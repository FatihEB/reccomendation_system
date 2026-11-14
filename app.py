"""Serve the local recommendation interface and keep imports in request memory."""

import argparse
import base64
import binascii
import csv
from email import policy
from email.parser import BytesParser
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from string import Template
import threading
import webbrowser
from zipfile import BadZipFile

import numpy as np

from catalog import Catalogue, import_profile, inspect_username
from models import TASTE, Models, pair_score, rank_candidates
from snapshots import DATA


WEB = Path(__file__).resolve().parent / 'web'
MODELS = {
    'taste': 'Personal Taste',
    'taste_genome': 'Personal Taste · genre, mood and storytelling',
    'taste_credits': 'Personal Taste · credits',
    'taste_traits': 'Personal Taste · mood and genre',
    'audience': 'TMDB audience',
    'mooduel': 'Mooduel',
    'ml_audience': 'MovieLens audience',
    'collaborative': 'MovieLens collaborative',
    'collaborative_mooduel': 'MovieLens + Mooduel',
}
DEFAULT_MODEL = 'audience'
VIEWERS = {'first': 'Viewer 1', 'second': 'Viewer 2', 'shared': 'Both viewers'}
SEEN = {'neither': 'Neither viewer has seen', 'one': 'Exactly one viewer has seen', 'both': 'Both viewers have seen'}
LIMIT = 16 * 1024 * 1024
UPLOAD_ERRORS = (UnicodeError, BadZipFile, binascii.Error, csv.Error, RuntimeError, NotImplementedError)


def watched_overrides(catalogue, data):
    path = data / 'watched-overrides.json'
    if not path.exists():
        return {}
    return {
        username.casefold(): {catalogue.imdb[film] for film in films if film in catalogue.imdb}
        for username, films in json.loads(path.read_text(encoding='utf-8')).items()
    }


def recommend(catalogue, models, payload, overrides=None):
    mode = payload.get('mode')
    if not isinstance(mode, str):
        raise ValueError('Choose a recommendation mode.')
    selected = {'first': [0], 'second': [1], 'shared': [0, 1]}.get(mode)
    if selected is None:
        raise ValueError('Choose a recommendation mode.')
    model = payload.get('model') or DEFAULT_MODEL
    if not isinstance(model, str) or model not in MODELS or model not in models.names:
        raise ValueError('That model is unavailable. Train MovieLens or select a snapshot baseline.')
    supplied = payload.get('imports')
    if not isinstance(supplied, list) or len(supplied) != 2:
        raise ValueError('Provide the two import slots.')
    allowed = catalogue.eligible.copy()
    profiles = []
    for slot in selected:
        upload = supplied[slot]
        if (not isinstance(upload, dict) or not isinstance(upload.get('name'), str)
                or not isinstance(upload.get('data'), (str, bytes))):
            raise ValueError(f'Choose a CSV or ZIP for viewer {slot + 1}.')
        data = upload['data']
        if isinstance(data, str):
            data = base64.b64decode(data, validate=True)
        if len(data) > 5 * 1024**2:
            raise ValueError('Each CSV or ZIP must be 5 MB or smaller.')
        profile = import_profile(catalogue, data, upload['name'])
        if not profile.ratings:
            raise ValueError(f'No valid ratings matched the catalogue for viewer {slot + 1}.')
        if overrides and profile.username:
            profile.watched.update(overrides.get(profile.username.casefold(), ()))
        profiles.append(profile)
    if mode == 'shared':
        seen = payload.get('seen', ['neither'])
        if not isinstance(seen, list) or not seen or any(not isinstance(key, str) or key not in SEEN for key in seen):
            raise ValueError('Select at least one viewing history filter.')
        seen_count = np.zeros(len(catalogue.movies), dtype=np.uint8)
        for profile in profiles:
            seen_count[list(profile.watched)] += 1
        allowed &= np.isin(seen_count, [index for index, key in enumerate(SEEN) if key in seen])
    else:
        allowed[list(profiles[0].watched)] = False
    runtime = payload.get('runtime')
    if runtime is not None:
        if type(runtime) is not int or not 1 <= runtime <= 1440:
            raise ValueError('Runtime limit must be a whole number from 1 to 1440 minutes.')
        allowed &= np.array([movie.runtime is not None and movie.runtime <= runtime for movie in catalogue.movies])
    scored = [models.score(model, profile.ratings) for profile in profiles]
    if len(scored) == 2:
        rank = pair_score(scored[0].stars, scored[1].stars)
        references = [score.stars[list(profile.ratings)].mean() for score, profile in zip(scored, profiles)]
        reference = pair_score(*references)
    else:
        rank = scored[0].rank
    coverage_adjusted = model not in TASTE and models.metadata is not None
    if len(scored) == 2:
        ordering = models.supported_rank(rank, reference) if coverage_adjusted else rank
    else:
        ordering = models.individual_rank(model, scored[0], profiles[0].ratings)
    indices = rank_candidates(ordering, allowed)
    films = []
    for index in indices[:24]:
        movie = catalogue.movies[index]
        directors, writers = models.taste.credits.display(index) if models.taste.credits else ([], [])
        films.append({'title': movie.title, 'year': movie.year, 'runtime': movie.runtime, 'genres': movie.genres,
                      'tmdb': movie.tmdb, 'imdb': movie.imdb, 'tmdb_rating': movie.tmdb_rating,
                      'directors': directors, 'writers': writers,
                      'movielens_ratings': int(models.counts[index]) if models.metadata else None,
                      'stars': [round(float(score.stars[index]), 1) for score in scored],
                      'shared': round(float(rank[index]), 2) if len(scored) == 2 else None,
                      'seen_by': [slot for slot, profile in zip(selected, profiles) if index in profile.watched],
                      'reasons': [models.explain(score, index) for score in scored]})
    return {'model': model, 'candidates': len(indices), 'slots': selected, 'films': films,
            'coverage_adjusted': coverage_adjusted,
            'profiles': [{'username': profile.username, 'ratings': len(profile.ratings), 'watched': len(profile.watched),
                          'rating_rows': sum(row[0] == 'ratings.csv' for row in profile.audit)} for profile in profiles]}


def form_payload(content_type, body):
    message = BytesParser(policy=policy.default).parsebytes(
        b'Content-Type: ' + content_type.encode('ascii') + b'\r\n\r\n' + body
    )
    if not message.is_multipart() or message.defects:
        raise ValueError('The browser could not send the uploads. Select the files and try again.')
    fields, uploads = {}, {}
    for part in message.iter_parts():
        name = part.get_param('name', header='content-disposition')
        data = part.get_payload(decode=True)
        if name in {'first', 'second'} and part.get_filename() and data is not None:
            uploads[name] = {'name': part.get_filename(), 'data': data}
        elif name in {'mode', 'model', 'runtime'} and data is not None:
            fields[name] = data.decode('utf-8')
        elif name == 'seen' and data is not None:
            choice = data.decode('utf-8')
            selected = fields.setdefault('seen', [])
            if choice:
                selected.append(choice)
    runtime = fields.get('runtime') or None
    if runtime is not None:
        try:
            minutes = float(runtime)
        except ValueError:
            minutes = 0.0
        if minutes.is_integer():
            runtime = int(minutes)
    fields['runtime'] = runtime
    fields['imports'] = [uploads.get('first'), uploads.get('second')]
    return fields


def options(labels, selected):
    return ''.join(
        f'<option value="{key}"{" selected" if key == selected else ""}>{escape(label)}</option>'
        for key, label in labels.items()
    )


def seen_options(selected):
    if not isinstance(selected, list):
        selected = ['neither']
    return ''.join(
        f'<label><input type="checkbox" name="seen" value="{key}"{" checked" if key in selected else ""}> {label}</label>'
        for key, label in SEEN.items()
    )


def film_card(film, names):
    title = escape(film['title'])
    runtime = f"{film['runtime']} min" if film['runtime'] else 'Runtime unavailable'
    metadata = escape(' · '.join([str(film['year']), runtime, *film['genres']]))
    rating = film['tmdb_rating']
    public_score = 'TMDB score unavailable' if rating is None else f'TMDB {rating:.1f}/10 (Mooduel v1.0 snapshot)'
    count = film['movielens_ratings']
    coverage = '' if count is None else f'<p class="metadata">MovieLens 32M ratings: {count:,}</p>'
    credits = ''.join(f'<p class="metadata">{role}: {escape(", ".join(people))}</p>'
                      for role, people in [('Directed by', film['directors']), ('Writing credits', film['writers'])]
                      if people)
    links = []
    if film['tmdb']:
        links.append(f'<a href="https://www.themoviedb.org/movie/{film["tmdb"]}" target="_blank" rel="noopener noreferrer">TMDB</a>')
    if film['imdb']:
        links.append(f'<a href="https://www.imdb.com/title/{film["imdb"]}/" target="_blank" rel="noopener noreferrer">IMDb</a>')
    stars = ''.join(f'<p>{escape(name)} estimate · {score:.1f} / 5</p>' for name, score in zip(names.values(), film['stars']))
    if film['shared'] is not None:
        stars += f'<p class="shared">Shared fit · {film["shared"]:.2f} / 5</p>'
    seen = ''
    if len(names) == 2:
        viewers = film['seen_by']
        label = f'{names[viewers[0]]} has seen' if len(viewers) == 1 else SEEN['both' if viewers else 'neither']
        seen = f'<p class="metadata">{escape(label)}</p>'
    reasons = ''.join(f'<p>{escape(name)}: {escape(reason)}</p>' for name, reason in zip(names.values(), film['reasons']))
    return f"""
      <article>
        <div>
          <h2>{title}</h2>
          <p class="metadata">{metadata}</p>
          {seen}
          <p class="metadata">{public_score}</p>
          {coverage}
          {credits}
          <p class="metadata">Premise, verified awards, Letterboxd score and Letterboxd distribution unavailable.</p>
          <div class="links">{' '.join(links)}</div>
          <details><summary>About these estimates</summary>{reasons}</details>
        </div>
        <div class="scores">{stars}</div>
      </article>"""


def profile_controls(profiles, selected):
    if not profiles:
        return ''
    controls = []
    for slot, label in (('first', 'Viewer 1'), ('second', 'Viewer 2')):
        value = selected.get(f'{slot}_profile', '')
        choices = {'': 'Upload a CSV or ZIP', **profiles}
        controls.append(f'''<label>{label} profile
          <select id="{slot}-profile" name="{slot}_profile">{options(choices, value)}</select>
        </label>''')
    return f'''<div class="profile-pickers">
        {''.join(controls)}
      </div>
      <p class="hint">Choose a built-in Letterboxd profile or leave it on upload to use your own file.</p>'''


def render_page(models, payload=None, data=None, error='', hosted=False, profiles=None):
    payload = payload or {}
    available = {key: label for key, label in MODELS.items() if key in models.names}
    default = 'taste' if hosted else DEFAULT_MODEL
    notice = error
    cards = ''
    usernames = {}
    if data:
        usernames = {slot: profile['username'] for slot, profile in zip(data['slots'], data['profiles'])}
        names = {slot: f'@{username}' if username else f'Viewer {slot + 1}' for slot, username in usernames.items()}
        summary = [
            f"{names[slot]}: {profile['ratings']} matched ratings, {profile['watched']} watched films"
            for slot, profile in zip(data['slots'], data['profiles'])
        ]
        notice = ' · '.join([*summary, f"{data['candidates']:,} eligible films", MODELS[data['model']]])
        if data['coverage_adjusted']:
            notice += ' · Ranking adjusted for MovieLens coverage'
        cards = ''.join(film_card(film, names) for film in data['films']) or '<p>No films meet the current filters.</p>'
    elif not hosted and not notice and 'collaborative' not in available:
        notice = 'For MovieLens models, run python train.py and restart the app.'
    document = Template((WEB / 'index.html').read_text(encoding='utf-8')).substitute(
        viewer_options=options(VIEWERS, payload.get('mode', 'first')),
        model_options=options(available, payload.get('model') or default),
        seen_options=seen_options(payload.get('seen', ['neither'])),
        runtime=escape(str(payload['runtime'])) if payload.get('runtime') is not None else '',
        status=escape(notice),
        results=cards,
        first_username=escape('@' + usernames[0]) if usernames.get(0) else '',
        second_username=escape('@' + usernames[1]) if usernames.get(1) else '',
        hosting_notice=HOSTING_NOTICE if hosted else '',
        profile_controls=profile_controls(profiles, payload),
    )
    return document.encode()


HOSTING_NOTICE = """<p class="hint">Uploads are sent to this service for scoring and processed in memory.
Only upload exports you have permission to share. This is a noncommercial research demo.</p>
<details><summary>Data sources</summary><p>
<a href="https://huggingface.co/datasets/fractalintelligence/mooduel-v1.0">Mooduel v1.0</a>
by Sutirtha Basak and Mooduel contributors, CC BY-NC 4.0. Film facts and audience scores
in that snapshot originate from <a href="https://www.themoviedb.org">TMDB</a>.
The genre and mood features are transformed for scoring.</p><p>
<a href="https://grouplens.org/datasets/movielens/32m/">MovieLens 32M</a> by GroupLens Research,
University of Minnesota, provides catalogue links and research ratings.
<a href="https://grouplens.org/datasets/movielens/tag-genome-2021/">Tag Genome 2021</a>
provides relevance scores grouped into traits, CC BY-NC 3.0. Information courtesy of
<a href="https://www.imdb.com">IMDb</a>. Used with permission. IMDb's non-commercial
datasets provide director and writer credits.</p></details>"""

RESPONSE_HEADERS = {
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
    'Content-Security-Policy': "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
}


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        for key, value in RESPONSE_HEADERS.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def json(self, status, value):
        self.reply(status, json.dumps(value).encode(), 'application/json; charset=utf-8')

    def page(self, status=200, payload=None, data=None, error=''):
        self.reply(status, render_page(self.server.models, payload, data, error), 'text/html; charset=utf-8')

    def do_GET(self):
        if self.path == '/':
            self.page()
            return
        if self.path == '/favicon.ico':
            self.reply(204, b'', 'image/x-icon')
            return
        if self.path == '/api/models':
            self.json(200, [model for model in MODELS if model in self.server.models.names])
            return
        resource = {'/app.js': ('app.js', 'text/javascript'),
                    '/style.css': ('style.css', 'text/css')}.get(self.path)
        if resource:
            self.reply(200, (WEB / resource[0]).read_bytes(), resource[1] + '; charset=utf-8')
        else:
            self.json(404, {'error': 'Page unavailable.'})

    def do_POST(self):
        is_form = self.path == '/recommend'
        is_preview = self.path == '/api/import-profile'
        if not is_form and not is_preview and self.path != '/api/recommend':
            self.json(404, {'error': 'Route unavailable.'})
            return
        payload = {}

        def reply_error(status, message):
            if is_form:
                self.page(status, payload, error=message)
            else:
                self.json(status, {'error': message})

        address = self.headers.get('Host')
        port = self.server.server_port
        if (address not in {f'127.0.0.1:{port}', f'localhost:{port}'}
                or self.headers.get('Origin', f'http://{address}') != f'http://{address}'):
            reply_error(403, 'Open the app using its local address.')
            return
        is_multipart = is_form or is_preview
        expected_type = 'multipart/form-data' if is_multipart else 'application/json'
        if self.headers.get_content_type() != expected_type:
            reply_error(415, 'Use the upload form.' if is_multipart else 'Use a JSON request.')
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= LIMIT:
                raise ValueError('The upload is empty or too large.')
            body = self.rfile.read(length)
            submitted = form_payload(self.headers['Content-Type'], body) if is_multipart else json.loads(body)
            if not isinstance(submitted, dict):
                raise ValueError('Invalid recommendation request.')
            payload = submitted
            if is_preview:
                upload = payload['imports'][0]
                if not upload or not upload['name'].lower().endswith('.zip'):
                    raise ValueError('Choose a Letterboxd ZIP export.')
                if len(upload['data']) > 5 * 1024**2:
                    raise ValueError('Each CSV or ZIP must be 5 MB or smaller.')
                self.json(200, {'username': inspect_username(upload['data'])})
                return
            result = recommend(self.server.catalogue, self.server.models, payload, self.server.watched_overrides)
            if is_form:
                self.page(payload=payload, data=result)
            else:
                self.json(200, result)
        except (BrokenPipeError, ConnectionResetError):
            return
        except UPLOAD_ERRORS:
            reply_error(400, 'The upload could not be read. Use a UTF-8 CSV or a Letterboxd export ZIP.')
        except ValueError as error:
            reply_error(400, str(error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    catalogue = Catalogue(args.data)
    models = Models(catalogue, args.data)
    overrides = watched_overrides(catalogue, args.data)
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        server.catalogue = catalogue
        server.models = models
        server.watched_overrides = overrides
        address = f'http://127.0.0.1:{server.server_port}/'
        print(f'Open {address}', flush=True)
        if args.open_browser:
            threading.Thread(target=webbrowser.open, args=(address,), daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
