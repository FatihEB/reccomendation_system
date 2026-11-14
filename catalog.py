"""Canonical catalogue and conservative Letterboxd import matching."""

import csv
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from io import BytesIO, TextIOWrapper
import json
from pathlib import Path, PurePosixPath
import re
import unicodedata
from urllib.parse import urlparse
from zipfile import ZipFile

import numpy as np

from snapshots import ML, MOOD


def normalize(title):
    text = unicodedata.normalize('NFKD', title.casefold())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[’']", '', text)
    return ' '.join(re.findall(r'[^\W_]+', text))


def integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def number(value):
    try:
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def letterboxd_uri(value):
    parts = urlparse((value or '').strip())
    host = (parts.hostname or '').removeprefix('www.')
    path = parts.path.rstrip('/')
    if host == 'boxd.it' and re.fullmatch(r'/[A-Za-z0-9]+', path):
        return host + path
    if host == 'letterboxd.com' and re.fullmatch(r'/film/[\w-]+', path):
        return host + path.lower()
    return ''


def imdb_id(value):
    match = re.fullmatch(r'(?:tt)?(\d+)', str(value or '').strip())
    if match and int(match[1]) > 0:
        return 'tt' + str(int(match[1])).zfill(7)
    return None


def movie_title(value):
    match = re.fullmatch(r'(.*?)\s*\((\d{4})\)\s*', value)
    return (match[1], int(match[2])) if match else (value, None)


def aliases(title):
    alternate = re.findall(r'\(([^()]+)\)', title)
    for name in [title, re.sub(r'\s*\([^()]+\)', '', title), *alternate]:
        name = re.sub(r'^a\.k\.a\.\s*', '', name, flags=re.I)
        match = re.fullmatch(r'(.+), (The|A|An|Les|Le|La|L\'|Die|Der|Das|El|Los|Las|Il|I)', name, re.I)
        yield f'{match[2]} {match[1]}' if match else name


@dataclass
class Movie:
    title: str
    year: int | None
    tmdb: int | None = None
    imdb: str | None = None
    genres: tuple = ()
    runtime: int | None = None
    tmdb_rating: float | None = None
    mood: dict | None = None


class Catalogue:
    def __init__(self, data, release_cutoff_year=None):
        self.movies = []
        self.ml = {}
        self.tmdb = {}
        self.imdb = {}
        self.names = {}
        self.mood_rows = 0
        with (data / MOOD).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                tmdb = int(row['tmdb_id'])
                if tmdb in self.tmdb:
                    raise ValueError(f'Duplicate Mooduel TMDB ID: {tmdb}')
                year = integer(row['year'])
                runtime = integer(row['runtime'])
                rating = number(row['tmdb_rating'])
                self.tmdb[tmdb] = self.add(Movie(
                    row['title'], year, tmdb, genres=tuple(json.loads(row['genres'] or '[]')),
                    runtime=runtime if runtime and 0 < runtime <= 1440 else None,
                    tmdb_rating=rating if rating and 0 < rating <= 10 else None, mood=row,
                ))
                self.mood_rows += 1
        if (data / ML).is_file():
            self.load_movielens(data / ML)
        self.release_cutoff_year = release_cutoff_year or date.today().year
        # A snapshot year alone cannot confirm that a current-year film is already released.
        self.eligible = np.array([movie.year is not None and 1800 <= movie.year < self.release_cutoff_year for movie in self.movies])
        self.mood_present = np.array([movie.mood is not None for movie in self.movies])

    def add(self, movie):
        index = len(self.movies)
        self.movies.append(movie)
        self.index_name(index, movie.title, movie.year)
        return index

    def index_name(self, index, title, year):
        if title and year:
            self.names.setdefault((normalize(title), year), set()).add(index)

    def load_movielens(self, archive):
        with ZipFile(archive) as source:
            links = {int(row['movieId']): row for row in archive_rows(source, 'ml-32m/links.csv')}
            tmdb_counts = Counter(integer(row['tmdbId']) for row in links.values() if row['tmdbId'])
            self.ambiguous_tmdb_links = sum(count > 1 for count in tmdb_counts.values())
            for row in archive_rows(source, 'ml-32m/movies.csv'):
                movie_id = int(row['movieId'])
                link = links[movie_id]
                tmdb = integer(link['tmdbId']) or None
                if tmdb_counts[tmdb] > 1:
                    tmdb = None
                imdb = imdb_id(link['imdbId'])
                title, year = movie_title(row['title'])
                by_tmdb, by_imdb = self.tmdb.get(tmdb), self.imdb.get(imdb)
                if by_tmdb is not None and by_imdb is not None and by_tmdb != by_imdb:
                    raise ValueError(f'Conflicting MovieLens IDs for {movie_id}')
                index = by_tmdb if by_tmdb is not None else by_imdb
                if index is None:
                    genres = tuple(g for g in row['genres'].split('|') if g != '(no genres listed)')
                    index = self.add(Movie(next(aliases(title)), year, tmdb, imdb, genres))
                self.ml[movie_id] = index
                movie = self.movies[index]
                if imdb:
                    self.imdb[imdb] = index
                    movie.imdb = movie.imdb or imdb
                if tmdb:
                    self.tmdb[tmdb] = index
                for alias in aliases(title):
                    self.index_name(index, alias, year)

    def resolve(self, row):
        providers = ((self.tmdb, integer(row.get('TMDB ID'))), (self.imdb, imdb_id(row.get('IMDb ID'))),
                     (self.ml, integer(row.get('MovieLens ID'))))
        ids = {mapping[value] for mapping, value in providers if value in mapping}
        if len(ids) > 1:
            return None, 'conflicting IDs'
        if ids:
            return ids.pop(), 'provider ID'
        if any(value is not None for _, value in providers):
            return None, 'unknown provider ID'
        matches = self.names.get((normalize(row.get('Name') or row.get('Title') or ''), integer(row.get('Year'))), ())
        if len(matches) == 1:
            return next(iter(matches)), 'exact title/year'
        return None, 'ambiguous title/year' if matches else 'unmatched'


def archive_rows(archive, member):
    with archive.open(member) as stream:
        yield from csv.DictReader(TextIOWrapper(stream, encoding='utf-8-sig', newline=''))


@dataclass
class Profile:
    ratings: dict = field(default_factory=dict)
    watched: set = field(default_factory=set)
    watchlist: set = field(default_factory=set)
    favorites: list = field(default_factory=list)
    audit: list = field(default_factory=list)
    username: str | None = None
    rating_dates: dict[int, date] = field(default_factory=dict)


def export_members(archive, files):
    selected = {}
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        if path.name not in files or len(path.parts) > 2 or path.parent.name in {'deleted', 'orphaned'}:
            continue
        if path.name in selected or item.file_size > 25 * 1024**2:
            raise ValueError('ZIP contains duplicate or oversized export tables.')
        selected[path.name] = item
    return selected


def export_rows(stream):
    def lines():
        while line := stream.readline(131073):
            if len(line) > 131072:
                raise ValueError('An export contains an oversized CSV line.')
            yield line

    reader = csv.reader(lines(), strict=True)
    headers = next(reader, [])
    if len(headers) > 64 or len(set(headers)) != len(headers):
        raise ValueError('An export contains invalid CSV columns.')
    relevant = {'Name', 'Title', 'Year', 'Rating', 'Date', 'Letterboxd URI',
                'TMDB ID', 'IMDb ID', 'MovieLens ID', 'Username', 'Favorite Films'}
    columns = [(index, name) for index, name in enumerate(headers) if name in relevant]
    yield headers
    for count, row in enumerate(reader, 1):
        if count > 30000 or len(row) > len(headers):
            raise ValueError('An export contains too many rows or columns.')
        if row:
            yield {name: row[index] for index, name in columns if index < len(row)}


def upload_table(archive, member):
    with archive.open(member) as stream, TextIOWrapper(stream, encoding='utf-8-sig', newline='') as decoded:
        rows = export_rows(decoded)
        next(rows)
        yield from rows


def profile_username(row):
    return (row.get('Username') or '').strip() or None


def inspect_username(data):
    with ZipFile(BytesIO(data)) as archive:
        member = export_members(archive, {'profile.csv'}).get('profile.csv')
        if member is None:
            return None
        if member.file_size > 256 * 1024:
            raise ValueError('The ZIP profile table is too large.')
        rows = upload_table(archive, member)
        try:
            return profile_username(next(rows, {}))
        finally:
            rows.close()


def read_export(source, filename):
    if filename.lower().endswith('.csv'):
        reader = export_rows(TextIOWrapper(source, encoding='utf-8-sig', newline=''))
        fields = set(next(reader))
        if not (fields & {'Name', 'Title'} and {'Year', 'Rating'} <= fields):
            raise ValueError('The ratings CSV needs Name or Title, Year and Rating columns.')
        return {'ratings.csv': list(reader)}
    if not filename.lower().endswith('.zip'):
        raise ValueError('Choose a Letterboxd ZIP or a CSV with Name, Year and Rating columns.')
    files = {'ratings.csv', 'watched.csv', 'watchlist.csv', 'diary.csv', 'profile.csv'}
    with ZipFile(source) as archive:
        selected = export_members(archive, files)
        if sum(member.file_size for member in selected.values()) > 16 * 1024**2:
            raise ValueError('The ZIP history tables are too large when expanded.')
        if not selected.keys() & {'ratings.csv', 'watched.csv', 'diary.csv'}:
            raise ValueError('The ZIP does not contain a Letterboxd history table.')
        return {key: list(upload_table(archive, item)) for key, item in selected.items()}


def import_profile(catalogue, source, filename=None, manual_favorites=()):
    if filename is None:
        path = Path(source)
        with path.open('rb') as stream:
            tables = read_export(stream, path.name)
    else:
        tables = read_export(BytesIO(source), filename)
    profile_row = (tables.get('profile.csv') or [{}])[0]
    profile = Profile(username=profile_username(profile_row))
    matched_uris = {}
    by_uri = {}
    tables_with_matches = []
    for table in ('ratings.csv', 'watched.csv', 'diary.csv', 'watchlist.csv'):
        for row in tables.get(table, ()):
            if not ({'Name', 'Title'} & row.keys() and 'Year' in row):
                raise ValueError('Film rows need Name or Title and Year columns.')
            uri = letterboxd_uri(row.get('Letterboxd URI'))
            index, method = catalogue.resolve(row)
            tables_with_matches.append((table, row, uri, index, method))
            if uri:
                by_uri.setdefault(uri, row)
                if index is not None:
                    matched_uris.setdefault(uri, set()).add(index)
    chosen_ratings = {}
    for table, row, uri, index, method in tables_with_matches:
        uri_matches = matched_uris.get(uri, set())
        if len(uri_matches) > 1:
            index, method = None, 'conflicting Letterboxd URI'
        if len(uri_matches) == 1 and index is None and method == 'unmatched':
            index, method = next(iter(uri_matches)), 'export URI alias'
        profile.audit.append((table, row.get('Name') or row.get('Title'), row.get('Year'), uri, index, method))
        if index is None:
            continue
        if table == 'watchlist.csv':
            profile.watchlist.add(index)
            continue
        profile.watched.add(index)
        rating = number(row.get('Rating'))
        if rating is not None and 0.5 <= rating <= 5 and rating * 2 == int(rating * 2):
            priority = (table == 'ratings.csv', row.get('Date') or '')
            if index not in chosen_ratings or priority >= chosen_ratings[index]:
                chosen_ratings[index] = priority
                profile.ratings[index] = rating
                profile.rating_dates.pop(index, None)
                raw_date = priority[1].strip()
                if re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', raw_date):
                    try:
                        profile.rating_dates[index] = date.fromisoformat(raw_date)
                    except ValueError:
                        pass
    exported = profile_row.get('Favorite Films') or ''
    entries = manual_favorites or re.findall(r'https?://[^,;\s]+', exported)
    for entry in dict.fromkeys(entries):
        uri = letterboxd_uri(entry)
        matches = matched_uris.get(uri, ())
        row = by_uri.get(uri, {})
        if uri:
            index = next(iter(matches)) if len(matches) == 1 else None
        else:
            name, year = movie_title(entry)
            index, _ = catalogue.resolve({'Name': name, 'Year': year})
        label = row.get('Name') or entry
        if index is None or all(target[1] != index for target in profile.favorites):
            profile.favorites.append((label, index))
        if index is not None:
            profile.watched.add(index)
    return profile
