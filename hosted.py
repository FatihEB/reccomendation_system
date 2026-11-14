"""Serve the public research demo behind an HTTPS reverse proxy."""

from io import BytesIO
import json
import logging
import os
from pathlib import Path
import re
from threading import BoundedSemaphore, Lock
from time import monotonic

from waitress import serve
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from werkzeug.wrappers import Request, Response

from app import MODELS, RESPONSE_HEADERS, UPLOAD_ERRORS, WEB, recommend, render_page
from catalog import Catalogue, inspect_username
from models import Models
from snapshots import DATA


MAX_FILE = 5 * 1024**2
MAX_BODY = 2 * MAX_FILE + 64 * 1024
BUILT_IN_PROFILES = ('adil', 'aryan', 'fatih', 'xander', 'zeynep')
HEADERS = {
    **RESPONSE_HEADERS,
    'Referrer-Policy': 'no-referrer',
    'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
    'Strict-Transport-Security': 'max-age=31536000',
}


class Upload(Request):
    max_content_length = MAX_BODY
    max_form_memory_size = 128 * 1024
    max_form_parts = 16

    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return BytesIO()


class PublicApp:
    def __init__(self, data, host, profile_dir=None):
        if not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?(?::[0-9]{1,5})?', host):
            raise ValueError('Set PUBLIC_HOST to the HTTPS hostname, without a scheme or path.')
        self.host = host
        self.origin = f'https://{host}'
        self.catalogue = Catalogue(data)
        self.models = Models(self.catalogue, data)
        if 'taste' not in self.models.names or 'collaborative' not in self.models.names:
            raise ValueError('Prepare the hosted snapshots and model caches before starting the service.')
        profile_dir = Path(profile_dir) if profile_dir else Path(__file__).resolve().parent / 'profiles'
        self.profiles = {}
        for key in BUILT_IN_PROFILES:
            path = profile_dir / f'{key}.zip'
            if not path.is_file():
                raise ValueError(f'The built-in profile {path.name} is missing.')
            contents = path.read_bytes()
            if len(contents) > MAX_FILE:
                raise ValueError(f'The built-in profile {path.name} exceeds 5 MB.')
            username = inspect_username(contents)
            self.profiles[key] = {
                'name': path.name,
                'data': contents,
                'label': f'@{username}' if username else path.stem,
            }
        self.work = BoundedSemaphore(1)
        self.rate_lock = Lock()
        self.tokens = 8.0
        self.updated = monotonic()
        self.assets = {
            '/app.js': ((WEB / 'app.js').read_bytes(), 'text/javascript'),
            '/style.css': ((WEB / 'style.css').read_bytes(), 'text/css'),
        }

    def take_token(self):
        with self.rate_lock:
            now = monotonic()
            self.tokens = min(8, self.tokens + (now - self.updated) / 2)
            self.updated = now
            if self.tokens < 1:
                return False
            self.tokens -= 1
            return True

    def page(self, status=200, payload=None, result=None, error=''):
        labels = {key: profile['label'] for key, profile in self.profiles.items()}
        return Response(render_page(self.models, payload, result, error, hosted=True, profiles=labels),
                        status=status, mimetype='text/html')

    def upload(self, request):
        preview = request.path == '/api/import-profile'
        payload = {'imports': [None, None]}

        def error(status, message):
            if preview:
                return self.json({'error': message}, status)
            return self.page(status, payload, error=message)

        if request.headers.get('Origin') != self.origin:
            return error(403, 'Open the upload form on this website.')
        if request.mimetype != 'multipart/form-data':
            return error(415, 'Use the upload form.')
        if not request.content_length or request.content_length > MAX_BODY:
            return error(413, 'The upload is empty or too large.')
        if not self.work.acquire(blocking=False):
            return error(429, 'The demo is busy. Try again shortly.')
        try:
            if not self.take_token():
                return error(429, 'The demo is busy. Try again shortly.')
            if not request.mimetype_params.get('boundary'):
                return error(400, 'The upload is missing its multipart boundary.')
            for slot, name in enumerate(('first', 'second')):
                file = request.files.get(name)
                profile_key = request.form.get(f'{name}_profile', '')
                if profile_key:
                    if profile_key not in self.profiles:
                        return error(400, f'Choose a valid built-in profile for viewer {slot + 1}.')
                    if file and file.filename:
                        return error(400, f'Choose either a built-in profile or an upload for viewer {slot + 1}.')
                    profile = self.profiles[profile_key]
                    payload['imports'][slot] = {'name': profile['name'], 'data': profile['data']}
                    payload[f'{name}_profile'] = profile_key
                if file and file.filename:
                    data = file.stream.read(MAX_FILE + 1)
                    if len(data) > MAX_FILE:
                        return error(413, 'Each CSV or ZIP must be 5 MB or smaller.')
                    payload['imports'][slot] = {'name': file.filename, 'data': data}
            if preview:
                first = payload['imports'][0]
                if not first or not first['name'].lower().endswith('.zip'):
                    return error(400, 'Choose a Letterboxd ZIP export.')
                return self.json({'username': inspect_username(first['data'])})
            runtime = request.form.get('runtime', '')
            payload.update(mode=request.form.get('mode'), model=request.form.get('model') or 'taste',
                           seen=[choice for choice in request.form.getlist('seen') if choice],
                           runtime=int(runtime) if re.fullmatch(r'[0-9]{1,4}', runtime) else runtime or None)
            result = recommend(self.catalogue, self.models, payload)
            return self.page(payload=payload, result=result)
        except RequestEntityTooLarge:
            return error(413, 'The upload exceeds the form limits.')
        except (BadRequest, *UPLOAD_ERRORS):
            return error(400, 'The upload could not be read. Use a UTF-8 CSV or Letterboxd export ZIP.')
        except ValueError as exc:
            return error(400, str(exc))
        finally:
            self.work.release()

    @staticmethod
    def json(value, status=200):
        return Response(json.dumps(value), status=status, mimetype='application/json')

    def dispatch(self, request):
        if request.host.lower() != self.host:
            return Response('Unknown host.', status=400)
        if request.method in {'GET', 'HEAD'}:
            if request.path == '/':
                return self.page()
            if request.path == '/healthz':
                return Response('ready\n', mimetype='text/plain')
            if request.path == '/favicon.ico':
                return Response(status=204)
            if request.path == '/robots.txt':
                return Response('User-agent: *\nDisallow: /\n', mimetype='text/plain')
            if request.path == '/api/models':
                return self.json([key for key in MODELS if key in self.models.names])
            if request.path in self.assets:
                body, mimetype = self.assets[request.path]
                return Response(body, mimetype=mimetype)
        if request.method == 'POST' and request.path in {'/recommend', '/api/import-profile'}:
            return self.upload(request)
        return Response('Page unavailable.', status=404)

    def __call__(self, environ, start_response):
        with Upload(environ) as request:
            try:
                response = self.dispatch(request)
            except Exception:
                logging.error('Public app request failed.')
                response = Response('The service could not process the request.', status=500)
            response.headers.update(HEADERS)
            if response.status_code == 429:
                response.headers['Retry-After'] = '2'
            return response(environ, start_response)


if __name__ == '__main__':
    host = os.environ.get('PUBLIC_HOST') or os.environ.get('RENDER_EXTERNAL_HOSTNAME', '')
    data = Path(os.environ.get('HOSTED_DATA', DATA / 'hosted'))
    application = PublicApp(data, host.lower())
    serve(application, host=os.environ.get('HOSTED_BIND', '0.0.0.0'), port=int(os.environ.get('PORT', '10000')),
          threads=3, connection_limit=10, backlog=32, channel_timeout=15, cleanup_interval=5,
          max_request_header_size=16384, max_request_body_size=MAX_BODY, inbuf_overflow=MAX_BODY,
          url_scheme='https', ident='recommender', expose_tracebacks=False, log_socket_errors=False)
