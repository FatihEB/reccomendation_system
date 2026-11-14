"""Download and verify the two public research snapshots used by the prototype."""

import argparse
import hashlib
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
from zipfile import ZipFile


DATA = Path(__file__).resolve().parent / 'data'
MOOD = 'mooduel-v1.0.csv'
ML = 'ml-32m.zip'
MOOD_REVISION = '7230816e48ac18abc03297bb5ab9e612df248ef5'
MOOD_URL = f'https://huggingface.co/datasets/fractalintelligence/mooduel-v1.0/resolve/{MOOD_REVISION}/{MOOD}'
MOOD_SHA256 = '21e5060284853470ef651fb0e27470ff6b1efe7ffdbb76064d90cf468682c9b3'
ML_URL = 'https://files.grouplens.org/datasets/movielens/ml-32m.zip'
ML_SHA256 = 'e4a68655d7386b8f95f2f2424b2ff975dfdd15ffd59e0d864a14dca43e99d6ee'
ML_MEMBERS = {
    'links.csv': '8f033867bcb4e6be8792b21468b4fa6e',
    'movies.csv': '0df90835c19151f9d819d0822e190797',
    'ratings.csv': 'cf12b74f9ad4b94a011f079e26d4270a',
    'tags.csv': '963bf4fa4de6b8901868fddd3eb54567',
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(path, name):
    if name == MOOD:
        if digest(path) != MOOD_SHA256:
            raise ValueError('Mooduel SHA-256 differs from the publisher checksum')
        return
    if digest(path) != ML_SHA256:
        raise ValueError('MovieLens archive differs from the pinned SHA-256')
    with ZipFile(path) as archive:
        for member, expected in ML_MEMBERS.items():
            with archive.open(f'ml-32m/{member}') as stream:
                if hashlib.file_digest(stream, 'md5').hexdigest() != expected:
                    raise ValueError(f'MovieLens checksum mismatch for {member}')


def cache(name, url, directory, supplied=None):
    destination = directory / name
    if destination.exists():
        verify(destination, name)
        print(f'Verified cached {name}', flush=True)
        return
    with NamedTemporaryFile(dir=directory, suffix='.part', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        if supplied:
            shutil.copyfile(supplied, temporary)
        else:
            print(f'Downloading {name}', flush=True)
            with urlopen(Request(url, headers={'User-Agent': 'local-film-research/1.0'}), timeout=90) as source:
                with temporary.open('wb') as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
        verify(temporary, name)
        temporary.replace(destination)
        print(f'Verified {destination}', flush=True)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--mooduel-file', type=Path)
    parser.add_argument('--movielens-file', type=Path)
    parser.add_argument('--mood-only', action='store_true')
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    cache(MOOD, MOOD_URL, args.data, args.mooduel_file)
    if not args.mood_only:
        cache(ML, ML_URL, args.data, args.movielens_file)
