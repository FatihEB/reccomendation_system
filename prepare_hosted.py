"""Prepare only the snapshots and derived caches used by the hosted demo."""

import argparse
from pathlib import Path
import shutil

from catalog import Catalogue
from snapshots import DATA, ML, ML_URL, MOOD, MOOD_URL, cache
from taste import CREDITS_CACHE, GENOME_CACHE, prepare_credits, prepare_genome
from taste_data import (CREW, GENOME, GENOME_BYTES, GENOME_RANGE, GENOME_SHA256, GENOME_URL,
                        PEOPLE, fetch)
from train import MODEL, train


def prepare(data):
    data.mkdir(parents=True, exist_ok=True)
    cache(MOOD, MOOD_URL, data)
    cache(ML, ML_URL, data)
    if not (data / MODEL).is_file():
        compiler = shutil.which('g++') or shutil.which('clang++')
        if compiler is None:
            raise ValueError('A C++17 compiler is needed to train MovieLens.')
        train(data, compiler)
    if not (data / GENOME_CACHE).is_file():
        fetch(data / GENOME, GENOME_URL, sha256=GENOME_SHA256, byte_range=GENOME_RANGE,
              chunks=(8180, GENOME_BYTES, 1928028583, 8 * 1024**2))
        prepare_genome(Catalogue(data), data)
    if not (data / CREDITS_CACHE).is_file():
        for name in (CREW, PEOPLE):
            fetch(data / name, f'https://datasets.imdbws.com/{name}')
        prepare_credits(Catalogue(data), data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA / 'hosted')
    prepare(parser.parse_args().data.resolve())
