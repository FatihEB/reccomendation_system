"""Train explicit MovieLens factors and write an offline model cache."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from catalog import Catalogue
from snapshots import DATA, ML, MOOD, digest


MODEL = 'movielens-model.npz'
CONFIG = {'version': 1, 'factors': 32, 'epochs': 7, 'seed': 42, 'rating_rows': 32000204}
RECORD = np.dtype([('user', '<u4'), ('item', '<u4'), ('rating', '<f4')])


def train(data, compiler=None, prebuilt=None):
    import pandas as pd

    catalogue = Catalogue(data)
    archive_path = data / ML
    if not archive_path.exists():
        raise ValueError('Cache MovieLens 32M with snapshots.py first.')
    mapping = np.zeros(max(catalogue.ml) + 1, dtype=np.uint32)
    for source_id, index in catalogue.ml.items():
        mapping[source_id] = index
    counts = np.zeros(len(catalogue.movies), dtype=np.int64)
    totals = np.zeros(len(catalogue.movies))
    rows = 0
    users = 0
    with TemporaryDirectory(prefix='training-', dir=data) as work:
        folder = Path(work)
        records = folder / 'ratings.bin'
        with ZipFile(archive_path) as archive, archive.open('ml-32m/ratings.csv') as stream, records.open('wb') as binary:
            chunks = pd.read_csv(stream, usecols=['userId', 'movieId', 'rating'], chunksize=1_000_000)
            for chunk in chunks:
                block = np.empty(len(chunk), dtype=RECORD)
                block['user'] = chunk.userId.to_numpy() - 1
                block['item'] = mapping[chunk.movieId.to_numpy()]
                block['rating'] = chunk.rating.to_numpy()
                block.tofile(binary)
                counts += np.bincount(block['item'], minlength=len(counts))
                totals += np.bincount(block['item'], weights=block['rating'], minlength=len(counts))
                users = max(users, int(block['user'].max()) + 1)
                rows += len(block)
        if rows != CONFIG['rating_rows'] or users != 200948:
            raise ValueError(f'Unexpected MovieLens shape: {rows} ratings, {users} users')
        average = totals.sum() / counts.sum()
        source = Path(__file__).with_name('factorize.cpp')
        executable = prebuilt or folder / 'factorize.exe'
        output = folder / 'factors.bin'
        if prebuilt is None:
            subprocess.run([compiler, '-std=c++17', '-O3', str(source), '-o', str(executable)], check=True)
        subprocess.run([str(executable), str(records), str(output), str(rows), str(users), str(len(counts)),
                        str(CONFIG['epochs']), str(CONFIG['seed']), str(average)], check=True)
        values = np.fromfile(output, dtype='<f4').reshape(len(counts), CONFIG['factors'] + 1)
        if not np.isfinite(values).all():
            raise ValueError('MovieLens factor training produced nonfinite weights')
        metadata = {**CONFIG, 'users': users, 'movies': len(counts), 'mooduel_sha256': digest(data / MOOD),
                    'movielens_sha256': digest(archive_path), 'training_source_sha256': digest(source)}
        temporary = folder / MODEL
        np.savez_compressed(temporary, counts=counts, totals=totals, bias=values[:, 0],
                            factors=values[:, 1:], metadata=json.dumps(metadata))
        temporary.replace(data / MODEL)
        print(f'Wrote {data / MODEL}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--compiler', default=shutil.which('g++') or shutil.which('clang++'))
    parser.add_argument('--factorizer', type=Path, help='Path to the compiled factorizer.')
    args = parser.parse_args()
    if not args.compiler and not args.factorizer:
        parser.error('Install a C++17 compiler (g++ or clang++) and add it to PATH.')
    train(args.data.resolve(), args.compiler, args.factorizer.resolve() if args.factorizer else None)
