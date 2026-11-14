"""Cache the public data and model files used by the taste and review experiments."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen
from zipfile import ZipFile
import zlib

from snapshots import DATA, digest


CREW = 'title.crew.tsv.gz'
PEOPLE = 'name.basics.tsv.gz'
GENOME = 'genome-2021-glmer.deflate'
GENOME_URL = 'https://files.grouplens.org/datasets/tag-genome-2021/genome_2021.zip'
GENOME_RANGE = 'bytes=8180-123703877'
GENOME_BYTES = 123695698
GENOME_EXPANDED = 370383596
GENOME_CRC = 2609921574
GENOME_SHA256 = '46544a054f25e914b42d0b9dd8b06be9701861c6d555f24a52fc8e6b822d345d'
SOFT = 'soft-attributes.csv'
SOFT_REVISION = '8d8f42e46845bccdf0f988dbca3c350469faeede'
QWEN = 'Qwen3-4B-Q4_K_M.gguf'
QWEN_REVISION = 'bc640142c66e1fdd12af0bd68f40445458f3869b'
BERT_REVISION = 'b269c930ee0a3f3018eb59fc6b3ee2a64c527b2b'
RUNTIME_REVISION = 'b12336'
RUNTIME = f'llama-{RUNTIME_REVISION}-bin-win-vulkan-x64.zip'
RUNTIME_SHA = '7668c75c4fa97a5420ef025d51090ef3a612eee99497377c08409080b6124fe8'
QWEN_BYTES = 2497280256


def verify_genome(path):
    if path.stat().st_size != GENOME_BYTES:
        raise ValueError('Unexpected Tag Genome member size.')
    decompressor = zlib.decompressobj(-15)
    size = checksum = 0
    with path.open('rb') as source:
        while chunk := source.read(256 * 1024):
            decoded = decompressor.decompress(chunk)
            size += len(decoded)
            checksum = zlib.crc32(decoded, checksum)
    if not decompressor.eof or size != GENOME_EXPANDED or checksum != GENOME_CRC:
        raise ValueError('Tag Genome member does not match the release ZIP directory.')


def download_ranges(path, url, start, length, total, chunk_size):
    parts = path.with_name(path.name + '.parts')
    parts.mkdir(exist_ok=True)

    def download(offset):
        first = start + offset
        size = min(chunk_size, length - offset)
        last = first + size - 1
        part = parts / str(first)
        if part.is_file() and part.stat().st_size == size:
            return part
        incomplete = part.with_suffix('.part')
        for _ in range(3):
            try:
                request = Request(url, headers={'Range': f'bytes={first}-{last}'})
                with urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {first}-{last}/{total}':
                        raise ValueError('Publisher did not return the requested byte range.')
                    with incomplete.open('wb') as target:
                        shutil.copyfileobj(response, target, 512 * 1024)
                if incomplete.stat().st_size != size:
                    raise OSError('Incomplete publisher response.')
                incomplete.replace(part)
                return part
            except OSError as error:
                last_error = error
        raise last_error

    with ThreadPoolExecutor(max_workers=5) as pool:
        ordered = list(pool.map(download, range(0, length, chunk_size)))
    with path.open('wb') as target:
        for part in ordered:
            with part.open('rb') as stream:
                shutil.copyfileobj(stream, target, 1024 * 1024)
    return ordered


def fetch(path, url, *, sha256=None, git_blob=None, byte_range=None, chunks=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    manifest = path.with_name(path.name + '.source.json')
    parts = []
    if not path.exists():
        headers = {'User-Agent': 'letterboxd-local-research/1.0'}
        print(f'Downloading {path.name}', flush=True)
        if chunks:
            parts = download_ranges(partial, url, *chunks)
            metadata = {'url': url, 'range': byte_range}
        else:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                metadata = {'url': url, 'last_modified': response.headers.get('Last-Modified'),
                            'source_date': response.headers.get('x-amz-meta-run-date')}
                with partial.open('wb') as target:
                    shutil.copyfileobj(response, target, 1024 * 1024)
        current = partial
    else:
        metadata = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {'url': url}
        current = path
    actual = digest(current)
    expected = sha256 or metadata.get('sha256')
    if expected and actual != expected:
        raise ValueError(f'SHA-256 mismatch for {path.name}')
    if git_blob:
        content = current.read_bytes()
        blob_hash = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        if blob_hash != git_blob:
            raise ValueError(f'Git blob hash mismatch for {path.name}')
    if byte_range:
        verify_genome(current)
    if current != path:
        current.replace(path)
    if parts:
        for part in parts:
            part.unlink()
        parts[0].parent.rmdir()
    metadata['sha256'] = actual
    manifest.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'Verified {path.name}', flush=True)


def fetch_taste(data):
    for name in (CREW, PEOPLE):
        fetch(data / name, f'https://datasets.imdbws.com/{name}')
    fetch(data / GENOME, GENOME_URL, sha256=GENOME_SHA256,
          byte_range=GENOME_RANGE, chunks=(8180, GENOME_BYTES, 1928028583, 8 * 1024**2))
    fetch(data / SOFT, f'https://raw.githubusercontent.com/google-research-datasets/soft-attributes/{SOFT_REVISION}/{SOFT}',
          sha256='f4690bf4474b4271b3808d937c8d4d89cc19d9a20d48a30efac4dc4cfc25fbdf')


def fetch_review_models(data):
    fetch(data / QWEN, f'https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/{QWEN_REVISION}/{QWEN}',
          sha256='7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5',
          chunks=(0, QWEN_BYTES, QWEN_BYTES, 32 * 1024**2))
    base = f'https://huggingface.co/Lowerated/lm6-movie-aspect-extraction-bert/resolve/{BERT_REVISION}'
    for name, blob in {
        'config.json': 'eaa9b6f270ed375d4f90b31cdc54dbec7bdc0a4e',
        'special_tokens_map.json': '9bbecc17cabbcbd3112c14d6982b51403b264bfa',
        'tokenizer_config.json': '75305659f7795d4549f0e23688b52fa20a32f925',
        'vocab.txt': 'fb140275c155a9c7c5a3b3e0e77a9e839594a938',
    }.items():
        fetch(data / 'lowerated' / name, f'{base}/{name}', git_blob=blob)
    fetch(data / 'lowerated' / 'model.safetensors', f'{base}/model.safetensors',
          sha256='32c6481332524fe60637d2bf4c35e9fbae13551010b015f59206ecb6af13e018')


def fetch_runtime(data):
    path = data / RUNTIME
    fetch(path, f'https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_REVISION}/{RUNTIME}',
          sha256=RUNTIME_SHA)
    with ZipFile(path) as archive:
        archive.extractall(data / f'llama-{RUNTIME_REVISION}-vulkan')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=('taste', 'reviews', 'runtime'))
    parser.add_argument('--data', type=Path, default=DATA)
    args = parser.parse_args()
    {'taste': fetch_taste, 'reviews': fetch_review_models, 'runtime': fetch_runtime}[args.group](args.data)
