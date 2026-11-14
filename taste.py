"""Learn per-viewer affinities for genres, mood, narrative traits, and film credits."""

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import gzip
from io import BufferedReader, RawIOBase, TextIOWrapper
from itertools import combinations
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import zlib

import numpy as np

from taste_data import CREW, GENOME, PEOPLE


CREDITS_CACHE = 'imdb-credits-local.json'
GENOME_CACHE = 'tag-genome-traits.npz'
TRAITS = {
    'suspense': ('suspense', 'suspenseful', 'tense'),
    'slow pacing': ('slow paced', 'slow'),
    'fast pacing': ('fast paced',),
    'nonlinear storytelling': ('nonlinear', 'unusual plot structure'),
    'complex plotting': ('complicated plot', 'complex'),
    'perceived confusion': ('confusing',),
    'moral ambiguity': ('moral ambiguity',),
    'dialogue-driven storytelling': ('dialogue driven', 'entirely dialogue'),
    'praised dialogue': ('good dialogue', 'great dialogue'),
    'wit': ('witty',),
    'political themes': ('political', 'politics', 'world politics'),
    'institutional corruption': ('corruption', 'police corruption'),
    'atmosphere': ('atmospheric',),
    'originality': ('original', 'original plot'),
    'emotional intensity': ('intense', 'emotional'),
    'complex characters': ('complex characters', 'character study'),
    'perceived realism': ('realistic',),
    'cinematography': ('cinematography', 'great cinematography', 'amazing cinematography'),
    'soundtrack': ('good soundtrack', 'notable soundtrack', 'great soundtrack', 'awesome soundtrack'),
    'praised payoff': ('powerful ending', 'great ending'),
    'criticized payoff': ('bad ending', 'plot holes'),
    'predictability': ('predictable',),
}
MOOD_TERMS = {
    'suspense': {'suspense', 'suspenseful', 'tension', 'psychological-tension', 'psychological-suspense',
                 'slow-burn-tension', 'cat-and-mouse'},
    'dread': {'dread', 'existential-dread', 'psychological-dread', 'atmospheric-dread', 'slow-burn-dread'},
    'psychological thriller': {'psychological-thriller'},
    'noir': {'noir', 'neo-noir', 'film-noir'},
    'heist': {'heist', 'heist-gone-wrong', 'heist-caper'},
    'political themes': {'political-thriller', 'political-intrigue', 'political-corruption'},
    'atmosphere': {'atmospheric', 'atmospheric-dread', 'atmospheric-horror', 'gothic-atmosphere'},
}


def write_json(path, value):
    with NamedTemporaryFile('w', dir=path.parent, encoding='utf-8', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'))
    temporary.replace(path)


class DeflateReader(RawIOBase):
    def __init__(self, path):
        self.source = path.open('rb')
        self.decoder = zlib.decompressobj(-15)
        self.pending = b''

    def readable(self):
        return True

    def readinto(self, buffer):
        while not self.pending:
            chunk = self.source.read(65536)
            if not chunk:
                return 0
            self.pending = self.decoder.decompress(chunk)
        count = min(len(buffer), len(self.pending))
        buffer[:count] = self.pending[:count]
        self.pending = self.pending[count:]
        return count

    def close(self):
        self.source.close()
        super().close()


def prepare_credits(catalogue, data):
    credits, people = {}, set()
    with gzip.open(data / CREW, 'rt', encoding='utf-8', newline='') as stream:
        if stream.readline().rstrip() != 'tconst\tdirectors\twriters':
            raise ValueError('Unexpected IMDb crew columns.')
        for line in stream:
            title, directors, writers = line.rstrip('\n').split('\t')
            if title not in catalogue.imdb:
                continue
            roles = [list(dict.fromkeys(value.split(','))) if value != r'\N' else []
                     for value in (directors, writers)]
            if any(roles):
                credits[title] = roles
                people.update(person for role in roles for person in role)
    names = {}
    with gzip.open(data / PEOPLE, 'rt', encoding='utf-8', newline='') as stream:
        if not stream.readline().startswith('nconst\tprimaryName\t'):
            raise ValueError('Unexpected IMDb name columns.')
        for line in stream:
            person, name, _ = line.split('\t', 2)
            if person in people and name != r'\N':
                names[person] = name
    write_json(data / CREDITS_CACHE, {'credits': credits, 'names': names})
    print(f'Cached credits for {len(credits):,} catalogue entries and {len(names):,} names.', flush=True)


def prepare_genome(catalogue, data):
    columns = {tag: index for index, tags in enumerate(TRAITS.values()) for tag in tags}
    rows = {}
    with TextIOWrapper(BufferedReader(DeflateReader(data / GENOME)), encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream)
        if next(reader) != ['tag', 'item_id', 'score']:
            raise ValueError('Unexpected Tag Genome columns.')
        for tag, item, score in reader:
            column = columns.get(tag)
            if column is None:
                continue
            movie_id = int(item)
            if movie_id not in catalogue.ml:
                continue
            values = rows.setdefault(movie_id, np.zeros(len(TRAITS), dtype=np.float32))
            values[column] = max(values[column], float(score))
    if not rows:
        raise ValueError('No Tag Genome records joined to MovieLens.')
    with NamedTemporaryFile(dir=data, suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
    np.savez_compressed(temporary, movie_ids=np.array(list(rows), dtype=np.int64),
                        values=np.stack(list(rows.values())), labels=np.array(list(TRAITS)))
    temporary.replace(data / GENOME_CACHE)
    print(f'Cached {len(TRAITS)} Tag Genome traits for {len(rows):,} linked films.', flush=True)


@dataclass
class Block:
    source: str
    labels: tuple
    values: np.ndarray
    present: np.ndarray
    penalty: float
    spoiler_columns: tuple = ()
    row_lookup: np.ndarray | None = None

    def rows(self, indices):
        return indices if self.row_lookup is None else self.row_lookup[indices]

    def fit(self, indices, residual, sample_weights=None):
        valid = self.present[indices]
        if valid.sum() < 5:
            return None
        rows = self.values[self.rows(indices[valid])].astype(np.float64)
        target = residual[valid]
        if sample_weights is None:
            center = rows.mean(axis=0)
            scale = np.maximum(rows.std(axis=0), 0.32)
            target = target - target.mean()
        else:
            covered_weights = sample_weights[valid]
            center = np.average(rows, axis=0, weights=covered_weights)
            variance = np.average((rows - center) ** 2, axis=0, weights=covered_weights)
            scale = np.maximum(np.sqrt(variance), 0.32)
            target = target - np.average(target, weights=covered_weights)
        design = (rows - center) / scale
        active = (np.abs(design) > 0.1).sum(axis=0) >= 3
        if not active.any():
            return None
        design = design[:, active]
        if sample_weights is not None:
            root_weights = np.sqrt(covered_weights)
            design = design * root_weights[:, None]
            target = target * root_weights
        weights = np.zeros(rows.shape[1])
        weights[active] = np.linalg.solve(
            design.T @ design + self.penalty * np.eye(design.shape[1]), design.T @ target
        ) / scale[active]
        return center, weights

    def predict(self, fitted):
        center, weights = fitted
        effect = np.einsum('ij,j->i', self.values, weights, dtype=np.float64) - center @ weights
        if self.row_lookup is None:
            return np.where(self.present, effect, 0)
        prediction = np.zeros(len(self.present))
        prediction[self.present] = effect
        return prediction


class CreditFeatures:
    def __init__(self, catalogue, path):
        source = json.loads(path.read_text(encoding='utf-8'))
        self.names = source['names']
        self.by_movie = {}
        positions, people, roles, shares = [], [], [], []
        for imdb, credits in source['credits'].items():
            index = catalogue.imdb.get(imdb)
            if index is None:
                continue
            self.by_movie[index] = credits
            for role, ids in enumerate(credits):
                for person in ids:
                    positions.append(index)
                    people.append(person)
                    roles.append(role)
                    shares.append(1 / len(ids) ** 0.5)
        self.positions = np.asarray(positions, dtype=np.int64)
        self.person_ids, self.people = np.unique(people, return_inverse=True)
        self.roles = np.asarray(roles, dtype=np.int64)
        self.shares = np.asarray(shares)
        self.lookup = {person: i for i, person in enumerate(self.person_ids)}
        self.size = len(catalogue.movies)

    def fit(self, indices, residual, sample_weights):
        evidence = [defaultdict(list), defaultdict(list)]
        for index, delta, weight in zip(indices, residual, sample_weights):
            for role, people in enumerate(self.by_movie.get(index, ((), ()))):
                for person in people:
                    evidence[role][person].append((int(index), float(delta * weight), float(weight)))
        affinity = np.zeros((2, len(self.lookup)))
        totals = [{}, {}]
        for role, groups in enumerate(evidence):
            for person, observations in groups.items():
                numerator = sum(delta for _, delta, _ in observations)
                mass = sum(weight for _, _, weight in observations)
                totals[role][person] = (numerator, mass)
                affinity[role, self.lookup[person]] = numerator / (mass + 4)
        return (affinity, evidence), totals

    def predict(self, fitted):
        affinity, _ = fitted
        return np.bincount(self.positions, weights=affinity[self.roles, self.people] * self.shares,
                           minlength=self.size)

    def effects(self, index, fitted):
        affinity, evidence = fitted
        for role, people in enumerate(self.by_movie.get(index, ((), ()))):
            for person in people:
                observations = evidence[role].get(person)
                if observations:
                    yield (affinity[role, self.lookup[person]] / len(people) ** 0.5, role, person, observations)

    def display(self, index):
        return [[self.names[person] for person in people if person in self.names]
                for people in self.by_movie.get(index, ((), ()))]


@dataclass
class TasteFit:
    parts: list
    credits: tuple | None
    ratings: dict


class Taste:
    def __init__(self, catalogue, data, mood, audience, counts=None):
        self.catalogue = catalogue
        self.blocks = self.make_blocks(catalogue, mood)
        path = data / CREDITS_CACHE
        self.credits = CreditFeatures(catalogue, path) if path.is_file() else None
        self.genome = self.load_genome(catalogue, data / GENOME_CACHE)
        # Bound the public-rating contribution to one eighth of a star.
        self.audience = np.clip(audience - np.mean(audience), -1, 1) * 0.125
        if counts is not None:
            self.audience *= counts / (counts + 25)

    @staticmethod
    def make_blocks(catalogue, mood):
        movies = catalogue.movies
        genres = [tuple(sorted({genre.lower().replace('sci-fi', 'science fiction') for genre in m.genres})) for m in movies]
        combinations_by_movie = [list(combinations(names, 2)) for names in genres]
        common = Counter(pair for pairs in combinations_by_movie for pair in pairs)
        feature_names = sorted({name for names in genres for name in names}) + [
            ' + '.join(pair) for pair, count in sorted(common.items()) if count >= 250
        ]
        lookup = {label: col for col, label in enumerate(feature_names)}
        values = np.zeros((len(movies), len(feature_names)), dtype=np.float32)
        for row, (names, pairs) in enumerate(zip(genres, combinations_by_movie)):
            for label in [*names, *(' + '.join(pair) for pair in pairs if common[pair] >= 250)]:
                values[row, lookup[label]] = 0.7 if ' + ' in label else 1
        mood_terms = np.zeros((len(mood.values), len(MOOD_TERMS)), dtype=np.float32)
        for row, movie in enumerate(movies):
            if movie.mood:
                tags = set(json.loads(movie.mood['mood_tags'] or '[]'))
                mood_terms[mood.rows[row]] = [bool(tags & terms) for terms in MOOD_TERMS.values()]
        arcs = {movie.mood['emotional_arc'] for movie in movies if movie.mood}
        genre_labels = {genre for movie in movies for genre in movie.genres}
        retained = [index for index, label in enumerate(mood.names) if label not in arcs | genre_labels]
        labels = tuple(mood.names[index] for index in retained) + tuple(MOOD_TERMS)
        features = np.column_stack((np.nan_to_num(mood.values[:, retained]), mood_terms))
        return [Block('Genre', tuple(feature_names), values, values.any(axis=1), 40),
                Block('Mooduel', labels, features, catalogue.mood_present, 55, row_lookup=mood.rows)]

    @staticmethod
    def load_genome(catalogue, path):
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as cache:
            labels = tuple(cache['labels'].tolist())
            values = np.zeros((len(catalogue.movies), len(labels)), dtype=np.float32)
            present = np.zeros(len(values), dtype=bool)
            for movie_id, scores in zip(cache['movie_ids'], cache['values']):
                if movie_id in catalogue.ml:
                    index = catalogue.ml[movie_id]
                    values[index], present[index] = scores, True
        spoiler = tuple(i for i, label in enumerate(labels) if 'payoff' in label)
        return Block('Tag Genome', labels, values, present, 40, spoiler)

    def fit(self, model, history, indices, ratings, sample_weights=None):
        weights = np.ones(len(indices)) if sample_weights is None else sample_weights
        base = float(((weights * ratings).sum() + 5 * 3.3) / (weights.sum() + 5))
        residual = ratings - base - self.audience[indices]
        if self.credits and model in {'taste_credits', 'taste'}:
            credits, totals = self.credits.fit(indices, residual, weights)
            credit_fit = self.credits.predict(credits)
            # Bound co-credit accumulation and remove the in-sample contribution before fitting traits.
            credit_fit = np.clip(credit_fit, -1.15, 1.15)
            remaining = residual.copy()
            for row, (index, delta, weight) in enumerate(zip(indices, residual, weights)):
                contribution = 0.0
                for role, people in enumerate(self.credits.by_movie.get(index, ((), ()))):
                    for person in people:
                        numerator, mass = totals[role][person]
                        contribution += (numerator - weight * delta) / (mass - weight + 4) / len(people) ** 0.5
                remaining[row] -= np.clip(contribution, -1.15, 1.15)
        else:
            credits = None
            credit_fit = 0
            remaining = residual
        prediction = base + self.audience + credit_fit
        blocks = list(self.blocks)
        if self.genome is not None and model in {'taste', 'taste_genome'}:
            blocks.append(self.genome)
        parts = []
        for block in blocks:
            fitted = block.fit(indices, remaining, sample_weights)
            if fitted is not None:
                effect = np.clip(block.predict(fitted), -0.8, 0.8)
                remaining -= effect[indices]
                prediction = prediction + effect
                parts.append((block, fitted))
        return prediction, TasteFit(parts, credits, history)

    def explain(self, index, fit):
        messages = []
        if fit.credits is not None:
            contributions = sorted(self.credits.effects(index, fit.credits), key=lambda row: -abs(row[0]))
            for value, role, person, observations in contributions[:2]:
                name = self.credits.names.get(person)
                if not name or abs(value) < 0.035:
                    continue
                # Cite different rated films when scoring already-watched candidates.
                examples = [(i, delta) for i, delta, _ in observations if i != index and delta * value > 0]
                if examples:
                    example = max(examples, key=lambda row: abs(row[1]))[0]
                    title = self.catalogue.movies[example].title
                    details = f', including {title} ({fit.ratings[example]:g}★)'
                else:
                    details = ''
                label = 'Director' if role == 0 else 'Writing credit'
                support = f'{len(observations)} rated film' + ('s' if len(observations) != 1 else '')
                messages.append(f'{label}: {name} ({value:+.2f} before credit cap, {support}{details}).')
        strongest = []
        for block, (center, weights) in fit.parts:
            if block.present[index]:
                effects = (block.values[block.rows(index)] - center) * weights
                for col in np.argsort(-np.abs(effects))[:5]:
                    if col not in block.spoiler_columns and abs(effects[col]) >= 0.035:
                        strongest.append((abs(effects[col]), f'{block.labels[col]} ({effects[col]:+.2f}, {block.source})'))
        if strongest:
            messages.append('Taste signals: ' + ', '.join(text for _, text in sorted(strongest, reverse=True)[:3]) + '.')
        return ' '.join(messages) or 'There is limited matching taste evidence for this film.'


if __name__ == '__main__':
    import argparse
    from catalog import Catalogue
    from snapshots import DATA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--only', choices=('credits', 'genome'))
    args = parser.parse_args()
    catalogue = Catalogue(args.data)
    if args.only in (None, 'credits'):
        prepare_credits(catalogue, args.data)
    if args.only in (None, 'genome'):
        prepare_genome(catalogue, args.data)
