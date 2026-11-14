"""Auditable baselines, MovieLens fold-in, and two-person ranking."""

from dataclasses import dataclass
import json

import numpy as np

from snapshots import ML, MOOD, digest
from taste import Taste, TasteFit
from train import MODEL


CLEAN = ('random', 'audience', 'mooduel')
MOVIELENS = ('popularity', 'ml_audience', 'collaborative', 'collaborative_mooduel')
TASTE = ('taste_traits', 'taste_credits', 'taste', 'taste_genome')
NUMERIC = ('valence', 'arousal', 'dominance', 'absorption', 'hedonic', 'eudaimonic',
           'psych_rich', 'comfort_level', 'conversation_potential')
CATEGORIES = ('pacing', 'emotional_arc')


def ridge(design, ratings, penalties):
    return np.linalg.solve(design.T @ design + np.diag(penalties), design.T @ ratings)


class MoodFeatures:
    def __init__(self, movies):
        covered = [movie for movie in movies if movie.mood]
        self.rows = np.full(len(movies), -1, dtype=np.int32)
        present = np.fromiter((movie.mood is not None for movie in movies), dtype=bool, count=len(movies))
        self.rows[present] = np.arange(len(covered), dtype=np.int32)
        self.names = [name.replace('_', ' ') for name in NUMERIC]
        tokens = set()
        for movie in covered:
            tokens.update(movie.mood[name] for name in CATEGORIES if movie.mood[name])
            tokens.update(movie.genres)
            tokens.update(json.loads(movie.mood['dominant_emotions'] or '[]'))
        self.names += sorted(tokens)
        lookup = {name: k for k, name in enumerate(self.names)}
        self.values = np.zeros((len(covered), len(self.names)), dtype=np.float32)
        for index, movie in enumerate(covered):
            for k, field in enumerate(NUMERIC):
                self.values[index, k] = float(movie.mood[field]) if movie.mood[field] else np.nan
            flags = {*movie.genres, *json.loads(movie.mood['dominant_emotions'] or '[]'),
                     *(movie.mood[name] for name in CATEGORIES if movie.mood[name])}
            for token in flags:
                self.values[index, lookup[token]] = 0.6

    def fit(self, indices, errors, present):
        known = present[indices]
        if known.sum() < 3:
            return None
        train = self.values[self.rows[indices[known]]].astype(np.float64)
        count = np.isfinite(train[:, :len(NUMERIC)]).sum(axis=0)
        center = np.divide(np.nansum(train[:, :len(NUMERIC)], axis=0), np.maximum(count, 1))
        train[:, :len(NUMERIC)] = np.nan_to_num(train[:, :len(NUMERIC)] - center)
        weights = ridge(train, errors[known], np.full(train.shape[1], 22.0))
        return center, weights

    def effects(self, fit, indices=None):
        center, weights = fit
        x = self.values if indices is None else self.values[self.rows[indices]]
        x = x.copy()
        if indices is not None:
            x = np.where(np.expand_dims(self.rows[indices] >= 0, axis=-1), x, 0)
        x[..., :len(NUMERIC)] = np.nan_to_num(x[..., :len(NUMERIC)] - center)
        return x * weights

    def predict(self, fit, present):
        center, weights = fit
        numeric = np.nan_to_num(self.values[:, :len(NUMERIC)] - center)
        result = numeric @ weights[:len(NUMERIC)] + np.einsum(
            'ij,j->i', self.values[:, len(NUMERIC):], weights[len(NUMERIC):], dtype=np.float64
        )
        prediction = np.zeros(len(present))
        prediction[present] = result
        return prediction


@dataclass
class Scores:
    stars: np.ndarray
    rank: np.ndarray
    mood_fit: tuple | None = None
    collaborative_fit: bool = False
    taste_fit: TasteFit | None = None


class Models:
    def __init__(self, catalogue, data, use_movielens=True):
        self.catalogue = catalogue
        self.present = catalogue.mood_present
        self.tmdb = np.array([m.tmdb_rating / 2 if m.tmdb_rating else np.nan for m in catalogue.movies])
        self.tmdb_prior = float(np.nanmean(self.tmdb))
        self.tmdb = np.where(np.isfinite(self.tmdb), self.tmdb, self.tmdb_prior)
        self.mood = MoodFeatures(catalogue.movies)
        self.names = CLEAN
        self.metadata = None
        self.coverage = None
        if use_movielens and (data / MODEL).is_file():
            with np.load(data / MODEL, allow_pickle=False) as model:
                self.metadata = json.loads(model['metadata'].item())
                if (self.metadata['movies'] != len(catalogue.movies)
                        or self.metadata['mooduel_sha256'] != digest(data / MOOD)
                        or self.metadata['movielens_sha256'] != digest(data / ML)):
                    raise ValueError('MovieLens model cache and snapshots differ. Run train.py again.')
                self.counts = model['counts']
                totals = model['totals']
                self.bias = model['bias']
                self.factors = model['factors']
            self.global_mean = float(totals.sum() / self.counts.sum())
            self.ml_mean = (totals + 30 * self.global_mean) / (self.counts + 30)
            self.coverage = 0.4 + 0.6 * self.counts / (self.counts + 30)
            self.names += MOVIELENS
        self.taste = Taste(catalogue, data, self.mood, self.tmdb, self.counts if self.metadata else None)
        # Feature matrices now contain every Mooduel field used by scoring. Retain only
        # the arc needed by explanations instead of every original CSV string.
        for movie in catalogue.movies:
            if movie.mood:
                movie.mood = {'emotional_arc': movie.mood['emotional_arc']}
        self.names += ('taste_traits',)
        if self.taste.credits is not None:
            self.names += ('taste_credits',)
        if self.taste.genome is not None and self.taste.credits is not None:
            self.names += ('taste',)
        elif self.taste.genome is not None:
            self.names += ('taste_genome',)

    @staticmethod
    def audience(base, indices, ratings):
        delta = np.sum(ratings - base[indices]) / (len(ratings) + 12)
        return base + delta

    def score(self, model, history, seed=42):
        indices = np.fromiter(sorted(history), dtype=np.int64)
        ratings = np.array([history[i] for i in indices])
        if model in TASTE and model in self.names:
            prediction, taste_fit = self.taste.fit(model, history, indices, ratings)
            return Scores(np.clip(prediction, 0.5, 5.0), prediction, taste_fit=taste_fit)
        audience = self.audience(self.tmdb, indices, ratings)
        fitted_mood = None
        collaborative_fit = False
        if model in ('random', 'audience', 'mooduel'):
            predicted = audience
        elif model in self.names:
            base = np.where(self.counts > 0, self.ml_mean, self.tmdb)
            predicted = self.audience(base, indices, ratings)
            warm = self.counts[indices] >= 5
            if model.startswith('collaborative') and warm.sum() >= 3:
                collaborative_fit = True
                baseline = self.global_mean + self.bias
                design = np.column_stack((np.ones(warm.sum()), self.factors[indices[warm]]))
                penalties = [12.0, *([6.0] * self.factors.shape[1])]
                user = ridge(design, ratings[warm] - baseline[indices[warm]], penalties)
                collaborative = baseline + user[0] + self.factors @ user[1:]
                predicted = np.where(self.counts >= 5, collaborative, predicted)
        else:
            raise ValueError(f'Model unavailable: {model}')
        if model in ('mooduel', 'collaborative_mooduel'):
            fitted_mood = self.mood.fit(indices, ratings - predicted[indices], self.present)
            if fitted_mood is not None:
                predicted = predicted + self.mood.predict(fitted_mood, self.present)
        stars = np.clip(predicted, 0.5, 5.0)
        ranks = predicted
        if model == 'random':
            ranks = np.random.default_rng(seed).random(len(stars))
        if model == 'popularity':
            ranks = self.counts
        return Scores(stars, ranks, fitted_mood, collaborative_fit)

    def explain(self, scores, index):
        if scores.taste_fit is not None:
            return self.taste.explain(index, scores.taste_fit)
        if scores.mood_fit is not None and self.present[index]:
            effects = self.mood.effects(scores.mood_fit, index)
            arc = self.catalogue.movies[index].mood['emotional_arc']
            strong = [k for k in np.argsort(-np.abs(effects), kind='stable') if self.mood.names[k] != arc][:2]
            details = ', '.join(f'{self.mood.names[k]} ({effects[k]:+.2f} stars)' for k in strong if abs(effects[k]) >= 0.03)
            if details:
                return f'Mooduel traits contributing to this profile’s estimate: {details}.'
        if scores.collaborative_fit and self.counts[index] >= 5:
            return 'The estimate uses MovieLens rating patterns and the imported ratings.'
        return 'The estimate uses snapshot audience ratings and the imported profile’s rating scale.'

    def supported_rank(self, rank, reference):
        if self.coverage is None:
            return rank
        # MovieLens coverage is a separate signal, not a TMDB vote count.
        # Retain some lift for films missing from the older MovieLens snapshot.
        return reference + self.coverage * (rank - reference)

    def individual_rank(self, model, score, history):
        if model in TASTE or model in {'random', 'popularity'}:
            return score.rank
        return self.supported_rank(score.rank, score.rank[list(history)].mean())


def rank_candidates(scores, allowed):
    candidates = np.flatnonzero(allowed)
    return candidates[np.argsort(-scores[candidates], kind='stable')]


def pair_score(first, second):
    return 0.5 * (first + second) - 0.25 * np.abs(first - second)
