"""Retrospective full-catalogue comparison on imported Letterboxd histories."""

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from catalog import Catalogue, import_profile
from models import CLEAN, MOVIELENS, TASTE, TIME_WEIGHTED_TASTE, Models, rank_candidates
from snapshots import DATA, ML, MOOD, digest
from taste import CREDITS_CACHE, GENOME_CACHE


SEEDS = (171, 281, 391)
TOP = (10, 20, 50, 100)


@dataclass
class Split:
    suite: str
    key: str
    targets: list


def sample(indices, rng, minimum=1):
    if len(indices) <= minimum:
        return []
    size = max(1, round(len(indices) * 0.2))
    return sorted(rng.choice(indices, size=size, replace=False).tolist())


def splits(profile):
    if len(profile.favorites) == 4:
        for number, target in enumerate(profile.favorites, 1):
            yield Split('favorite_one', str(number), [target])
        yield Split('favorite_all', 'all', profile.favorites)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        high = [i for i in sorted(profile.ratings) if profile.ratings[i] >= 4.5]
        yield Split('high_rating', str(seed), [(None, i) for i in sample(high, rng)])
        by_level = {level: [] for level in np.arange(0.5, 5.5, 0.5)}
        for index, rating in sorted(profile.ratings.items()):
            by_level[rating].append(index)
        holdout = [index for group in by_level.values() for index in sample(group, rng, minimum=0)]
        yield Split('rating_stratified', str(seed), [(None, i) for i in sorted(holdout)])


def mean(values):
    return float(np.mean(values)) if len(values) else None


def matched_pairs(models, profile, hidden, allowed):
    if models.metadata is None:
        return []
    pairs = []
    for first, second in combinations(sorted(hidden), 2):
        first_rating, second_rating = profile.ratings[first], profile.ratings[second]
        if not (allowed[first] and allowed[second] and abs(first_rating - second_rating) >= 1.5):
            continue
        if min(models.counts[first], models.counts[second]) < 25:
            continue
        if abs(models.ml_mean[first] - models.ml_mean[second]) > 0.25:
            continue
        popularity_ratio = max(models.counts[first], models.counts[second]) / min(models.counts[first], models.counts[second])
        if popularity_ratio <= 2:
            pairs.append((first, second) if first_rating > second_rating else (second, first))
    return pairs


def evaluate_profile(catalogue, models, profile, identifier, detailed, app_order=False, selected=None, as_of=None):
    output = []
    allowed_base = catalogue.eligible.copy()
    for split in splits(profile):
        if not split.targets:
            continue
        hidden = {index for _, index in split.targets if index is not None}
        history = {index: rating for index, rating in profile.ratings.items() if index not in hidden}
        if not history:
            continue
        allowed = allowed_base.copy()
        allowed[list(profile.watched - hidden)] = False
        pairs = matched_pairs(models, profile, hidden, allowed) if split.suite == 'rating_stratified' else []
        for model in selected or models.names:
            scored = models.score(model, history, seed=42, rating_dates=profile.rating_dates, as_of=as_of)
            ordering = models.individual_rank(model, scored, history) if app_order else scored.rank
            order = rank_candidates(ordering, allowed)
            ranks = np.zeros(len(allowed), dtype=np.int32)
            ranks[order] = np.arange(1, len(order) + 1)
            targets = []
            for target, index in split.targets:
                rank = (int(ranks[index]) or None) if index is not None else None
                actual = profile.ratings.get(index)
                predicted = float(scored.stars[index]) if rank else None
                status = 'eligible' if rank else ('unmatched' if index is None else 'ineligible')
                row = {'status': status, 'rank': rank, 'actual': actual, 'prediction': predicted}
                if detailed:
                    row['title'] = target or catalogue.movies[index].title
                    row['catalogue_index'] = index
                    row['percentile'] = 100 * (len(order) - rank) / max(1, len(order) - 1) if rank else None
                targets.append(row)
            pair_scores = [float(ordering[high] > ordering[low]) + 0.5 * float(ordering[high] == ordering[low])
                           for high, low in pairs]
            record = {'profile': identifier, 'suite': split.suite, 'key': split.key, 'model': model,
                      'candidates': len(order), 'history': len(history), 'targets': targets,
                      'matched_pairs': len(pairs), 'matched_pair_credit': sum(pair_scores)}
            if detailed and pairs:
                record['pair_manifest'] = [[high, low, result] for (high, low), result in zip(pairs, pair_scores)]
            output.append(record)
    return output


def summarize(rows):
    results = {}
    for suite in sorted({row['suite'] for row in rows}):
        results[suite] = {}
        for model in sorted({row['model'] for row in rows}):
            records = [row for row in rows if row['suite'] == suite and row['model'] == model]
            targets = [target for row in records for target in row['targets']]
            eligible = [t for t in targets if t['rank'] is not None]
            observed = [t for t in eligible if t['actual'] is not None]
            errors = np.array([t['prediction'] - t['actual'] for t in observed])
            ranks = [t['rank'] for t in eligible]
            n = len(targets)
            if not n:
                continue
            def ndcg(record):
                recovered = [t['rank'] for t in record['targets'] if t['rank'] and t['rank'] <= 100]
                ideal = np.arange(1, min(len(record['targets']), 100) + 1)
                return float(sum(1 / np.log2(rank + 1) for rank in recovered) / (1 / np.log2(ideal + 1)).sum())

            results[suite][model] = {
                'targets': n, 'eligible': len(eligible), 'numeric': len(observed),
                'mean_candidates': mean([row['candidates'] for row in records]),
                'median_eligible_rank': float(np.median(ranks)) if ranks else None,
                'mrr': sum(1 / rank for rank in ranks) / n,
                **{f'recall_at_{k}': sum(rank <= k for rank in ranks) / n for k in TOP},
                **{f'eligible_recall_at_{k}': sum(rank <= k for rank in ranks) / len(ranks) if ranks else None for k in TOP},
                'ndcg_at_100': mean([ndcg(record) for record in records]) if suite != 'rating_stratified' else None,
                'mae': mean(np.abs(errors)), 'rmse': float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                'known_dislikes_in_top_100': sum(t['actual'] is not None and t['actual'] <= 2 and t['rank'] <= 100 for t in eligible),
            }
            if suite == 'rating_stratified':
                pair_count = sum(record['matched_pairs'] for record in records)
                results[suite][model]['matched_pairs'] = pair_count
                results[suite][model]['matched_pair_accuracy'] = (
                    sum(record['matched_pair_credit'] for record in records) / pair_count if pair_count else None
                )
                per_profile = Counter()
                correct = Counter()
                for record in records:
                    per_profile[record['profile']] += record['matched_pairs']
                    correct[record['profile']] += record['matched_pair_credit']
                results[suite][model]['matched_pair_macro_accuracy'] = mean(
                    [correct[profile] / n for profile, n in per_profile.items() if n]
                )
                results[suite][model]['by_level'] = {
                    str(level): {'n': len(group), 'mae': mean([abs(t['prediction'] - level) for t in group])}
                    for level in sorted({t['actual'] for t in observed})
                    for group in [[t for t in observed if t['actual'] == level]]
                }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exports', type=Path, nargs='+')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=DATA / 'evaluation.json')
    parser.add_argument('--details', type=Path, help='Private split manifests, per-title ranks and import match audit.')
    parser.add_argument('--favorites', type=Path, help='JSON array of four favorite strings per input export, in input order. Use null for an export with pinned favorites.')
    parser.add_argument('--clean-only', action='store_true', help='Disable models using MovieLens ratings.')
    parser.add_argument('--app-order', action='store_true', help='Apply the same individual ordering as the app.')
    parser.add_argument('--models', nargs='+', help='Limit the comparison to named models.')
    parser.add_argument('--release-cutoff-year', type=int, help='Exclude this year and later. Defaults to the current year.')
    parser.add_argument('--rating-as-of', type=date.fromisoformat, default=date.today(),
                        help='Reference date for time weights (YYYY-MM-DD). Defaults to today.')
    args = parser.parse_args()
    catalogue = Catalogue(args.data, args.release_cutoff_year)
    models = Models(catalogue, args.data, use_movielens=not args.clean_only)
    available = [model for model in models.names if not (args.clean_only and model in {'taste', TIME_WEIGHTED_TASTE})]
    selected = args.models or available
    if set(selected) - set(available):
        parser.error('Unavailable models: ' + ', '.join(sorted(set(selected) - set(available))))
    overrides = json.loads(args.favorites.read_text(encoding='utf-8')) if args.favorites else [None] * len(args.exports)
    if not isinstance(overrides, list) or len(overrides) != len(args.exports):
        parser.error('Provide one favorites list or null per export.')
    for override in overrides:
        if override is not None and (not isinstance(override, list) or len(override) != 4
                                     or not all(isinstance(value, str) and value.strip() for value in override)):
            parser.error('Each manual favorites list must contain four titles with years or known export URIs.')
    rows, coverage, audits = [], {}, {}
    for i, (export, favorites) in enumerate(zip(args.exports, overrides), 1):
        identifier = f'P{i}'
        profile = import_profile(catalogue, export, manual_favorites=favorites or ())
        if len(profile.favorites) != 4:
            print(f'{identifier}: provide four pinned favorites with --favorites to run favorite recovery.')
        rows.extend(evaluate_profile(catalogue, models, profile, identifier, bool(args.details),
                                     args.app_order, selected, args.rating_as_of))
        coverage[identifier] = {'matched_ratings': len(profile.ratings), 'matched_watched': len(profile.watched),
                                'favorite_entries': len(profile.favorites),
                                'matched_favorites': sum(index is not None for _, index in profile.favorites),
                                'rating_row_matching': dict(Counter(record[-1] for record in profile.audit if record[0] == 'ratings.csv'))}
        if args.details:
            audits[identifier] = {'export_sha256': digest(export), 'match_rows': profile.audit}
    manifest = {'seeds': SEEDS, 'holdout_fraction': 0.2, 'minimum_high_rating': 4.5,
                'catalogue_size': len(catalogue.movies), 'eligible_size': int(catalogue.eligible.sum()),
                'release_cutoff_year': catalogue.release_cutoff_year,
                'ambiguous_movielens_tmdb_ids': getattr(catalogue, 'ambiguous_tmdb_links', 0),
                'mooduel_sha256': digest(args.data / MOOD),
                'movielens_sha256': digest(args.data / ML) if (args.data / ML).is_file() else None,
                'population_model': models.metadata, 'clean_models': CLEAN,
                'movielens_models': [model for model in MOVIELENS if model in selected],
                'taste_models': [model for model in TASTE if model in selected],
                'rating_as_of': args.rating_as_of.isoformat(),
                'app_order': args.app_order,
                'matched_pair_protocol': {'minimum_personal_gap': 1.5, 'max_audience_gap': 0.25,
                                          'minimum_ratings': 25, 'maximum_popularity_ratio': 2},
                'taste_data_sha256': {name: digest(args.data / name) for name in (CREDITS_CACHE, GENOME_CACHE)
                                      if (args.data / name).is_file()}}
    summary = {'manifest': manifest, 'coverage': coverage, 'aggregate': summarize(rows),
               'per_profile': {profile: summarize([row for row in rows if row['profile'] == profile]) for profile in coverage}}
    for path, content in [(args.output, summary), (args.details, {'manifest': manifest, 'audits': audits, 'splits': rows})]:
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content, indent=2) + '\n', encoding='utf-8')
            print(f'Wrote {path}')


if __name__ == '__main__':
    main()
