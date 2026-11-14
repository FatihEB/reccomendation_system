"""Compare the selected Tag Genome traits with human SoftAttributes judgments."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from catalog import Catalogue, movie_title
from snapshots import DATA, digest
from taste import GENOME_CACHE, Taste, write_json
from taste_data import SOFT


LABELS = {'slow-paced': 'slow pacing', 'confusing': 'perceived confusion',
          'complicated': 'complex plotting', 'original': 'originality',
          'intense': 'emotional intensity', 'realistic': 'perceived realism'}


def compare(data):
    catalogue = Catalogue(data)
    traits = Taste.load_genome(catalogue, data / GENOME_CACHE)
    scores, totals = Counter(), Counter()
    matched = {}

    def lookup(value):
        if value not in matched:
            title, year = movie_title(value)
            matched[value] = catalogue.resolve({'Name': title, 'Year': year})[0]
        return matched[value]

    with (data / SOFT).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            label = LABELS.get(row['soft_attribute'])
            if label is None:
                continue
            anchor = lookup(row['reference_title'])
            if anchor is None or not traits.present[anchor]:
                continue
            column = traits.labels.index(label)
            for field, sign in [('more_than', 1), ('less_than', -1)]:
                for title in json.loads(row[field] or '[]'):
                    index = lookup(title)
                    if index is None or not traits.present[index]:
                        continue
                    margin = float(traits.values[index, column] - traits.values[anchor, column]) * sign
                    totals[label] += 1
                    scores[label] += float(margin > 0) + 0.5 * float(margin == 0)
    return {'agreement': {label: {'pairs': n, 'agreement': scores[label] / n} for label, n in sorted(totals.items())},
            'sha256': {name: digest(data / name) for name in (SOFT, GENOME_CACHE)}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    args = parser.parse_args()
    report = compare(args.data)
    write_json(args.data / 'soft-attribute-check.json', report)
    print(json.dumps(report['agreement'], indent=2))
