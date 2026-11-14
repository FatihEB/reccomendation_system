"""Compare two offline film-review extractors on locally annotated excerpts."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from snapshots import DATA, digest
from taste import write_json
from taste_data import BERT_REVISION, QWEN_REVISION, QWEN


ASPECTS = {
    'pacing': 'speed or momentum, whether it worked',
    'suspense': 'tension, anticipation, or dread felt by the viewer',
    'payoff': 'satisfaction with an ending, reveal, or setup paying off',
    'dialogue': 'naturalness, wit, exposition, or a particular line or monologue',
    'complexity': 'nonlinearity, predictability, or how hard it was to follow',
    'coherence': 'whether the story holds together logically',
    'political_presentation': 'reaction to how a political theme is presented',
    'characters': 'character engagement, characterization, or acting',
    'atmosphere': 'the overall mood or atmosphere experienced',
    'emotional_impact': 'emotion felt by the viewer, apart from whether they liked it',
    'emotional_authenticity': 'whether emotions or reactions feel believable',
    'originality': 'freshness or familiarity of the film or story, excluding its title',
    'visuals': 'cinematography, editing as visual style, or other visual craft',
    'sound': 'music, soundtrack, or sound design',
    'direction': 'explicit evaluation of a director’s work',
}
BROAD = {
    'Cinematography': ('visuals',),
    'Direction': ('direction',),
    'Story': ('pacing', 'suspense', 'payoff', 'dialogue', 'complexity', 'coherence', 'political_presentation'),
    'Characters': ('characters',),
    'Production Design': (),
    'Unique Concept': ('originality',),
    'Emotions': ('emotional_impact', 'emotional_authenticity'),
}
SENTIMENT = ('positive', 'negative', 'mixed', 'neutral', 'unknown')
SEGMENTS = ('opening', 'middle', 'ending', 'whole', 'unknown')
SCHEMA = {
    'type': 'object', 'required': ['claims'], 'additionalProperties': False,
    'properties': {'claims': {'type': 'array', 'maxItems': 5, 'items': {
        'type': 'object', 'required': ['aspect', 'sentiment', 'segment', 'evidence'], 'additionalProperties': False,
        'properties': {'aspect': {'enum': list(ASPECTS)}, 'sentiment': {'enum': list(SENTIMENT)},
                       'segment': {'enum': list(SEGMENTS)}, 'evidence': {'type': 'string'}}
    }}}
}


def qwen(text, url):
    prompt = (
        'Extract only explicit film-aspect observations or reactions by the reviewer. Skip plot retelling, general praise, '
        'jokes about unrelated things, and factual mentions of who directed or acted. '
        'Output an object with claims, an array of objects with aspect, sentiment, segment and evidence. '
        'Evidence must be an exact contiguous quote from the input, with enough words to support the claim. '
        'Use unknown sentiment when the experience is described but there is no praise or criticism. '
        'Use mixed if praise and criticism concern the same aspect. Use whole when no particular section is mentioned. '
        'Use opening, middle or ending only if the review specifically indicates it. Return an empty array when unsupported. '
        'Allowed aspects: ' + '; '.join(f'{key} = {value}' for key, value in ASPECTS.items()) + '. '
        'Allowed sentiments: ' + ', '.join(SENTIMENT) + '. Allowed segments: ' + ', '.join(SEGMENTS) + '.'
    )
    payload = {
        'messages': [{'role': 'system', 'content': prompt},
                     {'role': 'user', 'content': 'Review excerpt as a JSON string:\n' + json.dumps(text, ensure_ascii=False)}],
        'chat_template_kwargs': {'enable_thinking': False}, 'temperature': 0, 'seed': 42, 'max_tokens': 360,
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'film_review', 'schema': SCHEMA}},
    }
    if callable(url):
        answer = url(**payload)['choices'][0]
    else:
        answer = qwen_request(url, payload)
    if answer['finish_reason'] != 'stop':
        return [], 0, True
    try:
        raw = json.loads(answer['message']['content'])['claims']
    except (ValueError, KeyError, TypeError):
        return [], 0, True
    valid = []
    for item in raw:
        evidence = item.get('evidence', '').strip()
        if (evidence in text and len(evidence.split()) >= 2 and item.get('aspect') in ASPECTS
                and item.get('sentiment') in SENTIMENT and item.get('segment') in SEGMENTS):
            valid.append({**item, 'evidence': evidence})
    return valid, len(raw) - len(valid), False


def qwen_request(url, payload):
    headers = {'Content-Type': 'application/json'}
    if key := os.environ.get('TASTE_LLAMA_KEY'):
        headers['Authorization'] = f'Bearer {key}'
    request = Request(url.rstrip('/') + '/v1/chat/completions', data=json.dumps(payload).encode(), headers=headers)
    with urlopen(request, timeout=100) as response:
        return json.load(response)['choices'][0]


def lowerated(data):
    import torch
    from transformers import AutoTokenizer, BertForTokenClassification

    torch.set_num_threads(4)
    path = data / 'lowerated'
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = BertForTokenClassification.from_pretrained(path, local_files_only=True).eval()
    if model.config.num_labels != 2:
        raise ValueError('Expected the published binary token head.')

    def extract(text):
        labels = list(BROAD)
        tokens = tokenizer([text] * len(labels), labels, padding=True, truncation=True, max_length=512,
                           return_offsets_mapping=True, return_tensors='pt')
        offsets = tokens.pop('offset_mapping').tolist()
        # The publisher uses label 1 for evidence and omits token types at inference.
        tokens.pop('token_type_ids', None)
        with torch.inference_mode():
            predictions = model(**tokens).logits.argmax(dim=-1).tolist()
        output = []
        for row, label in enumerate(labels):
            sequence = tokens.sequence_ids(row)
            spans = [tuple(offset) for offset, tag, kind in zip(offsets[row], predictions[row], sequence)
                     if tag == 1 and kind == 0 and offset[1] > offset[0]]
            if not spans:
                continue
            first, last = spans[0]
            for start, end in spans[1:]:
                if text[last:start].strip():
                    output.append({'aspect': label, 'evidence': text[first:last]})
                    first = start
                last = end
            output.append({'aspect': label, 'evidence': text[first:last]})
        return output

    return extract


def metrics(pairs):
    tp = fp = fn = 0
    for gold, predicted in pairs:
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None, 'f1': 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None}


def evaluate(rows):
    broad = lambda claims: {label for label, aspects in BROAD.items() if any(c['aspect'] in aspects for c in claims)}
    comparable = set().union(*map(set, BROAD.values()))
    return {
        'cases': len(rows),
        'qwen_fine_aspect': metrics([({c['aspect'] for c in row['gold']}, {c['aspect'] for c in row['qwen']}) for row in rows]),
        'qwen_aspect_and_sentiment': metrics([
            ({(c['aspect'], c['sentiment']) for c in row['gold']},
             {(c['aspect'], c['sentiment']) for c in row['qwen']}) for row in rows
        ]),
        'qwen_segment_on_explicit_mentions': metrics([
            ({(c['aspect'], c['segment']) for c in row['gold'] if c['segment'] not in {'whole', 'unknown'}},
             {(c['aspect'], c['segment']) for c in row['qwen'] if c['segment'] not in {'whole', 'unknown'}}) for row in rows
        ]),
        'qwen_broad': metrics([(broad(row['gold']), broad(row['qwen'])) for row in rows]),
        'lowerated_broad': metrics([
            (broad(row['gold']), {c['aspect'] for c in row['lowerated'] if BROAD[c['aspect']]}) for row in rows
        ]),
        'fine_gold_claims_outside_comparison': sum(c['aspect'] not in comparable for row in rows for c in row['gold']),
        'qwen_unsupported_spans': sum(row['rejected'] for row in rows),
        'qwen_invalid_responses': sum(row['invalid'] for row in rows),
        'per_aspect': {aspect: metrics([
            ({c['aspect'] for c in row['gold'] if c['aspect'] == aspect},
             {c['aspect'] for c in row['qwen'] if c['aspect'] == aspect}) for row in rows
        ]) for aspect in ASPECTS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', type=Path, required=True, help='Private JSON array of labeled excerpts.')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--server', default='http://127.0.0.1:8767')
    parser.add_argument('--in-process', action='store_true', help='Run Qwen in-process with llama-cpp-python.')
    args = parser.parse_args()
    if urlsplit(args.server).hostname not in {'127.0.0.1', 'localhost', '::1'}:
        parser.error('The private review pilot requires a loopback inference server.')
    cases = json.loads(args.gold.read_text(encoding='utf-8'))
    if not cases:
        parser.error('Provide labeled excerpts.')
    extract = lowerated(args.data)
    backend = args.server
    if args.in_process:
        from llama_cpp import Llama

        local_model = Llama(model_path=str(args.data / QWEN), n_ctx=2048, n_threads=8, verbose=False)

        def backend(**payload):
            payload.pop('chat_template_kwargs')
            payload['messages'][-1]['content'] += '\n/no_think'
            payload['response_format'] = {'type': 'json_object', 'schema': SCHEMA}
            return local_model.create_chat_completion(**payload)

    output = []
    for case in cases:
        spans, rejected, invalid = qwen(case['text'], backend)
        output.append({**case, 'qwen': spans, 'lowerated': extract(case['text']), 'rejected': rejected, 'invalid': invalid})
        print(f"Processed {case['id']}", flush=True)
    write_json(args.data / 'review-pilot-private.json', output)
    report = evaluate(output)
    report['manifest'] = {'gold_sha256': digest(args.gold), 'script_sha256': digest(Path(__file__)),
                          'qwen_revision': QWEN_REVISION, 'qwen_weights_sha256': digest(args.data / QWEN),
                          'lowerated_revision': BERT_REVISION}
    write_json(args.data / 'review-pilot-metrics.json', report)
    print(json.dumps({key: value for key, value in report.items() if key not in {'per_aspect', 'manifest'}}, indent=2))


if __name__ == '__main__':
    main()
