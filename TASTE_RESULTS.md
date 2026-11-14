# Personal Taste evaluation

The comparison used five local Letterboxd exports containing 1,013 matched ratings and 20 pinned favorites. The catalogue linked 87,080 films to at least one IMDb director or writing credit and 9,585 to the selected Tag Genome scores.

| Model | All favorites hidden, hits@100 | High ratings hidden, hits@100 | Matched-pair accuracy | Stratified MAE |
| --- | ---: | ---: | ---: | ---: |
| TMDB audience | 6/20 | 30/96 | 64.8% | 0.672 |
| Mooduel | 5/20 | 20/96 | **73.0%** | 0.651 |
| MovieLens collaborative | 2/20 | 11/96 | 66.0% | **0.643** |
| MovieLens + Mooduel | 2/20 | 16/96 | 69.2% | 0.645 |
| Personal Taste · mood and genre | 1/20 | 8/96 | 60.7% | 0.708 |
| Personal Taste · credits | 4/20 | 11/96 | 61.9% | 0.707 |
| Personal Taste | 5/20 | 15/96 | 64.5% | 0.700 |

Director/writer credits and Tag Genome each improved favorite recovery within the Personal Taste variants. The full variant also had lower all-favorites-out rating error than the audience model (1.093 versus 1.291 MAE). Audience led full-catalogue recovery of high ratings and favorites. Mooduel led the matched-pair comparison. Personal Taste is available as an explicit choice in the app, with the audience model selected by default.

Both movies in each matched pair were held out from the same fit. Their personal ratings differed by at least 1.5 stars. Their shrinkage-adjusted MovieLens means differed by at most 0.25 stars, and both had at least 25 MovieLens ratings with counts within a factor of two. The 341 pair instances reuse films and can recur across seeds. Equal-weight per-profile accuracy was 68.7% for Personal Taste, 68.9% for Mooduel, and 63.7% for TMDB audience. Stratified MAE uses the 620 eligible numeric targets out of 633 instances. This is retrospective evaluation on five profiles, with no group-enjoyment outcomes. The evaluation used app ordering and the source hashes in `data/taste-evaluation.json`. The optional private report records per-title and per-pair results. MovieLens users are anonymous and overlap with the local profiles cannot be verified.

## Trait validation

Compared six Tag Genome features with directional human judgments from [SoftAttributes](https://github.com/google-research-datasets/soft-attributes), using only pairs where both titles joined and the rater indicated a direction. A tied prediction receives half credit. The data describe perceived traits, with repeated ratings and overlapping titles.

| Trait | Comparisons | Agreement |
| --- | ---: | ---: |
| Emotional intensity | 663 | 72.2% |
| Perceived realism | 646 | 68.7% |
| Complex plotting | 626 | 68.5% |
| Perceived confusion | 608 | 67.8% |
| Originality | 548 | 60.6% |
| Slow pacing | 581 | 54.9% |

## Review extraction

The pilot used 31 real excerpts from the local Letterboxd exports. I assigned reference labels before running either extractor. These are single-annotator labels and have not been independently reviewed. The sample includes explicit evaluations, mixed reactions, descriptive emotion, plot-related text, and cases without a supported aspect. Excerpts and review identifiers stay in the ignored local cache.

| Task | Qwen3-4B | Lowerated BERT |
| --- | ---: | ---: |
| Broad aspect detection, micro-F1 | 0.694 | 0.667 |
| Fine-grained aspect detection, micro-F1 | 0.603 | — |
| Fine-grained aspect and sentiment, micro-F1 | 0.476 | — |
| Explicit segment detection, micro-F1 | 0.286 | — |

The common comparison maps overlapping fine labels to the publisher's broad categories. Lowerated's published weights use a binary token classifier conditioned on the requested category. The package's inference code identifies label 1 as evidence. Its BERT checkpoint produces evidence spans, with sentiment handled by a separate model. Qwen returned valid JSON for all 31 excerpts, and all accepted evidence strings matched the source text. Exact matching validates provenance. The aspect and sentiment results show that a real quote can still be assigned the wrong interpretation.

The review sample supplied only 1–4 rated films per profile with independently sourced labeled review evidence. This and the low segment recall make a review-based ranking result uninformative at this stage. Review features were excluded from the live ranker and the recommendation holdouts. See `data/review-pilot-metrics.json` for precision, recall, and per-aspect counts.
