# Initial evaluation

Four real Letterboxd ZIP exports supplied 875 ratings rows. The combined catalogue matched all 16 pinned favorites and used 786 distinct profile ratings. Across the ratings tables, 85 rows were unmatched and five were ambiguous. One additional matched rating came from a diary entry. The catalogue contained 95,743 records, with 94,766 eligible before profile-specific watched exclusions. The matching stage quarantined 36 duplicate MovieLens TMDB identifiers from cross-source joins.

## Results

All rank comparisons use the same full eligible catalogue per split. Each favorite test contains 16 target instances. The high-rating suite has 60 held-out target instances and the stratified suite has 498, both pooled across three fixed seeds. The same film can contribute in multiple seeds. Of the stratified targets, 487 passed catalogue eligibility and contributed to rating error.

| Model | One-at-a-time favorites in top 100 | All-four-out favorites in top 100 | All-four-out median rank | High-rating recall@100 | Stratified MAE | Stratified RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random ranking | 0/16 | 0/16 | 50,585 | 1.7% | — | — |
| TMDB audience | 4/16 | 4/16 | 402 | 16.7% | 0.674 | 0.895 |
| Mooduel | 3/16 | 3/16 | 504 | 10.0% | 0.655 | 0.877 |
| MovieLens popularity | 4/16 | 4/16 | 1,460.5 | 21.7% | — | — |
| MovieLens audience | 3/16 | 3/16 | 505 | 10.0% | 0.675 | 0.893 |
| MovieLens collaborative | 1/16 | 1/16 | 2,184.5 | 3.3% | 0.646 | 0.872 |
| MovieLens + Mooduel | 1/16 | 1/16 | 1,557 | 6.7% | 0.653 | 0.879 |

Random and popularity are ranking controls. Their attached star predictions reuse the corresponding audience baselines. The two collaborative variants exposed zero held-out known dislikes (ratings of 2 or below) in the stratified top 100, compared with one for Mooduel, four for TMDB audience and six for MovieLens audience. Unrated candidates have unknown relevance.

The TMDB audience baseline is the app default. It had the best all-four-out median rank and tied MovieLens popularity on recall@100. MovieLens popularity had the highest all-four-out NDCG@100 (0.0915 versus 0.0754 for TMDB audience). Collaborative filtering had the lowest rating error but lower recovery on both favorite tests and on repeated high-rating holdouts. Adding Mooduel to collaborative improved all-four-out median rank and high-rating recall@100 while increasing stratified rating error. These results support retaining the simpler ranking default and keeping collaborative estimates available for comparison.

## Protocol and limits

- Pinned favorites came from `profile.csv`. For each split, hidden canonical film IDs were removed from all personal ratings and watched exclusions before fitting. Watchlists, likes, and review text were unused throughout. Each target had to match and pass normal catalogue eligibility.
- Broad holdouts sample 20% of each observed star level, rounded to the nearest count with a minimum of one in a populated level. High-rating holdouts sample about 20% of rated films at 4.5 stars or above. The seeds are 171, 281 and 391. The release cutoff year is 2026, with earlier years eligible.
- All seven rankers share the combined catalogue. The random, TMDB audience and Mooduel rows use MovieLens catalogue metadata only. The four MovieLens-signal rows use the anonymous MovieLens rating corpus, whose overlap with these Letterboxd users is unknown. The fitted 32-factor model used all 32,000,204 MovieLens ratings from 200,948 accounts over seven epochs. Population hyperparameters were fixed for this run. No held-out MovieLens population benchmark was run.
- This retrospective sample has four profiles. Repeated splits share observations, and these results do not establish population-level accuracy, chronological prediction, confidence intervals or shared-viewing enjoyment. Predictions are point estimates on the imported scale. No joint probability is displayed.
- `data/evaluation.json` contains the full aggregate and per-profile metrics. `data/evaluation-private.json` contains per-title ranks, coverage and import decisions. Both are local ignored files. The archive and model checksums and split parameters are recorded in the evaluation manifest.
