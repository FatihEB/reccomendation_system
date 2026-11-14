# How Personal Taste works

The model fits one profile per viewer from imported star ratings. The full version combines IMDb directing and writing credits, genre and genre combinations, Mooduel descriptors, and 22 selected Tag Genome traits.

## Inputs and matching

The catalogue links Mooduel and MovieLens by verified identifiers where possible. Imports prefer explicit provider IDs and otherwise use unique normalized title/year matches and consistent export URI aliases. Ambiguous matches are withheld. The fit uses one numeric rating per canonical film. Watched history determines candidate eligibility. Pinned favorites supply evaluation targets and do not add bonus ratings.

## Scoring

1. **Set the viewer's baseline.** For `n` ratings, the baseline is `(sum(ratings) + 5 × 3.3) / (n + 5)`. The five prior ratings stabilize sparse profiles.
2. **Add a small audience prior.** Convert the TMDB snapshot rating to five stars, substitute the catalogue mean when missing, and subtract the catalogue mean. Clip that difference to ±1 and multiply by 0.125. When MovieLens counts are available, multiply again by `count / (count + 25)`. The largest possible audience contribution is ±0.125 stars.
3. **Learn separate director and writer effects.** For each person and role, sum the viewer's rating residuals (actual stars minus baseline and audience prior) across credited films and divide by `credited films + 4`. Repeated likes or dislikes strengthen the effect. For a candidate, sum its affinities within each role, dividing each role's sum by the square root of its number of credited people. Sum both roles and clip the total to ±1.15 stars. When preparing the history residuals for the next step, calculate each film's credit contribution with its own rating omitted.
4. **Fit the remaining taste effects in sequence.** Fit genre first, Mooduel second, and Tag Genome third. Genres use indicators plus pair indicators for combinations appearing in at least 250 catalogue entries. Mooduel contributes numeric mood dimensions, dominant emotions, pacing and selected tags such as dread, suspense and noir. Tag Genome contributes estimated relevance for traits such as nonlinear storytelling, dialogue, atmosphere and originality. Synonymous tags in a trait use their maximum relevance. The traits-only variant skips credits and Tag Genome. The credits variant adds the credit step and skips Tag Genome.
5. **Regularize each feature block.** Use only history films covered by that source, requiring at least five. Center columns at their profile means and scale by their standard deviations with a floor of 0.32. Keep columns with at least three standardized deviations greater than 0.1 in magnitude. Fit ridge regression to centered remaining residuals with penalties of 40 for genre, 55 for Mooduel and 40 for Tag Genome. Clip each predicted block effect to ±0.8 stars and subtract that contribution before fitting the next block. A film outside a source's coverage receives zero contribution from it.

```text
raw prediction = viewer baseline + audience prior
               + director/writer effect
               + genre effect + Mooduel effect + Tag Genome effect
```

## Ranking and explanations

Individual results sort by the raw prediction. The displayed prediction is clipped to 0.5–5 stars and rounded to one decimal. Shared mode fits both viewers independently, then ranks by `(A + B) / 2 − 0.25 × |A − B|` using their clipped predictions before display rounding. The shared value measures fit on the star scale.

The app applies catalogue eligibility, watched-history and optional runtime filters before sorting. Individual mode excludes matched watched films. Shared mode can include any combination of films seen by neither, exactly one, or both viewers. Cards can cite actual training-history ratings for director or writer affinities and show the strongest signed trait contributions. Explanations report contributions before the applicable caps and omit payoff tags to limit spoilers.

The model has no standalone award, release-year preference, or review-extraction term. Tag Genome and Mooduel supply estimated traits, with partial coverage. The rating estimate is a point prediction. The separate [evaluation](TASTE_RESULTS.md) covers favorite recovery, rating error and the ability to distinguish similarly rated and similarly popular films.

## Personal Taste - Time Weight

This dropdown option blends 70% of the standard raw prediction with 30% of a second fit based on dated ratings. The second fit uses the exported `Date` from the selected rating row for each film. Its weight is `0.35 + 0.65 × 2^(−age_in_days / 730)`. Today's rating has weight 1.0, a rating from 730 days ago has weight 0.675, and the weight approaches 0.35 for older ratings. Missing, invalid, and future dates contribute only to the long-term fit. With no usable dates, the result is exactly standard Personal Taste.

The second fit uses weighted rating totals for the viewer baseline and weighted residuals and evidence counts for director and writer affinities. When excluding a film's own contribution from its historical credit fit, it removes both the weighted residual and its weight. Genre, Mooduel, and Tag Genome use weighted centering, variance, and ridge regression. The blend is computed independently for each viewer before clipping and shared scoring.
