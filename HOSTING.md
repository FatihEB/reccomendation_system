# Host the recommender

This deployment is intended for personal, noncommercial use among a small group. It preserves the full Personal Taste model, including director and writer affinities, genre, Mooduel, and Tag Genome. IMDb permits limited personal and noncommercial use subject to its conditions, including attribution and restrictions on republishing or creating a movie-information database. Review those conditions before broadening the audience or changing the use. [IMDb dataset terms](https://help.imdb.com/article/imdb/general-information/can-i-use-imdb-data-in-my-software/G5JTRESSHJBBHTGX)

The hosted app keeps two CSV or ZIP uploads, username preview, individual and shared scoring, and the watched and runtime filters. Personal Taste is selected by default. MovieLens and Mooduel baselines remain available.

## Render

1. Open the [Render deploy form](https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2FFatihEB%2Freccomendation_system) and sign in.
2. Review the Blueprint and deploy the **Free** Docker web service. The build downloads and verifies the public snapshots, trains MovieLens, and prepares the complete Personal Taste caches. The first build can take several minutes.
3. When the service is live, open its HTTPS URL and visit `/healthz`. It should return `ready`. Test individual and shared results with your exports.

The free instance sleeps after 15 idle minutes and can take about a minute to wake. Automatic deploys are disabled so later Git pushes do not unexpectedly replace the running version. [Render free plan](https://render.com/docs/free)

## Run the container locally

```sh
docker build -t what-to-watch .
docker run --rm -e PUBLIC_HOST=localhost:10000 -p 127.0.0.1:10000:10000 what-to-watch
```

Use [the local health endpoint](http://localhost:10000/healthz) to check startup. Browser uploads in hosted mode require the configured HTTPS origin. For the local browser flow with plain HTTP, use the existing Windows launcher or `python app.py`.

The build context and image include only the five sanitized sample profile ZIPs tracked in `profiles/` and their two-film watched-history correction. They are selectable in the interface but are never served as downloadable assets. Private evaluations, other local overrides, raw Tag Genome scores, raw IMDb files, and review extraction assets remain excluded. The final image contains the derived credit cache needed for Personal Taste. The service processes uploads in memory, limits request and expanded ZIP size, bounds connections and scoring, sends security headers and no-store responses, and disallows search indexing. It serves a fixed list of assets and routes. The datasets stay outside Git. [Mooduel v1.0](https://huggingface.co/datasets/fractalintelligence/mooduel-v1.0) · [MovieLens terms](https://files.grouplens.org/datasets/movielens/ml-32m-README.html) · [Tag Genome terms](https://files.grouplens.org/datasets/tag-genome-2021/genome_2021_readme.txt)
