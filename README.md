# Letterboxd recommender

A local research prototype for individual and two-person movie recommendations from Letterboxd exports. Results show personal star estimates, available movie facts, and the sources behind them. Uploads are processed in memory by a server bound to `127.0.0.1`.

## Setup

On Windows x64, extract the GitHub download and double-click **Run Letterboxd.exe**. Keep it beside the other repository files. The first run downloads Python, dependencies and about 775 MB of data, prepares the credits and traits, and trains MovieLens locally with the bundled factorizer. The console shows progress and stays open while the app runs. Subsequent runs use the cache. Choose any of the five sample exports in `profiles/` in the app.

The sample exports include usernames, ratings, watched history, watchlists, diary dates and pinned favorites. Contact fields, review text and comments were omitted. Original exports, dataset archives, trained weights and private evaluation detail are kept outside Git.

Use 64-bit Python 3.12 or 3.13 and, for MovieLens training, a C++17 compiler such as `g++` or `clang++` on `PATH`. Allow about 1 GB of free disk space and 1 GB of available RAM for training. Python commands below work from the repository root. On Windows, use `.venv\Scripts\python.exe` in place of `python` after creating the environment. On macOS or Linux, use `.venv/bin/python`.

```sh
python -m venv .venv
python -m pip install -r requirements.txt
python snapshots.py
python train.py
python app.py
```

Open [the local app](http://127.0.0.1:8765). Upload up to two Letterboxd ZIP exports or ratings CSVs and select either viewer or both. A ZIP includes watched history. A standalone CSV marks only the films listed in that file as watched. In shared mode, select any combination of films neither, exactly one, or both viewers have seen. The default is films neither viewer has seen. Shared fit is the mean estimated stars minus one quarter of the absolute difference. The runtime filter requires a known runtime. The app uses the TMDB audience baseline for initial ranking, with the Mooduel and MovieLens models available in the model selector.

With the MovieLens cache loaded, the app moves the ordering score for sparsely rated films toward the model’s average score on the imported history. The adjustment uses MovieLens rating counts as coverage, including for the TMDB baseline, and retains titles with zero MovieLens ratings. The cards show the original star estimates, shared fit, and MovieLens rating count. The published benchmark compares the original model scores.

If a ZIP export is missing a watched film, add its IMDb ID under the ZIP's Letterboxd username in the local `data/watched-overrides.json` file and restart the app. The file is a JSON object mapping usernames to lists of IMDb IDs. These entries update watched status for recommendations without adding ratings.

`snapshots.py` caches files under `data/` and verifies them on repeat runs. To reuse local files, run `python snapshots.py --mooduel-file PATH --movielens-file PATH`. For an audience and Mooduel-only install, run `python snapshots.py --mood-only`, then start the app. Keep exports and evaluation files containing film titles in `data/`, `private/`, or outside the repository. The app needs internet access only if you follow an external movie link. Dataset and dependency downloads require network access during setup. No API keys or pretrained weights are required.

## Data and matching

| Snapshot | Used fields | License and download |
| --- | --- | --- |
| [Mooduel v1.0](https://huggingface.co/datasets/fractalintelligence/mooduel-v1.0), Fractal Intelligence | TMDB IDs, titles, years, genres, runtime, snapshot TMDB score, inferred affective traits | CC BY-NC 4.0, 13.6 MB CSV, pinned revision and publisher SHA-256 |
| [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/), GroupLens | `movies.csv` titles/genres, `links.csv` external IDs, `ratings.csv` user/item/stars | GroupLens research terms, 239 MB ZIP, pinned archive SHA-256 and publisher member checksums. See [official terms and citation](https://files.grouplens.org/datasets/movielens/ml-32m-README.html). |

TMDB and IMDb IDs identify catalogue records where available. Duplicate MovieLens TMDB links are quarantined from cross-source joins, preserving the distinct MovieLens and IMDb records. Letterboxd URIs connect records within an export. Imports prefer explicit provider IDs when supplied and otherwise use unique normalized title/year matches, including MovieLens aliases. Ambiguous matches are reported in the local audit. The initial release filter uses the snapshot year and excludes current-year titles pending dated release evidence. Country-specific release status and media-type verification for MovieLens-only titles are unavailable in these snapshots.

The available snapshots have no verified awards, premise-only overviews, or Letterboxd score/distribution fields. Cards mark those fields unavailable and link to source film pages where IDs exist. Mooduel traits are inferred descriptors. The app displays point estimates on a 0.5–5-star scale. MovieLens factors are trained locally on explicit ratings, and a ridge fit adapts the frozen movie factors to each import. The hybrid adds a regularized Mooduel residual where available.

## Personal Taste

See [how Personal Taste works](PERSONAL_TASTE.md) for the full scoring and ranking equations.

After caching Mooduel and MovieLens, run the following and restart the app.

```sh
python taste_data.py taste
python taste.py
python check_traits.py
```

Select **Personal Taste** to rank by preferences learned from your ratings. **Personal Taste · credits** removes Tag Genome and **Personal Taste · mood and genre** uses genres and Mooduel alone. Director and writing affinities use distinct IMDb person IDs, can move estimates in either direction, and shrink toward zero with limited examples. The TMDB rating contribution is bounded to 0.125 stars. Tag Genome supplies estimated trait relevance for covered films. It does not provide spoiler-free descriptions, official awards, or a measured audience judgment for each act. Review text never appears in recommendation cards.

| Source | Files and usage |
| --- | --- |
| [IMDb non-commercial datasets](https://data.imdb.com/non-commercial-datasets/) | `title.crew.tsv.gz` (84 MB) and `name.basics.tsv.gz` (313 MB). Local personal, non-commercial use. Daily URLs are mutable. Preserve the recorded files and hashes to reproduce this run. See the [IMDb terms and required acknowledgment](https://help.imdb.com/article/imdb/general-information/can-i-use-imdb-data-in-my-software/G5JTRESSHJBBHTGX). |
| [Tag Genome 2021](https://grouplens.org/datasets/movielens/tag-genome-2021/) | CC BY-NC 3.0. The downloader fetches only the GLMER score member, about 124 MB, and checks the pinned SHA-256, expanded size, and publisher ZIP CRC. The full 1.93 GB archive includes materials unused in this experiment. |
| [SoftAttributes](https://github.com/google-research-datasets/soft-attributes) | CC BY 4.0, 2.1 MB. Pinned CSV used to compare selected traits with human judgments. |

Information courtesy of IMDb (https://www.imdb.com). Used with permission.

### Compare rankings

```sh
python evaluate.py profile1.zip profile2.zip profile3.zip profile4.zip profile5.zip --release-cutoff-year 2026 --app-order --models audience mooduel collaborative collaborative_mooduel taste_traits taste_credits taste --output data/taste-evaluation.json --details data/taste-evaluation-private.json
```

The matched-pair diagnostic scores held-out film pairs with similar MovieLens means and counts and different personal ratings. It complements hidden-favorite retrieval and rating error. See [the taste evaluation](TASTE_RESULTS.md).

### Run the review-extraction pilot

Use a separate Python environment with the base requirements plus the CPU PyTorch wheel, Transformers, and a Windows CPU wheel of llama-cpp-python. The main app does not import the inference packages.

```sh
python -m pip install -r requirements.txt
python -m pip install torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers==4.57.6
python -m pip install llama-cpp-python==0.3.36 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python taste_data.py reviews
python review_pilot.py --gold data/review-gold.json --in-process
```

The Qwen3 4B Q4_K_M model is 2.50 GB and Lowerated BERT is 436 MB. Both model cards list Apache 2.0. They are cached by revision and checksum. The pilot runs offline when passed `--in-process`. The alternate server path accepts a local llama.cpp address and uses `TASTE_LLAMA_KEY` if configured. A Windows Vulkan release can be cached with `python taste_data.py runtime`. [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-GGUF) · [Lowerated model card](https://huggingface.co/Lowerated/lm6-movie-aspect-extraction-bert)

The gold file is a private JSON array. Each case has `id`, `text` and a `gold` list of objects with `aspect`, `sentiment`, and `segment`. Use `review_pilot.py` for allowed values. An empty list means no supported aspect. The report separates the common broad-category comparison from Qwen's fine-grained aspect, sentiment, and segment results. Raw excerpts and model outputs stay in ignored local files.

## Reproduce the evaluation

```sh
python evaluate.py profile1.zip profile2.zip profile3.zip profile4.zip --release-cutoff-year 2026 --details data/evaluation-private.json
```

The command accepts any number of exports. Use `--clean-only` to disable models that use MovieLens ratings while preserving the common catalogue. A `--favorites` JSON file can supply four pinned favorites per export when missing. Use `null` for profiles with exported pins. Entries can be known film URLs from the export or `Title (Year)` strings. The output includes aggregate and per-profile metrics in `data/evaluation.json`. The optional private file contains individual ranks, split targets and matching decisions.

See [the measured evaluation](RESULTS.md) for the initial model comparison. Tag Genome, plot embeddings, review extraction, population tuning and prospective shared-viewing validation are subsequent experiments. Tag Genome 2021 is a separate 1.8 GB archive and is reserved for a controlled feature comparison.
