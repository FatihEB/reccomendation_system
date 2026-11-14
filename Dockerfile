FROM python:3.13-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt requirements-hosted.txt ./
RUN python -m pip install --no-cache-dir -r requirements-hosted.txt

FROM base AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir pandas==3.0.1
COPY catalog.py factorize.cpp prepare_hosted.py snapshots.py taste.py taste_data.py train.py ./
RUN python prepare_hosted.py --data /snapshots

FROM base
COPY app.py catalog.py hosted.py models.py snapshots.py taste.py taste_data.py train.py ./
COPY web/ ./web/
COPY --from=build /snapshots/mooduel-v1.0.csv /snapshots/ml-32m.zip /snapshots/movielens-model.npz /snapshots/tag-genome-traits.npz /snapshots/imdb-credits-local.json /snapshots/
RUN chmod -R a=rX /app /snapshots
ENV HOSTED_DATA=/snapshots PORT=10000
USER 10001:10001
EXPOSE 10000
CMD ["python", "hosted.py"]
