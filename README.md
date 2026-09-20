# Connections solver

A local experiment that fetches the daily NYT Connections puzzle, embeds its
words with Google News Word2Vec, and searches for the lowest-distance
four-by-four partition.

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

The first run that constructs `Word2VecEmbedding` downloads the approximately
1.6 GB Google News model into `models/`. Scraped puzzles are cached under
`connections_solver/data/`.

## Run

```bash
venv/bin/python -m connections_solver.main
```

The scraper uses the NYT Connections API endpoint confirmed by the current
Connections page and falls back to its local date cache when available.
