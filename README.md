# literature-review-mining

Light-weight utilities for **downloading, embedding, and clustering recent arXiv papers** in a few lines of code.

## Features
| Module | What it does |
|--------|--------------|
| `fetch.py`   | Query the arXiv API with keywords and save the raw metadata to JSON. |
| `embed.py`   | Convert titles + abstracts to dense vectors with **Sentence-Transformers** (`all-mpnet-base-v2`). |
| `cluster.py` | Group similar papers with **HDBSCAN**, label each cluster with TF-IDF keywords, and (optionally) add Semantic Scholar citation counts. |
| `demo.py`    | End-to-end quick start: fetch → cluster. |