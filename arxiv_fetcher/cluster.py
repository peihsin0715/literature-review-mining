from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import hdbscan
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from .embed import get_embeddings

S2_API = "https://api.semanticscholar.org/graph/v1"
DEFAULT_OUTPUT: Path = Path("clusters.json")

@dataclass(slots=True)
class ClusterParams:
    min_cluster_size: int = 3
    metric: str = "precomputed"
    cluster_selection_method: str = "eom"

def _extract_keywords(texts: List[str], *, top_k: int = 10) -> List[str]:
    """Return *top_k* TF-IDF keywords for a list of texts."""
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    scores = X.sum(axis=0).A1
    return vocab[scores.argsort()[::-1][:top_k]].tolist()

def _citation_count(arxiv_url: str, *, sleep_secs: float = 1.0) -> int:
    headers = {"x-api-key": os.getenv("S2_API_KEY")} if os.getenv("S2_API_KEY") else {}
    resp = requests.get(
        f"{S2_API}/paper/URL:{arxiv_url}/citations?limit=1", headers=headers, timeout=10
    )
    time.sleep(sleep_secs)
    return resp.json().get("total", 0) if resp.ok else 0

def cluster_papers(
    papers: List[Dict],
    *,
    min_cluster_size: int = 3,
    top_k_keywords: int = 10,
    include_citations: bool = False,
    citation_sleep: float = 1.0,
    output_path: str | Path | None = DEFAULT_OUTPUT,
) -> Dict[int, Dict]:
    texts = [f"{p['title']}. {p['summary']}" for p in papers]
    embeddings = get_embeddings(texts)

    dist = cosine_distances(embeddings).astype(np.float64)
    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="precomputed",
        cluster_selection_method="eom",
    ).fit_predict(dist)

    for idx, p in enumerate(papers):
        p["cluster"] = int(labels[idx])

    clusters: Dict[int, Dict] = {}
    unique_cids = sorted({int(lbl) for lbl in labels if lbl != -1})

    for cid in unique_cids:
        idxs = [i for i, lbl in enumerate(labels) if int(lbl) == cid]
        summaries = [papers[i]["summary"] for i in idxs]
        centre = embeddings[idxs].mean(axis=0)

        items = []
        for i in tqdm(idxs, desc=f"Cluster {cid}"):
            record = {
                "title": papers[i]["title"],
                "summary": papers[i]["summary"],
                "distance": float(np.linalg.norm(embeddings[i] - centre)),
            }
            if include_citations:
                record["citations"] = _citation_count(papers[i]["link"], sleep_secs=citation_sleep)
            items.append(record)

        items.sort(key=lambda d: (-d.get("citations", 0), d["distance"]))
        clusters[cid] = {
            "keywords": _extract_keywords(summaries, top_k=top_k_keywords),
            "papers": items,
        }

    if output_path is not None:
        out = Path(output_path)
        out.write_text(json.dumps(clusters, indent=2, ensure_ascii=False))
        print(f"Saved {len(clusters)} clusters → {out}")

    return clusters

def _cli() -> None:
    p = argparse.ArgumentParser(description="Cluster and label arXiv papers.")
    p.add_argument("--in", dest="input_json", default="arxiv_papers.json")
    p.add_argument("--out", dest="output_json", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--min-cluster-size", type=int, default=3)
    p.add_argument("--include-citations", action="store_true")
    args = p.parse_args()

    papers = json.loads(Path(args.input_json).read_text())
    cluster_papers(
        papers,
        min_cluster_size=args.min_cluster_size,
        include_citations=args.include_citations,
        output_path=args.output_json,
    )


if __name__ == "__main__":
    _cli()