from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def get_embeddings(
    texts: List[str],
    *,
    cache_path: Path | None = None,
    model_name: str = MODEL_NAME,
    **encode_kwargs,
) -> np.ndarray:
    if cache_path and cache_path.exists():
        emb = np.load(cache_path)
        if len(emb) == len(texts):
            return emb

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, **encode_kwargs)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)

    return emb