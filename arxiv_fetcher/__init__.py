from __future__ import annotations

from importlib import metadata as _metadata

try:
    __version__: str = _metadata.version(__name__)
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from .fetch import fetch_arxiv_data
from .embed import get_embeddings
from .cluster import cluster_papers

__all__ = [
    "__version__",
    "fetch_arxiv_data",
    "get_embeddings",
    "cluster_papers"
]