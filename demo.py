from pathlib import Path
from arxiv_fetcher import fetch_arxiv_data,cluster_papers,get_embeddings

papers = fetch_arxiv_data(["construction", "vision"], limit=500, year_threshold=2018)
cluster_papers(papers, min_cluster_size=3)

#texts = [f"{p['title']}. {p['summary']}" for p in papers]
#emb   = get_embeddings(texts, cache_path=Path("emb.npy"))