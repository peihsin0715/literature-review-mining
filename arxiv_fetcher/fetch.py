from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import requests

ARXIV_API_URL: str = "http://export.arxiv.org/api/query"
ATOM_NS: dict[str, str] = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_PAGE_SIZE: int = 100
DEFAULT_OUTPUT: Path = Path("arxiv_papers.json")

def _build_search_query(keywords: List[str]) -> str:
    return " AND ".join(f"all:{kw}" for kw in keywords)


def _parse_entry(entry: ET.Element) -> Dict:
    title = entry.find("atom:title", ATOM_NS).text 
    published_year = int(entry.find("atom:published", ATOM_NS).text[:4])
    authors = [
        author.find("atom:name", ATOM_NS).text 
        for author in entry.findall("atom:author", ATOM_NS)
    ]
    link = entry.find("atom:id", ATOM_NS).text
    summary = entry.find("atom:summary", ATOM_NS).text.strip()
    return {
        "title": title,
        "authors": authors,
        "year": published_year,
        "link": link,
        "summary": summary,
    }

def fetch_arxiv_data(
    keywords: List[str],
    *,
    limit: int = 100,
    year_threshold: int = 2021,
    page_size: int = DEFAULT_PAGE_SIZE,
    pause_seconds: float = 1.0,
    output_path: str | Path | None = DEFAULT_OUTPUT,
) -> List[Dict]:
    results: List[Dict] = []
    start = 0
    query = _build_search_query(keywords)

    while start < limit:
        params = {
            "search_query": query,
            "start": start,
            "max_results": min(page_size, limit - start),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = requests.get(ARXIV_API_URL, params=params, timeout=10)
        resp.raise_for_status()

        entries = ET.fromstring(resp.content).findall("atom:entry", ATOM_NS)
        if not entries:
            break
        for entry in entries:
            rec = _parse_entry(entry)
            if rec["year"] >= year_threshold:
                results.append(rec)
        start += page_size
        time.sleep(pause_seconds)

    if output_path is not None:
        out = Path(output_path)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Saved {len(results)} papers → {out}")

    return results

def _cli() -> None:
    p = argparse.ArgumentParser(description="Fetch papers from the arXiv API.")
    p.add_argument("-k", "--keywords", nargs="+", required=True)
    p.add_argument("-l", "--limit", type=int, default=100)
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--year-threshold", type=int, default=2021)
    args = p.parse_args()

    fetch_arxiv_data(
        args.keywords,
        limit=args.limit,
        year_threshold=args.year_threshold,
        output_path=args.output,
    )

if __name__ == "__main__":
    _cli()