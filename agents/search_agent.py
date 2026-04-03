"""
SearchAgent — retrieves and ranks web sources for a given query.
Uses DuckDuckGo (free, no API key needed) via duckduckgo-search.
"""

from duckduckgo_search import DDGS
from dataclasses import dataclass
from typing import List
import time


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0


class SearchAgent:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.name = "SearchAgent"

    def log(self, msg: str):
        print(f"[{self.name}] {msg}")

    def search(self, query: str) -> List[SearchResult]:
        self.log(f"Decomposing query: '{query}'")
        sub_queries = self._build_queries(query)

        self.log(f"Dispatching {len(sub_queries)} parallel queries...")
        all_results = []

        with DDGS() as ddgs:
            for q in sub_queries:
                try:
                    hits = list(ddgs.text(q, max_results=4))
                    for h in hits:
                        all_results.append(SearchResult(
                            title=h.get("title", ""),
                            url=h.get("href", ""),
                            snippet=h.get("body", ""),
                        ))
                    time.sleep(0.3)  # polite delay
                except Exception as e:
                    self.log(f"Query failed for '{q}': {e}")

        self.log(f"Retrieved {len(all_results)} candidate documents")
        ranked = self._rank(all_results, query)
        top = ranked[:self.max_results]
        self.log(f"Top {len(top)} sources selected. Passing to SummaryAgent.")
        return top

    def _build_queries(self, query: str) -> List[str]:
        """Break query into 3 focused sub-queries."""
        return [
            query,
            f"{query} latest research 2025",
            f"{query} key findings overview",
        ]

    def _rank(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Simple keyword overlap scoring."""
        keywords = set(query.lower().split())
        for r in results:
            text = (r.title + " " + r.snippet).lower()
            overlap = sum(1 for k in keywords if k in text)
            r.relevance_score = round(overlap / max(len(keywords), 1), 3)
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
