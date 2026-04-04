"""
SearchAgent — retrieves and ranks web sources for a given query.
Uses DuckDuckGo (free, no API key needed) via duckduckgo-search.

FIX: Replaced flat keyword-overlap ranking with TF-IDF style scoring
     + URL deduplication so every query returns unique, relevant results.
"""

from duckduckgo_search import DDGS
from dataclasses import dataclass
from typing import List
import time
import math
import re


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0
    domain: str = ""


class SearchAgent:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.name = "SearchAgent"

    def log(self, msg: str):
        print(f"[{self.name}] {msg}")

    def search(self, query: str) -> List[SearchResult]:
        self.log(f"Decomposing query: '{query}'")
        sub_queries = self._build_queries(query)

        self.log(f"Dispatching {len(sub_queries)} sub-queries...")
        all_results = []
        seen_urls = set()

        with DDGS() as ddgs:
            for q in sub_queries:
                try:
                    hits = list(ddgs.text(q, max_results=6))
                    for h in hits:
                        url = h.get("href", "")
                        # Deduplicate by URL — each source appears only once
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            domain = re.sub(r"https?://(www\.)?", "", url).split("/")[0]
                            all_results.append(SearchResult(
                                title=h.get("title", ""),
                                url=url,
                                snippet=h.get("body", ""),
                                domain=domain,
                            ))
                    time.sleep(0.3)
                except Exception as e:
                    self.log(f"Sub-query failed for '{q}': {e}")

        self.log(f"Retrieved {len(all_results)} unique candidate documents")
        ranked = self._rank(all_results, query)
        top = ranked[:self.max_results]
        self.log(f"Top {len(top)} sources selected. Passing to SummaryAgent.")
        return top

    def _build_queries(self, query: str) -> List[str]:
        """
        Three focused sub-queries targeting different result slices.
        """
        return [
            query,
            f"{query} 2024 2025 latest news",
            f"{query} research analysis findings",
        ]

    def _rank(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """
        TF-IDF style relevance scoring.

        OLD (broken): count how many query keywords appear in the text.
        Problem — every doc with the same common words scores identically.

        NEW: for each query term, compute TF (frequency in this doc) × IDF
        (how rare this term is across all docs). Rare, specific terms score
        higher than generic ones, so each document gets a unique score that
        reflects how well it actually matches THIS query.

        Extra signals:
        - Title match boost (title relevance is a strong indicator)
        - Snippet length boost (longer = richer content)
        """
        if not results:
            return results

        query_terms = self._tokenise(query)
        corpus = [self._tokenise(r.title + " " + r.snippet) for r in results]
        N = len(corpus)

        # IDF: smoothed log(N / document_frequency)
        idf = {}
        for term in set(query_terms):
            df = sum(1 for doc in corpus if term in doc)
            idf[term] = math.log((N + 1) / (df + 1)) + 1

        for i, result in enumerate(results):
            doc_terms = corpus[i]
            doc_len = max(len(doc_terms), 1)

            # TF-IDF score across all query terms
            score = 0.0
            for term in query_terms:
                tf = doc_terms.count(term) / doc_len
                score += tf * idf.get(term, 0)

            # Title match boost
            title_terms = self._tokenise(result.title)
            title_overlap = sum(1 for t in query_terms if t in title_terms)
            score += 0.3 * (title_overlap / max(len(query_terms), 1))

            # Snippet length boost — richer snippets are more useful
            snippet_len = len(result.snippet.split())
            score += 0.05 * min(snippet_len / 100, 1.0)

            result.relevance_score = round(score, 4)

        return sorted(results, key=lambda x: x.relevance_score, reverse=True)

    def _tokenise(self, text: str) -> List[str]:
        """Lowercase, remove punctuation, strip stopwords."""
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "it", "its", "this", "that", "as", "be", "been", "has", "have",
        }
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in tokens if t not in stopwords and len(t) > 1]
