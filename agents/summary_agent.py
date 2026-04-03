"""
SummaryAgent — reads search results and uses Gemini API (free)
to produce a structured research summary + extracted key facts.
"""

import os
import json
from google import genai
from dataclasses import dataclass, field
from typing import List
from agents.search_agent import SearchResult


@dataclass
class ResearchSummary:
    query: str
    summary: str
    key_facts: List[str] = field(default_factory=list)
    sources_used: List[SearchResult] = field(default_factory=list)


class SummaryAgent:
    def __init__(self):
        self.name = "SummaryAgent"
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = "gemini-2.0-flash"

    def log(self, msg: str):
        print(f"[{self.name}] {msg}")

    def summarize(self, query: str, sources: List[SearchResult]) -> ResearchSummary:
        self.log(f"Received {len(sources)} sources from SearchAgent")
        context = self._build_context(sources)
        self.log("Calling Gemini API for synthesis...")
        summary_text, facts = self._call_llm(query, context)
        self.log(f"Extraction complete. {len(facts)} key facts generated.")

        return ResearchSummary(
            query=query,
            summary=summary_text,
            key_facts=facts,
            sources_used=sources,
        )

    def _build_context(self, sources: List[SearchResult]) -> str:
        parts = []
        for i, s in enumerate(sources, 1):
            parts.append(f"[Source {i}] {s.title}\n{s.snippet}\nURL: {s.url}")
        return "\n\n".join(parts)

    def _call_llm(self, query: str, context: str):
        prompt = f"""You are a research summarizer. Based on the sources below, answer the query.

QUERY: {query}

SOURCES:
{context}

Respond ONLY with a JSON object (no markdown, no backticks):
{{
  "summary": "3-4 sentence research summary",
  "key_facts": ["fact 1", "fact 2", "fact 3", "fact 4", "fact 5"]
}}"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            raw = response.text.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            return parsed.get("summary", ""), parsed.get("key_facts", [])
        except json.JSONDecodeError:
            self.log("JSON parse failed, using raw text as summary")
            return response.text.strip(), []
        except Exception as e:
            self.log(f"Gemini API error: {e}")
            return f"Research summary for '{query}' could not be generated.", []
