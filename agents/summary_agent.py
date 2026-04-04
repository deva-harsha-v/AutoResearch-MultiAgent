"""
SummaryAgent — reads search results and uses Gemini API to produce
a structured research summary + extracted key facts.

FIX: Prompt now explicitly forces Gemini to ground every sentence in the
     actual source snippets, citing [Source N] inline. Generic summaries
     from training data are blocked by the prompt instruction.
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
        """
        Build a numbered source block that the LLM can cite by index.
        Including domain and URL helps the model judge source credibility.
        """
        parts = []
        for i, s in enumerate(sources, 1):
            parts.append(
                f"[Source {i}] {s.title}\n"
                f"Domain: {s.domain}\n"
                f"Snippet: {s.snippet}\n"
                f"URL: {s.url}"
            )
        return "\n\n".join(parts)

    def _call_llm(self, query: str, context: str):
        """
        FIX: The old prompt let Gemini summarise from its own training data.
        The new prompt:
        1. Explicitly forbids information not in the sources
        2. Requires inline [Source N] citations so output is traceable
        3. Asks for SPECIFIC facts (numbers, names, dates) — not generics
        4. Each key_fact must quote which source supports it
        """
        prompt = f"""You are a research summarizer. Your ONLY job is to synthesize the sources below.

STRICT RULES:
- Use ONLY information explicitly stated in the numbered sources below.
- Do NOT use your training knowledge. Do NOT add any information not present in the sources.
- Every sentence in your summary must be traceable to a specific source.
- Key facts must be specific (include numbers, names, dates, percentages where available).
- If a source contains no relevant information, skip it.

QUERY: {query}

SOURCES:
{context}

Respond ONLY with a JSON object (no markdown, no backticks, no extra text):
{{
  "summary": "3-4 sentence summary using ONLY the source content above. Cite sources inline as [Source 1], [Source 2], etc.",
  "key_facts": [
    "Specific fact from Source N: ...",
    "Specific fact from Source N: ...",
    "Specific fact from Source N: ...",
    "Specific fact from Source N: ...",
    "Specific fact from Source N: ..."
  ]
}}"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            raw = response.text.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            summary = parsed.get("summary", "")
            facts = parsed.get("key_facts", [])
            return summary, facts

        except json.JSONDecodeError:
            self.log("JSON parse failed, using raw text as summary")
            return response.text.strip(), []
        except Exception as e:
            self.log(f"Gemini API error: {e}")
            return f"Research summary for '{query}' could not be generated.", []
