"""
FactCheckAgent — verifies each extracted fact using Gemini.
Assigns a confidence score and verdict to each claim.

FIX: Old code verified facts against research.summary (the LLM's own output).
     This made it self-referential — it scored its own text, always giving
     similar high confidence regardless of the actual query.

     New code verifies each fact against the RAW SOURCE SNIPPETS from
     SearchAgent, so confidence scores reflect real source support.
     Each fact also gets a unique score based on how strongly the
     specific sources back it up.
"""

import os
import json
from google import genai
from dataclasses import dataclass
from typing import List
from agents.summary_agent import ResearchSummary
from agents.search_agent import SearchResult


@dataclass
class FactResult:
    claim: str
    verdict: str        # "VERIFIED" | "UNVERIFIED" | "DISPUTED"
    confidence: float   # 0.0 to 1.0
    reasoning: str
    source_ref: str     # which source(s) support or contradict this fact


@dataclass
class FactCheckReport:
    query: str
    results: List[FactResult]
    overall_confidence: float
    summary: str


class FactCheckAgent:
    def __init__(self):
        self.name = "FactCheckAgent"
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = "gemini-2.0-flash"

    def log(self, msg: str):
        print(f"[{self.name}] {msg}")

    def verify(self, research: ResearchSummary) -> FactCheckReport:
        self.log(f"Received {len(research.key_facts)} facts from SummaryAgent")
        self.log("Cross-referencing facts against raw source snippets...")

        # FIX: Build source context from the actual search results,
        # NOT from the summary text the LLM generated.
        source_context = self._build_source_context(research.sources_used)

        results = []
        for fact in research.key_facts:
            result = self._check_fact(fact, research.query, source_context)
            results.append(result)

        verified = [r for r in results if r.verdict == "VERIFIED"]
        overall = round(sum(r.confidence for r in results) / max(len(results), 1), 2)

        self.log(f"Verified {len(verified)}/{len(results)} claims. Avg confidence: {overall:.2f}")

        return FactCheckReport(
            query=research.query,
            results=results,
            overall_confidence=overall,
            summary=f"{len(verified)}/{len(results)} claims verified. Overall confidence: {overall*100:.0f}%"
        )

    def _build_source_context(self, sources: List[SearchResult]) -> str:
        """
        FIX: Use the raw snippets fetched from the web, not the
        LLM-generated summary. This grounds fact-checking in actual
        source content so each fact gets a score tied to real evidence.
        """
        parts = []
        for i, s in enumerate(sources, 1):
            parts.append(
                f"[Source {i}] {s.title}\n"
                f"Domain: {s.domain}\n"
                f"Content: {s.snippet}"
            )
        return "\n\n".join(parts)

    def _check_fact(self, claim: str, query: str, source_context: str) -> FactResult:
        """
        FIX: Old prompt gave Gemini the summary as context — it was just
        checking whether the fact matched the summary it had already written,
        which always agreed and produced near-identical confidence scores.

        New prompt gives Gemini the raw source snippets and asks it to:
        1. Find which specific source supports or contradicts the claim
        2. Assign confidence based on how explicitly the source states it
        3. Mark as UNVERIFIED if no source mentions it at all
        4. Return source_ref so the user can trace the reasoning
        """
        prompt = f"""You are a fact-checking agent. Verify the claim below using ONLY the source snippets provided.

CLAIM: {claim}

RAW SOURCE SNIPPETS (from live web search about "{query}"):
{source_context}

INSTRUCTIONS:
- VERIFIED: The claim is directly and explicitly supported by at least one source snippet.
- UNVERIFIED: No source snippet mentions this claim — it may be true but is not in these sources.
- DISPUTED: A source snippet contradicts or significantly conflicts with this claim.
- confidence: How explicitly does the best-matching source support the claim?
  - 0.85-0.99: Source states it directly and clearly
  - 0.65-0.84: Source implies it or partially confirms it
  - 0.40-0.64: Tangentially related but not clear confirmation
  - 0.10-0.39: No support or contradicted
- source_ref: Name the specific source (e.g. "Source 2 — Reuters") that most supports or disputes the claim.

Respond ONLY with JSON (no markdown, no backticks):
{{
  "verdict": "VERIFIED" or "UNVERIFIED" or "DISPUTED",
  "confidence": 0.0 to 1.0,
  "reasoning": "One sentence citing the specific source evidence or lack thereof.",
  "source_ref": "Source N — domain name"
}}"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            raw = response.text.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            return FactResult(
                claim=claim,
                verdict=parsed.get("verdict", "UNVERIFIED"),
                confidence=float(parsed.get("confidence", 0.5)),
                reasoning=parsed.get("reasoning", ""),
                source_ref=parsed.get("source_ref", ""),
            )
        except Exception as e:
            self.log(f"Fact check failed for claim: {e}")
            return FactResult(
                claim=claim,
                verdict="UNVERIFIED",
                confidence=0.4,
                reasoning="Could not verify automatically.",
                source_ref="",
            )
