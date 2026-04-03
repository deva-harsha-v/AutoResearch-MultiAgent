"""
FactCheckAgent — verifies each extracted fact using Gemini.
Assigns a confidence score and verdict to each claim.
"""

import os
import json
from google import genai
from dataclasses import dataclass
from typing import List
from agents.summary_agent import ResearchSummary


@dataclass
class FactResult:
    claim: str
    verdict: str          # "VERIFIED" | "UNVERIFIED" | "DISPUTED"
    confidence: float     # 0.0 to 1.0
    reasoning: str


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
        self.log("Cross-referencing against knowledge base...")

        results = []
        for fact in research.key_facts:
            result = self._check_fact(fact, research.query, research.summary)
            results.append(result)

        verified = [r for r in results if r.verdict == "VERIFIED"]
        overall = round(sum(r.confidence for r in results) / max(len(results), 1), 2)

        self.log(f"Verified {len(verified)}/{len(results)} claims with avg confidence {overall}")

        return FactCheckReport(
            query=research.query,
            results=results,
            overall_confidence=overall,
            summary=f"{len(verified)}/{len(results)} claims verified. Overall confidence: {overall*100:.0f}%"
        )

    def _check_fact(self, claim: str, query: str, context: str) -> FactResult:
        prompt = f"""You are a fact-checking agent. Evaluate this claim about "{query}".

CLAIM: {claim}
CONTEXT: {context}

Respond ONLY with JSON (no markdown, no backticks):
{{
  "verdict": "VERIFIED" or "UNVERIFIED" or "DISPUTED",
  "confidence": 0.0 to 1.0,
  "reasoning": "one sentence explanation"
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
            )
        except Exception as e:
            self.log(f"Fact check failed for claim: {e}")
            return FactResult(
                claim=claim,
                verdict="UNVERIFIED",
                confidence=0.4,
                reasoning="Could not verify automatically."
            )
