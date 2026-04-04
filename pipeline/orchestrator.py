"""
Orchestrator — coordinates SearchAgent → SummaryAgent → FactCheckAgent
and produces the final research report.
"""

import time
from dataclasses import dataclass
from agents.search_agent import SearchAgent
from agents.summary_agent import SummaryAgent
from agents.factcheck_agent import FactCheckAgent, FactCheckReport
from agents.summary_agent import ResearchSummary


@dataclass
class ResearchReport:
    query: str
    summary: ResearchSummary
    fact_check: FactCheckReport
    elapsed_seconds: float

    def display(self):
        print("\n" + "="*60)
        print(f"  RESEARCH REPORT")
        print(f"  Query: {self.query}")
        print(f"  Time:  {self.elapsed_seconds:.1f}s")
        print("="*60)

        print("\n📝 SUMMARY")
        print("-"*40)
        print(self.summary.summary)

        print("\n✅ FACT CHECK RESULTS")
        print("-"*40)
        for r in self.fact_check.results:
            icon = "✓" if r.verdict == "VERIFIED" else ("✗" if r.verdict == "DISPUTED" else "?")
            pct = int(r.confidence * 100)
            print(f"  {icon} [{pct}%] {r.claim}")
            print(f"       → {r.reasoning}")

        print(f"\n  Overall confidence: {self.fact_check.overall_confidence*100:.0f}%")

        print("\n🔗 SOURCES")
        print("-"*40)
        for i, s in enumerate(self.summary.sources_used, 1):
            print(f"  [{i}] {s.title}")
            print(f"      {s.url}")

        print("\n" + "="*60)


class Orchestrator:
    def __init__(self):
        self.search_agent  = SearchAgent(max_results=5)
        self.summary_agent = SummaryAgent()
        self.fact_agent    = FactCheckAgent()

    def run(self, query: str) -> ResearchReport:
        print(f"\n🚀 Pipeline started for: '{query}'\n")
        start = time.time()

        # Stage 1 — Search
        sources = self.search_agent.search(query)

        # Stage 2 — Summarize
        summary = self.summary_agent.summarize(query, sources)

        # Stage 3 — Fact Check
        fact_check = self.fact_agent.verify(summary)

        elapsed = round(time.time() - start, 2)
        print(f"\n✅ Pipeline complete in {elapsed}s")

        return ResearchReport(
            query=query,
            summary=summary,
            fact_check=fact_check,
            elapsed_seconds=elapsed,
        )
def run_pipeline(query: str):
    orchestrator = Orchestrator()
    report = orchestrator.run(query)

    # Convert complex object → simple JSON-friendly output
    return {
        "summary": report.summary.summary,
        "facts": [
            {
                "claim": r.claim,
                "verdict": r.verdict,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
                "source_ref": getattr(r, "source_ref", ""),
            }
            for r in report.fact_check.results
        ],
        "sources": [
            {
                "title": s.title,
                "url": s.url,
                "domain": getattr(s, "domain", ""),
                "relevance_score": getattr(s, "relevance_score", 0.0),
            }
            for s in report.summary.sources_used
        ],
        "confidence": report.fact_check.overall_confidence
    }