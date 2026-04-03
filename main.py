"""
main.py — Entry point for AutoResearch Multi-Agent System
Usage: python main.py "your research query here"
"""

import sys
import os
from pipeline.orchestrator import Orchestrator


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"your research query\"")
        print("Example: python main.py \"Quantum computing advancements 2025\"")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        print("  $env:GEMINI_API_KEY = 'your_key_here'   (PowerShell)")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    orchestrator = Orchestrator()
    report = orchestrator.run(query)
    report.display()


if __name__ == "__main__":
    main()
