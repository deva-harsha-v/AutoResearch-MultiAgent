# ⬡ AutoResearch — Multi-Agent Research System

A multi-agent AI pipeline that autonomously researches any topic using three specialized agents: **SearchAgent**, **SummaryAgent**, and **FactCheckAgent** — orchestrated by a central pipeline controller.

Built with Python + Claude API (Anthropic) + DuckDuckGo Search.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│   Orchestrator  │  ← pipeline/orchestrator.py
└────────┬────────┘
         │
    ┌────▼────┐       Web Search (DuckDuckGo)
    │ Search  │ ─────────────────────────────► 5 ranked sources
    │  Agent  │
    └────┬────┘
         │
    ┌────▼────┐       Claude API
    │ Summary │ ─────────────────────────────► Summary + Key Facts
    │  Agent  │
    └────┬────┘
         │
    ┌────▼────┐       Claude API
    │  Fact   │ ─────────────────────────────► Verified Claims + Confidence
    │  Check  │
    └────┬────┘
         │
    ┌────▼────────────┐
    │  Research Report│
    └─────────────────┘
```

---

## Project Structure

```
AutoResearch-MultiAgent/
├── agents/
│   ├── __init__.py
│   ├── search_agent.py      # Web retrieval + ranking
│   ├── summary_agent.py     # LLM synthesis + extraction
│   └── factcheck_agent.py   # Claim verification + scoring
├── pipeline/
│   ├── __init__.py
│   └── orchestrator.py      # Pipeline controller
├── ui/
│   └── dashboard.html       # Interactive browser demo
├── main.py                  # CLI entry point
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/AutoResearch-MultiAgent.git
cd AutoResearch-MultiAgent
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your Anthropic API key**
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```
Get a free key at [console.anthropic.com](https://console.anthropic.com)

---

## Usage

**CLI (Python backend)**
```bash
python main.py "Quantum computing advancements 2025"
python main.py "Impact of AI on healthcare"
python main.py "Renewable energy trends"
```

**Browser UI**
Open `ui/dashboard.html` in any browser. Enter your query and click **RUN AGENTS**.

---

## Sample Output

```
🚀 Pipeline started for: 'Large Language Models 2025'

[SearchAgent]  Decomposing query...
[SearchAgent]  Top 5 sources selected. Passing to SummaryAgent.
[SummaryAgent] Calling Claude API for synthesis...
[SummaryAgent] Extraction complete. 5 key facts generated.
[FactCheckAgent] Cross-referencing against knowledge base...
[FactCheckAgent] Verified 4/5 claims. Avg confidence: 0.84

============================================================
  RESEARCH REPORT
  Query: Large Language Models 2025
  Time:  8.3s
============================================================

📝 SUMMARY
LLMs have seen rapid capability growth through 2025...

✅ FACT CHECK RESULTS
  ✓ [91%] GPT-4 and Claude are leading commercial models
       → Well-documented across major AI benchmarks
  ✓ [88%] Context windows now exceed 100k tokens
       → Confirmed by official model documentation
  ...

🔗 SOURCES
  [1] MIT Technology Review - AI Models 2025
      https://...
```

---

## Agents

| Agent | Role | Tech |
|---|---|---|
| `SearchAgent` | Retrieves & ranks sources | DuckDuckGo, BM25 scoring |
| `SummaryAgent` | Synthesizes + extracts facts | Claude API |
| `FactCheckAgent` | Verifies claims + confidence | Claude API |
| `Orchestrator` | Coordinates the pipeline | Python |

---

## Tech Stack

- **Python 3.10+**
- **Anthropic Claude API** — claude-opus-4-5
- **DuckDuckGo Search** — free, no API key required
- **Vanilla HTML/JS** — browser UI demo

---

## License

MIT License — free to use and modify.

---

*Built as an internship project demo — AI Agents / Multi-Agent Systems domain.*
