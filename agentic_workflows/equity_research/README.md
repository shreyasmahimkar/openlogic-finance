# equity_research (vertical slice)

The **Equity Research Assistant** — a grounded RAG agent over earnings-call
transcripts. Ties together `data_prep.rag` (index) → `model_library.retrieval`
(retrieve) → `risk_management.governance` (grounding) → a Gemini ADK agent.

```bash
adk run agentic_workflows/equity_research      # needs GEMINI_API_KEY
```

Full write-up + how-to: [`docs/EQUITY_RESEARCH.md`](../../docs/EQUITY_RESEARCH.md).
