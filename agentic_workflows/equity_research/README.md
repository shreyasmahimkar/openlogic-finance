# equity_research (vertical slice)

The **Equity Research Assistant** — a grounded ADK agent that orchestrates
**RAG ⇄ the return/regime model** with a **human-in-the-loop** approval gate.
Tools: `retrieve_context` (RAG), `predict_regime` (model), `publish_recommendation`
(consequential, gated). Spans `data_prep.rag` → `model_library` (retriever + return
model) → `risk_management.governance` (grounding + HITL approval).

```bash
adk run agentic_workflows/equity_research      # needs GEMINI_API_KEY
```

Full write-up + how-to: [`docs/EQUITY_RESEARCH.md`](../../docs/EQUITY_RESEARCH.md).
