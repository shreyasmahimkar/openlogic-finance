# Equity Research Assistant

A grounded **RAG agent over earnings-call transcripts** — an analyst asks a
question, the agent retrieves cited passages from the call/filings and answers
**only** from them (no hallucinated guidance), under responsible-AI grounding
governance. It's a second **vertical slice** through OpenLogic's 6-box
architecture (the first being the MoE-F market forecaster), and it's directly
on-mission: *research → ML → AI agents for transparent market foresight.*

> **Status:** Phase 1 (transcript RAG) implemented and tested offline. The
> return/regime model, monitoring, data platforms, and Streamlit console are
> phased below.

## What was done (Phase 1)

A complete, offline-runnable RAG slice, folded into the existing boxes (reuses the
ported FinSentinel example; no new top-level repo):

| Box | Module | Role |
|---|---|---|
| **1 — `data_prep/rag/`** | `embeddings.py` · `vector_store.py` · `indexing.py` | embed + index transcripts (Vertex AI, offline hashing fallback; pluggable vector store) |
| **2 — `model_library/retrieval/`** | `retriever.py` | query embedding + vector search + **cited** context block |
| **4 — `risk_management/governance/`** | `grounding.py` | grounding instruction + `is_grounded()` citation check (responsible AI) |
| **vertical — `agentic_workflows/equity_research/`** | `agent.py` · `corpus.py` | the ADK Gemini agent (model from the central registry) + sample transcripts |
| **3 — `strategy_testing/tests/`** | `test_equity_research_rag.py` | 6 offline tests (chunk/embed/retrieve/grounding/agent) |

**Verified:** 114 tests pass (108 existing + 6 new); ruff clean. The agent answers
*"What is the fiscal 2026 revenue guidance?"* by retrieving and citing the CFO
remarks — and refuses (cites nothing / abstains) when the transcripts don't cover it.

## How to run

Always run from the repo root, with the project env active.

```bash
# 1) Offline tests (no cloud, no keys)
make test                       # or: pytest strategy_testing/tests/test_equity_research_rag.py -q

# 2) Quick retrieval check (offline)
python -c "from agentic_workflows.equity_research import agent; \
print(agent.retrieve_context('fiscal 2026 revenue guidance'))"

# 3) Full agent (needs a key) — put GEMINI_API_KEY in .env, then:
adk run agentic_workflows/equity_research
#   ask: "What did management guide for fiscal 2026 revenue and margins?"
#   ask: "What is the dividend policy?"   → it should abstain (not in the transcripts)

# 4) Real Vertex embeddings instead of the offline fallback:
export GOOGLE_CLOUD_PROJECT=my-project GOOGLE_CLOUD_LOCATION=us-central1
#   (embeddings auto-switch to Vertex text-embedding-004; same code path)
```

**What to look for:** every answer cites bracketed sources `[1][2]`; questions
outside the transcripts get *"I don't have that in the provided transcripts."* —
never a fabricated number.

## Design notes

- **Grounding is a hard rule** (Box 4): the agent must cite or abstain. `is_grounded()`
  is the deterministic guardrail; a production system adds an LM-judge faithfulness
  score (Phase 3 evals).
- **Model routing:** the agent uses `model_registry.get_model("orchestration")` —
  Gemini by default, so it runs on a Google account alone (see
  `docs/memory/0004-model-provider-status.md`).
- **Import, don't copy:** the retriever imports the RAG infra from `data_prep.rag`;
  the agent imports the retriever + governance. One source each.

## Roadmap (next phases)

| Phase | Adds | Boxes |
|---|---|---|
| **P2** | a **return/regime prediction model** (direction next period) with full **validate → deploy → monitor** MDLC | `model_library` · `strategy_testing` · `live_paper_execution` |
| **P3** | the agent orchestrates **RAG ⇄ the return model**, + risk-veto + **human-in-the-loop** approval; agent/RAG **evals** | `agentic_workflows` · `risk_management` |
| **P4** | data platforms: transcripts in **GCS/S3**, **Snowflake** feature/audit marts, a **Databricks** feature job (AWS/Snowflake/Databricks/SQL) | `data_prep` · `live_paper_execution` |
| **P5** | a **Streamlit** research console (ask → retrieve → predict → explain → approve) + case study | `interface` |

## Skills demonstrated (for reference)

RAG · vector DBs · embeddings · prompt orchestration (P1) · classical ML + MDLC
monitoring (P2) · agentic AI + governance + HITL (P3) · AWS/GCP + Snowflake/
Databricks/SQL (P4) · Streamlit (P5) — all on Google ADK.
