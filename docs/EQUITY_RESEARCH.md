# Equity Research Assistant

A grounded **RAG agent over earnings-call transcripts** — an analyst asks a
question, the agent retrieves cited passages from the call/filings and answers
**only** from them (no hallucinated guidance), under responsible-AI grounding
governance. It's a second **vertical slice** through OpenLogic's 6-box
architecture (the first being the MoE-F market forecaster), and it's directly
on-mission: *research → ML → AI agents for transparent market foresight.*

> **Status:** Phases 1–5 complete, tested offline — transcript RAG (P1), the
> return/regime model with a full validate → deploy → monitor MDLC (P2), the agent
> orchestrating RAG ⇄ the model with a human-in-the-loop approval gate + agent
> evals (P3), data platforms (Snowflake/Databricks/AWS/SQL, P4), and a Streamlit
> research console + case study (P5).

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

## Phase 2 — return/regime model (validate → deploy → monitor) ✅

The classical-ML MDLC, end to end, across three boxes:

| Box | Module | Role |
|---|---|---|
| **2 — `model_library/ml_zoo/`** | `return_regime.py` | features + labels + sklearn model; `predict_proba_up` → regime (bear/neutral/bull); save/load |
| **3 — `strategy_testing/validation/`** | `report.py` | **validate**: AUC, accuracy, Brier (calibration), KS, time-series CV, feature **PSI**, confusion matrix + a **sign-off gate** |
| **5 — `live_paper_execution/serving/`** | `predict.py` | **deploy**: load a promoted model + score (Vertex/SageMaker in prod) |
| **5 — `live_paper_execution/monitoring/`** | `drift.py` | **monitor**: data drift (PSI), prediction drift, performance decay → **retrain trigger** |
| **foundation** | `horizontal_foundation/stats.py` | shared PSI + KS (used by both validation and monitoring) |

```python
# train → validate (gate) → save/serve → monitor for drift
import pandas as pd
from model_library.ml_zoo.return_regime import ReturnRegimeModel, build_training_frame
from strategy_testing.validation.report import validate
from live_paper_execution.monitoring.drift import monitor

df = pd.read_csv("assets/SPY_10y.csv")          # any OHLCV with a Close column
X, y = build_training_frame(df, horizon=5)
cut = int(len(X) * 0.7); Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

model = ReturnRegimeModel().train(Xtr, ytr)
report = validate(model, Xtr, ytr, Xte, yte)
print(report.summary())                          # [PASS/FAIL] AUC=… Brier=… PSI=…
if report.passes_gate():
    model.save("return_model.joblib")            # promote only on PASS

# later, in production:
print(monitor(Xtr, Xte, model.predict_proba_up(Xtr),
              model.predict_proba_up(Xte)).summary())   # [OK/RETRAIN] …
```

> Note on honesty: predicting short-horizon market direction is genuinely hard —
> real AUC hovers near 0.5. The **tests** verify the *machinery* (the model
> trains, the gate blocks weak models, drift fires the retrain trigger), not that
> it beats the market. That's the right thing to test in an MDLC.

## Phase 3 — agent orchestration + human-in-the-loop ✅

The agent now ties the workflow together with **three tools + a governance callback**:

| Piece | Where | Role |
|---|---|---|
| `retrieve_context` | `agentic_workflows/equity_research/tools.py` | RAG — what management *said* (cited) |
| `predict_regime(ticker)` | same | the return model's quantitative regime signal (Box 2) |
| `publish_recommendation` | same | the **consequential** action (BUY/HOLD/SELL note) |
| HITL approval gate | `risk_management/governance/approval.py` | `before_tool_callback` blocks publishing with `PENDING_HUMAN_APPROVAL` until `state["human_approved"]` is True |
| agent evals | `agentic_workflows/equity_research/eval/` | trajectory: retrieve → predict → cited thesis (schema-validated in CI) |

```bash
adk run agentic_workflows/equity_research        # needs GEMINI_API_KEY
#  ask: "Give me a research call on SPY using the earnings call and the model signal."
#   → it retrieves cited guidance, gets the model regime, drafts a rated thesis,
#     and when it tries to publish, returns PENDING_HUMAN_APPROVAL (HITL gate).
```

The approval gate reuses the **same `before_tool_callback` pattern as the trade
risk-veto** (`risk_management/portfolio/guardrail.py`) — one governance idiom, two
uses. Grounding still applies: cite transcript passages or abstain.

## Phase 4 — data platforms (AWS / Snowflake / Databricks / SQL) ✅

Built **local-first** (SQLite, local FS, pandas) so the SQL/interfaces are real and
run offline, with the identical code targeting the cloud in production. Full
breakdown: [`docs/DATA_PLATFORMS.md`](DATA_PLATFORMS.md).

| Platform | Module | Local → prod |
|---|---|---|
| **Snowflake** (SQL feature store; point-in-time + monitoring marts) | `data_prep/feature_store.py` | SQLite → Snowflake (same SQL) |
| **Snowflake** (governance audit log) | `risk_management/governance/audit.py` | SQLite → Snowflake table |
| **S3 / GCS** (transcripts + model artifacts) | `horizontal_foundation/storage.py` | local FS → boto3 / google-cloud-storage |
| **Databricks** (feature job) | `data_prep/pipelines/feature_pipeline.py` | pandas → Spark/Delta |

## Phase 5 — Streamlit research console + case study ✅

`interface/streamlit/equity_research_app.py` — a stakeholder console that ties
P1–P4 together: ask → **retrieve** cited evidence → **model regime** → draft a
rated note → **human approval (HITL)** → publish, with a live **governance audit
trail**. Runs offline.

```bash
make research-console        # streamlit run interface/streamlit/equity_research_app.py
```

Business framing for non-technical stakeholders:
[`docs/EQUITY_RESEARCH_CASE_STUDY.md`](EQUITY_RESEARCH_CASE_STUDY.md).

**All phases (P1–P5) complete.** The slice demonstrates RAG · vector DBs ·
embeddings · prompt orchestration · classical ML + validate→deploy→monitor MDLC ·
agentic AI · governance · HITL · evals · AWS/GCP + Snowflake/Databricks/SQL ·
Streamlit — one reusable 6-box architecture on Google ADK.

## Skills demonstrated (for reference)

RAG · vector DBs · embeddings · prompt orchestration (P1) · classical ML + MDLC
monitoring (P2) · agentic AI + governance + HITL (P3) · AWS/GCP + Snowflake/
Databricks/SQL (P4) · Streamlit (P5) — all on Google ADK.
