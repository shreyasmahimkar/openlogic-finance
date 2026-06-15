# OpenLogic Finance — Implementation Document

- **GitHub:** https://github.com/shreyasmahimkar/openlogic-finance
- **Deployed app:** https://openlogicfinance.streamlit.app
- **Branch documented:** `main`

## Summary

OpenLogic Finance is an open-source quantitative-finance platform that
democratizes institutional-grade market foresight by replacing "black box" models
with **transparent, multi-agent AI systems** on the **Google Agent Development Kit
(ADK)**. Rather than isolated proprietary strategies, it offers a collaborative,
auditable forecasting environment whose models can be understood at multiple
levels of expertise.

The repository is built on a **6-Box Model Architecture** with a strict boundary
between horizontal infrastructure and vertical agent orchestration. Crucially, the
same architecture now powers **two distinct products**, proving it is a reusable
pattern rather than a one-off:

1. **MoE-F Market Forecaster** — a Level-3 multi-agent system (a parallel swarm of
   LLM experts + stochastic filtering + PAC-Bayes Gibbs aggregation) that forecasts
   market regime.
2. **Equity Research Assistant** — a governed RAG agent over earnings-call
   transcripts, combined with a monitored ML model and a human-in-the-loop
   approval gate.

The codebase is engineered as **agentic engineering, not vibe coding**: every
capability is backed by specs, unit tests, scored evals, runtime guardrails, CI,
and observability (~9k LOC · **146 tests** · 6 ADK evalsets + a scored RAG eval ·
8 `AGENTS.md` rulebooks · CI-gated).

---

## Architecture

```
                 ┌──────────────────────────────────────────────────────┐
                 │      AGENTIC WORKFLOWS (Vertical orchestration)        │
                 │  MoE-F coordinator · Equity Research Assistant (ADK)   │
                 └───────────────────────────┬──────────────────────────┘
 ┌───────────────────────────────────────────▼───────────────────────────────────┐
 │                                 6-BOX MODEL                                     │
 │ Box1 Data Prep → Box2 Model Library → Box3 Strategy Testing →                  │
 │ Box4 Risk Mgmt → Box5 Live/Paper Execution → Box6 Interface                    │
 └───────────────────────────────────────────▲───────────────────────────────────┘
                 ┌───────────────────────────┴──────────────────────────┐
                 │   HORIZONTAL FOUNDATION (Infrastructure)               │
                 │  config · stats · storage · observability · interpret. │
                 └────────────────────────────────────────────────────────┘
```

**Dependency direction is one-way:** every box imports from the horizontal
foundation (which imports from nothing); later boxes may use earlier ones, never
the reverse. Each box carries its own `AGENTS.md` with local rules.

---

## 1. Horizontal Foundation

`horizontal_foundation/` — the shared substrate every box imports.

- **System config & logging** — `config/system_config.py` (central paths, default
  ticker/period, cache TTL) and `utils/logging_helpers.py`.
- **Interpretability Engine** — `interpretability/explain_engine.py`
  (`ExplanationEngine`): multi-tier natural-language translation of quantitative
  events. *Beginner (age 11+)* simplifies data into real-world analogies (SPY ≈ a
  basket of the 500 largest US companies); *Academic (Jim Simons level)* gives
  rigorous, math-heavy breakdowns (sample size, stationarity, dividend adjustment,
  drift).
- **Observability** — `observability.py`: OpenTelemetry tracing for local agent
  runs (`OPENLOGIC_TRACING=1`); the cloud deploy traces on Vertex. The
  interpretability engine is the human-readable layer over these machine traces.
- **Shared statistics** — `stats.py`: `population_stability_index` (PSI) and
  `ks_statistic`, used by both model validation (Box 3) and drift monitoring (Box 5)
  without violating box direction.
- **Object storage** — `storage.py`: an object-store abstraction (`LocalObjectStore`)
  that mirrors **S3 / GCS** with the same `put`/`get`/`list` surface.

---

## 2. Box 1: Data Prep

`data_prep/` — historical data, news, macro context, feature engineering, and the
RAG ingestion stack.

**Market & news connectors (ADK agents):**
- `connectors/market_data/` — `MarketDataConnector` downloads daily OHLCV (SPY,
  AAPL, GOOG, BTC-USD, …) from Yahoo Finance, persists to CSV, and parses metadata.
- `connectors/financial_news/` — the `financial_news_agent` fetches date-windowed
  news from a **New York Times MCP server** (Model Context Protocol), caching to CSV.
- `connectors/global_events/` — the `global_events_agent` overlays historical
  macro regimes (Bull/Bear/Neutral) onto price charts and web-searches to patch gaps.

**Feature engineering:**
- `model_library/technical/indicators.py::enrich_ohlcv_data` computes MACD
  (EMA12−EMA26), Bollinger Bands (SMA20 ± 2σ), 30-day RSI/CCI/DX, and 20/30/60-day SMAs.
- Semantic news filtering via TF-IDF + cosine similarity (0.2 threshold) strips noise.

**RAG ingestion stack (`data_prep/rag/`) — real semantic retrieval:**
- `embeddings.py` — real **Google `text-embedding-004`** via the `google-genai`
  SDK (auto-used with `GEMINI_API_KEY`/Vertex); a deterministic L2-normalized
  hashing fallback keeps everything runnable offline/CI.
- `vector_store.py` (in-memory cosine) and `chroma_store.py` (a real **ChromaDB**
  vector database, HNSW + cosine) — same `add`/`search` interface; swappable.
- `indexing.py` — chunk → embed → index (`build_index(corpus, backend="chroma")`).
- `loaders.py` — ingest real `.txt` / `.md` / `.pdf` filings (`sample_filings/` holds
  a 10-K MD&A excerpt).

**Data-platform layer (local-first; cloud-identical):**
- `feature_store.py` — a SQL feature store (SQLite → **Snowflake**): point-in-time
  reads (no lookahead leakage) + monitoring-aggregation marts.
- `pipelines/feature_pipeline.py` — a feature job (pandas → **Databricks/Spark**).

---

## 3. Box 2: Model Library

`model_library/` — the canonical home for forecasting models, the expert swarm,
aggregation math, retrieval, and model routing.

**MoE-F expert swarm (`agentic_ai/experts.py`):** `build_moe_parallel_swarm()`
fans out three `LlmAgent` perspectives in parallel:
1. **Technical / Momentum** (`Llama_Expert`),
2. **Fundamental / Macro** (`GPT4o_Expert`),
3. **Contrarian / Mean-Reversion** (`Mixtral_Expert`).
Each emits a single float in [0,1] (the output contract) and uses the
`read_market_indicators` tool. Experts are **factories** (fresh per pipeline) so
the same swarm can be built for multiple entrypoints.

**Model routing (`agentic_ai/model_registry.py`):** `get_model(role)` maps a
logical role → a concrete model. Defaults to **Gemini** so the full pipeline runs
on a Google account alone; `OPENLOGIC_HETEROGENEOUS_EXPERTS=1` restores the
Llama/GPT/Mixtral mix via LiteLLM. No hard-coded model strings in agents.

**Stochastic filtering (`ml_zoo/filters.py`):** `stochastic_filter_update`
implements the discrete Euler–Maruyama update for a **Wonham–Shiryaev filter**,
adjusting the belief vector π over which expert is currently most accurate from
forecasting innovations.

**PAC-Bayes Gibbs aggregation (`ml_zoo/filters.py`):** `robust_gibbs_aggregation`
computes a softmin ensemble and an outer-loop update of the transition-intensity
**Q-matrix** via a principal matrix logarithm (`scipy.linalg.logm`) with a
regularization perturbation.

**Pure-Python logistic regression (`ml_zoo/logistic_regression.py`):** a
zero-dependency LR (scales `sma_ratio`/`rsi_norm`/`momentum`, numerically stable
sigmoid, BUY/SELL/NONE transitions, weight projection back to raw feature space).

**Return/regime model (`ml_zoo/return_regime.py`):** a scikit-learn model that
predicts P(up) over a horizon and maps it to a regime (bear/neutral/bull); the
*build* half of the classical-ML MDLC (validation lives in Box 3, monitoring in Box 5).

**RAG retriever (`retrieval/retriever.py`):** query embedding + vector search +
**cited** context formatting — the retrieval half of RAG.

---

## 4. Box 3: Strategy Testing & Evaluation

`strategy_testing/` — backtesting plus the **evaluation/validation** discipline.

**QuantConnect LEAN bridge (`lean_engine/`):** `LeanEngineBridge` syncs local
strategy files to QuantConnect Cloud, runs backtests, patches `config.json`
dynamically, and parses Net Return, CAGR, Max Drawdown, and order counts. Golden/
Death crossover and LR-probability strategies bridge into the LEAN runtime. The
project copies of strategy code are **generated** from `model_library` via
`scripts/sync_lean_strategies.py` (LEAN runs each project self-contained).

**Model validation — the *validate* half of the MDLC (`validation/report.py`):**
`validate()` → a `ValidationReport` with AUC, accuracy, Brier (calibration), KS,
time-series cross-validation, and feature PSI (stability), plus a **`passes_gate()`
sign-off** that blocks weak models from promotion.

**Scored RAG eval (`validation/rag_eval.py`, `rag_eval_cases.py`, `llm_judge.py`):**
RAG evaluation is *actually scored*, not just schema-valid — `context_recall@k`
(deterministic) + groundedness via an **LM-as-judge** (Gemini with `GEMINI_API_KEY`,
deterministic heuristic fallback for CI) + a `passes_gate()`. On the benchmark:
`context_recall@k=1.00, grounded(good)=0.94, grounded(bad)=0.23` — the judge
separates faithful answers from fabricated ones.

---

## 5. Box 4: Risk Management & Governance

`risk_management/` — risk controls and Responsible-AI governance, all as **code**.

**Active drawdown auditor (`portfolio/auditor.py`):** `run_audited_simulation`
fires a **Risk Veto** when a long position breaches a drawdown limit (15% standard
/ 8% strict), liquidating to cash and halting trading until a fresh Golden Cross.

**Trade risk-veto guardrail (`portfolio/guardrail.py`):** the auditor as a real
ADK `before_tool_callback` — it intercepts trade-shaped tool calls and **vetoes**
them on a drawdown breach (and latches the halt). A rule the machine enforces.

**RAG governance (`governance/`):**
- `grounding.py` — the grounding instruction + `is_grounded()` (cite or abstain;
  no fabricated guidance).
- `approval.py` — `make_research_approval_callback()`: the **human-in-the-loop**
  gate (an ADK `before_tool_callback`) that blocks publishing a recommendation with
  `PENDING_HUMAN_APPROVAL` until a human signs off. Same idiom as the trade veto.
- `audit.py` — a queryable **SQL audit log** (SQLite → Snowflake) of every
  retrieval, score, recommendation, and approval ("why BUY, and who signed off?").

---

## 6. Box 5: Live & Paper Execution

`live_paper_execution/` — deployment, serving, and production monitoring — the
*deploy → monitor* half of the MDLC.

- **GCP Vertex AI deploy (`cloud_deploy/deploy_vertex.py`):** packages the ADK app
  (`AdkApp`) and deploys to the **Vertex AI Agent Engine**; fully env-driven
  (`GOOGLE_CLOUD_PROJECT`/`STAGING_BUCKET`) with tracing enabled. See
  `docs/DEPLOY_VERTEX.md`.
- **Model serving (`serving/predict.py`):** loads a promoted model and scores
  (Vertex / **AWS SageMaker** endpoint in prod) — the *deploy* surface.
- **Monitoring (`monitoring/drift.py`):** data drift + prediction drift (PSI),
  performance decay, and an automated **retrain trigger** — the *monitor* surface.
- **Docker simulator (`simulators/docker-compose.yml`):** isolated containers for
  market-data ingestion and the agent swarm.

---

## 7. Box 6: Interface

`interface/` — user-facing tools.

- **MoE-F Streamlit dashboard (`streamlit/app.py`)** — the deployed app
  (https://openlogicfinance.streamlit.app): select parameters, run manual or
  autonomous 6-box simulations, trigger remote LEAN backtests, review scikit-learn
  metrics, inspect risk-auditor logs, and plot comparisons via Plotly.
- **Equity Research Console (`streamlit/equity_research_app.py`)** — ask → retrieve
  cited evidence (RAG) → model regime → draft a rated note → **human approval** →
  publish, with a live SQL **audit trail** (`make research-console`).
- **CLI (`cli/agent.py`)** and research **notebooks/**.

---

## 8. Agentic Workflows (Vertical Orchestration)

`agentic_workflows/` — the ADK layer that ties the boxes into workflows.

- **MoE-F coordinator (`model_library/agentic_ai/moe_coordinator/`):** the
  `moef_level_3_system` `SequentialAgent` — Data Ingestion → Expert Swarm
  (`ParallelAgent`) → Gibbs Synthesizer → Plotter. Run: `adk run
  model_library/agentic_ai/moe_coordinator`.
- **Equity Research Assistant (`agentic_workflows/equity_research/`):** an ADK
  agent with three tools — `retrieve_context` (RAG), `predict_regime` (the model),
  `publish_recommendation` (consequential, HITL-gated) — under grounding governance.
  Vector-DB backend is `OPENLOGIC_RAG_BACKEND` (`memory` default, `chroma` real DB).
  Run: `adk run agentic_workflows/equity_research`.

---

## The two products on one architecture

| | MoE-F Market Forecaster | Equity Research Assistant |
|---|---|---|
| **Goal** | forecast next-period market regime | answer "what did the call say, where's the stock headed?" |
| **Box 1** | OHLCV + news + indicators | RAG indexing (embeddings, Chroma, loaders) |
| **Box 2** | expert swarm + stochastic filter + Gibbs | retriever + return/regime model |
| **Box 3** | LEAN backtests | model validation + scored RAG eval |
| **Box 4** | drawdown risk-veto | grounding + HITL approval + audit |
| **Box 5** | Vertex deploy | serving + drift monitoring |
| **Box 6** | MoE-F dashboard | research console |

---

## Engineering discipline (MLOps / agentic engineering)

- **Reproducible env:** `uv` + `pyproject.toml` + `uv.lock`; optional extras
  (`rag`, `lean`, `interface`, `dev`).
- **Tests:** 146 pytest tests across all boxes (deterministic math, model MDLC,
  governance, RAG, the vector DB).
- **Evals:** 6 schema-valid ADK evalsets + a **scored** LM-judge RAG eval, all
  guarded in CI.
- **CI:** `.github/workflows/ci.yml` — lint (ruff) → tests → gitleaks secret-scan,
  plus an optional scored-eval job.
- **Guardrails:** pre-commit hooks (ruff + gitleaks) and runtime ADK callbacks
  (risk veto, HITL approval, grounding).
- **Context engineering:** root + per-box `AGENTS.md`, `docs/specs/`, `docs/memory/`
  (ADR-style decisions) — the harness is versioned like code.

**Companion docs:** `AGENTIC_ENGINEERING_SDLC_PLAN.md` (the vibe-coded → governed
journey), `EQUITY_RESEARCH.md`, `RAG.md`, `DATA_PLATFORMS.md`, `DEPLOY_VERTEX.md`,
`QUALITY_FLYWHEEL.md`, `BACKLOG.md`, `HITL.md`.

---

## Future Considerations

1. **Broker integration for live trading** — move Box 5 from mocked execution to
   live order routing (Alpaca, Interactive Brokers, Coinbase Advanced Trade).
2. **Multi-asset & universe selection** — dynamic screening across sectors/indices/
   commodities/crypto, running parallel swarms per asset class.
3. **Adaptive online reinforcement learning** — replace static LR weights and fixed
   Gibbs hyperparameters with an online RL recalibrator.
4. **Alternative-data connectors** — SEC filings, FOMC transcripts, and social
   sentiment (Reddit, X) via LLM parsers; promote `search_recent_events` from a
   simulated stub to a real web-search MCP.
5. **Real-document RAG at scale** — ingest live transcripts/filings into Chroma /
   Vertex Vector Search; add a reranker.
6. **CI/CD to Vertex Agent Engine** — auto-deploy on `main` when tests + evals pass.
7. **Cloud data platforms** — stand up the Snowflake / Databricks / SageMaker
   slices on trial accounts (the local-first code is already cloud-identical).

---

*This document reflects the `main` branch and supersedes earlier summaries by
adding the Equity Research Assistant, the real RAG stack (semantic embeddings +
Chroma vector DB + scored LM-judge evals), the classical-ML MDLC
(validate → deploy → monitor), the governance/HITL/audit layer, the data-platform
layer, and the agentic-engineering harness.*
