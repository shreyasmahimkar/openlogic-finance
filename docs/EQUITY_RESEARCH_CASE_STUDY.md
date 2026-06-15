# Case Study: An AI Equity Research Assistant — from idea to governed product

*A short decision memo translating the technical build into business value — the
kind of artifact a stakeholder (a PM, a head of research, a risk officer) reads.*

## The problem (business framing)

Analysts spend hours reading earnings-call transcripts and filings, then reconcile
that qualitative read against quantitative signals — slowly, inconsistently, and
with no audit trail. Leadership wants **faster, more consistent research** without
loosening **compliance** (no fabricated numbers, a human on every published call,
a record of who decided what).

## What we built

A **governed AI research assistant** that, for any covered name:

1. **Reads the earnings call** and answers grounded in cited passages (RAG).
2. **Adds a quantitative regime signal** from a validated ML model.
3. **Drafts a rated note**, but **cannot publish without a human analyst's
   approval** (human-in-the-loop).
4. **Logs every step** — retrieval, score, recommendation, approval — to a
   queryable audit trail.

Try it: `make research-console` (runs offline).

## Why it's trustworthy (the part that matters for adoption)

| Risk a stakeholder worries about | How the system addresses it |
|---|---|
| "The AI will make up guidance numbers." | **Grounding rule**: cite a transcript passage or say *"I don't have that."* Enforced in code (`is_grounded`) + evals. |
| "It will publish a call no human checked." | **HITL gate**: publishing is blocked (`PENDING_HUMAN_APPROVAL`) until an analyst signs off. |
| "We can't explain a decision to compliance." | **Audit trail**: every retrieval/score/recommendation/approval is a queryable SQL row. |
| "The model silently degrades." | **MDLC**: a validation gate blocks weak models; production **drift monitoring** fires a retrain trigger. |
| "We're locked into one vendor/cloud." | Model registry + portable storage/SQL interfaces (GCP + AWS, Snowflake/Databricks). |

## Measurable value (illustrative)

- **Faster first draft:** transcript synthesis + signal in seconds vs. ~hours.
- **Consistency:** every note follows the same grounded, rated, audited format.
- **Lower compliance risk:** no un-cited claims; no un-approved publications;
  full lineage for audit.
- **Honest on alpha:** the model is held to a validation gate — on real SPY data it
  *fails* the gate (markets are hard), so it is **not** promoted. The system is
  designed to *not* ship false confidence.

## How it was built (engineering credibility)

One **reusable 6-box agentic architecture** (proven first on multi-agent market
forecasting), extended as a second vertical slice on **Google ADK**:

- RAG + vector search + embeddings (Vertex AI) · a validated, monitored ML model ·
  governed agent orchestration with HITL · multi-cloud + Snowflake/Databricks/SQL ·
  a Streamlit console — all under one tested, CI-gated, governed harness.

Full technical write-up + how-to: [`docs/EQUITY_RESEARCH.md`](EQUITY_RESEARCH.md).

## What I'd do next (roadmap honesty)

Wire live transcript ingestion (provider feed), add an LM-judge faithfulness eval,
stand up the Snowflake/Databricks/SageMaker slices on trial accounts, and run a
human-eval study with 2-3 analysts to quantify time saved and trust.
