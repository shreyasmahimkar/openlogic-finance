# Spec 0002: Financial News Agent

**Status:** Implemented
**Owner:** Data Prep
**Box(es):** 1 (Data Prep)

## Problem
Fetch, cache, and summarize date-bounded financial news to feed the SBERT/news
context used by the MoE-F experts.

## Behavior
- Input: a time period in natural language ("March 2026").
- The agent computes `begin_date`/`end_date` (YYYYMMDD) itself — never asks the user.
- Trajectory (cache-miss): `check_news_cache` → `search_articles` → `save_news_to_csv`.
  On cache hit: summarize without re-fetching.
- Output: a professional summary of top headlines + cache confirmation.

## Tools & dependencies
- `check_news_cache`, `save_news_to_csv` (FunctionTools, local files).
- `search_articles` via **MCP** (NYT server, `uvx`) — the canonical MCP pattern
  (`docs/memory/0005-mcp-policy.md`). Needs `NYT_API_KEY`.
- Model via `model_registry.get_model("orchestration")`.

## Success criteria (→ eval rubric)
- Correct date parsing; cache-aware trajectory; persists to `assets/`.
- Eval: `data_prep/connectors/financial_news/eval/financial_news.evalset.json`.

## Out of scope
Sentiment scoring (handled downstream by the SBERT filter).
