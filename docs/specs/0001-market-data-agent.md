# Spec 0001: Market Data Agent

**Status:** Implemented
**Owner:** Data Prep
**Box(es):** 1 (Data Prep)

## Problem
Ingest historical OHLCV for an asset and explain it at the user's expertise level,
so downstream boxes (and humans) start from clean, understood data.

## Behavior
- Input: a natural-language asset request (e.g. "ingest SPY 10y for a beginner").
- Output: dataset metadata + a human-readable explanation.
- Trajectory: call `fetch_and_explain(ticker, period, explanation_level)`.
- Constraints: translate names → Yahoo tickers (Bitcoin → BTC-USD); default SPY/10y;
  crypto is supported.

## Tools & dependencies
- `fetch_and_explain` (FunctionTool, yfinance) — see `docs/memory/0005-mcp-policy.md`.
- Indicator math imported from `model_library/technical/indicators.py`.
- Model via `model_registry.get_model("orchestration")`.

## Success criteria (→ eval rubric)
- Calls `fetch_and_explain` with the right ticker/period (tool_trajectory).
- Honors `explanation_level`; no rejection of crypto.
- Eval: `data_prep/connectors/market_data/eval/market_data.evalset.json`.

## Out of scope
Live streaming; feature engineering (see `data_prep/features`).
