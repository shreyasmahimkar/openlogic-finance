# Spec 0003: Global Events Agent

**Status:** Implemented (one tool is a stub — see below)
**Owner:** Data Prep
**Box(es):** 1 (Data Prep)

## Problem
Overlay macroeconomic regimes onto a price chart so users see *why* price moved.

## Behavior
- Input: a request to visualize an asset's macro context.
- Trajectory: `get_global_events` → (if the window exceeds stored history)
  `search_recent_events(start, end)` → `plot_asset_data`.
- Output: a chart artifact + a textual explanation aligning events to price.

## Tools & dependencies
- `get_global_events` (local CSV), `plot_asset_data` (matplotlib) — FunctionTools.
- `search_recent_events` — **currently returns a simulated result**; flagged as the
  next real MCP migration (web-search server) in `docs/memory/0005-mcp-policy.md`.
- Model via `model_registry.get_model("orchestration")`.

## Success criteria (→ eval rubric)
- Retrieves events before plotting; explanation references real regimes.
- Eval: `data_prep/connectors/global_events/eval/global_events.evalset.json`.

## Out of scope
Real-time macro feeds (pending the web-search MCP migration).
