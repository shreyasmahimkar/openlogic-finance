# AGENTS.md — data_prep (Box 1: Data Preparation)

Root rules: [`/AGENTS.md`](../AGENTS.md).

Real-time market pipelines, unstructured news connectors, and financial feature
engineering. Entry point of the 6-box flow.

## Public surface

- `connectors/market_data/` — ADK agent + tools for OHLCV ingestion (Yahoo Finance).
- `connectors/financial_news/` — ADK agent + tools for news ingestion (cache + fetch).
- `connectors/global_events/` — ADK agent + tools for macro/global-event context.
- `connectors/mcp_client.py` — MCP client layer (YFinance MCP, SBERT semantic filter).
- `features/`, `pipelines/` — feature engineering and pipeline assembly (build out as needed).

## Rules

- Each connector is a standard ADK agent package (`agent.py` exposing `root_agent`,
  `tools.py`, `__init__.py`). Tools have typed signatures and docstrings stating
  *when* the model should call them.
- Standardize external access on **MCP** where possible (keeps vendor optionality).
- Technical-indicator math lives in `model_library/technical/indicators.py` — import it, don't reimplement here.
- Cache datasets under `assets/`; resolve paths via `horizontal_foundation` `SystemConfig`, not hard-coded strings.
