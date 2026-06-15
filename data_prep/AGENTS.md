# AGENTS.md — data_prep (Box 1: Data Preparation)

Root rules: [`/AGENTS.md`](../AGENTS.md).

Real-time market pipelines, unstructured news connectors, and financial feature
engineering. Entry point of the 6-box flow.

## Public surface

- `connectors/market_data/` — ADK agent + tools for OHLCV ingestion (Yahoo Finance).
- `connectors/financial_news/` — ADK agent + tools for news ingestion (cache + fetch).
- `connectors/global_events/` — ADK agent + tools for macro/global-event context.
- `rag/` — reusable RAG ingestion infra: `embeddings.py` (Vertex + offline fallback), `vector_store.py` (pluggable), `indexing.py` (chunk + index). Powers the Equity Research Assistant; see `docs/EQUITY_RESEARCH.md`.
- `features/`, `pipelines/` — feature engineering and pipeline assembly (build out as needed).

## Rules

- Each connector is a standard ADK agent package (`agent.py` exposing `root_agent`,
  `tools.py`, `__init__.py`). Tools have typed signatures and docstrings stating
  *when* the model should call them.
- **MCP vs FunctionTool** (see `docs/memory/0005-mcp-policy.md`): external
  third-party services go through **MCP** (wired like `financial_news/agent.py`);
  local deterministic computation stays a **FunctionTool**. Don't force MCP onto
  in-process pandas/plot code. `global_events.search_recent_events` is the next
  MCP migration (currently a simulated result).
- Technical-indicator math lives in `model_library/technical/indicators.py` — import it, don't reimplement here.
- Cache datasets under `assets/`; resolve paths via `horizontal_foundation` `SystemConfig`, not hard-coded strings.
