# 0005 — MCP vs FunctionTool policy (Box 1)

**Status:** Active

**Decision:** Use the **Model Context Protocol (MCP)** for tools that reach an
**external, third-party service** (live APIs, web/news/search servers). Use plain
**ADK `FunctionTool`s** for **local, deterministic computation** (indicator math,
plotting, static-CSV lookups). MCP is for *tool access to external systems* — it
is not a wrapper to force onto in-process pandas/matplotlib code.

**Why:** the SDLC paper's "adopt MCP for tool access" is about external tool
servers and vendor optionality, not about replacing local functions. Forcing MCP
onto deterministic local code adds a server dependency and failure mode for zero
benefit.

**Current Box 1 mapping:**

| Connector | Tool | Access | Correct? |
|---|---|---|---|
| financial_news | `search_articles` | **MCP** (NYT server via `uvx`) | ✅ |
| financial_news | `check_news_cache`, `save_news_to_csv` | FunctionTool (local files) | ✅ |
| market_data | `fetch_and_explain` | FunctionTool (yfinance library) | ✅ acceptable¹ |
| global_events | `get_global_events`, `plot_asset_data` | FunctionTool (local CSV / matplotlib) | ✅ |
| global_events | `search_recent_events` | FunctionTool returning a **simulated** result | ⚠️ **next MCP migration** |

¹ `market_data` could move to a Yahoo Finance MCP server if/when one is adopted;
the yfinance library call is a reasonable interim.

**How to apply:**
- New external integrations go through MCP, wired like `financial_news/agent.py`
  (`McpToolset` + `StdioServerParameters`, command resolved via `shutil.which`).
- `global_events.search_recent_events` is the flagged candidate: replace the
  simulated string with a real web-search MCP server (needs a server + key).
- The old stub `data_prep/connectors/mcp_client.py` was deleted (dead code that
  pretended to be the MCP layer). See [[0003-import-dont-copy]].
