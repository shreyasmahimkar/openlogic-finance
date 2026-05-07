# OpenLogic Finance: Data Ingestion Agent

A Level 1 foundational agent configured using the Google Agent Development Kit (ADK) curriculum. Its sole purpose is parsing requests to fetch historical market data and automatically plotting visual insights for downstream usage in the multi-agent framework.

## Overview

The Data Ingestion Agent runs autonomously using Google Gemini 2.5. It features robust thick tools built upon the `yfinance` and `matplotlib` packages to interact with external financial data sources and visually articulate asset history.

### Core Abilities:
- Automatically identifies the user's intended stock ticker (falling back to default limits if ambiguous).
- Fetches accurate financial context with built-in networking failover routines (exponential backoff) using `tenacity` against transient Yahoo Finance outages.
- Persists data natively as static localized `.csv` files.
- Generates polished dark-mode temporal visual representations and exposes them as `png` assets.

## Documentation Navigation

This package strictly conforms to standard AgentOps deployment paradigms. Please refer to the specific documentation files included within this module:

* [**USAGE.md**](./docs/USAGE.md): Complete instructions on launching the ADK interface (`adk web`), executing the `adk eval` automated tracing test harness to verify metric efficiency, and containerizing this pipeline via `uv`/Docker for a seamless stateless deployment on Google Cloud Run.
* [**ARCHITECTURE.md**](./docs/ARCHITECTURE.md): Our engineering design choices. Details regarding its adherence as an ADK Level 1 simple agent, the roadmap to migrate tooling to the Model Context Protocol (MCP) paradigm, and our Cloud Storage (GCS) artifact integration plans to circumvent ephemeral container constraints.
* [**implementation-plan.md**](./docs/implementation-plan.md): The project iteration history showing its development footprint across the ADK's core pillars of effectiveness and observability.

## Getting Started Quickly

Start by bootstrapping your `uv` environment from the repository root:

```bash
uv venv --python "python3.11" "data-ingestion-env"
source data-ingestion-env/bin/activate
uv pip install -r market_data/requirements.txt
```

**1. Trace Cognitive Flows in the Browser:**
```bash
adk web market_data
```

**2. Verify Deterministic Tool Accuracy:**
```bash
adk eval market_data market_data/eval/spy_fetch.test.json
```
