# Implementation Plan - Level 1 Data Ingestion Agent (with Quality Focus)

This plan outlines the creation of a **Level 1 Data Ingestion Agent** in `openlogic-finance/data-ingestion`. Building on the "Agent Quality" pillars from Day 4 of the ADK whitepaper, we will treat evaluation and observability as first-class citizens, while ensuring easy deployment to Google Cloud Run.

## User Review Required

> [!IMPORTANT]
> **Quality as a Pillar**: We are not just building a "script"; we are building an observable and evaluatable agent. This requires adding a telemetry layer and a benchmark dataset.
>
> **Secrets**: You will need to provide a `yfinance` compatible environment (typically no key for basic features, but robustness testing will simulate API failures).

## Proposed Changes

### 🛡️ Quality Architecture (Day 4 Alignment)

1.  **Observability**: The agent will be instrumented using ADK's Telemetry.
    -   **Logs**: Structured JSON logging for every tool call.
    -   **Traces**: Full trajectory tracking (Thought → Action → Observation).
2.  **Robustness**: Implementation of exponential backoff for `yfinance` calls to handle transient network issues or rate limits.
3.  **Evaluation**: Creation of a `spy_fetch.test.json` evaluation set to lock in expected behavior (10 years of data, correct chart output).

### 🚀 Deployment Architecture (Day 5 Alignment)

To seamlessly deploy this agent to Google Cloud Run (aligning with the "Prototype to Production" principles of ADK Day 5), the agent will be containerized. We will use the ultra-fast `uv` package manager within a `python:3.11-slim` Docker image and serve the agent using ADK's built-in `api_server`.

---

### `openlogic-finance` Repository

#### [NEW] [data-ingestion/](file:///Users/shreyas/gitrepos/OpenSource/openlogic-finance/data-ingestion/)
#### [NEW] [data-ingestion/Dockerfile](file:///Users/shreyas/gitrepos/OpenSource/openlogic-finance/data-ingestion/Dockerfile)
Production-ready container definition using Python 3.11 and the `uv` package manager, configured to expose the ADK API server for Cloud Run.

#### [NEW] [data-ingestion/requirements.txt](file:///Users/shreyas/gitrepos/OpenSource/openlogic-finance/data-ingestion/requirements.txt)
Strictly pinned dependency file for reproducibility.

#### [NEW] [data-ingestion/__init__.py](file:///Users/shreyas/gitrepos/OpenSource/openlogic-finance/data-ingestion/__init__.py)
#### [NEW] [data-ingestion/agent.py](file:///Users/shreyas/gitrepos/OpenSource/openlogic-finance/data-ingestion/agent.py)
Root agent definition using `LlmAgent` for complex reasoning over long-term data.

#### [NEW] [data-ingestion/tools.py](file:///Users/shreyas/gitrepos/OpenSource/openlogic-finance/data-ingestion/tools.py)
Enhanced with quality-centric checks:
-   `fetch_stock_data`: Returns metadata about the size of the dataset fetched.
-   `plot_stock_data`: Validates that the input data matches the expected time range.

#### [NEW] [data-ingestion/eval/spy_fetch.test.json](file:///Users/shreyas/gitrepos/OpenSource/openlogic-finance/data-ingestion/eval/spy_fetch.test.json)
Initial evaluation set for regression testing output quality.

#### [MODIFY] [implementation-plan.md](file:///Users/shreyas/gitrepos/OpenSource/openlogic-finance/data-ingestion/implementation-plan.md)
Updated local copy for the repository.

---

## Open Questions

-   **Safety**: Should we implement PII scrubbing for the logs if they contain user-specific request metadata?
-   **Metrics**: Are there specific "efficiency" KPIs you want to track (e.g., maximum execution time for the 10-year fetch)?
-   **Evaluation**: Should we use an "LLM-as-a-Judge" to evaluate the "descriptive quality" of the agent's explanation of the chart?

## Verification Plan

### Automated Tests
- `adk eval openlogic-finance/data-ingestion`: Run the evaluation suite.
- `pytest tests/`: Unit tests for the data fetching tools.

### Manual Verification
1.  **Trace Review**: Use `adk web` to inspect the agent's "Thought" process for efficiency.
2.  **Artifact Audit**: Check the generated `assets/spy_chart.png` for visual accuracy.
3.  **Container Build**: Run `docker build -t data-ingestion-agent .` to ensure the container builds successfully.
