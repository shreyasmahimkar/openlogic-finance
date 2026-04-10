# Architectural Design & Future Considerations

This document outlines the design philosophy behind the Data Ingestion Agent within OpenLogic Finance, linking specific technical decisions directly to the Google Agent Development Kit (ADK) curriculum.

## 1. How and Why It Was Built

We designed this module not as a loose collection of Python scripts, but as a production-grade component, adhering strictly to the ADK AgentOps Lifecycle.

### A. The Level 1 Agent Paradigm (Day 1)
This agent follows the simplest ADK blueprint. It is given a single, explicit goal (fetching and charting) and provided thick tools that encapsulate their dependencies (e.g., `yfinance` context). It handles the reasoning loop of mapping the user intent ("get SPY") to the execution sequence (`fetch_stock_data` → `plot_stock_data`).

### B. Agent Quality & Observability (Day 4)
*   **Logging**: Built-in Python logging provides the exact parameters used during tool execution. When run through `adk web`, these form the "Traces" that show the trajectory of the agent's logic.
*   **Robustness**: Network requests to external APIs like Yahoo Finance are inherently flaky. We introduced the `@retry` decorator via the `tenacity` library to provide exponential backoff, shifting a failure mode from "Error" to "Recovered."
*   **Evaluation as a Gate**: We established the `eval/` structure. The `spy_fetch.test.json` acts as our "Glass Box" evaluatable regression check, ensuring prompt adjustments don't break tool mappings.

### C. The Path to Production (Day 5)
*   **Dependency Speed**: Using `uv` within the `Dockerfile` reduces image build times from minutes to seconds, which is crucial for modern CI/CD agent deployments.
*   **Stateless Scaling**: The agent logic uses the runtime directory `/app/assets` explicitly, separating logic from data outputs. 

---

## 2. Future Considerations & Technical Debt

As you scale the OpenLogic "CandleSage" framework, you must address the following limitations of this v1 prototype:

### Ephemeral Storage (Critical Cloud Run Issue)
Currently, `tools.py` saves CSV files and PNG charts to the local file system (`/app/assets`). **Google Cloud Run container instances are ephemeral.** Upon termination, all local assets will be wiped out.
*   **Action Required**: Update `tools.py` to upload the generated artifacts (the CSV and the chart) to **Google Cloud Storage (GCS)**, and return the GCS signed URLs back to the LLM context.

### Migration to MCP (Model Context Protocol)
Right now, `yfinance` is baked directly into the ADK agent dependencies. To expand capabilities without bloating the agent's container limit:
*   **Action Required**: Move the financial data fetching logic into a standalone Model Context Protocol (MCP) server. The agent would then be re-configured to use the `mcp_tool` integration, meaning any agent in your ecosystem could query financial history without downloading `yfinance`.

### Agent-to-Agent (A2A) Scaling
This agent should not be the endpoint. It should act as a reliable sensory organ for the rest of your system.
*   **Action Required**: Expose this agent over the `A2A` protocol using `to_a2a()`. This will generate an Agent Card, allowing your future high-level `Root Orchestrator Agent` to delegate work to this deployment across the network seamlessly.
