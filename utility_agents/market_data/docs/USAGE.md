# Usage Guide: Data Ingestion Agent

This guide outlines how to interact with the Data Ingestion Agent locally using the Google ADK CLI, and how to deploy it using Cloud Run.

## Prerequisites

Ensure you have initialized the virtual environment with Python 3.11 as defined in the deployment plan:
```bash
uv venv --python "python3.11" "data-ingestion-env"
source data-ingestion-env/bin/activate
uv pip install -r market_data/requirements.txt
```

## Running the Agent Locally

The ADK framework offers multiple ways to interact with agents natively. You must be in the `openlogic-finance` repository root to correctly pass the module paths.

### 1. Interactive Web UI (Recommended for Trace Inspection)
You can test the agent and view its cognitive trace using the ADK Web UI:
```bash
adk web .
```
*   This will start a local server. Open the displayed URL in your browser.
*   Ask: *"Please fetch and plot SPY for the last 10 years."*
*   Navigate to the **Trace tab** in the UI to see the exact payloads sent to the `yfinance` tools, verifying the "Glass Box" observability principle.

### 2. Interactive CLI
For terminal-only environments:
```bash
adk run data-ingestion
```

## Running Evaluations

To verify the quality metrics of the agent against our established benchmark dataset, use the `eval` command:
```bash
adk eval market_data market_data/eval/spy_fetch.test.json
```
This ensures the agent is satisfying the Day 4 "Effectiveness" criteria by picking the intended tools without hallucinations.

## Deploying to Cloud Run

The included Dockerfile is ready for Cloud Run. The container runs the `adk api_server` on startup, which exposes the agent as a REST API.

```bash
cd market_data
docker build -t market_data_agent .

# Assuming you have gcloud installed:
# gcloud run deploy market_data_agent --image market_data_agent --region us-central1 --allow-unauthenticated
```
