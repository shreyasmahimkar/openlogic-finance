# Google ADK Usage Guide: MoE-F Coordinator

Welcome to the native Google Agent Development Kit (ADK) integration of the **Mixture of Experts - Filter (MoE-F) Coordinator**! 

Because this application relies on a sophisticated Level 3 Multi-Agent architecture consisting of Sequential Pipelines and Parallel Fan-Out swarms, managing state and evaluating pathways requires explicit tooling. This guide details how you can trace the algorithmic flow, mathematically test trajectory outputs, and seamlessly transition into production on the cloud.

---

## 1. Tracing Cognitive Flows: The Interactive UI

Google ADK drastically reduces debugging time by exposing a built-in server-sent-event (SSE) Web UI out of the box. Instead of pouring through static terminal logs, you can trace the exact token outputs mapping from the Data Ingestion Phase all the way through the Swarm.

To trigger the interface, you must execute it from the **root of the `openlogic-finance` repository** so that it can correctly resolve both the coordinator and its dependencies (like `utility_agents`):

```bash
# Navigate to the root of the repository
# cd /Users/shreyas/gitrepos/OpenSource/openlogic-finance

# Make sure your shared isolated environment is active
source .openlogic-env/bin/activate

# Execute the local web harness for the full package space
PYTHONPATH=. adk web research_papers_to_agents
```

Alternatively, if you are already inside the `research_papers_to_agents/` folder, you must explicitly set the `PYTHONPATH` to include the root directory:
```bash
PYTHONPATH=.. adk web .
```

Open your browser to the local port specified (usually `http://localhost:8000`). From there, you can interact with the master `moef_level_3_system` natively. Look out for the Tool Call hooks to see the true transition dynamics logged by the `stochastic_filter_update` events!

> [!WARNING]
> **Local Web UI Execution Error (`ValueError: Model ... not found`)**
> If you trigger a full end-to-end execution through the Web UI without Vertex integration, it will crash when it hits the parallel layer because models like `llama-3-8b` and `mixtral-8x7b` are not natively registered in ADK's open-source local LLM registry (which defaults to Gemini). 
> **To fix this for local UI testing:** Temporarily change the model parameters in `experts.py` to `gemini-2.5-flash`, OR stick to running `python final_test.py` which isolates the logic and safely stubs these LLM calls entirely.

---

## 2. Running the Gen AI Evaluation Service

Unlike rule-based systems, generating quantitative certainty across LLM endpoints requires rigorous heuristic evaluation. The MoE-F Coordinator possesses custom json definitions tracing both component-level assertions and complete system routing.

Execute these commands to test the swarm before any deployments:

### Test Component Integrity (Phase 1 Ingestion)
Ensures the YFinance MCP and SBERT semantic tools trigger without hallucination.
```bash
adk eval moe_coordinator research_papers_to_agents/moe_coordinator/eval/ingestion.test.json
```

### Test Parallel Constraints (Phase 2 Swarm)
Tests the strict constraints placed horizontally across Llama/GPT/Mixtral LLMs guaranteeing discrete float output generation.
```bash
adk eval moe_coordinator research_papers_to_agents/moe_coordinator/eval/swarm.test.json
```

### Test Full Orchestration Handoffs
Uses the `trajectory_exact_match` metric to score 1 if the master ADK pipeline moves flawlessly from Extractor -> Swarm -> Gibbs Synthesis -> Plotter.
```bash
adk eval moe_coordinator research_papers_to_agents/moe_coordinator/eval/trajectory.test.json
```

---

## 3. Prototype to Production: Vertex AI Deployment

When your evaluation scores hit comfortable confidence intervals, you are ready to abstract away the local execution latency and tap into massive concurrency by migrating to Google Cloud's fully managed environment.

We have provided a streamlined deploy script leveraging the Vertex AI Python SDK.

### Deployment Prerequisites
Ensure your IAM identities are correct and you invoke the application with sufficient Cloud Storage privileges:

1. Authenticate with `gcloud` locally:
    ```bash
    gcloud auth application-default login
    ```
2. Navigate to the `deploy_vertex.py` script and update your `YOUR_PROJECT_ID` and `YOUR_STAGING_BUCKET` constants to match your GCP footprint.

### Execution
Run the deployment packaging tool:
```bash
python3 research_papers_to_agents/moe_coordinator/deploy_vertex.py
```

This encapsulates your `SequentialAgent` logic into an `AdkApp` and spins it up natively within the **Vertex AI Agent Engine**. Once deployed, your Swarm gains out-of-the-box infrastructure scaling and deep OpenTelemetry support to monitor HMM mathematical state health in the Cloud!

---

## 4. Testing & Running SPY 2025 Mock Loop
Since live LLM inference endpoints inside `agent.py` and `experts.py` are stubbed out via comments rather than using live execution API keys natively during local development runs, you can natively simulate a full 1-year SPY execution block which mimics the Stochastic filter pipeline:

1. Ensure the dummy dataset exists: `data/spy_2025_mock.csv` exists. If not, generate via `python data/generate_mock_data.py`.
2. Execute the test loop:
   ```bash
   # From the root of your repository
   PYTHONPATH=. python -m research_papers_to_agents.moe_coordinator.final_test
   ```
This loop sequentially applies the Llama/GPT/Mixtral mocked responses mapped against ground truth, integrates belief states incrementally via the Wonham-Shiryaev filter, normalizes with the Gibbs aggregator, and graphs the finalized rolling-trajectory locally in `moe_regimes.png` showcasing Bullish (1.0), Neutral (0.5), Bearish (0.0) market signals across the simulated 252 days.
