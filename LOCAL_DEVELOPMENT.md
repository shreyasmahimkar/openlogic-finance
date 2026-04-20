# Local Development Guide

Welcome to the `openlogic-finance` local development guide! This repository employs a multi-agent microservice architecture utilizing the Google Agent Development Kit (ADK). 

To ensure a smooth developer experience, we utilize a **Hybrid Environment Approach**:
* **One unified virtual environment locally** for seamless IDE integration, easy debugging, and frictionless cross-module imports.
* **Separated Docker containers** (via `docker-compose`) for isolated, production-ready deployments.

## Prerequisites

We highly recommend using `uv` over standard `pip` or `conda`. It is written in Rust, blazingly fast, and effortlessly handles our requirements.

If you don't have `uv` installed, install it via:
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

*(Note: Ensure you have Docker installed if you plan on running the entire system interactively).*

---

## 1. Setting Up the Unified Local Environment

At the root of the repository, initialize a single, unified virtual environment. This resolves all the `google.adk` integrations and stops your IDE (VS Code, PyCharm, etc.) from showing unresolved import errors on relative paths.

### Create and Activate the Environment
```bash
# Generate the virtual environment (.openlogic-env)
uv venv .openlogic-env

# Activate it (macOS/Linux)
source .openlogic-env/bin/activate
```
*(Tip: Point your IDE's Python Interpreter to this new `.openlogic-env/bin/python` path).*

### Install Agent Dependencies
Because our agents are developed independently but run together, we aggregate their requirements linearly into our local environment:

```bash
uv pip install -r utility_agents/market_data/requirements.txt
uv pip install -r research_papers_to_agents/moe_coordinator/requirements.txt
```

---

## 2. Interactive Development and Testing

With our `.openlogic-env` populated, you can easily develop and test logic. The Google ADK provides tools to immediately launch and evaluate specialized agents.

### Testing Individual Agents

You can fire up the local development UI or run evaluation cases against specific agents without engaging the full Docker ecosystem:

**Testing Market Data Agent:**
```bash
adk web utility_agents/market_data
# or run evaluations:
adk eval utility_agents/market_data utility_agents/market_data/eval/spy_fetch.test.json
```

**Testing MoE-F Coordinator:**
```bash
adk web research_papers_to_agents/moe_coordinator
# or run evaluations:
adk eval research_papers_to_agents/moe_coordinator research_papers_to_agents/moe_coordinator/eval/moef_eval.test.json
```

---

## 3. Running the Full System (Docker Compose)

For production orchestration, we fall back to Docker. When you invoke `docker-compose`, it ignores your local `.openlogic-env` entirely. Instead, it builds individual isolated environments for `market_data` and `moef_coordinator` by processing their respective `Dockerfiles`.

1. Ensure the Docker daemon is running on your system.
2. Spin up the cluster:
```bash
docker-compose up --build
```
This will mount the local `./shared-assets` directory into the containers and wire the network so that the Coordinator can successfully interface with the Market Data backend natively.
