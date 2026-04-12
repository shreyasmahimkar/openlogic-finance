# MoE-F Coordinator Agent

The mathematical brain of the OpenLogic ecosystem! 
It leverages a continuous-time Hidden Markov Model (Wonham-Shiryaev Filter) to evaluate its ensemble of foundation LLMs against true market moves retrieved by the `data_ingestion` agent. At each time step, it recalculates trust weights ($\pi_n$) via soft-min Gibbs aggregation, outputting an incredibly resilient stock prediction.

## 🏗️ Architecture Overview

It manages an ensemble of distinct foundational experts out-of-the-box:
- **The Technician** (Momentum tracking)
- **The Fundamentalist** (Macro regimes)
- **The Contrarian** (Mean reversal)

Rather than static routing, the filter dynamically shifts its trust to whichever expert provides the highest mathematical probability of success based on immediately historical ground-truth.

---

## 🚀 Running the Ecosystem (Docker Compose)

Because this agent relies heavily on the `data_ingestion` microservice to fetch its context and evaluate its ground-truth, they are orchestrated together via a root `docker-compose.yml` network.

Instead of paying for AWS S3 or GCP Storage, the containers share a local mounted directory (`/shared-assets`). When the ingestion handler pulls a massive CSV, it drops it logically onto your laptop so the `moe_coordinator` can read it simultaneously without network latency!

### Booting the Network

1. **Inject your core LLM keys:**
   The agents run on Gemini. Ensure you have a `.env` file at the root of the repository:
   ```bash
   echo "GEMINI_API_KEY=your_key_here" > .env
   ```

2. **Start all nodes in the background:**
   From the root repository directory run:
   ```bash
   docker compose up --build -d
   ```

### Accessing the Brains

Because we use the native Google ADK framework, each microservice exposes an interactive Server-Sent-Events (SSE) Web UI for "Glass Box" observability. You can watch the filters execute natively in your browser!

* **Data Ingestion Interface**: `http://localhost:8000`
* **MoE-F Coordinator Interface**: `http://localhost:8001`
