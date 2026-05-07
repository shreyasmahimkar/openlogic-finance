# MoE-F Coordinator (Level 3 Multi-Agent Ecosystem)

The mathematical brain of the OpenLogic ecosystem! Designed with the **Google Agent Development Kit (ADK)**, this package implements the **Mixture-of-Experts Filter (MoE-F)** using a sophisticated **Level 3 Multi-Agent Collaborative Architecture**. 

It transcends static ML routing, leveraging continuous-time Hidden Markov Models (the Wohman-Shiryaev Filter) to evaluate an ensemble of foundational LLMs iteratively. At each time step, it dynamically adjusts expert trust weights ($\pi_n$) based on previous realized performance against the true market via soft-min Gibbs aggregation.

## 🏗️ Google ADK Architecture Overview

Rather than a simplistic script, the MoE-F Coordinator orchestrates a massive 5-phase pipeline built entirely with programmatic ADK primitives:

1. **Phase 1: Data Ingestion (`SequentialAgent`)**
   - **MarketDataExtractor**: Leverages the `MCPToolset` communicating with the Yahoo Finance Model Context Protocol to fetch 10-years of SPY OHLCV data. Generates rolling Appendix C technical indicators (MACD, Bollinger, RSI, CCI).
   - **SBERT_SemanticFilter**: Validates semantic relevance of financial headlines utilizing tf-idf bounds to strip algorithmic noise.

2. **Phase 2: Stochastic Mathematical Filtering (`AgentTool`)**
   - The localized SDEs of the Wohman-Shiryaev filter are executed deterministically. Calculates gradient losses and diffusions for continuous Bayesian belief state ($pi_{agent\_id}$) updates natively persisting in ADK's `SessionState`.

3. **Phase 3: The MoE-F Swarm (`ParallelAgent`)**
   - Implements a parallel fan-out structure executing `expert_llama`, `expert_gpt`, and `expert_mixtral` `LlmAgent` objects synchronously across 7-day windows.

4. **Phase 4: Synthesis & Visualization Harmonization**
   - Uses `robust_gibbs_aggregation` inside the **SynthesizerAgent** to produce the final filtered ensemble prediction.
   - Triggers the Pandas rolling visualization (`df.rolling(window=7).mean()`) simulating the MoE-F tracked performance trajectory exactly as specified in the original research papers.

---

## 🚀 Native Agent Operations 

Because the `moe_coordinator` acts as an entirely independent distributed multi-agent system, it requires its own isolated virtual ecosystem to accurately execute tools and dependencies untouched by peripheral apps.

### 1. Initializing the UV Environment
We recommend utilizing `uv` for blistering-fast python environment spin-ups.
```bash
uv venv --python "python3.11" "moe-coordinator-env"
source moe-coordinator-env/bin/activate
uv pip install -r research_papers_to_agents/moe_coordinator/requirements.txt
```

### 2. Tracing Cognitive Flows in the Browser
Google ADK natively serves up interactive SSE (Server-Sent-Events) channels allowing you to observe the exact tool calls, multi-agent conversational handshaking, and filtering mathematics dynamically.
```bash
adk web moe_coordinator
```

### 3. Verify Deterministic Tool Accuracy
Run the Gen AI Evaluation service against the pipeline to trace trajectory outputs precisely, verify that the Yahoo Finance MCP schema holds structurally, and ensure hallucination limits are respected.
```bash
adk eval moe_coordinator research_papers_to_agents/moe_coordinator/eval/moef_eval.test.json
```

*(Note: If you wish to execute the MoE-F ecosystem in Docker mirroring the `market_data` network, see the root-level docker-compose strategy.)*
