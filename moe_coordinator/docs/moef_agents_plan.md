# Level 3 Multi-Agent Architecture Plan for MoE-F 
*(Fully Integrated within the `moe_coordinator` Ecosystem)*

This architecture plan updates the initial agent planning document using the advanced guidelines dictated by the **"Stochastic Filtering in Level 3 Multi-Agent Architectures"** analysis. To ensure absolute compliance with enterprise deployment standards, the entire MoE-F pipeline is reconstructed using the **Google Agent Development Kit (ADK)** capabilities, replacing generic agent structures with programmatic ADK primitives (`LlmAgent`, `ParallelAgent`, `SequentialAgent`, `AgentTool`, and `MCPToolset`) natively inside the `moe_coordinator`.

---

## 1. Phase 1: Data Ingestion Pipeline (`SequentialAgent`)
*Replaces the isolated 1st Agent (Technical Indicators) and 2nd Agent (News Ingestion).*

To align with the ADK standards and ingest 10 years of SPY data organically, the data fetching pipeline is executed chronologically as a Level-3 sequence pipeline:

### A. MarketDataExtractor (`LlmAgent`)
**Objective**: Leverage the `MCPToolset` using the Yahoo Finance Model Context Protocol (`yahoo-finance-mcp`).
- **Data Pull**: Invokes the `get_historical_stock_prices` MCP tool to fetch trailing 10 years of SPY (OHLCV) and `get_yahoo_finance_news` for the historical back-dated news.
- **Indicator Attachment Constraints**: Maps the technical indicators defined in the NIFTY-LM datasets onto the context window.
   - **MACD**: EMA(12) - EMA(26)
   - **Bollinger Bands**: SMA(20) ± [2 × σ(20)]
   - **30-Day RSI & CCI**
   - **30-Day DX**: 100 × [(+DI) - (-DI)] / [(+DI) + (-DI)]
   - **30-Day & 60-Day SMAs**.

### B. SBERT_SemanticFilter (`LlmAgent`)
**Objective**: Cleanse the incoming text data fed by the Extractor.
- **Rule**: Applies semantic similarity via the "all-MiniLM-L6-v2" models. News headlines returning a similarity score (or tf-idf) strictly below the 0.2 threshold against baseline financial definitions are discarded, directly mirroring Appendix C constraints.

---

## 2. Phase 2: Mathematical Stochastic Filtering (`AgentTool`)
*Replaces the previous standalone HMM evaluation agent recommendation.*

Instead of building a separate microservice to maintain weights, the **Wohman-Shiryaev Filter** and localized math from the research paper is encoded as a strictly deterministic Python `AgentTool` (e.g. `stochastic_filter_update`).
- Calculates continuous Euclidean loss (or cross-entropy).
- Manages the local Bayesian belief states ($\pi$), calculating drift against the overall Global Q-Matrix.
- Passes the updated simplex probabilities natively back into the ADK `SessionState` context.

---

## 3. Phase 3: The MoE-F Swarm (`ParallelAgent`)
*The computational core.*

By instantiating an ADK `ParallelAgent`, the architecture performs a "Parallel Fan-Out" execution.
- **Setup**: Sub-agents like `expert_llama`, `expert_gpt`, and `expert_mixtral` are initialized as individual `LlmAgent`s.
- **Execution**: The 7-Day windowed historical string arrays built by the SBERT filter are routed simultaneously to the parallel experts. The `filter_tool` is attached to these agents sequentially so their localized belief matrices are generated upon prediction.

---

## 4. Phase 4: Robust Aggregation & Visualization (`LlmAgent` + AgentTool)
*Replaces the standalone 3rd Agent Plotter Placeholder.*

### A. Coordinator Synthesizer
- Uses the `robust_gibbs_aggregation` function bound to a `SynthesizerAgent`.
- Applies the outer-loop Q-Matrix bi-level update by perturbing the stochastic matrices via the Softmin Gibbs measure ($1 - \alpha + \alpha I_N$).

### B. Visualization Harness (The Plotter)
As detailed in the document, to replicate Figure 1 smoothly:
- The synthesizer takes the raw prediction sequence and applies a Pandas rolling window: `df['MoE_F_Prediction'].rolling(window=7).mean()`.
- Generates a Plotly/Matplotlib image charting the True Market Trajectory (black line) against the MoE-F Filtered Trajectory (green dashed line), overlaying the parallel swarm experts' nodes as scatter clusters.

---

## 5. Master Pipeline Deployment
All phases are encapsulated into a master ADK pipeline object `moef_level_3_system = SequentialAgent(...)`.

**Vertex AI Ecosystem Integration:**
Rather than managing a custom Docker-heavy orchestrator, the final assembled architecture is packaged into an ADK App and deployed programmatically using GCP `Vertex AI Agent Engine`. This enables native telemetry, HMM state persistence, and highly scaled concurrent prediction across 10 years of rolling context without latency bottlenecks.
