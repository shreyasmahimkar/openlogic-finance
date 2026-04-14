# MoE-F Coordinator Integration & Architecture Plan

This document outlines the end-to-end implementation plan to transition the MoE-F (Stochastic Filter) Coordinator from a prototype into a production-grade, stateful, and context-aware orchestration network.

## User Review Required

> [!IMPORTANT]  
> Please review the architectural breakdown below. It aligns the codebase with a unified 5-Day progression framework—spanning taxonomy, tool design, robust ADK memory engineering, quality testing, and GCP production readiness.

---

## Part I: System Architecture & Bootcamp Alignment

### Day 1: Taxonomy & Multi-Agent Architecture
We are building a **Level 3 Multi-Agent System** focusing on **Ensemble Orchestration**.
- The `data_ingestion_agent` operates as a utility (Level 1) to fetch high-fidelity OHLCV market state.
- The `moef_coordinator` operates as the Orchestrator, dynamically routing intelligence across three separate expert personas (Technician, Fundamentalist, Contrarian). 

### Day 2: Tooling & Interoperability
We will codify and implement the following specific tools for the agents:
1. `fetch_stock_data` / `plot_stock_data`: Sourced from the ingestion agent.
2. `invoke_moe_filter`: The mathematical core that recalibrates trust weights based on stochastic equations.
3. **Execution Plugins**: Parallel asynchronous pollers to ping our three Expert Sub-Agents seamlessly.
4. **Visualization Tool**: A new tool that plots the ensemble's time-series predictions against the ground truth, matching the benchmark design concept (e.g., green MoE-F line threading through regime shifts).

### Day 3: Context Engineering & Memory (Data First, AI Second)
The MoE-F algorithm requires a continuous "memory" of yesterday's predictions. We will drop superficial DB placeholders and adopt native ADK Session & Memory concepts:
- **Procedural / Working Memory**: We will leverage ADK Session states to persist the running `cumulative_expert_scores` and `last_expert_preds`. 
- **Context Engineering**: Instead of relying purely on generalized AI prompt logic, we ensure the ecosystem is *Data First*. We will extract and slice actual data snippets from the `data_ingestion` CSVs and forcefully inject them into the system prompts of every expert at runtime.

### Day 4: Agent Quality & Robust Testing
- **Backtesting & Evaluation**: We will implement evaluation scripts testing the filter against multi-year historical benchmarks to ensure the "Alpha" holds.
- **Constraints Testing**: Implement validation protocols enforcing that experts return exclusively floats (0.0 to 1.0). If they diverge, their ADK memory score will be artificially penalized via the transition matrix.
- **Ablation Testing**: Evaluate the mathematically filtered MoE against a naive equal-weighted average to prove the system works.

### Day 5: Prototype to Production (Deploying to GCP)
Once the local metrics succeed, we will wrap the orchestrator into a containerized deployment target:
- Deploy using **Google Cloud Run** running the ADK REST API backend. 
- Sidecar instances of **AgentOps** to visualize real-time cost, token utilization, and expert routing trace failures so debugging in production is seamless.

---

## Part II: Execution Phases

### Phase 1: Refining the Filtering Logic (`filters.py`)
The current filter implements Robust Aggregation but lacks the Optimal Parallel Filtering that uses the Wonham-Shiryaev equations to update expert reliability based on continuous innovations.
- **Implement the Innovations Process:** Update the filter to calculate the "innovations" ($\Delta W$), which represents the difference between the observed loss and the expected average loss, normalized by the gradient of the loss function.
- **Add Stochastic Drift and Diffusion:** Modify `update_filter` to include the drift ($Q^T \pi$) and diffusion ($\pi(A-\bar{A})/B$) terms. This ensures that the expert rankings ($\pi$) evolve as a system of stochastic differential equations (SDEs) rather than just a simple running average.
- **Transition to Q-Matrix Dynamics:** Implement the bi-level optimization where the transition intensity matrix ($Q$) is updated using the matrix logarithm of the row-stochastic matrix ($P$).

### Phase 2: Stateful Persistence & Real-World Data (`agent.py`)
- **Implement Session / ADK Memory Backend for:**
  - `last_expert_preds`: The float predictions ([0.0,1.0]) made by each expert in the previous session.
  - `cumulative_expert_scores`: The running loss totals ($s_n$) used to calculate Gibbs-aggregation weights.
  - `current_q_matrix`: The $N \times N$ intensity matrix governing regime shifts.
- **Dynamic Ground-Truth Calculation:** Load the OHLCV CSV from the data ingestion agent via `pandas`. Calculate the actual movement label by comparing the two most recent closing prices. A change $>0.5\%$ is a Rise (1.0), $<-0.5\%$ is a Fall (0.0), and anything else is Neutral (0.5).

### Phase 3: Orchestrating the Expert Ensemble (`experts.py`)
- **Context Injection:** Update the root agent to slice the last 10–30 days of the SPY CSV and safely inject it into the prompt of each expert.
- **Parallel Execution:** Use `Agent.run_async()` to poll the three Gemini-2.5-flash experts simultaneously.
- **Constraint Enforcement:** Enforce float boundaries. If an expert hallucination occurs, assign it a maximum loss for that time step.

### Phase 4: Output Rendering & Visualization
- **Data Export:** Export the step-by-step predictions and updated $\pi_n$ probabilities as an orchestrated `.csv` file.
- **Graphical Regimes:** Develop a visualization hook using Python (e.g. Matplotlib/Seaborn) to mirror the provided concept art. Plot the market predictions over time, overlaying Ground Truth vs MoE-F, and visually map out Mixed, Neutral, and Bearish regimes.

## Open Questions

- What specific metric threshold for evaluation should we use on Day 4? E.g., >80% Rouge or simply lower Absolute Error?
