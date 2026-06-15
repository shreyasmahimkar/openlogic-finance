# AGENTS.md — model_library (Box 2: Models & Agents)

Root rules: [`/AGENTS.md`](../AGENTS.md).

Translates financial/ML research into agentic models. **This is the canonical
home for prediction math and the expert agents** — other boxes import from here.

## Public surface

- `ml_zoo/filters.py` — `stochastic_filter_update` (Wonham-Shiryaev), `robust_gibbs_aggregation` (PAC-Bayes Softmin) + their `FunctionTool` wrappers. **Canonical.**
- `ml_zoo/logistic_regression.py` — LR model.
- `technical/indicators.py` — `enrich_ohlcv_data` (MACD, Bollinger, RSI, CCI, DX, SMAs). **Canonical.**
- `technical/signals/` — e.g. `sma_crossover_signal`.
- `agentic_ai/experts.py` — `build_experts()` / `build_moe_parallel_swarm()` factories. **Canonical experts** (factories, so each pipeline gets fresh agents — ADK agents have one parent).
- `agentic_ai/model_registry.py` — `get_model(role)`: central model routing (Gemini default; env overrides). **Don't hard-code models in agents.**
- `agentic_ai/coordinator.py` — `build_moef_level_3_system(artifact_dir)`: the **single** MoE-F pipeline builder.
- `agentic_ai/moe_coordinator/` — flagship ADK app, a thin wrapper over the builder (`adk run model_library/agentic_ai/moe_coordinator`).
- `retrieval/retriever.py` — RAG retriever (query embedding + vector search + cited context) for the Equity Research Assistant. *(P2: the return/regime model lands here too.)*
- `tests/` — deterministic unit tests for the math above.

## Rules

- **Experts emit a single float in `[0.0, 1.0]`** — no prose. Enforce in evals (Phase 2).
- Math here must be **deterministic and unit-tested** (it is what evals trust).
- The pipeline is defined **once** in `coordinator.py`; `moe_coordinator/agent.py` and `interface/cli/agent.py` are thin importers. Don't re-inline it.
- All model choice goes through `model_registry.get_model(role)` — never a bare model string in an agent.
