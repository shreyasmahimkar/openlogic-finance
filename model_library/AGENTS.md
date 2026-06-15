# AGENTS.md — model_library (Box 2: Models & Agents)

Root rules: [`/AGENTS.md`](../AGENTS.md).

Translates financial/ML research into agentic models. **This is the canonical
home for prediction math and the expert agents** — other boxes import from here.

## Public surface

- `ml_zoo/filters.py` — `stochastic_filter_update` (Wonham-Shiryaev), `robust_gibbs_aggregation` (PAC-Bayes Softmin) + their `FunctionTool` wrappers. **Canonical.**
- `ml_zoo/logistic_regression.py` — LR model.
- `technical/indicators.py` — `enrich_ohlcv_data` (MACD, Bollinger, RSI, CCI, DX, SMAs). **Canonical.**
- `technical/signals/` — e.g. `sma_crossover_signal`.
- `agentic_ai/experts.py` — `expert_llama` / `expert_gpt` / `expert_mixtral` + `moe_parallel_swarm`. **Canonical experts.**
- `agentic_ai/moe_coordinator/` — flagship MoE-F Level-3 ADK pipeline (`adk run model_library/agentic_ai/moe_coordinator`).
- `tests/` — deterministic unit tests for the math above.

## Rules

- **Experts emit a single float in `[0.0, 1.0]`** — no prose. Enforce in evals (Phase 2).
- Math here must be **deterministic and unit-tested** (it is what evals trust).
- `moe_coordinator/agent.py` imports experts/filters/indicators from this box — never re-vendors them. (`interface/cli/agent.py` currently duplicates the pipeline; Phase 3 consolidates it. Don't add a third copy.)
- Keep model ids referenced in one place per agent (central registry comes in Phase 3/8).
