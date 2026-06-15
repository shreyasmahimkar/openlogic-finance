# AGENTS.md — strategy_testing (Box 3: Strategy Testing)

Root rules: [`/AGENTS.md`](../AGENTS.md).

Validates strategies with a lightweight simulator and the high-fidelity
QuantConnect **LEAN** backtest engine.

## Public surface

- `backtesting/simulator.py` — lightweight in-process simulator.
- `backtesting/final_test.py` — MoE-F end-to-end backtest harness.
- `backtesting/data/generate_mock_data.py` — deterministic mock data for tests.
- `lean_engine/` — LEAN CLI bridge (`agent.py`, `lean_bridge.py`, `lean_tool.py`) + strategy projects.

## Rules

- LEAN is heavy (Docker + multi-GB market data). Install on demand: `uv sync --extra lean`.
- `lean_workspace/`, `backtests/`, and `.lean/` are runtime/output — gitignored. **Source of truth for strategy code is the project folders under `lean_engine/`.**
- Strategy logic (signals, models) is **imported from `model_library`** — the LEAN project copies (`logistic_regression.py`, `sma_crossover_signal.py`) are a known duplication slated for consolidation; don't deepen it.
- `final_test.py` imports the plotter from the shared `model_library/agentic_ai/coordinator.py` (Phase 3 moved it off `interface/cli`).
