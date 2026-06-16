# AGENTS.md — strategy_testing (Box 3: Strategy Testing)

Root rules: [`/AGENTS.md`](../AGENTS.md).

Validates strategies with a lightweight simulator and the high-fidelity
QuantConnect **LEAN** backtest engine.

## Public surface

- `backtesting/simulator.py` — lightweight in-process simulator.
- `backtesting/final_test.py` — MoE-F end-to-end backtest harness.
- `backtesting/data/generate_mock_data.py` — deterministic mock data for tests.
- `validation/report.py` — the *validate* half of the MDLC: `validate()` → `ValidationReport` (AUC/Brier/KS/CV/PSI) + a `passes_gate()` sign-off. See `docs/EQUITY_RESEARCH.md`.
- `validation/rag_eval.py` + `llm_judge.py` — **scored** RAG eval: context_recall@k + LM-as-judge groundedness (Gemini judge, heuristic fallback) + `passes_gate()`. See `docs/RAG.md`.
- `validation/agent_contracts.py` — `analyze_pipeline()`: static handoff-contract analysis (dangling `{placeholder}` references + orphan `output_key`s) for any ADK agent tree. See `docs/AGENT_CONTRACT_TESTING.md`.
- `lean_engine/` — LEAN CLI bridge (`agent.py`, `lean_bridge.py`, `lean_tool.py`) + strategy projects.

## Rules

- LEAN is heavy (Docker + multi-GB market data). Install on demand: `uv sync --extra lean`.
- `lean_workspace/`, `backtests/`, and `.lean/` are runtime/output — gitignored. **Source of truth for strategy code is the project folders under `lean_engine/`.**
- Strategy logic (signals, models) lives once in **`model_library`**. LEAN runs
  each project self-contained in a container, so the project copies of
  `logistic_regression.py` / `sma_crossover_signal.py` are **generated** from
  `model_library` — regenerate with `make sync-lean`, never hand-edit them.
  `test_lean_sync.py` fails CI if they drift.
- `final_test.py` imports the plotter from the shared `model_library/agentic_ai/coordinator.py` (Phase 3 moved it off `interface/cli`).
