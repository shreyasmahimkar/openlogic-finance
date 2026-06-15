# 0003 — Import, don't copy (no vendoring)

**Status:** Active

Shared math, agents, and config exist **once** and are imported:

- Prediction math → `model_library/ml_zoo/filters.py`
- Indicators → `model_library/technical/indicators.py`
- Experts / swarm → `model_library/agentic_ai/experts.py`
- Paths / config → `horizontal_foundation/config/system_config.py`

**Why:** the original `moe_coordinator` package vendored its own copies of
`experts.py` / `filters.py` / `indicators.py`. The source was later lost (only
`.pyc` survived, never committed), and the vendored copies drifted from the
canonical ones. Duplication is how this codebase lost working code.

**How to apply:** never create a second copy of a shared module. Known remaining
duplications to *consolidate, not extend* (Phase 3):
- `interface/cli/agent.py` duplicates the MoE-F pipeline in `moe_coordinator/agent.py`.
- LEAN strategy projects copy `logistic_regression.py` / `sma_crossover_signal.py`.

See [[0001-six-box-architecture]].
