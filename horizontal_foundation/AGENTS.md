# AGENTS.md — horizontal_foundation (Infrastructure)

Root rules: [`/AGENTS.md`](../AGENTS.md).

The shared substrate **every box imports from**. This package has **no upstream
dependencies on any box** — the dependency arrow only points *into* here.

## Public surface

- `config/system_config.py` — `SystemConfig`: workspace paths, default ticker/period, cache TTL. The single source of paths.
- `utils/logging_helpers.py` — logging setup helpers.
- `core/base_connector.py` — base class for data connectors.
- `interpretability/explain_engine.py` — `ExplanationEngine`: multi-tier human-readable explanations (beginner → academic). This is the observability/explainability surface other boxes wrap.

## Rules

- This is the **canonical home** for cross-cutting primitives. If two boxes need
  the same helper, it belongs here — imported, never copied.
- Keep it dependency-light and side-effect-free on import (it is loaded everywhere).
- Never import from `data_prep`, `model_library`, etc. (would create a cycle).
- Changes here ripple repo-wide — cover them with `tests/`.
