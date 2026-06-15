# AGENTS.md — horizontal_foundation (Infrastructure)

Root rules: [`/AGENTS.md`](../AGENTS.md).

The shared substrate **every box imports from**. This package has **no upstream
dependencies on any box** — the dependency arrow only points *into* here.

## Public surface

- `config/system_config.py` — `SystemConfig`: workspace paths, default ticker/period, cache TTL. The single source of paths.
- `utils/logging_helpers.py` — logging setup helpers.
- `core/base_connector.py` — base class for data connectors.
- `observability.py` — `setup_tracing()` / `setup_from_env()`: local OpenTelemetry tracing for agent runs (`OPENLOGIC_TRACING=1` enables it; the Vertex deploy traces in the cloud). ADK emits spans once the provider is set.
- `interpretability/explain_engine.py` — `ExplanationEngine`: multi-tier human-readable explanations (beginner → academic). The human-readable layer on top of the machine traces.

## Rules

- This is the **canonical home** for cross-cutting primitives. If two boxes need
  the same helper, it belongs here — imported, never copied.
- Keep it dependency-light and side-effect-free on import (it is loaded everywhere).
- Never import from `data_prep`, `model_library`, etc. (would create a cycle).
- Changes here ripple repo-wide — cover them with `tests/`.
