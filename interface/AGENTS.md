# AGENTS.md — interface (Box 6: Interface)

Root rules: [`/AGENTS.md`](../AGENTS.md).

How humans observe and drive the system: CLI agent consoles, Streamlit
dashboards, and Jupyter research templates.

## Public surface

- `cli/agent.py` — CLI orchestration. A **thin importer** over the shared
  builder `model_library/agentic_ai/coordinator.py` (Phase 3 consolidated the old
  duplicate). Keep it thin; don't re-inline pipeline logic here.
- `streamlit/app.py` — MoE-F monitoring dashboard (`make web-dash`).
- `streamlit/equity_research_app.py` — the Equity Research Assistant console (RAG evidence + model regime + HITL approval + audit trail; `make research-console`). See `docs/EQUITY_RESEARCH.md`.
- `notebooks/` — research templates (`research_template.ipynb`, etc.).

## Rules

- This box **presents** results; it must not own model or risk logic. Import the
  pipeline, experts, and aggregation from `model_library` — don't define them here.
- Keep heavy deps (streamlit) behind the `interface` optional group.
- Notebooks are for research/exploration; promote anything reusable into the
  appropriate box rather than leaving logic in a notebook.
