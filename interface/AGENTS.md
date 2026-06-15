# AGENTS.md — interface (Box 6: Interface)

Root rules: [`/AGENTS.md`](../AGENTS.md).

How humans observe and drive the system: CLI agent consoles, Streamlit
dashboards, and Jupyter research templates.

## Public surface

- `cli/agent.py` — CLI orchestration. **Note:** currently duplicates the MoE-F
  pipeline assembled in `model_library/agentic_ai/moe_coordinator/agent.py`;
  Phase 3 factors both into one shared builder. Don't extend the duplication.
- `streamlit/app.py` — monitoring dashboard (`make web-dash` / `streamlit run interface/streamlit/app.py`).
- `notebooks/` — research templates (`research_template.ipynb`, etc.).

## Rules

- This box **presents** results; it must not own model or risk logic. Import the
  pipeline, experts, and aggregation from `model_library` — don't define them here.
- Keep heavy deps (streamlit) behind the `interface` optional group.
- Notebooks are for research/exploration; promote anything reusable into the
  appropriate box rather than leaving logic in a notebook.
