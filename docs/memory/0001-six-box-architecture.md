# 0001 — The 6-Box architecture

**Status:** Active

The codebase is organized as six sequential "boxes" over a shared horizontal
foundation, with a cross-cutting orchestration layer:

1. `data_prep` — ingestion, news, feature engineering
2. `model_library` — ML/research models + expert agents (canonical math)
3. `strategy_testing` — lightweight simulator + LEAN backtests
4. `risk_management` — risk limits + trade-vetoing auditor
5. `live_paper_execution` — connectivity, Docker sims, Vertex deploy
6. `interface` — CLI, Streamlit, notebooks

- `horizontal_foundation/` — config, utils, core, interpretability. **Everything
  imports from here; it imports from nothing.**
- `agentic_workflows/` — cross-cutting ADK orchestration (primitives, orchestrators, tools).

**Dependency direction is one-way:** boxes → `horizontal_foundation`; later boxes
may use earlier ones, never the reverse. Violating this creates import cycles and
erodes the boundaries that keep the system auditable.

Each box carries its own `AGENTS.md` with local rules and public surface.
