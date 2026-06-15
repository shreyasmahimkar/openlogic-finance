# AGENTS.md — OpenLogic Finance

The single source of truth for any AI coding agent (and human) working in this
repo. Read this first. `CLAUDE.md` and `GEMINI.md` point here.

> OpenLogic Finance is an open-source, **multi-agent financial engineering**
> platform built on **Google ADK**. It replaces black-box trading models with
> transparent, auditable agentic workflows. This is financial software — it is
> held to **agentic-engineering** discipline (specs, tests, evals, guardrails),
> not vibe coding. See `docs/AGENTIC_ENGINEERING_SDLC_PLAN.md`.

## Stack & environment

- **Python ≥ 3.11**, dependency-managed with **uv** (`pyproject.toml` + `uv.lock`).
- **Google ADK 1.31** for all agents (`google.adk.agents`, `google.adk.tools`).
- Project env lives in `.openlogic-env/` (Python 3.11). Recreate with `uv sync`.
- Common tasks are in the `Makefile` (`make setup`, `make test`, `make run`, `make web`, `make lint`).

```bash
make setup          # uv sync — build the env from the lockfile
make test           # pytest (57 tests today)
make run            # adk run the MoE-F coordinator (interactive)
make web            # adk web UI
```

Always launch ADK from the **repo root** so absolute `model_library...` imports resolve.

## Architecture: the 6-Box model + horizontal foundation

```
data_prep (Box 1) → model_library (Box 2) → strategy_testing (Box 3) →
risk_management (Box 4) → live_paper_execution (Box 5) → interface (Box 6)
        ▲ all boxes import shared primitives from ▼
horizontal_foundation/  (config · utils · core · interpretability)
agentic_workflows/      (cross-cutting ADK orchestration: primitives · orchestrators · tools)
```

Each box has its own `AGENTS.md` with local rules and public surface. The
**dependency direction is one-way**: boxes depend on `horizontal_foundation`,
never the reverse; later boxes may depend on earlier ones, not vice versa.

## Hard rules (non-negotiable)

1. **No money movement without a human.** An agent must never place a live order,
   transfer funds, or execute a real trade autonomously. Live/paper execution is
   human-gated, and the Risk Auditor (`risk_management/portfolio/auditor.py`) can
   **veto** any trade-shaped action.
2. **Expert output contract.** Prediction experts emit **exactly one float in
   `[0.0, 1.0]`** (1.0 = strong rise, 0.5 = neutral, 0.0 = strong fall) — no prose,
   no preamble. See `docs/memory/`.
3. **Import, don't copy.** Shared math/agents/config live once in `model_library`
   and `horizontal_foundation` and are **imported**. Never vendor a second copy of
   `experts.py`, `filters.py`, `indicators.py`, etc. (This is the failure that lost
   the original `moe_coordinator` source — see the SDLC plan.)
4. **No secrets in code.** Credentials come from `.env` / environment only. Never
   hard-code API keys; never commit `.env`. The pre-commit secret scanner enforces this.
5. **`scratch/` is throwaway.** It is gitignored and must never be imported by
   production code.
6. **One canonical ADK agent per capability.** Don't fork an agent to tweak it;
   parameterize the canonical one.

## Conventions

- **ADK agent package shape:** `agent.py` (defines `root_agent`), `tools.py`
  (typed tools with docstrings stating *when* to call them), `__init__.py`
  (`from . import agent`), `evals/` (Phase 2), `README.md`.
- **Models:** keep the model id in one place per agent (moving toward a central
  registry in Phase 3/8). Today: orchestration glue uses `gemini-2.5-flash`; the
  swarm experts reference non-Gemini models (`gpt-4o`, `llama-3-8b`, `mixtral-8x7b`)
  that need LiteLLM + provider keys to run end-to-end — see `docs/memory/`.
- **Tests** mirror source under `*/tests/`; deterministic math is unit-tested,
  agent behavior is eval-tested (Phase 2).

## Workflow

1. Branch off `main` (never commit directly to `main`).
2. For a new agent or non-trivial feature, write/update a **spec** in `docs/specs/` first.
3. Write tests (and, for agents, evals) alongside the change.
4. `make lint && make test` before committing; pre-commit hooks must pass.
5. Keep this file current: **when an agent does something it shouldn't, add a rule here.**

## Pointers

- Plan & roadmap: `docs/AGENTIC_ENGINEERING_SDLC_PLAN.md`
- Specs (intent): `docs/specs/`
- Durable decisions / facts: `docs/memory/`
- Local dev guide: `LOCAL_DEVELOPMENT.md`
- Flagship agent: `model_library/agentic_ai/moe_coordinator/`
