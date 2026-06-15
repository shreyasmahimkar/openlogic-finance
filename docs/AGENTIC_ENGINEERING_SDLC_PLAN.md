# From Vibe-Coded to Agentic Engineering: An SDLC Plan for OpenLogic Finance

**Status:** Draft for review (no implementation yet)
**Author:** Engineering
**Date:** 2026-06-15
**Source framework:** *The New SDLC With Vibe Coding* (Osmani, Saboo, Kartakis — May 2026) — the "factory model", "harness engineering", and "context engineering" mental models.
**Scope:** Turn `openlogic-finance` from an ad-hoc, partly-ADK codebase into a *purely agentic* repository governed by a deliberate harness, verified by tests + evals, and deployed through Google ADK / Vertex AI Agent Engine.

> **Read this first.** This is a planning document. It proposes *what to build and in what order*. Nothing in here is implemented yet. Each phase below ends with a concrete, reviewable deliverable so we can stop, inspect, and course-correct between milestones.

---

## 1. Executive summary

The whitepaper's central claim is that **an agent = model + harness**, and that the difference between "vibe coding" and "agentic engineering" is not the tools you use but **how deliberately you configure the harness** — instructions, tools, sandboxes, orchestration, guardrails, and observability — around the model. Today OpenLogic Finance already *uses* Google ADK in several places (`google.adk.agents.LlmAgent`, `ParallelAgent`, `Agent`), but it has **almost none of the harness**. It is, in the paper's terms, a sophisticated *vibe-coded* codebase: real agents, no scaffolding to keep them correct, reproducible, or safe.

This plan does three things:

1. **Diagnoses** the concrete vibe-coding symptoms already present in this repo (Section 3).
2. **Defines the target "factory"** — the harness, the verification layer (tests *and* evals), and the model-routing economics — mapped onto our existing 6-Box architecture (Sections 4–7).
3. **Lays out how Google ADK becomes the agentic substrate** that makes this seamless: a single `adk web` / `adk eval` / Agent Engine lifecycle from prototype to production (Section 6), with a phased roadmap (Section 9).

The north star: **a developer (or a coding agent) should be able to go from a research paper → spec → ADK agent → eval-gated PR → Vertex deployment without leaving the terminal, and without any step depending on undocumented tribal knowledge or a lost `.py` file.**

---

## 2. Where we are on the spectrum

Using Table 1 from the whitepaper as a yardstick:

| Dimension | Vibe Coding | **OpenLogic today** | Agentic Engineering (target) |
|---|---|---|---|
| Intent specification | Casual prompts | **Prose READMEs + a few plan `.md` docs** | Formal specs, architecture docs, memory files |
| Verification | "Does it seem to work?" | **3 unit tests, 0 evals** | Automated test suites, CI/CD gates, LM judges |
| Codebase understanding | May not read code | **Inconsistent; source for `moe_coordinator` is lost** | Comprehensive review of architecture |
| Error handling | Paste errors back | **Manual** | Agents self-diagnose within bounds |
| Scope | Prototypes | **Production aspirations, prototype discipline** | Production systems, team-scale |
| Risk profile | High | **High (financial code, no guardrails)** | Low; systematic verification |

We are roughly at **"Structured AI-Assisted Coding"** for *some* files and **pure vibe coding** for others. The goal is to move the whole repo to the right-hand column — *especially* because this is **financial software**, where the paper is explicit: "A production API handling financial transactions demands agentic engineering."

---

## 3. Current-state assessment (evidence in this repo)

These are the specific symptoms this plan must fix. Each is a real observation from the tree.

### 3.1 No harness at all
- **No `AGENTS.md` / `CLAUDE.md` / `GEMINI.md`** anywhere in the repo. There is no static context defining who the agents are, the 6-Box rules, the float-output contract, or hard constraints. Every coding-agent session starts from a blank slate and re-derives conventions — the paper's definition of high-OpEx vibe coding.
- **No skills, no MCP manifest, no hooks, no observability config.** The harness surface area the paper enumerates (instructions, tools, sandboxes, orchestration, guardrails, observability) is essentially empty.

### 3.2 Lost source — the canonical vibe-coding failure
- The original `research_papers_to_agents/moe_coordinator/` directory contained **only `__pycache__/*.pyc` and `.adk/session.db`** — `agent.py`, `experts.py`, `filters.py`, `indicators.py`, `final_test.py`, `prismtrace_client.py` existed **as compiled bytecode only; the source was gone and was never committed.** This is exactly the "spaghetti you must reverse-engineer six months later" the paper warns about — except here it was worse: the source didn't exist to reverse-engineer. *(Phase 0: reconstructed and relocated to `model_library/agentic_ai/moe_coordinator/`; the `research_papers_to_agents/` directory has been removed.)*
- The `ADK_USAGE_GUIDE.md` documents `adk eval` commands against `model_library/agentic_ai/moe_coordinator/eval/*.test.json` files that **do not exist yet** (created in Phase 2). Documentation and reality had already diverged.

### 3.3 Duplication instead of shared foundation
- `model_library/technical/indicators.py` and `model_library/ml_zoo/filters.py` appear to have been **copied** into `moe_coordinator` (per the `.pyc` names `indicators`, `filters`).
- `logistic_regression.py` and `sma_crossover_signal.py` are **duplicated** across `strategy_testing/lean_engine/logistic_regression_project/` and `.../sma_crossover_project/`. The `agentic_workflows/` and `utility_agents/` packages that should hold shared orchestration are **empty `__init__.py` skeletons.**

### 3.4 No verification discipline
- **Three** unit tests total (`test_foundation.py`, `test_logistic_regression.py`, `test_sma_crossover_signal.py`). **Zero evals.** For LLM agents whose output is non-deterministic, the paper is unambiguous: "Without both [tests and evals], the practice is always vibe coding, regardless of how sophisticated the prompts are."

### 3.5 No reproducible environment or CI
- **No `pyproject.toml` / `requirements.txt` at root**; dependencies are implicit. Two committed-around virtualenvs (`.openlogic-env/`, `.adk/`).
- **No `.github/workflows/`** — nothing runs tests, evals, lint, or secret-scanning on a PR. The harness's "feedback loop" and "observing" phases (paper §Harness in SDLC) have no automation.

### 3.6 Scratch and secrets in the tree
- `scratch/` debug scripts (`debug_lr.py`, `inspect_*.py`) are **tracked in git** — exploration residue mixed with production code, the blurred prototype/production boundary the paper tells leaders to make explicit.
- A root `.env` exists; `*block_convey*` is gitignored. Secret-handling is informal and there is no hook to block a hard-coded credential from being committed (the paper's canonical guardrail example).

### 3.7 Model references are inconsistent / unroutable
- `model_library/agentic_ai/experts.py` hard-codes `model="llama-3-8b"`, `"gpt-4o"`, `"mixtral-8x7b"`; `data_prep/.../market_data/agent.py` uses `"gemini-2.5-flash"`. There is no central model registry, no routing policy, and no abstraction to swap models — so the "intelligent model routing" lever the paper identifies for OpEx control cannot be pulled.

---

## 4. Target state: the Factory Model for OpenLogic Finance

The paper's factory model: *the developer's primary output is not code — it's the system that produces code.* For us that system has five parts. This is the spine of the whole plan.

```
   SPEC  ──►  AGENTS (ADK)  ──►  TESTS + EVALS  ──►  FEEDBACK LOOP  ──►  GUARDRAILS
 (what to     (translate spec    (verify correctness    (route failures     (constrain to
  build)       to implementation)  — deterministic +      back to agent)      safe behavior)
                                    trajectory/LM-judge)
```

Mapped onto our existing 6-Box + horizontal architecture:

- **Specifications & context** → a new top-level harness (Section 5): `AGENTS.md`, `docs/specs/`, per-box `AGENTS.md` files, memory files.
- **Agents that implement** → ADK agents, consolidated and de-duplicated, one canonical location per capability (Section 6).
- **Tests & quality gates** → `tests/` (pytest, deterministic) + `evals/` (ADK evalsets, non-deterministic) wired into CI (Section 7).
- **Feedback loops** → ADK orchestration + CI that routes failures back; the `interpretability/explain_engine` as a first-class observability surface.
- **Guardrails** → pre-commit/CI hooks (secret-scan, the risk-auditor veto, output-format validators) + ADK callbacks.

---

## 5. The Harness — what to build (and where)

The harness is the highest-leverage investment (the paper's benchmark anecdote: +13.7 points / outside-Top-30 → Top-5 *by changing only the harness*). Concrete artifacts, in priority order:

### 5.1 Instructions & rule files (static context)
- **Root `AGENTS.md`** (≤ ~40 high-signal lines): stack, the 6-Box boundary rules, the universal expert-output contract ("single float 0.0–1.0, no prose"), hard rules (no live order execution from an agent without human sign-off; risk auditor can veto), and the dev workflow. Mirror to `CLAUDE.md`/`GEMINI.md` via symlink or a thin include so every coding agent reads the same source of truth.
- **Per-box `AGENTS.md`** in each of `data_prep/`, `model_library/`, `strategy_testing/`, `risk_management/`, `live_paper_execution/`, `interface/` — the local conventions, the public surface, and "do not duplicate X, import it from horizontal_foundation."
- **`horizontal_foundation/` as the single source of shared truth** — config, logging, base connectors, interpretability. Rule: boxes import from it; they never copy.

### 5.2 Tools (with the prose around them)
- A **tool registry / MCP manifest** describing every tool an agent may call (the Yahoo Finance MCP, SBERT semantic filter, `stochastic_filter_update`, the LEAN backtest bridge, the risk auditor). The paper stresses MCP as the open standard for tool access — standardize on it now to keep vendor optionality.
- Each tool gets a typed signature + docstring that states *when* to call it, not just *what* it does (see the good example already in `market_data/agent.py::fetch_and_explain`).

### 5.3 Sandboxes & execution environments
- A reproducible env: **`pyproject.toml` + lockfile** at root, replacing implicit deps and the two stray venvs. A devcontainer already exists (`.devcontainer/devcontainer.json`) — make it the canonical sandbox.
- LEAN backtests and live execution already run in Docker (`live_paper_execution/simulators/docker-compose.yml`); document these as the agent's sanctioned execution surfaces.

### 5.4 Guardrails & hooks (deterministic, non-negotiable)
- **Pre-commit / CI hooks:** secret scanner (block hard-coded keys — the paper's headline example), `ruff`/`black`, "no new file under `scratch/` gets imported by production code", and an **output-format validator** for expert agents (must emit a float in [0,1]).
- **The Risk Auditor as a runtime guardrail:** `risk_management/portfolio/auditor.py` becomes an ADK `before_model`/`before_tool` callback that can **veto** any trade-shaped action — encoded as harness policy, not left to agent goodwill.

### 5.5 Observability
- Standardize on **ADK's built-in tracing + OpenTelemetry** (the `ADK_USAGE_GUIDE.md` already promises this for Vertex). Every agent run emits a trace; CI surfaces token cost + latency per eval. The `horizontal_foundation/interpretability/` engine becomes the human-readable layer on top of the machine traces.

### 5.6 Memory
- A persistent project memory (long-term: what the project is; short-term: session logs) so agents don't re-derive context. ADK `SessionState` + a committed `docs/memory/` for durable facts (architecture decisions, the float contract, why crypto is supported, etc.).

---

## 6. Google ADK as the agentic substrate

This is the "make it seamless" core. ADK is already a dependency; the plan is to **go from ad-hoc ADK usage to a governed ADK lifecycle.**

### 6.1 Consolidate to canonical ADK agents
- **One agent package per capability**, no duplicates. Recover/rebuild `moe_coordinator` from `experts.py` + `filters.py` + `indicators.py` (which still exist in `model_library/`) and **import** the shared math rather than copying it. The lost `.pyc`-only source must be reconstructed *as committed source* and never allowed to drift again.
- Use ADK primitives as the paper describes them and as `moef_agents_plan.md` already specifies: `SequentialAgent` (ingestion pipeline), `ParallelAgent` (the MoE-F expert swarm — already built in `experts.py::moe_parallel_swarm`), `AgentTool` (the deterministic `stochastic_filter_update`), and `MCPToolset` (Yahoo Finance MCP).

### 6.2 Adopt the Agents-CLI lifecycle (build → eval → deploy → observe)
The whitepaper's recommended workflow (its "Where to start" #2) is to install a **set of ADK skills** for the coding agent so the *whole* lifecycle is driven from natural language in the terminal:

```bash
# One-time
uvx google-agents-cli setup          # gives the coding agent the 7 ADK lifecycle skills

# Then, in the coding agent (Claude Code / Gemini CLI / Codex):
> Build the MoE-F coordinator from the moef_agents_plan spec
> Evaluate it on evals/moe_coordinator/trajectory.evalset.json
> Deploy it to Vertex AI Agent Engine
```

Direct-drive equivalents (`adk web`, `adk eval`, `agents-cli create|playground|eval|deploy`) stay available for conductor-mode work. This is what turns "we use ADK" into "ADK is our SDLC."

### 6.3 Standard project shape per agent
Every ADK agent package converges on:
```
<agent_name>/
  AGENTS.md            # local rules + tool contract
  agent.py             # root_agent definition
  tools.py             # typed tools (MCP-wrapped where possible)
  evals/
    *.evalset.json     # trajectory + final-response evals
  README.md            # spec link + usage
```
The existing `data_prep/connectors/*/` already roughly follow this (`agent.py` + `tools.py`) — we standardize it everywhere and add the missing `evals/` + `AGENTS.md`.

### 6.4 Inter-agent communication
- **MCP** for tool access, **A2A (Agent2Agent)** for cross-agent delegation, **shared `SessionState`** for simple coordination — exactly the three mechanisms the paper names. This lets the 6 boxes talk as a multi-agent system instead of as imported Python functions.

### 6.5 Deployment
- `live_paper_execution/cloud_deploy/deploy_vertex.py` already targets Vertex AI Agent Engine. Harden it: parameterize `PROJECT_ID`/`STAGING_BUCKET` (no hard-coded constants), wire OpenTelemetry, and make deployment a CI step gated on passing evals.

---

## 7. The new SDLC, phase by phase (for this repo)

Mapping the paper's "Harness in SDLC" onto OpenLogic Finance:

| Phase | What changes here | Harness components |
|---|---|---|
| **Requirements & planning** | Research paper → spec in `docs/specs/`. `moef_agents_plan.md` is the model; every new agent starts as a spec, not a prompt. | Instructions, rule files |
| **Design & architecture** | Humans own the 6-Box boundaries and the math; AI scaffolds within them. Architecture decisions recorded in `docs/memory/`. | Rule files, ADRs |
| **Implementation** | Coding agent writes ADK agents inside the devcontainer sandbox, importing shared foundation code. | Sandboxes, tools |
| **Testing & QA** | `pytest` for deterministic math (filters, indicators, signals); **ADK evals** for agent trajectories + LM-judge on prediction quality. | Orchestration, guardrails |
| **Code review & deploy** | AI first-pass review + human review; CI gates on tests **and** evals; secret-scan hook; eval-gated Vertex deploy. | Hooks, observability |
| **Maintenance** | Traces + interpretability engine explain every run; regressions caught by the eval suite before they ship. | Observability |

### 7.1 Tests vs. evals — the non-negotiable split
- **Tests (deterministic):** `stochastic_filter_update`, indicator math (MACD/RSI/Bollinger), Gibbs aggregation, SMA crossover signal, LEAN bridge parsing. These already have a foothold (3 tests) — expand to cover all of `model_library/` and `horizontal_foundation/`.
- **Evals (non-deterministic):** ADK evalsets per the `ADK_USAGE_GUIDE.md` plan — `ingestion`, `swarm`, `trajectory` — scored with `trajectory_exact_match` for routing and **LM-judge** rubrics for prediction quality. **These files must actually exist** (today they're referenced but absent). Rubric dimensions per the paper: task success, tool-use quality, trajectory compliance, hallucination, response quality.

---

## 8. Economics & model routing

The paper frames this as CapEx vs. OpEx and names **context engineering + model routing** as the financial levers.

- **Context engineering as a financial lever:** a dense `AGENTS.md` raises first-pass success and kills the "fix-its-own-mistakes" token loop. This is the cheapest, highest-ROI item in the plan.
- **Intelligent model routing:** introduce a central **model registry** (replacing the scattered hard-coded `"gemini-2.5-flash"`, `"gpt-4o"`, `"llama-3-8b"`, `"mixtral-8x7b"`). Policy: frontier models for architecture/initial implementation/synthesis; small fast cheap models for test generation, code review, CI monitoring, and the deterministic-ish routing checks. ADK supports multi-model agents, so routing becomes config, not code changes.
- **Static vs. dynamic context budget:** keep `AGENTS.md` lean; push specialized procedural knowledge into **skills** loaded on demand (progressive disclosure) rather than stuffing every agent's system prompt.

---

## 9. Phased roadmap

Each phase is independently shippable and ends in a reviewable artifact. **No phase begins until this doc is approved.**

### Phase 0 — Stop the bleeding (½ day)
- Recover or reconstruct `moe_coordinator` source from `model_library` equivalents; commit it. Delete orphaned `.pyc`-only dirs.
- Add `pyproject.toml` + lockfile; remove committed venvs from the tree; confirm `.gitignore` covers them.
- Move `scratch/` out of the production import path (or untrack it).
- **Deliverable:** clean, reproducible tree; no lost source.

### Phase 1 — Lay the harness (1–2 days)
- Root `AGENTS.md` + per-box `AGENTS.md`; `CLAUDE.md`/`GEMINI.md` includes.
- `docs/specs/` (seed with `moef_agents_plan.md`) and `docs/memory/` (architecture + the float contract).
- Pre-commit hooks: secret-scan, ruff/black, output-format validator.
- **Deliverable:** a coding agent opening this repo reads one coherent rulebook.

### Phase 2 — Verification layer (2–3 days)
- Expand `pytest` coverage across `model_library/` + `horizontal_foundation/`.
- Create the **real** ADK evalsets (`ingestion`, `swarm`, `trajectory`) with rubrics + LM-judge.
- `.github/workflows/ci.yml`: lint → tests → evals → secret-scan on every PR.
- **Deliverable:** red/green PR gates; `adk eval` runs in CI.

### Phase 3 — Consolidate on ADK (2–4 days)
- Standardize every agent to the canonical package shape (§6.3); de-duplicate (`indicators`, `filters`, LEAN project copies) behind `horizontal_foundation`/`model_library`.
- Tool registry + MCP manifest; wrap the risk auditor as an ADK veto callback.
- Central model registry + routing policy.
- **Deliverable:** one canonical, governed ADK agent per capability.

### Phase 4 — Lifecycle & deploy (2–3 days)
- `uvx google-agents-cli setup`; document the build→eval→deploy loop in the README.
- Harden `deploy_vertex.py` (parameterized, OTel, eval-gated); make Vertex deploy a CI job.
- A2A wiring between the 6 boxes where cross-agent delegation is real.
- **Deliverable:** terminal-driven path from spec → deployed Vertex agent with observability.

### Phase 5 — Compounding loop (ongoing)
- The quality flywheel: benchmark → cluster failures → optimize prompts/tools → regression-gate → monitor production traces. Treat harness files as versioned, owned, PR-reviewed infrastructure.

---

## 10. Risks, trade-offs, and open decisions

- **CapEx up front.** This is real engineering time before new features ship. The paper's argument — and ours — is that for financial software the OpEx/maintenance/security savings dominate. We accept higher CapEx deliberately.
- **`moe_coordinator` reconstruction may be imperfect.** Bytecode is decompilable but the cleanest path is to rebuild from the still-present `model_library` sources + `moef_agents_plan.md` spec, then lock it behind evals so it can't silently rot again.
- **Multi-vendor model access** (OpenAI/Llama/Mixtral alongside Gemini) needs keys + cost controls; the model registry must make per-model spend observable.

### Decisions needed before Phase 0 (for the author/reviewer)
1. **Rule-file convention:** `AGENTS.md` as source of truth with `CLAUDE.md`/`GEMINI.md` includes — confirm, or pick a single primary.
2. **`moe_coordinator`:** reconstruct from `model_library` sources (recommended) vs. decompile the `.pyc`?
3. **Dependency manager:** `uv` (matches the paper's `uvx google-agents-cli`) vs. Poetry vs. pip-tools.
4. **Primary coding agent** the harness targets first (affects which rule-file name is canonical).
5. **Hosting target** for production agents: Vertex AI Agent Engine only, or also self-hosted ADK?

---

## 11. First concrete step

On approval, **Phase 0**: reconstruct and commit `moe_coordinator` source, add `pyproject.toml`, and untrack the stray venvs/scratch — establishing a clean, reproducible base before any harness work begins.

> *"Generation is solved. Verification, judgment, and direction are the new craft."* — this plan is about building the verification, judgment, and direction layer that OpenLogic Finance is currently missing.
