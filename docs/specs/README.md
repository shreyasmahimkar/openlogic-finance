# Specs — intent before implementation

A **spec** is the contract for a new agent or non-trivial feature: what it does,
its inputs/outputs, tools, success criteria, and constraints. In agentic
engineering the spec (plus tests/evals) is how we communicate intent to AI
agents — write it **before** generating code (root `AGENTS.md`, workflow step 2).

## Convention

- One file per capability: `NNNN-short-name.md` (e.g. `0001-moe-f-coordinator.md`).
- Start from [`TEMPLATE.md`](TEMPLATE.md).
- Specs are reviewed in PRs and versioned with the code, like any other artifact.
- A spec's **success criteria** become the agent's eval rubric (Phase 2).

## Index

| Spec | Status | Implementation |
|---|---|---|
| MoE-F Level-3 Coordinator | Implemented (Phase 0 reconstruction) | [`model_library/agentic_ai/docs/moef_agents_plan.md`](../../model_library/agentic_ai/docs/moef_agents_plan.md) → `model_library/agentic_ai/moe_coordinator/` |

> The MoE-F design spec currently lives next to the agent
> (`model_library/agentic_ai/docs/moef_agents_plan.md`). New specs start here in
> `docs/specs/`.
