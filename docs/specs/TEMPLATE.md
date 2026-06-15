# Spec NNNN: <Capability name>

**Status:** Draft | Approved | Implemented
**Owner:** <name>
**Box(es):** <which of the 6 boxes / horizontal_foundation>

## Problem
What are we building and why. The user/business need in one paragraph.

## Behavior
- Inputs (types, sources, session-state keys consumed).
- Outputs (types, session-state keys produced). For experts: **single float `[0.0, 1.0]`**.
- Step-by-step / agent trajectory at a high level.

## Tools & dependencies
- ADK tools / MCP servers the agent may call.
- Shared code imported from `model_library` / `horizontal_foundation` (import, don't copy).
- Models used (and routing rationale).

## Constraints & guardrails
- Hard rules from root `AGENTS.md` that apply (money movement, output contract, …).
- Risk/limits enforced.

## Success criteria (→ eval rubric)
- Task success: …
- Tool-use quality: …
- Trajectory compliance: …
- Hallucination / output-format: …

## Out of scope
What this explicitly does not do.
