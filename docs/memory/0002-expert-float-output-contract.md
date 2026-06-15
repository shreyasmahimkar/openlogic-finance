# 0002 — Expert float-output contract

**Status:** Active

Every prediction **expert** agent (`model_library/agentic_ai/experts.py`:
`expert_llama`, `expert_gpt`, `expert_mixtral`) must output **exactly one float in
`[0.0, 1.0]`** and nothing else:

- `1.0` = strong rise, `0.5` = neutral, `0.0` = strong fall.
- No prose, no preamble, no units — just the number.

**Why:** the downstream stochastic filter and Gibbs aggregation
(`model_library/ml_zoo/filters.py`) consume these as numeric beliefs. Any prose
breaks parsing and corrupts the simplex math.

**How to apply:** keep the "single float, no text" instruction in every expert's
prompt, and verify it with an output-format eval (Phase 2) — it cannot be checked
by a static pre-commit hook because it is runtime LLM behavior.
