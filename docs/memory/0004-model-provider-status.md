# 0004 — Model routing & provider status

**Status:** Active (registry landed in Phase 3)

All model choice goes through the central registry
`model_library/agentic_ai/model_registry.py` (`get_model(role)`). It maps a
**logical role** → a concrete model, so routing is config, not scattered code.

| Role | Default | Runs on a Google account alone? |
|---|---|---|
| `orchestration` (extractor, feature, SBERT, synthesizer, plotter) | `gemini-2.5-flash` | ✅ |
| `expert_technical` | `gemini-2.5-flash` | ✅ |
| `expert_fundamental` | `gemini-2.5-flash` | ✅ |
| `expert_contrarian` | `gemini-2.5-flash` | ✅ |

**Defaults route everything to Gemini**, so a full MoE-F run completes with only
`GEMINI_API_KEY` (local) or Vertex (deploy) — no OpenAI/Groq keys. This closed
the earlier gap where the swarm died at `ParallelFilterPhase`.

**Overrides:**
- Per role: `OPENLOGIC_MODEL_<ROLE>` (e.g. `OPENLOGIC_MODEL_EXPERT_FUNDAMENTAL=gpt-4o`).
- Research-faithful heterogeneous swarm (Llama/GPT/Mixtral): `OPENLOGIC_HETEROGENEOUS_EXPERTS=1`.
  Non-Gemini ids are wrapped in ADK's `LiteLlm` and need `litellm` + provider keys.

**How to apply:** never hard-code a model in an agent — call `get_model(role)`.
Cost-based routing (frontier models for hard tasks, cheap models for deterministic
ones) is a future extension of this same registry.
