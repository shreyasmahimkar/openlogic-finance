# 0004 — Multi-provider model status

**Status:** Active (gap to close in Phase 3/8)

Models referenced today, and what actually runs:

| Agent(s) | Model id | Runs with current setup? |
|---|---|---|
| Orchestration glue (extractor, feature, SBERT, synthesizer, plotter) | `gemini-2.5-flash` | ✅ `GEMINI_API_KEY` in `.env` |
| `expert_llama` | `llama-3-8b` | ❌ needs LiteLLM + provider key |
| `expert_gpt` | `gpt-4o` | ❌ needs LiteLLM + OpenAI key |
| `expert_mixtral` | `mixtral-8x7b` | ❌ needs LiteLLM + provider key |

**Consequence:** a full MoE-F run reaches `ParallelFilterPhase` and then fails at
the experts until the non-Gemini providers are wired. ADK treats a bare model
string as Gemini; non-Gemini models must be wrapped with `LiteLlm`.

**How to apply:** don't assume end-to-end works out of the box. Closing this is
the **central model registry + routing** item (Phase 3/8): one place that maps a
logical role → concrete model + provider wrapper, enabling cost-based routing
(frontier models for hard tasks, cheap models for deterministic ones).
