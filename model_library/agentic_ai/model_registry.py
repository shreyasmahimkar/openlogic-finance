"""Central model registry — the one place that maps a logical *role* to a
concrete model, so model choice is config, not code scattered across agents.

This is the "intelligent model routing" lever from the SDLC plan (Phase 3/8) and
it closes the gap documented in `docs/memory/0004-model-provider-status.md`.

Defaults route **everything to Gemini** so the full MoE-F pipeline runs on a
Google account alone (`GEMINI_API_KEY` locally, or Vertex on deploy) — no
OpenAI/Groq keys required. Two override paths:

- Per role: set `OPENLOGIC_MODEL_<ROLE>` (e.g. `OPENLOGIC_MODEL_EXPERT_TECHNICAL=gpt-4o`).
- Research-faithful heterogeneous swarm: set `OPENLOGIC_HETEROGENEOUS_EXPERTS=1`
  to use Llama/GPT/Mixtral for the three experts (needs LiteLLM + provider keys).

Non-Gemini ids are wrapped in ADK's `LiteLlm` so ADK can route them.
"""

import os

# Logical roles used across the agentic_ai package.
ROLES = (
    "orchestration",       # ingestion / synthesis / plotting glue
    "expert_technical",    # momentum / technical expert
    "expert_fundamental",  # macro / fundamental expert
    "expert_contrarian",   # mean-reversion / contrarian expert
)

# Runs on a Google account with no extra provider keys.
_DEFAULTS = {
    "orchestration": "gemini-2.5-flash",
    "expert_technical": "gemini-2.5-flash",
    "expert_fundamental": "gemini-2.5-flash",
    "expert_contrarian": "gemini-2.5-flash",
}

# Research-faithful heterogeneous experts (opt-in; needs LiteLLM + provider keys).
_HETEROGENEOUS = {
    "expert_technical": "llama-3-8b",
    "expert_fundamental": "gpt-4o",
    "expert_contrarian": "mixtral-8x7b",
}


def _resolve(model_id: str):
    """Return a value ADK accepts as a model: a bare Gemini id, or a LiteLlm wrapper."""
    if model_id.startswith("gemini"):
        return model_id
    try:
        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(model=model_id)
    except Exception:
        # litellm not installed — return the raw id so the missing-provider
        # setup surfaces explicitly at call time rather than being hidden here.
        return model_id


def get_model(role: str):
    """Resolve the model for a logical role, honoring env overrides."""
    if role not in ROLES:
        raise ValueError(f"Unknown model role: {role!r}. Known roles: {ROLES}")

    override = os.environ.get(f"OPENLOGIC_MODEL_{role.upper()}")
    if override:
        return _resolve(override)

    if os.environ.get("OPENLOGIC_HETEROGENEOUS_EXPERTS") == "1" and role in _HETEROGENEOUS:
        return _resolve(_HETEROGENEOUS[role])

    return _resolve(_DEFAULTS[role])
