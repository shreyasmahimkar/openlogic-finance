# risk_management/agents (planned)

LLM-driven risk agents (Box 4) — e.g. a narrative risk reviewer that explains
*why* an exposure is unsafe. The deterministic veto already lives in
`portfolio/guardrail.py` (an ADK `before_tool_callback`); agentic risk reasoning
that wraps it belongs here.

**Status:** scaffolding. See `docs/BACKLOG.md`.
