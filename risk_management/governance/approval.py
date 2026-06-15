"""Human-in-the-loop approval gate (Box 4) for the equity-research agent.

A consequential action — publishing a rated research recommendation — must not go
out without a human analyst's sign-off. This is the same ADK `before_tool_callback`
pattern as the trade risk-veto (`risk_management/portfolio/guardrail.py`): the
callback intercepts consequential tools and short-circuits them with a
`PENDING_HUMAN_APPROVAL` response until `state["human_approved"]` is True.
"""

# Tools that publish/commit a recommendation and therefore require human approval.
CONSEQUENTIAL_TOOLS = frozenset({"publish_recommendation", "submit_research_note"})

STATE_APPROVED = "human_approved"


def is_consequential(tool_name: str) -> bool:
    return tool_name in CONSEQUENTIAL_TOOLS


def make_research_approval_callback():
    """Build an ADK `before_tool_callback` that gates consequential tools on approval."""

    def before_tool_callback(tool, args, tool_context):
        if not is_consequential(getattr(tool, "name", "")):
            return None  # non-consequential (retrieve/predict) — let it run

        if tool_context.state.get(STATE_APPROVED) is True:
            return None  # a human approved this session — proceed

        return {
            "status": "PENDING_HUMAN_APPROVAL",
            "blocked_tool": getattr(tool, "name", "unknown"),
            "reason": "A human analyst must approve before a recommendation is published.",
            "how_to_approve": f"set session state '{STATE_APPROVED}' = true after review.",
        }

    return before_tool_callback
