"""Equity Research Assistant — orchestrating RAG ⇄ the return model with HITL (P3).

The agent now wields three tools and a governance callback, tying the boxes into
one workflow:
- `retrieve_context` (Box 1/2, RAG) — what management *said*.
- `predict_regime` (Box 2, model) — what the *numbers* say.
- `publish_recommendation` (consequential) — gated by a **human-in-the-loop**
  approval callback (Box 4) until an analyst signs off.

Grounding governance still applies: cite transcript passages or abstain.
Run: `adk run agentic_workflows/equity_research` (needs GEMINI_API_KEY).
"""

from model_library.agentic_ai.model_registry import get_model
from risk_management.governance.approval import make_research_approval_callback
from risk_management.governance.grounding import GROUNDING_INSTRUCTION

from . import tools
from .tools import retrieve_context  # re-exported for convenience/tests  # noqa: F401

INSTRUCTION = (
    GROUNDING_INSTRUCTION + "\n\nYou also have two more tools:\n"
    "- predict_regime(ticker): the quantitative model's regime signal — present it "
    "as the *model view*, separate from management's words.\n"
    "- publish_recommendation(thesis, rating): publishes a BUY/HOLD/SELL note. It is "
    "consequential and requires human approval; if you receive PENDING_HUMAN_APPROVAL, "
    "tell the user the note is awaiting analyst sign-off — do NOT claim it was published.\n"
    "For a research call: retrieve the relevant transcript passages, get the model "
    "regime, then synthesize a concise, cited thesis with a rating. Distinguish the "
    "qualitative (transcript) and quantitative (model) evidence."
)


def build_equity_research_agent():
    """Build the ADK research agent (requires google-adk)."""
    from google.adk.agents import LlmAgent
    from google.adk.tools import FunctionTool

    return LlmAgent(
        name="equity_research_agent",
        model=get_model("orchestration"),
        instruction=INSTRUCTION,
        tools=[
            FunctionTool(func=tools.retrieve_context),
            FunctionTool(func=tools.predict_regime),
            FunctionTool(func=tools.publish_recommendation),
        ],
        before_tool_callback=make_research_approval_callback(),
        output_key="research_note",
    )


try:  # pragma: no cover - import-time convenience for `adk run`
    root_agent = build_equity_research_agent()
except Exception:
    root_agent = None
