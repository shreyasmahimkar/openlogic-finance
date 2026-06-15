"""Equity Research Assistant — grounded RAG agent (vertical slice over the 6 boxes).

Wires the boxes together: ingestion/index (Box 1, `data_prep.rag`) → retriever
(Box 2, `model_library.retrieval`) → grounding governance (Box 4,
`risk_management.governance`) → a Gemini ADK agent. The model comes from the
central registry. ADK is optional at import; the retrieval core works without it.

Run: `adk run agentic_workflows/equity_research` (needs GEMINI_API_KEY).
P2 will add the return/regime model as a second tool on this agent.
"""

from data_prep.rag.indexing import build_index
from model_library.agentic_ai.model_registry import get_model
from model_library.retrieval.retriever import Retriever
from risk_management.governance.grounding import GROUNDING_INSTRUCTION

from .corpus import SAMPLE_TRANSCRIPTS

# Build the index + retriever once at import.
_store, _embedder = build_index(SAMPLE_TRANSCRIPTS)
_retriever = Retriever(_store, _embedder)


def retrieve_context(query: str) -> str:
    """Retrieve the most relevant cited earnings-call passages for a question.

    Args:
        query: the analyst's natural-language question about the company.

    Returns:
        A numbered, source-cited context block (or NO_CONTEXT_FOUND).
    """
    return _retriever.retrieve(query, k=4).answer_context


def build_equity_research_agent():
    """Build the ADK grounded equity-research agent (requires google-adk)."""
    from google.adk.agents import LlmAgent
    from google.adk.tools import FunctionTool

    return LlmAgent(
        name="equity_research_agent",
        model=get_model("orchestration"),
        instruction=GROUNDING_INSTRUCTION,
        tools=[FunctionTool(func=retrieve_context)],
        output_key="research_note",
    )


try:  # pragma: no cover - import-time convenience for `adk run`
    root_agent = build_equity_research_agent()
except Exception:
    root_agent = None
