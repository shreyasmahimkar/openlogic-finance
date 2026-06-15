"""The MoE-F expert swarm.

Exposed as **factories** rather than module-level singletons: an ADK agent can
only have one parent, so the coordinator builder must be able to assemble a fresh,
independent swarm per pipeline instance (e.g. the ADK app and the CLI twin).

Models come from the central registry (`model_registry.py`): Gemini by default so
the swarm runs on a Google account alone; set `OPENLOGIC_HETEROGENEOUS_EXPERTS=1`
for the research-faithful Llama/GPT/Mixtral mix (needs LiteLLM + provider keys).
"""

from google.adk.agents import LlmAgent, ParallelAgent

from model_library.agentic_ai.model_registry import get_model
from model_library.ml_zoo.filters import stochastic_filter_update_tool

_FLOAT_CONTRACT = (
    "Output your prediction EXACTLY as a single float between 0.0 and 1.0, where "
    "1.0=Strong Rise, 0.5=Neutral, 0.0=Strong Fall. No text, no preamble, just the float value."
)


def build_experts() -> list[LlmAgent]:
    """Build the three fresh expert agents (technical / fundamental / contrarian)."""
    expert_technical = LlmAgent(
        name="Llama_Expert",
        model=get_model("expert_technical"),
        instruction=(
            "You are a Technical Analyst Expert evaluating SPY.\n"
            "Given standard OHLCV prices and moving averages for the past 10 days in "
            "the context, predict if the price will Rise, Fall, or remain Neutral tomorrow.\n"
            "Use {enriched_market_data} and {filtered_news_context} for context.\n" + _FLOAT_CONTRACT
        ),
        tools=[stochastic_filter_update_tool],
        output_key="pred_llama",
    )
    expert_fundamental = LlmAgent(
        name="GPT4o_Expert",
        model=get_model("expert_fundamental"),
        instruction=(
            "You are a Fundamental Macroeconomic Analyst Expert evaluating the broader "
            "stock market (SPY).\n"
            "Predict a macro-level Rise, Fall, or Neutral move tomorrow. Ignore short-term "
            "technical noise; focus on structural gravity and long-horizon price memory.\n"
            "Use {enriched_market_data} and {filtered_news_context} for context.\n" + _FLOAT_CONTRACT
        ),
        tools=[stochastic_filter_update_tool],
        output_key="pred_gpt",
    )
    expert_contrarian = LlmAgent(
        name="Mixtral_Expert",
        model=get_model("expert_contrarian"),
        instruction=(
            "You are a High-Frequency Mean-Reverting Analyst Expert.\n"
            "Look at the past 10 days of price context. If it rallied hard, bet that it "
            "Falls. If it dumped, bet that it Rises. You believe markets are rubber bands.\n"
            "Use {enriched_market_data} and {filtered_news_context} for context.\n" + _FLOAT_CONTRACT
        ),
        tools=[stochastic_filter_update_tool],
        output_key="pred_mixtral",
    )
    return [expert_technical, expert_fundamental, expert_contrarian]


def build_moe_parallel_swarm() -> ParallelAgent:
    """Build a fresh parallel fan-out swarm of the three experts."""
    return ParallelAgent(name="ParallelFilterPhase", sub_agents=build_experts())
