import os
import sys
from google.adk.agents import LlmAgent, ParallelAgent
from .filters import stochastic_filter_update_tool

# Ensure root path is in PYTHONPATH for importing prismtrace
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from prismtrace import PRISMtraceADKAdapter

expert_adapter = PRISMtraceADKAdapter(
    api_key=os.environ.get("PRISMTRACE_API_KEY", ""),
    project_id="3a06f38f-3103-4c38-a38d-b6e3c68414d3",
    agent_name="my-adk-expert",
)

# Expert 1: Llama (The Technician / Momentum)
expert_llama = LlmAgent(
    name="Llama_Expert",
    model="gemini-2.5-flash",
    instruction="""You are a Technical Analyst Expert evaluating SPY.
Given standard OHLCV prices and moving averages for the past 10 days provided in the context, predict if the price will Rise, Fall, or remain Neutral tomorrow.
Use {enriched_market_data} and {filtered_news_context} for context.
Output your prediction EXACTLY as a single float between 0.0 and 1.0, where 1.0=Strong Rise, 0.5=Neutral, 0.0=Strong Fall. 
No text, no preamble, just the float value.""",
    tools=[stochastic_filter_update_tool],
    output_key="pred_llama",
    before_model_callback=expert_adapter.before_model,
    after_model_callback=expert_adapter.after_model,
    before_tool_callback=expert_adapter.before_tool,
    after_tool_callback=expert_adapter.after_tool,
    before_agent_callback=expert_adapter.before_agent,
    after_agent_callback=expert_adapter.after_agent,
)

# Expert 2: GPT4o (The Fundamentalist / Macro)
expert_gpt = LlmAgent(
    name="GPT4o_Expert",
    model="gemini-2.5-flash",
    instruction="""You are a Fundamental Macroeconomic Analyst Expert evaluating the broader stock market (SPY).
Given the market context provided, predict if the asset will experience a macro-level Rise, Fall, or Neutral move tomorrow. 
Ignore short-term technical noise, focus on structural gravity and long-horizon price memory.
Use {enriched_market_data} and {filtered_news_context} for context.
Output your prediction EXACTLY as a single float between 0.0 and 1.0, where 1.0=Strong Rise, 0.5=Neutral, 0.0=Strong Fall.
No text, no preamble, just the float value.""",
    tools=[stochastic_filter_update_tool],
    output_key="pred_gpt",
    before_model_callback=expert_adapter.before_model,
    after_model_callback=expert_adapter.after_model,
    before_tool_callback=expert_adapter.before_tool,
    after_tool_callback=expert_adapter.after_tool,
    before_agent_callback=expert_adapter.before_agent,
    after_agent_callback=expert_adapter.after_agent,
)

# Expert 3: Mixtral (The Contrarian / Mean-Reversion)
expert_mixtral = LlmAgent(
    name="Mixtral_Expert",
    model="gemini-2.5-flash",
    instruction="""You are a High-Frequency Mean-Reverting Analyst Expert.
Look at the past 10 days of price context. If it rallied hard, bet that it Falls. If it dumped, bet that it Rises. You believe markets are rubber bands.
Use {enriched_market_data} and {filtered_news_context} for context.
Output your prediction EXACTLY as a single float between 0.0 and 1.0, where 1.0=Strong Rise, 0.5=Neutral, 0.0=Strong Fall.
No text, no preamble, just the float value.""",
    tools=[stochastic_filter_update_tool],
    output_key="pred_mixtral",
    before_model_callback=expert_adapter.before_model,
    after_model_callback=expert_adapter.after_model,
    before_tool_callback=expert_adapter.before_tool,
    after_tool_callback=expert_adapter.after_tool,
    before_agent_callback=expert_adapter.before_agent,
    after_agent_callback=expert_adapter.after_agent,
)

# Group them into a Parallel Swarm for Fan-Out execution
moe_parallel_swarm = ParallelAgent(
    name="ParallelFilterPhase",
    sub_agents=[expert_llama, expert_gpt, expert_mixtral]
)
