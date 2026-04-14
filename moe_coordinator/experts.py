from google.adk.agents import LlmAgent, ParallelAgent
from .filters import stochastic_filter_update_tool

# Expert 1: Llama (The Technician / Momentum)
expert_llama = LlmAgent(
    name="Llama_Expert",
    model="llama-3-8b",
    instruction="""You are a Technical Analyst Expert evaluating SPY.
Given standard OHLCV prices and moving averages for the past 10 days provided in the context, predict if the price will Rise, Fall, or remain Neutral tomorrow.
Use {structured_market_data} and {filtered_news_context} for context.
Output your prediction EXACTLY as a single float between 0.0 and 1.0, where 1.0=Strong Rise, 0.5=Neutral, 0.0=Strong Fall. 
No text, no preamble, just the float value.""",
    tools=[stochastic_filter_update_tool],
    output_key="pred_llama"
)

# Expert 2: GPT4o (The Fundamentalist / Macro)
expert_gpt = LlmAgent(
    name="GPT4o_Expert",
    model="gpt-4o",
    instruction="""You are a Fundamental Macroeconomic Analyst Expert evaluating the broader stock market (SPY).
Given the market context provided, predict if the asset will experience a macro-level Rise, Fall, or Neutral move tomorrow. 
Ignore short-term technical noise, focus on structural gravity and long-horizon price memory.
Use {structured_market_data} and {filtered_news_context} for context.
Output your prediction EXACTLY as a single float between 0.0 and 1.0, where 1.0=Strong Rise, 0.5=Neutral, 0.0=Strong Fall.
No text, no preamble, just the float value.""",
    tools=[stochastic_filter_update_tool],
    output_key="pred_gpt"
)

# Expert 3: Mixtral (The Contrarian / Mean-Reversion)
expert_mixtral = LlmAgent(
    name="Mixtral_Expert",
    model="mixtral-8x7b",
    instruction="""You are a High-Frequency Mean-Reverting Analyst Expert.
Look at the past 10 days of price context. If it rallied hard, bet that it Falls. If it dumped, bet that it Rises. You believe markets are rubber bands.
Use {structured_market_data} and {filtered_news_context} for context.
Output your prediction EXACTLY as a single float between 0.0 and 1.0, where 1.0=Strong Rise, 0.5=Neutral, 0.0=Strong Fall.
No text, no preamble, just the float value.""",
    tools=[stochastic_filter_update_tool],
    output_key="pred_mixtral"
)

# Group them into a Parallel Swarm for Fan-Out execution
moe_parallel_swarm = ParallelAgent(
    name="ParallelFilterPhase",
    sub_agents=[expert_llama, expert_gpt, expert_mixtral]
)
