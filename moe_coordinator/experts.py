from google.adk.agents import Agent

# Expert 1: The Technician
# Strictly analyzes OHLCV metrics looking for breakouts or collapses.
expert_technician = Agent(
    name="expert_technician",
    model="gemini-2.5-flash",
    instruction="""You are a Technical Analyst Expert evaluating SPY.
Given standard OHLCV prices and moving averages for the past 10 days provided in the context, predict if the price will Rise, Fall, or remain Neutral tomorrow.
Output your prediction EXACTLY as a single float between 0.0 and 1.0, where 1.0=Strong Rise, 0.5=Neutral, 0.0=Strong Fall. 
No text, no preamble, just the float value.""",
    description="A specialist focused purely on momentum oscillators and MACD."
)

# Expert 2: The Fundamentlist
# Tends to act conservatively, reacting only to massive macro shifts.
expert_fundamental = Agent(
    name="expert_fundamental",
    model="gemini-2.5-flash",
    instruction="""You are a Fundamental Macroeconomic Analyst Expert evaluating the broader stock market (SPY).
Given the market context provided, predict if the asset will experience a macro-level Rise, Fall, or Neutral move tomorrow. 
Ignore short-term technical noise, focus on structural gravity and long-horizon price memory.
Output your prediction EXACTLY as a single float between 0.0 and 1.0, where 1.0=Strong Rise, 0.5=Neutral, 0.0=Strong Fall.
No text, no preamble, just the float value.""",
    description="A conservative specialist anchoring on regime shifts."
)

# Expert 3: The Contrarian
# Looks for mean-reversion anomalies.
expert_contrarian = Agent(
    name="expert_contrarian",
    model="gemini-2.5-flash",
    instruction="""You are a High-Frequency Mean-Reverting Analyst Expert.
Look at the past 10 days of price context. If it rallied hard, bet that it Falls. If it dumped, bet that it Rises. You believe markets are rubber bands.
Output your prediction EXACTLY as a single float between 0.0 and 1.0, where 1.0=Strong Rise, 0.5=Neutral, 0.0=Strong Fall.
No text, no preamble, just the float value.""",
    description="A specialist betting against the herd."
)

# Total Experts = 3
EXPERTS = [expert_technician, expert_fundamental, expert_contrarian]
