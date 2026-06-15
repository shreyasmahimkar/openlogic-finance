from google.adk.agents import Agent
from .tools import fetch_asset_data
from horizontal_foundation.interpretability.explain_engine import ExplanationEngine


def fetch_and_explain(
    ticker: str = "SPY", period: str = "10y", explanation_level: str = "beginner"
) -> dict:
    """
    Fetches historical asset data and generates a detailed human-readable explanation at the requested level.

    Args:
        ticker: The stock or crypto ticker (default SPY).
        period: Lookback period (default 10y).
        explanation_level: The user target audience level ('beginner' or 'academic').

    Returns:
        Metadata dict containing 'explanation' and other dataset stats.
    """
    result = fetch_asset_data(ticker, period)
    explanation = ExplanationEngine.explain_data_prep(result, level=explanation_level)
    result["explanation"] = explanation
    return result


# Define the Level 1 root agent.
root_agent = Agent(
    name="market_data_agent",
    model="gemini-2.5-flash",
    instruction="""You are a Level 1 Data Ingestion Agent for OpenLogic Finance.
Your primary objective is to fetch historical asset data and explain it using the Interpretability Engine tool.

When a user asks to ingest or review an asset (like SPY, Apple, or Bitcoin):
1. Think step-by-step. Invoke `fetch_and_explain` to retrieve the historical dataset.
2. Ensure you pass the appropriate `explanation_level` ('beginner' for simple terms or 'academic' for quant formulas) if the user specifies a target audience style. Otherwise, default to 'beginner'.
3. Present the retrieved metadata and the generated human-readable explanation directly to the user in a professional, structured manner.

Constraints:
- Translate assets to their official Yahoo Finance ticker (e.g. Bitcoin -> BTC-USD, Solana -> SOL-USD, Apple -> AAPL). If no asset is mentioned at all, default to "SPY" and "10y".
- DO NOT reject cryptocurrencies. They are fully supported.
""",
    description="A foundational Agent responsible for ingesting historical data streams and explaining them at different user expertise levels.",
    tools=[fetch_and_explain],
)
