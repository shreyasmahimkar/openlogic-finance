from google.adk.agents import Agent
from .tools import fetch_stock_data

# Define the Level 1 root agent.
root_agent = Agent(
    name="data_ingestion_agent",
    model="gemini-2.5-flash",
    instruction="""You are a Level 1 Data Ingestion Agent for OpenLogic Finance.
Your primary objective is to fetch historical stock data when requested.

When a user asks to review an asset (like SPY):
1. Think step-by-step. First, use `fetch_stock_data` to retrieve the data.
2. Provide a brief, professional summary to the user. Describe the metadata returned from the fetch (e.g., rows fetched, date range, and the latest close price).

Constraints:
- Translate assets to their official Yahoo Finance ticker (e.g. Bitcoin -> BTC-USD, Apple -> AAPL). If no asset is mentioned at all, default to "SPY" and "10y".
""",
    description="A foundational Agent responsible for ingesting historical data streams.",
    tools=[fetch_stock_data]
)
