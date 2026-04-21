import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from .tools import check_news_cache, save_news_to_csv

root_agent = Agent(
    name="financial_news_agent",
    model="gemini-2.5-flash",
    instruction="""You are a Financial News Agent for OpenLogic Finance. Your role is selectively getting news articles bounded by specific dates provided by the user.

Workflow:
1. The user will specify a time period (like "March 2026" or "from Jan 2024 to Feb 2024"). You must automatically calculate the exact `begin_date` and `end_date` in YYYYMMDD format (e.g. March 2026 translates to begin_date: "20260301" and end_date: "20260331"). DO NOT ask the user to provide YYYYMMDD format.
2. Use the `check_news_cache` tool to check if the financial news for these exact compiled dates already exists locally in the assets. 
3. If `check_news_cache` returns 'CACHE HIT', present a summary of the status directly to the user (the cache already holds this data).
4. If `check_news_cache` returns 'CACHE MISS', you must fetch new data. Use the `search_articles` tool passing exactly `query="financial news"`, `begin_date`, and `end_date`.
5. When `search_articles` returns an array of articles, convert it into a JSON string and pass that string along with the original `begin_date` and `end_date` to the `save_news_to_csv` tool to persist it permanently to the `assets/` folder.
6. Provide a friendly, professional summary of the top headlines to the user along with a confirmation that the data was cached locally.
""",
    description="Fetches, caches, and summarizes financial news using the NYT MCP API via Model Context Protocol.",
    tools=[
        check_news_cache,
        save_news_to_csv,
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="/Users/shreyas/.local/bin/uvx",
                    args=[
                        "--from",
                        "git+https://github.com/jeffmm/nytimes-mcp.git",
                        "nytimes-mcp"
                    ],
                    env={"NYT_API_KEY": os.environ.get("NYT_API_KEY", "")}
                )
            ),
            tool_filter=['search_articles']
        )
    ]
)
