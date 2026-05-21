"""
MCP Client Module
Integrates the YFinance MCP and SBERT semantic tools for robust data ingestion.
This logic acts as the data connectivity layer for Box 1.
"""

class MCPClient:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        print("Initialized MCP Client connected to ", self.server_url)
        
    def fetch_market_data(self, ticker: str, period: str = "10y"):
        """
        Uses the YFinance MCP to fetch market data.
        """
        print(f"Fetching {period} market data for {ticker} via YFinance MCP...")
        # Stub logic
        return f"assets/{ticker}_{period}.csv"
        
    def semantic_filter_news(self, query: str, context_data: str):
        """
        Uses the SBERT semantic tools to filter out noise from news ingestion.
        """
        print(f"Applying SBERT semantic filter on {context_data} for '{query}'...")
        # Stub logic
        return "filtered_news_context"

if __name__ == "__main__":
    client = MCPClient()
    client.fetch_market_data("SPY", "10y")
