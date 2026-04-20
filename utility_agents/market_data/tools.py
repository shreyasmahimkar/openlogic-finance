import logging
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
from tenacity import retry, wait_exponential, stop_after_attempt

logger = logging.getLogger(__name__)

# Apply robust fallback for API flakiness (Day 4: Robustness)
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def fetch_stock_data(ticker: str = "SPY", period: str = "10y") -> dict:
    """
    Fetches historical stock prices using yfinance.

    Args:
        ticker: The stock ticker to fetch (default SPY).
        period: The historical period to fetch (default 10y).
        
    Returns:
        JSON/dict summarizing the fetched data including the latest price metrics.
    """
    logger.info(f"Fetching {period} historical data for {ticker}")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        error_msg = f"Failed to fetch data for {ticker}. The returned dataframe is empty."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Contextual directory check based on runtime environment
    # Useful for differentiating between local testing and Cloud Run (/app context)
    asset_dir = "/app/assets" if os.environ.get("APP_HOME") else "assets"
    os.makedirs(asset_dir, exist_ok=True)
    
    csv_path = os.path.join(asset_dir, f"{ticker}_{period}.csv")
    df.to_csv(csv_path)
    
    logger.info(f"Successfully fetched {len(df)} rows for {ticker}. Saved raw to {csv_path}")
    
    return {
        "status": "success",
        "ticker": ticker,
        "rows_fetched": len(df),
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "csv_path": csv_path,
        "latest_close_price": round(df["Close"].iloc[-1], 2)
    }

