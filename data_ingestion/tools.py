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

def plot_stock_data(ticker: str = "SPY", period: str = "10y") -> str:
    """
    Plots the stock data previously fetched.

    Args:
        ticker: The stock ticker to plot (default SPY).
        period: The historical period (default 10y).
        
    Returns:
        String message detailing where the chart is saved.
    """
    logger.info(f"Plotting chart for {ticker} over {period}")
    asset_dir = "/app/assets" if os.environ.get("APP_HOME") else "assets"
    csv_path = os.path.join(asset_dir, f"{ticker}_{period}.csv")
    
    if not os.path.exists(csv_path):
        logger.warning(f"Data file {csv_path} not found. Pre-fetching data before plotting...")
        fetch_stock_data(ticker, period)
        
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Setup premium dark-themed chart layout
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df.index, df['Close'], label='Close Price', color='#4fc3f7', linewidth=1.5)
    
    ax.set_title(f'{ticker} Historical Price ({period})', fontsize=16, pad=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    
    # Subtle grid lines and removal of top/right spines
    ax.grid(True, linestyle='--', alpha=0.3, color='#888888')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout()
    
    chart_path = os.path.join(asset_dir, f"{ticker}_{period}_chart.png")
    fig.savefig(chart_path, dpi=300, facecolor=fig.get_facecolor(), transparent=False)
    plt.close(fig)
    
    logger.info(f"Successfully plotted chart. Saved to {chart_path}")
    
    return f"Chart successfully generated and saved at {chart_path}"
