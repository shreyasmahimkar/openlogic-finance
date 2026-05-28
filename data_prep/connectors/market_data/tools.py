import os
import yfinance as yf
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt

from horizontal_foundation.config.system_config import SystemConfig
from horizontal_foundation.utils.logging_helpers import get_logger
from horizontal_foundation.core.base_connector import BaseConnector

logger = get_logger(__name__)

class MarketDataConnector(BaseConnector):
    """Data connector representing the market data ingestion pipeline (Box 1)."""

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def fetch(self, ticker: str = "SPY", period: str = "10y") -> dict:
        """
        Ingests daily historical OHLCV data for an asset from yfinance.
        
        Args:
            ticker: The asset ticker (e.g. SPY, BTC-USD).
            period: The historical lookback period (e.g. 10y, 1y).
            
        Returns:
            A metadata dictionary describing the fetched series.
        """
        logger.info(f"Ingesting {period} data for {ticker} using MarketDataConnector")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            error_msg = f"Failed to fetch data for {ticker}. The returned dataframe is empty."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Resolve standard asset destination path via system configuration
        csv_path = SystemConfig.get_asset_path(f"{ticker}_{period}.csv")
        df.to_csv(csv_path)
        
        logger.info(f"Successfully fetched {len(df)} rows for {ticker}. Saved raw to {csv_path}")
        
        return {
            "status": "success",
            "ticker": ticker,
            "rows_fetched": len(df),
            "start_date": str(df.index.min().date()),
            "end_date": str(df.index.max().date()),
            "csv_path": str(csv_path),
            "latest_close_price": round(df["Close"].iloc[-1], 2)
        }

# Maintain 100% backward compatibility for all other callers (e.g., in other boxes or scripts)
def fetch_asset_data(ticker: str = "SPY", period: str = "10y") -> dict:
    """Backward-compatible function hook referencing the core MarketDataConnector."""
    connector = MarketDataConnector()
    return connector.fetch(ticker, period)
