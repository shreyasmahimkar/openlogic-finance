import os
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def get_asset_dir() -> str:
    """Helper to get correct assets directory depending on runtime environment."""
    return "/app/assets" if os.environ.get("APP_HOME") else "assets"

def check_news_cache(begin_date: str, end_date: str) -> str:
    """
    Checks if a financial news CSV file exists for the given date range.

    Args:
        begin_date: The start date in YYYYMMDD format.
        end_date: The end date in YYYYMMDD format.

    Returns:
        A string summarizing the cached data if found. Otherwise, "CACHE MISS".
    """
    asset_dir = get_asset_dir()
    csv_path = os.path.join(asset_dir, f"financial_news_{begin_date}_{end_date}.csv")
    
    if os.path.exists(csv_path):
        logger.info(f"Cache hit for {csv_path}")
        df = pd.read_csv(csv_path)
        return f"CACHE HIT: Found {len(df)} news articles for {begin_date} to {end_date} in local cache. Summarize this data."
    
    logger.info(f"Cache miss for {csv_path}")
    return "CACHE MISS: Data not found in cache. Proceed to fetch from MCP."

def save_news_to_csv(articles_json_str: str, begin_date: str, end_date: str) -> str:
    """
    Saves the fetched news articles to a CSV file in the assets folder.

    Args:
        articles_json_str: JSON string of articles fetched from NYTimes API.
        begin_date: The start date in YYYYMMDD format.
        end_date: The end date in YYYYMMDD format.
        
    Returns:
        Status message about the save operation.
    """
    try:
        articles = json.loads(articles_json_str)
        if not articles:
            return "No articles were found to save."
            
        df = pd.DataFrame(articles)
        asset_dir = get_asset_dir()
        os.makedirs(asset_dir, exist_ok=True)
        
        csv_path = os.path.join(asset_dir, f"financial_news_{begin_date}_{end_date}.csv")
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved {len(df)} articles to {csv_path}")
        return f"SUCCESS: Successfully saved {len(df)} articles to {csv_path}."
    except Exception as e:
        logger.error(f"Failed to save to CSV: {str(e)}")
        return f"ERROR: Failed to save data: {str(e)}"
