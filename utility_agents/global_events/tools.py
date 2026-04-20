import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def get_global_events() -> str:
    """
    Returns the global macro regime events as a stringified Pandas DataFrame.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "global_events.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path).to_string()
    return "No global events found."

def search_recent_events(start_date: str, end_date: str) -> str:
    """
    Searches the internet for global financial events in the requested date range.
    Use this if the requested date is beyond standard temporal bounds (e.g. > 2026-05-12).
    """
    logger.info(f"Searching internet for macro events between {start_date} and {end_date}...")
    # Mocking standard internet search hook for the Agent
    return f"Simulated Web Search Results for {start_date} to {end_date}:\n- Market adjusting to post-tariff environment.\n- Geopolitical tensions stabilize.\n- Fed rate trajectory maps a gentle pivot based on May CPI data."

def plot_stock_data(ticker: str = "SPY", period: str = "10y") -> str:
    """
    Plots the stock data previously fetched by the market data system, visually overlaying historical regimes.
    """
    logger.info(f"Plotting Global Events Chart for {ticker} over {period}")
    
    asset_dir = "/app/assets" if os.environ.get("APP_HOME") else "assets"
    csv_path = os.path.join(asset_dir, f"{ticker}_{period}.csv")

    if not os.path.exists(csv_path):
        logger.warning(f"Data file {csv_path} not found. Automatically invoking market_data tools to fetch it...")
        from utility_agents.market_data.tools import fetch_stock_data
        fetch_stock_data(ticker, period)
        
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    
    # Setup premium dark-themed chart layout
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df.index, df['Close'], label='Close Price', color='#4fc3f7', linewidth=1.5)
    
    ax.set_title(f'{ticker} Historical Price with Macro Regimes ({period})', fontsize=16, pad=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.3, color='#888888')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Overlay global events as shaded regime rectangles 
    events_path = os.path.join(os.path.dirname(__file__), "global_events.csv")
    if os.path.exists(events_path):
        events_df = pd.read_csv(events_path, parse_dates=["Start date", "End date"])
        events_df.dropna(subset=["Start date", "End date"], inplace=True)
        
        min_date, max_date = df.index.min(), df.index.max()
        
        for _, row in events_df.iterrows():
            sd = pd.to_datetime(row["Start date"], utc=True)
            ed = pd.to_datetime(row["End date"], utc=True)
            sentiment = str(row["Market sentiment"]).lower().strip()
            
            if sd > max_date or ed < min_date:
                continue
                
            sd_bound = max(sd, min_date)
            ed_bound = min(ed, max_date)
            
            color = 'gray'
            if sentiment == 'bull': color = 'green'
            elif sentiment == 'bear': color = 'red'
            
            ax.axvspan(sd_bound, ed_bound, color=color, alpha=0.1)
            
            # Annotate the Market State text inside the regime block
            state_label = str(row.get("Market State", ""))
            if state_label and state_label != "nan":
                mid_date = sd_bound + (ed_bound - sd_bound) / 2
                y_pos = df['Close'].max()
                
                # Add text vertically spanning down from the top 
                ax.text(mid_date, y_pos, state_label, 
                        rotation=90, color='white', alpha=0.7, 
                        fontsize=9, va='top', ha='center',
                        bbox=dict(facecolor='black', alpha=0.3, edgecolor='none'))

    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout()
    
    out_dir = "/app/assets" if os.environ.get("APP_HOME") else "assets"
    os.makedirs(out_dir, exist_ok=True)
    chart_path = os.path.join(out_dir, f"{ticker}_{period}_global_events_chart.png")
    fig.savefig(chart_path, dpi=300, facecolor=fig.get_facecolor(), transparent=False)
    plt.close(fig)
    
    logger.info(f"Successfully plotted global events chart. Saved to {chart_path}")
    return f"Global Events Chart successfully generated and saved at {chart_path}"
