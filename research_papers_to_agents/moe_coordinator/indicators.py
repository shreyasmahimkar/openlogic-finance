import logging
import pandas as pd
import numpy as np
import os
import time
from .block_convey.prismtrace_client import send_trace_async

logger = logging.getLogger(__name__)

def enrich_ohlcv_data(csv_path: str, state=None) -> str:
    """
    Reads an OHLCV CSV file, calculates MoE-F required technical indicators using 
    pandas and numpy exclusively, and saves it to a new enriched CSV.
    
    Required Indicators:
    - MACD: EMA(12) - EMA(26)
    - Bollinger Bands: SMA(20) ± [2 × σ(20)]
    - 30-Day RSI
    - 30-Day CCI
    - 30-Day DX
    - 30-Day & 60-Day SMAs
    """
    t0 = time.time()
    
    # If the LLM agent hallucinated a relative path or stripped the absolute path prefix, 
    # force it back to the local data directory ONLY if the relative path doesn't actually exist.
    if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "data", os.path.basename(csv_path))

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot enrich data. File not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    if 'Close' not in df.columns:
        raise ValueError("CSV must contain a 'Close' column.")
        
    logger.info(f"Calculating technical indicators for {csv_path}...")
    
    # 1. SMAs
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    
    # 2. MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    
    # 3. Bollinger Bands
    std_20 = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['SMA_20'] + (2 * std_20)
    df['Bollinger_Lower'] = df['SMA_20'] - (2 * std_20)
    
    # 4. 30-Day RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    # Standard RSI uses smoothed moving average (EMA)
    avg_gain = gain.ewm(alpha=1/30, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/30, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df['RSI_30'] = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    
    # Ensure High and Low exist for CCI and DX
    if 'High' in df.columns and 'Low' in df.columns:
        # 5. 30-Day CCI
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=30).mean()
        
        # Calculate Mean Deviation carefully to avoid deprecated df.mad()
        # pandas rolling.apply is slow, we use a rolling window directly
        def mean_deviation(x):
            return np.abs(x - np.mean(x)).mean()
            
        md = tp.rolling(window=30).apply(mean_deviation, raw=True)
        # Add epsilon to prevent division by zero
        df['CCI_30'] = (tp - sma_tp) / (0.015 * (md + 1e-8))
        
        # 6. 30-Day DX
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
        
        tr1 = df['High'] - df['Low']
        tr2 = np.abs(df['High'] - df['Close'].shift(1))
        tr3 = np.abs(df['Low'] - df['Close'].shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth with Wilder's method equivalent (alpha = 1/n)
        smoothed_plus_dm = pd.Series(plus_dm).ewm(alpha=1/30, adjust=False).mean()
        smoothed_minus_dm = pd.Series(minus_dm).ewm(alpha=1/30, adjust=False).mean()
        smoothed_tr = tr.ewm(alpha=1/30, adjust=False).mean()
        
        plus_di = 100 * (smoothed_plus_dm / (smoothed_tr + 1e-8))
        minus_di = 100 * (smoothed_minus_dm / (smoothed_tr + 1e-8))
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['DX_30'] = dx
    else:
        logger.warning("High/Low columns missing. Skipping CCI and DX indicators.")
        
    # Save to _enriched.csv
    base, ext = os.path.splitext(csv_path)
    enriched_path = f"{base}_enriched{ext}"
    df.to_csv(enriched_path, index=False)
    
    logger.info(f"Enriched CSV securely saved to {enriched_path}")
    
    ms = int((time.time() - t0) * 1000)
    session_id = state.get("session_id", "live_adk_run") if hasattr(state, "get") else "live_adk_run"
    send_trace_async(f"Enrich {csv_path} with quantitative indicators", f"Saved to {enriched_path}", "quantitative_indicator", ms, "math", 2, session_id)
    
    return enriched_path
