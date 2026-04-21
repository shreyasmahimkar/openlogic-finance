import os
import pandas as pd
import numpy as np

def generate_spy_mock_data():
    # 2025 typically has 252 trading days
    dates = pd.date_range(start='2025-01-02', periods=252, freq='B')
    
    np.random.seed(42) # Deterministic for testing
    
    # Simulate a SPY-like price path using geometric brownian motion
    S0 = 475.0 # Starting price near end of 2024
    mu = 0.08 / 252
    sigma = 0.15 / np.sqrt(252)
    
    returns = np.random.normal(mu, sigma, len(dates))
    prices = S0 * np.exp(np.cumsum(returns))
    
    # Derive OHLC
    open_prices = prices * (1 + np.random.normal(0, 0.001, len(dates)))
    high_prices = np.maximum(open_prices, prices) * (1 + np.abs(np.random.normal(0, 0.002, len(dates))))
    low_prices = np.minimum(open_prices, prices) * (1 - np.abs(np.random.normal(0, 0.002, len(dates))))
    close_prices = prices
    volume = np.random.randint(40000000, 100000000, len(dates))
    
    # Simulate an actual market regime for our Ground Truth labels (0: Bearish, 0.5: Neutral, 1.0: Bullish)
    # We will compute a simple 10-day moving average to decide the ground truth regime for the sake of the mock
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    })
    
    # Ground truth simulation:
    # Compute 5-day return
    df['5d_return'] = df['Close'].pct_change(5)
    
    def get_regime(ret):
        if pd.isna(ret):
            return 0.5
        if ret > 0.01:
            return 1.0
        elif ret < -0.01:
            return 0.0
        else:
            return 0.5
            
    df['Ground_Truth_Regime'] = df['5d_return'].apply(get_regime)
    
    # Mocking semantic news scores between 0 and 1, loosely correlated to recent returns
    df['SBERT_News_Sentiment'] = np.clip(df['Ground_Truth_Regime'] + np.random.normal(0, 0.2, len(dates)), 0.0, 1.0)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/spy_2025_mock.csv', index=False)
    print(f"Generated {len(df)} rows of SPY mock data in data/spy_2025_mock.csv")

if __name__ == '__main__':
    generate_spy_mock_data()
