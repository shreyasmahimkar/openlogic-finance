"""
strategy_testing/backtesting/simulator.py

Lightweight, event-driven local simulator and evaluation engine for testing crossover strategies.
"""

import pandas as pd
import numpy as np

def get_performance_summary(series: pd.Series, name: str) -> dict:
    """
    Calculate CAGR, total return, and max drawdown for an equity curve.
    """
    total_return = (series.iloc[-1] / series.iloc[0]) - 1
    n_days = len(series)
    years = n_days / 252.0
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1.0 / years) - 1 if years > 0 else 0.0
    
    roll_max = series.cummax()
    drawdowns = (series - roll_max) / roll_max
    max_dd = drawdowns.min()
    
    return {
        "Strategy": name,
        "Final Value": f"${series.iloc[-1]:,.2f}",
        "Total Return": f"{total_return * 100:.2f}%",
        "CAGR": f"{cagr * 100:.2f}%",
        "Max Drawdown": f"{max_dd * 100:.2f}%"
    }

def run_local_simulation(df: pd.DataFrame, initial_capital: float = 100000.0) -> pd.DataFrame:
    """
    Run a local crossover strategy simulation without risk management.
    """
    sim_df = df.copy()
    
    # 1. Benchmark: Buy & Hold
    sim_df["Buy_Hold_Shares"] = initial_capital / sim_df["Close"].iloc[0]
    sim_df["Buy_Hold_Value"] = sim_df["Buy_Hold_Shares"] * sim_df["Close"]
    
    # 2. Standard Crossover Strategy (Without Risk Management)
    portfolio_values = []
    cash = initial_capital
    shares = 0.0
    position = 0  # 0: Cash, 1: Long
    
    for date, row in sim_df.iterrows():
        close = row["Close"]
        sig = row["Signal"]
        
        # Process signal at market close
        if sig == "GOLDEN_CROSS" and position == 0:
            shares = cash / close
            cash = 0.0
            position = 1
        elif sig == "DEATH_CROSS" and position == 1:
            cash = shares * close
            shares = 0.0
            position = 0
            
        current_val = cash + (shares * close)
        portfolio_values.append(current_val)
        
    sim_df["Strat_Value"] = portfolio_values
    return sim_df
