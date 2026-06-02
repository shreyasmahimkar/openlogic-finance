import os
import pandas as pd
from strategy_testing.backtesting.simulator import run_local_simulation, get_performance_summary
from risk_management.portfolio.auditor import run_audited_simulation

repo_root = "/Users/shreyas/gitrepos/OpenSource/openlogic-finance"
csv_path = os.path.join(repo_root, "assets", "SPY_10y.csv")
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df.set_index('Date', inplace=True)
df = df.sort_index()

lean_start_date = "2016-05-27"
lean_end_date = "2026-05-12"
sim_df = df.loc[lean_start_date:lean_end_date].copy()

# Fast/Slow SMA, RSI
sim_df['Fast_SMA'] = sim_df['Close'].rolling(window=50).mean()
sim_df['Slow_SMA'] = sim_df['Close'].rolling(window=200).mean()

# Calculate signals exactly like app.py / notebook
signals_a = []
prev_prob = None
d_weights = {'sma_ratio': 2.5, 'rsi_norm': 0.5, 'momentum': 1.0}
d_intercept = 0.1
d_means = {'sma_ratio': 0.005, 'rsi_norm': 0.02, 'momentum': 0.0003}
d_stds = {'sma_ratio': 0.03, 'rsi_norm': 0.35, 'momentum': 0.015}

# Compute RSI
delta = sim_df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(com=13, adjust=False).mean()
avg_loss = loss.ewm(com=13, adjust=False).mean()
rs = avg_gain / avg_loss
sim_df['RSI'] = 100 - (100 / (1 + rs))
sim_df['Prev_Close'] = sim_df['Close'].shift(1)

import math
for idx, row in sim_df.iterrows():
    fs = row['Fast_SMA']
    ss = row['Slow_SMA']
    r = row['RSI']
    pc = row['Prev_Close']
    c = row['Close']
    
    if pd.isna(fs) or pd.isna(ss) or pd.isna(r) or pd.isna(pc):
        signals_a.append("NONE")
        continue
        
    sma_ratio = (fs / ss) - 1.0 if ss != 0.0 else 0.0
    rsi_norm = (r - 50.0) / 50.0
    momentum = (c / pc) - 1.0 if pc > 0.0 else 0.0
    
    z = d_intercept
    z += d_weights['sma_ratio'] * (sma_ratio - d_means['sma_ratio']) / d_stds['sma_ratio']
    z += d_weights['rsi_norm'] * (rsi_norm - d_means['rsi_norm']) / d_stds['rsi_norm']
    z += d_weights['momentum'] * (momentum - d_means['momentum']) / d_stds['momentum']
    
    prob = 1.0 / (1.0 + math.exp(-z)) if z >= 0.0 else math.exp(z) / (1.0 + math.exp(z))
    
    if prev_prob is None:
        signals_a.append("NONE")
    elif prev_prob <= 0.5 and prob > 0.5:
        signals_a.append("GOLDEN_CROSS")
    elif prev_prob > 0.5 and prob <= 0.5:
        signals_a.append("DEATH_CROSS")
    else:
        signals_a.append("NONE")
    prev_prob = prob

sim_df['Signal'] = signals_a
local_df_a = run_local_simulation(sim_df, 100000.0)
sum_a = get_performance_summary(local_df_a['Strat_Value'], "Model A Local")
print("Model A Local Simulation Return:")
print(sum_a)

# Model B
signals_b = []
prev_fast = None
prev_slow = None
for idx, row in sim_df.iterrows():
    fs = row['Fast_SMA']
    ss = row['Slow_SMA']
    if pd.isna(fs) or pd.isna(ss):
        signals_b.append("NONE")
        continue
    if prev_fast is None or prev_slow is None:
        signals_b.append("NONE")
    elif prev_fast <= prev_slow and fs > ss:
        signals_b.append("GOLDEN_CROSS")
    elif prev_fast >= prev_slow and fs < ss:
        signals_b.append("DEATH_CROSS")
    else:
        signals_b.append("NONE")
    prev_fast = fs
    prev_slow = ss

sim_df['Signal'] = signals_b
local_df_b = run_local_simulation(sim_df, 100000.0)
sum_b = get_performance_summary(local_df_b['Strat_Value'], "Model B Local")
print("Model B Local Simulation Return:")
print(sum_b)
