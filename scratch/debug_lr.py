import sys
import os
import pandas as pd
import numpy as np
import math

repo_root = "/Users/shreyas/gitrepos/OpenSource/openlogic-finance"
sys.path.insert(0, repo_root)

from model_library.ml_zoo.logistic_regression import engineer_features

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def train_logistic_regression(df):
    df_ml = df.copy()
    
    delta = df_ml['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df_ml['RSI'] = 100 - (100 / (1 + rs))
    
    df_ml['Fast_SMA'] = df_ml['Close'].rolling(window=50).mean()
    df_ml['Slow_SMA'] = df_ml['Close'].rolling(window=200).mean()
    df_ml['Prev_Close'] = df_ml['Close'].shift(1)
    
    features_list = []
    for idx, row in df_ml.iterrows():
        raw_item = {
            "close": row['Close'],
            "fast_sma": row['Fast_SMA'],
            "slow_sma": row['Slow_SMA'],
            "rsi": row['RSI'],
            "prev_close": row['Prev_Close']
        }
        features_list.append(engineer_features(raw_item))
        
    feat_df = pd.DataFrame(features_list, index=df_ml.index)
    feat_df['Target'] = (df_ml['Close'].shift(-5) > df_ml['Close']).astype(int)
    
    clean_df = feat_df.dropna().copy()
    
    # Let's localize slicing to prevent timezone mismatch empty slices!
    train_df = clean_df.loc[:pd.to_datetime('2021-05-31', utc=True)]
    test_df = clean_df.loc[pd.to_datetime('2021-06-01', utc=True):pd.to_datetime('2022-05-31', utc=True)]
    
    feature_names = ['sma_ratio', 'rsi_norm', 'momentum']
    
    print("train_df size:", len(train_df))
    print("test_df size:", len(test_df))
    
    X_train = train_df[feature_names]
    y_train = train_df['Target']
    
    X_test = test_df[feature_names]
    y_test = test_df['Target']
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(penalty='l2', C=1.0, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    weights = dict(zip(feature_names, lr_model.coef_[0]))
    intercept = lr_model.intercept_[0]
    
    feature_means = dict(zip(feature_names, scaler.mean_))
    feature_stds = dict(zip(feature_names, scaler.scale_))
    
    return weights, intercept, feature_means, feature_stds

csv_path = os.path.join(repo_root, "assets", "SPY_10y.csv")
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df.set_index('Date', inplace=True)
df = df.sort_index()

weights, intercept, feature_means, feature_stds = train_logistic_regression(df)
print("\nDynamic Weights:", weights)
print("Dynamic Intercept:", intercept)

# Now check run simulations
fast_period = 50
slow_period = 200
rsi_period = 14
prob_threshold = 0.5

df['Fast_SMA'] = df['Close'].rolling(window=fast_period).mean()
df['Slow_SMA'] = df['Close'].rolling(window=slow_period).mean()
df['RSI'] = compute_rsi(df['Close'], rsi_period)
df['Prev_Close'] = df['Close'].shift(1)

lean_start_date = "2016-05-27"
lean_end_date = "2026-05-12"
sim_df = df.loc[lean_start_date:lean_end_date].copy()

signals_a = []
probabilities_a = []
prev_prob = None

for idx, row in sim_df.iterrows():
    fs = row['Fast_SMA']
    ss = row['Slow_SMA']
    r = row['RSI']
    pc = row['Prev_Close']
    c = row['Close']
    
    if pd.isna(fs) or pd.isna(ss) or pd.isna(r) or pd.isna(pc):
        signals_a.append("NONE")
        probabilities_a.append(0.5)
        continue
        
    sma_ratio = (fs / ss) - 1.0 if ss != 0.0 else 0.0
    rsi_norm = (r - 50.0) / 50.0
    momentum = (c / pc) - 1.0 if pc > 0.0 else 0.0
    
    z = intercept
    z += weights['sma_ratio'] * (sma_ratio - feature_means['sma_ratio']) / feature_stds['sma_ratio']
    z += weights['rsi_norm'] * (rsi_norm - feature_means['rsi_norm']) / feature_stds['rsi_norm']
    z += weights['momentum'] * (momentum - feature_means['momentum']) / feature_stds['momentum']
    
    prob = 1.0 / (1.0 + math.exp(-z)) if z >= 0.0 else math.exp(z) / (1.0 + math.exp(z))
    probabilities_a.append(prob)
    
    if prev_prob is None:
        signals_a.append("NONE")
    elif prev_prob <= prob_threshold and prob > prob_threshold:
        signals_a.append("GOLDEN_CROSS")
    elif prev_prob > prob_threshold and prob <= prob_threshold:
        signals_a.append("DEATH_CROSS")
    else:
        signals_a.append("NONE")
    prev_prob = prob

print("\nProbabilities Min:", min(probabilities_a))
print("Probabilities Max:", max(probabilities_a))
print("Probabilities Mean:", np.mean(probabilities_a))

sim_df['ModelA_Signal'] = signals_a
print("ModelA_Signal value counts:")
print(sim_df['ModelA_Signal'].value_counts())
