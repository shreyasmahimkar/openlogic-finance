import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

repo_root = "/Users/shreyas/gitrepos/OpenSource/openlogic-finance"
csv_path = os.path.join(repo_root, "assets", "SPY_10y.csv")
df = pd.read_csv(csv_path)

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

from model_library.ml_zoo.logistic_regression import engineer_features

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
train_df = clean_df.loc[:'2021-05-31']
test_df = clean_df.loc['2021-06-01':'2022-05-31']
feature_names = ['sma_ratio', 'rsi_norm', 'momentum']

X_train = train_df[feature_names]
y_train = train_df['Target']
X_test = test_df[feature_names]
y_test = test_df['Target']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(penalty='l2', C=1.0, random_state=42)
lr_model.fit(X_train_scaled, y_train)

y_pred = lr_model.predict(X_test_scaled)
y_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

print("Test Set Size:", len(y_test))
print("Target Value Counts:")
print(y_test.value_counts())
print("Unique Predictions:", np.unique(y_pred))
print("Max Predicted Probability:", np.max(y_prob))
print("Min Predicted Probability:", np.min(y_min := y_prob))
print("Mean Predicted Probability:", np.mean(y_prob))
