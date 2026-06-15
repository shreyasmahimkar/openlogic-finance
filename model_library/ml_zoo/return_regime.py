"""Return/regime prediction model (Box 2) for the Equity Research Assistant.

Predicts the probability that the asset rises over the next `horizon` days from
technical features, and maps that probability to a **regime** (bearish / neutral /
bullish). A calibrated probability (binary up/down) keeps validation rich (AUC,
Brier, calibration) while the regime mapping gives the agent an interpretable
signal that lines up with the MoE-F 0-1 convention.

This is the *build* half of the MDLC; validation lives in
`strategy_testing/validation/`, monitoring in `live_paper_execution/monitoring/`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["ret_1d", "ret_5d", "sma_ratio", "rsi", "volatility"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized technical features from an OHLCV frame (needs a 'Close' column)."""
    close = df["Close"]
    out = pd.DataFrame(index=df.index)
    out["ret_1d"] = close.pct_change()
    out["ret_5d"] = close.pct_change(5)
    out["sma_ratio"] = close.rolling(20).mean() / close.rolling(50).mean() - 1.0
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = (100 - 100 / (1 + rs)) / 100.0  # normalized to [0, 1]
    out["volatility"] = close.pct_change().rolling(20).std()
    return out


def make_labels(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Binary label: 1 if the forward `horizon`-day return is positive."""
    fwd_return = df["Close"].shift(-horizon) / df["Close"] - 1.0
    return (fwd_return > 0).astype(int)


def build_training_frame(df: pd.DataFrame, horizon: int = 5):
    """Return (X, y) aligned and NaN-free, ready for train/validate."""
    X = engineer_features(df)
    y = make_labels(df, horizon)
    frame = X.assign(_y=y).dropna()
    return frame[FEATURES], frame["_y"]


def prob_to_regime(p: float) -> str:
    return "bullish" if p >= 0.6 else "bearish" if p < 0.4 else "neutral"


@dataclass
class ReturnRegimeModel:
    horizon: int = 5
    pipeline: Pipeline | None = None

    def build(self) -> "ReturnRegimeModel":
        self.pipeline = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
        )
        return self

    def train(self, X, y) -> "ReturnRegimeModel":
        if self.pipeline is None:
            self.build()
        self.pipeline.fit(X, y)
        return self

    def predict_proba_up(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_regime(self, X) -> list[str]:
        return [prob_to_regime(p) for p in self.predict_proba_up(X)]

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ReturnRegimeModel":
        import joblib

        return joblib.load(path)
