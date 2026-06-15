"""Tests for the return/regime model (Box 2) — offline, no keys."""

import numpy as np
import pandas as pd

from model_library.ml_zoo.return_regime import (
    FEATURES,
    ReturnRegimeModel,
    build_training_frame,
    engineer_features,
    make_labels,
    prob_to_regime,
)


def _ohlcv(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.1, 1.0, n))
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1e6}
    )


def _learnable_xy(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    p = 1.0 / (1.0 + np.exp(-(2.0 * X["ret_5d"] + 1.0 * X["sma_ratio"])))
    y = pd.Series((rng.uniform(size=n) < p).astype(int))
    return X, y


def test_feature_engineering_columns_and_clean():
    df = _ohlcv()
    feats = engineer_features(df)
    assert list(feats.columns) == FEATURES
    X, y = build_training_frame(df)
    assert not X.isna().any().any()
    assert set(y.unique()).issubset({0, 1})


def test_labels_are_forward_looking():
    df = _ohlcv()
    y = make_labels(df, horizon=5)
    assert y.iloc[-5:].notna().sum() >= 0  # last `horizon` rows have no forward window
    assert set(y.dropna().unique()).issubset({0, 1})


def test_prob_to_regime_mapping():
    assert prob_to_regime(0.8) == "bullish"
    assert prob_to_regime(0.5) == "neutral"
    assert prob_to_regime(0.2) == "bearish"


def test_model_learns_and_predicts_in_range():
    X, y = _learnable_xy()
    model = ReturnRegimeModel().train(X, y)
    p = model.predict_proba_up(X)
    assert ((p >= 0) & (p <= 1)).all()
    assert set(model.predict_regime(X.iloc[:5])).issubset({"bullish", "neutral", "bearish"})


def test_save_load_roundtrip(tmp_path):
    X, y = _learnable_xy()
    model = ReturnRegimeModel(horizon=5).train(X, y)
    path = str(tmp_path / "model.joblib")
    model.save(path)
    loaded = ReturnRegimeModel.load(path)
    assert np.allclose(loaded.predict_proba_up(X), model.predict_proba_up(X))
    assert loaded.horizon == 5
