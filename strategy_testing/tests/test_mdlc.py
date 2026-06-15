"""MDLC tests: validate (Box 3) → serve (Box 5) → monitor (Box 5). Offline."""

import numpy as np
import pandas as pd

from horizontal_foundation.stats import population_stability_index
from live_paper_execution.monitoring.drift import monitor
from live_paper_execution.serving.predict import load_and_predict
from model_library.ml_zoo.return_regime import FEATURES, ReturnRegimeModel
from strategy_testing.validation.report import validate


def _learnable_xy(n=600, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    p = 1.0 / (1.0 + np.exp(-(2.5 * X["ret_5d"] + 1.0 * X["sma_ratio"])))
    y = pd.Series((rng.uniform(size=n) < p).astype(int))
    return X, y


def _split(X, y, frac=0.7):
    cut = int(len(X) * frac)
    return X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]


# ── validate ────────────────────────────────────────────────────────────────
def test_validation_report_and_gate():
    X, y = _learnable_xy()
    Xtr, ytr, Xte, yte = _split(X, y)
    model = ReturnRegimeModel().train(Xtr, ytr)
    report = validate(model, Xtr, ytr, Xte, yte)

    assert report.auc > 0.55  # learnable signal → real discrimination
    assert 0.0 <= report.brier <= 1.0
    assert report.n_train == len(Xtr) and report.n_test == len(Xte)
    assert len(report.confusion) == 2
    assert report.passes_gate() is True
    assert "PASS" in report.summary()


def test_gate_fails_on_weak_model():
    # Random labels → no signal → should not pass the gate.
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, len(FEATURES))), columns=FEATURES)
    y = pd.Series(rng.integers(0, 2, 400))
    Xtr, ytr, Xte, yte = _split(X, y)
    report = validate(ReturnRegimeModel().train(Xtr, ytr), Xtr, ytr, Xte, yte)
    assert report.passes_gate(min_auc=0.6) is False


# ── PSI / drift ───────────────────────────────────────────────────────────────
def test_psi_low_for_same_high_for_shifted():
    rng = np.random.default_rng(2)
    base = rng.normal(size=2000)
    assert population_stability_index(base, rng.normal(size=2000)) < 0.1
    assert population_stability_index(base, rng.normal(loc=3, size=2000)) > 0.25


# ── serve + monitor ───────────────────────────────────────────────────────────
def test_serving_roundtrip(tmp_path):
    X, y = _learnable_xy()
    model = ReturnRegimeModel().train(X, y)
    path = str(tmp_path / "m.joblib")
    model.save(path)
    probs, regimes = load_and_predict(path, X.iloc[:10])
    assert ((probs >= 0) & (probs <= 1)).all()
    assert len(regimes) == 10


def test_monitor_flags_drift_and_retrain():
    rng = np.random.default_rng(3)
    baseline = pd.DataFrame(rng.normal(size=(1000, 2)), columns=["a", "b"])
    drifted = pd.DataFrame({"a": rng.normal(loc=3, size=1000), "b": rng.normal(size=1000)})
    rep = monitor(baseline, drifted, rng.uniform(size=1000), rng.uniform(size=1000))
    assert rep.needs_retrain is True
    assert any("data drift: a" in alert for alert in rep.alerts)


def test_monitor_clean_when_stable():
    rng = np.random.default_rng(4)
    base = pd.DataFrame(rng.normal(size=(1000, 2)), columns=["a", "b"])
    cur = pd.DataFrame(rng.normal(size=(1000, 2)), columns=["a", "b"])
    scores = rng.uniform(size=1000)
    rep = monitor(base, cur, scores, rng.uniform(size=1000))
    assert rep.needs_retrain is False and rep.alerts == []
