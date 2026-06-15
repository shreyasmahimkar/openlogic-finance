"""
model_library/tests/test_logistic_regression.py

Unit tests for the Logistic Regression model strategy logic.
These tests require ZERO LEAN dependencies — just run:
    pytest model_library/tests/test_logistic_regression.py -v
"""

import pytest
import math
from model_library.ml_zoo.logistic_regression import (
    LRSignalType,
    LogisticStrategyConfig,
    LogisticModelPayload,
    engineer_features,
    predict_probability,
    evaluate_signal,
    project_weights,
)


# ══════════════════════════════════════════════════════════════════════════════
# LogisticStrategyConfig validation tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLogisticStrategyConfig:
    def test_default_construction(self):
        """Default config builds without error."""
        cfg = LogisticStrategyConfig()
        assert cfg.ticker == "SPY"
        assert cfg.fast_period == 50
        assert cfg.slow_period == 200
        assert cfg.rsi_period == 14
        assert cfg.probability_threshold == 0.5
        assert cfg.position_size == 1.0
        assert cfg.max_drawdown_pct == 0.15

    def test_custom_construction(self):
        """Custom config builds correctly."""
        cfg = LogisticStrategyConfig(
            ticker="QQQ",
            fast_period=20,
            slow_period=50,
            rsi_period=10,
            probability_threshold=0.6,
            position_size=0.8,
            max_drawdown_pct=0.1,
        )
        assert cfg.ticker == "QQQ"
        assert cfg.fast_period == 20
        assert cfg.slow_period == 50
        assert cfg.rsi_period == 10
        assert cfg.probability_threshold == 0.6
        assert cfg.position_size == 0.8
        assert cfg.max_drawdown_pct == 0.1

    def test_invalid_fast_gte_slow_raises(self):
        """fast_period >= slow_period should raise ValueError."""
        with pytest.raises(ValueError, match="fast_period"):
            LogisticStrategyConfig(fast_period=200, slow_period=50)

    def test_invalid_position_size_raises(self):
        """position_size out of bounds raises ValueError."""
        with pytest.raises(ValueError, match="position_size"):
            LogisticStrategyConfig(position_size=0.0)
        with pytest.raises(ValueError, match="position_size"):
            LogisticStrategyConfig(position_size=1.2)

    def test_invalid_max_drawdown_raises(self):
        """max_drawdown_pct out of bounds raises ValueError."""
        with pytest.raises(ValueError, match="max_drawdown_pct"):
            LogisticStrategyConfig(max_drawdown_pct=0.0)
        with pytest.raises(ValueError, match="max_drawdown_pct"):
            LogisticStrategyConfig(max_drawdown_pct=1.0)

    def test_invalid_threshold_raises(self):
        """probability_threshold out of bounds raises ValueError."""
        with pytest.raises(ValueError, match="probability_threshold"):
            LogisticStrategyConfig(probability_threshold=0.0)
        with pytest.raises(ValueError, match="probability_threshold"):
            LogisticStrategyConfig(probability_threshold=1.0)

    def test_invalid_rsi_period_raises(self):
        """rsi_period <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="rsi_period"):
            LogisticStrategyConfig(rsi_period=0)


# ══════════════════════════════════════════════════════════════════════════════
# LogisticModelPayload validation tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLogisticModelPayload:
    def test_valid_payload(self):
        """Valid payload constructed without error."""
        payload = LogisticModelPayload(
            weights={"f1": 1.0, "f2": -2.0},
            intercept=0.5,
            feature_means={"f1": 0.0, "f2": 1.0},
            feature_stds={"f1": 1.0, "f2": 0.5},
        )
        assert payload.weights["f1"] == 1.0
        assert payload.intercept == 0.5

    def test_feature_mismatch_means_raises(self):
        """Feature mismatch in means raises ValueError."""
        with pytest.raises(ValueError, match="means"):
            LogisticModelPayload(
                weights={"f1": 1.0},
                intercept=0.5,
                feature_means={"f2": 0.0},
                feature_stds={"f1": 1.0},
            )

    def test_feature_mismatch_stds_raises(self):
        """Feature mismatch in stds raises ValueError."""
        with pytest.raises(ValueError, match="stds"):
            LogisticModelPayload(
                weights={"f1": 1.0},
                intercept=0.5,
                feature_means={"f1": 0.0},
                feature_stds={"f2": 1.0},
            )

    def test_non_positive_std_raises(self):
        """Standard deviation <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            LogisticModelPayload(
                weights={"f1": 1.0},
                intercept=0.5,
                feature_means={"f1": 0.0},
                feature_stds={"f1": 0.0},
            )
        with pytest.raises(ValueError, match="positive"):
            LogisticModelPayload(
                weights={"f1": 1.0},
                intercept=0.5,
                feature_means={"f1": 0.0},
                feature_stds={"f1": -0.5},
            )


# ══════════════════════════════════════════════════════════════════════════════
# engineer_features tests
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineerFeatures:
    def test_standard_feature_engineering(self):
        """Features engineered correctly with normal values."""
        raw = {
            "close": 105.0,
            "fast_sma": 102.0,
            "slow_sma": 100.0,
            "rsi": 60.0,
            "prev_close": 100.0,
        }
        feats = engineer_features(raw)
        assert abs(feats["sma_ratio"] - 0.02) < 1e-7
        assert abs(feats["rsi_norm"] - 0.2) < 1e-7
        assert abs(feats["momentum"] - 0.05) < 1e-7

    def test_zero_slow_sma_safety(self):
        """slow_sma of 0 does not raise division by zero error."""
        raw = {
            "close": 100.0,
            "fast_sma": 50.0,
            "slow_sma": 0.0,
            "rsi": 50.0,
        }
        feats = engineer_features(raw)
        assert feats["sma_ratio"] == 0.0

    def test_missing_prev_close_safety(self):
        """Missing or zero prev_close is handled safely."""
        raw = {
            "close": 100.0,
            "fast_sma": 100.0,
            "slow_sma": 100.0,
            "rsi": 50.0,
        }
        feats = engineer_features(raw)
        assert feats["momentum"] == 0.0

        raw_zero = raw.copy()
        raw_zero["prev_close"] = 0.0
        feats_zero = engineer_features(raw_zero)
        assert feats_zero["momentum"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# predict_probability tests
# ══════════════════════════════════════════════════════════════════════════════


class TestPredictProbability:
    def test_predict_probability_exact(self):
        """Verify the exact math of standardization and sigmoid."""
        # Simple setup:
        # x_raw = 12.0, mean = 10.0, std = 2.0 -> x_scaled = (12-10)/2 = 1.0
        # w = 2.0, intercept = -1.0
        # z = -1.0 + 2.0 * 1.0 = 1.0
        # p = 1 / (1 + exp(-1)) = 0.7310585786300049
        payload = LogisticModelPayload(
            weights={"f1": 2.0},
            intercept=-1.0,
            feature_means={"f1": 10.0},
            feature_stds={"f1": 2.0},
        )
        features = {"f1": 12.0}
        prob = predict_probability(features, payload)

        expected_z = 1.0
        expected_p = 1.0 / (1.0 + math.exp(-expected_z))
        assert abs(prob - expected_p) < 1e-9

    def test_sigmoid_math_stability_large_positive(self):
        """Extremely large positive z (e.g. z=1000) does not overflow and returns ~1.0."""
        payload = LogisticModelPayload(
            weights={"f1": 1000.0},
            intercept=0.0,
            feature_means={"f1": 0.0},
            feature_stds={"f1": 1.0},
        )
        features = {"f1": 1.0}
        # z = 1000
        prob = predict_probability(features, payload)
        assert prob == 1.0

    def test_sigmoid_math_stability_large_negative(self):
        """Extremely large negative z (e.g. z=-1000) does not overflow and returns ~0.0."""
        payload = LogisticModelPayload(
            weights={"f1": -1000.0},
            intercept=0.0,
            feature_means={"f1": 0.0},
            feature_stds={"f1": 1.0},
        )
        features = {"f1": 1.0}
        # z = -1000
        prob = predict_probability(features, payload)
        assert prob == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# evaluate_signal tests
# ══════════════════════════════════════════════════════════════════════════════


class TestEvaluateSignal:
    def test_buy_signal_crossover_above(self):
        """BUY triggered when probability crosses ABOVE threshold."""
        # Previous is at or below threshold (0.5), current is above (0.55)
        signal = evaluate_signal(probability=0.55, prev_probability=0.45, threshold=0.5)
        assert signal == LRSignalType.BUY

        # Exact border: from exactly 0.5 to 0.51
        signal = evaluate_signal(probability=0.51, prev_probability=0.5, threshold=0.5)
        assert signal == LRSignalType.BUY

    def test_sell_signal_crossover_below(self):
        """SELL triggered when probability crosses BELOW threshold."""
        # Previous is above (0.55), current is at or below (0.45)
        signal = evaluate_signal(probability=0.45, prev_probability=0.55, threshold=0.5)
        assert signal == LRSignalType.SELL

        # Exact border: from 0.51 to exactly 0.5
        signal = evaluate_signal(probability=0.5, prev_probability=0.51, threshold=0.5)
        assert signal == LRSignalType.SELL

    def test_no_signal_holding_above(self):
        """No signal when remaining above the threshold."""
        signal = evaluate_signal(probability=0.6, prev_probability=0.55, threshold=0.5)
        assert signal == LRSignalType.NONE

    def test_no_signal_holding_below(self):
        """No signal when remaining below the threshold."""
        signal = evaluate_signal(probability=0.4, prev_probability=0.45, threshold=0.5)
        assert signal == LRSignalType.NONE

    def test_first_bar_no_signal(self):
        """No signal on first bar (prev_probability is None)."""
        signal = evaluate_signal(probability=0.6, prev_probability=None, threshold=0.5)
        assert signal == LRSignalType.NONE


# ══════════════════════════════════════════════════════════════════════════════
# project_weights tests
# ══════════════════════════════════════════════════════════════════════════════


class TestProjectWeights:
    def test_projection_equivalence(self):
        """Verify raw-space projection yields IDENTICAL logits z as scaled-space computation."""
        payload = LogisticModelPayload(
            weights={"sma_ratio": 2.5, "rsi_norm": -1.2, "momentum": 0.8},
            intercept=0.1,
            feature_means={"sma_ratio": 0.005, "rsi_norm": 0.1, "momentum": 0.002},
            feature_stds={"sma_ratio": 0.03, "rsi_norm": 0.35, "momentum": 0.012},
        )

        # 1. Project weights to raw space
        raw_weights, raw_intercept = project_weights(payload)

        # 2. Random raw feature values
        raw_features = {"sma_ratio": 0.02, "rsi_norm": -0.15, "momentum": 0.005}

        # 3. Calculate z using scaled space
        z_scaled = payload.intercept
        for f, w in payload.weights.items():
            x_scaled = (raw_features[f] - payload.feature_means[f]) / payload.feature_stds[f]
            z_scaled += w * x_scaled

        # 4. Calculate z using raw space
        z_raw = raw_intercept
        for f, w_raw in raw_weights.items():
            z_raw += w_raw * raw_features[f]

        # 5. Assert equivalence (within float rounding limits)
        assert abs(z_scaled - z_raw) < 1e-12

        # Verify both predict identical probabilities
        prob_scaled = predict_probability(raw_features, payload)

        # Evaluate sigmoid directly with z_raw
        prob_raw = 1.0 / (1.0 + math.exp(-z_raw))
        assert abs(prob_scaled - prob_raw) < 1e-12
