"""
model_library/ml_zoo/logistic_regression.py

Pure Python Logistic Regression model strategy module.
Strictly zero dependencies on third-party libraries (no pandas, no numpy, no sklearn).
Operates entirely on plain Python floats, dicts, and built-in standard library structures.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional


class LRSignalType(Enum):
    """
    Signals emitted by the Logistic Regression strategy.
    """
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"


@dataclass
class LogisticStrategyConfig:
    """
    Configuration parameters for the Logistic Regression trading strategy.
    """
    ticker: str = "SPY"
    fast_period: int = 50
    slow_period: int = 200
    rsi_period: int = 14
    probability_threshold: float = 0.5
    position_size: float = 1.0
    max_drawdown_pct: float = 0.15

    def __post_init__(self):
        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"fast_period ({self.fast_period}) must be < slow_period ({self.slow_period})"
            )
        if not 0.0 < self.position_size <= 1.0:
            raise ValueError(
                f"position_size ({self.position_size}) must be in (0.0, 1.0]"
            )
        if not 0.0 < self.max_drawdown_pct < 1.0:
            raise ValueError(
                f"max_drawdown_pct ({self.max_drawdown_pct}) must be in (0.0, 1.0)"
            )
        if not 0.0 < self.probability_threshold < 1.0:
            raise ValueError(
                f"probability_threshold ({self.probability_threshold}) must be in (0.0, 1.0)"
            )
        if self.rsi_period <= 0:
            raise ValueError(f"rsi_period ({self.rsi_period}) must be positive")


@dataclass
class LogisticModelPayload:
    """
    Standardized payload for a pre-trained Logistic Regression model.
    Contains weights, intercept, and feature scaling statistics (means and standard deviations).
    """
    weights: dict[str, float]
    intercept: float
    feature_means: dict[str, float]
    feature_stds: dict[str, float]

    def __post_init__(self):
        # Validate that feature_means and feature_stds contain the exact same features as weights
        weight_features = set(self.weights.keys())
        mean_features = set(self.feature_means.keys())
        std_features = set(self.feature_stds.keys())

        if weight_features != mean_features:
            raise ValueError(
                f"Feature mismatch between weights {weight_features} and feature_means {mean_features}"
            )
        if weight_features != std_features:
            raise ValueError(
                f"Feature mismatch between weights {weight_features} and feature_stds {std_features}"
            )

        # Validate that standard deviations are positive (non-zero, non-negative)
        for feature, std in self.feature_stds.items():
            if std <= 0.0:
                raise ValueError(
                    f"Standard deviation for feature '{feature}' must be positive. Got: {std}"
                )


def engineer_features(raw_data: dict[str, float]) -> dict[str, float]:
    """
    Lightweight feature engineering from raw market indicator values.

    Expected raw_data dict keys:
        - 'close': float (current close price)
        - 'fast_sma': float (fast moving average)
        - 'slow_sma': float (slow moving average)
        - 'rsi': float (Relative Strength Index value, typical range [0, 100])
        - 'prev_close': float (previous bar close price, optional)

    Returns:
        dict[str, float]: Dict containing:
            - 'sma_ratio': (fast_sma / slow_sma) - 1.0 (or 0.0 if slow_sma is 0.0)
            - 'rsi_norm': (rsi - 50.0) / 50.0 (scales RSI from [0, 100] to [-1, 1])
            - 'momentum': (close / prev_close) - 1.0 (or 0.0 if prev_close <= 0.0)
    """
    features = {}

    close = raw_data.get("close", 0.0)
    fast_sma = raw_data.get("fast_sma", 0.0)
    slow_sma = raw_data.get("slow_sma", 0.0)
    rsi = raw_data.get("rsi", 50.0)
    prev_close = raw_data.get("prev_close", 0.0)

    # 1. SMA Ratio feature
    if slow_sma != 0.0:
        features["sma_ratio"] = (fast_sma / slow_sma) - 1.0
    else:
        features["sma_ratio"] = 0.0

    # 2. RSI Normalized feature
    features["rsi_norm"] = (rsi - 50.0) / 50.0

    # 3. Simple Momentum feature
    if prev_close > 0.0:
        features["momentum"] = (close / prev_close) - 1.0
    else:
        features["momentum"] = 0.0

    return features


def predict_probability(features: dict[str, float], payload: LogisticModelPayload) -> float:
    """
    Predict signal probability using standardized weights, intercept, and feature engineering.
    Implements a numerically stable sigmoid to prevent overflow.

    z = intercept + sum( w_i * (x_i - mean_i) / std_i )
    p = 1 / (1 + exp(-z))

    Args:
        features: Dict of raw engineered features (e.g. {'sma_ratio': 0.02, 'rsi_norm': -0.1, ...})
        payload: Trained LogisticModelPayload with weights and scaling parameters.

    Returns:
        float: Prediction probability in interval (0.0, 1.0).
    """
    z = payload.intercept

    for feature, weight in payload.weights.items():
        val = features.get(feature, 0.0)
        mean = payload.feature_means[feature]
        std = payload.feature_stds[feature]
        
        # Standardize the feature value
        scaled_val = (val - mean) / std
        z += weight * scaled_val

    # Numerically stable sigmoid function
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)


def evaluate_signal(
    probability: float,
    prev_probability: Optional[float],
    threshold: float = 0.5,
) -> LRSignalType:
    """
    Evaluate state crossover transition signals based on probability crossing the threshold.

    - BUY: Probability crosses above threshold (prev_prob <= threshold, current_prob > threshold)
    - SELL: Probability crosses below threshold (prev_prob > threshold, current_prob <= threshold)
    - NONE: Otherwise

    Args:
        probability: Current probability estimate.
        prev_probability: Previous bar's probability estimate. None on first bar.
        threshold: Decision probability threshold (default: 0.5).

    Returns:
        LRSignalType: BUY, SELL, or NONE.
    """
    if prev_probability is None:
        return LRSignalType.NONE

    if prev_probability <= threshold and probability > threshold:
        return LRSignalType.BUY

    if prev_probability > threshold and probability <= threshold:
        return LRSignalType.SELL

    return LRSignalType.NONE


def project_weights(payload: LogisticModelPayload) -> tuple[dict[str, float], float]:
    """
    Map scaled weights back to the raw feature data space parameters.
    
    Mathematical proof of raw weights projection:
        Let x_scaled_i = (x_raw_i - mean_i) / std_i.
        Then the logit z is:
            z = intercept + sum( w_i * (x_raw_i - mean_i) / std_i )
            z = intercept - sum( w_i * mean_i / std_i ) + sum( (w_i / std_i) * x_raw_i )

        Thus, the equivalent linear relationship in raw space is:
            z = raw_intercept + sum( raw_weight_i * x_raw_i )
            where:
                raw_weight_i = w_i / std_i
                raw_intercept = intercept - sum( w_i * mean_i / std_i )

    Args:
        payload: The trained payload in scaled feature space.

    Returns:
        tuple[dict[str, float], float]: A tuple (raw_weights, raw_intercept).
    """
    raw_weights = {}
    raw_intercept = payload.intercept

    for feature, weight in payload.weights.items():
        mean = payload.feature_means[feature]
        std = payload.feature_stds[feature]
        
        # Compute raw feature space parameters
        raw_weights[feature] = weight / std
        raw_intercept -= weight * mean / std

    return raw_weights, raw_intercept
