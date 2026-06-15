from model_library.ml_zoo.logistic_regression import (
    LogisticStrategyConfig,
    LogisticModelPayload
)

# 1. Initialize our strategy config and pretrained model payload
config = LogisticStrategyConfig(
    ticker="SPY",
    fast_period=50,
    slow_period=200,
    rsi_period=14,
    probability_threshold=0.5,
    position_size=1.0,
    max_drawdown_pct=0.15
)

model_payload = LogisticModelPayload(
    weights={
        "sma_ratio": 2.5,
        "rsi_norm": 0.5,
        "momentum": 1.0
    },
    intercept=0.1,
    feature_means={
        "sma_ratio": 0.005,
        "rsi_norm": 0.02,
        "momentum": 0.0003
    },
    feature_stds={
        "sma_ratio": 0.03,
        "rsi_norm": 0.35,
        "momentum": 0.015
    }
)

print("Pre-Trained Model weights (scaled feature space):")
for f, w in model_payload.weights.items():
    print(f"  - {f}: {w}")
print(f"  - intercept (bias): {model_payload.intercept}")