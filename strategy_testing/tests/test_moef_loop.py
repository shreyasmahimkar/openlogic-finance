"""Box 3 backtest math: the MoE-F filter + Gibbs loop produces a valid forecast."""

import numpy as np
import pandas as pd

from strategy_testing.backtesting.final_test import SessionState, run_simulation


def _mock_df(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Ground_Truth_Regime": rng.uniform(0, 1, n),
            "SBERT_News_Sentiment": rng.uniform(0, 1, n),
        }
    )


def test_simulation_produces_valid_final_prediction():
    np.random.seed(42)  # final_test uses np.random.normal internally
    state = run_simulation(_mock_df())
    final = state.get("final_prediction")
    assert final is not None
    assert 0.0 <= final <= 1.0


def test_simulation_updates_each_expert_belief():
    np.random.seed(1)
    state = run_simulation(_mock_df())
    # Each expert maintains a normalized belief simplex (pi) in state.
    for agent in ("Llama_Expert", "GPT4o_Expert", "Mixtral_Expert"):
        pi = state.get(f"pi_{agent}")
        assert pi is not None
        assert abs(float(np.sum(pi)) - 1.0) < 1e-6


def test_simulation_is_resumable():
    np.random.seed(7)
    state = SessionState()
    run_simulation(_mock_df(n=10, seed=1), state)
    run_simulation(_mock_df(n=10, seed=2), state)  # continue with same state
    assert 0.0 <= state.get("final_prediction") <= 1.0
