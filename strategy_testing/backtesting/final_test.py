import os
import pandas as pd
import numpy as np


# A minimal mock of ADK SessionState since it is only passed as dict-like Context
class SessionState:
    def __init__(self):
        self._state = {}

    def get(self, key, default=None):
        return self._state.get(key, default)

    def set(self, key, val):
        self._state[key] = val


from model_library.agentic_ai.coordinator import render_moe_trajectories
from model_library.ml_zoo.filters import stochastic_filter_update, robust_gibbs_aggregation


def run_simulation(df: pd.DataFrame, state: "SessionState | None" = None) -> "SessionState":
    """Run the MoE-F filter + Gibbs-aggregation loop over a dataframe.

    Pure math (no file I/O / plotting) so it is unit-testable. Expects columns
    `Ground_Truth_Regime` and `SBERT_News_Sentiment`. Returns the final state,
    whose `final_prediction` holds the last aggregated MoE-F output.
    """
    state = state or SessionState()
    for _, row in df.iterrows():
        gt = row["Ground_Truth_Regime"]
        news = row["SBERT_News_Sentiment"]

        state.set("current_ground_truth", gt)

        # MOCK EXPERT PREDICTIONS:
        pred_llama = float(np.clip(news + np.random.normal(0, 0.1), 0.0, 1.0))
        pred_gpt = float(np.clip(gt + np.random.normal(0, 0.05), 0.0, 1.0))
        pred_mixtral = float(np.clip(1.0 - news + np.random.normal(0, 0.2), 0.0, 1.0))

        state.set("pred_llama", pred_llama)
        state.set("pred_gpt", pred_gpt)
        state.set("pred_mixtral", pred_mixtral)
        state.set("all_expert_predictions", [pred_llama, pred_gpt, pred_mixtral])

        # 1. Update filters; 2. Gibbs aggregation
        stochastic_filter_update("Llama_Expert", pred_llama, gt, state)
        stochastic_filter_update("GPT4o_Expert", pred_gpt, gt, state)
        stochastic_filter_update("Mixtral_Expert", pred_mixtral, gt, state)
        robust_gibbs_aggregation(state)

    return state


def run_test():
    base_dir = os.path.dirname(__file__)
    history_file = os.path.join(base_dir, "moe_history.csv")
    if os.path.exists(history_file):
        os.remove(history_file)

    data_file = os.path.join(base_dir, "data/spy_2025_mock.csv")
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Run generate_mock_data.py first.")
        return

    df = pd.read_csv(data_file)
    print("Beginning 2025 SPY MoE-F Simulation Loop (252 Days)...")

    state = SessionState()
    msg = "no steps run"
    for idx in range(len(df)):
        run_simulation(df.iloc[[idx]], state)
        msg = render_moe_trajectories(state)  # appends + plots
        if (idx + 1) % 50 == 0:
            print(f"Propagated Day {idx + 1}... Current predictions aggregated.")

    print(f"Simulation Complete. Final status: {msg}")
    print("Output available at: moe_regimes.png")


if __name__ == "__main__":
    run_test()
