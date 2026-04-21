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

from .agent import render_moe_trajectories
from .filters import stochastic_filter_update, robust_gibbs_aggregation

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
    state = SessionState()
    
    print("Beginning 2025 SPY MoE-F Simulation Loop (252 Days)...")
    
    for idx, row in df.iterrows():
        gt = row['Ground_Truth_Regime']
        news = row['SBERT_News_Sentiment']
        
        state.set("current_ground_truth", gt)
        
        # MOCK EXPERT PREDICTIONS:
        # Llama_Expert (Technical) follows recent news
        pred_llama = float(np.clip(news + np.random.normal(0, 0.1), 0.0, 1.0))
        # GPT4o_Expert (Macro) more closely aligned with GT
        pred_gpt = float(np.clip(gt + np.random.normal(0, 0.05), 0.0, 1.0))
        # Mixtral_Expert (Contrarian) mean-reverting behaviour
        pred_mixtral = float(np.clip(1.0 - news + np.random.normal(0, 0.2), 0.0, 1.0))
        
        state.set("pred_llama", pred_llama)
        state.set("pred_gpt", pred_gpt)
        state.set("pred_mixtral", pred_mixtral)
        state.set("all_expert_predictions", [pred_llama, pred_gpt, pred_mixtral])
        
        # 1. Update Filters
        stochastic_filter_update("Llama_Expert", pred_llama, gt, state)
        stochastic_filter_update("GPT4o_Expert", pred_gpt, gt, state)
        stochastic_filter_update("Mixtral_Expert", pred_mixtral, gt, state)
        
        # 2. Gibbs Aggregation
        robust_gibbs_aggregation(state)
        
        # 3. Render Plot (appends internally)
        msg = render_moe_trajectories(state)
        
        if (idx + 1) % 50 == 0:
            print(f"Propagated Day {idx+1}... Current predictions aggregated.")

    print(f"Simulation Complete. Final status: {msg}")
    print("Output available at: moe_regimes.png")

if __name__ == '__main__':
    run_test()
