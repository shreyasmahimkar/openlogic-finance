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
import uuid
import time
from .block_convey.prismtrace_client import send_trace_async

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
    # Explicitly enforce simulated files so live ADK won't read them
    state.set("history_file", "moe_history.csv")
    state.set("chart_file", "moe_regimes.png")
    
    print("Beginning 2025 SPY MoE-F Simulation Loop (252 Days)...")
    
    for idx, row in df.iterrows():
        # 1. Generate unique session ID for this step
        turn_id = str(uuid.uuid4())
        state.set("session_id", turn_id)
        state.set("step_order", 1)
        
        gt = row['Ground_Truth_Regime']
        news = row['SBERT_News_Sentiment']
        
        state.set("current_ground_truth", gt)
        
        # MOCK EXPERT PREDICTIONS:
        # Llama_Expert (Technical) follows recent news
        t0 = time.time()
        pred_llama = float(np.clip(news + np.random.normal(0, 0.1), 0.0, 1.0))
        ms0 = int((time.time()-t0)*1000)
        send_trace_async(
            user_input=f"Analyze {row['Date']} technical indicators", 
            output=str(pred_llama), model="llama-3-8b", latency_ms=ms0,
            step="llm_call", step_order=state.get("step_order"), session_id=turn_id
        )
        state.set("step_order", state.get("step_order") + 1)
        
        # GPT4o_Expert (Macro) more closely aligned with GT
        t1 = time.time()
        pred_gpt = float(np.clip(gt + np.random.normal(0, 0.05), 0.0, 1.0))
        ms1 = int((time.time()-t1)*1000)
        send_trace_async(
            user_input=f"Analyze {row['Date']} macro economy", 
            output=str(pred_gpt), model="gpt-4o", latency_ms=ms1,
            step="llm_call", step_order=state.get("step_order"), session_id=turn_id
        )
        state.set("step_order", state.get("step_order") + 1)
        
        # Mixtral_Expert (Contrarian) mean-reverting behaviour
        t2 = time.time()
        pred_mixtral = float(np.clip(1.0 - news + np.random.normal(0, 0.2), 0.0, 1.0))
        ms2 = int((time.time()-t2)*1000)
        send_trace_async(
            user_input=f"Analyze {row['Date']} mean reversion", 
            output=str(pred_mixtral), model="mixtral-8x7b", latency_ms=ms2,
            step="llm_call", step_order=state.get("step_order"), session_id=turn_id
        )
        state.set("step_order", state.get("step_order") + 1)
        
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
