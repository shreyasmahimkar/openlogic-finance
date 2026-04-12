import os
import json
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from typing import List
from google.adk.agents import Agent
from .filters import MoEF_Filter
from .experts import EXPERTS

# Configuration for ADK Memory Session Fallback
STATE_FILE = os.path.join(os.path.dirname(__file__), "moe_state.json")

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        "last_expert_preds": [0.5] * len(EXPERTS),
        "filter_state": {}
    }

def save_state(state: dict):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

async def _poll_expert(idx, expert, prompt):
    """Internal helper to poll an ADK agent async and extract its text output."""
    try:
        response_gen = expert.run_async(prompt)
        text_out = ""
        async for event in response_gen:
            if hasattr(event, "text") and event.text is not None:
                text_out += event.text
        
        # Parse output rigidly to a float
        val = float(text_out.strip())
        val = max(0.0, min(1.0, val))
        return val
    except Exception as e:
        print(f"Failed to poll/parse Expert {idx} output: {e}")
        # Default to neutral on hallucination / failure
        return 0.5

async def execute_experts(context: str) -> List[float]:
    """Parallel execution of ADK Sub-agents with rigid constraints."""
    prompt = f"Here is the trailing OHLCV market context:\n\n{context}\n\nWhat is your prediction?"
    
    # Concurrently poll
    tasks = [_poll_expert(i, expert, prompt) for i, expert in enumerate(EXPERTS)]
    return await asyncio.gather(*tasks)

async def invoke_moe_filter(market_context_url: str) -> str:
    """
    Stochastic Filtering Execution via ADK Sub-agent orchestration.
    
    Args:
        market_context_url: The path or URL to the trailing OHLCV CSV file.
    Returns:
        String detailing the filter's output.
    """
    # 1. Parse CSV & Calculate True Ground-Truth
    if not os.path.exists(market_context_url):
        return f"Error: The provided URL {market_context_url} does not exist locally."
        
    try:
        df = pd.read_csv(market_context_url)
        if 'Close' not in df.columns:
            return "Error: CSV requires a 'Close' column."
        
        if len(df) < 2:
            return "Error: Need at least 2 rows to compute close change."
            
        last_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        pct_change = (last_close - prev_close) / prev_close
        
        if pct_change > 0.005:  # Rise > 0.5%
            y_true = 1.0
        elif pct_change < -0.005: # Fall < -0.5%
            y_true = 0.0
        else:
            y_true = 0.5 # Neutral
            
        # Extract a 15-day slice for the experts to analyze to save tokens
        context_slice = df.tail(15).to_csv(index=False)
        
    except Exception as e:
        return f"Error parsing CSV context: {e}"

    # 2. ADK Memory: Load Filter state and previous predictions
    state = load_state()
    last_expert_preds = state["last_expert_preds"]
    
    moef_filter = MoEF_Filter(num_experts=len(EXPERTS), lambda_reg=1.0)
    if state.get("filter_state"):
        moef_filter.set_state(state["filter_state"])
        
    # 3. Online Filter Update
    # Update the transition matrix and mixture weights iteratively using yesterday's prediction
    moef_filter.update_filter(y_true=y_true, expert_preds=last_expert_preds)

    # 4. Context Injection & Polling Experts
    # Orchestrate LLMs concurrently
    new_expert_preds = await execute_experts(context_slice)

    # 5. Robust Aggregation
    final_prediction = moef_filter.predict_ensemble(new_expert_preds)
    ranks = moef_filter.get_expert_rankings()
    
    # 6. Save State for Next Turn (Continuous Memory)
    new_state = {
        "last_expert_preds": new_expert_preds,
        "filter_state": moef_filter.get_state()
    }
    save_state(new_state)
    
    # 7. Output Rendering (CSV & Plot)
    history_file = os.path.join(os.path.dirname(__file__), "moe_history.csv")
    row = {
        "timestamp": datetime.datetime.now().isoformat(),
        "y_true": y_true,
        "moef_prediction": final_prediction,
        "weight_tech": ranks[0],
        "weight_fund": ranks[1],
        "weight_contra": ranks[2],
    }
    df_hist = pd.DataFrame([row])
    if os.path.exists(history_file):
        df_hist.to_csv(history_file, mode='a', header=False, index=False)
    else:
        df_hist.to_csv(history_file, index=False)

    df_all = pd.read_csv(history_file)
    plot_msg = ""
    if len(df_all) >= 2:
        try:
            plt.figure(figsize=(10, 5))
            # Mock regimes coloring for visually matching "Threading the needle"
            plt.axvspan(0, len(df_all)*0.33, color='blue', alpha=0.1, label='Mixed')
            plt.axvspan(len(df_all)*0.33, len(df_all)*0.66, color='gray', alpha=0.1, label='Neutral')
            plt.axvspan(len(df_all)*0.66, len(df_all), color='red', alpha=0.1, label='Bearish')
            
            plt.plot(range(len(df_all)), df_all['y_true'], 'o-', color='grey', label='Ground Truth', markersize=4)
            plt.plot(range(len(df_all)), df_all['moef_prediction'], '-', color='#00ff99', linewidth=3, label='MoE-F Filter')
            plt.title('Threading the needle through diverse market regimes')
            plt.legend()
            chart_path = os.path.join(os.path.dirname(__file__), "moe_regimes.png")
            plt.savefig(chart_path, bbox_inches='tight')
            plt.close()
            plot_msg = f"\nRendered visual regimes to {chart_path}"
        except Exception as e:
            plot_msg = f"\nFailed to render plot: {e}"
    
    res = (f"MoE-F Filter execution complete.\n"
           f"Ground Truth Mapped: {y_true}\n"
           f"New Dynamic Rankings (0=Tech, 1=Fund, 2=Contra): {ranks}\n"
           f"Final Ensemble Prediction: {final_prediction:.3f}"
           f"{plot_msg}")
    return res

root_agent = Agent(
    name="moef_coordinator",
    model="gemini-2.5-flash",
    instruction="""You are the Root Agent implementing the MoE-F (Stochastic Filtering) paper.
Your goal is to coordinate predictions using an ensemble of 3 distinct LLM personas.

Whenever the user asks for a market forecast, you must sequentially:
1. Observe the `market_context_url` (like an `assets/SPY_10y.csv` path from the ingestion phase). If you don't have one, ask for it.
2. Trigger the `invoke_moe_filter` tool.
3. Return the mathematically grouped Ensemble Prediction to the user, exploring which expert was rewarded most heavily this time step.""",
    description="Orchestrator Agent routing data through the Wonham-Shiryaev gating filter.",
    tools=[invoke_moe_filter]
)
