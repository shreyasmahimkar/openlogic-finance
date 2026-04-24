import os
import pandas as pd
import numpy as np
import asyncio
import time
import uuid

# Minimal mock of ADK SessionState
class SessionState:
    def __init__(self):
        self._state = {}
    def get(self, key, default=None):
        return self._state.get(key, default)
    def set(self, key, val):
        self._state[key] = val

from .agent import render_moe_trajectories, market_data_tool, technical_indicators_tool, sbert_news_filter
from .experts import moe_parallel_swarm
from .filters import stochastic_filter_update, robust_gibbs_aggregation
from .block_convey.prismtrace_client import send_trace_async
from google.adk.agents import LlmAgent

# Create an explainer agent uniquely for generating the final analysis CSV
explainer_agent = LlmAgent(
    name="ExplainabilityAgent",
    model="gemini-2.5-flash",
    instruction="""You are the Explainability Expert for the MoE-F output.
Given the technical/news context, the GT (Ground Truth) direction (1.0=Rise, 0.0=Fall), and the 3 expert scores (Llama, GPT4o, Mixtral) along with the final aggregated score:
1. Explain the state of the market (why it moved the way it did).
2. Explain the mathematical divergence between the agents (e.g. why did the Contrarian disagree with the Momentum agent?). 
Keep the entire explanation concise and formatted as a single small paragraph (1-3 sentences maximum).""",
    output_key="explanation"
)

async def run_simulation():
    base_dir = os.path.dirname(__file__)
    history_file = os.path.join(base_dir, "moe_history.csv")
    if os.path.exists(history_file):
        os.remove(history_file)
        
    print("Step 1: Fetching 90-Day Base Data ONCE...")
    # 1. Fetch 3mo data manually using the un-wrapped native python tools
    raw_csv = market_data_tool.func(ticker="SPY", period="3mo", state=SessionState())
    
    print("Step 2: Calculating Quantitative Indicators...")
    # 2. Enrich the data 
    enriched_msg = technical_indicators_tool.func(raw_csv, state=SessionState())
    
    enriched_csv = raw_csv.replace(".csv", "_enriched.csv")
    if not os.path.exists(enriched_csv):
        print(f"Failed to locate enriched CSV: {enriched_csv}")
        return
        
    df = pd.read_csv(enriched_csv)
    # Drop rows without enough SMA_60 history
    df_valid = df.dropna().reset_index(drop=True)
    
    if len(df_valid) < 8:
        print("Not enough trading days calculated after 60-day MA. Needs a wider window.")
        return
        
    print("Step 3: Fetching 30-Day Sub-Window NYT Financial News ONCE...")
    # 3. Pull news context ONCE to avoid API rate limits
    news_context = ""
    try:
        sbert_gen = sbert_news_filter.run_async("Fetch recent financial news for SPY from NYTimes and extract macro insights.")
        async for ev in sbert_gen:
            if hasattr(ev, 'data') and isinstance(ev.data, dict) and "filtered_news_context" in ev.data:
                news_context = ev.data["filtered_news_context"]
            elif hasattr(ev, 'data') and isinstance(ev.data, str):
                news_context = ev.data
    except Exception as e:
        print(f"SBERT News Fetch Failed (API / Rate limit timeout): {e}")
        
    if not news_context:
        news_context = "Fallback News: Markets digesting recent macroeconomic data."

    print("Step 4: Booting 7-Day Rolling Swarm Simulation...\n")
    state = SessionState()
    state.set("history_file", "moe_history.csv")
    state.set("chart_file", "moe_regimes.png")
    
    explanation_results = []
    
    num_days = len(df_valid)
    start_idx = num_days - 7
    
    for i in range(start_idx, num_days):
        date_str = str(df_valid.iloc[i]['Date'])
        
        # Context is exactly 10 days leading up to THIS specific loop iteration
        df_slice = df_valid.iloc[:i+1].tail(10)
        data_text = df_slice.to_string(index=False)
        
        # Determine Ground Truth
        # (Compare close of current day vs previous day)
        current_close = df_valid.iloc[i]['Close']
        prev_close = df_valid.iloc[i-1]['Close']
        gt = 1.0 if current_close > prev_close else 0.0
        state.set("current_ground_truth", gt)
        
        turn_id = str(uuid.uuid4())
        state.set("session_id", turn_id)
        state.set("step_order", 1)
        
        print(f"--- Swarm Evaluation for Day {i - start_idx + 1}/7 : [{date_str}] ---")
        
        # 1. Trigger true parallel swarm!
        swarm_gen = moe_parallel_swarm.run_async(
            user_input=f"Analyze {date_str} market direction for tomorrow.",
            variables={"enriched_market_data": data_text, "filtered_news_context": news_context}
        )
        
        swarm_outputs = []
        async for ev in swarm_gen:
            # We catch any string data emitted from the parallel agents
            # Since ADK ParallelAgent encapsulates sub-agents, we collect raw strings and guess order based on typical resolution
            if hasattr(ev, 'type') and "ResponseEvent" in str(type(ev)):
                 if hasattr(ev, 'data'):
                     swarm_outputs.append((ev.source.name if hasattr(ev, 'source') and hasattr(ev.source, 'name') else str(len(swarm_outputs)), ev.data))
        
        # The ParallelAgent returns a single event containing a List at the very end
        # We handle extraction resiliently:
        pred_llama, pred_gpt, pred_mixtral = 0.5, 0.5, 0.5

        # If data arrived as an aggregated list
        for _, out_text in swarm_outputs:
            if isinstance(out_text, list) and len(out_text) >= 3:
                try: pred_llama = float(''.join(c for c in str(out_text[0]) if c.isdigit() or c=='.'))
                except: pass
                try: pred_gpt = float(''.join(c for c in str(out_text[1]) if c.isdigit() or c=='.'))
                except: pass
                try: pred_mixtral = float(''.join(c for c in str(out_text[2]) if c.isdigit() or c=='.'))
                except: pass
        
        # Ensure constraints
        pred_llama = float(np.clip(pred_llama, 0.0, 1.0))
        pred_gpt = float(np.clip(pred_gpt, 0.0, 1.0))
        pred_mixtral = float(np.clip(pred_mixtral, 0.0, 1.0))

        print(f"Swarm Preds --> Llama: {pred_llama:.2f} | GPT: {pred_gpt:.2f} | Mixtral: {pred_mixtral:.2f} | (Market GT: {gt})")
        
        state.set("pred_llama", pred_llama)
        state.set("pred_gpt", pred_gpt)
        state.set("pred_mixtral", pred_mixtral)
        state.set("all_expert_predictions", [pred_llama, pred_gpt, pred_mixtral])
        
        # 2. Filter Updates
        stochastic_filter_update("Llama_Expert", pred_llama, gt, state)
        stochastic_filter_update("GPT4o_Expert", pred_gpt, gt, state)
        stochastic_filter_update("Mixtral_Expert", pred_mixtral, gt, state)
        
        # 3. Gibbs Aggregation
        final_agg = robust_gibbs_aggregation(state)
        print(f"Gibbs Aggregated Score: {final_agg:.3f}")
        
        # 4. Render Point Append
        render_moe_trajectories(state)
        
        # 5. Explainability AI
        explain_prompt = f"Day: {date_str}. Actual Market GT: {gt}. Llama (Technical): {pred_llama}, GPT (Macro): {pred_gpt}, Mixtral (Contrarian): {pred_mixtral}. Aggregated Final: {final_agg:.3f}. Quantitative Data Context text: {data_text}. News Context text: {news_context}."
        
        exp_gen = explainer_agent.run_async(user_input=explain_prompt)
        explanation_text = ""
        async for ev in exp_gen:
            if hasattr(ev, 'data') and isinstance(ev.data, str):
                explanation_text = ev.data
        
        if not explanation_text:
            explanation_text = "Analysis completed."
            
        print(f"Agent Explainer: {explanation_text}\n")
        
        # Save to explanation log
        explanation_results.append({
            "Date": date_str,
            "Ground_Truth": gt,
            "Llama": pred_llama,
            "GPT": pred_gpt,
            "Mixtral": pred_mixtral,
            "Aggregated_Prediction": final_agg,
            "Explanation": explanation_text
        })
        
    print("\nSimulation Complete!")
    # Save the explanations to CSV
    exp_df = pd.DataFrame(explanation_results)
    exp_csv_path = os.path.join(base_dir, "7_day_simulation_analysis.csv")
    exp_df.to_csv(exp_csv_path, index=False)
    print(f"Explanations securely saved to {exp_csv_path}")
    print("Graph securely rendered to moe_regimes.png")

if __name__ == '__main__':
    asyncio.run(run_simulation())
