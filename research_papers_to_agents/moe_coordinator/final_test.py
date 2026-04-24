import os
import pandas as pd
import numpy as np
import asyncio
import time
import uuid
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

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
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

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
        
    print("Step 1: Fetching 6mo Base Data ONCE (to satisfy 60-day SMA warmup)...")
    raw_response = market_data_tool.func(ticker="SPY", period="6mo", state=SessionState())
    raw_csv = raw_response.get("csv_path", raw_response) if isinstance(raw_response, dict) else str(raw_response)
    
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
    session_service = InMemorySessionService()
    
    try:
        session = await session_service.create_session(app_name="simulation", user_id="test", session_id="test_sbert")
        sbert_runner = Runner(app_name="simulation", agent=sbert_news_filter, session_service=session_service, auto_create_session=False)
        msg_obj = Content(role="user", parts=[Part.from_text(text="Fetch recent financial news for SPY from NYTimes and extract macro insights.")])
        sbert_gen = sbert_runner.run_async(user_id="test", session_id="test_sbert", new_message=msg_obj)
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
        
        # 1. Trigger true parallel swarm via Runner!
        session = await session_service.create_session(app_name="simulation", user_id="test", session_id=turn_id)
        session.state["enriched_market_data"] = data_text
        session.state["filtered_news_context"] = news_context
        
        # Get previous day's ground truth for the filter tool
        prev_gt = 1.0
        if i - start_idx > 0:
            past_row = test_data.iloc[i-1]
            if past_row['Close'] > test_data.iloc[i-2]['Close']:
                prev_gt = 1.0
            else:
                prev_gt = 0.0
                
        swarm_runner = Runner(app_name="simulation", agent=moe_parallel_swarm, session_service=session_service, auto_create_session=False)
        full_user_prompt = f"Analyze {date_str} market direction for tomorrow.\n\nQuantitative Context:\n{data_text}\n\nNews Context:\n{news_context}\n\nYesterday's Actual Ground Truth was {prev_gt}. If calling stochastic_filter_update_tool, use this truth value."
        msg_obj = Content(role="user", parts=[Part.from_text(text=full_user_prompt)])
        swarm_gen = swarm_runner.run_async(
            user_id="test",
            session_id=turn_id,
            new_message=msg_obj
        )
        
        swarm_outputs = []
        async for ev in swarm_gen:
            # We catch any string data emitted from the parallel agents
            # Since ADK ParallelAgent encapsulates sub-agents, we collect raw strings and guess order based on typical resolution
            if hasattr(ev, 'type') and "ResponseEvent" in str(type(ev)):
                 if hasattr(ev, 'data'):
                     swarm_outputs.append((ev.source.name if hasattr(ev, 'source') and hasattr(ev.source, 'name') else str(len(swarm_outputs)), ev.data))
        
        # Handle extraction resiliently by unwrapping ADK types and capturing regex
        pred_llama, pred_gpt, pred_mixtral = 0.5, 0.5, 0.5

        for src, out_data in swarm_outputs:
            import re
            text_str = ""
            if hasattr(out_data, 'text') and getattr(out_data, 'text'):
                text_str = out_data.text
            elif hasattr(out_data, 'parts') and len(out_data.parts) > 0:
                # Handle text or FunctionCall
                part = out_data.parts[0]
                if getattr(part, 'function_call', None):
                    import json
                    try:
                        args = part.function_call.args
                        if 'prediction' in args:
                            text_str = str(args['prediction'])
                        else:
                            text_str = str(args)
                    except:
                        text_str = str(part)
                else:
                    text_str = getattr(part, 'text', str(out_data))
            elif isinstance(out_data, list):
                if len(out_data) >= 3:
                    try: pred_llama = float(''.join(c for c in str(out_data[0]) if c.isdigit() or c=='.'))
                    except: pass
                    try: pred_gpt = float(''.join(c for c in str(out_data[1]) if c.isdigit() or c=='.'))
                    except: pass
                    try: pred_mixtral = float(''.join(c for c in str(out_data[2]) if c.isdigit() or c=='.'))
                    except: pass
                text_str = "" # already parsed
            else:
                text_str = str(out_data)
                
            if text_str:
                try:
                    match = re.search(r"(0\.\d+|1\.0)", text_str)
                    val = float(match.group(1)) if match else 0.5
                except:
                    val = 0.5
                if "Llama" in str(src): pred_llama = val
                elif "GPT" in str(src): pred_gpt = val
                elif "Mixtral" in str(src): pred_mixtral = val
        
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
        
        # Use single runner invocation to generate the explanation text
        explanation_session_id = turn_id + "_explain"
        session_exp = await session_service.create_session(app_name="simulation", user_id="test", session_id=explanation_session_id)
        
        exp_runner = Runner(app_name="simulation", agent=explainer_agent, session_service=session_service, auto_create_session=False)
        msg_obj = Content(role="user", parts=[Part.from_text(text=explain_prompt)])
        exp_gen = exp_runner.run_async(user_id="test", session_id=explanation_session_id, new_message=msg_obj)
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
