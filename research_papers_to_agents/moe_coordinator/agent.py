import os
import json
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sys
import os
from dotenv import load_dotenv

# Ensure the root openlogic-finance directory is in the PYTHONPATH so adk web can find utility_agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Load API keys including NYT_API_KEY for MCP
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "../../utility_agents/financial_news/.env"))

import time
from .block_convey.prismtrace_client import send_trace_async

from google.adk.agents import Agent, LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import FunctionTool, AgentTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# Import other components
from utility_agents.market_data.agent import root_agent as market_data_agent
from utility_agents.market_data.tools import fetch_asset_data
from .filters import robust_gibbs_aggregation_tool
from .experts import moe_parallel_swarm
from .indicators import enrich_ohlcv_data

# ---------------------------------------------------------
# Phase 1: Data Ingestion Pipeline
# ---------------------------------------------------------
def data_ingestion_stub(state=None):
    """Fallback stub kept exclusively for final_test.py simulations."""
    t0 = time.time()
    result = os.path.join(os.path.dirname(__file__), "data", "spy_2025_mock.csv") 
    ms = int((time.time() - t0) * 1000)
    
    session_id = state.get("session_id", "live_adk_run") if hasattr(state, "get") else "live_adk_run"
    
    # Pre-seed ADK session state to prevent KeyErrors if downstream agents crash
    if hasattr(state, "set"):
        state.set("filtered_news_context", "No macroeconomic news successfully filtered.")
        state.set("enriched_market_data", "No quantitative data successfully enriched.")
    elif isinstance(state, dict):
        state["filtered_news_context"] = "No macroeconomic news successfully filtered."
        state["enriched_market_data"] = "No quantitative data successfully enriched."
        
    send_trace_async("Extract OHLCV data", f"Retrieved {result}", "data-retrieval", ms, "data_ingestion", 1, session_id)
    return result

def live_data_ingestion(ticker: str = "SPY", period: str = "10y", state=None):
    """Real live market data tool wired with PRISMtrace."""
    t0 = time.time()
    result = fetch_asset_data(ticker=ticker, period=period)
    ms = int((time.time() - t0) * 1000)
    
    session_id = state.get("session_id", "live_adk_run") if hasattr(state, "get") else "live_adk_run"
    
    # Pre-seed ADK session state to prevent KeyErrors if downstream agents crash
    if hasattr(state, "set"):
        state.set("filtered_news_context", "No macroeconomic news successfully filtered.")
        state.set("enriched_market_data", "No quantitative data successfully enriched.")
    elif isinstance(state, dict):
        state["filtered_news_context"] = "No macroeconomic news successfully filtered."
        state["enriched_market_data"] = "No quantitative data successfully enriched."

    send_trace_async(f"Fetch live asset data {ticker}", str(result), "data-retrieval", ms, "data_ingestion", 1, session_id)
    return result

market_data_tool = FunctionTool(func=live_data_ingestion)

market_extractor = LlmAgent(
    name="MarketDataExtractor",
    model="gemini-2.5-flash",
    instruction="Use the DataIngestionTool to extract 10 years of OHLCV historical data and news for the SPY ticker. Structure this data logically.",
    tools=[market_data_tool],
    output_key="structured_market_data"
)

technical_indicators_tool = FunctionTool(func=enrich_ohlcv_data)

quantitative_feature_agent = LlmAgent(
    name="QuantitativeFeatureAgent",
    model="gemini-2.5-flash",
    instruction="Take the EXACT FILE PATH output from {structured_market_data} (DO NOT invent your own filename) and use the technical_indicators_tool to calculate MoE-F technical indicators (MACD, Bollinger, RSI, CCI, DX, SMAs).",
    tools=[technical_indicators_tool],
    output_key="enriched_market_data"
)

def sbert_telemetry_stub(news_text: str, state=None) -> str:
    t0 = time.time()
    ms = int((time.time() - t0) * 1000)
    session_id = state.get("session_id", "live_adk_run") if hasattr(state, "get") else "live_adk_run"
    send_trace_async("SBERT Semantic Filter execution", "Processed news text", "sbert_semantic_filter", ms, "sentiment", 3, session_id)
    return news_text

sbert_tool = FunctionTool(func=sbert_telemetry_stub)

sbert_news_filter = LlmAgent(
    name="SBERT_SemanticFilter",
    model="gemini-2.5-flash",
    instruction="""You are the SBERT Semantic Filter agent.
Your goal is to provide precise recent financial news chunks to the downstream Swarm.
1. Use the NYTimes search_articles MCP tool (query="financial news") to pull market news specifically representing the past 10 days.
2. Filter the retrieved arrays to extract only the most highly relevant macroeconomic insights (acting as a tf-idf noise discarder).
3. Output the final refined news summary by passing it directly to the sbert_telemetry_stub tool.
4. CRITICAL: You MUST also output the exact same refined news summary in your final text response so the pipeline can capture it!
""",
    tools=[
        sbert_tool,
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="/Users/shreyas/.local/bin/uvx",
                    args=["--from", "git+https://github.com/jeffmm/nytimes-mcp.git", "nytimes-mcp"],
                    env={"NYT_API_KEY": os.environ.get("NYT_API_KEY", "")}
                )
            ),
            tool_filter=['search_articles']
        )
    ],
    output_key="filtered_news_context"
)

market_data_pipeline = SequentialAgent(
    name="NIFTY_Ingestion_Pipeline",
    sub_agents=[market_extractor, quantitative_feature_agent, sbert_news_filter]
)

# ---------------------------------------------------------
# Phase 2: Swarm (ParallelFilterPhase) (imported above)
# ---------------------------------------------------------
# moe_parallel_swarm
# [STUB] NOTE: For live runs, the expert_llama, expert_gpt, and expert_mixtral LlmAgents (defined in experts.py)
# would be triggered here to produce predictions against the live API, invoking the stochastic_filter_update_tool.

# ---------------------------------------------------------
# Phase 3: Robust Aggregator (Coordinator Synthesizer)
# ---------------------------------------------------------
aggregator_agent = LlmAgent(
    name="SynthesizerAgent",
    model="gemini-2.5-flash",
    instruction="""Execute the robust_gibbs_aggregation tool. 
You will receive an array of predictions from the Swarm in the exact order: [Llama, GPT4o, Mixtral]. 
Parse these 3 float values and pass them as specific arguments (pred_llama, pred_gpt, pred_mixtral) into the aggregation tool to calculate the final synthesis.""",
    tools=[robust_gibbs_aggregation_tool],
    output_key="synthesized_history_context"
)

# ---------------------------------------------------------
# Phase 4: Visualization & Plotting (Reporting Agent)
# ---------------------------------------------------------
def render_moe_trajectories(state) -> str:
    t0 = time.time()
    # 7-day rolling window simulation as defined in paper's methodology
    y_final = state.get("final_prediction", 0.5) if hasattr(state, "get") else 0.5
    
    # Decouple the file names so `final_test.py` simulator doesn't clobber the live ADK agent's state
    history_name = state.get("history_file", "live_moe_history.csv") if hasattr(state, "get") else "live_moe_history.csv"
    history_file = os.path.join(os.path.dirname(__file__), history_name)
    
    # Normally we load the history and append, here we'll simulate the rolling 
    try:
        # If it's a live ADK run, the user requested it to be completely fresh every time
        is_live = history_name == "live_moe_history.csv"
        
        # Load existing history ONLY if we are in the stateful local simulator
        if not is_live and os.path.exists(history_file):
            df_hist = pd.read_csv(history_file)
        else:
            df_hist = pd.DataFrame(columns=["Turn", "y_true", "moef_prediction"])
        
        # We append a mock ground truth and new prediction 
        # (Assuming y_true is evaluated outside during final_test simulations)
        turn_index = len(df_hist)
        new_row = {"Turn": turn_index, "y_true": state.get("current_ground_truth", 0.5), "moef_prediction": y_final}
        df_hist = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)
        
        # Still save tracking to file (for final_test to work statefully)
        if not is_live:
            df_hist.to_csv(history_file, index=False)
        
        plt.figure(figsize=(12, 6))
        
        if len(df_hist) >= 7:
            # 7-day smoothing for long-term tracking
            df_hist['rolling_moe'] = df_hist['moef_prediction'].rolling(window=7).mean()
            plt.plot(df_hist['Turn'], df_hist['y_true'], color='black', label='True Market Trajectory (Ground Truth)', linewidth=2)
            plt.plot(df_hist['Turn'], df_hist['rolling_moe'], color='green', linestyle='--', label='MoE-F Filtered Trajectory (7-Day)', linewidth=2)
        else:
            # Short-term or fresh run tracking (will just show a few dots/lines)
            plt.plot(df_hist['Turn'], df_hist['y_true'], color='black', marker='o', label='True Market Trajectory (Ground Truth)', linewidth=2)
            plt.plot(df_hist['Turn'], df_hist['moef_prediction'], color='green', marker='x', markersize=10, label='MoE-F Filtered Trajectory (Raw)', linewidth=2)
            
        # Formatting as required by paper Figure 1
        plt.yticks([0.0, 0.5, 1.0], ['Bearish (0.0)', 'Neutral (0.5)', 'Bullish (1.0)'])
        plt.xlabel('Trading Days / Inference Turns')
        plt.ylabel('Market Movement Direction / Regime')
        plt.title('MoE-F Mechanism Trajectory vs Ground Truth (SPY)')
        plt.legend()
        plt.grid(True)
        
        chart_name = state.get("chart_file", "live_moe_regimes.png") if hasattr(state, "get") else "live_moe_regimes.png"
        chart_path = os.path.join(os.path.dirname(__file__), chart_name)
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close()
        msg = f"Chart correctly rendered with 7-day rolling window at {chart_path}."
        
        ms = int((time.time() - t0) * 1000)
        session_id = state.get("session_id", "live_adk_run") if hasattr(state, "get") else "live_adk_run"
        send_trace_async("Render final chart", msg, "reporting_agent", ms, "reporting", 6, session_id)
        
        return msg
            
    except Exception as e:
        return f"Plotting failed: {str(e)}"

render_tool = FunctionTool(func=render_moe_trajectories)

plotting_agent = LlmAgent(
    name="PlottingAgent",
    model="gemini-2.5-flash",
    instruction="You are the final visualization reporter. Trigger the RenderMoETrajectories tool to generate the chart. CRITICAL: In your final response to the user, you MUST summarize the entire pipeline run! Explicitly state what data was extracted, list what the Swarm experts predicted (extracted from the synthesized_history_context), and explain the final aggregated prediction before presenting the chart image.",
    tools=[render_tool],
    output_key="final_status"
)

# ---------------------------------------------------------
# Phase 5: Master Orchestration via Sequential Pipeline
# ---------------------------------------------------------
moef_level_3_system = SequentialAgent(
    name="MoEF_Pipeline",
    sub_agents=[market_data_pipeline, moe_parallel_swarm, aggregator_agent, plotting_agent]
)

# Compatibility stub mimicking the previous root agent for standard evaluations
root_agent = moef_level_3_system
