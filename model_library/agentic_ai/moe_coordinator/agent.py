"""MoE-F Coordinator — Level 3 Multi-Agent ADK pipeline.

Reconstructed source (Phase 0, "stop the bleeding").
--------------------------------------------------------------------------
The original ``moe_coordinator/agent.py`` was lost: only compiled
``__pycache__/agent.cpython-311.pyc`` survived and the source was never
committed to git. This module reconstructs the canonical pipeline from two
authoritative, surviving sources:

  1. The design spec: ``model_library/agentic_ai/docs/moef_agents_plan.md``
  2. The committed twin: ``interface/cli/agent.py`` (assembles the same
     ``moef_level_3_system`` and is tracked in git).

Per the agentic-engineering plan (``docs/AGENTIC_ENGINEERING_SDLC_PLAN.md``),
all shared math is **imported** from ``model_library`` rather than copied —
the original package vendored its own ``experts.py``/``filters.py``/
``indicators.py``, which is exactly the duplication we are removing.

This is the ADK app referenced by ``model_library/agentic_ai/docs/
ADK_USAGE_GUIDE.md`` (``adk web moe_coordinator``).

NOTE: ``interface/cli/agent.py`` currently duplicates this assembly. Phase 3
(consolidation) will refactor that module to import from here. Tracked as a
follow-up, not done in Phase 0.
"""

import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless: no display in agent/CI environments
import matplotlib.pyplot as plt

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

# Shared building blocks — imported, never copied (see module docstring).
from model_library.agentic_ai.experts import moe_parallel_swarm
from model_library.ml_zoo.filters import robust_gibbs_aggregation_tool
from model_library.technical.indicators import enrich_ohlcv_data

# Artifacts (rolling-window chart + prediction history) are written next to
# this module so the package is self-contained.
ARTIFACT_DIR = os.path.dirname(__file__)

# The default LLM for the orchestration glue agents. Centralizing the model
# choice here is a deliberate first step toward the Phase 8 model registry.
DEFAULT_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Phase 1: Data Ingestion Pipeline (SequentialAgent)
# ---------------------------------------------------------------------------
def data_ingestion_stub() -> str:
    """Return the path to the OHLCV CSV the rest of the pipeline operates on.

    [STUB] Phase 0 keeps the stubbed path from the committed twin. Phase 3
    wires the live Yahoo Finance MCP (``data_prep/connectors/market_data``)
    in its place.
    """
    return "assets/SPY_10y.csv"


market_data_tool = FunctionTool(func=data_ingestion_stub)

market_extractor = LlmAgent(
    name="MarketDataExtractor",
    model=DEFAULT_MODEL,
    instruction=(
        "Use the data_ingestion_stub tool to extract 10 years of OHLCV "
        "historical data and news for the SPY ticker. Structure this data "
        "logically and emit the resulting CSV path."
    ),
    tools=[market_data_tool],
    output_key="structured_market_data",
)

technical_indicators_tool = FunctionTool(func=enrich_ohlcv_data)

quantitative_feature_agent = LlmAgent(
    name="QuantitativeFeatureAgent",
    model=DEFAULT_MODEL,
    instruction=(
        "Take the output from {structured_market_data} (a CSV file path) and "
        "use the technical_indicators_tool to calculate the MoE-F technical "
        "indicators (MACD, Bollinger Bands, RSI, CCI, DX, SMAs). Emit the path "
        "to the enriched CSV."
    ),
    tools=[technical_indicators_tool],
    output_key="enriched_market_data",
)

sbert_news_filter = LlmAgent(
    name="SBERT_SemanticFilter",
    model=DEFAULT_MODEL,
    instruction=(
        "Apply semantic similarity search to the news headlines associated "
        "with {enriched_market_data}. Discard noise scoring below the 0.2 "
        "tf-idf / cosine threshold and output the precise, high-signal news "
        "chunks as filtered context."
    ),
    output_key="filtered_news_context",
)

market_data_pipeline = SequentialAgent(
    name="NIFTY_Ingestion_Pipeline",
    sub_agents=[market_extractor, quantitative_feature_agent, sbert_news_filter],
)


# ---------------------------------------------------------------------------
# Phase 2: The MoE-F Swarm (ParallelAgent) — imported from model_library
# ---------------------------------------------------------------------------
# ``moe_parallel_swarm`` fans out to expert_llama / expert_gpt / expert_mixtral,
# each of which invokes the stochastic_filter_update_tool to maintain its local
# Bayesian belief state (pi) in the shared ADK SessionState.


# ---------------------------------------------------------------------------
# Phase 3: Robust Aggregation (Coordinator Synthesizer)
# ---------------------------------------------------------------------------
aggregator_agent = LlmAgent(
    name="SynthesizerAgent",
    model=DEFAULT_MODEL,
    instruction=(
        "Execute the robust_gibbs_aggregation tool to combine the expert "
        "predictions using the PAC-Bayes Softmin measure and apply the "
        "bi-level Q-matrix update produced by the swarm."
    ),
    tools=[robust_gibbs_aggregation_tool],
    output_key="synthesized_history_context",
)


# ---------------------------------------------------------------------------
# Phase 4: Visualization & Plotting (Reporting Agent)
# ---------------------------------------------------------------------------
def render_moe_trajectories(state) -> str:
    """Append the latest prediction to history and render the 7-day rolling chart.

    Reproduces the paper's Figure 1: true market trajectory (black) vs. the
    MoE-F filtered trajectory (green dashed, 7-day rolling mean).
    """
    y_final = state.get("final_prediction", 0.5)
    history_file = os.path.join(ARTIFACT_DIR, "moe_history.csv")

    try:
        if os.path.exists(history_file):
            df_hist = pd.read_csv(history_file)
        else:
            df_hist = pd.DataFrame(columns=["Turn", "y_true", "moef_prediction"])

        turn_index = len(df_hist)
        new_row = {
            "Turn": turn_index,
            "y_true": state.get("current_ground_truth", 0.5),
            "moef_prediction": y_final,
        }
        df_hist = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)
        df_hist.to_csv(history_file, index=False)

        if len(df_hist) < 7:
            return "Not enough data for 7-day rolling window. Accumulating predictions."

        df_hist["rolling_moe"] = df_hist["moef_prediction"].rolling(window=7).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(
            df_hist["Turn"], df_hist["y_true"],
            color="black", label="True Market Trajectory (Ground Truth)", linewidth=2,
        )
        plt.plot(
            df_hist["Turn"], df_hist["rolling_moe"],
            color="green", linestyle="--", label="MoE-F Filtered Trajectory (7-Day)", linewidth=2,
        )
        plt.yticks([0.0, 0.5, 1.0], ["Bearish (0.0)", "Neutral (0.5)", "Bullish (1.0)"])
        plt.xlabel("Trading Days")
        plt.ylabel("Market Movement Direction / Regime")
        plt.title("MoE-F 7-Day Rolling Trajectory vs Ground Truth (SPY)")
        plt.legend()
        plt.grid(True)

        chart_path = os.path.join(ARTIFACT_DIR, "moe_regimes.png")
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()
        return f"Chart rendered with 7-day rolling window at {chart_path}."
    except Exception as e:  # noqa: BLE001 - surface the error back to the agent
        return f"Plotting failed: {e}"


render_tool = FunctionTool(func=render_moe_trajectories)

plotting_agent = LlmAgent(
    name="PlottingAgent",
    model=DEFAULT_MODEL,
    instruction=(
        "You are the visualization reporter. Trigger the render_moe_trajectories "
        "tool using the history from {synthesized_history_context}."
    ),
    tools=[render_tool],
    output_key="final_status",
)


# ---------------------------------------------------------------------------
# Phase 5: Master Orchestration (SequentialAgent)
# ---------------------------------------------------------------------------
moef_level_3_system = SequentialAgent(
    name="MoEF_Pipeline",
    sub_agents=[market_data_pipeline, moe_parallel_swarm, aggregator_agent, plotting_agent],
)

# ADK entrypoint: ``adk web moe_coordinator`` discovers ``root_agent``.
root_agent = moef_level_3_system
