"""Canonical MoE-F Level-3 pipeline builder — the single source of the pipeline.

Both entrypoints import from here (Phase 3 consolidation, per
`docs/memory/0003-import-dont-copy.md`):

- `model_library/agentic_ai/moe_coordinator/agent.py` (the ADK app)
- `interface/cli/agent.py` (the CLI twin)

`build_moef_level_3_system(artifact_dir=...)` returns the assembled
`SequentialAgent`; each entrypoint passes its own artifact directory so the
rolling-window chart / history CSV land next to that entrypoint.
"""

import os

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless: no display in agent/CI environments
import matplotlib.pyplot as plt

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

from model_library.agentic_ai.experts import build_moe_parallel_swarm
from model_library.agentic_ai.model_registry import get_model
from model_library.ml_zoo.filters import robust_gibbs_aggregation_tool
from model_library.technical.indicators import enrich_ohlcv_data, apply_semantic_news_filter

DEFAULT_ARTIFACT_DIR = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Phase 4 plotting helper (shared; used as a tool and directly by final_test)
# ---------------------------------------------------------------------------
def render_moe_trajectories(state, artifact_dir: str = DEFAULT_ARTIFACT_DIR) -> str:
    """Append the latest prediction to history and render the 7-day rolling chart.

    Reproduces the paper's Figure 1: true market trajectory (black) vs. the
    MoE-F filtered trajectory (green dashed, 7-day rolling mean).
    """
    if isinstance(state, (int, float)):
        y_final = float(state)
        y_true = 0.5
    elif hasattr(state, "get"):
        y_final = state.get("final_prediction", 0.5)
        y_true = state.get("current_ground_truth", 0.5)
    else:
        try:
            y_final = float(state)
        except (ValueError, TypeError):
            y_final = 0.5
        y_true = 0.5

    history_file = os.path.join(artifact_dir, "moe_history.csv")

    try:
        if os.path.exists(history_file):
            df_hist = pd.read_csv(history_file)
        else:
            df_hist = pd.DataFrame(columns=["Turn", "y_true", "moef_prediction"])

        turn_index = len(df_hist)
        new_row = {
            "Turn": turn_index,
            "y_true": y_true,
            "moef_prediction": y_final,
        }
        df_hist = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)
        df_hist.to_csv(history_file, index=False)

        if len(df_hist) < 7:
            return "Not enough data for 7-day rolling window. Accumulating predictions."

        df_hist["rolling_moe"] = df_hist["moef_prediction"].rolling(window=7).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(
            df_hist["Turn"],
            df_hist["y_true"],
            color="black",
            label="True Market Trajectory (Ground Truth)",
            linewidth=2,
        )
        plt.plot(
            df_hist["Turn"],
            df_hist["rolling_moe"],
            color="green",
            linestyle="--",
            label="MoE-F Filtered Trajectory (7-Day)",
            linewidth=2,
        )
        plt.yticks([0.0, 0.5, 1.0], ["Bearish (0.0)", "Neutral (0.5)", "Bullish (1.0)"])
        plt.xlabel("Trading Days")
        plt.ylabel("Market Movement Direction / Regime")
        plt.title("MoE-F 7-Day Rolling Trajectory vs Ground Truth (SPY)")
        plt.legend()
        plt.grid(True)

        chart_path = os.path.join(artifact_dir, "moe_regimes.png")
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()
        return f"Chart rendered with 7-day rolling window at {chart_path}."
    except Exception as e:  # noqa: BLE001 - surface the error back to the agent
        return f"Plotting failed: {e}"


def data_ingestion_stub() -> str:
    """Return the OHLCV CSV path the pipeline operates on.

    [STUB] Phase 3 still stubs ingestion; wiring the live Yahoo Finance MCP
    (`data_prep/connectors/market_data`) replaces this.
    """
    return "assets/SPY_10y.csv"


def build_moef_level_3_system(artifact_dir: str = DEFAULT_ARTIFACT_DIR) -> SequentialAgent:
    """Assemble the full MoE-F Level-3 pipeline. Artifacts land under `artifact_dir`."""
    glue = get_model("orchestration")

    # Phase 1: ingestion pipeline
    market_extractor = LlmAgent(
        name="MarketDataExtractor",
        model=glue,
        instruction=(
            "Use the data_ingestion_stub tool to extract 10 years of OHLCV "
            "historical data and news for the SPY ticker. Emit the CSV path."
        ),
        tools=[FunctionTool(func=data_ingestion_stub)],
        output_key="structured_market_data",
    )
    quantitative_feature_agent = LlmAgent(
        name="QuantitativeFeatureAgent",
        model=glue,
        instruction=(
            "Take the output from {structured_market_data} (a CSV path) and use "
            "the technical_indicators tool to calculate the MoE-F indicators "
            "(MACD, Bollinger, RSI, CCI, DX, SMAs). Emit the enriched CSV path."
        ),
        tools=[FunctionTool(func=enrich_ohlcv_data)],
        output_key="enriched_market_data",
    )
    sbert_news_filter = LlmAgent(
        name="SBERT_SemanticFilter",
        model=glue,
        instruction=(
            "Apply semantic similarity to the news for {enriched_market_data} by "
            "calling the apply_semantic_news_filter tool. Discard noise below the 0.2 "
            "threshold; output the high-signal news chunks returned by the tool."
        ),
        tools=[FunctionTool(func=apply_semantic_news_filter)],
        output_key="filtered_news_context",
    )
    market_data_pipeline = SequentialAgent(
        name="NIFTY_Ingestion_Pipeline",
        sub_agents=[market_extractor, quantitative_feature_agent, sbert_news_filter],
    )

    # Phase 3: synthesizer
    aggregator_agent = LlmAgent(
        name="SynthesizerAgent",
        model=glue,
        instruction=(
            "Execute the robust_gibbs_aggregation tool to combine expert "
            "predictions via the PAC-Bayes Softmin measure and bi-level Q-update."
        ),
        tools=[robust_gibbs_aggregation_tool],
        output_key="synthesized_history_context",
    )

    # Phase 4: plotter (tool closes over the chosen artifact_dir)
    def render(state) -> str:
        return render_moe_trajectories(state, artifact_dir)

    render.__name__ = "render_moe_trajectories"
    render.__doc__ = "Render the MoE-F 7-day rolling trajectory chart."
    plotting_agent = LlmAgent(
        name="PlottingAgent",
        model=glue,
        instruction=(
            "You are the visualization reporter. Trigger the render_moe_trajectories "
            "tool using the history from {synthesized_history_context}."
        ),
        tools=[FunctionTool(func=render)],
        output_key="final_status",
    )

    # Phase 5: master pipeline (fresh swarm each call — ADK agents have one parent)
    return SequentialAgent(
        name="MoEF_Pipeline",
        sub_agents=[
            market_data_pipeline,
            build_moe_parallel_swarm(),
            aggregator_agent,
            plotting_agent,
        ],
    )
