# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import os

# Streamlit visualization code relocated here.
# This represents Box 5: UI & UX

st.set_page_config(page_title="OpenLogic Finance Dashboard", layout="wide")

st.title("OpenLogic Finance - Box 5: UI & UX")

st.sidebar.header("Controls")
selected_ticker = st.sidebar.selectbox("Select Asset", ["SPY", "BTC"])

st.write(f"### {selected_ticker} Market Analysis")

# Fetch local generated data if available
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
history_file = os.path.join(base_dir, "strategy_testing", "backtesting", "moe_history.csv")
regime_img = os.path.join(base_dir, "strategy_testing", "backtesting", "moe_regimes.png")

if os.path.exists(regime_img):
    st.image(regime_img, caption="MoE-F Mechanism 7-Day Rolling Trajectory")
else:
    st.info("Run strategy tests to generate the MoE trajectory visualization.")

if os.path.exists(history_file):
    st.write("#### Agent Evaluation History")
    df = pd.read_csv(history_file)
    st.dataframe(df.tail(20))
else:
    st.info("No backtest history found.")

st.sidebar.button("Run Simulation", help="Triggers strategy_testing backtest locally.")
