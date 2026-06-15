"""Equity Research Assistant — Streamlit console (Box 6, P5).

A stakeholder-facing console that ties the whole vertical slice together:
ask → **retrieve** cited earnings-call evidence (RAG) → **predict** the model
regime → draft a rated note → **human approval** (HITL) → publish, with a
**governance audit trail**. Runs offline (retrieval + model need no API key).

    streamlit run interface/streamlit/equity_research_app.py
"""

import os
import sys

import streamlit as st

# Ensure the repo root is importable.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from agentic_workflows.equity_research import tools  # noqa: E402
from risk_management.governance.approval import (  # noqa: E402
    STATE_APPROVED,
    make_research_approval_callback,
)
from risk_management.governance.audit import AuditLog  # noqa: E402

st.set_page_config(page_title="Equity Research Assistant", page_icon="📈", layout="wide")

if "audit" not in st.session_state:
    st.session_state.audit = AuditLog()  # in-memory governance log for this session

st.title("📈 Equity Research Assistant")
st.caption(
    "Grounded RAG over earnings calls + a quantitative regime model, with "
    "human-in-the-loop governance. OpenLogic Finance · Google ADK."
)

with st.sidebar:
    st.header("Research request")
    ticker = st.selectbox("Ticker", ["SPY", "AAPL", "GOOG", "BTC"])
    question = st.text_input(
        "Question", "What did management guide for next year, and what is the model signal?"
    )
    if st.button("Run research", type="primary"):
        st.session_state.context = tools.retrieve_context(question, ticker=ticker)
        st.session_state.audit.record("retrieval", ticker, f"q={question[:60]}")
        st.session_state.regime = tools.predict_regime(ticker)
        st.session_state.audit.record("prediction", ticker, st.session_state.regime[:80])
        st.session_state.ticker = ticker

if "context" in st.session_state:
    left, right = st.columns(2)
    with left:
        st.subheader("📑 Earnings-call evidence (RAG, cited)")
        st.text(st.session_state.context)
    with right:
        st.subheader("📊 Quantitative model signal")
        st.info(st.session_state.regime)

    st.divider()
    st.subheader("📝 Draft research note")
    rating = st.selectbox("Rating", ["BUY", "HOLD", "SELL"])
    thesis = st.text_area(
        "Thesis (cite transcript sources, e.g. [1][2])",
        "Management guidance [1] and the model regime support the rating below.",
    )
    approved = st.checkbox("✅ Analyst approval (human-in-the-loop sign-off)")

    if st.button("Publish recommendation"):
        # Same governance callback the agent uses — approval is enforced in code.
        callback = make_research_approval_callback()
        tool = type("Tool", (), {"name": "publish_recommendation"})()
        ctx = type("Ctx", (), {"state": {STATE_APPROVED: approved}})()
        veto = callback(tool, {"thesis": thesis, "rating": rating}, ctx)
        if veto is not None:
            st.error(f"🚫 {veto['status']}: {veto['reason']}")
        else:
            msg = tools.publish_recommendation(thesis, rating)
            tkr = st.session_state.get("ticker", ticker)
            st.session_state.audit.record("recommendation", tkr, f"rating={rating}")
            st.session_state.audit.record("approval", tkr, f"rating={rating}", actor="analyst")
            st.success(f"✅ {msg}")

st.divider()
st.subheader("🔒 Governance audit trail")
st.caption("Every retrieval, model score, recommendation, and approval is logged (SQL).")
st.dataframe(st.session_state.audit.query(), width="stretch")
