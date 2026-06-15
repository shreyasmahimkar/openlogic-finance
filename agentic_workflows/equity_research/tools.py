"""Tools for the Equity Research Assistant agent (P3).

Three tools the agent orchestrates:
- `retrieve_context` — RAG over earnings-call transcripts (the qualitative view).
- `predict_regime` — the return/regime model's quantitative signal (the numbers).
- `publish_recommendation` — the consequential action, gated by human approval
  (see `risk_management/governance/approval.py`).
"""

import os

import pandas as pd

from data_prep.rag.indexing import build_index
from model_library.ml_zoo.return_regime import (
    ReturnRegimeModel,
    build_training_frame,
    prob_to_regime,
)
from model_library.retrieval.retriever import Retriever

from .corpus import SAMPLE_TRANSCRIPTS

# Build the transcript index + retriever once.
_store, _embedder = build_index(SAMPLE_TRANSCRIPTS)
_retriever = Retriever(_store, _embedder)

# repo root = up 3 from agentic_workflows/equity_research/tools.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def retrieve_context(query: str) -> str:
    """Retrieve the most relevant cited earnings-call passages for a question.

    Args:
        query: the analyst's natural-language question about the company.

    Returns:
        A numbered, source-cited context block (or NO_CONTEXT_FOUND).
    """
    return _retriever.retrieve(query, k=4).answer_context


def predict_regime(ticker: str = "SPY") -> str:
    """Predict the next-period return regime for a ticker from its price history.

    Args:
        ticker: the asset symbol (e.g. SPY, AAPL, GOOG, BTC).

    Returns:
        The model's regime (bearish/neutral/bullish) + calibrated P(up). This is a
        quantitative model signal, distinct from management's qualitative guidance.
    """
    path = os.path.join(_REPO_ROOT, "assets", f"{ticker}_10y.csv")
    if not os.path.exists(path):
        return f"No price history for {ticker}. Available: SPY, AAPL, GOOG, BTC."
    X, y = build_training_frame(pd.read_csv(path))
    if len(X) < 100:
        return f"Insufficient price history for {ticker}."
    model = ReturnRegimeModel().train(X.iloc[:-1], y.iloc[:-1])
    p = float(model.predict_proba_up(X.iloc[[-1]])[0])
    return (
        f"{ticker} next-{model.horizon}d model regime: {prob_to_regime(p)} "
        f"(P(up)={p:.2f}). Quantitative model signal only — not investment advice."
    )


def publish_recommendation(thesis: str, rating: str) -> str:
    """Publish a rated research note. CONSEQUENTIAL — requires human approval.

    Args:
        thesis: the grounded investment thesis (must cite sources).
        rating: BUY / HOLD / SELL.

    Returns:
        Confirmation of publication (only reached if a human has approved).
    """
    return f"PUBLISHED research note — rating={rating}. Thesis: {thesis[:200]}"
