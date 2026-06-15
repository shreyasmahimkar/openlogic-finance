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

from collections import defaultdict

# Vector-DB backend: "memory" (default, fast/offline) or "chroma" (real vector DB).
# Embeddings auto-upgrade to Google text-embedding-004 when GEMINI_API_KEY is set.
_RAG_BACKEND = os.environ.get("OPENLOGIC_RAG_BACKEND", "memory")

# Group transcripts by ticker to build ticker-specific indices.
_by_ticker = defaultdict(list)
for record in SAMPLE_TRANSCRIPTS:
    _by_ticker[record.get("ticker", "NMBS")].append(record)

# Build a base index first to get the embedder (shared across ticker indices).
_default_store, _embedder = build_index(
    SAMPLE_TRANSCRIPTS, backend=_RAG_BACKEND, collection_name="eqr_global"
)

# Build a retriever per ticker (unique Chroma collection per ticker when chroma).
_retrievers = {}
for t, records in _by_ticker.items():
    store, _ = build_index(
        records, _embedder, backend=_RAG_BACKEND, collection_name=f"eqr_{t.lower()}"
    )
    _retrievers[t] = Retriever(store, _embedder)
_retrievers["GLOBAL"] = Retriever(_default_store, _embedder)

# repo root = up 3 from agentic_workflows/equity_research/tools.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def retrieve_context(query: str, ticker: str = "") -> str:
    """Retrieve the most relevant cited earnings-call passages for a question.

    Args:
        query: the analyst's natural-language question about the company.
        ticker: the optional asset symbol (e.g. SPY, AAPL, GOOG, BTC, NMBS).

    Returns:
        A numbered, source-cited context block (or NO_CONTEXT_FOUND).
    """
    # Normalize ticker
    t = (ticker or "").upper().strip()

    # If no ticker is passed, try to detect it from the query text
    if not t:
        for possible_t in _retrievers.keys():
            if possible_t != "GLOBAL" and possible_t in query.upper():
                t = possible_t
                break

    # Fallback to NMBS if the ticker is empty or not in our retrievers
    if not t or t not in _retrievers:
        t = "NMBS"

    return _retrievers[t].retrieve(query, k=4).answer_context


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
