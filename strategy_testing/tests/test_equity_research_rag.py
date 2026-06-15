"""Equity-research RAG vertical slice — offline tests across Boxes 1/2/4 + workflow.

No cloud, no API keys: exercises ingestion (data_prep.rag), retrieval
(model_library.retrieval), and grounding governance (risk_management.governance).
"""

import numpy as np

from agentic_workflows.equity_research.corpus import SAMPLE_TRANSCRIPTS
from data_prep.rag.embeddings import EmbeddingProvider
from data_prep.rag.indexing import build_index, chunk_text
from data_prep.rag.vector_store import InMemoryVectorStore
from model_library.retrieval.retriever import Retriever
from risk_management.governance.grounding import is_grounded


def _retriever():
    store, embedder = build_index(SAMPLE_TRANSCRIPTS)
    return Retriever(store, embedder)


def test_chunking_overlaps_and_covers():
    chunks = chunk_text("word " * 300, max_chars=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)


def test_embeddings_shape_and_norm():
    vecs = EmbeddingProvider(dim=256).embed_texts(["revenue guidance", "gross margin"])
    assert vecs.shape == (2, 256)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_retrieval_is_relevant_and_cited():
    result = _retriever().retrieve("What is the fiscal 2026 revenue guidance?", k=2)
    sources = " ".join(h.document.metadata["source"] for h in result.hits)
    assert "Q4 2025" in sources
    assert "[1]" in result.answer_context  # numbered, cited context block


def test_empty_store_returns_no_context():
    r = Retriever(InMemoryVectorStore(), EmbeddingProvider())
    assert r.retrieve("anything", k=3).answer_context == "NO_CONTEXT_FOUND"


def test_grounding_guardrail():
    context = "[1] (source: NMBS Q4 2025) Revenue growth guided 8 to 10 percent."
    assert is_grounded("Guidance is 8-10% revenue growth [1].", context) is True
    assert is_grounded("Guidance is 20% growth.", context) is False  # uncited / fabricated
    assert is_grounded("I don't have that in the provided transcripts.", context) is True


def test_agent_module_imports():
    # root_agent builds when google-adk is installed (Gemini default, no live call).
    from agentic_workflows.equity_research import agent

    assert hasattr(agent, "retrieve_context")
    ctx = agent.retrieve_context("revenue guidance")
    assert "[1]" in ctx
