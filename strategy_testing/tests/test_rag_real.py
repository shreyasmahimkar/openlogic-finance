"""Gap #1: real vector DB (Chroma) + real embeddings + document loading. Offline."""

import numpy as np

from data_prep.rag.embeddings import EmbeddingProvider
from data_prep.rag.indexing import build_index
from data_prep.rag.loaders import load_documents
from model_library.retrieval.retriever import Retriever

CORPUS = [
    {
        "id": "guid",
        "source": "NMBS CFO",
        "text": "Nimbus guides fiscal 2026 revenue growth of 8 to 10 percent.",
    },
    {
        "id": "marg",
        "source": "NMBS margins",
        "text": "Gross margin reached 79 percent on data-center efficiency.",
    },
    {
        "id": "risk",
        "source": "NMBS risk",
        "text": "A stronger dollar is a 2 to 3 point revenue headwind.",
    },
]


def test_embeddings_fallback_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    emb = EmbeddingProvider(dim=256)
    vecs = emb.embed_texts(["revenue guidance", "gross margin"])
    assert vecs.shape == (2, 256)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert emb.backend == "offline-hashing"


def test_chroma_vector_db_backend():
    store, embedder = build_index(CORPUS, backend="chroma", collection_name="test_rag_real")
    assert len(store) == 3
    retriever = Retriever(store, embedder)
    result = retriever.retrieve("What is the revenue guidance?", k=1)
    assert "[1]" in result.answer_context
    assert result.hits[0].score is not None  # cosine similarity from Chroma


def test_chroma_and_memory_agree_on_top_hit():
    mem, emb = build_index(CORPUS, backend="memory")
    chr, _ = build_index(CORPUS, backend="chroma", collection_name="test_rag_agree")
    q = "gross margin efficiency"
    top_mem = Retriever(mem, emb).retrieve(q, k=1).hits[0].document.metadata["source"]
    top_chr = Retriever(chr, emb).retrieve(q, k=1).hits[0].document.metadata["source"]
    assert top_mem == top_chr  # same embeddings → same nearest neighbor


def test_document_loader_reads_txt(tmp_path):
    (tmp_path / "a.txt").write_text("Nimbus fiscal 2026 outlook: revenue up 8 to 10 percent.")
    (tmp_path / "skip.csv").write_text("ignore,me")
    records = load_documents(str(tmp_path), source_prefix="NMBS/")
    assert len(records) == 1
    assert records[0]["source"] == "NMBS/a.txt"
    assert "fiscal 2026" in records[0]["text"]
