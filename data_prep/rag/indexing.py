"""Ingestion (Box 1, RAG): chunk a corpus and build a vector index.

The data-prep half of RAG — turning raw earnings-call transcripts into an
embedded, searchable index. The retrieval half lives in
`model_library/retrieval/retriever.py`.
"""

from .embeddings import EmbeddingProvider
from .vector_store import Document, InMemoryVectorStore


def chunk_text(text: str, max_chars: int = 400, overlap: int = 50) -> list[str]:
    """Naive char-window chunker with overlap (sentence-aware splitters go here in prod)."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def build_index(
    corpus: list[dict],
    embedder: EmbeddingProvider | None = None,
    backend: str = "memory",
    **store_kwargs,
):
    """Chunk + embed + index a corpus of {id, source, text} records.

    Args:
        backend: "memory" (default, offline-safe) or "chroma" (real vector DB).
        store_kwargs: passed to the store (e.g. persist_dir / collection_name for chroma).

    Returns: (store, embedder).
    """
    embedder = embedder or EmbeddingProvider()
    docs: list[Document] = []
    for record in corpus:
        for i, chunk in enumerate(chunk_text(record["text"])):
            docs.append(
                Document(
                    id=f"{record['id']}#{i}",
                    text=chunk,
                    metadata={"source": record.get("source", record["id"])},
                )
            )

    if backend == "chroma":
        from .chroma_store import ChromaVectorStore

        store = ChromaVectorStore(**store_kwargs)
    else:
        store = InMemoryVectorStore()

    store.add(docs, embedder.embed_texts([d.text for d in docs]))
    return store, embedder
