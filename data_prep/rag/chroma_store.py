"""Chroma vector store (Box 1, RAG) — a real vector database backend.

Implements the same `add` / `search` interface as `InMemoryVectorStore`, backed
by **ChromaDB** (HNSW, cosine). We pass precomputed embeddings (from
`EmbeddingProvider`) so the same vectors power both backends. Use a `persist_dir`
for an on-disk DB; otherwise it's ephemeral (in-process).

In production this swaps for **Vertex AI Vector Search** or **pgvector** with the
same surface — see `docs/RAG.md`.
"""

from .vector_store import Document, Hit

_COLLECTION = "openlogic_rag_docs"


class ChromaVectorStore:
    def __init__(self, collection_name: str = _COLLECTION, persist_dir: str | None = None):
        import chromadb
        from chromadb.config import Settings

        settings = Settings(anonymized_telemetry=False)
        self._client = (
            chromadb.PersistentClient(path=persist_dir, settings=settings)
            if persist_dir
            else chromadb.EphemeralClient(settings)
        )
        try:  # rebuild a clean collection
            self._client.delete_collection(collection_name)
        except Exception:
            pass
        self._col = self._client.create_collection(
            name=collection_name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def add(self, docs: list[Document], embeddings) -> None:
        self._col.add(
            ids=[d.id for d in docs],
            embeddings=[e.tolist() for e in embeddings],
            documents=[d.text for d in docs],
            metadatas=[{"source": d.metadata.get("source", d.id)} for d in docs],
        )

    def search(self, query_embedding, k: int = 4) -> list[Hit]:
        res = self._col.query(query_embeddings=[query_embedding.ravel().tolist()], n_results=k)
        hits = []
        for text, meta, dist, _id in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0], res["ids"][0]
        ):
            # cosine distance → similarity score
            hits.append(
                Hit(document=Document(id=_id, text=text, metadata=meta), score=1.0 - float(dist))
            )
        return hits

    def __len__(self) -> int:
        return self._col.count()
