"""Vector store (Box 1, RAG) — in-memory cosine search for the example.

Production options (same `add` / `search` interface):
- **Vertex AI Vector Search** (managed ANN on GCP) — recommended cloud path.
- **pgvector** (Postgres) — SQL-native.
- **Chroma / FAISS** — local/dev.

Embeddings are L2-normalized, so cosine similarity is a dot product.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Hit:
    document: Document
    score: float


class InMemoryVectorStore:
    def __init__(self):
        self._docs: list[Document] = []
        self._matrix: np.ndarray | None = None

    def add(self, docs: list[Document], embeddings: np.ndarray) -> None:
        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings length mismatch")
        self._docs.extend(docs)
        self._matrix = embeddings if self._matrix is None else np.vstack([self._matrix, embeddings])

    def search(self, query_embedding: np.ndarray, k: int = 4) -> list[Hit]:
        if self._matrix is None or not self._docs:
            return []
        scores = self._matrix @ query_embedding.ravel()  # cosine (vectors are L2-normalized)
        top = np.argsort(scores)[::-1][:k]
        return [Hit(document=self._docs[i], score=float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self._docs)
