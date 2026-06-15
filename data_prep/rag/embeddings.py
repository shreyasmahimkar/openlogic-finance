"""Embedding provider (Box 1, RAG) — Vertex AI in production, offline fallback.

GCP path: Vertex AI text embeddings (`text-embedding-004`), used automatically
when `GOOGLE_CLOUD_PROJECT` is set and the Vertex SDK is available. Offline path:
a deterministic L2-normalized hashing vectorizer so the equity-research RAG slice
runs and tests pass with no cloud and no keys — identical interface either way.
"""

import os

import numpy as np


class EmbeddingProvider:
    def __init__(self, model: str = "text-embedding-004", dim: int = 512):
        self.model = model
        self.dim = dim
        self._using_vertex = False

    def _embed_vertex(self, texts: list[str]) -> np.ndarray | None:
        if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
            return None
        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel

            vertexai.init(
                project=os.environ["GOOGLE_CLOUD_PROJECT"],
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )
            model = TextEmbeddingModel.from_pretrained(self.model)
            embs = model.get_embeddings(texts)
            self._using_vertex = True
            return np.array([e.values for e in embs], dtype=np.float32)
        except Exception:
            return None  # fall back offline

    def _embed_offline(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import HashingVectorizer

        vec = HashingVectorizer(n_features=self.dim, norm="l2", alternate_sign=False)
        return vec.transform(texts).toarray().astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float array of L2-normalized embeddings."""
        vertex = self._embed_vertex(texts)
        return vertex if vertex is not None else self._embed_offline(texts)

    @property
    def backend(self) -> str:
        return "vertex" if self._using_vertex else "offline-hashing"
