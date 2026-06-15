"""Embedding provider (Box 1, RAG) — real semantic embeddings with an offline fallback.

Backends, in priority order:
  1. **Google `text-embedding-004`** via the `google-genai` SDK — real semantic
     embeddings, used automatically when `GEMINI_API_KEY` is set (or Vertex via
     `GOOGLE_GENAI_USE_VERTEXAI=TRUE`). This is the production path.
  2. **Offline hashing vectorizer** — deterministic, L2-normalized, lexical. Needs
     no cloud/keys so the RAG slice + tests run anywhere (CI). Same interface.

All vectors are L2-normalized, so cosine similarity is a dot product.
"""

import os

import numpy as np

_GENAI_MODEL = "text-embedding-004"


class EmbeddingProvider:
    def __init__(self, model: str = _GENAI_MODEL, dim: int = 512, prefer_genai: bool = True):
        self.model = model
        self.dim = dim
        self.prefer_genai = prefer_genai
        self._backend = "offline-hashing"

    def _embed_genai(self, texts: list[str]) -> np.ndarray | None:
        if not (self.prefer_genai and os.environ.get("GEMINI_API_KEY")):
            return None
        try:
            from google import genai

            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            resp = client.models.embed_content(model=self.model, contents=texts)
            vecs = np.array([e.values for e in resp.embeddings], dtype=np.float32)
            # L2-normalize so cosine == dot product across backends.
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            self._backend = f"genai:{self.model}"
            return vecs / np.clip(norms, 1e-8, None)
        except Exception:
            return None  # fall back offline

    def _embed_offline(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import HashingVectorizer

        vec = HashingVectorizer(n_features=self.dim, norm="l2", alternate_sign=False)
        self._backend = "offline-hashing"
        return vec.transform(texts).toarray().astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float array of L2-normalized embeddings."""
        genai_vecs = self._embed_genai(texts)
        return genai_vecs if genai_vecs is not None else self._embed_offline(texts)

    @property
    def backend(self) -> str:
        return self._backend
