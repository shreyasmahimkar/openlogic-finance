"""Retriever (Box 2) — query embedding + vector search + cited-context formatting.

The retrieval half of RAG: turns an analyst's question into a numbered,
source-cited context block the agent is instructed to answer *only* from. Sits
in the model library alongside the MoE-F experts and (P2) the return/regime model.
"""

from dataclasses import dataclass

from data_prep.rag.embeddings import EmbeddingProvider
from data_prep.rag.vector_store import Hit, InMemoryVectorStore


@dataclass
class RagResult:
    answer_context: str
    hits: list[Hit]


class Retriever:
    def __init__(self, store: InMemoryVectorStore, embedder: EmbeddingProvider):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 4) -> RagResult:
        q_emb = self.embedder.embed_texts([query])[0]
        hits = self.store.search(q_emb, k=k)
        return RagResult(answer_context=self.format_context(hits), hits=hits)

    @staticmethod
    def format_context(hits: list[Hit]) -> str:
        """Render retrieved chunks as a numbered, cited context block for the LLM."""
        if not hits:
            return "NO_CONTEXT_FOUND"
        lines = []
        for n, hit in enumerate(hits, 1):
            src = hit.document.metadata.get("source", hit.document.id)
            lines.append(f"[{n}] (source: {src}) {hit.document.text}")
        return "\n".join(lines)
