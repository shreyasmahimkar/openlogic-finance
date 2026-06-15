# RAG: real embeddings, a real vector DB, and scored evals

Upgrades the equity-research RAG layer from a lexical demo to a **real semantic
stack** with **scored evaluation** — closing the two biggest gaps from the
codebase overview.

## Gap #1 — real semantic RAG

| Piece | Module | What changed |
|---|---|---|
| **Embeddings** | `data_prep/rag/embeddings.py` | Real **Google `text-embedding-004`** via the `google-genai` SDK (auto-used with `GEMINI_API_KEY` / Vertex). Deterministic **hashing fallback** keeps offline/CI working — same interface. |
| **Vector DB** | `data_prep/rag/chroma_store.py` | A real **ChromaDB** backend (HNSW, cosine) with the same `add`/`search` surface as the in-memory store. `build_index(corpus, backend="chroma")`. Swaps to Vertex Vector Search / pgvector in prod. |
| **Real documents** | `data_prep/rag/loaders.py` | `load_documents()` ingests real **`.txt` / `.md` / `.pdf`** filings from a folder. A sample 10-K MD&A excerpt lives in `data_prep/rag/sample_filings/`. |

```python
from data_prep.rag.loaders import load_documents
from data_prep.rag.indexing import build_index
from model_library.retrieval.retriever import Retriever

docs = load_documents("data_prep/rag/sample_filings")          # real .txt/.pdf files
store, emb = build_index(docs, backend="chroma")               # real vector DB
print(emb.backend)                                             # genai:text-embedding-004 (with key) or offline-hashing
print(Retriever(store, emb).retrieve("fiscal 2026 outlook", k=3).answer_context)
```

> Install the stack: `uv sync --extra rag` (adds chromadb + pypdf). Set
> `GEMINI_API_KEY` for real embeddings; otherwise the offline fallback is used.
>
> **Live agent:** the Equity Research Assistant uses the in-memory backend by
> default (fast, offline); set `OPENLOGIC_RAG_BACKEND=chroma` to run it on the real
> Chroma vector DB (`adk run agentic_workflows/equity_research`).

## Gap #2 — scored LM-judge RAG eval

Evaluation moves from "schema-valid evalset" to **actually scored**:

| Metric | How | Module |
|---|---|---|
| **context_recall@k** | deterministic — did retrieval surface the gold doc? | `strategy_testing/validation/rag_eval.py` |
| **groundedness** | **LM-as-judge** (Gemini with `GEMINI_API_KEY`) scoring how well an answer is supported by context; deterministic heuristic fallback for CI | `strategy_testing/validation/llm_judge.py` |
| **gate** | `passes_gate()` — recall above floor AND judge separates faithful from fabricated answers | same |

Result on the labeled benchmark (offline heuristic judge):

```
[PASS] context_recall@k=1.00 grounded(good)=0.94 grounded(bad)=0.23 judge=heuristic (n=4)
```

The judge cleanly separates a grounded answer (0.94) from a fabricated one (0.23).
With `GEMINI_API_KEY` set, `judge=llm:gemini-2.5-flash` and the same harness runs a
real LLM-judged faithfulness eval. Labeled cases: `rag_eval_cases.py`.

```python
from strategy_testing.validation.rag_eval import evaluate_rag
from strategy_testing.validation.rag_eval_cases import EVAL_CASES, build_eval_retriever
print(evaluate_rag(EVAL_CASES, build_eval_retriever(), k=3).summary())
```

## Why it matters
This converts the two honest caveats from the codebase overview — "lexical
fallback, no real vector DB" and "evals are schema-valid but not scored" — into
real, tested capabilities: semantic embeddings, a real vector database, real
document ingestion, and a faithfulness eval with an LLM judge.
