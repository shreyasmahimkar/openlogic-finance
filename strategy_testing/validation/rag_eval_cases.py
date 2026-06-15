"""Labeled RAG eval cases (Box 3) — questions with a gold source + good/bad answers.

A small benchmark for the equity-research RAG: each case has a question, the
distinctive substring of the **gold** source document (for context-recall@k), and
a grounded **good** answer vs. a fabricated **bad** answer (to validate the judge
discriminates faithfulness).
"""

from data_prep.rag.indexing import build_index
from model_library.retrieval.retriever import Retriever

# A self-contained eval corpus (3 companies x 3 topics) so recall@k is non-trivial.
EVAL_CORPUS = [
    {
        "id": "nmbs-guid",
        "source": "NMBS Q4 — CFO guidance",
        "text": "Nimbus guides fiscal 2026 revenue growth of 8 to 10 percent and 120 bps of operating margin expansion.",
    },
    {
        "id": "nmbs-marg",
        "source": "NMBS Q4 — margins",
        "text": "Nimbus gross margin reached 79 percent, up 200 basis points, on data-center efficiency.",
    },
    {
        "id": "nmbs-risk",
        "source": "NMBS Q4 — risk",
        "text": "Nimbus notes a stronger dollar is a 2 to 3 point revenue headwind and higher rates extend sales cycles.",
    },
    {
        "id": "aapl-guid",
        "source": "AAPL Q4 — CFO guidance",
        "text": "Apple expects fiscal 2026 revenue to accelerate on iPhone and services, with record gross margin near 47 percent.",
    },
    {
        "id": "aapl-serv",
        "source": "AAPL Q4 — services",
        "text": "Apple services subscriber base reached 1.1 billion, growing 14 percent year over year.",
    },
    {
        "id": "goog-cloud",
        "source": "GOOG Q4 — cloud",
        "text": "Alphabet Google Cloud backlog grew 30 percent on multi-year enterprise AI commitments.",
    },
    {
        "id": "goog-guid",
        "source": "GOOG Q4 — guidance",
        "text": "Alphabet guides search and services revenue growth of 10 to 12 percent for fiscal 2026.",
    },
    {
        "id": "btc-flows",
        "source": "BTC Q4 — flows",
        "text": "Bitcoin institutional inflows accelerate on spot ETF adoption, targeting a 15 percent increase in holdings.",
    },
]

EVAL_CASES = [
    {
        "question": "What is Nimbus fiscal 2026 revenue guidance?",
        "gold_source_substring": "CFO guidance",
        "good_answer": "Nimbus guided fiscal 2026 revenue growth of 8 to 10 percent [1].",
        "bad_answer": "Nimbus guided 25 percent revenue growth and announced a special dividend.",
    },
    {
        "question": "What drove Nimbus gross margin?",
        "gold_source_substring": "margins",
        "good_answer": "Nimbus gross margin rose to 79 percent driven by data-center efficiency [1].",
        "bad_answer": "Nimbus gross margin collapsed to 40 percent in a price war.",
    },
    {
        "question": "How fast is Apple services growing?",
        "gold_source_substring": "services",
        "good_answer": "Apple services grew 14 percent year over year to 1.1 billion subscribers [1].",
        "bad_answer": "Apple services shrank as subscribers cancelled en masse.",
    },
    {
        "question": "What is Alphabet's cloud backlog trend?",
        "gold_source_substring": "cloud",
        "good_answer": "Alphabet's Google Cloud backlog grew 30 percent on enterprise AI commitments [1].",
        "bad_answer": "Alphabet exited the cloud business this quarter.",
    },
]


def build_eval_retriever(backend: str = "memory") -> Retriever:
    store, embedder = build_index(EVAL_CORPUS, backend=backend)
    return Retriever(store, embedder)
