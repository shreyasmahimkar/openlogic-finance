"""Gap #2: scored LM-judge RAG eval. Offline (heuristic judge), no keys."""

from strategy_testing.validation.llm_judge import judge_backend, judge_groundedness
from strategy_testing.validation.rag_eval import evaluate_rag
from strategy_testing.validation.rag_eval_cases import EVAL_CASES, build_eval_retriever


def test_scored_rag_eval_passes_gate():
    report = evaluate_rag(EVAL_CASES, build_eval_retriever(), k=3)
    assert report.n == len(EVAL_CASES)
    assert report.context_recall >= 0.7  # retrieval surfaces the gold doc
    assert report.grounded_good > report.grounded_bad  # judge separates faithful vs fabricated
    assert report.passes_gate()
    assert "PASS" in report.summary()


def test_judge_scores_grounded_above_fabricated():
    context = "[1] (source: NMBS CFO) Nimbus guided fiscal 2026 revenue growth of 8 to 10 percent."
    good = judge_groundedness("Revenue guidance is 8 to 10 percent growth [1].", context)
    bad = judge_groundedness("Revenue will triple next year.", context)
    assert good > bad
    assert 0.0 <= bad <= 1.0 and 0.0 <= good <= 1.0


def test_judge_backend_is_heuristic_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert judge_backend() == "heuristic"
