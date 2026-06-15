"""Scored RAG evaluation (Box 3) — the *evals* the LLM layer was missing.

Moves RAG evaluation from "schema-valid evalset" to **actually scored**:
- **context_recall@k** (deterministic): did retrieval surface the gold document?
- **groundedness** (LM-judge): is a good (grounded) answer scored above a fabricated
  one? Uses a Gemini judge with `GEMINI_API_KEY`, else a deterministic fallback.

A `passes_gate()` makes it a CI quality gate, like the model validation report.
"""

from dataclasses import dataclass, field
from statistics import mean

from .llm_judge import judge_backend, judge_groundedness


@dataclass
class RagEvalReport:
    n: int
    context_recall: float
    grounded_good: float  # mean LM-judge score for the grounded answers
    grounded_bad: float  # mean LM-judge score for the fabricated answers
    judge: str
    per_case: list = field(default_factory=list)

    def passes_gate(self, min_recall: float = 0.7, min_separation: float = 0.2) -> bool:
        """Retrieval finds the gold doc AND the judge separates good from fabricated."""
        return (
            self.context_recall >= min_recall
            and (self.grounded_good - self.grounded_bad) >= min_separation
        )

    def summary(self) -> str:
        verdict = "PASS" if self.passes_gate() else "FAIL"
        return (
            f"[{verdict}] context_recall@k={self.context_recall:.2f} "
            f"grounded(good)={self.grounded_good:.2f} grounded(bad)={self.grounded_bad:.2f} "
            f"judge={self.judge} (n={self.n})"
        )


def evaluate_rag(cases: list[dict], retriever, k: int = 3) -> RagEvalReport:
    """Run retrieval + LM-judge groundedness over labeled cases."""
    recalls, goods, bads, per = [], [], [], []
    for case in cases:
        context = retriever.retrieve(case["question"], k=k).answer_context
        recall = 1.0 if case["gold_source_substring"].lower() in context.lower() else 0.0
        good = judge_groundedness(case["good_answer"], context, case["question"])
        bad = judge_groundedness(case["bad_answer"], context, case["question"])
        recalls.append(recall)
        goods.append(good)
        bads.append(bad)
        per.append({"question": case["question"], "recall": recall, "good": good, "bad": bad})
    return RagEvalReport(
        n=len(cases),
        context_recall=mean(recalls) if recalls else 0.0,
        grounded_good=mean(goods) if goods else 0.0,
        grounded_bad=mean(bads) if bads else 0.0,
        judge=judge_backend(),
        per_case=per,
    )
