"""LM-as-judge scorer (Box 3, eval) — a Gemini judge with a deterministic fallback.

`judge_groundedness` scores 0..1 how well an answer is supported by retrieved
context (RAG faithfulness). With `GEMINI_API_KEY` set it uses a real **LLM judge**
(gemini-2.5-flash); otherwise a deterministic heuristic (citation validity +
lexical overlap) so the eval still produces scores in CI without keys.
"""

import os
import re


def _parse_score(text: str | None) -> float:
    match = re.search(r"\d*\.?\d+", text or "")
    if not match:
        return 0.0
    return max(0.0, min(1.0, float(match.group(0))))


def _heuristic_groundedness(answer: str, context: str) -> float:
    """Deterministic proxy: cited sources must exist + claims should overlap context."""
    cited = set(re.findall(r"\[(\d+)\]", answer))
    available = set(re.findall(r"\[(\d+)\]", context))
    cite_ok = bool(cited) and cited.issubset(available)
    answer_terms = set(re.findall(r"[a-z]{4,}", answer.lower()))
    context_terms = set(re.findall(r"[a-z]{4,}", context.lower()))
    overlap = len(answer_terms & context_terms) / max(len(answer_terms), 1)
    return round(0.5 * float(cite_ok) + 0.5 * min(overlap, 1.0), 3)


def judge_groundedness(answer: str, context: str, question: str = "") -> float:
    """Score 0..1 how fully `answer` is supported by `context` (RAG faithfulness)."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        try:
            from google import genai

            client = genai.Client(api_key=key)
            prompt = (
                "You are a strict RAG faithfulness judge. Score from 0.0 to 1.0 how fully "
                "the ANSWER is supported by the CONTEXT (1.0 = every claim supported and "
                "correctly cited; 0.0 = fabricated/contradicted). Return ONLY the number.\n\n"
                f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
            )
            resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return _parse_score(resp.text)
        except Exception:
            pass
    return _heuristic_groundedness(answer, context)


def judge_backend() -> str:
    return "llm:gemini-2.5-flash" if os.environ.get("GEMINI_API_KEY") else "heuristic"
