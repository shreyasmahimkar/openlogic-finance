"""Responsible-AI grounding controls (Box 4) for the equity-research agent.

Governance as *code*, not prose: the grounding instruction the agent must follow,
and a deterministic check that an answer is actually grounded in retrieved
transcript passages (no hallucinated guidance numbers). This complements the
trade-veto guardrail in `risk_management/portfolio/guardrail.py`.
"""

import re

GROUNDING_INSTRUCTION = (
    "You are an equity research analyst. For every question:\n"
    "1. Call the retrieve_context tool to fetch passages from the earnings call / filings.\n"
    "2. Answer ONLY from those passages. If they don't contain the answer, say "
    "'I don't have that in the provided transcripts.' — never invent guidance, "
    "numbers, or management quotes.\n"
    "3. Cite the bracketed source numbers you used, e.g. [1][3].\n"
    "Be concise and precise; this supports an investment thesis."
)

# Phrase that signals an honest abstention (allowed, not a grounding failure).
_ABSTENTION = "don't have that in the provided transcripts"


def is_grounded(answer: str, context: str) -> bool:
    """True if the answer cites a source present in the context, or honestly abstains.

    A lightweight guardrail/eval check: ungrounded, uncited answers are a
    responsible-AI failure (material-misstatement risk in research). Real
    deployments add an LM-judge faithfulness score on top.
    """
    if _ABSTENTION in answer.lower():
        return True
    cited = set(re.findall(r"\[(\d+)\]", answer))
    available = set(re.findall(r"\[(\d+)\]", context))
    return bool(cited) and cited.issubset(available)
