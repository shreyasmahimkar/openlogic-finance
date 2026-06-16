"""Agent handoff-contract tests (Box 3). Offline — no model calls, no keys.

Covers the reusable analyzer + regression tests locking in the two contract fixes
found by an external handoff review (the MarketDataExtractor completeness gap and
the SBERT referential-integrity gap).
"""

from model_library.agentic_ai.coordinator import build_moef_level_3_system
from strategy_testing.validation.agent_contracts import _flatten, analyze_pipeline


def _agents_by_name(root) -> dict:
    return {a.name: a for a in _flatten(root, [])}


# ── the reusable analyzer ─────────────────────────────────────────────────────
def test_moef_pipeline_has_no_dangling_references():
    report = analyze_pipeline(build_moef_level_3_system())
    assert report.ok(), f"dangling handoffs: {[i.detail for i in report.dangling]}"
    # every {placeholder} an agent consumes is produced by an upstream output_key
    assert "structured_market_data" in report.produced
    assert "enriched_market_data" in report.produced
    assert "filtered_news_context" in report.produced


def test_analyzer_catches_a_dangling_reference():
    """A synthetic pipeline that references a key nobody produces must be flagged."""
    from google.adk.agents import LlmAgent, SequentialAgent

    bad = SequentialAgent(
        name="bad",
        sub_agents=[
            LlmAgent(
                name="A",
                model="gemini-2.5-flash",
                instruction="Emit a path.",
                output_key="data_path",
            ),
            LlmAgent(
                name="B", model="gemini-2.5-flash", instruction="Use {nonexistent_key} to do work."
            ),
        ],
    )
    report = analyze_pipeline(bad)
    assert not report.ok()
    assert any("nonexistent_key" in i.detail for i in report.dangling)


# ── regression: the two fixes from the external review ────────────────────────
def test_fix1_extractor_does_not_overpromise_news():
    extractor = _agents_by_name(build_moef_level_3_system())["MarketDataExtractor"]
    # It resolves the OHLCV dataset path only; it must not claim to fetch news.
    assert "and news" not in extractor.instruction.lower()
    assert "market-data path" in extractor.instruction.lower()


def test_fix2_sbert_references_raw_dataset_not_enriched():
    sbert = _agents_by_name(build_moef_level_3_system())["SBERT_SemanticFilter"]
    # News filtering uses the raw dataset path to locate news, not the indicator CSV.
    assert "{structured_market_data}" in sbert.instruction
    assert "{enriched_market_data}" not in sbert.instruction
