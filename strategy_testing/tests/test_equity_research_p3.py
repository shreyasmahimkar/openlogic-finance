"""P3 tests: agent orchestration (RAG + model) + HITL approval gate. Offline."""

from agentic_workflows.equity_research import agent, tools
from risk_management.governance.approval import (
    make_research_approval_callback,
)


class _Tool:
    def __init__(self, name):
        self.name = name


class _Ctx:
    def __init__(self, state):
        self.state = dict(state)


# ── tools ─────────────────────────────────────────────────────────────────────
def test_predict_regime_tool_on_real_data():
    out = tools.predict_regime("SPY")
    assert "regime:" in out
    assert "P(up)=" in out
    assert any(r in out for r in ("bullish", "neutral", "bearish"))


def test_predict_regime_unknown_ticker():
    assert "No price history" in tools.predict_regime("NOPE")


def test_retrieve_context_still_grounded():
    ctx = tools.retrieve_context("fiscal 2026 revenue guidance")
    assert "[1]" in ctx


# ── HITL approval gate ────────────────────────────────────────────────────────
def test_publish_blocked_without_approval():
    cb = make_research_approval_callback()
    result = cb(_Tool("publish_recommendation"), {"thesis": "x", "rating": "BUY"}, _Ctx({}))
    assert result is not None
    assert result["status"] == "PENDING_HUMAN_APPROVAL"


def test_publish_allowed_after_approval():
    cb = make_research_approval_callback()
    assert cb(_Tool("publish_recommendation"), {}, _Ctx({"human_approved": True})) is None


def test_non_consequential_tools_not_gated():
    cb = make_research_approval_callback()
    assert cb(_Tool("retrieve_context"), {"query": "x"}, _Ctx({})) is None
    assert cb(_Tool("predict_regime"), {"ticker": "SPY"}, _Ctx({})) is None


# ── agent wiring ──────────────────────────────────────────────────────────────
def test_agent_has_three_tools_and_callback():
    a = agent.build_equity_research_agent()
    tool_names = {t.name for t in a.tools}
    assert {"retrieve_context", "predict_regime", "publish_recommendation"} <= tool_names
    assert a.before_tool_callback is not None
