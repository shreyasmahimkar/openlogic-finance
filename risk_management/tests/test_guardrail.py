"""Unit tests for the Box 4 risk-veto guardrail (no API keys needed)."""

from google.adk.agents import LlmAgent

from risk_management.portfolio.guardrail import (
    RiskLimits,
    evaluate_trade,
    make_risk_veto_callback,
)


class _FakeTool:
    def __init__(self, name):
        self.name = name


class _FakeContext:
    """Mimics ADK ToolContext: a `.state` that behaves like a dict."""

    def __init__(self, state):
        self.state = dict(state)


# ── pure decision ──────────────────────────────────────────────────────────────
def test_evaluate_allows_within_limit():
    state = {"portfolio_peak_value": 100000, "portfolio_current_value": 90000}
    vetoed, _ = evaluate_trade(state, RiskLimits(max_drawdown_pct=0.15))
    assert vetoed is False


def test_evaluate_vetoes_on_breach():
    state = {"portfolio_peak_value": 100000, "portfolio_current_value": 80000}
    vetoed, reason = evaluate_trade(state, RiskLimits(max_drawdown_pct=0.15))
    assert vetoed is True
    assert "limit" in reason


def test_evaluate_allows_without_telemetry():
    # No peak/current yet → nothing to breach against → allow.
    vetoed, _ = evaluate_trade({}, RiskLimits())
    assert vetoed is False


def test_evaluate_vetoes_when_already_halted():
    vetoed, reason = evaluate_trade({"risk_halted": True}, RiskLimits())
    assert vetoed is True
    assert "halted" in reason.lower()


# ── callback behavior ──────────────────────────────────────────────────────────
def test_callback_ignores_non_trade_tool():
    cb = make_risk_veto_callback()
    ctx = _FakeContext({"portfolio_peak_value": 100000, "portfolio_current_value": 50000})
    # A read-only tool must never be vetoed, even deep in a drawdown.
    assert cb(_FakeTool("read_market_indicators"), {"csv_path": "x.csv"}, ctx) is None


def test_callback_allows_trade_within_limit():
    cb = make_risk_veto_callback(RiskLimits(max_drawdown_pct=0.15))
    ctx = _FakeContext({"portfolio_peak_value": 100000, "portfolio_current_value": 92000})
    assert cb(_FakeTool("place_order"), {"side": "buy", "symbol": "SPY"}, ctx) is None
    assert ctx.state.get("risk_halted") is not True


def test_callback_vetoes_trade_on_breach_and_latches_halt():
    cb = make_risk_veto_callback(RiskLimits(max_drawdown_pct=0.15))
    ctx = _FakeContext({"portfolio_peak_value": 100000, "portfolio_current_value": 80000})
    result = cb(_FakeTool("place_order"), {"side": "buy", "symbol": "SPY"}, ctx)
    assert result is not None
    assert result["status"] == "VETOED"
    assert ctx.state["risk_halted"] is True
    # Once halted, even a healthy-looking later trade stays blocked.
    healthy = _FakeContext({**ctx.state, "portfolio_current_value": 100000})
    assert cb(_FakeTool("place_order"), {"side": "buy"}, healthy)["status"] == "VETOED"


def test_callback_detects_trade_by_args_shape():
    cb = make_risk_veto_callback()
    ctx = _FakeContext({"portfolio_peak_value": 100000, "portfolio_current_value": 70000})
    # Unknown tool name, but order-shaped args → treated as a trade and vetoed.
    result = cb(_FakeTool("custom_exec"), {"side": "sell", "quantity": 10}, ctx)
    assert result is not None and result["status"] == "VETOED"


def test_callback_attaches_to_llm_agent():
    """The guardrail must be a valid ADK before_tool_callback."""
    agent = LlmAgent(
        name="execution_agent",
        model="gemini-2.5-flash",
        instruction="Place orders.",
        before_tool_callback=make_risk_veto_callback(),
    )
    assert agent.before_tool_callback is not None
