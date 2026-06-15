"""Risk-veto guardrail (Box 4) — the Risk Auditor as a real ADK callback.

`AGENTS.md` hard rule #1 says the Risk Auditor can **veto any trade-shaped
action**. `auditor.py` holds the drawdown math but only as a backtest simulator;
this module turns that decision into a deterministic ADK `before_tool_callback`
that runs *before* a trade tool executes and can short-circuit it.

Wire it onto any agent that can place orders::

    from risk_management.portfolio.guardrail import make_risk_veto_callback

    execution_agent = LlmAgent(
        ...,
        before_tool_callback=make_risk_veto_callback(),
    )

ADK contract: a `before_tool_callback` that returns a dict **replaces** the tool
result and the real tool never runs (the veto); returning ``None`` lets the tool
proceed.
"""

from dataclasses import dataclass

# Reuse the canonical drawdown math — import, don't copy (docs/memory/0003).
from model_library.technical.signals.sma_crossover_signal import drawdown_breached

# Tool names that move money / open positions and are therefore subject to veto.
TRADE_TOOL_NAMES = frozenset(
    {"place_order", "execute_trade", "submit_order", "place_paper_order", "buy", "sell"}
)

# Session-state keys the guardrail reads to assess current portfolio risk.
STATE_PEAK = "portfolio_peak_value"
STATE_CURRENT = "portfolio_current_value"
STATE_HALTED = "risk_halted"


@dataclass(frozen=True)
class RiskLimits:
    """Risk thresholds (sourced from config/env, never magic numbers in logic)."""

    max_drawdown_pct: float = 0.15


def _is_trade_tool(tool_name: str, args: dict) -> bool:
    """A call is trade-shaped if its tool name is known, or its args look like an order."""
    if tool_name in TRADE_TOOL_NAMES:
        return True
    order_keys = {"quantity", "qty", "shares", "side", "notional", "symbol", "ticker"}
    return "side" in args and bool(order_keys & set(args))


def evaluate_trade(state, limits: RiskLimits) -> tuple[bool, str]:
    """Pure veto decision. Returns (vetoed, reason). Reusable + easy to unit-test."""
    if state.get(STATE_HALTED):
        return True, "Trading is halted: a prior drawdown breach liquidated the book."

    peak = state.get(STATE_PEAK)
    current = state.get(STATE_CURRENT)
    if peak is None or current is None:
        # No risk telemetry yet — fail safe is to allow (nothing to breach against).
        return False, ""

    if drawdown_breached(current, peak, threshold=limits.max_drawdown_pct):
        return (
            True,
            f"Drawdown {((peak - current) / peak):.1%} exceeds the "
            f"{limits.max_drawdown_pct:.0%} limit.",
        )
    return False, ""


def make_risk_veto_callback(limits: RiskLimits | None = None):
    """Build an ADK `before_tool_callback` enforcing the drawdown veto on trade tools."""
    limits = limits or RiskLimits()

    def before_tool_callback(tool, args, tool_context):
        if not _is_trade_tool(getattr(tool, "name", ""), args or {}):
            return None  # not a trade — let it run

        vetoed, reason = evaluate_trade(tool_context.state, limits)
        if not vetoed:
            return None  # within limits — let the trade run

        # Latch the halt so subsequent trades stay blocked this session.
        tool_context.state[STATE_HALTED] = True
        return {
            "status": "VETOED",
            "blocked_tool": getattr(tool, "name", "unknown"),
            "reason": reason,
            "auditor": "risk_management.portfolio.guardrail",
        }

    return before_tool_callback
