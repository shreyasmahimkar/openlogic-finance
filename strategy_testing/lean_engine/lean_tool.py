"""
strategy_testing/lean_engine/lean_tool.py

ADK-compatible tool that exposes LEAN backtesting to the strategy_testing agent.

The function signature follows Google ADK's tool convention:
  - All args typed and documented (used as the LLM's tool schema)
  - Returns a plain dict (JSON-serialisable)

Register with an Agent like:
    from strategy_testing.lean_engine.lean_tool import run_sma_backtest
    root_agent = Agent(..., tools=[run_sma_backtest])
"""

import logging
from .lean_bridge import LeanEngineBridge

logger = logging.getLogger(__name__)

_bridge = LeanEngineBridge()


def run_sma_backtest(
    ticker:        str   = "SPY",
    fast_period:   int   = 50,
    slow_period:   int   = 200,
    position_size: float = 1.0,
) -> dict:
    """
    Run a QuantConnect LEAN backtest for the SMA Golden Cross strategy.

    This executes a full historical backtest using the local LEAN engine,
    testing the strategy where a BUY signal fires when the fast SMA crosses
    above the slow SMA (Golden Cross), and a SELL fires when it crosses below
    (Death Cross).

    Args:
        ticker:        The asset ticker symbol to backtest (e.g. "SPY", "QQQ", "AAPL").
                       Defaults to "SPY".
        fast_period:   The lookback period for the fast Simple Moving Average.
                       Defaults to 50 (the standard short-term trend indicator).
        slow_period:   The lookback period for the slow Simple Moving Average.
                       Defaults to 200 (the standard long-term trend indicator).
        position_size: Fraction of portfolio to allocate on a Golden Cross signal.
                       1.0 = 100% (fully invested). Defaults to 1.0.

    Returns:
        A dict summarising the backtest result with keys:
            - strategy_name (str): Human-readable label for the run.
            - ticker (str): Asset tested.
            - fast_period (int): Fast SMA period used.
            - slow_period (int): Slow SMA period used.
            - success (bool): Whether LEAN completed without errors.
            - return_code (int): LEAN CLI exit code (0 = success).
            - total_return_pct (float | None): Parsed total return percentage.
            - output_dir (str | None): Path where LEAN wrote results.
            - started_at (str): ISO 8601 timestamp.
            - completed_at (str | None): ISO 8601 timestamp.
    """
    logger.info(
        f"[ADK TOOL] run_sma_backtest called | "
        f"ticker={ticker} fast={fast_period} slow={slow_period} size={position_size}"
    )

    result = _bridge.run_backtest(
        ticker        = ticker,
        fast_period   = fast_period,
        slow_period   = slow_period,
        position_size = position_size,
    )

    d = result.to_dict()

    # Build a human-readable summary the ADK agent can show directly
    status = "✅ SUCCESS" if d["success"] else "❌ FAILED"
    ret    = f"{d['total_return_pct']:.2f}%" if d["total_return_pct"] is not None else "N/A"

    summary_lines = [
        f"## LEAN Backtest Result — {d['strategy_name']}",
        f"**Status**: {status}",
        f"**Ticker**: {d['ticker']}",
        f"**Fast SMA**: {d['fast_period']} | **Slow SMA**: {d['slow_period']}",
        f"**Net Profit**: {ret}",
        f"**Completed**: {d['completed_at']}",
        "",
    ]

    if d.get("full_summary"):
        summary_lines += ["### Full Statistics", "```", d["full_summary"], "```"]
    elif not d["success"]:
        summary_lines.append(f"**Error**: Check LEAN logs. Return code: {d['return_code']}")

    d["formatted_summary"] = "\n".join(summary_lines)
    return d


def check_lean_cli() -> dict:
    """
    Check whether the QuantConnect LEAN CLI is installed and available.

    Use this tool before running a backtest to confirm the environment is ready.

    Returns:
        A dict with:
            - installed (bool): True if `lean` CLI is found.
            - version (str): Version string or error message.
    """
    return _bridge.check_lean_installed()
