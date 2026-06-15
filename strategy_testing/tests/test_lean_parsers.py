"""Unit tests for the LEAN stdout parsers (Box 3) — no LEAN/Docker required."""

from strategy_testing.lean_engine.lean_bridge import LeanEngineBridge

# A representative slice of LEAN cloud-backtest stdout (box-drawing stats table).
SAMPLE = """
Statistics
┌─────────────────────┬──────────────────┬─────────────────────┬──────────────────┐
│ Net Profit          │ -3.738%          │ Compounding Annual  │ 8.072%           │
│ Drawdown            │ 19.900%          │ Total Orders        │ 7                │
└─────────────────────┴──────────────────┴─────────────────────┴──────────────────┘
"""


def test_parse_return():
    assert LeanEngineBridge._parse_return(SAMPLE) == -3.738


def test_parse_cagr():
    assert LeanEngineBridge._parse_cagr(SAMPLE) == 8.072


def test_parse_drawdown():
    assert LeanEngineBridge._parse_drawdown(SAMPLE) == 19.900


def test_parse_orders():
    assert LeanEngineBridge._parse_orders(SAMPLE) == 7


def test_parse_return_plain_log_fallback():
    assert LeanEngineBridge._parse_return("Total Return  -3.74%") == -3.74


def test_parsers_return_none_on_empty():
    assert LeanEngineBridge._parse_return("") is None
    assert LeanEngineBridge._parse_orders("nothing here") is None


def test_extract_summary_table():
    table = LeanEngineBridge._extract_summary_table(SAMPLE)
    assert "Net Profit" in table and "└" in table
