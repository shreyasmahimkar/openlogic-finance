"""Unit tests for the technical-indicator tools (Box 2; consumed by Box 1 experts)."""

import pandas as pd

from model_library.technical.indicators import enrich_ohlcv_data, read_market_indicators


def _write_ohlcv(path, n=80):
    # Deterministic ascending series so indicators are well-defined.
    close = [100 + i for i in range(n)]
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=n).astype(str),
            "Open": close,
            "High": [c + 1 for c in close],
            "Low": [c - 1 for c in close],
            "Close": close,
            "Volume": [1_000_000] * n,
        }
    )
    df.to_csv(path, index=False)


def test_enrich_adds_indicator_columns(tmp_path):
    csv = tmp_path / "SPY.csv"
    _write_ohlcv(csv)
    out = enrich_ohlcv_data(str(csv))
    df = pd.read_csv(out)
    for col in ["SMA_20", "SMA_30", "SMA_60", "MACD", "Bollinger_Upper", "RSI_30"]:
        assert col in df.columns
    assert out.endswith("_enriched.csv")


def test_read_market_indicators_returns_last_10(tmp_path):
    csv = tmp_path / "SPY.csv"
    _write_ohlcv(csv)
    enriched = enrich_ohlcv_data(str(csv))
    text = read_market_indicators(enriched)
    assert "Close" in text and "RSI_30" in text
    # header + separator + 10 data rows
    assert len(text.strip().splitlines()) == 12


def test_read_market_indicators_missing_file():
    assert "Error" in read_market_indicators("/no/such/file.csv")
