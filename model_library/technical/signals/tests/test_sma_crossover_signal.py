"""
model_library/technical/signals/tests/test_sma_crossover_signal.py

Unit tests for the SMA crossover signal logic.

These tests require ZERO LEAN dependencies — just:
    pytest model_library/technical/signals/tests/

Or from repo root:
    python -m pytest model_library/technical/signals/tests/ -v
"""

import pytest
from model_library.technical.signals.sma_crossover_signal import (
    SignalType,
    StrategyConfig,
    detect_crossover,
    drawdown_breached,
)


# ══════════════════════════════════════════════════════════════════════════════
# detect_crossover tests
# ══════════════════════════════════════════════════════════════════════════════


class TestDetectCrossover:
    # ── Golden Cross ──────────────────────────────────────────────────────────

    def test_golden_cross_exact_crossover(self):
        """Fast SMA moves from below to above slow SMA."""
        signal = detect_crossover(fast=201.0, slow=200.0, prev_fast=199.0, prev_slow=200.0)
        assert signal == SignalType.GOLDEN_CROSS

    def test_golden_cross_from_equal_to_above(self):
        """Fast SMA moves from equal to slow to above it."""
        signal = detect_crossover(fast=201.0, slow=200.0, prev_fast=200.0, prev_slow=200.0)
        assert signal == SignalType.GOLDEN_CROSS

    def test_golden_cross_large_gap(self):
        """Golden Cross with a big numerical gap still detected."""
        signal = detect_crossover(fast=300.0, slow=200.0, prev_fast=100.0, prev_slow=200.0)
        assert signal == SignalType.GOLDEN_CROSS

    # ── Death Cross ───────────────────────────────────────────────────────────

    def test_death_cross_exact_crossover(self):
        """Fast SMA moves from above to below slow SMA."""
        signal = detect_crossover(fast=199.0, slow=200.0, prev_fast=201.0, prev_slow=200.0)
        assert signal == SignalType.DEATH_CROSS

    def test_death_cross_from_equal_to_below(self):
        """Fast SMA moves from equal to slow to below it."""
        signal = detect_crossover(fast=199.0, slow=200.0, prev_fast=200.0, prev_slow=200.0)
        assert signal == SignalType.DEATH_CROSS

    def test_death_cross_large_gap(self):
        """Death Cross with large numerical gap still detected."""
        signal = detect_crossover(fast=100.0, slow=200.0, prev_fast=300.0, prev_slow=200.0)
        assert signal == SignalType.DEATH_CROSS

    # ── No Signal ─────────────────────────────────────────────────────────────

    def test_no_signal_fast_already_above(self):
        """Fast already above slow on both bars — no crossover."""
        signal = detect_crossover(fast=205.0, slow=200.0, prev_fast=203.0, prev_slow=200.0)
        assert signal == SignalType.NONE

    def test_no_signal_fast_already_below(self):
        """Fast already below slow on both bars — no crossover."""
        signal = detect_crossover(fast=195.0, slow=200.0, prev_fast=198.0, prev_slow=200.0)
        assert signal == SignalType.NONE

    def test_no_signal_both_equal(self):
        """Both bars have fast == slow — no directional crossover."""
        signal = detect_crossover(fast=200.0, slow=200.0, prev_fast=200.0, prev_slow=200.0)
        assert signal == SignalType.NONE

    # ── First bar (None prev values) ─────────────────────────────────────────

    def test_first_bar_none_prev_fast(self):
        """No previous fast value — cannot detect crossover."""
        signal = detect_crossover(fast=205.0, slow=200.0, prev_fast=None, prev_slow=200.0)
        assert signal == SignalType.NONE

    def test_first_bar_none_prev_slow(self):
        """No previous slow value — cannot detect crossover."""
        signal = detect_crossover(fast=205.0, slow=200.0, prev_fast=199.0, prev_slow=None)
        assert signal == SignalType.NONE

    def test_first_bar_both_none(self):
        """Both previous values are None — first bar, no signal."""
        signal = detect_crossover(fast=205.0, slow=200.0, prev_fast=None, prev_slow=None)
        assert signal == SignalType.NONE

    # ── Return type ───────────────────────────────────────────────────────────

    def test_returns_signal_type_enum(self):
        """Return value is always a SignalType enum member."""
        result = detect_crossover(205.0, 200.0, 199.0, 200.0)
        assert isinstance(result, SignalType)


# ══════════════════════════════════════════════════════════════════════════════
# drawdown_breached tests
# ══════════════════════════════════════════════════════════════════════════════


class TestDrawdownBreached:
    def test_breach_at_exactly_threshold(self):
        """Drawdown == threshold is a breach (>=)."""
        assert drawdown_breached(85_000, 100_000, threshold=0.15) is True

    def test_breach_above_threshold(self):
        """Drawdown greater than threshold is a breach."""
        assert drawdown_breached(80_000, 100_000, threshold=0.15) is True

    def test_no_breach_just_below_threshold(self):
        """Drawdown just below threshold is not a breach."""
        assert drawdown_breached(86_000, 100_000, threshold=0.15) is False

    def test_no_breach_at_zero_drawdown(self):
        """Portfolio at peak — no drawdown."""
        assert drawdown_breached(100_000, 100_000, threshold=0.15) is False

    def test_no_breach_portfolio_growing(self):
        """Current value above peak (shouldn't happen in real use, but safe)."""
        assert drawdown_breached(110_000, 100_000, threshold=0.15) is False

    def test_zero_peak_value_safe(self):
        """Peak value of 0 should never trigger a breach (guards division by zero)."""
        assert drawdown_breached(0, 0, threshold=0.15) is False

    def test_custom_threshold_20pct(self):
        """Custom threshold: 20% drawdown with 20% threshold should breach."""
        assert drawdown_breached(80_000, 100_000, threshold=0.20) is True

    def test_custom_threshold_20pct_no_breach(self):
        """Custom threshold: 15% drawdown with 20% threshold should not breach."""
        assert drawdown_breached(85_000, 100_000, threshold=0.20) is False

    def test_returns_bool(self):
        """Always returns a bool."""
        result = drawdown_breached(90_000, 100_000)
        assert isinstance(result, bool)


# ══════════════════════════════════════════════════════════════════════════════
# StrategyConfig validation tests
# ══════════════════════════════════════════════════════════════════════════════


class TestStrategyConfig:
    def test_default_construction(self):
        """Default config builds without error."""
        cfg = StrategyConfig()
        assert cfg.fast_period == 50
        assert cfg.slow_period == 200
        assert cfg.position_size == 1.0
        assert cfg.max_drawdown_pct == 0.15
        assert cfg.ticker == "SPY"

    def test_custom_construction(self):
        """Custom config builds correctly."""
        cfg = StrategyConfig(fast_period=20, slow_period=50, ticker="QQQ")
        assert cfg.fast_period == 20
        assert cfg.slow_period == 50
        assert cfg.ticker == "QQQ"

    def test_invalid_fast_gte_slow_raises(self):
        """fast_period >= slow_period should raise ValueError."""
        with pytest.raises(ValueError, match="fast_period"):
            StrategyConfig(fast_period=200, slow_period=50)

    def test_invalid_fast_equal_slow_raises(self):
        """fast_period == slow_period should raise ValueError."""
        with pytest.raises(ValueError, match="fast_period"):
            StrategyConfig(fast_period=50, slow_period=50)

    def test_invalid_position_size_zero_raises(self):
        """position_size of 0 is invalid."""
        with pytest.raises(ValueError, match="position_size"):
            StrategyConfig(position_size=0.0)

    def test_invalid_position_size_over_one_raises(self):
        """position_size > 1.0 is invalid (no leverage)."""
        with pytest.raises(ValueError, match="position_size"):
            StrategyConfig(position_size=1.5)

    def test_invalid_max_drawdown_zero_raises(self):
        """max_drawdown_pct of 0 is invalid."""
        with pytest.raises(ValueError, match="max_drawdown_pct"):
            StrategyConfig(max_drawdown_pct=0.0)

    def test_invalid_max_drawdown_one_raises(self):
        """max_drawdown_pct of 1.0 means 100% loss — invalid."""
        with pytest.raises(ValueError, match="max_drawdown_pct"):
            StrategyConfig(max_drawdown_pct=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end simulation test (no LEAN required)
# ══════════════════════════════════════════════════════════════════════════════


class TestEndToEndSimulation:
    """
    Simulates a sequence of bars to verify the crossover logic produces
    the expected pattern of signals over time.
    """

    def test_golden_then_death_cross_sequence(self):
        """
        Bar sequence:
          Bars 1-3: fast below slow (below regime)
          Bar 4:    fast crosses above → GOLDEN CROSS
          Bars 5-6: fast stays above (holding)
          Bar 7:    fast crosses below → DEATH CROSS
        """
        bars = [
            # (fast, slow)
            (195.0, 200.0),  # bar 1 — below
            (197.0, 200.0),  # bar 2 — still below
            (199.0, 200.0),  # bar 3 — still below
            (202.0, 200.0),  # bar 4 — GOLDEN CROSS
            (204.0, 200.0),  # bar 5 — above, no cross
            (206.0, 200.0),  # bar 6 — above, no cross
            (198.0, 200.0),  # bar 7 — DEATH CROSS
        ]

        signals = []
        prev_fast, prev_slow = None, None

        for fast, slow in bars:
            signal = detect_crossover(fast, slow, prev_fast, prev_slow)
            signals.append(signal)
            prev_fast, prev_slow = fast, slow

        assert signals[0] == SignalType.NONE  # bar 1: first bar
        assert signals[1] == SignalType.NONE  # bar 2: already below
        assert signals[2] == SignalType.NONE  # bar 3: still below
        assert signals[3] == SignalType.GOLDEN_CROSS  # bar 4: crossover UP
        assert signals[4] == SignalType.NONE  # bar 5: holding
        assert signals[5] == SignalType.NONE  # bar 6: holding
        assert signals[6] == SignalType.DEATH_CROSS  # bar 7: crossover DOWN
