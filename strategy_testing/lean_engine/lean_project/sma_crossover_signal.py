"""
model_library/technical/signals/sma_crossover_signal.py

Pure Python signal logic for the SMA Golden Cross / Death Cross strategy.

This module has ZERO dependencies on QuantConnect LEAN or any execution framework.
It operates only on plain Python floats and is independently testable.

Design contract:
  - Box 2 (Model Library) owns WHAT to decide given indicator values.
  - Box 3 (LEAN Adapter in Main.py) owns HOW to compute those values and execute orders.

This file is the single source of truth.
lean_bridge.py automatically syncs it into lean_project/ before every cloud push.

Usage (anywhere in the project, no LEAN needed):
    from model_library.technical.signals import detect_crossover, drawdown_breached, SignalType
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ── Signal Types ──────────────────────────────────────────────────────────────

class SignalType(Enum):
    """
    Crossover signal emitted by detect_crossover() on each bar.

    GOLDEN_CROSS : Fast SMA just crossed ABOVE slow SMA → enter long
    DEATH_CROSS  : Fast SMA just crossed BELOW slow SMA → exit long
    NONE         : No crossover event on this bar
    """
    GOLDEN_CROSS = "GOLDEN_CROSS"
    DEATH_CROSS  = "DEATH_CROSS"
    NONE         = "NONE"


# ── Strategy Configuration ────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """
    Canonical parameter set for the SMA crossover strategy.

    This dataclass is the single place to define strategy parameters.
    LEAN's Main.py reads them via get_parameter() and constructs this config,
    so LEAN and any local backtesting scripts always use identical defaults.

    Fields:
        fast_period:      Lookback for the fast SMA (default 50).
        slow_period:      Lookback for the slow SMA (default 200).
        position_size:    Fractional portfolio allocation on Golden Cross (default 1.0 = 100%).
        max_drawdown_pct: Liquidate if portfolio falls this far from peak (default 0.15 = 15%).
        ticker:           Asset ticker symbol (default "SPY").
    """
    fast_period:      int   = 50
    slow_period:      int   = 200
    position_size:    float = 1.0
    max_drawdown_pct: float = 0.15
    ticker:           str   = "SPY"

    def __post_init__(self):
        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"fast_period ({self.fast_period}) must be < slow_period ({self.slow_period})"
            )
        if not 0.0 < self.position_size <= 1.0:
            raise ValueError(
                f"position_size ({self.position_size}) must be in (0.0, 1.0]"
            )
        if not 0.0 < self.max_drawdown_pct < 1.0:
            raise ValueError(
                f"max_drawdown_pct ({self.max_drawdown_pct}) must be in (0.0, 1.0)"
            )


# ── Signal Functions ──────────────────────────────────────────────────────────

def detect_crossover(
    fast:      float,
    slow:      float,
    prev_fast: Optional[float],
    prev_slow: Optional[float],
) -> SignalType:
    """
    Detect whether a Golden Cross or Death Cross occurred between the previous
    bar and the current bar.

    A Golden Cross occurs when:
        previous bar: fast_sma <= slow_sma
        current  bar: fast_sma >  slow_sma

    A Death Cross occurs when:
        previous bar: fast_sma >= slow_sma
        current  bar: fast_sma <  slow_sma

    Args:
        fast:      Current bar fast SMA value.
        slow:      Current bar slow SMA value.
        prev_fast: Previous bar fast SMA value. None on the first bar.
        prev_slow: Previous bar slow SMA value. None on the first bar.

    Returns:
        SignalType.GOLDEN_CROSS, SignalType.DEATH_CROSS, or SignalType.NONE.

    Examples:
        >>> detect_crossover(fast=205.0, slow=200.0, prev_fast=199.0, prev_slow=200.0)
        <SignalType.GOLDEN_CROSS: 'GOLDEN_CROSS'>

        >>> detect_crossover(fast=198.0, slow=200.0, prev_fast=201.0, prev_slow=200.0)
        <SignalType.DEATH_CROSS: 'DEATH_CROSS'>

        >>> detect_crossover(fast=205.0, slow=200.0, prev_fast=203.0, prev_slow=200.0)
        <SignalType.NONE: 'NONE'>
    """
    # First bar — no previous state to compare
    if prev_fast is None or prev_slow is None:
        return SignalType.NONE

    # Golden Cross: fast was at or below slow, now above
    if prev_fast <= prev_slow and fast > slow:
        return SignalType.GOLDEN_CROSS

    # Death Cross: fast was at or above slow, now below
    if prev_fast >= prev_slow and fast < slow:
        return SignalType.DEATH_CROSS

    return SignalType.NONE


def drawdown_breached(
    current_value: float,
    peak_value:    float,
    threshold:     float = 0.15,
) -> bool:
    """
    Determine whether the portfolio has dropped beyond the maximum allowed drawdown.

    Drawdown = (peak_value - current_value) / peak_value

    Args:
        current_value: Current total portfolio value.
        peak_value:    Highest portfolio value observed so far.
        threshold:     Maximum tolerated drawdown as a decimal (default 0.15 = 15%).

    Returns:
        True if the drawdown equals or exceeds the threshold, False otherwise.

    Examples:
        >>> drawdown_breached(current_value=85_000, peak_value=100_000, threshold=0.15)
        True

        >>> drawdown_breached(current_value=90_000, peak_value=100_000, threshold=0.15)
        False

        >>> drawdown_breached(current_value=85_000, peak_value=100_000, threshold=0.15)
        True

        # Edge case: peak_value is zero (should not breach)
        >>> drawdown_breached(current_value=0, peak_value=0, threshold=0.15)
        False
    """
    if peak_value <= 0:
        return False

    drawdown = (peak_value - current_value) / peak_value
    return drawdown >= threshold


def generate_crossover_signals(df, config: StrategyConfig):
    """
    Enriches a price DataFrame with Fast SMA, Slow SMA, and raw crossover signals.
    """
    import pandas as pd
    df = df.copy()
    df["Fast_SMA"] = df["Close"].rolling(window=config.fast_period).mean()
    df["Slow_SMA"] = df["Close"].rolling(window=config.slow_period).mean()
    
    signals = []
    prev_fast = None
    prev_slow = None
    
    for idx, row in df.iterrows():
        fast = row["Fast_SMA"]
        slow = row["Slow_SMA"]
        
        if pd.isna(fast) or pd.isna(slow):
            signals.append(SignalType.NONE)
        else:
            sig = detect_crossover(fast, slow, prev_fast, prev_slow)
            signals.append(sig)
            prev_fast = fast
            prev_slow = slow
            
    df["Signal"] = [s.value for s in signals]
    return df

