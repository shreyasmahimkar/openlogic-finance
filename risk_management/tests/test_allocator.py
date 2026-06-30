"""Unit tests for the Box 4 capital-allocation orchestrator (no API keys)."""

import numpy as np

from risk_management.enterprise.allocator import (
    AllocationConfig,
    StrategyStat,
    allocate,
)


def _series(mean, vol, n=300, seed=0):
    return np.random.default_rng(seed).normal(mean, vol, n)


def _two_decent():
    return [
        StrategyStat("Model A", _series(0.0006, 0.010, seed=1)),
        StrategyStat("Model B", _series(0.0008, 0.012, seed=2)),
    ]


def test_weights_and_cash_sum_to_one():
    res = allocate(_two_decent(), AllocationConfig(method="max_sharpe"))
    assert np.isclose(sum(res.weights.values()) + res.cash_weight, 1.0, atol=1e-6)
    assert all(w >= -1e-9 for w in res.weights.values())


def test_per_strategy_cap_binds_and_forces_cash():
    # Equal-weight split is 0.5/0.5; a 0.4 cap clips both -> 0.2 cash, deterministic.
    res = allocate(_two_decent(), AllocationConfig(method="equal_weight", max_strategy_weight=0.40))
    for w in res.weights.values():
        assert w <= 0.40 + 1e-9
    assert np.isclose(res.cash_weight, 0.20, atol=1e-6)
    assert any(p["bound_by"] == "per_strategy_cap" for p in res.per_strategy.values())


def test_halted_strategy_gets_zero():
    stats = [
        StrategyStat("Model A", _series(0.0006, 0.010, seed=1)),
        StrategyStat("Model B", _series(0.0008, 0.012, seed=2), is_halted=True),
    ]
    res = allocate(stats, AllocationConfig(method="equal_weight", max_strategy_weight=1.0))
    assert res.weights["Model B"] == 0.0
    assert res.per_strategy["Model B"]["bound_by"] == "halted"
    # the non-halted strategy absorbs the book
    assert res.weights["Model A"] > 0.9


def test_thin_history_excluded():
    stats = [
        StrategyStat("Model A", _series(0.0006, 0.010, n=300, seed=1)),
        StrategyStat("Thin", _series(0.001, 0.01, n=10, seed=3)),
    ]
    res = allocate(stats, AllocationConfig(method="equal_weight", min_observations=60))
    assert res.weights["Thin"] == 0.0
    assert res.per_strategy["Thin"]["bound_by"] == "min_history"


def test_all_halted_is_all_cash():
    stats = [
        StrategyStat("A", _series(0.0006, 0.01, seed=1), is_halted=True),
        StrategyStat("B", _series(0.0008, 0.01, seed=2), is_halted=True),
    ]
    res = allocate(stats, AllocationConfig())
    assert np.isclose(res.cash_weight, 1.0)
    assert all(w == 0.0 for w in res.weights.values())
    assert "cash" in res.explanation.lower()


def test_high_correlation_raises_warning():
    base = np.random.default_rng(7).normal(0.0006, 0.01, 300)
    noise = np.random.default_rng(8).normal(0, 0.0005, 300)
    stats = [
        StrategyStat("A", base),
        StrategyStat("B", base + noise),  # ~0.99 correlated
    ]
    res = allocate(stats, AllocationConfig(method="equal_weight", max_correlation=0.80))
    assert any("correlated" in w.lower() for w in res.warnings)
    assert res.per_strategy["A"]["max_correlation"] > 0.80


def test_kelly_leverage_capped():
    # strong positive drift + low vol -> realized Sharpe is high, so Kelly clearly
    # wants leverage > 1 and must be scaled back to the cap.
    a = np.random.default_rng(11).normal(0.0010, 0.002, 400)
    b = np.random.default_rng(12).normal(0.0010, 0.002, 400)
    stats = [StrategyStat("A", a), StrategyStat("B", b)]
    res = allocate(stats, AllocationConfig(method="kelly", max_gross_leverage=1.0))
    assert res.portfolio_metrics["gross_exposure"] <= 1.0 + 1e-9
    assert res.portfolio_metrics["raw_kelly_gross"] > 1.0
    assert any("leverage cap" in w.lower() for w in res.warnings)


def test_explanation_is_grounded_nonempty():
    res = allocate(_two_decent(), AllocationConfig(method="max_sharpe"))
    assert res.explanation
    assert "Sharpe" in res.explanation
    assert "%" in res.explanation
