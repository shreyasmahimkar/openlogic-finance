"""Model Portfolio Capital Allocation Engine (Box 4, enterprise level).

Composes the pure sizing primitives in ``portfolio/sizing.py`` with the cross-strategy
**risk constraints** (§4 of ``PORTFOLIO_ALLOCATION_REQUIREMENTS.md``) and produces a
recommended capital split **plus a grounded explanation of *why***:

    result = allocate(stats, AllocationConfig(method="max_sharpe"))
    result.weights          # {strategy_name: fraction of capital}
    result.explanation      # plain-English "why", every claim citing a number

This is **advisory** — it recommends weights, it never places orders. The hard drawdown
veto in ``portfolio/guardrail.py`` remains the non-overridable backstop, and a strategy
that is ``is_halted`` here is forced to weight 0 (the allocator never sizes *into* a
vetoed strategy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from risk_management.portfolio import sizing


@dataclass(frozen=True)
class StrategyStat:
    """One selected strategy's risk telemetry (the allocator's input contract)."""

    name: str
    daily_returns: np.ndarray  # 1-D array of daily simple returns (e.g. equity.pct_change())
    is_halted: bool = False  # True if the drawdown veto has halted this strategy

    @property
    def n_observations(self) -> int:
        return int(np.asarray(self.daily_returns).size)


@dataclass(frozen=True)
class AllocationConfig:
    """Method + risk limits. Config, never magic numbers buried in logic (AGENTS.md)."""

    method: str = "max_sharpe"  # one of sizing.METHODS
    kelly_fraction: float = 0.5  # half-Kelly
    max_strategy_weight: float = 0.60  # per-strategy cap
    max_gross_leverage: float = 1.0  # gross exposure cap (1.0 = no leverage)
    max_correlation: float = 0.80  # above this, flag a redundancy warning
    min_observations: int = 60  # refuse to size a strategy with less history


@dataclass
class AllocationResult:
    """The recommendation + the audit trail behind it."""

    weights: dict[str, float]
    cash_weight: float
    method: str
    per_strategy: dict[str, dict] = field(default_factory=dict)
    portfolio_metrics: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    warnings: list[str] = field(default_factory=list)


def _apply_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """Water-fill a long-only weight vector to honor a per-strategy ``cap``.

    Clip any strategy above ``cap`` and redistribute the freed capital **equally to the
    strategies still below the cap** (keeping the book fully invested when possible).
    Only when every strategy is at the cap does the remainder fall to cash — e.g. 2
    strategies at a 0.40 cap can hold at most 0.80, leaving 0.20 in cash.
    """
    if cap >= 1.0:
        return w
    w = np.minimum(w.astype(float).copy(), cap)
    for _ in range(100):
        deficit = 1.0 - float(w.sum())  # capital we still want to put to work
        if deficit <= 1e-12:
            break
        free = w < cap - 1e-12  # strategies that can still absorb more
        if not free.any():  # all at the cap — remainder stays in cash
            break
        w[free] = np.minimum(w[free] + deficit / int(free.sum()), cap)
    return w


def _standalone(returns: np.ndarray) -> tuple[float, float]:
    """Per-strategy annualized ``(vol, sharpe)`` from its own return series."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2 or r.std() == 0:
        return 0.0, 0.0
    vol = float(r.std() * np.sqrt(sizing.TRADING_DAYS))
    sharpe = float(np.sqrt(sizing.TRADING_DAYS) * r.mean() / r.std())
    return vol, sharpe


def allocate(stats: list[StrategyStat], config: AllocationConfig | None = None) -> AllocationResult:
    """Recommend a capital split across ``stats`` under ``config``. See module docstring."""
    config = config or AllocationConfig()
    if config.method not in sizing.METHODS:
        raise ValueError(f"unknown method {config.method!r}; expected {sizing.METHODS}")

    names = [s.name for s in stats]
    warnings: list[str] = []
    per_strategy: dict[str, dict] = {s.name: {"weight": 0.0, "bound_by": None} for s in stats}

    # --- 1. Eligibility: drop halted / thin-history strategies (they get weight 0). ---
    usable: list[StrategyStat] = []
    for s in stats:
        if s.is_halted:
            per_strategy[s.name]["bound_by"] = "halted"
            warnings.append(f"{s.name}: weight 0 — drawdown veto has halted this strategy.")
        elif s.n_observations < config.min_observations:
            per_strategy[s.name]["bound_by"] = "min_history"
            warnings.append(
                f"{s.name}: weight 0 — only {s.n_observations} obs "
                f"(< {config.min_observations} required)."
            )
        else:
            usable.append(s)

    # Standalone read for every strategy (for the "why", even halted ones).
    for s in stats:
        vol, sharpe = _standalone(s.daily_returns)
        per_strategy[s.name].update(vol=vol, sharpe=sharpe)

    if not usable:
        return AllocationResult(
            weights={n: 0.0 for n in names},
            cash_weight=1.0,
            method=config.method,
            per_strategy=per_strategy,
            warnings=warnings,
            explanation="No eligible strategies — all halted or below the minimum history. "
            "Recommendation: hold 100% cash.",
        )

    # --- 2. Build the joint return matrix (aligned, common length). ---
    arrs = [np.asarray(s.daily_returns, dtype=float) for s in usable]
    T = min(a.size for a in arrs)
    R = np.column_stack([a[-T:] for a in arrs])  # (T, k)
    mask = ~np.isnan(R).any(axis=1)
    R = R[mask]
    mu, sigma = sizing.annualize(R)
    corr = sizing.correlation_matrix(sigma)

    # --- 3. Raw weights from the chosen philosophy. ---
    split, gross = sizing.weights_for_method(config.method, mu, sigma, config.kelly_fraction)

    # --- 4. Per-strategy cap (redistribute excess). ---
    capped = _apply_cap(split, config.max_strategy_weight)
    for i, s in enumerate(usable):
        if split[i] > config.max_strategy_weight + 1e-9:
            per_strategy[s.name]["bound_by"] = "per_strategy_cap"

    # --- 5. Gross-leverage cap → invested fraction & cash. ---
    invested = min(gross, config.max_gross_leverage)
    if gross > config.max_gross_leverage + 1e-9:
        warnings.append(
            f"{config.method}: raw gross exposure {gross:.2f}x exceeds the "
            f"{config.max_gross_leverage:.2f}x leverage cap — scaled down."
        )
    weights_vec = capped * invested
    cash_weight = float(max(0.0, 1.0 - weights_vec.sum()))

    # --- 6. Correlation / redundancy read. ---
    k = len(usable)
    for i, s in enumerate(usable):
        others = [corr[i, j] for j in range(k) if j != i]
        max_corr = max(others) if others else 0.0
        per_strategy[s.name]["max_correlation"] = float(max_corr)
        per_strategy[s.name]["kelly_f"] = float(np.linalg.solve(sigma, mu)[i])
        if k > 1 and max_corr > config.max_correlation:
            warnings.append(
                f"{s.name}: highly correlated (ρ={max_corr:.2f}) with another strategy — "
                f"little diversification; the book is paying for near-duplicate risk."
            )

    # --- 7. Record final weights + portfolio metrics. ---
    for i, s in enumerate(usable):
        per_strategy[s.name]["weight"] = float(weights_vec[i])
    weights = {n: per_strategy[n]["weight"] for n in names}

    ret, vol, sharpe = sizing.portfolio_stats(capped, mu, sigma)  # of the invested sleeve
    best_single = max((per_strategy[s.name]["sharpe"] for s in usable), default=0.0)
    portfolio_metrics = {
        "expected_return": float(ret * invested),
        "expected_vol": float(vol * invested),
        "sharpe": float(sharpe),
        "gross_exposure": float(invested),
        "raw_kelly_gross": float(gross),
        "best_single_sharpe": float(best_single),
        "diversification_gain": float(sharpe - best_single),
    }

    explanation = _explain(config, usable, per_strategy, portfolio_metrics, cash_weight, warnings)
    return AllocationResult(
        weights=weights,
        cash_weight=cash_weight,
        method=config.method,
        per_strategy=per_strategy,
        portfolio_metrics=portfolio_metrics,
        explanation=explanation,
        warnings=warnings,
    )


# Human-readable labels for the method names.
_METHOD_LABEL = {
    "max_sharpe": "Mean-Variance (max-Sharpe)",
    "min_variance": "Minimum-Variance",
    "risk_parity": "Risk-Parity",
    "kelly": "Fractional-Kelly",
    "equal_weight": "Equal-Weight (1/N)",
}


def _explain(config, usable, per_strategy, metrics, cash_weight, warnings) -> str:
    """Tier-1 grounded narrative: every claim cites a number computed above (cite-or-abstain)."""
    label = _METHOD_LABEL.get(config.method, config.method)
    ranked = sorted(usable, key=lambda s: per_strategy[s.name]["weight"], reverse=True)
    lines = [f"Method: {label}.", ""]

    top = ranked[0]
    lines.append(
        f"• Largest allocation → {top.name} at {per_strategy[top.name]['weight'] * 100:.0f}% "
        f"(Sharpe {per_strategy[top.name]['sharpe']:.2f}, "
        f"vol {per_strategy[top.name]['vol'] * 100:.0f}%)."
    )
    for s in ranked[1:]:
        ps = per_strategy[s.name]
        bound = f" — {ps['bound_by'].replace('_', ' ')}" if ps.get("bound_by") else ""
        lines.append(
            f"• {s.name}: {ps['weight'] * 100:.0f}% "
            f"(Sharpe {ps['sharpe']:.2f}, vol {ps['vol'] * 100:.0f}%){bound}."
        )

    if cash_weight > 0.01:
        lines.append(f"• Cash: {cash_weight * 100:.0f}% (held back by the caps above).")

    gain = metrics["diversification_gain"]
    if gain <= 0.02:
        lines.append(
            f"\nDiversification: blended Sharpe {metrics['sharpe']:.2f} vs best single "
            f"{metrics['best_single_sharpe']:.2f} (gain {gain:+.2f}) — the strategies are too "
            f"correlated to add much; capital concentrates in the strongest edge by design."
        )
    else:
        lines.append(
            f"\nDiversification: blended Sharpe {metrics['sharpe']:.2f} beats the best single "
            f"strategy {metrics['best_single_sharpe']:.2f} (gain {gain:+.2f}) — the blend earns "
            f"its place."
        )

    if config.method == "kelly":
        lines.append(
            f"Kelly note: raw gross exposure was {metrics['raw_kelly_gross']:.2f}x; "
            f"applied at {metrics['gross_exposure']:.2f}x after the leverage cap "
            f"(fraction λ={config.kelly_fraction})."
        )
    if warnings:
        lines.append("\nConstraints / flags:")
        lines.extend(f"  - {w}" for w in warnings)
    return "\n".join(lines)
