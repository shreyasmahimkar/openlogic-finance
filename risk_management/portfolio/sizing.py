"""Capital-allocation sizing primitives (Box 4).

Pure, deterministic, dependency-light math. Given the per-strategy **daily return
series** of the selected strategies, compute a recommended weight vector under several
allocation philosophies. No I/O, no LLM, no session state — this is the math the
`enterprise/allocator.py` orchestrator and the Box 4 dashboard panel compose.

Each `*_weights` function returns a long-only weight vector that sums to 1 (a *split* of
the strategy sleeve); leverage / cash decisions live one level up in the allocator.

See `risk_management/PORTFOLIO_ALLOCATION_REQUIREMENTS.md` §3.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

TRADING_DAYS = 252


def annualize(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Daily simple returns ``(T, N)`` → annualized ``(mu (N,), Sigma (N, N))``.

    ``mu`` is the annualized mean; ``Sigma`` the annualized covariance matrix.
    """
    returns = np.atleast_2d(np.asarray(returns, dtype=float))
    if returns.shape[0] == 1:  # a single row is a degenerate sample, not N assets
        returns = returns.T
    mu = returns.mean(axis=0) * TRADING_DAYS
    sigma = np.cov(returns, rowvar=False) * TRADING_DAYS
    sigma = np.atleast_2d(sigma)
    return mu, sigma


def portfolio_stats(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> tuple[float, float, float]:
    """Return ``(expected_return, volatility, sharpe)`` for weight vector ``w`` (rf = 0)."""
    w = np.asarray(w, dtype=float)
    ret = float(w @ mu)
    var = float(w @ sigma @ w)
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe = ret / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def equal_weights(n: int) -> np.ndarray:
    """1/N — the humility baseline that uses no estimates."""
    return np.full(n, 1.0 / n)


def _solve_long_only(objective, n: int, x0: np.ndarray | None = None) -> np.ndarray:
    """Minimize ``objective(w)`` over the long-only simplex (``w ≥ 0``, ``Σw = 1``)."""
    x0 = equal_weights(n) if x0 is None else np.asarray(x0, dtype=float)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    w = np.clip(res.x, 0.0, None)
    total = w.sum()
    return w / total if total > 0 else equal_weights(n)


def min_variance_weights(sigma: np.ndarray) -> np.ndarray:
    """Long-only minimum-variance portfolio: ``argmin wᵀΣw``."""
    n = sigma.shape[0]
    return _solve_long_only(lambda w: float(w @ sigma @ w), n)


def max_sharpe_weights(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Long-only tangency (max-Sharpe) portfolio. The mean-variance recommendation.

    Maximizes ``wᵀμ / √(wᵀΣw)`` — the only method that prices the *covariance* between
    strategies, so it is the one that decides whether a second strategy diversifies or is
    redundant.
    """
    n = sigma.shape[0]

    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(max(w @ sigma @ w, 1e-18)))
        return -ret / vol

    return _solve_long_only(neg_sharpe, n)


def risk_parity_weights(sigma: np.ndarray) -> np.ndarray:
    """Long-only risk-parity: each strategy contributes equal risk to the book.

    Robust when expected returns are untrustworthy (they usually are), so a good prior.
    """
    n = sigma.shape[0]

    def risk_dispersion(w):
        vol = float(np.sqrt(max(w @ sigma @ w, 1e-18)))
        marginal = (sigma @ w) / vol
        contrib = w * marginal
        return float(np.sum((contrib - contrib.mean()) ** 2))

    return _solve_long_only(risk_dispersion, n)


def kelly_weights(
    mu: np.ndarray, sigma: np.ndarray, fraction: float = 0.5
) -> tuple[np.ndarray, float]:
    """Continuous Kelly direction ``f* = Σ⁻¹μ``.

    Returns ``(weights, gross_exposure)`` where ``weights`` is the long-only normalized
    split (negative legs clipped) and ``gross_exposure`` is ``fraction · Σ f*`` — how much
    *total* leverage (fractional-)Kelly asks for. The allocator caps this; full Kelly is
    notoriously over-levered, hence ``fraction`` (half-Kelly default).
    """
    f_star = np.linalg.solve(sigma, mu)
    gross = float(fraction * f_star.sum())
    long_only = np.clip(f_star, 0.0, None)
    total = long_only.sum()
    weights = long_only / total if total > 0 else equal_weights(len(mu))
    return weights, gross


def correlation_matrix(sigma: np.ndarray) -> np.ndarray:
    """Covariance → correlation matrix (for the diversification / redundancy read)."""
    d = np.sqrt(np.clip(np.diag(sigma), 1e-18, None))
    return sigma / np.outer(d, d)


METHODS = ("max_sharpe", "min_variance", "risk_parity", "kelly", "equal_weight")


def weights_for_method(
    method: str, mu: np.ndarray, sigma: np.ndarray, kelly_fraction: float = 0.5
) -> tuple[np.ndarray, float]:
    """Dispatch a method name → ``(weights, gross_exposure)``.

    ``gross_exposure`` is 1.0 for fully-invested methods and the Kelly leverage for kelly.
    """
    n = sigma.shape[0]
    if method == "max_sharpe":
        return max_sharpe_weights(mu, sigma), 1.0
    if method == "min_variance":
        return min_variance_weights(sigma), 1.0
    if method == "risk_parity":
        return risk_parity_weights(sigma), 1.0
    if method == "equal_weight":
        return equal_weights(n), 1.0
    if method == "kelly":
        return kelly_weights(mu, sigma, fraction=kelly_fraction)
    raise ValueError(f"unknown allocation method: {method!r} (expected one of {METHODS})")
