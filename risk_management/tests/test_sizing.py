"""Unit tests for the Box 4 capital-allocation sizing primitives (no API keys)."""

import numpy as np
import pytest

from risk_management.portfolio import sizing


def test_equal_weights_sum_to_one():
    w = sizing.equal_weights(4)
    assert w.shape == (4,)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, 0.25)


def test_portfolio_stats_basic():
    mu = np.array([0.10, 0.10])
    sig = np.diag([0.04, 0.04])
    ret, vol, sharpe = sizing.portfolio_stats([0.5, 0.5], mu, sig)
    assert np.isclose(ret, 0.10)
    # var = 0.25*0.04 + 0.25*0.04 = 0.02 -> vol = sqrt(0.02)
    assert np.isclose(vol, np.sqrt(0.02))
    assert np.isclose(sharpe, ret / vol)


def test_max_sharpe_tilts_to_higher_return():
    # equal vol, uncorrelated; asset 0 has the higher mean -> gets more weight.
    mu = np.array([0.20, 0.10])
    sig = np.diag([0.04, 0.04])
    w = sizing.max_sharpe_weights(mu, sig)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-9).all()
    assert w[0] > w[1]


def test_max_sharpe_symmetric_is_balanced():
    mu = np.array([0.10, 0.10])
    sig = np.diag([0.04, 0.04])
    w = sizing.max_sharpe_weights(mu, sig)
    assert np.allclose(w, [0.5, 0.5], atol=1e-3)


def test_min_variance_prefers_lower_vol():
    # asset 0 much lower variance -> min-variance overweights it.
    sig = np.diag([0.01, 0.09])
    w = sizing.min_variance_weights(sig)
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > w[1]


def test_risk_parity_inverse_to_vol():
    # diagonal cov, vols 0.1 and 0.2 -> weight ~ inverse vol -> asset 0 heavier.
    sig = np.diag([0.01, 0.04])
    w = sizing.risk_parity_weights(sig)
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > w[1]
    # equal risk contribution: w0*sig0 ≈ w1*sig1 in vol terms
    assert np.isclose(w[0] * 0.1, w[1] * 0.2, atol=0.05)


def test_kelly_returns_gross_and_normalized_split():
    mu = np.array([0.10, 0.10])
    sig = np.diag([0.04, 0.04])
    w, gross = sizing.kelly_weights(mu, sig, fraction=0.5)
    # f* = Sigma^-1 mu = [2.5, 2.5]; gross = 0.5 * 5.0 = 2.5
    assert np.isclose(gross, 2.5)
    assert np.allclose(w, [0.5, 0.5])


def test_annualize_shapes():
    rng = np.random.default_rng(0)
    R = rng.normal(0.0005, 0.01, size=(300, 3))
    mu, sig = sizing.annualize(R)
    assert mu.shape == (3,)
    assert sig.shape == (3, 3)
    # annualized vol of ~0.01 daily ≈ 0.16
    assert 0.10 < np.sqrt(sig[0, 0]) < 0.25


def test_correlation_matrix_unit_diagonal():
    sig = np.array([[0.04, 0.018], [0.018, 0.09]])
    c = sizing.correlation_matrix(sig)
    assert np.allclose(np.diag(c), 1.0)
    assert -1.0 <= c[0, 1] <= 1.0


def test_weights_for_method_dispatch_and_unknown():
    mu = np.array([0.12, 0.10])
    sig = np.diag([0.04, 0.05])
    for m in sizing.METHODS:
        w, gross = sizing.weights_for_method(m, mu, sig)
        assert np.isclose(w.sum(), 1.0)
        assert gross > 0
    with pytest.raises(ValueError):
        sizing.weights_for_method("nope", mu, sig)
