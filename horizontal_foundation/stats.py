"""Shared statistical utilities (horizontal foundation).

Lives here so both validation (Box 3) and monitoring (Box 5) can import it without
violating the one-way box dependency direction (everything imports from the
horizontal foundation; it imports from nothing).
"""

import numpy as np


def population_stability_index(expected, actual, bins: int = 10) -> float:
    """PSI between a baseline (`expected`) and a new (`actual`) distribution.

    Bins are taken from the expected quantiles. Rule of thumb: <0.1 stable,
    0.1-0.25 moderate shift, >0.25 significant drift. Used for both feature
    stability (validation) and data/prediction drift (monitoring).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return 0.0

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_counts, _ = np.histogram(expected, edges)
    a_counts, _ = np.histogram(actual, edges)
    eps = 1e-6
    e_pct = np.clip(e_counts / e_counts.sum(), eps, None)
    a_pct = np.clip(a_counts / a_counts.sum(), eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def ks_statistic(y_true, scores) -> float:
    """Kolmogorov-Smirnov statistic (max separation of TPR-FPR) for a binary classifier."""
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, np.asarray(scores))
    return float(np.max(tpr - fpr))
