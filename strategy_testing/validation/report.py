"""Model validation (Box 3) — the *validate* half of the MDLC.

Produces a `ValidationReport` (discrimination, calibration, stability, time-series
CV) and a **sign-off gate**: a model is not promoted unless it clears the
thresholds. This is what turns "it trained" into "it's fit to deploy".
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from horizontal_foundation.stats import ks_statistic, population_stability_index
from model_library.ml_zoo.return_regime import ReturnRegimeModel


@dataclass
class ValidationReport:
    auc: float
    accuracy: float
    brier: float  # calibration (lower is better)
    ks: float
    cv_auc_mean: float
    feature_psi: float  # train→test stability of a key feature
    n_train: int
    n_test: int
    confusion: list = field(default_factory=list)

    def passes_gate(self, min_auc: float = 0.52, max_brier: float = 0.30) -> bool:
        """Sign-off gate: discrimination above floor AND calibration within bound."""
        return self.auc >= min_auc and self.brier <= max_brier

    def summary(self) -> str:
        verdict = "PASS" if self.passes_gate() else "FAIL"
        return (
            f"[{verdict}] AUC={self.auc:.3f} acc={self.accuracy:.3f} "
            f"Brier={self.brier:.3f} KS={self.ks:.3f} cvAUC={self.cv_auc_mean:.3f} "
            f"PSI={self.feature_psi:.3f} (train={self.n_train}, test={self.n_test})"
        )


def _safe_auc(y, p) -> float:
    return roc_auc_score(y, p) if np.unique(y).size > 1 else 0.5


def validate(model: ReturnRegimeModel, X_train, y_train, X_test, y_test) -> ValidationReport:
    """Validate a trained model on a held-out test set + time-series CV on train."""
    # Time-series cross-validation (no leakage) on the training window.
    cv_aucs = []
    tscv = TimeSeriesSplit(n_splits=4)
    for tr_idx, va_idx in tscv.split(X_train):
        if np.unique(y_train.iloc[tr_idx]).size < 2:
            continue
        m = ReturnRegimeModel(horizon=model.horizon).train(
            X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        )
        cv_aucs.append(_safe_auc(y_train.iloc[va_idx], m.predict_proba_up(X_train.iloc[va_idx])))

    p_test = model.predict_proba_up(X_test)
    preds = (p_test > 0.5).astype(int)
    return ValidationReport(
        auc=_safe_auc(y_test, p_test),
        accuracy=accuracy_score(y_test, preds),
        brier=brier_score_loss(y_test, p_test) if np.unique(y_test).size > 1 else 1.0,
        ks=ks_statistic(y_test, p_test),
        cv_auc_mean=float(np.mean(cv_aucs)) if cv_aucs else 0.5,
        feature_psi=population_stability_index(X_train["ret_5d"], X_test["ret_5d"]),
        n_train=len(X_train),
        n_test=len(X_test),
        confusion=confusion_matrix(y_test, preds).tolist(),
    )
