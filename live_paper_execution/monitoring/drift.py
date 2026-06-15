"""Production monitoring (Box 5) — the *monitor* half of the MDLC.

Watches a deployed model for **data drift** (feature distributions move),
**prediction drift** (score distribution moves), and **performance decay** (rolling
accuracy once labels arrive), and raises a **retraining trigger** when thresholds
break. In production these metrics are computed from the Snowflake marts daily
(see docs/EQUITY_RESEARCH.md roadmap).
"""

from dataclasses import dataclass, field

import pandas as pd
from sklearn.metrics import accuracy_score

from horizontal_foundation.stats import population_stability_index


@dataclass
class MonitoringReport:
    feature_psi: dict
    prediction_psi: float
    rolling_accuracy: float | None
    alerts: list = field(default_factory=list)
    needs_retrain: bool = False

    def summary(self) -> str:
        status = "RETRAIN" if self.needs_retrain else "OK"
        return f"[{status}] pred_PSI={self.prediction_psi:.3f} alerts={len(self.alerts)}"


def monitor(
    baseline_features: pd.DataFrame,
    current_features: pd.DataFrame,
    baseline_scores,
    current_scores,
    recent_labels=None,
    recent_preds=None,
    psi_threshold: float = 0.2,
    accuracy_floor: float = 0.5,
) -> MonitoringReport:
    """Compare current production data/scores against the training baseline."""
    feature_psi = {
        col: population_stability_index(baseline_features[col], current_features[col])
        for col in baseline_features.columns
    }
    prediction_psi = population_stability_index(baseline_scores, current_scores)

    alerts = [f"data drift: {c} PSI={v:.3f}" for c, v in feature_psi.items() if v > psi_threshold]
    if prediction_psi > psi_threshold:
        alerts.append(f"prediction drift PSI={prediction_psi:.3f}")

    rolling_accuracy = None
    if recent_labels is not None and recent_preds is not None and len(recent_labels):
        rolling_accuracy = accuracy_score(recent_labels, recent_preds)
        if rolling_accuracy < accuracy_floor:
            alerts.append(f"performance decay: accuracy={rolling_accuracy:.3f}")

    return MonitoringReport(
        feature_psi=feature_psi,
        prediction_psi=prediction_psi,
        rolling_accuracy=rolling_accuracy,
        alerts=alerts,
        needs_retrain=bool(alerts),
    )
