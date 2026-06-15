"""P4 tests: SQL feature store, object storage, audit log, feature pipeline. Offline."""

import numpy as np
import pandas as pd

from data_prep.feature_store import FeatureStore
from data_prep.pipelines.feature_pipeline import run_feature_pipeline
from horizontal_foundation.storage import LocalObjectStore
from risk_management.governance.audit import AuditLog


def _features():
    idx = pd.date_range("2026-01-01", periods=5, name="date")
    return pd.DataFrame({"ret_5d": [0.01, -0.02, 0.03, 0.0, 0.05]}, index=idx)


# ── feature store (Snowflake/SQL stand-in) ───────────────────────────────────
def test_feature_store_roundtrip_and_point_in_time():
    store = FeatureStore()
    n = store.write_features(_features())
    assert n == 5
    assert len(store.read_features()) == 5

    # point-in-time: latest row at-or-before a date (no lookahead)
    pit = store.point_in_time("2026-01-03")
    assert pit.iloc[0]["date"] == "2026-01-03"


def test_feature_store_summary_stats():
    store = FeatureStore()
    store.write_features(_features())
    stats = store.summary_stats("ret_5d")
    assert int(stats.iloc[0]["n"]) == 5
    assert stats.iloc[0]["max"] == 0.05


# ── object storage (S3/GCS stand-in) ──────────────────────────────────────────
def test_object_store_put_get_list(tmp_path):
    store = LocalObjectStore(str(tmp_path / "bucket"))
    store.put("corpus/nmbs_q4.txt", b"earnings transcript")
    assert store.exists("corpus/nmbs_q4.txt")
    assert store.get("corpus/nmbs_q4.txt") == b"earnings transcript"
    assert "corpus/nmbs_q4.txt" in store.list(prefix="corpus/")


# ── audit log (governance SQL) ────────────────────────────────────────────────
def test_audit_log_records_and_queries():
    log = AuditLog()
    log.record("retrieval", "SPY", "fetched 4 passages")
    log.record("recommendation", "SPY", "rating=BUY")
    log.record("approval", "SPY", "rating=BUY approved", actor="analyst.jane")

    assert len(log.query("SPY")) == 3
    approvals = log.approvals()
    assert len(approvals) == 1
    assert approvals.iloc[0]["actor"] == "analyst.jane"


# ── feature pipeline (Databricks stand-in) ────────────────────────────────────
def test_feature_pipeline_lands_features():
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0.1, 1.0, 120))
    prices = pd.DataFrame({"Close": close})
    store = FeatureStore()
    n = run_feature_pipeline(prices, store)
    assert n > 0
    assert "ret_5d" in store.read_features().columns
