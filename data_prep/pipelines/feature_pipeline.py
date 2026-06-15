"""Feature pipeline (Box 1) — engineer features and land them in the feature store.

Local implementation uses pandas; the **Databricks** production job runs the same
transform with Spark over Delta Lake (the feature logic is identical — only the
execution engine differs). See `docs/DATA_PLATFORMS.md`.
"""

import pandas as pd

from data_prep.feature_store import FeatureStore
from model_library.ml_zoo.return_regime import engineer_features


def run_feature_pipeline(
    price_df: pd.DataFrame, store: FeatureStore, table: str = "features"
) -> int:
    """Compute model features from prices and persist them to the feature store.

    Returns the number of feature rows written. In production a Databricks job
    schedules this; the SQL/feature contract downstream is unchanged.
    """
    features = engineer_features(price_df).dropna()
    return store.write_features(features, table=table)
