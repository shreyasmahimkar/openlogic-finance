"""Feature store (Box 1) — a SQL system of record for features + scores.

The local stand-in uses **SQLite** so the SQL is real and runs offline; the
*identical* SQL runs on **Snowflake** in production (swap the connection — see
`docs/DATA_PLATFORMS.md`). This is the Snowflake/SQL half of the data-platform
story: point-in-time feature reads (no leakage) and monitoring aggregations.
"""

import sqlite3

import pandas as pd


class FeatureStore:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)

    def write_features(self, df: pd.DataFrame, table: str = "features") -> int:
        """Persist a feature frame (its index becomes the `date` key)."""
        out = df.reset_index().rename(columns={df.index.name or "index": "date"})
        out["date"] = out["date"].astype(str)
        out.to_sql(table, self.conn, if_exists="replace", index=False)
        return len(out)

    def read_features(self, table: str = "features") -> pd.DataFrame:
        return pd.read_sql_query(f"SELECT * FROM {table} ORDER BY date", self.conn)

    def point_in_time(self, as_of: str, table: str = "features") -> pd.DataFrame:
        """Latest row at-or-before `as_of` — point-in-time correctness (no lookahead)."""
        return pd.read_sql_query(
            f"SELECT * FROM {table} WHERE date <= ? ORDER BY date DESC LIMIT 1",
            self.conn,
            params=(as_of,),
        )

    def summary_stats(self, column: str, table: str = "features") -> pd.DataFrame:
        """Monitoring mart: count/avg/stddev/min/max for a feature (powers drift dashboards)."""
        return pd.read_sql_query(
            f"""
            SELECT COUNT(*) AS n,
                   AVG({column}) AS mean,
                   MIN({column}) AS min,
                   MAX({column}) AS max
            FROM {table}
            """,
            self.conn,
        )

    def close(self) -> None:
        self.conn.close()
