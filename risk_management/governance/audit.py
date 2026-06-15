"""Audit log (Box 4 governance) — a queryable record of decisions & approvals.

Every retrieval, model score, recommendation, and human approval is logged to a
SQL table so governance is *auditable* — "why did the agent recommend BUY, and who
signed off?" Local stand-in is SQLite; in production this is a **Snowflake** table
with row-level access + masking (see `docs/DATA_PLATFORMS.md`).
"""

import sqlite3
from datetime import datetime, timezone


class AuditLog:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit (
                ts          TEXT NOT NULL,
                event_type  TEXT NOT NULL,   -- retrieval | prediction | recommendation | approval
                ticker      TEXT,
                detail      TEXT,
                actor       TEXT             -- 'agent' or the approving human
            )
            """
        )
        self.conn.commit()

    def record(
        self, event_type: str, ticker: str = "", detail: str = "", actor: str = "agent"
    ) -> None:
        self.conn.execute(
            "INSERT INTO audit (ts, event_type, ticker, detail, actor) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), event_type, ticker, detail, actor),
        )
        self.conn.commit()

    def query(self, ticker: str | None = None):
        import pandas as pd

        if ticker:
            return pd.read_sql_query(
                "SELECT * FROM audit WHERE ticker = ? ORDER BY ts", self.conn, params=(ticker,)
            )
        return pd.read_sql_query("SELECT * FROM audit ORDER BY ts", self.conn)

    def approvals(self):
        import pandas as pd

        return pd.read_sql_query(
            "SELECT * FROM audit WHERE event_type = 'approval' ORDER BY ts", self.conn
        )

    def close(self) -> None:
        self.conn.close()
