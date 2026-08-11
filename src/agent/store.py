"""Application tracker keyed by job_id.

Replaces the (company, title) dedup key in src/tracker.py, which collapsed
two genuinely different roles that shared a title at one company. The
original schema is migrated in place -- existing rows are preserved and
back-filled with a derived job_id.

Connections are opened per operation and always closed, rather than held
open at Streamlit module scope across reruns.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .identity import make_job_id

logger = logging.getLogger(__name__)

DEFAULT_DB = "data/tracker.db"
STATUSES = ["not_applied", "applied", "interview", "offer", "rejected", "ghosted"]


@contextmanager
def connect(db_path: str = DEFAULT_DB) -> Iterator[sqlite3.Connection]:
    """Open, commit-or-rollback, and always close."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB) -> None:
    """Create the table if absent; migrate it if it predates job_id."""
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                company TEXT,
                title TEXT,
                job_url TEXT,
                location TEXT,
                match_score REAL,
                priority TEXT,
                status TEXT DEFAULT 'not_applied',
                date_applied TEXT,
                cover_note TEXT
            )
            """
        )
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(applications)")}
        for column, ddl in [
            ("job_id", "ALTER TABLE applications ADD COLUMN job_id TEXT"),
            ("job_url", "ALTER TABLE applications ADD COLUMN job_url TEXT"),
            ("location", "ALTER TABLE applications ADD COLUMN location TEXT"),
            ("priority", "ALTER TABLE applications ADD COLUMN priority TEXT"),
        ]:
            if column not in existing:
                logger.info("migrating tracker.db: adding %s", column)
                conn.execute(ddl)

        # Back-fill job_id for pre-migration rows so they remain addressable.
        for row in conn.execute("SELECT id, company, title FROM applications WHERE job_id IS NULL OR job_id = ''"):
            conn.execute(
                "UPDATE applications SET job_id = ? WHERE id = ?",
                (make_job_id(row["company"], row["title"]), row["id"]),
            )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_applications_job_id ON applications(job_id)")


def save_application(
    job_id: str, company: str, title: str, cover_note: str = "",
    match_score: Optional[float] = None, priority: str = "", job_url: str = "",
    location: str = "", status: str = "not_applied", db_path: str = DEFAULT_DB,
) -> Dict[str, object]:
    """Insert or update by job_id. Returns {'action': 'inserted'|'updated', 'id': int}."""
    init_db(db_path)
    with connect(db_path) as conn:
        existing = conn.execute("SELECT id FROM applications WHERE job_id = ?", (job_id,)).fetchone()
        if existing:
            conn.execute(
                """UPDATE applications SET company=?, title=?, job_url=?, location=?,
                   match_score=?, priority=?, cover_note=? WHERE id=?""",
                (company, title, job_url, location, match_score, priority, cover_note, existing["id"]),
            )
            return {"action": "updated", "id": existing["id"]}
        cursor = conn.execute(
            """INSERT INTO applications
               (job_id, company, title, job_url, location, match_score, priority, status, cover_note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (job_id, company, title, job_url, location, match_score, priority, status, cover_note),
        )
        return {"action": "inserted", "id": cursor.lastrowid}


def list_applications(db_path: str = DEFAULT_DB) -> List[Dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM applications ORDER BY id DESC")]


def get_application(job_id: str, db_path: str = DEFAULT_DB) -> Optional[Dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM applications WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def update_status(app_id: int, status: str, db_path: str = DEFAULT_DB) -> None:
    init_db(db_path)
    applied_on = date.today().isoformat() if status != "not_applied" else None
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE applications SET status = ?, date_applied = ? WHERE id = ?",
            (status, applied_on, app_id),
        )


def status_counts(db_path: str = DEFAULT_DB) -> Dict[str, int]:
    init_db(db_path)
    with connect(db_path) as conn:
        return {r["status"]: r["n"] for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM applications GROUP BY status")}
