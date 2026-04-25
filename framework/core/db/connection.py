"""Connection and initialization helpers for the db package."""

from __future__ import annotations

import os
import sqlite3

from .migrations import apply_migrations
from .schema_base import SCHEMA_SQL

DB_PATH = "output/tracker.db"


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = _connect(db_path)
    conn.executescript(SCHEMA_SQL)
    apply_migrations(conn)
    return conn
