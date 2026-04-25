"""Persistence helpers for frontier snapshots."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone


def save_frontier_snapshot(
    conn: sqlite3.Connection,
    *,
    front: list[dict],
    dominated: list[dict],
    maximize: list[str],
    minimize: list[str],
    eval_level: int | None,
    sweep_tag: str | None,
    campaign_id: str | None = None,
    objective_profile_id: str | None = None,
    metric_source: str | None = None,
    constraints_json: str | None = None,
    knee_run_id: str | None = None,
    knee_rationale_json: str | None = None,
) -> str | None:
    try:
        snapshot_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO frontier_snapshots
                (id, created_at, maximize_axes, minimize_axes, front_run_ids,
                 dominated_count, total_runs, eval_level, sweep_tag, campaign_id,
                 objective_profile_id, metric_source, constraints_json,
                 knee_run_id, knee_rationale_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                snapshot_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(maximize),
                json.dumps(minimize),
                json.dumps([p.get("run_full", p.get("run", "")) for p in front]),
                len(dominated),
                len(front) + len(dominated),
                eval_level,
                sweep_tag,
                campaign_id,
                objective_profile_id,
                metric_source,
                constraints_json,
                knee_run_id,
                knee_rationale_json,
            ),
        )
        conn.commit()
        return snapshot_id
    except sqlite3.OperationalError:
        return None
