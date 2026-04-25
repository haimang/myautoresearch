"""Promotion decision repository."""

from __future__ import annotations

import sqlite3

from .common import stable_json, utc_now_iso


def save_promotion_decision(
    conn: sqlite3.Connection,
    *,
    campaign_id: str,
    from_stage: str,
    to_stage: str,
    candidate_key: str,
    axis_values: dict,
    aggregated_metrics: dict,
    seed_count: int,
    decision: str,
    decision_rank: int | None = None,
    reason: str = "",
) -> None:
    conn.execute(
        """INSERT INTO promotion_decisions
           (campaign_id, from_stage, to_stage, candidate_key, axis_values_json,
            aggregated_metrics_json, seed_count, decision, decision_rank, reason, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(campaign_id, from_stage, to_stage, candidate_key) DO UPDATE SET
               axis_values_json = excluded.axis_values_json,
               aggregated_metrics_json = excluded.aggregated_metrics_json,
               seed_count = excluded.seed_count,
               decision = excluded.decision,
               decision_rank = excluded.decision_rank,
               reason = excluded.reason""",
        (
            campaign_id,
            from_stage,
            to_stage,
            candidate_key,
            stable_json(axis_values),
            stable_json(aggregated_metrics),
            seed_count,
            decision,
            decision_rank,
            reason,
            utc_now_iso(),
        ),
    )
    conn.commit()


def get_promotion_decisions(
    conn: sqlite3.Connection,
    campaign_id: str,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> list[dict]:
    sql = "SELECT * FROM promotion_decisions WHERE campaign_id = ?"
    params: list = [campaign_id]
    if from_stage is not None:
        sql += " AND from_stage = ?"
        params.append(from_stage)
    if to_stage is not None:
        sql += " AND to_stage = ?"
        params.append(to_stage)
    sql += " ORDER BY from_stage, decision_rank, created_at"
    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]
