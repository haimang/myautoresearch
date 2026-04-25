"""Selector recommendation and surrogate repositories."""

from __future__ import annotations

import sqlite3

from .common import utc_now_iso


def save_recommendation_batch(
    conn: sqlite3.Connection,
    *,
    batch_id: str,
    campaign_id: str,
    selector_name: str,
    selector_version: str,
    selector_hash: str,
    frontier_snapshot_id: str | None = None,
    acquisition_name: str | None = None,
    acquisition_version: str | None = None,
    surrogate_snapshot_id: str | None = None,
) -> None:
    conn.execute(
        """INSERT INTO recommendation_batches
           (id, campaign_id, selector_name, selector_version, selector_hash,
            frontier_snapshot_id, acquisition_name, acquisition_version,
            surrogate_snapshot_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               campaign_id = excluded.campaign_id,
               selector_name = excluded.selector_name,
               selector_version = excluded.selector_version,
               selector_hash = excluded.selector_hash,
               frontier_snapshot_id = excluded.frontier_snapshot_id,
               acquisition_name = excluded.acquisition_name,
               acquisition_version = excluded.acquisition_version,
               surrogate_snapshot_id = excluded.surrogate_snapshot_id""",
        (
            batch_id,
            campaign_id,
            selector_name,
            selector_version,
            selector_hash,
            frontier_snapshot_id,
            acquisition_name,
            acquisition_version,
            surrogate_snapshot_id,
            utc_now_iso(),
        ),
    )
    conn.commit()


def save_recommendation(
    conn: sqlite3.Connection,
    *,
    recommendation_id: str,
    batch_id: str,
    candidate_type: str,
    candidate_key: str | None,
    rank: int,
    score_total: float,
    score_breakdown_json: str,
    rationale_json: str,
    axis_values_json: str | None = None,
    branch_reason: str | None = None,
    delta_json: str | None = None,
    selector_score_total: float | None = None,
    acquisition_score: float | None = None,
    parent_run_id: str | None = None,
    parent_checkpoint_id: int | None = None,
    candidate_payload_json: str | None = None,
    objective_metrics_json: str | None = None,
    status: str = "planned",
) -> None:
    conn.execute(
        """INSERT INTO recommendations
           (id, batch_id, candidate_type, candidate_key, rank, score_total,
             score_breakdown_json, rationale_json, axis_values_json,
             branch_reason, delta_json, selector_score_total, acquisition_score,
             parent_run_id, parent_checkpoint_id, candidate_payload_json,
             objective_metrics_json, status, created_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(id) DO UPDATE SET
                batch_id = excluded.batch_id,
                candidate_type = excluded.candidate_type,
                candidate_key = excluded.candidate_key,
                rank = excluded.rank,
                score_total = excluded.score_total,
                score_breakdown_json = excluded.score_breakdown_json,
                rationale_json = excluded.rationale_json,
                axis_values_json = excluded.axis_values_json,
                branch_reason = excluded.branch_reason,
                delta_json = excluded.delta_json,
                selector_score_total = excluded.selector_score_total,
                acquisition_score = excluded.acquisition_score,
                parent_run_id = excluded.parent_run_id,
                parent_checkpoint_id = excluded.parent_checkpoint_id,
                candidate_payload_json = excluded.candidate_payload_json,
                objective_metrics_json = excluded.objective_metrics_json,
                status = excluded.status""",
        (
            recommendation_id,
            batch_id,
            candidate_type,
            candidate_key,
            rank,
            score_total,
            score_breakdown_json,
            rationale_json,
            axis_values_json,
            branch_reason,
            delta_json,
            selector_score_total,
            acquisition_score,
            parent_run_id,
            parent_checkpoint_id,
            candidate_payload_json,
            objective_metrics_json,
            status,
            utc_now_iso(),
        ),
    )
    conn.commit()


def save_surrogate_snapshot(
    conn: sqlite3.Connection,
    *,
    snapshot_id: str,
    campaign_id: str,
    frontier_snapshot_id: str | None,
    acquisition_name: str,
    acquisition_version: str,
    policy_hash: str,
    objectives_json: str,
    feature_schema_json: str,
    summary_json: str,
    candidate_count: int,
) -> None:
    conn.execute(
        """INSERT INTO surrogate_snapshots
           (id, campaign_id, frontier_snapshot_id, acquisition_name,
            acquisition_version, policy_hash, objectives_json,
            feature_schema_json, summary_json, candidate_count, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               campaign_id = excluded.campaign_id,
               frontier_snapshot_id = excluded.frontier_snapshot_id,
               acquisition_name = excluded.acquisition_name,
               acquisition_version = excluded.acquisition_version,
               policy_hash = excluded.policy_hash,
               objectives_json = excluded.objectives_json,
               feature_schema_json = excluded.feature_schema_json,
               summary_json = excluded.summary_json,
               candidate_count = excluded.candidate_count""",
        (
            snapshot_id,
            campaign_id,
            frontier_snapshot_id,
            acquisition_name,
            acquisition_version,
            policy_hash,
            objectives_json,
            feature_schema_json,
            summary_json,
            candidate_count,
            utc_now_iso(),
        ),
    )
    conn.commit()


def get_surrogate_snapshot(conn: sqlite3.Connection, snapshot_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM surrogate_snapshots WHERE id = ?", (snapshot_id,)).fetchone()
    return dict(row) if row else None


def list_surrogate_snapshots(conn: sqlite3.Connection, campaign_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM surrogate_snapshots WHERE campaign_id = ? ORDER BY created_at DESC",
        (campaign_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def update_recommendation_status(conn: sqlite3.Connection, *, recommendation_id: str, status: str) -> None:
    conn.execute("UPDATE recommendations SET status = ? WHERE id = ?", (status, recommendation_id))
    conn.commit()


def save_recommendation_outcome(
    conn: sqlite3.Connection,
    *,
    recommendation_id: str,
    run_id: str | None = None,
    branch_id: str | None = None,
    observed_metrics_json: str = "{}",
    frontier_delta_json: str | None = None,
    constraint_status_json: str | None = None,
    outcome_label: str = "unknown",
) -> None:
    conn.execute(
        """INSERT INTO recommendation_outcomes
           (recommendation_id, run_id, branch_id, observed_metrics_json,
             frontier_delta_json, constraint_status_json, outcome_label, evaluated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            recommendation_id,
            run_id,
            branch_id,
            observed_metrics_json,
            frontier_delta_json,
            constraint_status_json,
            outcome_label,
            utc_now_iso(),
        ),
    )
    conn.commit()


def list_recommendation_batches(conn: sqlite3.Connection, campaign_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM recommendation_batches WHERE campaign_id = ? ORDER BY created_at DESC",
        (campaign_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def list_recommendations_for_batch(conn: sqlite3.Connection, batch_id: str) -> list[dict]:
    rows = conn.execute(
        """SELECT r.*, b.campaign_id
           FROM recommendations r
           JOIN recommendation_batches b ON b.id = r.batch_id
           WHERE r.batch_id = ?
           ORDER BY r.rank""",
        (batch_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def list_recommendation_outcomes(conn: sqlite3.Connection, recommendation_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM recommendation_outcomes WHERE recommendation_id = ? ORDER BY evaluated_at",
        (recommendation_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_latest_recommendation_batch(conn: sqlite3.Connection, campaign_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM recommendation_batches WHERE campaign_id = ? ORDER BY created_at DESC LIMIT 1",
        (campaign_id,),
    ).fetchone()
    return dict(row) if row else None


def get_recommendation_by_id(conn: sqlite3.Connection, recommendation_id: str) -> dict | None:
    row = conn.execute(
        """SELECT r.*, b.campaign_id, b.frontier_snapshot_id,
                  b.acquisition_name, b.acquisition_version, b.surrogate_snapshot_id
           FROM recommendations r
           JOIN recommendation_batches b ON b.id = r.batch_id
           WHERE r.id = ?""",
        (recommendation_id,),
    ).fetchone()
    return dict(row) if row else None
