"""Promotion, branch, and recommendation repositories."""

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
    return [dict(r) for r in rows]


def save_run_branch(
    conn: sqlite3.Connection,
    *,
    branch_id: str,
    campaign_id: str,
    parent_run_id: str,
    parent_checkpoint_id: int | None,
    from_stage: str,
    branch_reason: str,
    branch_params_json: str,
    delta_json: str,
    status: str = "planned",
    result_summary_json: str = "{}",
) -> None:
    if parent_checkpoint_id is None:
        existing = conn.execute(
            """SELECT id FROM run_branches
               WHERE campaign_id = ? AND parent_run_id = ?
                 AND branch_reason = ? AND delta_json = ?
                 AND parent_checkpoint_id IS NULL""",
            (campaign_id, parent_run_id, branch_reason, delta_json),
        ).fetchone()
        if existing and existing["id"] != branch_id:
            return
    conn.execute(
        """INSERT INTO run_branches
           (id, campaign_id, parent_run_id, parent_checkpoint_id, child_run_id,
            from_stage, branch_reason, branch_params_json, delta_json,
            status, result_summary_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id)
           DO UPDATE SET
               campaign_id = excluded.campaign_id,
               parent_run_id = excluded.parent_run_id,
               parent_checkpoint_id = excluded.parent_checkpoint_id,
               child_run_id = excluded.child_run_id,
               from_stage = excluded.from_stage,
               branch_reason = excluded.branch_reason,
               branch_params_json = excluded.branch_params_json,
               delta_json = excluded.delta_json,
               status = excluded.status,
               result_summary_json = excluded.result_summary_json""",
        (
            branch_id,
            campaign_id,
            parent_run_id,
            parent_checkpoint_id,
            None,
            from_stage,
            branch_reason,
            branch_params_json,
            delta_json,
            status,
            result_summary_json,
            utc_now_iso(),
        ),
    )
    conn.commit()


def bind_branch_child_run(
    conn: sqlite3.Connection,
    *,
    branch_id: str,
    child_run_id: str,
    status: str = "running",
) -> None:
    conn.execute(
        """UPDATE run_branches
           SET child_run_id = ?, status = ?, started_at = ?
           WHERE id = ?""",
        (child_run_id, status, utc_now_iso(), branch_id),
    )
    conn.commit()


def update_branch_status(
    conn: sqlite3.Connection,
    *,
    branch_id: str,
    status: str,
    result_summary_json: str | None = None,
) -> None:
    if result_summary_json is not None:
        conn.execute(
            """UPDATE run_branches
               SET status = ?, finished_at = ?, result_summary_json = ?
               WHERE id = ?""",
            (status, utc_now_iso(), result_summary_json, branch_id),
        )
    else:
        conn.execute(
            """UPDATE run_branches
               SET status = ?, finished_at = ?
               WHERE id = ?""",
            (status, utc_now_iso(), branch_id),
        )
    conn.commit()


def list_branches_for_campaign(conn: sqlite3.Connection, campaign_id: str) -> list[dict]:
    rows = conn.execute(
        """SELECT rb.*,
               p.sweep_tag AS parent_sweep_tag,
               c.sweep_tag AS child_sweep_tag
         FROM run_branches rb
         LEFT JOIN runs p ON p.id = rb.parent_run_id
         LEFT JOIN runs c ON c.id = rb.child_run_id
         WHERE rb.campaign_id = ?
         ORDER BY rb.created_at""",
        (campaign_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_branches_for_checkpoint(conn: sqlite3.Connection, checkpoint_id: int) -> list[dict]:
    rows = conn.execute(
        """SELECT rb.*,
               p.sweep_tag AS parent_sweep_tag,
               c.sweep_tag AS child_sweep_tag
         FROM run_branches rb
         LEFT JOIN runs p ON p.id = rb.parent_run_id
         LEFT JOIN runs c ON c.id = rb.child_run_id
         WHERE rb.parent_checkpoint_id = ?
         ORDER BY rb.created_at""",
        (checkpoint_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_branch_tree(conn: sqlite3.Connection, campaign_id: str) -> list[dict]:
    rows = conn.execute(
        """SELECT rb.id, rb.parent_run_id, rb.child_run_id,
               rb.parent_checkpoint_id, rb.branch_reason, rb.delta_json,
               rb.status, rb.result_summary_json,
               p.sweep_tag AS parent_tag, p.final_win_rate AS parent_wr,
               c.sweep_tag AS child_tag, c.final_win_rate AS child_wr,
               c.wall_time_s AS child_wall_s, c.num_params AS child_params
         FROM run_branches rb
         LEFT JOIN runs p ON p.id = rb.parent_run_id
         LEFT JOIN runs c ON c.id = rb.child_run_id
         WHERE rb.campaign_id = ?
         ORDER BY rb.created_at""",
        (campaign_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_branch_by_id(conn: sqlite3.Connection, branch_id: str) -> dict | None:
    row = conn.execute(
        """SELECT rb.*,
               p.sweep_tag AS parent_sweep_tag,
               c.sweep_tag AS child_sweep_tag
         FROM run_branches rb
         LEFT JOIN runs p ON p.id = rb.parent_run_id
         LEFT JOIN runs c ON c.id = rb.child_run_id
         WHERE rb.id = ?""",
        (branch_id,),
    ).fetchone()
    return dict(row) if row else None


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
    return [dict(r) for r in rows]


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
    return [dict(r) for r in rows]


def list_recommendations_for_batch(conn: sqlite3.Connection, batch_id: str) -> list[dict]:
    rows = conn.execute(
        """SELECT r.*, b.campaign_id
           FROM recommendations r
           JOIN recommendation_batches b ON b.id = r.batch_id
           WHERE r.batch_id = ?
           ORDER BY r.rank""",
        (batch_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_recommendation_outcomes(conn: sqlite3.Connection, recommendation_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM recommendation_outcomes WHERE recommendation_id = ? ORDER BY evaluated_at",
        (recommendation_id,),
    ).fetchall()
    return [dict(r) for r in rows]


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
