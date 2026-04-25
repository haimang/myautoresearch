"""Branch planning and execution repository."""

from __future__ import annotations

import sqlite3

from .common import utc_now_iso


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
    return [dict(row) for row in rows]


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
    return [dict(row) for row in rows]


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
    return [dict(row) for row in rows]


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
