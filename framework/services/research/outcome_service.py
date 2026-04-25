"""Observed-outcome helpers shared by research executors."""

from __future__ import annotations

import json


def build_frontier_delta(conn, campaign_id: str, previous_best_wr: float | None, run_id: str) -> tuple[str, str, str]:
    row = conn.execute(
        """SELECT final_win_rate, wall_time_s, num_params, total_games, status
           FROM runs WHERE id = ?""",
        (run_id,),
    ).fetchone()
    observed = {
        "final_win_rate": row["final_win_rate"],
        "wall_time_s": row["wall_time_s"],
        "num_params": row["num_params"],
        "total_games": row["total_games"],
        "status": row["status"],
    }
    latest_best = conn.execute(
        """SELECT MAX(r.final_win_rate) AS best_wr
           FROM campaign_runs cr
           JOIN runs r ON r.id = cr.run_id
           WHERE cr.campaign_id = ?
             AND r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')""",
        (campaign_id,),
    ).fetchone()
    new_best_wr = latest_best["best_wr"] if latest_best else None
    old_best_wr = previous_best_wr or 0.0
    current_wr = observed["final_win_rate"] or 0.0
    if current_wr > old_best_wr:
        outcome_label = "new_front"
    elif current_wr >= max(0.0, old_best_wr - 0.02):
        outcome_label = "near_front"
    else:
        outcome_label = "no_gain"
    frontier_delta = {
        "old_best_wr": old_best_wr,
        "new_best_wr": new_best_wr,
        "delta": round((new_best_wr or 0.0) - old_best_wr, 6),
    }
    return (
        json.dumps(observed, ensure_ascii=False, sort_keys=True),
        json.dumps(frontier_delta, ensure_ascii=False, sort_keys=True),
        outcome_label,
    )
