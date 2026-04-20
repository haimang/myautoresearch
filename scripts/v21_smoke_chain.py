#!/usr/bin/env python3
"""v21 smoke chain: create campaign → generate recommendation → accept → execute → backfill outcome."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from core.db import (
    create_run,
    finish_run,
    get_or_create_campaign,
    init_db,
    save_campaign_stage,
    save_checkpoint,
    save_search_space,
    update_recommendation_status,
    save_recommendation_outcome,
    get_latest_recommendation_batch,
    list_recommendations_for_batch,
)
from search_space import load_profile

PROFILE_PATH = ROOT / "domains" / "gomoku" / "search_space.json"
SELECTOR_POLICY = ROOT / "domains" / "gomoku" / "selector_policy.json"
ANALYZE = ROOT / "framework" / "analyze.py"

DB_PATH = ROOT / "output" / "v21_smoke.db"


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db_path = DB_PATH
    conn = init_db(str(db_path))

    # 1. Setup campaign with search space
    profile = load_profile(str(PROFILE_PATH))
    space_id = save_search_space(conn, profile)
    campaign = get_or_create_campaign(
        conn,
        name="v21-smoke",
        domain="gomoku",
        train_script="t.py",
        search_space_id=space_id,
        protocol={"eval_level": 0},
    )

    # 2. Seed runs with axis_values (numeric so perturbation works)
    configs = [
        ({"lr": 0.005, "blocks": 10, "filters": 128}, 0.88),
        ({"lr": 0.003, "blocks": 8,  "filters": 128}, 0.72),
        ({"lr": 0.001, "blocks": 6,  "filters": 64},  0.55),
    ]

    for i, (axis, wr) in enumerate(configs):
        rid = f"run-{i}"
        candidate_key = json.dumps(axis, sort_keys=True, separators=(",", ":"))
        create_run(conn, rid, {
            "sweep_tag": f"r{i}",
            "eval_level": 0,
            "learning_rate": axis["lr"],
            "num_res_blocks": axis["blocks"],
            "num_filters": axis["filters"],
        }, is_benchmark=True)
        finish_run(conn, rid, {
            "status": "completed",
            "final_win_rate": wr,
            "wall_time_s": 120.0,
            "num_params": 200000,
            "total_games": 500,
        })
        conn.execute(
            """INSERT INTO campaign_runs
               (campaign_id, run_id, stage, sweep_tag, seed, axis_values_json, status, created_at, candidate_key)
               VALUES (?, ?, 'C', ?, ?, ?, 'linked', '2024-01-01', ?)""",
            (campaign["id"], rid, f"r{i}", i, candidate_key, candidate_key),
        )
        save_checkpoint(conn, rid, {
            "tag": "final",
            "cycle": 100,
            "step": 1000,
            "loss": 0.5,
            "win_rate": wr,
            "eval_level": 0,
            "eval_games": 100,
            "model_path": f"/tmp/models/{rid}.npz",
        })

    conn.commit()
    conn.close()
    print(f"[1/5] Campaign + {len(configs)} seed runs created in {db_path}")

    # 3. Generate recommendation batch via CLI (dry-run first, then persist)
    proc = subprocess.run(
        [sys.executable, str(ANALYZE),
         "--db", str(db_path),
         "--recommend-next", "v21-smoke",
         "--selector-policy", str(SELECTOR_POLICY),
         "--limit", "5"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if proc.returncode != 0:
        print("recommend-next failed:", proc.stderr)
        return 1
    print("[2/5] Recommendation batch generated")
    print(proc.stdout)

    # 4. Accept top recommendation
    conn = init_db(str(db_path))
    batch = get_latest_recommendation_batch(conn, campaign["id"])
    if not batch:
        print("No batch generated!")
        return 1
    recs = list_recommendations_for_batch(conn, batch["id"])
    if not recs:
        print("No recommendations in batch!")
        return 1

    top = recs[0]
    update_recommendation_status(conn, recommendation_id=top["id"], status="accepted")
    print(f"[3/5] Top recommendation accepted: {top['id'][:24]} (type={top['candidate_type']})")

    # 5. Simulate execution: create a run
    exec_run_id = f"run-exec-{top['id'][:8]}"
    axis_vals = json.loads(top.get("axis_values_json") or "{}")
    if not axis_vals and top.get("candidate_key"):
        try:
            axis_vals = json.loads(top["candidate_key"])
        except Exception:
            axis_vals = {}

    create_run(conn, exec_run_id, {
        "sweep_tag": "v21-smoke-exec",
        "eval_level": 0,
        "learning_rate": axis_vals.get("lr", 0.003),
        "num_res_blocks": axis_vals.get("blocks", 8),
        "num_filters": axis_vals.get("filters", 128),
    }, is_benchmark=True)
    finish_run(conn, exec_run_id, {
        "status": "completed",
        "final_win_rate": 0.92,
        "wall_time_s": 130.0,
        "num_params": 210000,
        "total_games": 500,
    })
    update_recommendation_status(conn, recommendation_id=top["id"], status="executed")
    print(f"[4/5] Simulated execution: {exec_run_id} with WR=0.92")

    # 6. Backfill outcome
    save_recommendation_outcome(
        conn,
        recommendation_id=top["id"],
        run_id=exec_run_id,
        observed_metrics_json=json.dumps({"final_win_rate": 0.92, "wall_time_s": 130.0}),
        frontier_delta_json=json.dumps({"old_best_wr": 0.88, "new_best_wr": 0.92, "delta": +0.04}),
        outcome_label="new_front",
    )
    print(f"[5/5] Outcome backfilled: new_front (WR 0.88 → 0.92)")

    # 7. Verify via CLI
    conn.close()
    proc = subprocess.run(
        [sys.executable, str(ANALYZE),
         "--db", str(db_path),
         "--recommendation-outcomes", "v21-smoke"],
        capture_output=True, text=True, cwd=ROOT,
    )
    print("\n--- CLI outcome report ---")
    print(proc.stdout)
    if proc.returncode != 0:
        print("outcome report stderr:", proc.stderr)

    print(f"\nPersistent DB left at: {db_path}")
    print("v21 smoke chain complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
