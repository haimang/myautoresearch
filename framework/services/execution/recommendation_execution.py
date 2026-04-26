"""Execution of accepted point recommendations."""

from __future__ import annotations

import json
import os

from framework.core.db import (
    find_run_by_sweep_tag,
    get_campaign,
    get_campaign_stages,
    get_recommendation_by_id,
    link_run_to_campaign_v20,
    save_recommendation_outcome,
    update_recommendation_status,
)
from framework.policies.selector_policy import get_candidate_kind_config, load_selector_policy
from framework.services.execution.matrix import stable_candidate_key
from framework.services.research.outcome_service import build_frontier_delta

from .sweep_runner import run_one


def infer_recommendation_stage(conn, campaign_id: str) -> str | None:
    stages = get_campaign_stages(conn, campaign_id)
    if stages:
        open_stages = [stage for stage in stages if stage.get("status") == "open"]
        source = sorted(open_stages or stages, key=lambda stage: stage["stage"], reverse=True)
        return source[0]["stage"]
    row = conn.execute(
        "SELECT stage FROM campaign_runs WHERE campaign_id = ? AND stage IS NOT NULL ORDER BY stage DESC LIMIT 1",
        (campaign_id,),
    ).fetchone()
    return row["stage"] if row else None


def next_seed_for_candidate(conn, campaign_id: str, candidate_key: str | None) -> int:
    if not candidate_key:
        return 1
    rows = conn.execute(
        """SELECT DISTINCT seed FROM campaign_runs
           WHERE campaign_id = ? AND candidate_key = ? AND seed IS NOT NULL
           ORDER BY seed""",
        (campaign_id, candidate_key),
    ).fetchall()
    used = {row["seed"] for row in rows if row["seed"] is not None}
    seed = 1
    while seed in used:
        seed += 1
    return seed


def default_budget_for_recommendation(project_root: str, campaign: dict, candidate_type: str) -> int:
    selector_path = os.path.join(project_root, "domains", campaign["domain"], "manifest", "selector_policy.json")
    if os.path.isfile(selector_path):
        policy = load_selector_policy(selector_path)
        cfg = get_candidate_kind_config(policy, candidate_type)
        if cfg and cfg.get("default_budget_s"):
            return int(cfg["default_budget_s"])
    return 60


def execute_point_recommendation(conn, args, *, project_root: str) -> None:
    rec = get_recommendation_by_id(conn, args.execute_recommendation)
    if not rec:
        raise ValueError(f"Recommendation not found: {args.execute_recommendation}")
    if rec["status"] != "accepted":
        raise ValueError(f"Recommendation {rec['id']} status is '{rec['status']}', expected 'accepted'")
    if rec["candidate_type"] not in ("new_point", "seed_recheck"):
        raise ValueError(f"Recommendation {rec['id']} is '{rec['candidate_type']}', use branch.py for branch execution")

    campaign_row = get_campaign(conn, rec["campaign_id"])
    if not campaign_row:
        raise ValueError(f"Campaign not found for recommendation {rec['id']}")
    campaign = dict(campaign_row)
    if not args.train_script:
        args.train_script = campaign["train_script"]
    if not args.train_script:
        raise ValueError("Error: train_script is required for recommendation execution")

    args.time_budget = args.time_budget or default_budget_for_recommendation(project_root, campaign, rec["candidate_type"])
    stage = args.stage or infer_recommendation_stage(conn, campaign["id"])
    axis_values = json.loads(rec.get("axis_values_json") or "{}")
    if not axis_values and rec.get("candidate_key"):
        try:
            axis_values = json.loads(rec["candidate_key"])
        except json.JSONDecodeError:
            axis_values = {}

    seed = next_seed_for_candidate(conn, campaign["id"], rec.get("candidate_key"))
    tag = f"{campaign['name']}_rec_{rec['id'][:8]}_sd{seed}"
    cfg = {**axis_values, "seed": seed, "sweep_tag": tag}
    if campaign.get("objective_profile_id"):
        cfg["__candidate_payload"] = axis_values
        cfg["__candidate_key"] = stable_candidate_key(axis_values)

    previous_best = conn.execute(
        """SELECT MAX(r.final_win_rate) AS best_wr
           FROM campaign_runs cr
           JOIN runs r ON r.id = cr.run_id
           WHERE cr.campaign_id = ?
             AND r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')""",
        (campaign["id"],),
    ).fetchone()
    previous_best_wr = previous_best["best_wr"] if previous_best else None

    _, ok, _ = run_one(cfg, args, 1, 1, campaign=campaign)
    run_id = find_run_by_sweep_tag(conn, tag)
    if run_id:
        axis_identity = {k: v for k, v in cfg.items() if k not in ("seed", "sweep_tag")}
        link_run_to_campaign_v20(
            conn,
            campaign_id=campaign["id"],
            run_id=run_id,
            stage=stage,
            sweep_tag=tag,
            seed=seed,
            axis_values=axis_identity,
            status="linked" if ok else "failed",
        )

    if ok and run_id:
        update_recommendation_status(conn, recommendation_id=rec["id"], status="executed")
        observed_json, frontier_delta_json, outcome_label = build_frontier_delta(conn, campaign["id"], previous_best_wr, run_id)
        save_recommendation_outcome(
            conn,
            recommendation_id=rec["id"],
            run_id=run_id,
            observed_metrics_json=observed_json,
            frontier_delta_json=frontier_delta_json,
            outcome_label=outcome_label,
        )
        print(f"Recommendation executed: {rec['id']} -> run {run_id[:8]} ({outcome_label})")
    else:
        print(f"Recommendation execution failed: {rec['id']}")
