"""Branch planning and execution services."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

from branch_policy import (
    apply_delta,
    get_allowed_protocol_changes,
    get_reason_config,
    load_branch_policy,
    reason_preserves_protocol,
    validate_delta,
)
from core.db import (
    bind_branch_child_run,
    get_branch_by_id,
    get_campaign,
    get_latest_checkpoint,
    get_recommendation_by_id,
    link_run_to_campaign_v20,
    save_recommendation_outcome,
    save_run_branch,
    update_branch_status,
    update_recommendation_status,
)

from .outcome_service import build_frontier_delta


def resolve_campaign(conn, name: str) -> dict:
    row = get_campaign(conn, name)
    if not row:
        raise ValueError(f"Campaign not found: {name}")
    return dict(row)


def resolve_parent_checkpoint(conn, campaign, args):
    if args.parent_checkpoint:
        checkpoint = conn.execute("SELECT * FROM checkpoints WHERE tag = ?", (args.parent_checkpoint,)).fetchone()
        if not checkpoint:
            raise ValueError(f"Checkpoint not found: {args.parent_checkpoint}")
        return dict(checkpoint), checkpoint["run_id"]

    stages = conn.execute("SELECT stage FROM campaign_stages WHERE campaign_id = ?", (campaign["id"],)).fetchall()
    stage_names = {row["stage"] for row in stages}
    from_stage = "C" if "C" in stage_names else "B" if "B" in stage_names else "A"
    rows = conn.execute(
        """SELECT cr.run_id FROM campaign_runs cr
           JOIN runs r ON r.id = cr.run_id
           WHERE cr.campaign_id = ? AND cr.stage = ?
             AND r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')
           ORDER BY r.final_win_rate DESC LIMIT 10""",
        (campaign["id"], from_stage),
    ).fetchall()
    if not rows:
        raise ValueError(f"No completed runs found in campaign '{campaign['name']}' stage {from_stage}")
    for row in rows:
        checkpoint = get_latest_checkpoint(conn, row["run_id"])
        if checkpoint:
            print(f"Auto-selected parent: run {row['run_id'][:8]}, checkpoint {checkpoint['tag']} (cycle {checkpoint['cycle']})")
            return checkpoint, row["run_id"]
        print(f"Skipping run {row['run_id'][:8]}: no checkpoint found, trying next best run...")
    raise ValueError(f"No run with a checkpoint found in campaign '{campaign['name']}' stage {from_stage}")


def get_parent_params(conn, run_id: str) -> dict:
    row = conn.execute(
        """SELECT num_res_blocks, num_filters, learning_rate,
                   replay_buffer_size, train_steps_per_cycle,
                   mcts_simulations, eval_level, time_budget, seed
            FROM runs WHERE id = ?""",
        (run_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Run not found: {run_id}")
    return {
        "num_blocks": row["num_res_blocks"],
        "num_filters": row["num_filters"],
        "learning_rate": row["learning_rate"],
        "buffer_size": row["replay_buffer_size"],
        "steps_per_cycle": row["train_steps_per_cycle"],
        "mcts_simulations": row["mcts_simulations"],
        "eval_level": row["eval_level"],
        "time_budget": row["time_budget"],
        "seed": row["seed"],
    }


def build_branch_id(campaign_id: str, parent_run_id: str, reason: str, delta_json: str) -> str:
    base = f"{campaign_id}_{parent_run_id}_{reason}_{delta_json}"
    return f"br-{hashlib.sha256(base.encode()).hexdigest()[:16]}"


def build_child_tag(campaign_name: str, parent_tag: str, reason: str, delta_json: str) -> str:
    delta_hash = hashlib.sha256(delta_json.encode()).hexdigest()[:8]
    return f"{campaign_name}_branch_{reason}_{delta_hash}_{parent_tag}"


def build_child_cmd(child_params: dict, parent_run_id: str, parent_ckpt_tag: str, tag: str, args, train_script: str) -> list[str]:
    cmd = [
        sys.executable,
        train_script,
        "--resume",
        parent_run_id,
        "--resume-checkpoint-tag",
        parent_ckpt_tag,
        "--sweep-tag",
        tag,
        "--time-budget",
        str(args.time_budget),
    ]
    param_map = {
        "num_blocks": "--num-blocks",
        "num_filters": "--num-filters",
        "learning_rate": "--learning-rate",
        "steps_per_cycle": "--steps-per-cycle",
        "buffer_size": "--buffer-size",
        "mcts_simulations": "--mcts-sims",
        "eval_level": "--eval-level",
        "seed": "--seed",
    }
    for key, flag in param_map.items():
        if key in child_params and child_params[key] is not None:
            cmd += [flag, str(child_params[key])]
    if args.eval_level is not None:
        cmd += ["--eval-level", str(args.eval_level)]
    if args.db:
        cmd += ["--db", args.db]
    return cmd


def infer_delta(parent: dict, child: dict) -> dict:
    delta = {}
    for key, value in child.items():
        if key in parent and parent[key] != value:
            delta[key] = value
    return delta


def plan_branches(conn, campaign, policy, parent_ckpt, parent_run_id, args) -> list[dict]:
    parent_params = get_parent_params(conn, parent_run_id)
    plans = []
    delta_overrides = {}
    if args.delta:
        for index, raw in enumerate(args.delta):
            if index < len(args.reason):
                delta_overrides[args.reason[index]] = json.loads(raw)

    for reason in args.reason:
        config = get_reason_config(policy, reason)
        if config is None:
            raise ValueError(f"Error: unknown branch reason '{reason}'")

        override = delta_overrides.get(reason, {})
        if override:
            validate_delta(reason, override, policy)
        child_params = apply_delta(parent_params, reason, policy, override)

        if reason == "seed_recheck" and (child_params.get("seed") is None or child_params.get("seed") == parent_params.get("seed")):
            parent_seed = parent_params.get("seed") or 0
            new_seed = (parent_seed % 997) + 1
            if new_seed == parent_seed:
                new_seed = (new_seed % 997) + 1
            child_params["seed"] = new_seed

        if not reason_preserves_protocol(policy, reason):
            allowed_changes = get_allowed_protocol_changes(policy, reason)
            for key in override:
                if key in ("eval_level", "eval_opponent") and key not in allowed_changes:
                    raise ValueError(f"Error: reason '{reason}' not allowed to change '{key}'")
        else:
            for key in ("eval_level", "eval_opponent"):
                if key in child_params and key in parent_params and child_params[key] != parent_params[key]:
                    raise ValueError(
                        f"Error: reason '{reason}' declares preserves_protocol=true, "
                        f"but delta would change '{key}' from {parent_params[key]} to {child_params[key]}"
                    )

        delta_json = json.dumps(override if override else infer_delta(parent_params, child_params), ensure_ascii=False, sort_keys=True)
        plans.append(
            {
                "branch_id": build_branch_id(campaign["id"], parent_run_id, reason, delta_json),
                "reason": reason,
                "parent_run_id": parent_run_id,
                "parent_checkpoint_id": parent_ckpt["id"],
                "child_params": child_params,
                "delta_json": delta_json,
                "tag": build_child_tag(campaign["name"], parent_ckpt["tag"], reason, delta_json),
            }
        )
    return plans


def print_plan(plans: list[dict], parent_tag: str):
    print(f"\nBranch Plan: from checkpoint {parent_tag}")
    print("=" * 72)
    print(f"{'#':>3}  {'Reason':>18}  {'Tag':>30}  {'Delta'}")
    print("-" * 72)
    for index, plan in enumerate(plans, 1):
        print(f"{index:>3}  {plan['reason']:>18}  {plan['tag']:>30}  {plan['delta_json'][:40]}")
    print(f"\nTotal branches: {len(plans)}\n")


def execute_branches(conn, campaign, plans: list[dict], parent_ckpt: dict, args, recommendation_by_branch_id: dict[str, dict] | None = None):
    print(f"\nExecuting {len(plans)} branches...")
    print("=" * 72)
    for plan in plans:
        branch_id = plan["branch_id"]
        reason = plan["reason"]
        tag = plan["tag"]
        child_params = plan["child_params"]
        recommendation = recommendation_by_branch_id.get(branch_id) if recommendation_by_branch_id else None
        previous_best_wr = None
        if recommendation:
            best_row = conn.execute(
                """SELECT MAX(r.final_win_rate) AS best_wr
                   FROM campaign_runs cr
                   JOIN runs r ON r.id = cr.run_id
                   WHERE cr.campaign_id = ?
                     AND r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')""",
                (campaign["id"],),
            ).fetchone()
            previous_best_wr = best_row["best_wr"] if best_row else None

        existing = get_branch_by_id(conn, branch_id)
        if existing and existing.get("child_run_id") and existing["status"] in ("completed", "running"):
            print(f"  Skip {reason}: already has child run {existing['child_run_id'][:8]} ({existing['status']})")
            continue

        if not existing:
            save_run_branch(
                conn,
                branch_id=branch_id,
                campaign_id=campaign["id"],
                parent_run_id=plan["parent_run_id"],
                parent_checkpoint_id=plan["parent_checkpoint_id"],
                from_stage="D",
                branch_reason=reason,
                branch_params_json=json.dumps(child_params, ensure_ascii=False, sort_keys=True),
                delta_json=plan["delta_json"],
                status="planned",
            )

        command = build_child_cmd(child_params, plan["parent_run_id"], parent_ckpt["tag"] if parent_ckpt else "", tag, args, args.train_script)
        print(f"\n[{'=' * 60}")
        print(f"  {tag}")
        print(f"  reason={reason}  delta={plan['delta_json']}")
        print(f"{'=' * 60}")
        update_branch_status(conn, branch_id=branch_id, status="running")
        started = time.time()
        proc = subprocess.run(command, env={**os.environ, 'PYTHONUNBUFFERED': '1'}, capture_output=True, text=True)
        elapsed = time.time() - started
        if proc.returncode != 0:
            print(f"  失败 (退出码 {proc.returncode})")
            if proc.stderr:
                for line in proc.stderr.strip().split("\n")[-5:]:
                    print(f"  {line}")
            update_branch_status(conn, branch_id=branch_id, status="failed")
            continue

        for line in proc.stdout.splitlines():
            if "Win rate:" in line or "win_rate" in line.lower():
                print(f"  {line.strip()}")
        print(f"  完成，耗时 {elapsed:.0f}s")
        child_row = conn.execute("SELECT id FROM runs WHERE sweep_tag = ? ORDER BY started_at DESC LIMIT 1", (tag,)).fetchone()
        if not child_row:
            update_branch_status(conn, branch_id=branch_id, status="completed")
            continue

        child_run_id = child_row["id"]
        bind_branch_child_run(conn, branch_id=branch_id, child_run_id=child_run_id, status="running")
        update_branch_status(conn, branch_id=branch_id, status="completed", result_summary_json=json.dumps({"elapsed_s": round(elapsed, 1)}))
        print(f"  Linked child run: {child_run_id[:8]}")
        child_axis_values = {key: value for key, value in child_params.items() if key not in {"seed", "time_budget"}}
        link_run_to_campaign_v20(
            conn,
            campaign_id=campaign["id"],
            run_id=child_run_id,
            stage="D",
            sweep_tag=tag,
            seed=child_params.get("seed"),
            axis_values=child_axis_values,
            status="linked",
        )
        if recommendation:
            update_recommendation_status(conn, recommendation_id=recommendation["id"], status="executed")
            observed_json, frontier_delta_json, outcome_label = build_frontier_delta(conn, campaign["id"], previous_best_wr, child_run_id)
            save_recommendation_outcome(
                conn,
                recommendation_id=recommendation["id"],
                run_id=child_run_id,
                branch_id=branch_id,
                observed_metrics_json=observed_json,
                frontier_delta_json=frontier_delta_json,
                outcome_label=outcome_label,
            )
            print(f"  Recommendation executed: {recommendation['id'][:16]} ({outcome_label})")
    print("\n" + "=" * 72)
    print("Branch execution complete.")


def resolve_branch_policy_path(project_root: str, campaign: dict, explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path
    default_path = os.path.join(project_root, "domains", campaign["domain"], "branch_policy.json")
    if os.path.isfile(default_path):
        return default_path
    raise ValueError(f"No branch policy found for domain '{campaign['domain']}'")


def plan_from_recommendation(conn, campaign, policy, rec: dict) -> tuple[list[dict], dict]:
    row = None
    if rec.get("parent_checkpoint_id") is not None:
        row = conn.execute("SELECT * FROM checkpoints WHERE id = ?", (rec["parent_checkpoint_id"],)).fetchone()
    parent_ckpt = dict(row) if row else None
    if parent_ckpt is None and rec.get("parent_run_id"):
        parent_ckpt = get_latest_checkpoint(conn, rec["parent_run_id"])
    if parent_ckpt is None:
        raise ValueError(f"Recommendation {rec['id']} has no resolvable parent checkpoint")

    parent_run_id = rec["parent_run_id"]
    reason = rec["branch_reason"]
    parent_params = get_parent_params(conn, parent_run_id)
    override = json.loads(rec.get("delta_json") or "{}")
    child_params = apply_delta(parent_params, reason, policy, override)
    if reason == "seed_recheck" and (child_params.get("seed") is None or child_params.get("seed") == parent_params.get("seed")):
        parent_seed = parent_params.get("seed") or 0
        new_seed = (parent_seed % 997) + 1
        if new_seed == parent_seed:
            new_seed = (new_seed % 997) + 1
        child_params["seed"] = new_seed

    delta_json = rec.get("delta_json") or json.dumps(infer_delta(parent_params, child_params), ensure_ascii=False, sort_keys=True)
    plan = {
        "branch_id": build_branch_id(campaign["id"], parent_run_id, reason, delta_json),
        "reason": reason,
        "parent_run_id": parent_run_id,
        "parent_checkpoint_id": parent_ckpt["id"],
        "child_params": child_params,
        "delta_json": delta_json,
        "tag": build_child_tag(campaign["name"], parent_ckpt["tag"], reason, delta_json),
    }
    return [plan], parent_ckpt


def execute_branch_recommendation(conn, args, *, project_root: str) -> None:
    recommendation = get_recommendation_by_id(conn, args.execute_recommendation)
    if not recommendation:
        raise ValueError(f"Recommendation not found: {args.execute_recommendation}")
    if recommendation["status"] != "accepted":
        raise ValueError(f"Recommendation {recommendation['id']} status is '{recommendation['status']}', expected 'accepted'")
    if recommendation["candidate_type"] not in ("continue_branch", "eval_upgrade"):
        raise ValueError(f"Recommendation {recommendation['id']} is '{recommendation['candidate_type']}', use sweep.py for point execution")

    campaign = resolve_campaign(conn, recommendation["campaign_id"])
    if not args.train_script:
        args.train_script = campaign["train_script"]
    policy_path = resolve_branch_policy_path(project_root, campaign, args.branch_policy)
    policy = load_branch_policy(policy_path)
    plans, parent_ckpt = plan_from_recommendation(conn, campaign, policy, recommendation)
    execute_branches(conn, campaign, plans, parent_ckpt, args, recommendation_by_branch_id={plans[0]["branch_id"]: recommendation})
