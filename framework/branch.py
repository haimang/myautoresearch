#!/usr/bin/env python3
"""autoresearch branch planner / executor — v20.3 Continuation / Trajectory Explorer.

Plan:   read parent checkpoint, apply branch reason deltas, output branch plan.
Execute: launch child continuation runs and record them in run_branches ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from core.db import (
    DB_PATH,
    bind_branch_child_run,
    find_checkpoint_by_tag,
    get_branch_by_id,
    get_campaign,
    get_campaign_stage,
    get_campaign_stages,
    get_latest_checkpoint,
    get_recommendation_by_id,
    get_search_space,
    init_db,
    link_run_to_campaign_v20,
    list_branches_for_campaign,
    save_recommendation_outcome,
    save_run_branch,
    update_recommendation_status,
    update_branch_status,
)
from branch_policy import (
    apply_delta,
    get_allowed_protocol_changes,
    get_reason_config,
    load_branch_policy,
    reason_preserves_protocol,
    validate_delta,
)


def parse_args():
    p = argparse.ArgumentParser(description="autoresearch branch planner / executor")
    p.add_argument("--db", type=str, default=DB_PATH)
    p.add_argument("--campaign", type=str, default=None)
    p.add_argument("--branch-policy", type=str, default=None)
    p.add_argument("--parent-checkpoint", type=str, default=None,
                   help="Explicit checkpoint tag to branch from; default: latest")
    p.add_argument("--reason", type=str, action="append", default=None,
                   help="Branch reason(s) to apply (can specify multiple)")
    p.add_argument("--delta", type=str, action="append", default=None,
                   help="Optional JSON delta override per reason (same order as --reason)")
    p.add_argument("--plan", action="store_true", help="Output branch plan only")
    p.add_argument("--dry-run", action="store_true", help="With --plan: show but do not persist")
    p.add_argument("--execute", action="store_true", help="Execute branch plan")
    p.add_argument("--execute-recommendation", type=str, default=None,
                   help="Execute an accepted branch recommendation by id (v21.1)")
    p.add_argument("--train-script", type=str, default=None)
    p.add_argument("--time-budget", type=int, default=60,
                   help="Time budget for each child continuation run")
    p.add_argument("--eval-level", type=int, default=None)
    return p.parse_args()


def _resolve_campaign(conn, name: str):
    row = get_campaign(conn, name)
    if not row:
        print(f"Campaign not found: {name}")
        sys.exit(1)
    return dict(row)


def _resolve_parent_checkpoint(conn, campaign, args):
    """Resolve the parent checkpoint to branch from."""
    # Find Stage D-ready candidates (or use explicit parent)
    if args.parent_checkpoint:
        ckpt = find_checkpoint_by_tag(conn, args.parent_checkpoint)
        if not ckpt:
            print(f"Checkpoint not found: {args.parent_checkpoint}")
            sys.exit(1)
        run_id = ckpt["run_id"]
        return ckpt, run_id

    # Auto-select: find latest checkpoint from campaign's Stage C (or D-ready) runs
    stages = get_campaign_stages(conn, campaign["id"])
    stage_names = {s["stage"] for s in stages}
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
        print(f"No completed runs found in campaign '{campaign['name']}' stage {from_stage}")
        sys.exit(1)

    for row in rows:
        run_id = row["run_id"]
        ckpt = get_latest_checkpoint(conn, run_id)
        if ckpt:
            print(f"Auto-selected parent: run {run_id[:8]}, checkpoint {ckpt['tag']} (cycle {ckpt['cycle']})")
            return ckpt, run_id
        print(f"Skipping run {run_id[:8]}: no checkpoint found, trying next best run...")

    print(f"No run with a checkpoint found in campaign '{campaign['name']}' stage {from_stage}")
    sys.exit(1)


def _get_parent_params(conn, run_id: str) -> dict:
    """Extract parent run parameters as a dict suitable for delta application."""
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


def _build_branch_id(campaign_id: str, parent_run_id: str, reason: str, delta_json: str) -> str:
    """Stable branch id from campaign + parent + reason + delta."""
    base = f"{campaign_id}_{parent_run_id}_{reason}_{delta_json}"
    h = hashlib.sha256(base.encode()).hexdigest()[:16]
    return f"br-{h}"


def _build_child_tag(campaign_name: str, parent_tag: str, reason: str, delta_json: str) -> str:
    """Construct a unique sweep_tag for the child run.

    Includes a short hash of the delta JSON to disambiguate branches
    with the same parent + reason but different delta values.
    """
    delta_hash = hashlib.sha256(delta_json.encode()).hexdigest()[:8]
    return f"{campaign_name}_branch_{reason}_{delta_hash}_{parent_tag}"


def _build_child_cmd(child_params: dict, parent_run_id: str, parent_ckpt_tag: str, tag: str,
                     args, train_script: str) -> list[str]:
    """Build the train.py command for a child continuation run."""
    cmd = [sys.executable, train_script,
           "--resume", parent_run_id,
           "--resume-checkpoint-tag", parent_ckpt_tag,
           "--sweep-tag", tag,
           "--time-budget", str(args.time_budget)]

    # Apply delta params (keys match branch_policy.json allowed_deltas names)
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


def plan_branches(conn, campaign, policy, parent_ckpt, parent_run_id, args) -> list[dict]:
    """Generate branch plan entries for each requested reason."""
    parent_params = _get_parent_params(conn, parent_run_id)
    plans = []

    delta_overrides = {}
    if args.delta:
        for i, d in enumerate(args.delta):
            if i < len(args.reason):
                try:
                    delta_overrides[args.reason[i]] = json.loads(d)
                except json.JSONDecodeError as exc:
                    print(f"Invalid delta JSON for reason '{args.reason[i]}': {exc}")
                    sys.exit(1)

    for reason in args.reason:
        # Validate reason exists
        cfg = get_reason_config(policy, reason)
        if cfg is None:
            print(f"Error: unknown branch reason '{reason}'")
            sys.exit(1)

        # Build delta
        override = delta_overrides.get(reason, {})
        try:
            if override:
                validate_delta(reason, override, policy)
            child_params = apply_delta(parent_params, reason, policy, override)
        except ValueError as exc:
            print(f"Error applying delta for '{reason}': {exc}")
            sys.exit(1)

        # H-4: seed_recheck with null default → assign deterministic seed
        if reason == "seed_recheck" and (
            child_params.get("seed") is None
            or child_params.get("seed") == parent_params.get("seed")
        ):
            parent_seed = parent_params.get("seed") or 0
            new_seed = (parent_seed % 997) + 1
            # Avoid accidental collision with parent seed
            if new_seed == parent_seed:
                new_seed = (new_seed % 997) + 1
            child_params["seed"] = new_seed

        # Protocol guard: check override AND final child_params against parent
        if not reason_preserves_protocol(policy, reason):
            allowed_changes = get_allowed_protocol_changes(policy, reason)
            for key in override:
                if key in ("eval_level", "eval_opponent") and key not in allowed_changes:
                    print(f"Error: reason '{reason}' not allowed to change '{key}'")
                    sys.exit(1)
        else:
            # Reason claims protocol preservation — verify no protocol drift in final params
            PROTOCOL_FIELDS = ("eval_level", "eval_opponent")
            for key in PROTOCOL_FIELDS:
                if key in child_params and key in parent_params:
                    if child_params[key] != parent_params[key]:
                        print(
                            f"Error: reason '{reason}' declares preserves_protocol=true, "
                            f"but delta would change '{key}' from {parent_params[key]} to {child_params[key]}"
                        )
                        sys.exit(1)

        delta_json = json.dumps(override if override else _infer_delta(parent_params, child_params), ensure_ascii=False, sort_keys=True)
        branch_id = _build_branch_id(campaign["id"], parent_run_id, reason, delta_json)

        plans.append({
            "branch_id": branch_id,
            "reason": reason,
            "parent_run_id": parent_run_id,
            "parent_checkpoint_id": parent_ckpt["id"],
            "child_params": child_params,
            "delta_json": delta_json,
            "tag": _build_child_tag(campaign["name"], parent_ckpt["tag"], reason, delta_json),
        })

    return plans


def _infer_delta(parent: dict, child: dict) -> dict:
    """Infer a simple delta dict from parent→child param changes."""
    delta = {}
    for k, v in child.items():
        if k in parent and parent[k] != v:
            delta[k] = v
    return delta


def print_plan(plans: list[dict], parent_tag: str):
    print(f"\nBranch Plan: from checkpoint {parent_tag}")
    print("=" * 72)
    print(f"{'#':>3}  {'Reason':>18}  {'Tag':>30}  {'Delta'}")
    print("-" * 72)
    for i, p in enumerate(plans, 1):
        delta_short = p["delta_json"][:40]
        print(f"{i:>3}  {p['reason']:>18}  {p['tag']:>30}  {delta_short}")
    print(f"\nTotal branches: {len(plans)}")
    print()


def _build_frontier_delta(conn, campaign_id: str, previous_best_wr: float | None, run_id: str) -> tuple[str, str, str]:
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


def execute_branches(conn, campaign, policy, plans: list[dict], parent_ckpt: dict, args,
                     recommendation_by_branch_id: dict[str, dict] | None = None):
    """Launch child continuation runs and record them in the branch ledger."""
    print(f"\nExecuting {len(plans)} branches...")
    print("=" * 72)

    for p in plans:
        branch_id = p["branch_id"]
        reason = p["reason"]
        tag = p["tag"]
        child_params = p["child_params"]
        rec = recommendation_by_branch_id.get(branch_id) if recommendation_by_branch_id else None
        previous_best_wr = None
        if rec:
            best_row = conn.execute(
                """SELECT MAX(r.final_win_rate) AS best_wr
                   FROM campaign_runs cr
                   JOIN runs r ON r.id = cr.run_id
                   WHERE cr.campaign_id = ?
                     AND r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')""",
                (campaign["id"],),
            ).fetchone()
            previous_best_wr = best_row["best_wr"] if best_row else None

        # Check if this exact branch already has a completed child run
        existing = get_branch_by_id(conn, branch_id)
        if existing and existing.get("child_run_id") and existing["status"] in ("completed", "running"):
            print(f"  Skip {reason}: already has child run {existing['child_run_id'][:8]} ({existing['status']})")
            continue

        # Persist planned branch
        if not existing:
            save_run_branch(
                conn,
                branch_id=branch_id,
                campaign_id=campaign["id"],
                parent_run_id=p["parent_run_id"],
                parent_checkpoint_id=p["parent_checkpoint_id"],
                from_stage="D",
                branch_reason=reason,
                branch_params_json=json.dumps(child_params, ensure_ascii=False, sort_keys=True),
                delta_json=p["delta_json"],
                status="planned",
            )

        # Build and run child command
        ckpt_tag = parent_ckpt["tag"] if parent_ckpt else ""
        cmd = _build_child_cmd(child_params, p["parent_run_id"], ckpt_tag, tag, args, args.train_script)
        print(f"\n[{'='*60}")
        print(f"  {tag}")
        print(f"  reason={reason}  delta={p['delta_json']}")
        print(f"{'='*60}")

        # Update status to running
        update_branch_status(conn, branch_id=branch_id, status="running")

        t0 = time.time()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        elapsed = time.time() - t0
        ok = proc.returncode == 0

        if not ok:
            print(f"  失败 (退出码 {proc.returncode})")
            if proc.stderr:
                for line in proc.stderr.strip().split("\n")[-5:]:
                    print(f"  {line}")
            update_branch_status(conn, branch_id=branch_id, status="failed")
            continue

        # Extract child run info from output
        wr_line = ""
        for line in proc.stdout.splitlines():
            if "Win rate:" in line or "win_rate" in line.lower():
                wr_line = line.strip()
        if wr_line:
            print(f"  {wr_line}")
        print(f"  完成，耗时 {elapsed:.0f}s")

        # Find the newly created child run by sweep_tag (most recent first)
        child_row = conn.execute(
            "SELECT id FROM runs WHERE sweep_tag = ? ORDER BY started_at DESC LIMIT 1",
            (tag,),
        ).fetchone()
        if child_row:
            child_run_id = child_row["id"]
            bind_branch_child_run(conn, branch_id=branch_id, child_run_id=child_run_id, status="running")
            update_branch_status(
                conn,
                branch_id=branch_id,
                status="completed",
                result_summary_json=json.dumps({"elapsed_s": round(elapsed, 1)}),
            )
            print(f"  Linked child run: {child_run_id[:8]}")
            # v20.3: link child run into campaign_runs with proper hyperparam axis_values
            # Exclude non-identity fields (seed, time_budget) so candidate_key is meaningful
            child_axis_values = {
                k: v for k, v in child_params.items()
                if k not in {"seed", "time_budget"}
            }
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
            if rec:
                update_recommendation_status(conn, recommendation_id=rec["id"], status="executed")
                observed_json, frontier_delta_json, outcome_label = _build_frontier_delta(
                    conn, campaign["id"], previous_best_wr, child_run_id
                )
                save_recommendation_outcome(
                    conn,
                    recommendation_id=rec["id"],
                    run_id=child_run_id,
                    branch_id=branch_id,
                    observed_metrics_json=observed_json,
                    frontier_delta_json=frontier_delta_json,
                    outcome_label=outcome_label,
                )
                print(f"  Recommendation executed: {rec['id'][:16]} ({outcome_label})")
        else:
            update_branch_status(conn, branch_id=branch_id, status="completed")

    print("\n" + "=" * 72)
    print("Branch execution complete.")


def _resolve_branch_policy_path(campaign: dict, explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path
    default_path = os.path.join(_PROJECT_ROOT, "domains", campaign["domain"], "branch_policy.json")
    if os.path.isfile(default_path):
        return default_path
    raise ValueError(f"No branch policy found for domain '{campaign['domain']}'")


def _plan_from_recommendation(conn, campaign, policy, rec: dict, args) -> tuple[list[dict], dict]:
    row = None
    if rec.get("parent_checkpoint_id") is not None:
        row = conn.execute(
            "SELECT * FROM checkpoints WHERE id = ?",
            (rec["parent_checkpoint_id"],),
        ).fetchone()
    parent_ckpt = dict(row) if row else None
    if parent_ckpt is None and rec.get("parent_run_id"):
        parent_ckpt = get_latest_checkpoint(conn, rec["parent_run_id"])
    if parent_ckpt is None:
        raise ValueError(f"Recommendation {rec['id']} has no resolvable parent checkpoint")

    parent_run_id = rec["parent_run_id"]
    reason = rec["branch_reason"]
    parent_params = _get_parent_params(conn, parent_run_id)
    override = json.loads(rec.get("delta_json") or "{}")
    child_params = apply_delta(parent_params, reason, policy, override)

    if reason == "seed_recheck" and (
        child_params.get("seed") is None
        or child_params.get("seed") == parent_params.get("seed")
    ):
        parent_seed = parent_params.get("seed") or 0
        new_seed = (parent_seed % 997) + 1
        if new_seed == parent_seed:
            new_seed = (new_seed % 997) + 1
        child_params["seed"] = new_seed

    delta_json = rec.get("delta_json") or json.dumps(
        _infer_delta(parent_params, child_params),
        ensure_ascii=False,
        sort_keys=True,
    )
    branch_id = _build_branch_id(campaign["id"], parent_run_id, reason, delta_json)
    plan = {
        "branch_id": branch_id,
        "reason": reason,
        "parent_run_id": parent_run_id,
        "parent_checkpoint_id": parent_ckpt["id"],
        "child_params": child_params,
        "delta_json": delta_json,
        "tag": _build_child_tag(campaign["name"], parent_ckpt["tag"], reason, delta_json),
    }
    return [plan], parent_ckpt


def _execute_branch_recommendation(args) -> None:
    conn = init_db(args.db)
    rec = get_recommendation_by_id(conn, args.execute_recommendation)
    if not rec:
        print(f"Recommendation not found: {args.execute_recommendation}")
        conn.close()
        sys.exit(1)
    if rec["status"] != "accepted":
        print(f"Recommendation {rec['id']} status is '{rec['status']}', expected 'accepted'")
        conn.close()
        sys.exit(1)
    if rec["candidate_type"] not in ("continue_branch", "eval_upgrade"):
        print(f"Recommendation {rec['id']} is '{rec['candidate_type']}', use sweep.py for point execution")
        conn.close()
        sys.exit(1)

    campaign = _resolve_campaign(conn, rec["campaign_id"])
    if not args.train_script:
        args.train_script = campaign["train_script"]
    try:
        policy_path = _resolve_branch_policy_path(campaign, args.branch_policy)
    except ValueError as exc:
        print(f"Error: {exc}")
        conn.close()
        sys.exit(1)
    policy = load_branch_policy(policy_path)

    plans, parent_ckpt = _plan_from_recommendation(conn, campaign, policy, rec, args)
    execute_branches(
        conn,
        campaign,
        policy,
        plans,
        parent_ckpt,
        args,
        recommendation_by_branch_id={plans[0]["branch_id"]: rec},
    )
    conn.close()


def main():
    args = parse_args()

    if args.execute_recommendation:
        _execute_branch_recommendation(args)
        return

    conn = init_db(args.db)

    if not args.campaign:
        print("Error: --campaign is required unless --execute-recommendation is used")
        conn.close()
        sys.exit(1)

    campaign = _resolve_campaign(conn, args.campaign)
    if not args.train_script:
        args.train_script = campaign["train_script"]
    if not args.branch_policy:
        print("Error: --branch-policy is required unless --execute-recommendation is used")
        conn.close()
        sys.exit(1)
    if not args.reason:
        print("Error: at least one --reason is required unless --execute-recommendation is used")
        conn.close()
        sys.exit(1)

    policy = load_branch_policy(args.branch_policy)

    # Verify policy domain matches campaign domain
    if policy["domain"] != campaign["domain"]:
        print(f"Error: policy domain '{policy['domain']}' != campaign domain '{campaign['domain']}'")
        sys.exit(1)

    # Verify search_space compatibility
    ssr = policy.get("search_space_ref", {})
    space_row = get_search_space(conn, campaign["search_space_id"])
    if space_row:
        if ssr.get("name") != space_row["name"] or ssr.get("version") != space_row["version"]:
            print(
                f"Error: policy search_space_ref ({ssr.get('name')} v{ssr.get('version')}) "
                f"does not match campaign search space ({space_row['name']} v{space_row['version']})"
            )
            sys.exit(1)

    # Verify stage_policy_ref domain compatibility
    # NOTE: campaign_stages.policy_json stores per-stage config (name=A/B/C/D),
    # not the stage policy metadata. We only enforce domain alignment here;
    # name/version consistency is ensured by policy author at config time.
    spr = policy.get("stage_policy_ref", {})
    if spr.get("domain") and spr["domain"] != campaign["domain"]:
        print(
            f"Error: policy stage_policy_ref domain '{spr['domain']}' "
            f"does not match campaign domain '{campaign['domain']}'"
        )
        sys.exit(1)

    parent_ckpt, parent_run_id = _resolve_parent_checkpoint(conn, campaign, args)

    plans = plan_branches(conn, campaign, policy, parent_ckpt, parent_run_id, args)

    if not plans:
        print("No branches to plan.")
        conn.close()
        return

    print_plan(plans, parent_ckpt["tag"])

    if args.dry_run:
        print("(dry-run: branches not persisted)")
        conn.close()
        return

    if args.plan and not args.execute:
        # Persist planned branches
        for p in plans:
            existing = get_branch_by_id(conn, p["branch_id"])
            if not existing:
                save_run_branch(
                    conn,
                    branch_id=p["branch_id"],
                    campaign_id=campaign["id"],
                    parent_run_id=p["parent_run_id"],
                    parent_checkpoint_id=p["parent_checkpoint_id"],
                    from_stage="D",
                    branch_reason=p["reason"],
                    branch_params_json=json.dumps(p["child_params"], ensure_ascii=False, sort_keys=True),
                    delta_json=p["delta_json"],
                    status="planned",
                )
        print(f"Persisted {len(plans)} planned branches.")

    if args.execute:
        execute_branches(conn, campaign, policy, plans, parent_ckpt, args)

    conn.close()


if __name__ == "__main__":
    main()
