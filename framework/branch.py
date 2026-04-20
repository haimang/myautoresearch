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
    get_search_space,
    init_db,
    link_run_to_campaign,
    list_branches_for_campaign,
    save_run_branch,
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
    p.add_argument("--campaign", type=str, required=True)
    p.add_argument("--branch-policy", type=str, required=True)
    p.add_argument("--parent-checkpoint", type=str, default=None,
                   help="Explicit checkpoint tag to branch from; default: latest")
    p.add_argument("--reason", type=str, action="append", required=True,
                   help="Branch reason(s) to apply (can specify multiple)")
    p.add_argument("--delta", type=str, action="append", default=None,
                   help="Optional JSON delta override per reason (same order as --reason)")
    p.add_argument("--plan", action="store_true", help="Output branch plan only")
    p.add_argument("--dry-run", action="store_true", help="With --plan: show but do not persist")
    p.add_argument("--execute", action="store_true", help="Execute branch plan")
    p.add_argument("--train-script", type=str, default="domains/gomoku/train.py")
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
           ORDER BY r.final_win_rate DESC LIMIT 1""",
        (campaign["id"], from_stage),
    ).fetchall()

    if not rows:
        print(f"No completed runs found in campaign '{campaign['name']}' stage {from_stage}")
        sys.exit(1)

    run_id = rows[0]["run_id"]
    ckpt = get_latest_checkpoint(conn, run_id)
    if not ckpt:
        print(f"No checkpoint found for run {run_id[:8]}")
        sys.exit(1)

    print(f"Auto-selected parent: run {run_id[:8]}, checkpoint {ckpt['tag']} (cycle {ckpt['cycle']})")
    return ckpt, run_id


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


def _build_child_tag(campaign_name: str, parent_tag: str, reason: str) -> str:
    """Construct a sweep_tag for the child run."""
    return f"{campaign_name}_branch_{reason}_{parent_tag}"


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
            "tag": _build_child_tag(campaign["name"], parent_ckpt["tag"], reason),
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


def execute_branches(conn, campaign, policy, plans: list[dict], parent_ckpt: dict, args):
    """Launch child continuation runs and record them in the branch ledger."""
    print(f"\nExecuting {len(plans)} branches...")
    print("=" * 72)

    for p in plans:
        branch_id = p["branch_id"]
        reason = p["reason"]
        tag = p["tag"]
        child_params = p["child_params"]

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

        # Find the newly created child run by sweep_tag
        child_row = conn.execute(
            "SELECT id FROM runs WHERE sweep_tag = ?",
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
            # v20.3: also link child run into campaign_runs for unified campaign lineage
            link_run_to_campaign(
                conn,
                campaign_id=campaign["id"],
                run_id=child_run_id,
                stage="D",
                sweep_tag=tag,
                seed=child_params.get("seed"),
                axis_values={"branch_reason": reason, "parent_run_id": p["parent_run_id"]},
                status="linked",
            )
        else:
            update_branch_status(conn, branch_id=branch_id, status="completed")

    print("\n" + "=" * 72)
    print("Branch execution complete.")


def main():
    args = parse_args()
    conn = init_db(args.db)

    campaign = _resolve_campaign(conn, args.campaign)
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
