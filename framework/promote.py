#!/usr/bin/env python3
"""autoresearch promotion planner / executor.

Plan:   read campaign stage results, aggregate by candidate, rank, output decisions.
Execute: generate missing seeds for promoted candidates and run them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

# Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from core.db import (
    DB_PATH,
    aggregate_candidates_by_stage,
    close_campaign_stage,
    find_run_by_sweep_tag,
    get_campaign,
    get_campaign_runs_by_stage,
    get_campaign_stage,
    get_campaign_stages,
    get_search_space,
    init_db,
    link_run_to_campaign_v20,
    save_campaign_stage,
    save_promotion_decision,
)
from stage_policy import load_stage_policy, next_stage_name


def parse_args():
    p = argparse.ArgumentParser(description="autoresearch promotion planner")
    p.add_argument("--db", type=str, default=DB_PATH)
    p.add_argument("--campaign", type=str, required=True)
    p.add_argument("--stage-policy", type=str, required=True)
    p.add_argument("--from-stage", type=str, required=True)
    p.add_argument("--to-stage", type=str, required=True)
    p.add_argument("--plan", action="store_true", help="Output promotion plan only")
    p.add_argument("--execute", action="store_true", help="Execute promotion plan")
    p.add_argument("--train-script", type=str, default="domains/gomoku/train.py")
    p.add_argument("--eval-level", type=int, default=None)
    p.add_argument("--target-win-rate", type=float, default=None)
    p.add_argument("--parallel-games", type=int, default=None)
    return p.parse_args()


def _resolve_campaign(conn, name: str):
    row = get_campaign(conn, name)
    if not row:
        print(f"Campaign not found: {name}")
        sys.exit(1)
    return dict(row)


def _build_tag(campaign_name: str, stage: str, axis_values: dict, seed: int) -> str:
    parts = [f"{campaign_name}_{stage}"]
    short_map = {
        "num_blocks": "b",
        "num_filters": "f",
        "learning_rate": "lr",
        "steps_per_cycle": "s",
        "buffer_size": "buf",
    }
    for k, v in sorted(axis_values.items()):
        if k == "seed":
            continue
        short = short_map.get(k, k)
        parts.append(f"{short}{v}")
    parts.append(f"sd{seed}")
    return "_".join(parts)


def plan_promotion(conn, campaign_id: str, policy: dict,
                   from_stage: str, to_stage: str):
    """Aggregate candidates, rank, and return promotion decisions."""
    stage_cfg = None
    for s in policy["stages"]:
        if s["name"] == from_stage:
            stage_cfg = s
            break
    if stage_cfg is None:
        raise ValueError(f"stage '{from_stage}' not found in policy")

    promote_top_k = stage_cfg["promote_top_k"]
    seed_target = stage_cfg["seed_count"]
    min_runs = stage_cfg.get("min_runs", 1)
    metric = stage_cfg["metric"]

    # Aggregate candidates
    candidates = aggregate_candidates_by_stage(conn, campaign_id, from_stage)
    if not candidates:
        print(f"No completed candidates found for campaign {campaign_id[:8]} stage {from_stage}")
        return []

    # Build decision list
    decisions = []
    for rank, cand in enumerate(candidates, start=1):
        axis_values = json.loads(cand["axis_values_json"])
        seed_count = cand["seed_count"]
        run_count = cand["run_count"]
        mean_wr = cand["mean_wr"] or 0.0
        std_wr = cand["std_wr"] or 0.0

        # Determine decision
        if promote_top_k == 0:
            decision = "hold"
            reason = "promote_top_k=0 (handoff stage)"
        elif rank <= promote_top_k:
            if seed_count >= seed_target and run_count >= min_runs:
                decision = "promote"
                reason = (
                    f"rank #{rank} within top-{promote_top_k}, "
                    f"seed coverage {seed_count}/{seed_target} satisfied, "
                    f"runs {run_count}/{min_runs} satisfied"
                )
            else:
                decision = "hold"
                gaps = []
                if seed_count < seed_target:
                    gaps.append(f"seed coverage {seed_count}/{seed_target}")
                if run_count < min_runs:
                    gaps.append(f"runs {run_count}/{min_runs}")
                reason = (
                    f"rank #{rank} within top-{promote_top_k}, "
                    f"but insufficient data: {', '.join(gaps)}"
                )
        else:
            decision = "reject"
            reason = f"rank #{rank} outside top-{promote_top_k}"

        agg_metrics = {
            "mean_wr": round(mean_wr, 4),
            "std_wr": round(std_wr, 4),
            "min_wr": round(cand["min_wr"] or 0.0, 4),
            "max_wr": round(cand["max_wr"] or 0.0, 4),
            "mean_wall_s": round(cand["mean_wall_s"] or 0.0, 1),
            "games_total": cand["games_total"],
            "mean_params": round(cand["mean_params"] or 0.0, 0),
            "run_count": run_count,
            "seed_count": seed_count,
            "metric": metric,
        }

        decisions.append({
            "candidate_key": cand["candidate_key"],
            "axis_values": axis_values,
            "aggregated_metrics": agg_metrics,
            "seed_count": seed_count,
            "decision": decision,
            "decision_rank": rank,
            "reason": reason,
        })

    return decisions


def print_plan(decisions: list[dict], from_stage: str, to_stage: str):
    print(f"\nPromotion Plan: {from_stage} → {to_stage}")
    print("=" * 72)
    print(f"{'Rank':>4}  {'Decision':>8}  {'Seeds':>5}  {'Mean WR':>8}  {'Std WR':>7}  {'Reason'}")
    print("-" * 72)
    for d in decisions:
        rank = d["decision_rank"]
        dec = d["decision"]
        seeds = d["seed_count"]
        mwr = d["aggregated_metrics"]["mean_wr"]
        swr = d["aggregated_metrics"]["std_wr"]
        reason = d["reason"]
        print(f"{rank:>4}  {dec:>8}  {seeds:>5}  {mwr:>7.1%}  {swr:>6.1%}  {reason}")
    print()
    n_promote = sum(1 for d in decisions if d["decision"] == "promote")
    n_hold = sum(1 for d in decisions if d["decision"] == "hold")
    n_reject = sum(1 for d in decisions if d["decision"] == "reject")
    print(f"Summary: {n_promote} promote, {n_hold} hold, {n_reject} reject")
    print()


def _existing_seeds_for_candidate(conn, campaign_id: str,
                                  stage: str, candidate_key: str) -> set[int]:
    rows = get_campaign_runs_by_stage(conn, campaign_id, stage)
    seeds = set()
    for r in rows:
        if r.get("candidate_key") == candidate_key and r.get("seed") is not None:
            seeds.add(int(r["seed"]))
    return seeds


def execute_promotion(conn, campaign_id: str, policy: dict,
                      from_stage: str, to_stage: str,
                      args, decisions: list[dict]):
    """Run missing seeds for promoted candidates in the next stage."""
    to_stage_cfg = None
    for s in policy["stages"]:
        if s["name"] == to_stage:
            to_stage_cfg = s
            break
    if to_stage_cfg is None:
        raise ValueError(f"stage '{to_stage}' not found in policy")

    if to_stage == "D":
        print("\n⚠ Stage D execute is blocked in v20.2.")
        print("   Checkpoint continuation / branching belongs to v20.3.")
        sys.exit(1)

    time_budget = to_stage_cfg["time_budget"]
    seed_target = to_stage_cfg["seed_count"]

    # Collect work items: promoted candidates with missing seeds
    work_items = []
    for d in decisions:
        if d["decision"] != "promote":
            continue
        candidate_key = d["candidate_key"]
        axis_values = d["axis_values"]
        existing = _existing_seeds_for_candidate(conn, campaign_id, to_stage, candidate_key)
        # Also check if these seeds were already run in any stage for this candidate
        # but we specifically need them in the *target* stage.
        # However, for seed discipline, we look at the target stage only.
        # If seed already exists in target stage, skip.
        needed = []
        for seed in range(1, seed_target + 1):
            if seed not in existing:
                needed.append(seed)
        if needed:
            work_items.append({
                "axis_values": axis_values,
                "candidate_key": candidate_key,
                "needed_seeds": needed,
            })

    if not work_items:
        print("\nNo work needed: all promoted candidates already have required seeds in target stage.")
        return

    total_runs = sum(len(w["needed_seeds"]) for w in work_items)
    print(f"\nExecuting promotion: {len(work_items)} candidates, {total_runs} missing seeds")
    print(f"Target stage: {to_stage}, budget: {time_budget}s, seeds required: {seed_target}")
    print("=" * 60)

    # Open target stage record
    save_campaign_stage(
        conn,
        campaign_id=campaign_id,
        stage=to_stage,
        policy_json=json.dumps(to_stage_cfg, ensure_ascii=False, sort_keys=True),
        budget_json=json.dumps({"time_budget": time_budget}, ensure_ascii=False, sort_keys=True),
        seed_target=seed_target,
        status="open",
    )

    results = []
    for wi in work_items:
        axis_values = wi["axis_values"]
        for seed in wi["needed_seeds"]:
            tag = _build_tag(args.campaign, to_stage, axis_values, seed)
            cfg = {**axis_values, "seed": seed, "sweep_tag": tag}

            cmd = [sys.executable, args.train_script,
                   "--time-budget", str(time_budget),
                   "--seed", str(seed),
                   "--sweep-tag", tag]
            if args.db:
                cmd += ["--db", args.db]

            for k in ("num_blocks", "num_filters", "learning_rate",
                      "steps_per_cycle", "buffer_size"):
                if k in axis_values and axis_values[k] is not None:
                    cmd += [f"--{k.replace('_', '-')}", str(axis_values[k])]

            if args.eval_level is not None:
                cmd += ["--eval-level", str(args.eval_level)]
            if args.target_win_rate:
                cmd += ["--target-win-rate", str(args.target_win_rate)]
            if args.parallel_games:
                cmd += ["--parallel-games", str(args.parallel_games)]

            axis_desc = "  ".join(f"{k}={v}" for k, v in axis_values.items())
            print(f"\n[{'='*60}")
            print(f"  {tag}")
            print(f"  {axis_desc}  seed={seed}")
            print(f"{'='*60}")

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
            else:
                wr_line = ""
                for line in proc.stdout.splitlines():
                    if "Win rate:" in line or "win_rate" in line.lower():
                        wr_line = line.strip()
                if wr_line:
                    print(f"  {wr_line}")
                print(f"  完成，耗时 {elapsed:.0f}s")

            results.append((tag, ok, elapsed))

            # Link to campaign
            run_id = find_run_by_sweep_tag(conn, tag)
            if run_id:
                link_run_to_campaign_v20(
                    conn,
                    campaign_id=campaign_id,
                    run_id=run_id,
                    stage=to_stage,
                    sweep_tag=tag,
                    seed=seed,
                    axis_values=axis_values,
                    status="linked" if ok else "failed",
                )

    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_ok
    print(f"\n{'='*60}")
    print(f"Promotion execute 完成: {n_ok} 成功, {n_fail} 失败")
    print(f"{'='*60}")

    # Persist promotion decisions
    for d in decisions:
        save_promotion_decision(
            conn,
            campaign_id=campaign_id,
            from_stage=from_stage,
            to_stage=to_stage,
            candidate_key=d["candidate_key"],
            axis_values=d["axis_values"],
            aggregated_metrics=d["aggregated_metrics"],
            seed_count=d["seed_count"],
            decision=d["decision"],
            decision_rank=d["decision_rank"],
            reason=d["reason"],
        )

    # Close from_stage
    close_campaign_stage(conn, campaign_id, from_stage)


def main():
    args = parse_args()
    conn = init_db(args.db)

    campaign = _resolve_campaign(conn, args.campaign)
    campaign_id = campaign["id"]

    policy = load_stage_policy(args.stage_policy)

    # Verify policy domain matches campaign domain
    if policy["domain"] != campaign["domain"]:
        print(
            f"Error: policy domain '{policy['domain']}' != "
            f"campaign domain '{campaign['domain']}'"
        )
        sys.exit(1)

    # Verify policy search_space_ref matches campaign search space
    ssr = policy.get("search_space_ref", {})
    space_row = get_search_space(conn, campaign["search_space_id"])
    if space_row:
        if ssr.get("name") != space_row["name"] or ssr.get("version") != space_row["version"]:
            print(
                f"Error: policy search_space_ref ({ssr.get('name')} v{ssr.get('version')}) "
                f"does not match campaign search space ({space_row['name']} v{space_row['version']})"
            )
            sys.exit(1)
    else:
        print(f"Warning: campaign search space {campaign['search_space_id'][:8]} not found in DB")

    # Block Stage D execution in v20.2
    if args.execute and args.to_stage == "D":
        print("\n⚠ Stage D execute is blocked in v20.2.")
        print("   Checkpoint continuation / branching belongs to v20.3.")
        sys.exit(1)

    # Verify stage transition is valid
    expected_next = next_stage_name(policy, args.from_stage)
    if expected_next != args.to_stage:
        print(
            f"Error: invalid stage transition '{args.from_stage}' → '{args.to_stage}'. "
            f"Policy expects next stage to be '{expected_next}'."
        )
        sys.exit(1)

    # Ensure from_stage exists in ledger
    existing_stages = get_campaign_stages(conn, campaign_id)
    existing_names = {s["stage"] for s in existing_stages}
    if args.from_stage not in existing_names:
        # Auto-create from_stage entry if missing (e.g. Stage A created by sweep)
        from_cfg = None
        for s in policy["stages"]:
            if s["name"] == args.from_stage:
                from_cfg = s
                break
        if from_cfg:
            save_campaign_stage(
                conn,
                campaign_id=campaign_id,
                stage=args.from_stage,
                policy_json=json.dumps(from_cfg, ensure_ascii=False, sort_keys=True),
                budget_json=json.dumps(
                    {"time_budget": from_cfg["time_budget"]},
                    ensure_ascii=False, sort_keys=True
                ),
                seed_target=from_cfg["seed_count"],
                status="open",
            )

    decisions = plan_promotion(
        conn, campaign_id, policy,
        args.from_stage, args.to_stage
    )

    if not decisions:
        print("No candidates to evaluate.")
        sys.exit(0)

    print_plan(decisions, args.from_stage, args.to_stage)

    if args.execute:
        execute_promotion(
            conn, campaign_id, policy,
            args.from_stage, args.to_stage,
            args, decisions
        )

    conn.close()


if __name__ == "__main__":
    main()
