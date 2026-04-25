"""Promotion planning and execution services."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from core.db import (
    aggregate_candidates_by_stage,
    close_campaign_stage,
    find_run_by_sweep_tag,
    get_campaign,
    get_campaign_runs_by_stage,
    link_run_to_campaign_v20,
    save_campaign_stage,
    save_promotion_decision,
)


def resolve_campaign(conn, name: str) -> dict:
    row = get_campaign(conn, name)
    if not row:
        raise ValueError(f"Campaign not found: {name}")
    return dict(row)


def build_tag(campaign_name: str, stage: str, axis_values: dict, seed: int) -> str:
    parts = [f"{campaign_name}_{stage}"]
    short_map = {
        "num_blocks": "b",
        "num_filters": "f",
        "learning_rate": "lr",
        "steps_per_cycle": "s",
        "buffer_size": "buf",
    }
    for key, value in sorted(axis_values.items()):
        if key == "seed":
            continue
        parts.append(f"{short_map.get(key, key)}{value}")
    parts.append(f"sd{seed}")
    return "_".join(parts)


def plan_promotion(conn, campaign_id: str, policy: dict, from_stage: str, to_stage: str) -> list[dict]:
    stage_cfg = next((stage for stage in policy["stages"] if stage["name"] == from_stage), None)
    if stage_cfg is None:
        raise ValueError(f"stage '{from_stage}' not found in policy")

    promote_top_k = stage_cfg["promote_top_k"]
    seed_target = stage_cfg["seed_count"]
    min_runs = stage_cfg.get("min_runs", 1)
    metric = stage_cfg["metric"]
    candidates = aggregate_candidates_by_stage(conn, campaign_id, from_stage)
    if not candidates:
        print(f"No completed candidates found for campaign {campaign_id[:8]} stage {from_stage}")
        return []

    decisions = []
    for rank, candidate in enumerate(candidates, start=1):
        axis_values = json.loads(candidate["axis_values_json"])
        seed_count = candidate["seed_count"]
        run_count = candidate["run_count"]
        if promote_top_k == 0:
            decision = "hold"
            reason = "promote_top_k=0 (handoff stage)"
        elif rank <= promote_top_k:
            if seed_count >= seed_target and run_count >= min_runs:
                decision = "promote"
                reason = f"rank #{rank} within top-{promote_top_k}, seed coverage {seed_count}/{seed_target} satisfied, runs {run_count}/{min_runs} satisfied"
            else:
                decision = "hold"
                gaps = []
                if seed_count < seed_target:
                    gaps.append(f"seed coverage {seed_count}/{seed_target}")
                if run_count < min_runs:
                    gaps.append(f"runs {run_count}/{min_runs}")
                reason = f"rank #{rank} within top-{promote_top_k}, but insufficient data: {', '.join(gaps)}"
        else:
            decision = "reject"
            reason = f"rank #{rank} outside top-{promote_top_k}"

        decisions.append(
            {
                "candidate_key": candidate["candidate_key"],
                "axis_values": axis_values,
                "aggregated_metrics": {
                    "mean_wr": round(candidate["mean_wr"] or 0.0, 4),
                    "std_wr": round(candidate["std_wr"] or 0.0, 4),
                    "min_wr": round(candidate["min_wr"] or 0.0, 4),
                    "max_wr": round(candidate["max_wr"] or 0.0, 4),
                    "mean_wall_s": round(candidate["mean_wall_s"] or 0.0, 1),
                    "games_total": candidate["games_total"],
                    "mean_params": round(candidate["mean_params"] or 0.0, 0),
                    "run_count": run_count,
                    "seed_count": seed_count,
                    "metric": metric,
                },
                "seed_count": seed_count,
                "decision": decision,
                "decision_rank": rank,
                "reason": reason,
            }
        )
    return decisions


def print_plan(decisions: list[dict], from_stage: str, to_stage: str) -> None:
    print(f"\nPromotion Plan: {from_stage} → {to_stage}")
    print("=" * 72)
    print(f"{'Rank':>4}  {'Decision':>8}  {'Seeds':>5}  {'Mean WR':>8}  {'Std WR':>7}  {'Reason'}")
    print("-" * 72)
    for decision in decisions:
        print(
            f"{decision['decision_rank']:>4}  {decision['decision']:>8}  {decision['seed_count']:>5}  "
            f"{decision['aggregated_metrics']['mean_wr']:>7.1%}  {decision['aggregated_metrics']['std_wr']:>6.1%}  {decision['reason']}"
        )
    n_promote = sum(1 for decision in decisions if decision["decision"] == "promote")
    n_hold = sum(1 for decision in decisions if decision["decision"] == "hold")
    n_reject = sum(1 for decision in decisions if decision["decision"] == "reject")
    print(f"\nSummary: {n_promote} promote, {n_hold} hold, {n_reject} reject\n")


def existing_seeds_for_candidate(conn, campaign_id: str, stage: str, candidate_key: str) -> set[int]:
    rows = get_campaign_runs_by_stage(conn, campaign_id, stage)
    return {int(row["seed"]) for row in rows if row.get("candidate_key") == candidate_key and row.get("seed") is not None}


def execute_promotion(conn, campaign_id: str, policy: dict, from_stage: str, to_stage: str, args, decisions: list[dict]) -> None:
    to_stage_cfg = next((stage for stage in policy["stages"] if stage["name"] == to_stage), None)
    if to_stage_cfg is None:
        raise ValueError(f"stage '{to_stage}' not found in policy")
    if to_stage == "D":
        raise ValueError("Stage D execute is blocked in v20.2.")

    time_budget = to_stage_cfg["time_budget"]
    seed_target = to_stage_cfg["seed_count"]
    work_items = []
    for decision in decisions:
        if decision["decision"] != "promote":
            continue
        existing = existing_seeds_for_candidate(conn, campaign_id, to_stage, decision["candidate_key"])
        needed = [seed for seed in range(1, seed_target + 1) if seed not in existing]
        if needed:
            work_items.append({"axis_values": decision["axis_values"], "candidate_key": decision["candidate_key"], "needed_seeds": needed})

    if not work_items:
        print("\nNo work needed: all promoted candidates already have required seeds in target stage.")
        return

    total_runs = sum(len(item["needed_seeds"]) for item in work_items)
    print(f"\nExecuting promotion: {len(work_items)} candidates, {total_runs} missing seeds")
    print(f"Target stage: {to_stage}, budget: {time_budget}s, seeds required: {seed_target}")
    print("=" * 60)

    save_campaign_stage(
        conn,
        campaign_id=campaign_id,
        stage=to_stage,
        policy_json=json.dumps(to_stage_cfg, ensure_ascii=False, sort_keys=True),
        budget_json=json.dumps({"time_budget": time_budget}, ensure_ascii=False, sort_keys=True),
        seed_target=seed_target,
        status="open",
    )

    for decision in decisions:
        save_promotion_decision(
            conn,
            campaign_id=campaign_id,
            from_stage=from_stage,
            to_stage=to_stage,
            candidate_key=decision["candidate_key"],
            axis_values=decision["axis_values"],
            aggregated_metrics=decision["aggregated_metrics"],
            seed_count=decision["seed_count"],
            decision=decision["decision"],
            decision_rank=decision["decision_rank"],
            reason=decision["reason"],
        )

    results = []
    for item in work_items:
        axis_values = item["axis_values"]
        for seed in item["needed_seeds"]:
            tag = build_tag(args.campaign, to_stage, axis_values, seed)
            cmd = [sys.executable, args.train_script, "--time-budget", str(time_budget), "--seed", str(seed), "--sweep-tag", tag]
            if args.db:
                cmd += ["--db", args.db]
            for key in ("num_blocks", "num_filters", "learning_rate", "steps_per_cycle", "buffer_size"):
                if key in axis_values and axis_values[key] is not None:
                    cmd += [f"--{key.replace('_', '-')}", str(axis_values[key])]
            if args.eval_level is not None:
                cmd += ["--eval-level", str(args.eval_level)]
            if args.target_win_rate:
                cmd += ["--target-win-rate", str(args.target_win_rate)]
            if args.parallel_games:
                cmd += ["--parallel-games", str(args.parallel_games)]

            axis_desc = "  ".join(f"{k}={v}" for k, v in axis_values.items())
            print(f"\n[{'=' * 60}")
            print(f"  {tag}")
            print(f"  {axis_desc}  seed={seed}")
            print(f"{'=' * 60}")
            started = time.time()
            proc = subprocess.run(cmd, env={**os.environ, 'PYTHONUNBUFFERED': '1'}, capture_output=True, text=True)
            elapsed = time.time() - started
            if proc.returncode != 0:
                print(f"  失败 (退出码 {proc.returncode})")
                if proc.stderr:
                    for line in proc.stderr.strip().split("\n")[-5:]:
                        print(f"  {line}")
                ok = False
            else:
                ok = True
                for line in proc.stdout.splitlines():
                    if "Win rate:" in line or "win_rate" in line.lower():
                        print(f"  {line.strip()}")
                print(f"  完成，耗时 {elapsed:.0f}s")
            results.append((tag, ok, elapsed))
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

    print(f"\n{'=' * 60}")
    print(f"Promotion execute 完成: {sum(1 for _, ok, _ in results if ok)} 成功, {sum(1 for _, ok, _ in results if not ok)} 失败")
    print(f"{'=' * 60}")
    close_campaign_stage(conn, campaign_id, from_stage)
