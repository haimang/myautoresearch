#!/usr/bin/env python3
"""Constrained Bayesian-style frontier refinement for the fx_spot mock domain."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(1, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from bayes_adapter import (  # noqa: E402
    MAXIMIZE,
    MINIMIZE,
    build_universe,
    candidate_key,
    encode_candidates,
    evaluate_candidate,
    feasibility_from_result,
    point_from_result,
    prior_for_candidate,
    save_observation,
    utility_from_result,
)
from framework.core.db import (  # noqa: E402
    get_or_create_campaign,
    init_db,
    save_experiment_run,
    save_objective_profile,
    save_search_space,
)
from framework.profiles.objective_profile import load_objective_profile  # noqa: E402
from framework.services.frontier.plotting import plot_pareto_artifacts  # noqa: E402
from framework.profiles.search_space import load_profile  # noqa: E402
from framework.services.bayes.benchmark import summarize_frontier  # noqa: E402
from framework.services.bayes.loop import run_strategy  # noqa: E402
from framework.services.frontier.pareto import compute_knee_point, pareto_front  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v23 constrained BO replay for fx_spot")
    parser.add_argument("--db", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--search-space", default="domains/fx_spot/search_space.json")
    parser.add_argument("--objective-profile", default="domains/fx_spot/objective_profile.json")
    parser.add_argument("--scenarios", default="cn_exporter_core,usd_importer_mix,global_diversified,asia_procurement_hub")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--seed-observations", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-universe", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args.search_space)
    objective = load_objective_profile(args.objective_profile)
    scenarios = [scenario.strip() for scenario in args.scenarios.split(",") if scenario.strip()]
    fx_run_id = args.run_id or datetime.now(timezone.utc).strftime("fx-bo-%Y%m%d-%H%M%S")
    workspace = Path(args.output_root) / "fx_spot" / fx_run_id
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "benchmarks").mkdir(exist_ok=True)
    db_path = args.db or str(workspace / "tracker.db")
    manifest = {
        "fx_run_id": fx_run_id,
        "domain": "fx_spot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "constrained_bootstrap_ridge_bo",
        "budget": args.budget,
        "seed_observations": args.seed_observations,
        "batch_size": args.batch_size,
        "scenarios": scenarios,
        "db": db_path,
    }
    (workspace / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    universe = build_universe(profile, scenarios, args.max_universe, args.seed)
    results_cache = {candidate["_key"]: evaluate_candidate(candidate) for candidate in universe}
    full_points = [point_from_result(candidate, results_cache[candidate["_key"]], candidate["_key"]) for candidate in universe]
    full_feasible = [point for point in full_points if point["liquidity_floor_ok"] >= 1.0]
    true_front, true_dominated = pareto_front(full_feasible, maximize=MAXIMIZE, minimize=MINIMIZE)
    true_front_keys = {point["run_full"] for point in true_front}
    true_knee = compute_knee_point(true_front, maximize=MAXIMIZE, minimize=MINIMIZE)

    conn = init_db(db_path)
    space_id = save_search_space(conn, profile)
    objective_id = save_objective_profile(conn, objective)
    save_experiment_run(
        conn,
        run_id=fx_run_id,
        domain="fx_spot",
        output_root=str(workspace),
        manifest=manifest,
        objective_profile_id=objective_id,
    )

    summaries = []
    for strategy_name, seed_offset in [("random", 1000), ("bayesian", 2000)]:
        campaign = get_or_create_campaign(
            conn,
            name=f"v23-{strategy_name}-{fx_run_id}",
            domain="fx_spot",
            train_script="domains/fx_spot/bayesian_refine.py",
            search_space_id=space_id,
            protocol={"eval_level": None, "eval_opponent": None, "is_benchmark": False},
            objective_profile_id=objective_id,
        )
        conn.execute("UPDATE campaigns SET experiment_run_id = ? WHERE id = ?", (fx_run_id, campaign["id"]))
        conn.commit()
        selected_keys, points = run_strategy(
            name=strategy_name,
            universe=universe,
            results_cache=results_cache,
            budget=args.budget,
            seed_observations=args.seed_observations,
            batch_size=args.batch_size,
            seed=args.seed + seed_offset,
            point_from_result=point_from_result,
            encode_candidates=encode_candidates,
            utility_from_result=utility_from_result,
            feasibility_from_result=feasibility_from_result,
            prior_for_candidate=prior_for_candidate,
        )
        for idx, key in enumerate(selected_keys, 1):
            candidate = next(candidate for candidate in universe if candidate["_key"] == key)
            artifact_dir = workspace / "campaigns" / campaign["name"] / "runs" / f"{idx:04d}"
            save_observation(
                conn,
                campaign=campaign,
                candidate=candidate,
                result=results_cache[key],
                sweep_tag=f"{campaign['name']}_{idx:04d}",
                seed=args.seed,
                artifact_dir=str(artifact_dir),
            )
        feasible = [point for point in points if point["liquidity_floor_ok"] >= 1.0]
        summary, front, dominated, knee = summarize_frontier(
            points=points,
            feasible_points=feasible,
            maximize=MAXIMIZE,
            minimize=MINIMIZE,
            true_front_keys=true_front_keys,
            true_knee_key=true_knee["run_full"] if true_knee else None,
        )
        campaign_pareto_dir = workspace / "campaigns" / campaign["name"] / "pareto"
        artifacts = plot_pareto_artifacts(
            front,
            dominated,
            output_path=str(campaign_pareto_dir / "overview.png"),
            x_key="embedded_spread_bps",
            y_key="liquidity_headroom_ratio",
            axis_meta=objective.get("display", {}),
            knee_point=knee,
            metrics=MAXIMIZE + MINIMIZE,
        )
        summary.update({"strategy": strategy_name, "overview_plot": artifacts["overview"]})
        summaries.append(summary)
    conn.close()

    summary_path = workspace / "benchmarks" / "benchmark_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    md_path = workspace / "benchmarks" / "benchmark_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# v23 FX Bayesian Replay Benchmark\n\n")
        handle.write(f"- universe: {len(universe)}\n")
        handle.write(f"- true_feasible: {len(full_feasible)}\n")
        handle.write(f"- true_front: {len(true_front)}\n")
        handle.write(f"- true_dominated: {len(true_dominated)}\n\n")
        handle.write("| strategy | budget | front_points | true_front_recall | best_uplift_bps | infeasible_rate |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summaries:
            handle.write(
                f"| {row['strategy']} | {row['budget']} | {row['front_points']} | "
                f"{row['true_front_recall']:.3f} | {row['best_uplift_bps']:.3f} | {row['infeasible_rate']:.3f} |\n"
            )
    print(f"FX run id: {fx_run_id}")
    print(f"Workspace: {workspace}")
    print(f"Benchmark summary: {summary_path}")
    print(json.dumps({"summaries": summaries, "true_front": len(true_front)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
