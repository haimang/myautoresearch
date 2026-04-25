#!/usr/bin/env python3
"""Constrained Bayesian-style frontier refinement for the fx_spot mock domain.

The loop is intentionally local and deterministic: it builds a finite quote-surface
candidate universe, evaluates a small seed set, fits bootstrapped ridge surrogates
for feasibility and utility, and uses uncertainty-aware acquisition to pick the
next batch. This is a practical v23 BO layer for the mixed categorical/numeric
FX search space without adding heavyweight dependencies.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import itertools
import json
import os
from pathlib import Path
import random
import sys
import uuid

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_fw_path = os.path.join(_PROJECT_ROOT, "framework")
if _fw_path not in sys.path:
    sys.path.insert(1, _fw_path)
os.chdir(_PROJECT_ROOT)

from core.db import (  # noqa: E402
    collect_hardware_info,
    create_run,
    finish_run,
    get_or_create_campaign,
    init_db,
    link_run_to_campaign_v20,
    save_experiment_run,
    save_fx_quote,
    save_fx_route_leg,
    save_objective_profile,
    save_quote_window,
    save_run_metrics,
    save_search_space,
)
from metrics import metric_rows  # noqa: E402
from mock_provider import MockQuoteProvider  # noqa: E402
from objective_profile import load_objective_profile  # noqa: E402
from pareto_plot import plot_pareto_artifacts  # noqa: E402
from portfolio import clone_floors, clone_portfolio  # noqa: E402
from quote_graph import route_for_candidate  # noqa: E402
from route_eval import evaluate_route  # noqa: E402
from search_space import load_profile  # noqa: E402
from treasury_scenarios import apply_treasury_scenario  # noqa: E402


MAXIMIZE = ["liquidity_headroom_ratio", "preservation_ratio", "spot_uplift_bps", "quote_validity_remaining_s"]
MINIMIZE = ["embedded_spread_bps", "route_leg_count", "settlement_lag_s", "locked_funds_ratio"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v23 constrained BO replay for fx_spot")
    p.add_argument("--db", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--output-root", default="output")
    p.add_argument("--search-space", default="domains/fx_spot/search_space.json")
    p.add_argument("--objective-profile", default="domains/fx_spot/objective_profile.json")
    p.add_argument("--scenarios", default="cn_exporter_core,usd_importer_mix,global_diversified,asia_procurement_hub")
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--seed-observations", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--max-universe", type=int, default=900)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _candidate_key(candidate: dict) -> str:
    return _stable_json(candidate)


def _axis_values(profile: dict, name: str) -> list:
    axis = profile["axes"][name]
    return axis.get("values", [axis.get("default")])


def build_universe(profile: dict, scenarios: list[str], max_universe: int, seed: int) -> list[dict]:
    axes = {
        "treasury_scenario": scenarios,
        "sell_currency": _axis_values(profile, "sell_currency"),
        "buy_currency": _axis_values(profile, "buy_currency"),
        "route_template": _axis_values(profile, "route_template"),
        "rebalance_fraction": _axis_values(profile, "rebalance_fraction"),
        "sell_amount_mode": _axis_values(profile, "sell_amount_mode"),
        "sell_amount_ratio": _axis_values(profile, "sell_amount_ratio"),
        "floor_buffer_target": _axis_values(profile, "floor_buffer_target"),
        "max_legs": _axis_values(profile, "max_legs"),
        "provider": _axis_values(profile, "provider"),
        "quote_scenario": _axis_values(profile, "quote_scenario"),
    }
    candidates = []
    for combo in itertools.product(*axes.values()):
        candidate = dict(zip(axes.keys(), combo))
        if candidate["sell_currency"] == candidate["buy_currency"]:
            continue
        template = candidate["route_template"]
        if template.startswith("via_"):
            bridge = template[4:].upper()
            if bridge in (candidate["sell_currency"], candidate["buy_currency"]):
                continue
        materialized = apply_treasury_scenario(candidate)
        holding = float(materialized.get("portfolio", {}).get(candidate["sell_currency"], 0.0))
        floor = float(materialized.get("liquidity_floors", {}).get(candidate["sell_currency"], 0.0))
        if holding <= 0.0:
            continue
        if candidate["sell_amount_mode"] == "surplus_fraction" and holding <= floor:
            continue
        if candidate["sell_amount_mode"] == "floor_probe" and holding <= floor:
            continue
        if candidate["sell_amount_mode"] == "explicit_ratio":
            if holding * float(candidate["sell_amount_ratio"]) <= 0.0:
                continue
        if candidate["sell_amount_mode"] == "floor_probe":
            probe_amount = max(0.0, holding - floor * (1.0 + float(candidate["floor_buffer_target"])))
            if probe_amount <= 0.0:
                continue
        # Avoid meaningless unused-mode Cartesian blowup.
        if candidate["sell_amount_mode"] != "explicit_ratio":
            candidate["sell_amount_ratio"] = 0.0
        if candidate["sell_amount_mode"] != "floor_probe":
            candidate["floor_buffer_target"] = 0.0
        candidates.append(candidate)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:max_universe]


def evaluate_candidate(candidate: dict) -> dict:
    materialized = apply_treasury_scenario(candidate)
    provider = MockQuoteProvider(scenario=materialized.get("quote_scenario", "base"))
    portfolio = clone_portfolio(materialized.get("portfolio"))
    floors = clone_floors(materialized.get("liquidity_floors"))
    anchor = materialized["anchor_currency"]
    return evaluate_route(
        candidate=materialized,
        portfolio=portfolio,
        floors=floors,
        anchor_currency=anchor,
        provider=provider,
    )


def utility(metrics: dict) -> float:
    feasible = metrics.get("liquidity_floor_ok", 0.0) >= 1.0
    if not feasible:
        return -250.0 - 25.0 * metrics.get("liquidity_breach_count", 1.0)
    return (
        metrics.get("spot_uplift_bps", 0.0)
        + 12.0 * metrics.get("liquidity_headroom_ratio", 0.0)
        - 0.35 * metrics.get("embedded_spread_bps", 0.0)
        - 0.01 * metrics.get("settlement_lag_s", 0.0)
        - 1.5 * metrics.get("route_leg_count", 0.0)
    )


def _pareto_front(points: list[dict]) -> tuple[list[dict], list[dict]]:
    front = []
    dominated = []
    for i, p in enumerate(points):
        is_dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            better_or_equal = True
            strict = False
            for key in MAXIMIZE:
                if q[key] < p[key]:
                    better_or_equal = False
                    break
                if q[key] > p[key]:
                    strict = True
            if not better_or_equal:
                continue
            for key in MINIMIZE:
                if q[key] > p[key]:
                    better_or_equal = False
                    break
                if q[key] < p[key]:
                    strict = True
            if better_or_equal and strict:
                is_dominated = True
                break
        (dominated if is_dominated else front).append(p)
    return front, dominated


def _compute_knee(front: list[dict]) -> dict | None:
    if not front:
        return None
    axes = MAXIMIZE + MINIMIZE
    ranges = {k: (min(p[k] for p in front), max(p[k] for p in front)) for k in axes}
    best = None
    best_dist = None
    for p in front:
        total = 0.0
        for k in axes:
            lo, hi = ranges[k]
            norm = 1.0 if hi == lo else ((p[k] - lo) / (hi - lo) if k in MAXIMIZE else (hi - p[k]) / (hi - lo))
            total += (1.0 - norm) ** 2
        dist = float(np.sqrt(total / len(axes)))
        if best_dist is None or dist < best_dist:
            best = p
            best_dist = dist
    return best


def point_from_result(candidate: dict, result: dict, key: str) -> dict:
    point = {
        "run": key[:8],
        "run_full": key,
        "label": key[:8],
        "axis_values": candidate,
        "route_signature": result["route_signature"],
        "route_family": result["route_family"],
        "breach_reasons": result["breach_reasons"],
    }
    point.update(result["metrics"])
    return point


def encode_candidates(candidates: list[dict], feature_schema: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    cats = ["treasury_scenario", "sell_currency", "buy_currency", "route_template", "sell_amount_mode", "quote_scenario"]
    nums = ["rebalance_fraction", "sell_amount_ratio", "floor_buffer_target", "max_legs"]
    if feature_schema is None:
        schema = []
        for key in cats:
            for value in sorted({str(c.get(key)) for c in candidates}):
                schema.append(f"{key}={value}")
        schema.extend(nums)
    else:
        schema = feature_schema
    rows = []
    for candidate in candidates:
        row = []
        for feature in schema:
            if "=" in feature:
                key, value = feature.split("=", 1)
                row.append(1.0 if str(candidate.get(key)) == value else 0.0)
            else:
                row.append(float(candidate.get(feature, 0.0) or 0.0))
        rows.append(row)
    return np.array(rows, dtype=float), schema


def fit_ensemble(X: np.ndarray, y: np.ndarray, *, seed: int, n_models: int = 24, ridge: float = 1e-3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    models = []
    n = X.shape[0]
    X_aug = np.concatenate([np.ones((n, 1)), X], axis=1)
    for _ in range(n_models):
        idx = rng.integers(0, n, size=max(n, 2))
        xb = X_aug[idx]
        yb = y[idx]
        reg = np.eye(xb.shape[1]) * ridge
        reg[0, 0] = 0.0
        beta = np.linalg.pinv(xb.T @ xb + reg) @ xb.T @ yb
        models.append(beta)
    return models


def predict_ensemble(models: list[np.ndarray], X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_aug = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    preds = np.stack([X_aug @ beta for beta in models], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def select_bayesian(
    universe: list[dict],
    observed_keys: set[str],
    observations: dict[str, dict],
    batch_size: int,
    seed: int,
) -> list[dict]:
    observed_candidates = [p["candidate"] for p in observations.values()]
    feature_schema = encode_candidates(universe)[1]
    X_obs, _ = encode_candidates(observed_candidates, feature_schema)
    y_util = np.array([utility(p["result"]["metrics"]) for p in observations.values()], dtype=float)
    y_feas = np.array([p["result"]["metrics"].get("liquidity_floor_ok", 0.0) for p in observations.values()], dtype=float)
    util_models = fit_ensemble(X_obs, y_util, seed=seed)
    feas_models = fit_ensemble(X_obs, y_feas, seed=seed + 1)

    unseen = [c for c in universe if _candidate_key(c) not in observed_keys]
    X_unseen, _ = encode_candidates(unseen, feature_schema)
    mu, sigma = predict_ensemble(util_models, X_unseen)
    p_feas, _ = predict_ensemble(feas_models, X_unseen)
    p_feas = np.clip(p_feas, 0.0, 1.0)
    priors = np.array([
        (6.0 if c.get("quote_scenario") == "uplift_corridor" else 0.0)
        + (2.0 if str(c.get("route_template", "")).startswith("via_") else 0.0)
        + (1.5 if c.get("sell_amount_mode") == "explicit_ratio" else 0.0)
        for c in unseen
    ])
    acq = p_feas * (mu + 2.75 * sigma + priors) - (1.0 - p_feas) * 24.0
    order = np.argsort(-acq)
    return [unseen[int(i)] for i in order[:batch_size]]


def save_observation(
    conn,
    *,
    campaign: dict,
    candidate: dict,
    result: dict,
    sweep_tag: str,
    seed: int,
    artifact_dir: str,
) -> str:
    run_id = str(uuid.uuid4())
    os.makedirs(artifact_dir, exist_ok=True)
    create_run(
        conn,
        run_id,
        {"sweep_tag": sweep_tag, "seed": seed, "time_budget": 1, "artifact_dir": artifact_dir},
        hardware=collect_hardware_info(),
        output_dir=artifact_dir,
        is_benchmark=False,
    )
    now = datetime.now(timezone.utc)
    qw_id = f"qw-{uuid.uuid4().hex[:16]}"
    materialized = apply_treasury_scenario(candidate)
    save_quote_window(
        conn,
        window_id=qw_id,
        campaign_id=campaign["id"],
        anchor_currency=materialized["anchor_currency"],
        started_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=30)).isoformat(),
        max_quote_age_seconds=1800,
        portfolio_snapshot_json=json.dumps(materialized["portfolio"], ensure_ascii=False, sort_keys=True),
        liquidity_floor_json=json.dumps(materialized["liquidity_floors"], ensure_ascii=False, sort_keys=True),
        provider_config_json=json.dumps({"provider": "mock", "scenario": candidate["quote_scenario"]}, sort_keys=True),
    )
    quote_ids = []
    for quote in result["quotes"]:
        quote = dict(quote)
        quote["quote_window_id"] = qw_id
        quote_ids.append(save_fx_quote(conn, quote))
    for leg, quote_ref in zip(result["legs"], quote_ids):
        save_fx_route_leg(
            conn,
            run_id=run_id,
            leg_index=leg["leg_index"],
            sell_currency=leg["sell_currency"],
            buy_currency=leg["buy_currency"],
            sell_amount=leg["sell_amount"],
            buy_amount=leg["buy_amount"],
            quote_ref=quote_ref,
            route_state_before_json=leg["route_state_before_json"],
            route_state_after_json=leg["route_state_after_json"],
        )
    save_run_metrics(conn, run_id, metric_rows(result["metrics"]))
    finish_run(
        conn,
        run_id,
        {
            "status": "completed",
            "total_cycles": 1,
            "total_games": 1,
            "total_steps": len(result["legs"]),
            "wall_time_s": 0.001,
            "final_win_rate": result["metrics"]["preservation_ratio"],
            "num_params": len(result["legs"]),
        },
    )
    link_run_to_campaign_v20(
        conn,
        campaign_id=campaign["id"],
        run_id=run_id,
        stage=None,
        sweep_tag=sweep_tag,
        seed=seed,
        axis_values=candidate,
    )
    with open(os.path.join(artifact_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "candidate": candidate, "result": result}, f, indent=2, ensure_ascii=False, sort_keys=True)
    return run_id


def run_strategy(
    *,
    name: str,
    universe: list[dict],
    results_cache: dict[str, dict],
    budget: int,
    seed_observations: int,
    batch_size: int,
    seed: int,
) -> tuple[list[str], list[dict]]:
    rng = random.Random(seed)
    observed: dict[str, dict] = {}
    selected: list[str] = []
    seed_batch = rng.sample(universe, min(seed_observations, budget, len(universe)))
    for candidate in seed_batch:
        key = _candidate_key(candidate)
        observed[key] = {"candidate": candidate, "result": results_cache[key]}
        selected.append(key)
    while len(selected) < min(budget, len(universe)):
        if name == "random":
            unseen = [c for c in universe if _candidate_key(c) not in observed]
            batch = rng.sample(unseen, min(batch_size, len(unseen), budget - len(selected)))
        else:
            batch = select_bayesian(universe, set(observed), observed, min(batch_size, budget - len(selected)), seed + len(selected))
        if not batch:
            break
        for candidate in batch:
            key = _candidate_key(candidate)
            observed[key] = {"candidate": candidate, "result": results_cache[key]}
            selected.append(key)
    points = [point_from_result(observed[k]["candidate"], observed[k]["result"], k) for k in selected]
    return selected, points


def main() -> None:
    args = parse_args()
    profile = load_profile(args.search_space)
    objective = load_objective_profile(args.objective_profile)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
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
    (workspace / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    universe = build_universe(profile, scenarios, args.max_universe, args.seed)
    results_cache = {_candidate_key(c): evaluate_candidate(c) for c in universe}
    full_points = [point_from_result(c, results_cache[_candidate_key(c)], _candidate_key(c)) for c in universe]
    full_feasible = [p for p in full_points if p["liquidity_floor_ok"] >= 1.0]
    true_front, true_dominated = _pareto_front(full_feasible)
    true_front_keys = {p["run_full"] for p in true_front}
    true_knee = _compute_knee(true_front)

    conn = init_db(db_path)
    space_id = save_search_space(conn, profile)
    objective_id = save_objective_profile(conn, objective)
    save_experiment_run(conn, run_id=fx_run_id, domain="fx_spot", output_root=str(workspace), manifest=manifest, objective_profile_id=objective_id)

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
        )
        for idx, key in enumerate(selected_keys, 1):
            candidate = json.loads(key)
            result = results_cache[key]
            artifact_dir = workspace / "campaigns" / campaign["name"] / "runs" / f"{idx:04d}"
            save_observation(
                conn,
                campaign=campaign,
                candidate=candidate,
                result=result,
                sweep_tag=f"{campaign['name']}_{idx:04d}",
                seed=args.seed,
                artifact_dir=str(artifact_dir),
            )
        feasible = [p for p in points if p["liquidity_floor_ok"] >= 1.0]
        front, dominated = _pareto_front(feasible)
        knee = _compute_knee(front)
        selected_front_keys = {p["run_full"] for p in front}
        hit_count = len(set(selected_keys) & true_front_keys)
        front_recall = hit_count / len(true_front_keys) if true_front_keys else 0.0
        best_uplift = max((p["spot_uplift_bps"] for p in feasible), default=0.0)
        best_preservation = max((p["preservation_ratio"] for p in feasible), default=0.0)
        infeasible_rate = 1.0 - (len(feasible) / len(points) if points else 0.0)
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
        summaries.append({
            "strategy": strategy_name,
            "budget": len(points),
            "front_points": len(front),
            "selected_true_front_hits": hit_count,
            "true_front_recall": front_recall,
            "best_uplift_bps": best_uplift,
            "best_preservation_ratio": best_preservation,
            "infeasible_rate": infeasible_rate,
            "knee_hit": bool(knee and true_knee and knee["run_full"] == true_knee["run_full"]),
            "overview_plot": artifacts["overview"],
        })
    conn.close()

    summary_path = workspace / "benchmarks" / "benchmark_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    md_path = workspace / "benchmarks" / "benchmark_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# v23 FX Bayesian Replay Benchmark\n\n")
        f.write(f"- universe: {len(universe)}\n")
        f.write(f"- true_feasible: {len(full_feasible)}\n")
        f.write(f"- true_front: {len(true_front)}\n")
        f.write(f"- true_dominated: {len(true_dominated)}\n\n")
        f.write("| strategy | budget | front_points | true_front_recall | best_uplift_bps | infeasible_rate |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summaries:
            f.write(
                f"| {row['strategy']} | {row['budget']} | {row['front_points']} | "
                f"{row['true_front_recall']:.3f} | {row['best_uplift_bps']:.3f} | {row['infeasible_rate']:.3f} |\n"
            )
    print(f"FX run id: {fx_run_id}")
    print(f"Workspace: {workspace}")
    print(f"Benchmark summary: {summary_path}")
    print(json.dumps({"summaries": summaries, "true_front": len(true_front)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
