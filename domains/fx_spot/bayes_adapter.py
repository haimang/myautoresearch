"""FX-spot adapter for generic Bayesian-style refinement services."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import itertools
import json
import os
import uuid

import numpy as np

from core.db import (
    collect_hardware_info,
    create_run,
    finish_run,
    link_run_to_campaign_v20,
    save_fx_quote,
    save_fx_route_leg,
    save_quote_window,
    save_run_metrics,
)
from metrics import metric_rows
from mock_provider import MockQuoteProvider
from portfolio import clone_floors, clone_portfolio
from quote_graph import route_for_candidate
from route_eval import evaluate_route
from treasury_scenarios import apply_treasury_scenario

MAXIMIZE = ["liquidity_headroom_ratio", "preservation_ratio", "spot_uplift_bps", "quote_validity_remaining_s"]
MINIMIZE = ["embedded_spread_bps", "route_leg_count", "settlement_lag_s", "locked_funds_ratio"]


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def candidate_key(candidate: dict) -> str:
    return stable_json({k: v for k, v in candidate.items() if not k.startswith("_")})


def axis_values(profile: dict, name: str) -> list:
    axis = profile["axes"][name]
    return axis.get("values", [axis.get("default")])


def build_universe(profile: dict, scenarios: list[str], max_universe: int, seed: int) -> list[dict]:
    axes = {
        "treasury_scenario": scenarios,
        "sell_currency": axis_values(profile, "sell_currency"),
        "buy_currency": axis_values(profile, "buy_currency"),
        "route_template": axis_values(profile, "route_template"),
        "rebalance_fraction": axis_values(profile, "rebalance_fraction"),
        "sell_amount_mode": axis_values(profile, "sell_amount_mode"),
        "sell_amount_ratio": axis_values(profile, "sell_amount_ratio"),
        "floor_buffer_target": axis_values(profile, "floor_buffer_target"),
        "max_legs": axis_values(profile, "max_legs"),
        "provider": axis_values(profile, "provider"),
        "quote_scenario": axis_values(profile, "quote_scenario"),
    }
    candidates: list[dict] = []
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
        if candidate["sell_amount_mode"] in {"surplus_fraction", "floor_probe"} and holding <= floor:
            continue
        if candidate["sell_amount_mode"] == "explicit_ratio" and holding * float(candidate["sell_amount_ratio"]) <= 0.0:
            continue
        if candidate["sell_amount_mode"] == "floor_probe":
            probe_amount = max(0.0, holding - floor * (1.0 + float(candidate["floor_buffer_target"])))
            if probe_amount <= 0.0:
                continue
        if candidate["sell_amount_mode"] != "explicit_ratio":
            candidate["sell_amount_ratio"] = 0.0
        if candidate["sell_amount_mode"] != "floor_probe":
            candidate["floor_buffer_target"] = 0.0
        candidate["_key"] = candidate_key(candidate)
        candidates.append(candidate)
    rng = np.random.default_rng(seed)
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


def utility_from_result(result: dict) -> float:
    metrics = result["metrics"]
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


def feasibility_from_result(result: dict) -> float:
    return float(result["metrics"].get("liquidity_floor_ok", 0.0))


def prior_for_candidate(candidate: dict) -> float:
    return (
        (6.0 if candidate.get("quote_scenario") == "uplift_corridor" else 0.0)
        + (2.0 if str(candidate.get("route_template", "")).startswith("via_") else 0.0)
        + (1.5 if candidate.get("sell_amount_mode") == "explicit_ratio" else 0.0)
    )


def point_from_result(candidate: dict, result: dict, key: str) -> dict:
    point = {
        "run": key[:8],
        "run_full": key,
        "label": key[:8],
        "axis_values": {k: v for k, v in candidate.items() if not k.startswith("_")},
        "route_signature": result["route_signature"],
        "route_family": result["route_family"],
        "breach_reasons": result["breach_reasons"],
    }
    point.update(result["metrics"])
    return point


def encode_candidates(
    candidates: list[dict],
    feature_schema: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    cats = ["treasury_scenario", "sell_currency", "buy_currency", "route_template", "sell_amount_mode", "quote_scenario"]
    nums = ["rebalance_fraction", "sell_amount_ratio", "floor_buffer_target", "max_legs"]
    if feature_schema is None:
        schema: list[str] = []
        for key in cats:
            for value in sorted({str(candidate.get(key)) for candidate in candidates}):
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
        payload = dict(quote)
        payload["quote_window_id"] = qw_id
        quote_ids.append(save_fx_quote(conn, payload))
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
        axis_values={k: v for k, v in candidate.items() if not k.startswith("_")},
    )
    with open(os.path.join(artifact_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump({"run_id": run_id, "candidate": candidate, "result": result}, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return run_id
