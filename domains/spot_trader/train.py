#!/usr/bin/env python3
"""Spot FX quote-surface mock execution entrypoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import sys
import time
import uuid

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(1, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from framework.core.db import (  # noqa: E402
    DB_PATH,
    collect_hardware_info,
    create_run,
    finish_run,
    init_db,
    save_fx_quote,
    save_fx_route_leg,
    save_quote_window,
    save_run_metrics,
)
from metrics import metric_rows  # noqa: E402
from mock_provider import MockQuoteProvider  # noqa: E402
from portfolio import DEFAULT_ANCHOR, clone_floors, clone_portfolio  # noqa: E402
from route_eval import evaluate_route  # noqa: E402
from treasury_scenarios import apply_treasury_scenario  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spot FX quote-surface mock train/eval")
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--time-budget", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sweep-tag", default=None)
    p.add_argument("--campaign-id", default=None)
    p.add_argument("--candidate-json", default=None)
    p.add_argument("--sell-currency", default="EUR")
    p.add_argument("--buy-currency", default=DEFAULT_ANCHOR)
    p.add_argument("--route-template", default="direct")
    p.add_argument("--rebalance-fraction", type=float, default=0.5)
    p.add_argument("--max-legs", type=int, default=2)
    p.add_argument("--provider", default="mock")
    p.add_argument("--quote-scenario", default="base")
    p.add_argument("--run-id", default=None, help="Filesystem-level spot_trader experiment run id")
    p.add_argument("--artifact-root", default=None, help="Run-scoped artifact workspace root")
    return p.parse_args()


def _candidate_from_args(args: argparse.Namespace) -> dict:
    if args.candidate_json:
        data = json.loads(args.candidate_json)
        if not isinstance(data, dict):
            raise ValueError("--candidate-json must be a JSON object")
        return data
    return {
        "sell_currency": args.sell_currency,
        "buy_currency": args.buy_currency,
        "route_template": args.route_template,
        "rebalance_fraction": args.rebalance_fraction,
        "max_legs": args.max_legs,
        "provider": args.provider,
        "quote_scenario": args.quote_scenario,
    }


def main() -> None:
    args = parse_args()
    start = time.time()
    candidate = apply_treasury_scenario(_candidate_from_args(args))
    provider = MockQuoteProvider(scenario=candidate.get("quote_scenario", args.quote_scenario))
    portfolio = clone_portfolio(candidate.get("portfolio"))
    floors = clone_floors(candidate.get("liquidity_floors"))
    anchor = candidate.get("anchor_currency", DEFAULT_ANCHOR)

    run_id = str(uuid.uuid4())
    artifact_dir = None
    if args.artifact_root:
        safe_campaign = args.campaign_id or "standalone"
        artifact_dir = os.path.join(args.artifact_root, "runs", run_id)
        os.makedirs(artifact_dir, exist_ok=True)
        os.makedirs(os.path.join(args.artifact_root, "campaigns", safe_campaign), exist_ok=True)
    elif args.run_id:
        artifact_dir = os.path.join("output", "spot_trader", args.run_id, "runs", run_id)
        os.makedirs(artifact_dir, exist_ok=True)
    conn = init_db(args.db)
    create_run(
        conn,
        run_id,
        {
            "sweep_tag": args.sweep_tag,
            "seed": args.seed,
            "time_budget": args.time_budget,
            "artifact_dir": artifact_dir,
        },
        hardware=collect_hardware_info(),
        output_dir=artifact_dir,
        is_benchmark=False,
    )

    quote_window_id = None
    if args.campaign_id:
        now = datetime.now(timezone.utc)
        quote_window_id = f"qw-{uuid.uuid4().hex[:16]}"
        save_quote_window(
            conn,
            window_id=quote_window_id,
            campaign_id=args.campaign_id,
            anchor_currency=anchor,
            started_at=now.isoformat(),
            expires_at=(now + timedelta(minutes=30)).isoformat(),
            max_quote_age_seconds=1800,
            portfolio_snapshot_json=json.dumps(portfolio, ensure_ascii=False, sort_keys=True),
            liquidity_floor_json=json.dumps(floors, ensure_ascii=False, sort_keys=True),
            provider_config_json=json.dumps({"provider": provider.provider_name, "scenario": provider.scenario}, sort_keys=True),
            status="open",
        )

    try:
        result = evaluate_route(
            candidate=candidate,
            portfolio=portfolio,
            floors=floors,
            anchor_currency=anchor,
            provider=provider,
        )
        quote_ids: list[str | None] = []
        for quote in result["quotes"]:
            if quote_window_id:
                quote["quote_window_id"] = quote_window_id
                quote_ref = save_fx_quote(conn, quote)
            else:
                quote_ref = None
            quote_ids.append(quote_ref)
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
        if artifact_dir:
            with open(os.path.join(artifact_dir, "result.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fx_run_id": args.run_id,
                        "db_run_id": run_id,
                        "sweep_tag": args.sweep_tag,
                        "candidate": candidate,
                        "route": result["route"],
                        "route_signature": result["route_signature"],
                        "route_family": result["route_family"],
                        "breach_reasons": result["breach_reasons"],
                        "metrics": result["metrics"],
                        "portfolio_before": result["portfolio_before"],
                        "portfolio_after": result["portfolio_after"],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
        finish_run(
            conn,
            run_id,
            {
                "status": "completed",
                "total_cycles": 1,
                "total_games": 1,
                "total_steps": len(result["legs"]),
                "wall_time_s": time.time() - start,
                "final_win_rate": result["metrics"]["preservation_ratio"],
            },
        )
        print(f"Run: {run_id[:8]} ({args.sweep_tag or '-'})")
        if args.run_id:
            print(f"FX run id: {args.run_id}")
        print(f"Preservation ratio: {result['metrics']['preservation_ratio']:.6f}")
        print(f"Liquidity floor ok: {result['metrics']['liquidity_floor_ok']:.0f}")
    except Exception:
        finish_run(conn, run_id, {"status": "failed", "wall_time_s": time.time() - start})
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
