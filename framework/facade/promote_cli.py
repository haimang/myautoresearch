#!/usr/bin/env python3
"""autoresearch promotion planner / executor — facade over promotion services."""

from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, _FRAMEWORK_DIR)

from framework.core.db import DB_PATH, get_campaign_stages, get_search_space, init_db, save_campaign_stage
from framework.policies.stage_policy import load_stage_policy, next_stage_name
from framework.services.research.promotion_service import execute_promotion, plan_promotion, print_plan, resolve_campaign


def parse_args():
    parser = argparse.ArgumentParser(description="autoresearch promotion planner")
    parser.add_argument("--db", type=str, default=DB_PATH)
    parser.add_argument("--campaign", type=str, required=True)
    parser.add_argument("--stage-policy", type=str, required=True)
    parser.add_argument("--from-stage", type=str, required=True)
    parser.add_argument("--to-stage", type=str, required=True)
    parser.add_argument("--plan", action="store_true", help="Output promotion plan only")
    parser.add_argument("--execute", action="store_true", help="Execute promotion plan")
    parser.add_argument("--train-script", type=str, default="domains/gomoku/train.py")
    parser.add_argument("--eval-level", type=int, default=None)
    parser.add_argument("--target-win-rate", type=float, default=None)
    parser.add_argument("--parallel-games", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    conn = init_db(args.db)
    try:
        try:
            campaign = resolve_campaign(conn, args.campaign)
        except ValueError as exc:
            print(str(exc))
            sys.exit(1)
        campaign_id = campaign["id"]
        policy = load_stage_policy(args.stage_policy)

        if policy["domain"] != campaign["domain"]:
            print(f"Error: policy domain '{policy['domain']}' != campaign domain '{campaign['domain']}'")
            sys.exit(1)

        ref = policy.get("search_space_ref", {})
        space_row = get_search_space(conn, campaign["search_space_id"])
        if space_row:
            if ref.get("name") != space_row["name"] or ref.get("version") != space_row["version"]:
                print(
                    f"Error: policy search_space_ref ({ref.get('name')} v{ref.get('version')}) "
                    f"does not match campaign search space ({space_row['name']} v{space_row['version']})"
                )
                sys.exit(1)
        else:
            print(f"Warning: campaign search space {campaign['search_space_id'][:8]} not found in DB")

        if args.execute and args.to_stage == "D":
            print("\n⚠ Stage D execute is blocked in v20.2.")
            print("   Checkpoint continuation / branching belongs to v20.3.")
            sys.exit(1)

        expected_next = next_stage_name(policy, args.from_stage)
        if expected_next != args.to_stage:
            print(f"Error: invalid stage transition '{args.from_stage}' → '{args.to_stage}'. Policy expects next stage to be '{expected_next}'.")
            sys.exit(1)

        existing_names = {stage["stage"] for stage in get_campaign_stages(conn, campaign_id)}
        if args.from_stage not in existing_names:
            from_cfg = next((stage for stage in policy["stages"] if stage["name"] == args.from_stage), None)
            if from_cfg:
                save_campaign_stage(
                    conn,
                    campaign_id=campaign_id,
                    stage=args.from_stage,
                    policy_json=json.dumps(from_cfg, ensure_ascii=False, sort_keys=True),
                    budget_json=json.dumps({"time_budget": from_cfg["time_budget"]}, ensure_ascii=False, sort_keys=True),
                    seed_target=from_cfg["seed_count"],
                    status="open",
                )

        decisions = plan_promotion(conn, campaign_id, policy, args.from_stage, args.to_stage)
        if not decisions:
            print("No candidates to evaluate.")
            sys.exit(0)
        print_plan(decisions, args.from_stage, args.to_stage)
        if args.execute:
            try:
                execute_promotion(conn, campaign_id, policy, args.from_stage, args.to_stage, args, decisions)
            except ValueError as exc:
                print(str(exc))
                sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
