#!/usr/bin/env python3
"""autoresearch branch planner / executor — facade over branch services."""

from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, _FRAMEWORK_DIR)

from framework.policies.branch_policy import load_branch_policy
from framework.core.db import DB_PATH, get_branch_by_id, get_search_space, init_db, save_run_branch
from framework.services.research.branch_service import (
    execute_branch_recommendation,
    execute_branches,
    plan_branches,
    plan_from_recommendation,
    print_plan,
    resolve_branch_policy_path,
    resolve_campaign,
    resolve_parent_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="autoresearch branch planner / executor")
    parser.add_argument("--db", type=str, default=DB_PATH)
    parser.add_argument("--campaign", type=str, default=None)
    parser.add_argument("--branch-policy", type=str, default=None)
    parser.add_argument("--parent-checkpoint", type=str, default=None, help="Explicit checkpoint tag to branch from; default: latest")
    parser.add_argument("--reason", type=str, action="append", default=None, help="Branch reason(s) to apply (can specify multiple)")
    parser.add_argument("--delta", type=str, action="append", default=None, help="Optional JSON delta override per reason (same order as --reason)")
    parser.add_argument("--plan", action="store_true", help="Output branch plan only")
    parser.add_argument("--dry-run", action="store_true", help="With --plan: show but do not persist")
    parser.add_argument("--execute", action="store_true", help="Execute branch plan")
    parser.add_argument("--execute-recommendation", type=str, default=None, help="Execute an accepted branch recommendation by id (v21.1)")
    parser.add_argument("--train-script", type=str, default=None)
    parser.add_argument("--time-budget", type=int, default=60, help="Time budget for each child continuation run")
    parser.add_argument("--eval-level", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    conn = init_db(args.db)
    try:
        if args.execute_recommendation:
            execute_branch_recommendation(conn, args, project_root=_PROJECT_ROOT)
            return

        if not args.campaign:
            print("Error: --campaign is required unless --execute-recommendation is used")
            sys.exit(1)
        if not args.branch_policy:
            print("Error: --branch-policy is required unless --execute-recommendation is used")
            sys.exit(1)
        if not args.reason:
            print("Error: at least one --reason is required unless --execute-recommendation is used")
            sys.exit(1)

        campaign = resolve_campaign(conn, args.campaign)
        if not args.train_script:
            args.train_script = campaign["train_script"]
        policy = load_branch_policy(args.branch_policy)
        if policy["domain"] != campaign["domain"]:
            print(f"Error: policy domain '{policy['domain']}' != campaign domain '{campaign['domain']}'")
            sys.exit(1)

        space_row = get_search_space(conn, campaign["search_space_id"])
        if space_row:
            ref = policy.get("search_space_ref", {})
            if ref.get("name") != space_row["name"] or ref.get("version") != space_row["version"]:
                print(
                    f"Error: policy search_space_ref ({ref.get('name')} v{ref.get('version')}) "
                    f"does not match campaign search space ({space_row['name']} v{space_row['version']})"
                )
                sys.exit(1)
        ref = policy.get("stage_policy_ref", {})
        if ref.get("domain") and ref["domain"] != campaign["domain"]:
            print(f"Error: policy stage_policy_ref domain '{ref['domain']}' does not match campaign domain '{campaign['domain']}'")
            sys.exit(1)

        try:
            parent_ckpt, parent_run_id = resolve_parent_checkpoint(conn, campaign, args)
            plans = plan_branches(conn, campaign, policy, parent_ckpt, parent_run_id, args)
        except ValueError as exc:
            print(str(exc))
            sys.exit(1)

        if not plans:
            print("No branches to plan.")
            return

        print_plan(plans, parent_ckpt["tag"])
        if args.dry_run:
            print("(dry-run: branches not persisted)")
            return

        if args.plan and not args.execute:
            for plan in plans:
                if not get_branch_by_id(conn, plan["branch_id"]):
                    save_run_branch(
                        conn,
                        branch_id=plan["branch_id"],
                        campaign_id=campaign["id"],
                        parent_run_id=plan["parent_run_id"],
                        parent_checkpoint_id=plan["parent_checkpoint_id"],
                        from_stage="D",
                        branch_reason=plan["reason"],
                        branch_params_json=json.dumps(plan["child_params"], ensure_ascii=False, sort_keys=True),
                        delta_json=plan["delta_json"],
                        status="planned",
                    )
            print(f"Persisted {len(plans)} planned branches.")

        if args.execute:
            execute_branches(conn, campaign, plans, parent_ckpt, args)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
