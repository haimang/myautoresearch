#!/usr/bin/env python3
"""v21.1 replay benchmark: compare selector vs acquisition on labeled recommendation history."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from acquisition import replay_recommendation_history
from acquisition_policy import load_acquisition_policy
from core.db import DB_PATH, get_campaign, init_db


def parse_args():
    p = argparse.ArgumentParser(description="Replay selector vs acquisition ranking on recommendation history")
    p.add_argument("--db", type=str, default=DB_PATH)
    p.add_argument("--campaign", type=str, required=True)
    p.add_argument("--acquisition-policy", type=str, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--format", type=str, default="md", choices=["md", "json"])
    return p.parse_args()


def _load_campaign_and_policy(conn, campaign_name: str, acquisition_policy_path: str | None):
    campaign = get_campaign(conn, campaign_name)
    if not campaign:
        raise ValueError(f"Campaign not found: {campaign_name}")
    campaign = dict(campaign)
    if acquisition_policy_path:
        policy = load_acquisition_policy(acquisition_policy_path)
    else:
        default_path = ROOT / "domains" / campaign["domain"] / "acquisition_policy.json"
        if not default_path.is_file():
            raise ValueError(f"No acquisition policy found for domain '{campaign['domain']}'")
        policy = load_acquisition_policy(str(default_path))
    return campaign, policy


def _load_replay_rows(conn, campaign_id: str) -> list[dict]:
    rows = conn.execute(
        """SELECT b.id AS batch_id,
                  r.id AS recommendation_id,
                  r.rank,
                  r.score_total,
                  r.selector_score_total,
                  r.acquisition_score,
                  r.candidate_type,
                  o.outcome_label
           FROM recommendation_batches b
           JOIN recommendations r ON r.batch_id = b.id
           LEFT JOIN recommendation_outcomes o ON o.recommendation_id = r.id
           WHERE b.campaign_id = ?
           ORDER BY b.created_at, r.rank""",
        (campaign_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def main() -> int:
    args = parse_args()
    conn = init_db(args.db)
    try:
        campaign, policy = _load_campaign_and_policy(conn, args.campaign, args.acquisition_policy)
        rows = _load_replay_rows(conn, campaign["id"])
        if not rows:
            print(f"No recommendation history for campaign '{args.campaign}'")
            return 0

        top_k = args.top_k or int(policy["replay"]["top_k"])
        positive_outcomes = policy["replay"]["positive_outcomes"]
        summary = replay_recommendation_history(
            rows,
            top_k=top_k,
            positive_outcomes=positive_outcomes,
        )
        output = {
            "campaign": args.campaign,
            "policy": {
                "name": policy["name"],
                "version": policy["version"],
            },
            "summary": summary,
        }

        if args.format == "json":
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print(f"Replay Benchmark: {args.campaign}")
            print("=" * 72)
            print(f"Policy: {policy['name']} v{policy['version']}")
            print(f"Evaluated batches: {summary['evaluated_batches']}")
            print(f"Top-k: {summary['top_k']} | Positive outcomes: {', '.join(summary['positive_outcomes'])}")
            print(f"Selector hit rate:    {summary['selector_hit_rate']:.1%} ({summary['selector_hits']}/{summary['evaluated_batches'] or 1})")
            print(f"Acquisition hit rate: {summary['acquisition_hit_rate']:.1%} ({summary['acquisition_hits']}/{summary['evaluated_batches'] or 1})")
            print(f"Delta: {summary['acquisition_delta']:+.1%}")
        return 0
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
