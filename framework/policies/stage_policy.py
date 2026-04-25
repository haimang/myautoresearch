"""
framework/stage_policy.py — v20.2 Multi-Fidelity Promotion Engine

负责 stage policy JSON 的加载、校验、摘要，以及 promotion 规划逻辑。
"""

import json
import os
import sqlite3
import sys
from typing import Any

# Allow imports both as package (framework.stage_policy) and direct
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from framework.core.db import (
    get_campaign_stages,
    get_campaign_stage,
    close_campaign_stage,
    save_promotion_decision,
)


# ---------------------------------------------------------------------------
# 1. Load / validate stage policy JSON
# ---------------------------------------------------------------------------

STAGE_ORDER = ["A", "B", "C", "D"]

REQUIRED_STAGE_KEYS = {"name", "time_budget", "seed_count", "promote_top_k", "metric", "min_runs"}

ALLOWED_METRICS = {"win_rate", "final_win_rate"}


def load_stage_policy(path: str) -> dict:
    """Load a stage policy JSON from disk."""
    if not os.path.isfile(path):
        raise ValueError(f"Stage policy file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    validate_stage_policy(data)
    return data


def validate_stage_policy(data: dict) -> None:
    """Validate a stage policy object. Raises ValueError on failure."""
    if not isinstance(data, dict):
        raise ValueError("Stage policy must be a JSON object")

    top_keys = {"domain", "name", "version", "search_space_ref", "stages"}
    missing = top_keys - set(data.keys())
    if missing:
        raise ValueError(f"Missing top-level keys: {sorted(missing)}")

    stages = data["stages"]
    if not isinstance(stages, list) or len(stages) == 0:
        raise ValueError("stages must be a non-empty list")

    seen_names = set()
    for i, st in enumerate(stages):
        if not isinstance(st, dict):
            raise ValueError(f"Stage {i} must be an object")
        missing_keys = REQUIRED_STAGE_KEYS - set(st.keys())
        if missing_keys:
            raise ValueError(f"Stage {i} missing keys: {sorted(missing_keys)}")

        name = st["name"]
        if name in seen_names:
            raise ValueError(f"Duplicate stage name: {name}")
        seen_names.add(name)

        if st["promote_top_k"] < 0:
            raise ValueError(f"Stage {name}: promote_top_k must be >= 0")
        if st["seed_count"] is not None and st["seed_count"] < 1:
            raise ValueError(f"Stage {name}: seed_count must be >= 1 or null")
        metric = st["metric"]
        if not isinstance(metric, str) or metric not in ALLOWED_METRICS:
            raise ValueError(
                f"Stage {name}: metric '{metric}' must be one of {sorted(ALLOWED_METRICS)}"
            )

    # Validate stage order A→B→C→D (must be contiguous prefix)
    actual_order = [s["name"] for s in stages]
    expected = STAGE_ORDER[:len(actual_order)]
    if actual_order != expected:
        raise ValueError(
            f"Stage order must be {expected}, got {actual_order}"
        )

    # Stage D must exist (v20.3 handoff boundary)
    if "D" not in actual_order:
        raise ValueError("Stage D must be present in policy (v20.3 handoff slot)")

    # Validate search_space_ref
    ssr = data["search_space_ref"]
    if not isinstance(ssr, dict) or "domain" not in ssr or "name" not in ssr:
        raise ValueError("search_space_ref must be an object with domain and name")


def describe_stage_policy(policy: dict) -> str:
    """Return a human-readable summary of the policy."""
    lines = [
        f"Stage Policy: {policy['name']} v{policy['version']} ({policy['domain']})",
        f"Search space: {policy['search_space_ref']['name']} v{policy['search_space_ref'].get('version', '?')}",
        "Stages:",
    ]
    for st in policy["stages"]:
        tb = st.get("time_budget")
        tb_str = f"{tb}s" if tb is not None else "handoff"
        lines.append(
            f"  {st['name']}: budget={tb_str}  seeds={st['seed_count']}  "
            f"promote_top_k={st['promote_top_k']}  metric={st['metric']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Stage lookup helpers
# ---------------------------------------------------------------------------

def get_stage_by_name(policy: dict, name: str) -> dict | None:
    """Return the stage config matching `name`, or None."""
    for st in policy["stages"]:
        if st["name"] == name:
            return st
    return None


def get_next_stage(policy: dict, current: str) -> str | None:
    """Return the next stage after `current`, or None if at end."""
    names = [s["name"] for s in policy["stages"]]
    try:
        idx = names.index(current)
    except ValueError:
        return None
    if idx + 1 < len(names):
        return names[idx + 1]
    return None


# Alias used by promote.py
next_stage_name = get_next_stage


# ---------------------------------------------------------------------------
# 3. Promotion planner (read-only analysis of campaign runs)
# ---------------------------------------------------------------------------

def aggregate_stage_metrics(
    conn: sqlite3.Connection,
    campaign_id: str,
    stage: str,
    metric_col: str = "final_win_rate",
) -> list[dict]:
    """Aggregate runs in a campaign stage by candidate_key.

    Returns a list sorted by mean metric descending, each dict containing:
    - candidate_key
    - axis_values
    - mean_metric
    - seed_count
    - run_ids
    """
    if metric_col not in ALLOWED_METRICS:
        raise ValueError(
            f"metric_col '{metric_col}' is not allowed; must be one of {sorted(ALLOWED_METRICS)}"
        )
    rows = conn.execute(
        """
        SELECT cr.candidate_key, cr.axis_values_json, cr.seed, r.id, r.{metric}
        FROM campaign_runs cr
        JOIN runs r ON r.id = cr.run_id
        WHERE cr.campaign_id = ? AND cr.stage = ? AND cr.status != 'failed'
        """.format(metric=metric_col),
        (campaign_id, stage),
    ).fetchall()

    from collections import defaultdict
    groups: dict[str, dict] = defaultdict(lambda: {
        "axis_values": None,
        "metrics": [],
        "seeds": set(),
        "run_ids": [],
    })

    for row in rows:
        ck = row["candidate_key"]
        groups[ck]["axis_values"] = json.loads(row["axis_values_json"] or "{}")
        groups[ck]["metrics"].append(row[metric_col])
        if row["seed"] is not None:
            groups[ck]["seeds"].add(row["seed"])
        groups[ck]["run_ids"].append(row["id"])

    results = []
    for ck, g in groups.items():
        metrics = [m for m in g["metrics"] if m is not None]
        if not metrics:
            continue
        mean_metric = sum(metrics) / len(metrics)
        results.append({
            "candidate_key": ck,
            "axis_values": g["axis_values"],
            "mean_metric": mean_metric,
            "seed_count": len(g["seeds"]),
            "run_count": len(g["run_ids"]),
            "run_ids": g["run_ids"],
        })

    results.sort(key=lambda x: x["mean_metric"], reverse=True)
    return results


def plan_promotions(
    policy: dict,
    current_stage: str,
    aggregated: list[dict],
) -> list[dict]:
    """Generate promotion decisions for candidates in a stage.

    Returns a list of decision dicts with keys:
    - candidate_key, axis_values, mean_metric, seed_count
    - decision: "promote" | "hold" | "reject"
    - decision_rank
    - reason
    """
    stage_cfg = get_stage_by_name(policy, current_stage)
    if stage_cfg is None:
        raise ValueError(f"Stage {current_stage} not found in policy")

    top_k = stage_cfg["promote_top_k"]
    min_runs = stage_cfg["min_runs"]
    seed_target = stage_cfg.get("seed_count", 1)

    decisions = []
    for rank, cand in enumerate(aggregated, 1):
        if rank <= top_k:
            if cand["seed_count"] >= seed_target and cand["run_count"] >= min_runs:
                decision = "promote"
                reason = (
                    f"rank={rank} <= top_k={top_k}, "
                    f"seeds={cand['seed_count']}/{seed_target}, "
                    f"runs={cand['run_count']}/{min_runs}"
                )
            else:
                decision = "hold"
                reason = (
                    f"rank={rank} <= top_k={top_k} but insufficient data: "
                    f"seeds={cand['seed_count']}/{seed_target}, "
                    f"runs={cand['run_count']}/{min_runs}"
                )
        else:
            decision = "reject"
            reason = f"rank={rank} > top_k={top_k}"

        decisions.append({
            "candidate_key": cand["candidate_key"],
            "axis_values": cand["axis_values"],
            "mean_metric": cand["mean_metric"],
            "seed_count": cand["seed_count"],
            "decision": decision,
            "decision_rank": rank,
            "reason": reason,
        })

    return decisions


# ---------------------------------------------------------------------------
# 4. Promotion execution (DB write)
# ---------------------------------------------------------------------------

def execute_promotions(
    conn: sqlite3.Connection,
    campaign_id: str,
    from_stage: str,
    to_stage: str,
    decisions: list[dict],
) -> dict:
    """Persist promotion decisions and close the from_stage.

    Returns a summary dict with counts."""
    promote_count = 0
    hold_count = 0
    reject_count = 0

    for d in decisions:
        save_promotion_decision(
            conn,
            campaign_id=campaign_id,
            from_stage=from_stage,
            to_stage=to_stage,
            candidate_key=d["candidate_key"],
            axis_values=d["axis_values"],
            aggregated_metrics={"mean_metric": d["mean_metric"]},
            seed_count=d["seed_count"],
            decision=d["decision"],
            decision_rank=d["decision_rank"],
            reason=d["reason"],
        )
        if d["decision"] == "promote":
            promote_count += 1
        elif d["decision"] == "hold":
            hold_count += 1
        else:
            reject_count += 1

    close_campaign_stage(conn, campaign_id, from_stage)

    return {
        "promoted": promote_count,
        "held": hold_count,
        "rejected": reject_count,
        "total": len(decisions),
    }
