#!/usr/bin/env python3
"""autoresearch selector engine — v21 Surrogate-Guided Selector.

Generates and scores point / branch candidates for a campaign.
Produces recommendations with score breakdowns and rationales.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import uuid
from typing import Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from core.db import (
    get_campaign,
    get_campaign_runs_by_stage,
    get_branch_tree,
    get_latest_checkpoint,
    get_search_space,
    list_all_checkpoints,
)


def _stable_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# 1. Candidate generation
# ---------------------------------------------------------------------------

def generate_point_candidates(conn, campaign, policy, limit: int = 5) -> list[dict]:
    """Generate new_point and seed_recheck candidates from campaign runs."""
    candidates = []
    campaign_id = campaign["id"]

    # Aggregate completed campaign runs by candidate_key
    rows = conn.execute(
        """SELECT cr.candidate_key, cr.axis_values_json,
                  COUNT(DISTINCT cr.seed) AS seed_count,
                  COUNT(*) AS run_count,
                  AVG(r.final_win_rate) AS mean_wr,
                  MAX(r.final_win_rate) - MIN(r.final_win_rate) AS range_wr,
                  CASE WHEN COUNT(*) > 1
                       THEN SQRT(((COUNT(*)*SUM(r.final_win_rate*r.final_win_rate) -
                                  SUM(r.final_win_rate)*SUM(r.final_win_rate)) /
                                  (COUNT(*)*(COUNT(*)-1))))
                       ELSE 0.0
                  END AS std_wr,
                  AVG(r.wall_time_s) AS mean_wall_s,
                  AVG(r.num_params) AS mean_params
           FROM campaign_runs cr
           JOIN runs r ON r.id = cr.run_id
           WHERE cr.campaign_id = ?
             AND r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')
           GROUP BY cr.candidate_key
           ORDER BY mean_wr DESC""",
        (campaign_id,),
    ).fetchall()

    if not rows:
        return candidates

    # Load search space axes for bounds-checking new_point candidates
    space_axes: dict = {}
    space_row = get_search_space(conn, campaign["search_space_id"] or "")
    if space_row:
        try:
            profile = json.loads(space_row["profile_json"])
            space_axes = profile.get("axes", {})
        except (json.JSONDecodeError, KeyError):
            pass

    # Identify frontier points (top 3 by mean WR)
    frontier = [dict(r) for r in rows[:3]]
    frontier_keys = {p["candidate_key"] for p in frontier}

    # Build existing candidate keys set
    existing_keys = {r["candidate_key"] for r in rows}

    # 1a. seed_recheck candidates: high variance or single-seed frontier points
    min_runs_for_variance = policy.get("limits", {}).get("min_runs_for_variance", 2)
    max_wr_std = policy.get("limits", {}).get("max_wr_std_for_confident", 0.05)
    for r in rows:
        if r["candidate_key"] in frontier_keys:
            if r["seed_count"] < min_runs_for_variance or (r["std_wr"] or 0) > max_wr_std:
                candidates.append({
                    "candidate_type": "seed_recheck",
                    "candidate_key": r["candidate_key"],
                    "axis_values": json.loads(r["axis_values_json"] or "{}"),
                    "score_signals": {
                        "mean_wr": r["mean_wr"] or 0,
                        "std_wr": r["std_wr"] or 0,
                        "seed_count": r["seed_count"],
                        "on_frontier": True,
                    },
                })

    # 1b. new_point candidates: frontier-neighbour exploration
    # Perturb each axis of the best frontier point toward adjacent allowed values.
    if frontier and len(existing_keys) < limit * 2:
        best = frontier[0]
        axis_values = json.loads(best["axis_values_json"] or "{}")

        for axis_key, current_val in axis_values.items():
            if axis_key == "seed":
                continue
            axis_spec = space_axes.get(axis_key, {})
            allowed_values = axis_spec.get("values")

            if allowed_values and current_val in allowed_values:
                # Discrete axis: try next value in list (wrap to prev if at end)
                idx = allowed_values.index(current_val)
                next_idx = idx + 1 if idx + 1 < len(allowed_values) else idx - 1
                if next_idx < 0 or next_idx == idx:
                    continue
                new_val = allowed_values[next_idx]
            elif isinstance(current_val, (int, float)):
                # No explicit value list: +10% perturbation
                new_val = current_val * 1.1 if isinstance(current_val, float) else current_val + 1
                new_val = round(new_val, 4) if isinstance(current_val, float) else new_val
                # Bounds check against range spec
                if "min" in axis_spec and new_val < axis_spec["min"]:
                    new_val = axis_spec["min"]
                if "max" in axis_spec and new_val > axis_spec["max"]:
                    continue
            else:
                continue

            new_axis = dict(axis_values)
            new_axis[axis_key] = new_val
            new_key = _stable_json({k: v for k, v in new_axis.items() if k != "seed"})
            if new_key not in existing_keys:
                candidates.append({
                    "candidate_type": "new_point",
                    "candidate_key": new_key,
                    "axis_values": new_axis,
                    "score_signals": {
                        "parent_wr": best["mean_wr"] or 0,
                        "mean_params": best.get("mean_params") or 200000,
                        "perturbed_axis": axis_key,
                        "perturbation_ratio": round(new_val / current_val, 3) if current_val else 1.0,
                    },
                })
                if len(candidates) >= limit:
                    break

    return candidates


def generate_branch_candidates(conn, campaign, policy, limit: int = 3,
                               branch_policy: dict | None = None) -> list[dict]:
    """Generate continue_branch and eval_upgrade candidates from branch tree."""
    candidates = []
    campaign_id = campaign["id"]

    # Find top-performing parent runs with checkpoints
    rows = conn.execute(
        """SELECT cr.run_id, cr.candidate_key, cr.axis_values_json,
                  r.final_win_rate, r.eval_level, r.learning_rate,
                  r.mcts_simulations, r.num_res_blocks, r.num_filters,
                  r.num_params
           FROM campaign_runs cr
           JOIN runs r ON r.id = cr.run_id
           WHERE cr.campaign_id = ?
             AND r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')
             AND r.final_win_rate IS NOT NULL
           ORDER BY r.final_win_rate DESC
           LIMIT 5""",
        (campaign_id,),
    ).fetchall()

    if not rows:
        return candidates

    # Get branch policy reasons and factors
    branch_reasons = policy.get("branch_policy_ref", {}).get("reasons", [])
    if not branch_reasons:
        branch_reasons = ["lr_decay", "mcts_upshift", "seed_recheck"]

    # Extract lr_decay factor from branch_policy if available
    lr_decay_factor = 0.1
    if branch_policy:
        lr_spec = (branch_policy.get("branch_reasons", {})
                   .get("lr_decay", {})
                   .get("allowed_deltas", {})
                   .get("learning_rate", {}))
        lr_decay_factor = lr_spec.get("default_factor", 0.1)

    max_eval_level = policy.get("limits", {}).get("max_eval_level", 2)

    for r in rows:
        parent_run_id = r["run_id"]
        ckpt = get_latest_checkpoint(conn, parent_run_id)
        if not ckpt:
            continue

        # Only suggest branch if parent has decent WR
        if (r["final_win_rate"] or 0) < 0.5:
            continue

        # Check if this parent already has too many branches
        existing_branches = conn.execute(
            "SELECT COUNT(*) FROM run_branches WHERE parent_run_id = ?",
            (parent_run_id,),
        ).fetchone()[0]
        if existing_branches >= 3:
            continue

        # Suggest lr_decay for high-LR parents
        if (r["learning_rate"] or 0) > 0.001:
            # delta_json stores the factor for multiply-type deltas (consistent with validate_delta)
            delta_json = json.dumps({"learning_rate": lr_decay_factor}, sort_keys=True, separators=(",", ":"))
            candidates.append({
                "candidate_type": "continue_branch",
                "candidate_key": r["candidate_key"],
                "parent_run_id": parent_run_id,
                "parent_checkpoint_id": ckpt["id"],
                "branch_reason": "lr_decay",
                "delta_json": delta_json,
                "axis_values": json.loads(r["axis_values_json"] or "{}"),
                "score_signals": {
                    "parent_wr": r["final_win_rate"] or 0,
                    "mean_params": r["num_params"] if r["num_params"] is not None else 200000,
                    "existing_branches": existing_branches,
                    "reason": "lr_decay",
                },
            })

        # Suggest seed_recheck for single-seed parents
        seed_count = conn.execute(
            """SELECT COUNT(DISTINCT seed) FROM campaign_runs
               WHERE campaign_id = ? AND candidate_key = ?""",
            (campaign_id, r["candidate_key"]),
        ).fetchone()[0]
        if seed_count < 2:
            candidates.append({
                "candidate_type": "continue_branch",
                "candidate_key": r["candidate_key"],
                "parent_run_id": parent_run_id,
                "parent_checkpoint_id": ckpt["id"],
                "branch_reason": "seed_recheck",
                "delta_json": "{}",  # No explicit delta; branch.py uses policy default (deterministic seed)
                "axis_values": json.loads(r["axis_values_json"] or "{}"),
                "score_signals": {
                    "parent_wr": r["final_win_rate"] or 0,
                    "mean_params": r["num_params"] if r["num_params"] is not None else 200000,
                    "existing_branches": existing_branches,
                    "reason": "seed_recheck",
                },
            })

        # Suggest eval_upgrade for high-WR parents below max eval_level
        if (r["final_win_rate"] or 0) >= 0.6 and (r["eval_level"] or 0) < max_eval_level:
            candidates.append({
                "candidate_type": "eval_upgrade",
                "candidate_key": r["candidate_key"],
                "parent_run_id": parent_run_id,
                "parent_checkpoint_id": ckpt["id"],
                "branch_reason": "eval_upgrade",
                "delta_json": "{}",
                "axis_values": json.loads(r["axis_values_json"] or "{}"),
                "score_signals": {
                    "parent_wr": r["final_win_rate"] or 0,
                    "mean_params": r["num_params"] if r["num_params"] is not None else 200000,
                    "existing_branches": existing_branches,
                    "reason": "eval_upgrade",
                },
            })

    return candidates


# ---------------------------------------------------------------------------
# 2. Scoring
# ---------------------------------------------------------------------------

def _score_candidate(candidate: dict, weights: dict, all_candidates: list[dict]) -> tuple[float, dict, dict]:
    """Score a single candidate. Returns (score_total, breakdown, rationale)."""
    signals = candidate.get("score_signals", {})
    ctype = candidate["candidate_type"]
    breakdown = {}

    # frontier_gap: higher for frontier-adjacent / high-WR candidates
    parent_wr = signals.get("parent_wr", signals.get("mean_wr", 0.5))
    breakdown["frontier_gap"] = round(parent_wr * weights.get("frontier_gap", 1.0), 4)

    # uncertainty: bonus for high-variance or under-sampled candidates
    std_wr = signals.get("std_wr", 0)
    seed_count = signals.get("seed_count", 1)
    uncertainty_score = (std_wr * 2) + max(0, 2 - seed_count) * 0.3
    breakdown["uncertainty"] = round(uncertainty_score * weights.get("uncertainty", 0.8), 4)

    # cost_penalty: penalize expensive candidates (more params / longer wall time)
    params = signals.get("mean_params", 100000)
    cost_penalty = math.log10(max(params, 1000)) / 6.0  # normalize ~0.5-1.0
    breakdown["cost_penalty"] = round(-cost_penalty * weights.get("cost_penalty", 0.5), 4)

    # dominance_penalty: if candidate is clearly dominated by existing frontier
    dominated = _is_dominated(candidate, all_candidates)
    dom_weight = weights.get("dominance_penalty", 1.0)
    breakdown["dominance_penalty"] = round(-1.5 * dom_weight if dominated else 0.0, 4)

    score_total = sum(breakdown.values())

    rationale = {
        "summary": f"{ctype} candidate scored {score_total:.3f}",
        "frontier_position": "on_frontier" if signals.get("on_frontier") else "near_frontier" if parent_wr > 0.6 else "back",
        "variance_concern": std_wr > 0.05,
        "seed_shortage": seed_count < 2,
        "dominated": dominated,
    }

    return round(score_total, 4), breakdown, rationale


def _is_dominated(candidate: dict, all_candidates: list[dict]) -> bool:
    """Simple dominance check: candidate is dominated if another candidate has higher WR and fewer params."""
    c_wr = candidate.get("score_signals", {}).get("mean_wr", candidate.get("score_signals", {}).get("parent_wr", 0))
    c_params = candidate.get("score_signals", {}).get("mean_params", 100000)

    for other in all_candidates:
        if other is candidate:
            continue
        o_wr = other.get("score_signals", {}).get("mean_wr", other.get("score_signals", {}).get("parent_wr", 0))
        o_params = other.get("score_signals", {}).get("mean_params", 100000)
        if o_wr >= c_wr and o_params <= c_params and (o_wr > c_wr or o_params < c_params):
            return True
    return False


# ---------------------------------------------------------------------------
# 3. Orchestration
# ---------------------------------------------------------------------------

def recommend_for_campaign(conn, campaign, policy,
                           candidate_type: str | None = None,
                           limit: int = 5) -> list[dict]:
    """Generate and score recommendations for a campaign.

    Returns a list of recommendation dicts sorted by score descending.
    Each dict has: candidate_type, candidate_key, score_total, score_breakdown,
    rationale, axis_values, and optional parent_run_id / branch_reason / delta_json.
    """
    # Filter out recommendations already accepted or executed in prior batches.
    # Identity is defined by (candidate_type, candidate_key, branch_reason, delta_json)
    # to avoid cross-type suppression (e.g. seed_recheck vs continue_branch on same key).
    rows = conn.execute(
        """SELECT DISTINCT r.candidate_type, r.candidate_key, r.branch_reason, r.delta_json
           FROM recommendations r
           JOIN recommendation_batches b ON b.id = r.batch_id
           WHERE b.campaign_id = ? AND r.status IN ('accepted','executed')""",
        (campaign["id"],),
    ).fetchall()
    accepted_identities = {
        (r["candidate_type"], r["candidate_key"], r["branch_reason"], r["delta_json"])
        for r in rows
    }

    def _identity(c: dict) -> tuple:
        return (c["candidate_type"], c.get("candidate_key"), c.get("branch_reason"), c.get("delta_json"))

    candidates = []

    # Load branch policy for delta computation in branch candidates
    bp: dict | None = None
    bp_ref = policy.get("branch_policy_ref", {})
    if bp_ref.get("domain") and bp_ref.get("name"):
        bp_path = os.path.join(_PROJECT_ROOT, "domains", bp_ref["domain"], "branch_policy.json")
        if os.path.isfile(bp_path):
            try:
                with open(bp_path, "r", encoding="utf-8") as _f:
                    bp = json.load(_f)
            except (json.JSONDecodeError, OSError):
                pass

    if candidate_type is None or candidate_type in ("new_point", "seed_recheck"):
        point_cands = generate_point_candidates(conn, campaign, policy)
        if candidate_type:
            point_cands = [c for c in point_cands if c["candidate_type"] == candidate_type]
        candidates.extend(c for c in point_cands if _identity(c) not in accepted_identities)
    if candidate_type is None or candidate_type in ("continue_branch", "eval_upgrade"):
        branch_cands = generate_branch_candidates(conn, campaign, policy, branch_policy=bp)
        if candidate_type:
            branch_cands = [c for c in branch_cands if c["candidate_type"] == candidate_type]
        candidates.extend(c for c in branch_cands if _identity(c) not in accepted_identities)

    if not candidates:
        return []

    weights = policy.get("score_weights", {})
    scored = []
    for c in candidates:
        score, breakdown, rationale = _score_candidate(c, weights, candidates)
        scored.append({
            **c,
            "score_total": score,
            "score_breakdown": breakdown,
            "rationale": rationale,
        })

    scored.sort(key=lambda x: x["score_total"], reverse=True)
    return scored[:limit]


def build_recommendation_id(batch_id: str, candidate: dict) -> str:
    """Stable recommendation id from batch + candidate content."""
    base = f"{batch_id}_{candidate['candidate_type']}_{candidate.get('candidate_key', '')}_{candidate.get('branch_reason', '')}"
    h = hashlib.sha256(base.encode()).hexdigest()[:16]
    return f"rec-{h}"
