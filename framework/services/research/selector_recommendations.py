"""Recommendation assembly for selector batches."""

from __future__ import annotations

import hashlib
import json
import os

from .selector_candidates import generate_branch_candidates, generate_point_candidates
from .selector_scoring import score_candidate


def _selector_identity(candidate: dict) -> tuple:
    return (
        candidate["candidate_type"],
        candidate.get("candidate_key"),
        candidate.get("branch_reason"),
        candidate.get("delta_json"),
    )


def load_branch_policy(project_root: str, policy: dict) -> dict | None:
    policy_ref = policy.get("branch_policy_ref", {})
    if not (policy_ref.get("domain") and policy_ref.get("name")):
        return None
    policy_path = os.path.join(project_root, "domains", policy_ref["domain"], "branch_policy.json")
    if not os.path.isfile(policy_path):
        return None
    try:
        with open(policy_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def recommend_for_campaign(
    conn,
    campaign,
    policy,
    *,
    project_root: str | None = None,
    candidate_type: str | None = None,
    limit: int | None = 5,
) -> list[dict]:
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
    rows = conn.execute(
        """SELECT DISTINCT r.candidate_type, r.candidate_key, r.branch_reason, r.delta_json
           FROM recommendations r
           JOIN recommendation_batches b ON b.id = r.batch_id
           WHERE b.campaign_id = ? AND r.status IN ('accepted','executed')""",
        (campaign["id"],),
    ).fetchall()
    accepted = {
        (row["candidate_type"], row["candidate_key"], row["branch_reason"], row["delta_json"])
        for row in rows
    }

    branch_policy = load_branch_policy(project_root, policy)
    candidates = []
    if candidate_type is None or candidate_type in ("new_point", "seed_recheck"):
        point_candidates = generate_point_candidates(conn, campaign, policy)
        if candidate_type:
            point_candidates = [candidate for candidate in point_candidates if candidate["candidate_type"] == candidate_type]
        candidates.extend(candidate for candidate in point_candidates if _selector_identity(candidate) not in accepted)
    if candidate_type is None or candidate_type in ("continue_branch", "eval_upgrade"):
        branch_candidates = generate_branch_candidates(conn, campaign, policy, branch_policy=branch_policy)
        if candidate_type:
            branch_candidates = [candidate for candidate in branch_candidates if candidate["candidate_type"] == candidate_type]
        candidates.extend(candidate for candidate in branch_candidates if _selector_identity(candidate) not in accepted)
    if not candidates:
        return []

    weights = policy.get("score_weights", {})
    scored = []
    for candidate in candidates:
        score, breakdown, rationale = score_candidate(candidate, weights, candidates)
        scored.append(
            {
                **candidate,
                "score_total": score,
                "score_breakdown": breakdown,
                "rationale": rationale,
            }
        )
    scored.sort(key=lambda item: item["score_total"], reverse=True)
    return scored if limit is None else scored[:limit]


def build_recommendation_id(batch_id: str, candidate: dict) -> str:
    """Stable recommendation id from batch + candidate content."""
    base = f"{batch_id}_{candidate['candidate_type']}_{candidate.get('candidate_key', '')}_{candidate.get('branch_reason', '')}"
    return f"rec-{hashlib.sha256(base.encode()).hexdigest()[:16]}"
