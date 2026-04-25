"""
framework/acquisition.py — v21.1 candidate-pool acquisition reranker.

在 selector 生成的 point/branch candidate pool 上叠加轻量的、可迁移的多目标 acquisition 层。
这里刻意避免重型 raw-space BO；目标是：

1. 复用已有 search/stage/branch discipline
2. 对合法候选池做 portable reranking
3. 为 replay / cross-domain handoff 提供稳定证据
"""

from __future__ import annotations

import json
import math
from typing import Any


def _posterior_sigma(signals: dict, priors: dict) -> float:
    std_wr = float(signals.get("std_wr") or 0.0)
    seed_count = max(1, int(signals.get("seed_count") or 1))
    base_sigma = float(priors.get("base_sigma", 0.04))
    min_sigma = float(priors.get("min_sigma", 0.01))
    shortage_bonus = float(priors.get("seed_shortage_bonus", 0.15))
    target_seed_count = int(priors.get("target_seed_count", 2))

    empirical_sigma = max(std_wr, base_sigma / math.sqrt(seed_count))
    seed_shortage = max(0, target_seed_count - seed_count)
    posterior_sigma = empirical_sigma + (seed_shortage * shortage_bonus)
    return round(max(min_sigma, posterior_sigma), 6)


def _candidate_frontier_bonus(candidate: dict, signals: dict, weight: float) -> float:
    if signals.get("on_frontier"):
        return round(weight, 6)
    parent_wr = float(signals.get("parent_wr", signals.get("mean_wr", 0.0)) or 0.0)
    if parent_wr >= 0.7:
        return round(weight * 0.5, 6)
    return 0.0


def rerank_candidates(
    candidates: list[dict],
    policy: dict,
) -> tuple[list[dict], dict]:
    """Apply a lightweight acquisition reranker to selector-scored candidates.

    Returns (reranked_candidates, surrogate_summary).
    """
    if not candidates:
        summary = {
            "candidate_count": 0,
            "feature_schema": {
                "predicted_wr": "float",
                "posterior_sigma": "float",
                "log_params": "float",
                "log_wall_s": "float",
                "candidate_type_bonus": "float",
            },
        }
        return [], summary

    weights = policy["weights"]
    priors = policy["priors"]
    type_bonus = policy.get("candidate_type_bonus", {})

    reranked = []
    for candidate in candidates:
        signals = candidate.get("score_signals", {})
        predicted_wr = float(signals.get("mean_wr", signals.get("parent_wr", 0.5)) or 0.5)
        mean_params = float(signals.get("mean_params") or 100000.0)
        mean_wall_s = float(signals.get("mean_wall_s") or signals.get("mean_wall") or 120.0)
        posterior_sigma = _posterior_sigma(signals, priors)
        candidate_bonus = float(type_bonus.get(candidate["candidate_type"], 0.0))
        frontier_bonus = _candidate_frontier_bonus(candidate, signals, float(weights.get("frontier_bonus", 0.0)))

        predicted_component = predicted_wr * float(weights["predicted_wr"])
        uncertainty_component = posterior_sigma * float(weights["uncertainty"])
        params_penalty = (math.log10(max(mean_params, 1000.0)) / 6.0) * float(weights["params_penalty"])
        wall_penalty = (math.log10(max(mean_wall_s, 1.0)) / 3.0) * float(weights["wall_penalty"])

        acquisition_score = (
            predicted_component
            + uncertainty_component
            + candidate_bonus
            + frontier_bonus
            - params_penalty
            - wall_penalty
        )

        acquisition_breakdown = {
            "predicted_wr": round(predicted_component, 6),
            "uncertainty": round(uncertainty_component, 6),
            "candidate_type_bonus": round(candidate_bonus, 6),
            "frontier_bonus": round(frontier_bonus, 6),
            "params_penalty": round(-params_penalty, 6),
            "wall_penalty": round(-wall_penalty, 6),
        }

        rationale = dict(candidate.get("rationale", {}))
        rationale["acquisition_summary"] = (
            f"{policy['name']} scored {candidate['candidate_type']} at {acquisition_score:.3f}"
        )
        rationale["posterior_sigma"] = posterior_sigma

        reranked.append({
            **candidate,
            "selector_score_total": candidate.get("score_total"),
            "score_total": round(acquisition_score, 6),
            "score_breakdown": acquisition_breakdown,
            "rationale": rationale,
            "acquisition_name": policy["name"],
            "acquisition_version": policy["version"],
            "acquisition_score": round(acquisition_score, 6),
            "posterior_sigma": posterior_sigma,
        })

    reranked.sort(
        key=lambda row: (row["acquisition_score"], row.get("selector_score_total", row["score_total"])),
        reverse=True,
    )

    summary = {
        "candidate_count": len(reranked),
        "feature_schema": {
            "predicted_wr": "float",
            "posterior_sigma": "float",
            "mean_params": "float",
            "mean_wall_s": "float",
            "candidate_type_bonus": "float",
        },
        "top_candidate_types": [r["candidate_type"] for r in reranked[:3]],
        "top_scores": [r["acquisition_score"] for r in reranked[:3]],
        "policy": {
            "name": policy["name"],
            "version": policy["version"],
        },
    }
    return reranked, summary


def replay_recommendation_history(
    rows: list[dict],
    *,
    top_k: int,
    positive_outcomes: list[str],
) -> dict:
    """Compare selector vs acquisition ranking on historical labeled recommendations."""
    positive = set(positive_outcomes)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["batch_id"], []).append(row)

    evaluated_batches = 0
    selector_hits = 0
    acquisition_hits = 0

    for batch_rows in grouped.values():
        labeled = [r for r in batch_rows if r.get("outcome_label")]
        if not labeled:
            continue
        evaluated_batches += 1

        selector_ranked = sorted(
            batch_rows,
            key=lambda r: (
                float(r.get("selector_score_total") if r.get("selector_score_total") is not None else r.get("score_total", 0.0)),
                -int(r.get("rank") or 0),
            ),
            reverse=True,
        )
        acquisition_ranked = sorted(
            batch_rows,
            key=lambda r: (
                float(r.get("acquisition_score") if r.get("acquisition_score") is not None else r.get("score_total", 0.0)),
                -int(r.get("rank") or 0),
            ),
            reverse=True,
        )

        if any((row.get("outcome_label") in positive) for row in selector_ranked[:top_k]):
            selector_hits += 1
        if any((row.get("outcome_label") in positive) for row in acquisition_ranked[:top_k]):
            acquisition_hits += 1

    def _rate(hits: int) -> float:
        return round(hits / evaluated_batches, 4) if evaluated_batches else 0.0

    return {
        "evaluated_batches": evaluated_batches,
        "top_k": top_k,
        "positive_outcomes": sorted(positive),
        "selector_hits": selector_hits,
        "selector_hit_rate": _rate(selector_hits),
        "acquisition_hits": acquisition_hits,
        "acquisition_hit_rate": _rate(acquisition_hits),
        "acquisition_delta": round(_rate(acquisition_hits) - _rate(selector_hits), 4),
    }


def snapshot_payload(policy: dict, summary: dict) -> dict:
    """Build a compact persisted payload for surrogate_snapshots.summary_json."""
    return {
        "policy": {
            "name": policy["name"],
            "version": policy["version"],
        },
        "summary": summary,
        "objectives": policy["objectives"],
    }
