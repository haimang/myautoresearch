"""Scoring helpers for selector recommendations."""

from __future__ import annotations

import math


def is_dominated(candidate: dict, all_candidates: list[dict]) -> bool:
    """Simple dominance check: candidate is dominated if another has higher objective and lower cost."""
    signals = candidate.get("score_signals", {})
    candidate_objective = signals.get("predicted_objective", signals.get("mean_wr", signals.get("parent_wr", 0)))
    candidate_cost = signals.get("mean_params", signals.get("mean_cost", 100000))
    for other in all_candidates:
        if other is candidate:
            continue
        other_signals = other.get("score_signals", {})
        other_objective = other_signals.get(
            "predicted_objective",
            other_signals.get("mean_wr", other_signals.get("parent_wr", 0)),
        )
        other_cost = other_signals.get("mean_params", other_signals.get("mean_cost", 100000))
        if other_objective >= candidate_objective and other_cost <= candidate_cost:
            if other_objective > candidate_objective or other_cost < candidate_cost:
                return True
    return False


def score_candidate(candidate: dict, weights: dict, all_candidates: list[dict]) -> tuple[float, dict, dict]:
    """Score a single candidate. Returns (score_total, breakdown, rationale)."""
    signals = candidate.get("score_signals", {})
    candidate_type = candidate["candidate_type"]
    predicted = signals.get("predicted_objective")
    if predicted is None:
        predicted = signals.get("parent_wr", signals.get("mean_wr", 0.5))

    breakdown = {
        "frontier_gap": round(float(predicted) * weights.get("frontier_gap", 1.0), 4),
    }

    std_wr = signals.get("std_wr", signals.get("posterior_sigma", 0))
    seed_count = signals.get("seed_count", 1)
    uncertainty_score = (std_wr * 2) + max(0, 2 - seed_count) * 0.3
    breakdown["uncertainty"] = round(uncertainty_score * weights.get("uncertainty", 0.8), 4)

    params = signals.get("mean_params", signals.get("mean_cost", 100000))
    cost_penalty = math.log10(max(params, 1000)) / 6.0
    breakdown["cost_penalty"] = round(-cost_penalty * weights.get("cost_penalty", 0.5), 4)

    dominated = is_dominated(candidate, all_candidates)
    dom_weight = weights.get("dominance_penalty", 1.0)
    breakdown["dominance_penalty"] = round(-1.5 * dom_weight if dominated else 0.0, 4)

    score_total = round(sum(breakdown.values()), 4)
    rationale = {
        "summary": f"{candidate_type} candidate scored {score_total:.3f}",
        "frontier_position": "on_frontier" if signals.get("on_frontier") else "near_frontier" if float(predicted) > 0.6 else "back",
        "variance_concern": std_wr > 0.05,
        "seed_shortage": seed_count < 2,
        "dominated": dominated,
    }
    return score_total, breakdown, rationale
