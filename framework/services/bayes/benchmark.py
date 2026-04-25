"""Generic benchmark summaries for Bayesian-style refinement."""

from __future__ import annotations

from services.frontier.pareto import compute_knee_point, pareto_front


def summarize_frontier(
    *,
    points: list[dict],
    feasible_points: list[dict],
    maximize: list[str],
    minimize: list[str],
    true_front_keys: set[str],
    true_knee_key: str | None,
) -> tuple[dict, list[dict], list[dict], dict | None]:
    front, dominated = pareto_front(feasible_points, maximize=maximize, minimize=minimize)
    knee = compute_knee_point(front, maximize=maximize, minimize=minimize)
    selected_front_keys = {point["run_full"] for point in front}
    hit_count = len(selected_front_keys & true_front_keys)
    front_recall = hit_count / len(true_front_keys) if true_front_keys else 0.0
    best_uplift = max((point["spot_uplift_bps"] for point in feasible_points), default=0.0)
    best_preservation = max((point["preservation_ratio"] for point in feasible_points), default=0.0)
    infeasible_rate = 1.0 - (len(feasible_points) / len(points) if points else 0.0)
    summary = {
        "budget": len(points),
        "front_points": len(front),
        "selected_true_front_hits": hit_count,
        "true_front_recall": front_recall,
        "best_uplift_bps": best_uplift,
        "best_preservation_ratio": best_preservation,
        "infeasible_rate": infeasible_rate,
        "knee_hit": bool(knee and true_knee_key and knee["run_full"] == true_knee_key),
    }
    return summary, front, dominated, knee
