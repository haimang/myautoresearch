"""Candidate generation for selector recommendations."""

from __future__ import annotations

import json

from core.db import get_latest_checkpoint, get_objective_profile, get_search_space


def stable_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def objective_profile_for_campaign(conn, campaign) -> dict | None:
    profile_id = campaign.get("objective_profile_id")
    if not profile_id:
        return None
    row = get_objective_profile(conn, profile_id)
    if not row:
        return None
    try:
        return json.loads(row["profile_json"])
    except (json.JSONDecodeError, KeyError):
        return None


def generic_utility(metrics: dict[str, float], profile: dict) -> float:
    score = 0.0
    for metric in profile.get("maximize", []):
        score += float(metrics.get(metric, 0.0) or 0.0)
    for metric in profile.get("minimize", []):
        score -= float(metrics.get(metric, 0.0) or 0.0) * 0.001
    return score


def generate_point_candidates(conn, campaign, policy, limit: int = 5) -> list[dict]:
    """Generate new_point and seed_recheck candidates from campaign runs."""
    if campaign.get("objective_profile_id"):
        return generate_generic_point_candidates(conn, campaign, policy, limit=limit)

    candidates = []
    campaign_id = campaign["id"]
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

    space_axes: dict = {}
    space_row = get_search_space(conn, campaign["search_space_id"] or "")
    if space_row:
        try:
            profile = json.loads(space_row["profile_json"])
            space_axes = profile.get("axes", {})
        except (json.JSONDecodeError, KeyError):
            pass

    frontier = [dict(row) for row in rows[:3]]
    frontier_keys = {point["candidate_key"] for point in frontier}
    existing_keys = {row["candidate_key"] for row in rows}

    min_runs_for_variance = policy.get("limits", {}).get("min_runs_for_variance", 2)
    max_wr_std = policy.get("limits", {}).get("max_wr_std_for_confident", 0.05)
    for row in rows:
        if row["candidate_key"] in frontier_keys:
            if row["seed_count"] < min_runs_for_variance or (row["std_wr"] or 0) > max_wr_std:
                candidates.append(
                    {
                        "candidate_type": "seed_recheck",
                        "candidate_key": row["candidate_key"],
                        "axis_values": json.loads(row["axis_values_json"] or "{}"),
                        "score_signals": {
                            "mean_wr": row["mean_wr"] or 0,
                            "std_wr": row["std_wr"] or 0,
                            "seed_count": row["seed_count"],
                            "on_frontier": True,
                        },
                    }
                )

    if frontier and len(existing_keys) < limit * 2:
        best = frontier[0]
        axis_values = json.loads(best["axis_values_json"] or "{}")
        for axis_key, current_val in axis_values.items():
            if axis_key == "seed":
                continue
            axis_spec = space_axes.get(axis_key, {})
            allowed_values = axis_spec.get("values")
            if allowed_values and current_val in allowed_values:
                idx = allowed_values.index(current_val)
                next_idx = idx + 1 if idx + 1 < len(allowed_values) else idx - 1
                if next_idx < 0 or next_idx == idx:
                    continue
                new_val = allowed_values[next_idx]
            elif isinstance(current_val, (int, float)):
                new_val = current_val * 1.1 if isinstance(current_val, float) else current_val + 1
                new_val = round(new_val, 4) if isinstance(current_val, float) else new_val
                if "min" in axis_spec and new_val < axis_spec["min"]:
                    new_val = axis_spec["min"]
                if "max" in axis_spec and new_val > axis_spec["max"]:
                    continue
            else:
                continue

            new_axis = dict(axis_values)
            new_axis[axis_key] = new_val
            new_key = stable_json({k: v for k, v in new_axis.items() if k != "seed"})
            if new_key not in existing_keys:
                candidates.append(
                    {
                        "candidate_type": "new_point",
                        "candidate_key": new_key,
                        "axis_values": new_axis,
                        "score_signals": {
                            "parent_wr": best["mean_wr"] or 0,
                            "mean_params": best.get("mean_params") or 200000,
                            "perturbed_axis": axis_key,
                            "perturbation_ratio": round(new_val / current_val, 3) if current_val else 1.0,
                        },
                    }
                )
                if len(candidates) >= limit:
                    break

    return candidates


def generate_generic_point_candidates(conn, campaign, policy, limit: int = 5) -> list[dict]:
    """Generate point candidates for domains whose objective lives in run_metrics."""
    profile = objective_profile_for_campaign(conn, campaign)
    if not profile:
        return []

    metric_names = set(profile.get("maximize", [])) | set(profile.get("minimize", []))
    metric_names |= {constraint["metric"] for constraint in profile.get("hard_constraints", [])}
    if not metric_names:
        return []

    rows = conn.execute(
        """SELECT cr.candidate_key, cr.axis_values_json, cr.seed,
                  rm.metric_name, rm.metric_value
           FROM campaign_runs cr
           JOIN runs r ON r.id = cr.run_id
           JOIN run_metrics rm ON rm.run_id = r.id
           WHERE cr.campaign_id = ?
             AND r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')
             AND rm.metric_name IN ({})
           ORDER BY cr.candidate_key""".format(",".join("?" for _ in metric_names)),
        (campaign["id"], *sorted(metric_names)),
    ).fetchall()
    if not rows:
        return []

    grouped: dict[str, dict] = {}
    for row in rows:
        key = row["candidate_key"] or row["axis_values_json"]
        bucket = grouped.setdefault(
            key,
            {
                "candidate_key": key,
                "axis_values_json": row["axis_values_json"] or "{}",
                "seeds": set(),
                "metric_values": {},
            },
        )
        if row["seed"] is not None:
            bucket["seeds"].add(row["seed"])
        bucket["metric_values"].setdefault(row["metric_name"], []).append(float(row["metric_value"]))

    summaries = []
    for item in grouped.values():
        metrics = {name: mean(values) for name, values in item["metric_values"].items()}
        if not metric_names.issubset(metrics):
            continue
        hard_ok = all(
            metrics[constraint["metric"]] >= float(constraint["value"]) if constraint["op"] == "ge" else True
            for constraint in profile.get("hard_constraints", [])
        )
        if not hard_ok:
            continue
        summaries.append(
            {
                **item,
                "metrics": metrics,
                "utility": generic_utility(metrics, profile),
                "seed_count": len(item["seeds"]) or 1,
            }
        )
    if not summaries:
        return []

    summaries.sort(key=lambda row: row["utility"], reverse=True)
    frontier = summaries[: max(1, min(3, len(summaries)))]
    frontier_keys = {row["candidate_key"] for row in frontier}
    existing_keys = set(grouped)
    primary_max = profile.get("maximize", ["objective"])[0]
    primary_min = profile.get("minimize", ["cost"])[0] if profile.get("minimize") else None

    candidates = []
    min_runs_for_variance = policy.get("limits", {}).get("min_runs_for_variance", 2)
    for row in frontier:
        if row["seed_count"] < min_runs_for_variance:
            candidates.append(
                {
                    "candidate_type": "seed_recheck",
                    "candidate_key": row["candidate_key"],
                    "axis_values": json.loads(row["axis_values_json"] or "{}"),
                    "score_signals": {
                        "predicted_objective": row["metrics"].get(primary_max, row["utility"]),
                        "mean_cost": row["metrics"].get(primary_min, 1.0) if primary_min else 1.0,
                        "seed_count": row["seed_count"],
                        "on_frontier": True,
                        "objective_metrics": row["metrics"],
                    },
                }
            )

    space_axes: dict = {}
    space_row = get_search_space(conn, campaign["search_space_id"] or "")
    if space_row:
        try:
            space_axes = json.loads(space_row["profile_json"]).get("axes", {})
        except (json.JSONDecodeError, KeyError):
            space_axes = {}

    best = frontier[0]
    axis_values = json.loads(best["axis_values_json"] or "{}")
    for axis_key, current_val in axis_values.items():
        if axis_key == "seed":
            continue
        axis_spec = space_axes.get(axis_key, {})
        allowed_values = axis_spec.get("values")
        if allowed_values and current_val in allowed_values:
            idx = allowed_values.index(current_val)
            neighbours = []
            if idx + 1 < len(allowed_values):
                neighbours.append(allowed_values[idx + 1])
            if idx - 1 >= 0:
                neighbours.append(allowed_values[idx - 1])
        elif isinstance(current_val, (int, float)):
            neighbours = [round(current_val * 1.1, 6) if isinstance(current_val, float) else current_val + 1]
        else:
            continue
        for new_val in neighbours:
            new_axis = dict(axis_values)
            new_axis[axis_key] = new_val
            new_key = stable_json({k: v for k, v in new_axis.items() if k != "seed"})
            if new_key in existing_keys:
                continue
            candidates.append(
                {
                    "candidate_type": "new_point",
                    "candidate_key": new_key,
                    "axis_values": new_axis,
                    "score_signals": {
                        "predicted_objective": best["metrics"].get(primary_max, best["utility"]),
                        "mean_cost": best["metrics"].get(primary_min, 1.0) if primary_min else 1.0,
                        "perturbed_axis": axis_key,
                        "on_frontier": new_key in frontier_keys,
                        "objective_metrics": best["metrics"],
                    },
                }
            )
            if len(candidates) >= limit:
                return candidates

    return candidates


def generate_branch_candidates(conn, campaign, policy, limit: int = 3, branch_policy: dict | None = None) -> list[dict]:
    """Generate continue_branch and eval_upgrade candidates from branch tree."""
    candidates = []
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
        (campaign["id"],),
    ).fetchall()
    if not rows:
        return candidates

    lr_decay_factor = 0.1
    if branch_policy:
        lr_spec = (
            branch_policy.get("branch_reasons", {})
            .get("lr_decay", {})
            .get("allowed_deltas", {})
            .get("learning_rate", {})
        )
        lr_decay_factor = lr_spec.get("default_factor", 0.1)

    max_eval_level = policy.get("limits", {}).get("max_eval_level", 2)
    for row in rows:
        parent_run_id = row["run_id"]
        checkpoint = get_latest_checkpoint(conn, parent_run_id)
        if not checkpoint or (row["final_win_rate"] or 0) < 0.5:
            continue

        existing_branches = conn.execute(
            "SELECT COUNT(*) FROM run_branches WHERE parent_run_id = ?",
            (parent_run_id,),
        ).fetchone()[0]
        if existing_branches >= 3:
            continue

        axis_values = json.loads(row["axis_values_json"] or "{}")
        mean_params = row["num_params"] if row["num_params"] is not None else 200000
        if (row["learning_rate"] or 0) > 0.001:
            delta_json = json.dumps({"learning_rate": lr_decay_factor}, sort_keys=True, separators=(",", ":"))
            candidates.append(
                {
                    "candidate_type": "continue_branch",
                    "candidate_key": row["candidate_key"],
                    "parent_run_id": parent_run_id,
                    "parent_checkpoint_id": checkpoint["id"],
                    "branch_reason": "lr_decay",
                    "delta_json": delta_json,
                    "axis_values": axis_values,
                    "score_signals": {
                        "parent_wr": row["final_win_rate"] or 0,
                        "mean_params": mean_params,
                        "existing_branches": existing_branches,
                        "reason": "lr_decay",
                    },
                }
            )

        seed_count = conn.execute(
            """SELECT COUNT(DISTINCT seed) FROM campaign_runs
               WHERE campaign_id = ? AND candidate_key = ?""",
            (campaign["id"], row["candidate_key"]),
        ).fetchone()[0]
        if seed_count < 2:
            candidates.append(
                {
                    "candidate_type": "continue_branch",
                    "candidate_key": row["candidate_key"],
                    "parent_run_id": parent_run_id,
                    "parent_checkpoint_id": checkpoint["id"],
                    "branch_reason": "seed_recheck",
                    "delta_json": "{}",
                    "axis_values": axis_values,
                    "score_signals": {
                        "parent_wr": row["final_win_rate"] or 0,
                        "mean_params": mean_params,
                        "existing_branches": existing_branches,
                        "reason": "seed_recheck",
                    },
                }
            )

        if (row["final_win_rate"] or 0) >= 0.6 and (row["eval_level"] or 0) < max_eval_level:
            candidates.append(
                {
                    "candidate_type": "eval_upgrade",
                    "candidate_key": row["candidate_key"],
                    "parent_run_id": parent_run_id,
                    "parent_checkpoint_id": checkpoint["id"],
                    "branch_reason": "eval_upgrade",
                    "delta_json": "{}",
                    "axis_values": axis_values,
                    "score_signals": {
                        "parent_wr": row["final_win_rate"] or 0,
                        "mean_params": mean_params,
                        "existing_branches": existing_branches,
                        "reason": "eval_upgrade",
                    },
                }
            )

    return candidates[:limit] if limit is not None else candidates
