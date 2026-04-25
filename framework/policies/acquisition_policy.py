"""
framework/acquisition_policy.py — v21.1 candidate-pool acquisition policy.

定义 acquisition policy JSON 的加载、校验与摘要。
v21.1 不直接在 raw search space 上生成点，而是在 selector 产生的合法 candidate pool 上做二次排序。
"""

from __future__ import annotations

import hashlib
import json
import os

REQUIRED_TOP_KEYS = {
    "domain",
    "name",
    "version",
    "description",
    "selector_policy_ref",
    "objectives",
    "weights",
    "priors",
    "candidate_type_bonus",
    "replay",
}

REQUIRED_WEIGHT_KEYS = {
    "predicted_wr",
    "uncertainty",
    "params_penalty",
    "wall_penalty",
    "frontier_bonus",
}

REQUIRED_PRIOR_KEYS = {
    "base_sigma",
    "min_sigma",
    "seed_shortage_bonus",
    "target_seed_count",
}

VALID_CANDIDATE_TYPES = {
    "new_point",
    "seed_recheck",
    "continue_branch",
    "eval_upgrade",
}


def load_acquisition_policy(path: str) -> dict:
    """Load and validate an acquisition policy JSON file."""
    if not os.path.isfile(path):
        raise ValueError(f"Acquisition policy file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    validate_acquisition_policy(data)
    return data


def validate_acquisition_policy(data: dict) -> None:
    """Validate an acquisition policy object."""
    if not isinstance(data, dict):
        raise ValueError("Acquisition policy must be a JSON object")

    missing = REQUIRED_TOP_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing top-level keys: {sorted(missing)}")

    domain = data["domain"]
    ref = data["selector_policy_ref"]
    if not isinstance(ref, dict) or "domain" not in ref or "name" not in ref:
        raise ValueError("selector_policy_ref must contain domain and name")
    if ref["domain"] != domain:
        raise ValueError(
            f"selector_policy_ref domain '{ref['domain']}' does not match policy domain '{domain}'"
        )

    objectives = data["objectives"]
    if not isinstance(objectives, dict):
        raise ValueError("objectives must be an object")
    maximize = objectives.get("maximize")
    minimize = objectives.get("minimize")
    if not isinstance(maximize, list) or len(maximize) == 0:
        raise ValueError("objectives.maximize must be a non-empty list")
    if not isinstance(minimize, list):
        raise ValueError("objectives.minimize must be a list")

    weights = data["weights"]
    if not isinstance(weights, dict):
        raise ValueError("weights must be an object")
    missing_weights = REQUIRED_WEIGHT_KEYS - set(weights.keys())
    if missing_weights:
        raise ValueError(f"weights missing keys: {sorted(missing_weights)}")
    for key, value in weights.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"weights.{key} must be numeric")
        if value < 0:
            raise ValueError(f"weights.{key} must be non-negative")

    priors = data["priors"]
    if not isinstance(priors, dict):
        raise ValueError("priors must be an object")
    missing_priors = REQUIRED_PRIOR_KEYS - set(priors.keys())
    if missing_priors:
        raise ValueError(f"priors missing keys: {sorted(missing_priors)}")
    for key, value in priors.items():
        if key == "target_seed_count":
            if not isinstance(value, int) or value < 1:
                raise ValueError("priors.target_seed_count must be a positive integer")
        else:
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"priors.{key} must be a non-negative number")

    bonuses = data["candidate_type_bonus"]
    if not isinstance(bonuses, dict):
        raise ValueError("candidate_type_bonus must be an object")
    for key, value in bonuses.items():
        if key not in VALID_CANDIDATE_TYPES:
            raise ValueError(f"Unknown candidate_type_bonus key: {key}")
        if not isinstance(value, (int, float)):
            raise ValueError(f"candidate_type_bonus.{key} must be numeric")

    replay = data["replay"]
    if not isinstance(replay, dict):
        raise ValueError("replay must be an object")
    top_k = replay.get("top_k")
    positive_labels = replay.get("positive_outcomes")
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("replay.top_k must be a positive integer")
    if not isinstance(positive_labels, list) or len(positive_labels) == 0:
        raise ValueError("replay.positive_outcomes must be a non-empty list")


def describe_acquisition_policy(policy: dict) -> str:
    """Return a human-readable summary."""
    lines = [
        f"Acquisition Policy: {policy['name']} v{policy['version']} ({policy['domain']})",
        f"Selector policy: {policy['selector_policy_ref']['name']} v{policy['selector_policy_ref'].get('version', '?')}",
        f"Objectives: maximize={policy['objectives']['maximize']} minimize={policy['objectives']['minimize']}",
        "Weights:",
    ]
    for key, value in policy["weights"].items():
        lines.append(f"  {key}: {value}")
    lines.append("Priors:")
    for key, value in policy["priors"].items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def policy_hash(policy: dict) -> str:
    """Stable hash for persisted evidence lineage."""
    canonical = json.dumps(policy, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
