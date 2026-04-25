"""
framework/selector_policy.py — v21 Surrogate-Guided Selector

负责 selector policy JSON 的加载、校验、摘要。
定义 recommendation 类型词表与评分权重。
"""

import json
import os
import sys
from typing import Any

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# 1. Load / validate selector policy JSON
# ---------------------------------------------------------------------------

REQUIRED_TOP_KEYS = {
    "domain", "name", "version",
    "search_space_ref", "stage_policy_ref", "branch_policy_ref",
    "candidate_kinds", "score_weights",
}

VALID_CANDIDATE_KINDS = {
    "new_point", "seed_recheck", "continue_branch",
    "eval_upgrade", "skip_dominated",
}

REQUIRED_SCORE_WEIGHT_KEYS = {
    "frontier_gap", "uncertainty", "cost_penalty", "dominance_penalty",
}

REASON_KIND_KEYS = {
    "description", "max_per_batch", "default_budget_s",
}


def load_selector_policy(path: str) -> dict:
    """Load a selector policy JSON from disk."""
    if not os.path.isfile(path):
        raise ValueError(f"Selector policy file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    validate_selector_policy(data)
    return data


def validate_selector_policy(data: dict) -> None:
    """Validate a selector policy object. Raises ValueError on failure."""
    if not isinstance(data, dict):
        raise ValueError("Selector policy must be a JSON object")

    missing = REQUIRED_TOP_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing top-level keys: {sorted(missing)}")

    # Validate refs have consistent domain
    domain = data["domain"]
    for ref_name in ("search_space_ref", "stage_policy_ref", "branch_policy_ref"):
        ref = data.get(ref_name, {})
        if not isinstance(ref, dict) or "domain" not in ref or "name" not in ref:
            raise ValueError(f"{ref_name} must be an object with domain and name")
        if ref["domain"] != domain:
            raise ValueError(
                f"{ref_name} domain '{ref['domain']}' does not match policy domain '{domain}'"
            )

    # Validate candidate_kinds
    kinds = data["candidate_kinds"]
    if not isinstance(kinds, dict) or len(kinds) == 0:
        raise ValueError("candidate_kinds must be a non-empty object")
    for kind_name, kind_cfg in kinds.items():
        if kind_name not in VALID_CANDIDATE_KINDS:
            raise ValueError(f"Unknown candidate kind: '{kind_name}'; must be one of {VALID_CANDIDATE_KINDS}")
        if not isinstance(kind_cfg, dict):
            raise ValueError(f"Candidate kind '{kind_name}' must be an object")
        missing_keys = REASON_KIND_KEYS - set(kind_cfg.keys())
        if missing_keys:
            raise ValueError(f"Candidate kind '{kind_name}' missing keys: {sorted(missing_keys)}")

    # Validate score_weights
    weights = data["score_weights"]
    if not isinstance(weights, dict):
        raise ValueError("score_weights must be an object")
    missing_weights = REQUIRED_SCORE_WEIGHT_KEYS - set(weights.keys())
    if missing_weights:
        raise ValueError(f"score_weights missing keys: {sorted(missing_weights)}")
    for key, val in weights.items():
        if not isinstance(val, (int, float)):
            raise ValueError(f"score_weights['{key}'] must be numeric")
        if val < 0:
            raise ValueError(f"score_weights['{key}'] must be non-negative, got {val}")

    # Validate limits (optional field)
    limits = data.get("limits")
    if limits is not None:
        if not isinstance(limits, dict):
            raise ValueError("'limits' must be an object")
        for key in ("default_batch_size", "min_runs_for_variance"):
            if key in limits and not isinstance(limits[key], int):
                raise ValueError(f"limits.{key} must be an integer")
        if "max_wr_std_for_confident" in limits:
            val = limits["max_wr_std_for_confident"]
            if not isinstance(val, (int, float)) or val < 0:
                raise ValueError("limits.max_wr_std_for_confident must be a non-negative number")


def describe_selector_policy(policy: dict) -> str:
    """Return a human-readable summary of the selector policy."""
    lines = [
        f"Selector Policy: {policy['name']} v{policy['version']} ({policy['domain']})",
        f"Search space: {policy['search_space_ref']['name']} v{policy['search_space_ref'].get('version', '?')}",
        f"Stage policy: {policy['stage_policy_ref']['name']} v{policy['stage_policy_ref'].get('version', '?')}",
        f"Branch policy: {policy['branch_policy_ref']['name']} v{policy['branch_policy_ref'].get('version', '?')}",
        f"Candidate kinds ({len(policy['candidate_kinds'])}):",
    ]
    for name, cfg in policy["candidate_kinds"].items():
        lines.append(f"  {name}: {cfg['description']} (max/batch={cfg['max_per_batch']}, budget={cfg['default_budget_s']}s)")
    lines.append("Score weights:")
    for k, v in policy["score_weights"].items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------

def list_candidate_kinds(policy: dict) -> list[str]:
    """Return all candidate kind names."""
    return list(policy.get("candidate_kinds", {}).keys())


def get_candidate_kind_config(policy: dict, kind: str) -> dict | None:
    """Return the config for a specific candidate kind, or None."""
    return policy.get("candidate_kinds", {}).get(kind)


def get_score_weights(policy: dict) -> dict:
    """Return score weights dict."""
    return policy.get("score_weights", {})


def policy_hash(policy: dict) -> str:
    """Return a stable hash of the policy for cache / comparison."""
    canonical = json.dumps(policy, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    import hashlib
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
