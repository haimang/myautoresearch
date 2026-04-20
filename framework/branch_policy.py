"""
framework/branch_policy.py — v20.3 Continuation / Trajectory Explorer

负责 branch policy JSON 的加载、校验、摘要，以及 delta 应用逻辑。
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
# 1. Load / validate branch policy JSON
# ---------------------------------------------------------------------------

REQUIRED_TOP_KEYS = {"domain", "name", "version", "search_space_ref", "stage_policy_ref", "branch_reasons"}

REQUIRED_REASON_KEYS = {"description", "allowed_deltas", "preserves_protocol"}

DELTA_TYPE_KEYS = {"type", "default_factor", "default_delta", "default_value", "min_factor", "max_factor", "min_delta", "max_delta"}

VALID_DELTA_TYPES = {"multiply", "add", "set"}


def load_branch_policy(path: str) -> dict:
    """Load a branch policy JSON from disk."""
    if not os.path.isfile(path):
        raise ValueError(f"Branch policy file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    validate_branch_policy(data)
    return data


def validate_branch_policy(data: dict) -> None:
    """Validate a branch policy object. Raises ValueError on failure."""
    if not isinstance(data, dict):
        raise ValueError("Branch policy must be a JSON object")

    missing = REQUIRED_TOP_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing top-level keys: {sorted(missing)}")

    # Validate search_space_ref
    ssr = data["search_space_ref"]
    if not isinstance(ssr, dict) or "domain" not in ssr or "name" not in ssr:
        raise ValueError("search_space_ref must be an object with domain and name")

    # Validate stage_policy_ref
    spr = data["stage_policy_ref"]
    if not isinstance(spr, dict) or "domain" not in spr or "name" not in spr:
        raise ValueError("stage_policy_ref must be an object with domain and name")

    # Validate branch_reasons
    reasons = data["branch_reasons"]
    if not isinstance(reasons, dict) or len(reasons) == 0:
        raise ValueError("branch_reasons must be a non-empty object")

    for reason_name, reason_cfg in reasons.items():
        if not isinstance(reason_cfg, dict):
            raise ValueError(f"Reason '{reason_name}' must be an object")

        missing_keys = REQUIRED_REASON_KEYS - set(reason_cfg.keys())
        if missing_keys:
            raise ValueError(f"Reason '{reason_name}' missing keys: {sorted(missing_keys)}")

        allowed_deltas = reason_cfg["allowed_deltas"]
        if not isinstance(allowed_deltas, dict) or len(allowed_deltas) == 0:
            raise ValueError(f"Reason '{reason_name}': allowed_deltas must be a non-empty object")

        for param_name, delta_spec in allowed_deltas.items():
            if not isinstance(delta_spec, dict):
                raise ValueError(f"Reason '{reason_name}': delta for '{param_name}' must be an object")
            if "type" not in delta_spec:
                raise ValueError(f"Reason '{reason_name}': delta for '{param_name}' missing 'type'")
            if delta_spec["type"] not in VALID_DELTA_TYPES:
                raise ValueError(
                    f"Reason '{reason_name}': delta type '{delta_spec['type']}' invalid; "
                    f"must be one of {VALID_DELTA_TYPES}"
                )

        # eval_upgrade must have allowed_protocol_changes
        if reason_name == "eval_upgrade":
            if not reason_cfg.get("allowed_protocol_changes"):
                raise ValueError(
                    f"Reason '{reason_name}' must define allowed_protocol_changes"
                )

    # Validate that expected reasons exist
    expected_reasons = {"lr_decay", "mcts_upshift", "eval_upgrade", "seed_recheck", "buffer_or_spc_adjust"}
    missing_reasons = expected_reasons - set(reasons.keys())
    if missing_reasons:
        raise ValueError(f"Missing expected branch reasons: {sorted(missing_reasons)}")


def describe_branch_policy(policy: dict) -> str:
    """Return a human-readable summary of the branch policy."""
    lines = [
        f"Branch Policy: {policy['name']} v{policy['version']} ({policy['domain']})",
        f"Search space: {policy['search_space_ref']['name']} v{policy['search_space_ref']['version']}",
        f"Stage policy: {policy['stage_policy_ref']['name']} v{policy['stage_policy_ref']['version']}",
        f"Branch reasons ({len(policy['branch_reasons'])}):",
    ]
    for name, cfg in policy["branch_reasons"].items():
        deltas = ", ".join(cfg["allowed_deltas"].keys())
        proto = "protocol-preserved" if cfg["preserves_protocol"] else "protocol-may-change"
        lines.append(f"  {name}: {cfg['description']}")
        lines.append(f"      deltas=[{deltas}]  {proto}")
        if cfg.get("example"):
            lines.append(f"      example: {cfg['example']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Delta application
# ---------------------------------------------------------------------------

def apply_delta(parent_params: dict, reason: str, branch_policy: dict, override: dict | None = None) -> dict:
    """Apply a branch reason's default delta to parent parameters.

    Returns a new dict of child parameters.  Does not mutate parent_params.
    override: optional explicit delta values (bypass defaults).
    """
    reasons = branch_policy.get("branch_reasons", {})
    if reason not in reasons:
        raise ValueError(f"Unknown branch reason: {reason}")

    reason_cfg = reasons[reason]
    child = dict(parent_params)
    override = override or {}

    for param_name, delta_spec in reason_cfg["allowed_deltas"].items():
        if param_name not in child and param_name not in override:
            continue

        current = child.get(param_name)
        # Use override if provided, else default
        if param_name in override:
            new_val = override[param_name]
        else:
            new_val = _compute_default_delta(current, delta_spec)

        child[param_name] = new_val

    return child


def _compute_default_delta(current, delta_spec: dict):
    """Compute the default delta value from spec."""
    dtype = delta_spec["type"]
    if dtype == "multiply":
        factor = delta_spec.get("default_factor", 1.0)
        return (current * factor) if current is not None else factor
    elif dtype == "add":
        delta = delta_spec.get("default_delta", 0)
        return (current + delta) if current is not None else delta
    elif dtype == "set":
        return delta_spec.get("default_value")
    return current


def validate_delta(reason: str, delta: dict, branch_policy: dict) -> None:
    """Validate that an explicit delta override is within policy bounds.

    Raises ValueError on out-of-bounds or illegal parameter changes.
    """
    reasons = branch_policy.get("branch_reasons", {})
    if reason not in reasons:
        raise ValueError(f"Unknown branch reason: {reason}")

    reason_cfg = reasons[reason]
    allowed = reason_cfg["allowed_deltas"]

    for param_name, new_val in delta.items():
        if param_name not in allowed:
            raise ValueError(
                f"Reason '{reason}' does not allow delta on parameter '{param_name}'"
            )
        spec = allowed[param_name]
        dtype = spec["type"]

        if dtype == "multiply":
            if new_val is not None:
                min_f = spec.get("min_factor")
                max_f = spec.get("max_factor")
                if min_f is not None and new_val < min_f:
                    raise ValueError(f"Factor {new_val} < min {min_f} for '{param_name}'")
                if max_f is not None and new_val > max_f:
                    raise ValueError(f"Factor {new_val} > max {max_f} for '{param_name}'")
        elif dtype == "add":
            if new_val is not None:
                min_d = spec.get("min_delta")
                max_d = spec.get("max_delta")
                if min_d is not None and new_val < min_d:
                    raise ValueError(f"Delta {new_val} < min {min_d} for '{param_name}'")
                if max_d is not None and new_val > max_d:
                    raise ValueError(f"Delta {new_val} > max {max_d} for '{param_name}'")


# ---------------------------------------------------------------------------
# 3. Reason / delta helpers
# ---------------------------------------------------------------------------

def list_reasons(policy: dict) -> list[str]:
    """Return all branch reason names."""
    return list(policy.get("branch_reasons", {}).keys())


def get_reason_config(policy: dict, reason: str) -> dict | None:
    """Return the config for a specific reason, or None."""
    return policy.get("branch_reasons", {}).get(reason)


def reason_preserves_protocol(policy: dict, reason: str) -> bool:
    """Return True if the reason preserves campaign protocol."""
    cfg = get_reason_config(policy, reason)
    return cfg["preserves_protocol"] if cfg else True


def get_allowed_protocol_changes(policy: dict, reason: str) -> list[str]:
    """Return list of protocol fields this reason is allowed to change."""
    cfg = get_reason_config(policy, reason)
    return cfg.get("allowed_protocol_changes", []) if cfg else []
