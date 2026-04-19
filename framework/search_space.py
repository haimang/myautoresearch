#!/usr/bin/env python3
"""Structured search-space profiles for autoresearch campaigns.

Profiles are JSON documents that describe:
  - the domain / profile identity
  - the benchmark protocol under which runs are comparable
  - the searchable axes and their semantics
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

VALID_AXIS_TYPES = {"int", "float", "categorical"}
VALID_SCALES = {"linear", "log"}
VALID_ROLES = {"structure", "training", "slow"}


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def profile_hash(profile: dict) -> str:
    """Stable SHA-256 hash for a normalized profile."""
    clean = deepcopy(profile)
    clean.pop("source_path", None)
    return hashlib.sha256(_stable_json(clean).encode("utf-8")).hexdigest()


def normalize_axis(name: str, axis: dict) -> dict:
    """Validate and normalize one axis definition."""
    if not isinstance(axis, dict):
        raise ValueError(f"axis '{name}' must be an object")

    axis_type = axis.get("type")
    if axis_type not in VALID_AXIS_TYPES:
        raise ValueError(
            f"axis '{name}' has invalid type '{axis_type}'; expected one of {sorted(VALID_AXIS_TYPES)}"
        )

    role = axis.get("role")
    if role not in VALID_ROLES:
        raise ValueError(
            f"axis '{name}' has invalid role '{role}'; expected one of {sorted(VALID_ROLES)}"
        )

    scale = axis.get("scale", "linear")
    if scale not in VALID_SCALES:
        raise ValueError(
            f"axis '{name}' has invalid scale '{scale}'; expected one of {sorted(VALID_SCALES)}"
        )

    has_values = "values" in axis
    has_range = "min" in axis or "max" in axis
    if has_values and has_range:
        raise ValueError(f"axis '{name}' must use either 'values' or 'min/max', not both")
    if not has_values and not has_range:
        raise ValueError(f"axis '{name}' must define either 'values' or 'min/max'")

    out = {
        "type": axis_type,
        "role": role,
        "scale": scale,
        "allow_continuation": bool(axis.get("allow_continuation", False)),
    }
    if axis.get("description"):
        out["description"] = axis["description"]

    if has_values:
        values = axis.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"axis '{name}' has invalid 'values'; expected a non-empty list")
        if axis_type == "categorical":
            if any(v is None for v in values):
                raise ValueError(f"axis '{name}' categorical values cannot contain null")
        elif axis_type == "int":
            if any(not isinstance(v, int) for v in values):
                raise ValueError(f"axis '{name}' expected integer values")
        elif axis_type == "float":
            if any(not isinstance(v, (int, float)) for v in values):
                raise ValueError(f"axis '{name}' expected numeric values")
            values = [float(v) for v in values]
        if scale == "log":
            if any(float(v) <= 0 for v in values):
                raise ValueError(f"axis '{name}' uses scale=log but includes non-positive values")
        out["values"] = values
    else:
        if "min" not in axis or "max" not in axis:
            raise ValueError(f"axis '{name}' range axes must define both 'min' and 'max'")
        min_v = axis["min"]
        max_v = axis["max"]
        if axis_type == "categorical":
            raise ValueError(f"axis '{name}' categorical axes must use 'values', not 'min/max'")
        if axis_type == "int":
            if not isinstance(min_v, int) or not isinstance(max_v, int):
                raise ValueError(f"axis '{name}' expected integer min/max")
        else:
            if not isinstance(min_v, (int, float)) or not isinstance(max_v, (int, float)):
                raise ValueError(f"axis '{name}' expected numeric min/max")
            min_v = float(min_v)
            max_v = float(max_v)
        if min_v > max_v:
            raise ValueError(f"axis '{name}' has min > max")
        if scale == "log" and (float(min_v) <= 0 or float(max_v) <= 0):
            raise ValueError(f"axis '{name}' uses scale=log but range is not strictly positive")
        out["min"] = min_v
        out["max"] = max_v

    if "default" in axis:
        out["default"] = axis["default"]

    return out


def validate_profile(profile: dict) -> None:
    """Raise ValueError if the profile is invalid."""
    if not isinstance(profile, dict):
        raise ValueError("profile must be an object")

    for field in ("domain", "name", "version", "protocol", "axes"):
        if field not in profile:
            raise ValueError(f"profile missing required field '{field}'")

    if not isinstance(profile["protocol"], dict):
        raise ValueError("profile field 'protocol' must be an object")
    if not isinstance(profile["axes"], dict) or not profile["axes"]:
        raise ValueError("profile field 'axes' must be a non-empty object")

    for name, axis in profile["axes"].items():
        normalize_axis(name, axis)


def normalize_profile(profile: dict) -> dict:
    """Return a validated normalized profile copy."""
    validate_profile(profile)
    out = {
        "domain": profile["domain"],
        "name": profile["name"],
        "version": str(profile["version"]),
        "protocol": {
            "eval_level": profile["protocol"].get("eval_level"),
            "eval_opponent": profile["protocol"].get("eval_opponent"),
            "is_benchmark": bool(profile["protocol"].get("is_benchmark", False)),
        },
        "axes": {},
    }
    if profile.get("description"):
        out["description"] = profile["description"]
    for name, axis in profile["axes"].items():
        out["axes"][name] = normalize_axis(name, axis)
    return out


def load_profile(path: str) -> dict:
    """Load, validate, and normalize a JSON profile from disk."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    profile = normalize_profile(raw)
    profile["source_path"] = path
    profile["profile_hash"] = profile_hash(profile)
    return profile


def describe_profile(profile: dict) -> str:
    """Human-readable profile summary."""
    protocol = profile["protocol"]
    role_groups: dict[str, list[str]] = {"structure": [], "training": [], "slow": []}
    for axis_name, axis in profile["axes"].items():
        role_groups[axis["role"]].append(axis_name)

    lines = [
        f"Search Space: {profile['domain']}/{profile['name']} v{profile['version']}",
        f"  Protocol: eval_level={protocol.get('eval_level')}, "
        f"eval_opponent={protocol.get('eval_opponent') or '-'}, "
        f"is_benchmark={protocol.get('is_benchmark')}",
        f"  Axes ({len(profile['axes'])}):",
    ]
    for role in ("structure", "training", "slow"):
        names = sorted(role_groups[role])
        if names:
            lines.append(f"    - {role}: {', '.join(names)}")
    return "\n".join(lines)


def _axis_contains(axis: dict, value) -> bool:
    if "values" in axis:
        return value in axis["values"]
    try:
        numeric = float(value) if axis["type"] == "float" else int(value)
    except (TypeError, ValueError):
        return False
    return axis["min"] <= numeric <= axis["max"]


def validate_selected_axes(profile: dict, selected_axes: dict[str, list]) -> None:
    """Ensure CLI-selected sweep values are legal under the profile."""
    allowed_axes = profile["axes"]
    unknown = sorted(set(selected_axes) - set(allowed_axes))
    if unknown:
        raise ValueError(f"selected axes not defined in profile: {', '.join(unknown)}")

    for name, values in selected_axes.items():
        axis = allowed_axes[name]
        for value in values:
            if not _axis_contains(axis, value):
                raise ValueError(f"axis '{name}' value {value!r} is outside profile definition")
