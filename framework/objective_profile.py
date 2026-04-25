#!/usr/bin/env python3
"""Domain-generic objective profile loader for v22.

Objective profiles define the metrics, hard constraints, display metadata, and
knee-point strategy used by generic Pareto analysis.
"""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy


VALID_DIRECTIONS = {"maximize", "minimize"}
VALID_CONSTRAINT_OPS = {"eq", "le", "ge", "lt", "gt"}
VALID_FORMATS = {"number", "ratio", "percent", "bps", "seconds", "integer"}
VALID_KNEE_METHODS = {"utopia_distance", "none"}


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def profile_hash(profile: dict) -> str:
    clean = deepcopy(profile)
    clean.pop("source_path", None)
    clean.pop("profile_hash", None)
    return hashlib.sha256(_stable_json(clean).encode("utf-8")).hexdigest()


def validate_objective_profile(profile: dict) -> None:
    if not isinstance(profile, dict):
        raise ValueError("objective profile must be an object")
    for field in ("domain", "name", "version"):
        if field not in profile:
            raise ValueError(f"objective profile missing required field '{field}'")

    maximize = profile.get("maximize", [])
    minimize = profile.get("minimize", [])
    if not isinstance(maximize, list) or not isinstance(minimize, list):
        raise ValueError("objective profile maximize/minimize must be lists")
    if not maximize and not minimize:
        raise ValueError("objective profile must define at least one objective")
    for direction, metrics in (("maximize", maximize), ("minimize", minimize)):
        for metric in metrics:
            if not isinstance(metric, str) or not metric:
                raise ValueError(f"{direction} metrics must be non-empty strings")

    seen = set()
    for metric in maximize + minimize:
        if metric in seen:
            raise ValueError(f"metric '{metric}' appears more than once in objectives")
        seen.add(metric)

    constraints = profile.get("hard_constraints", [])
    if not isinstance(constraints, list):
        raise ValueError("hard_constraints must be a list")
    for constraint in constraints:
        if not isinstance(constraint, dict):
            raise ValueError("hard constraint must be an object")
        if not isinstance(constraint.get("metric"), str) or not constraint["metric"]:
            raise ValueError("hard constraint missing metric")
        if constraint.get("op") not in VALID_CONSTRAINT_OPS:
            raise ValueError(f"invalid hard constraint op: {constraint.get('op')}")
        if "value" not in constraint:
            raise ValueError("hard constraint missing value")

    display = profile.get("display", {})
    if not isinstance(display, dict):
        raise ValueError("display must be an object")
    for metric, meta in display.items():
        if not isinstance(metric, str) or not isinstance(meta, dict):
            raise ValueError("display entries must map metric names to objects")
        fmt = meta.get("format", "number")
        if fmt not in VALID_FORMATS:
            raise ValueError(f"display.{metric}.format invalid: {fmt}")

    knee = profile.get("knee", {"method": "none"})
    if not isinstance(knee, dict):
        raise ValueError("knee must be an object")
    if knee.get("method", "none") not in VALID_KNEE_METHODS:
        raise ValueError(f"invalid knee method: {knee.get('method')}")


def normalize_objective_profile(profile: dict) -> dict:
    validate_objective_profile(profile)
    out = {
        "domain": profile["domain"],
        "name": profile["name"],
        "version": str(profile["version"]),
        "hard_constraints": profile.get("hard_constraints", []),
        "maximize": profile.get("maximize", []),
        "minimize": profile.get("minimize", []),
        "display": profile.get("display", {}),
        "knee": profile.get("knee", {"method": "none"}),
    }
    if profile.get("description"):
        out["description"] = profile["description"]
    out["profile_hash"] = profile_hash(out)
    return out


def load_objective_profile(path: str) -> dict:
    if not os.path.isfile(path):
        raise ValueError(f"Objective profile file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    profile = normalize_objective_profile(raw)
    profile["source_path"] = path
    return profile


def describe_objective_profile(profile: dict) -> str:
    lines = [
        f"Objective Profile: {profile['domain']}/{profile['name']} v{profile['version']}",
        f"  Maximize: {', '.join(profile.get('maximize', [])) or '-'}",
        f"  Minimize: {', '.join(profile.get('minimize', [])) or '-'}",
    ]
    constraints = profile.get("hard_constraints", [])
    if constraints:
        lines.append("  Hard constraints:")
        for c in constraints:
            lines.append(f"    - {c['metric']} {c['op']} {c['value']}")
    return "\n".join(lines)
