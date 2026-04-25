"""Label and formatting helpers for frontier artifacts."""

from __future__ import annotations

from typing import Callable, Optional

_AXIS_META: dict[str, tuple[str, Callable]] = {
    "wr": ("Win Rate", lambda v: f"{v:.1%}"),
    "params": ("Params", lambda v: f"{v/1000:.0f}K" if v and v < 1e6 else f"{v/1e6:.1f}M" if v else "?"),
    "wall_s": ("Wall Time (s)", lambda v: f"{v:.0f}s" if v else "?"),
    "games": ("Total Games", lambda v: f"{v:.0f}" if v else "?"),
    "cycles": ("Training Cycles", lambda v: f"{v:.0f}" if v else "?"),
    "steps": ("Training Steps", lambda v: f"{v:.0f}" if v else "?"),
    "lr": ("Learning Rate", lambda v: f"{v:.1e}" if v else "?"),
    "throughput": ("Throughput (games/s)", lambda v: f"{v:.2f}" if v else "?"),
}


def format_by_kind(kind: str, val) -> str:
    if val is None:
        return "?"
    if kind in ("ratio", "percent"):
        return f"{float(val):.1%}"
    if kind == "bps":
        return f"{float(val):.1f} bps"
    if kind == "seconds":
        return f"{float(val):.0f}s"
    if kind == "integer":
        return f"{int(val)}"
    return f"{float(val):.4g}" if isinstance(val, (int, float)) else str(val)


def get_label(key: str, axis_meta: Optional[dict] = None) -> str:
    if axis_meta and key in axis_meta:
        return axis_meta[key].get("label", key)
    if key in _AXIS_META:
        return _AXIS_META[key][0]
    return key


def fmt_val(key: str, val, axis_meta: Optional[dict] = None) -> str:
    if val is None:
        return "?"
    if axis_meta and key in axis_meta:
        return format_by_kind(axis_meta[key].get("format", "number"), val)
    if key in _AXIS_META:
        return _AXIS_META[key][1](val)
    return str(val)


def short_label(value: object, max_len: int = 36) -> str:
    text = str(value or "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def point_label(point: dict, fallback: object, annotation_label_key: str | None = None) -> str:
    if annotation_label_key and point.get(annotation_label_key):
        return short_label(point[annotation_label_key])
    axis = point.get("axis_values") or {}
    sell = axis.get("sell_currency")
    buy = axis.get("buy_currency")
    template = axis.get("route_template")
    fraction = axis.get("rebalance_fraction") or axis.get("sell_amount_ratio")
    scenario = axis.get("quote_scenario")
    if sell and buy:
        parts = [f"{sell}->{buy}"]
        if template:
            parts.append(str(template))
        if fraction is not None:
            parts.append(str(fraction))
        if scenario:
            parts.append(str(scenario).replace("_corridor", "").replace("_", "-"))
        return short_label(" | ".join(parts), 48)
    return short_label(fallback, 48)


def annotation_point_ids(front: list[dict], x_key: str, y_key: str, knee_point: dict | None, limit: int) -> set[int]:
    selected: list[dict] = []
    if knee_point:
        for point in front:
            if point.get("run_full") == knee_point.get("run_full"):
                selected.append(point)
                break
    valid = [p for p in front if p.get(x_key) is not None and p.get(y_key) is not None]
    if valid:
        selected.append(min(valid, key=lambda p: p.get(x_key)))
        selected.append(max(valid, key=lambda p: p.get(y_key)))
        selected.append(max(valid, key=lambda p: p.get("preservation_ratio", float("-inf"))))
        selected.append(max(valid, key=lambda p: p.get("quote_validity_remaining_s", float("-inf"))))
    for point in valid:
        if len({id(p) for p in selected}) >= limit:
            break
        selected.append(point)
    return {id(p) for p in selected[:limit]}
