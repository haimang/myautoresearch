#!/usr/bin/env python3
"""Pareto 前沿散点图可视化模块。

生成 Pareto front 散点图：前沿点高亮 + frontier 折线，被支配点灰色。
支持任意 x/y 轴字段、可选的点大小映射、自动标注。

用法（被 analyze.py --pareto --plot 调用）:
    from pareto_plot import plot_pareto
    path = plot_pareto(front, dominated, x_key="params", y_key="wr")
"""

import csv
import json
import os
from typing import Callable, Optional

# Axis display metadata: (label, format function)
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


def _format_by_kind(kind: str, val) -> str:
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


def _get_label(key: str, axis_meta: Optional[dict] = None) -> str:
    """Get display label for an axis key."""
    if axis_meta and key in axis_meta:
        return axis_meta[key].get("label", key)
    if key in _AXIS_META:
        return _AXIS_META[key][0]
    return key


def _fmt_val(key: str, val, axis_meta: Optional[dict] = None) -> str:
    """Format a value for annotation."""
    if val is None:
        return "?"
    if axis_meta and key in axis_meta:
        return _format_by_kind(axis_meta[key].get("format", "number"), val)
    if key in _AXIS_META:
        return _AXIS_META[key][1](val)
    return str(val)


def plot_pareto(
    front: list[dict],
    dominated: list[dict],
    x_key: str = "params",
    y_key: str = "wr",
    size_key: Optional[str] = None,
    label_key: str = "arch",
    output_path: str = "output/pareto_front.png",
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    figsize: tuple = (12, 8),
    dpi: int = 150,
    eval_level: Optional[int] = None,
    sweep_tag: Optional[str] = None,
    axis_meta: Optional[dict] = None,
    knee_point: Optional[dict] = None,
    annotate_dominated: bool = False,
    max_front_annotations: int = 6,
    smooth_front: bool = True,
    annotation_label_key: str | None = None,
) -> str:
    """生成 Pareto 前沿散点图，返回输出文件路径。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # --- Dominated points (gray, small) ---
    dom_x = [p.get(x_key) for p in dominated if p.get(x_key) is not None and p.get(y_key) is not None]
    dom_y = [p.get(y_key) for p in dominated if p.get(x_key) is not None and p.get(y_key) is not None]
    dom_labels = [p.get(label_key, "") for p in dominated if p.get(x_key) is not None and p.get(y_key) is not None]

    if dom_x:
        ax.scatter(dom_x, dom_y, c="#BBBBBB", s=60, alpha=0.6, edgecolors="#999999",
                   linewidths=0.5, zorder=2, label=f"Dominated ({len(dom_x)})")
        if annotate_dominated:
            for x, y, lbl in zip(dom_x, dom_y, dom_labels):
                ax.annotate(_short_label(lbl), (x, y), textcoords="offset points", xytext=(6, -8),
                            fontsize=7, color="#888888", alpha=0.8)

    # --- Front points (blue, large) ---
    front_x = [p.get(x_key) for p in front if p.get(x_key) is not None and p.get(y_key) is not None]
    front_y = [p.get(y_key) for p in front if p.get(x_key) is not None and p.get(y_key) is not None]
    front_labels = [p.get(label_key, "") for p in front if p.get(x_key) is not None and p.get(y_key) is not None]
    front_filtered = [p for p in front if p.get(x_key) is not None and p.get(y_key) is not None]

    if size_key:
        sizes = []
        for p in front_filtered:
            v = p.get(size_key)
            sizes.append(v if v and v > 0 else 100)
        max_s = max(sizes) if sizes else 1
        sizes = [40 + 160 * (s / max_s) for s in sizes]
    else:
        sizes = [120] * len(front_x)

    if front_x:
        ax.scatter(front_x, front_y, c="#2563EB", s=sizes, alpha=0.85,
                   edgecolors="#1D4ED8", linewidths=1.2, zorder=4,
                   label=f"Pareto Front ({len(front_x)})")

        # Frontier line (sorted by x)
        if len(front_x) > 1:
            sorted_pairs = sorted(zip(front_x, front_y), key=lambda p: p[0])
            line_x, line_y = zip(*sorted_pairs)
            ax.plot(line_x, line_y, color="#EF4444", linestyle="--", linewidth=1.5,
                    alpha=0.7, zorder=3, label="Frontier")
            if smooth_front and len(set(front_x)) >= 3:
                smooth = _smooth_curve(front_x, front_y)
                if smooth:
                    sx, sy = smooth
                    ax.plot(sx, sy, color="#DC2626", linestyle="-", linewidth=1.8,
                            alpha=0.65, zorder=3, label="Smoothed frontier")

        # Annotations for front points
        annotate_ids = _annotation_point_ids(front_filtered, x_key, y_key, knee_point, max_front_annotations)
        for x, y, lbl, p in zip(front_x, front_y, front_labels, front_filtered):
            if id(p) not in annotate_ids:
                continue
            y_str = _fmt_val(y_key, y, axis_meta)
            annotation = f"{_point_label(p, lbl, annotation_label_key)}\n{y_str}"
            ax.annotate(annotation, (x, y), textcoords="offset points",
                        xytext=(8, 10), fontsize=8, fontweight="bold",
                        color="#1E40AF",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF",
                                  edgecolor="#93C5FD", alpha=0.9),
                        arrowprops=dict(arrowstyle="-", color="#93C5FD", lw=0.8),
                        zorder=5)

    if knee_point and knee_point.get(x_key) is not None and knee_point.get(y_key) is not None:
        ax.scatter([knee_point[x_key]], [knee_point[y_key]], c="#F59E0B", s=220,
                   marker="*", edgecolors="#B45309", linewidths=1.2, zorder=6,
                   label="Knee")
        ax.annotate(
            f"K\n{_point_label(knee_point, knee_point.get(label_key, knee_point.get('run', '')), annotation_label_key)}",
            (knee_point[x_key], knee_point[y_key]),
            textcoords="offset points",
            xytext=(10, -22),
            fontsize=8,
            fontweight="bold",
            color="#92400E",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF3C7",
                      edgecolor="#F59E0B", alpha=0.95),
            arrowprops=dict(arrowstyle="-", color="#F59E0B", lw=0.8),
            zorder=7,
        )

    # --- Labels and title ---
    ax.set_xlabel(x_label or _get_label(x_key, axis_meta), fontsize=11)
    ax.set_ylabel(y_label or _get_label(y_key, axis_meta), fontsize=11)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    else:
        level_str = f", eval_level={eval_level}" if eval_level is not None else ""
        tag_str = f", sweep={sweep_tag}" if sweep_tag else ""
        total = len(front) + len(dominated)
        ax.set_title(
            f"Pareto Front: {_get_label(y_key, axis_meta)} vs {_get_label(x_key, axis_meta)}"
            f"\n({len(front)} front / {len(dominated)} dominated / {total} total"
            f"{level_str}{tag_str})",
            fontsize=12, fontweight="bold"
        )

    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Y-axis: percentage format for WR
    if y_key == "wr":
        from matplotlib.ticker import PercentFormatter
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    if x_key == "wr":
        from matplotlib.ticker import PercentFormatter
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    fig.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def _short_label(value: object, max_len: int = 36) -> str:
    text = str(value or "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _point_label(point: dict, fallback: object, annotation_label_key: str | None = None) -> str:
    if annotation_label_key and point.get(annotation_label_key):
        return _short_label(point[annotation_label_key])
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
        return _short_label(" | ".join(parts), 48)
    return _short_label(fallback, 48)


def _annotation_point_ids(front: list[dict], x_key: str, y_key: str, knee_point: dict | None, limit: int) -> set[int]:
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


def _smooth_curve(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]] | None:
    try:
        import numpy as np
    except Exception:
        return None
    grouped: dict[float, list[float]] = {}
    for x, y in zip(xs, ys):
        grouped.setdefault(float(x), []).append(float(y))
    if len(grouped) < 3:
        return None
    x_unique = np.array(sorted(grouped), dtype=float)
    y_unique = np.array([max(grouped[x]) for x in x_unique], dtype=float)
    degree = min(3, len(x_unique) - 1)
    try:
        coeff = np.polyfit(x_unique, y_unique, degree)
        sx = np.linspace(float(x_unique.min()), float(x_unique.max()), 200)
        sy = np.polyval(coeff, sx)
    except Exception:
        return None
    lo = float(min(y_unique))
    hi = float(max(y_unique))
    pad = (hi - lo) * 0.15 if hi > lo else 1.0
    sy = np.clip(sy, lo - pad, hi + pad)
    return sx.tolist(), sy.tolist()


def export_front_table(
    front: list[dict],
    *,
    csv_path: str,
    md_path: str,
    metrics: list[str],
    label_key: str = "label",
) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    rows = []
    for idx, point in enumerate(front, 1):
        axis = point.get("axis_values") or {}
        row = {
            "id": "K" if point.get("_is_knee") else f"F{idx}",
            "run": point.get("run_full") or point.get("run"),
            "label": _point_label(point, point.get(label_key, point.get("run", ""))),
            "sell_currency": axis.get("sell_currency"),
            "buy_currency": axis.get("buy_currency"),
            "route_template": axis.get("route_template"),
            "quote_scenario": axis.get("quote_scenario"),
            "treasury_scenario": axis.get("treasury_scenario"),
            "rebalance_fraction": axis.get("rebalance_fraction"),
            "sell_amount_mode": axis.get("sell_amount_mode"),
        }
        for metric in metrics:
            row[metric] = point.get(metric)
        rows.append(row)
    fieldnames = list(rows[0].keys()) if rows else ["id", "run", "label"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(k, "")) for k in fieldnames) + " |\n")


def plot_pareto_artifacts(
    front: list[dict],
    dominated: list[dict],
    *,
    output_path: str,
    x_key: str,
    y_key: str,
    axis_meta: Optional[dict] = None,
    knee_point: Optional[dict] = None,
    label_key: str = "label",
    metrics: list[str] | None = None,
) -> dict[str, str]:
    """Write overview/front-only/knee/table artifacts next to output_path."""
    base_dir = os.path.dirname(output_path) or "."
    os.makedirs(base_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(output_path))[0]
    overview_path = output_path
    front_only_path = os.path.join(base_dir, f"{stem}_front_only.png")
    knee_path = os.path.join(base_dir, f"{stem}_knee_zoom.png")
    json_path = os.path.join(base_dir, f"{stem}_front.json")
    csv_path = os.path.join(base_dir, f"{stem}_front.csv")
    md_path = os.path.join(base_dir, f"{stem}_front.md")

    marked_front = []
    for point in front:
        clone = dict(point)
        clone["_is_knee"] = bool(knee_point and point.get("run_full") == knee_point.get("run_full"))
        marked_front.append(clone)

    plot_pareto(
        marked_front,
        dominated,
        x_key=x_key,
        y_key=y_key,
        label_key=label_key,
        output_path=overview_path,
        axis_meta=axis_meta,
        knee_point=knee_point,
        annotate_dominated=False,
        max_front_annotations=6,
        smooth_front=True,
        title=f"Pareto overview: {_get_label(y_key, axis_meta)} vs {_get_label(x_key, axis_meta)}",
    )
    plot_pareto(
        marked_front,
        [],
        x_key=x_key,
        y_key=y_key,
        label_key=label_key,
        output_path=front_only_path,
        axis_meta=axis_meta,
        knee_point=knee_point,
        annotate_dominated=False,
        max_front_annotations=8,
        smooth_front=True,
        title=f"Front-only decision plot: {_get_label(y_key, axis_meta)} vs {_get_label(x_key, axis_meta)}",
    )
    plot_pareto(
        marked_front,
        [],
        x_key=x_key,
        y_key=y_key,
        label_key=label_key,
        output_path=knee_path,
        axis_meta=axis_meta,
        knee_point=knee_point,
        annotate_dominated=False,
        max_front_annotations=5,
        smooth_front=True,
        title=f"Knee neighborhood: {_get_label(y_key, axis_meta)} vs {_get_label(x_key, axis_meta)}",
        figsize=(8, 6),
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"front": marked_front, "knee": knee_point}, f, indent=2, ensure_ascii=False, sort_keys=True)
    export_front_table(marked_front, csv_path=csv_path, md_path=md_path, metrics=metrics or [y_key, x_key])
    return {
        "overview": overview_path,
        "front_only": front_only_path,
        "knee_zoom": knee_path,
        "front_json": json_path,
        "front_csv": csv_path,
        "front_md": md_path,
    }
