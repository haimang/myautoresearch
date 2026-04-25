#!/usr/bin/env python3
"""Pareto 前沿散点图可视化模块。

生成 Pareto front 散点图：前沿点高亮 + frontier 折线，被支配点灰色。
支持任意 x/y 轴字段、可选的点大小映射、自动标注。

用法（被 analyze.py --pareto --plot 调用）:
    from framework.services.frontier.plotting import plot_pareto
    path = plot_pareto(front, dominated, x_key="params", y_key="wr")
"""

import json
import os
from typing import Optional

from .exports import export_front_table
from .labels import annotation_point_ids, fmt_val, get_label, point_label, short_label


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
                ax.annotate(short_label(lbl), (x, y), textcoords="offset points", xytext=(6, -8),
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
        annotate_ids = annotation_point_ids(front_filtered, x_key, y_key, knee_point, max_front_annotations)
        for x, y, lbl, p in zip(front_x, front_y, front_labels, front_filtered):
            if id(p) not in annotate_ids:
                continue
            y_str = fmt_val(y_key, y, axis_meta)
            annotation = f"{point_label(p, lbl, annotation_label_key)}\n{y_str}"
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
            f"K\n{point_label(knee_point, knee_point.get(label_key, knee_point.get('run', '')), annotation_label_key)}",
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
    ax.set_xlabel(x_label or get_label(x_key, axis_meta), fontsize=11)
    ax.set_ylabel(y_label or get_label(y_key, axis_meta), fontsize=11)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    else:
        level_str = f", eval_level={eval_level}" if eval_level is not None else ""
        tag_str = f", sweep={sweep_tag}" if sweep_tag else ""
        total = len(front) + len(dominated)
        ax.set_title(
            f"Pareto Front: {get_label(y_key, axis_meta)} vs {get_label(x_key, axis_meta)}"
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
        title=f"Pareto overview: {get_label(y_key, axis_meta)} vs {get_label(x_key, axis_meta)}",
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
        title=f"Front-only decision plot: {get_label(y_key, axis_meta)} vs {get_label(x_key, axis_meta)}",
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
        title=f"Knee neighborhood: {get_label(y_key, axis_meta)} vs {get_label(x_key, axis_meta)}",
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
