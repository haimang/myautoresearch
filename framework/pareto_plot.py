#!/usr/bin/env python3
"""Pareto 前沿散点图可视化模块。

生成 Pareto front 散点图：前沿点高亮 + frontier 折线，被支配点灰色。
支持任意 x/y 轴字段、可选的点大小映射、自动标注。

用法（被 analyze.py --pareto --plot 调用）:
    from pareto_plot import plot_pareto
    path = plot_pareto(front, dominated, x_key="params", y_key="wr")
"""

import os
from typing import Optional

# Axis display metadata: (label, format function)
_AXIS_META: dict[str, tuple[str, callable]] = {
    "wr": ("Win Rate", lambda v: f"{v:.1%}"),
    "params": ("Params", lambda v: f"{v/1000:.0f}K" if v and v < 1e6 else f"{v/1e6:.1f}M" if v else "?"),
    "wall_s": ("Wall Time (s)", lambda v: f"{v:.0f}s" if v else "?"),
    "games": ("Total Games", lambda v: f"{v:.0f}" if v else "?"),
    "cycles": ("Training Cycles", lambda v: f"{v:.0f}" if v else "?"),
    "steps": ("Training Steps", lambda v: f"{v:.0f}" if v else "?"),
    "lr": ("Learning Rate", lambda v: f"{v:.1e}" if v else "?"),
    "throughput": ("Throughput (games/s)", lambda v: f"{v:.2f}" if v else "?"),
}


def _get_label(key: str) -> str:
    """Get display label for an axis key."""
    if key in _AXIS_META:
        return _AXIS_META[key][0]
    return key


def _fmt_val(key: str, val) -> str:
    """Format a value for annotation."""
    if val is None:
        return "?"
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
        for x, y, lbl in zip(dom_x, dom_y, dom_labels):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, -8),
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

        # Annotations for front points
        for x, y, lbl, p in zip(front_x, front_y, front_labels, front_filtered):
            y_str = _fmt_val(y_key, y)
            annotation = f"{lbl}\n{y_str}"
            ax.annotate(annotation, (x, y), textcoords="offset points",
                        xytext=(8, 10), fontsize=8, fontweight="bold",
                        color="#1E40AF",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFF6FF",
                                  edgecolor="#93C5FD", alpha=0.9),
                        arrowprops=dict(arrowstyle="-", color="#93C5FD", lw=0.8),
                        zorder=5)

    # --- Labels and title ---
    ax.set_xlabel(x_label or _get_label(x_key), fontsize=11)
    ax.set_ylabel(y_label or _get_label(y_key), fontsize=11)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    else:
        level_str = f", eval_level={eval_level}" if eval_level is not None else ""
        tag_str = f", sweep={sweep_tag}" if sweep_tag else ""
        total = len(front) + len(dominated)
        ax.set_title(
            f"Pareto Front: {_get_label(y_key)} vs {_get_label(x_key)}"
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
