"""Export helpers for frontier artifacts."""

from __future__ import annotations

import csv
import os

from .labels import point_label


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
            "label": point_label(point, point.get(label_key, point.get("run", ""))),
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
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(fieldnames) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row.get(key, "")) for key in fieldnames) + " |\n")
