"""Metric names and schema mapping for the floorplan_checker domain."""

MAXIMIZE = [
    "val_acc_bedroom",
    "val_acc_bathroom",
    "val_acc_parking",
]

MINIMIZE = [
    "wall_time_s",
    "inference_latency_ms",
    "peak_memory_mb"
]

HARD = ["nan_loss_count"]

def metric_rows(metrics: dict[str, float]) -> list[dict]:
    rows = []
    for name, value in metrics.items():
        if name in MAXIMIZE:
            direction = "maximize"
            role = "objective"
        elif name in MINIMIZE:
            direction = "minimize"
            role = "objective"
        elif name in HARD:
            direction = "eq" # as defined in objective profile
            role = "constraint"
        else:
            direction = "none"
            role = "diagnostic"
            
        unit = "ratio" if "acc" in name else None
        if name.endswith("_s"):
            unit = "seconds"
        elif name.endswith("_ms"):
            unit = "ms"
        elif name.endswith("_mb"):
            unit = "MB"
            
        rows.append({
            "metric_name": name,
            "metric_value": float(value),
            "metric_unit": unit,
            "metric_role": role,
            "direction": direction,
            "source": "floorplan_mlx",
        })
    return rows
