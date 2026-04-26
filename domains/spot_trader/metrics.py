"""Metric names for the spot FX domain."""

MAXIMIZE = [
    "liquidity_headroom_ratio",
    "preservation_ratio",
    "spot_uplift_bps",
    "quote_validity_remaining_s",
]

MINIMIZE = [
    "embedded_spread_bps",
    "route_leg_count",
    "settlement_lag_s",
    "locked_funds_ratio",
    "insufficient_balance_count",
    "expired_quote_count",
]

HARD = ["liquidity_floor_ok", "liquidity_breach_count"]


def metric_rows(metrics: dict[str, float]) -> list[dict]:
    rows = []
    for name, value in metrics.items():
        if name in MAXIMIZE:
            direction = "maximize"
            role = "objective"
        elif name in MINIMIZE:
            direction = "minimize"
            role = "objective"
        elif name in HARD or name.endswith("_count"):
            direction = "none"
            role = "constraint"
        else:
            direction = "none"
            role = "diagnostic"
        unit = "ratio" if name.endswith("_ratio") or name.endswith("_ok") else None
        if name.endswith("_bps"):
            unit = "bps"
        if name.endswith("_s"):
            unit = "seconds"
        rows.append({
            "metric_name": name,
            "metric_value": float(value),
            "metric_unit": unit,
            "metric_role": role,
            "direction": direction,
            "source": "spot_trader_mock",
        })
    return rows
