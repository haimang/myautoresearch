"""Portfolio primitives for the spot FX quote-surface domain."""

from __future__ import annotations

from copy import deepcopy


DEFAULT_ANCHOR = "CNY"
DEFAULT_PORTFOLIO = {
    "CNY": 100000.0,
    "USD": 20000.0,
    "EUR": 8000.0,
    "JPY": 1200000.0,
}
DEFAULT_LIQUIDITY_FLOORS = {
    "CNY": 70000.0,
    "USD": 10000.0,
    "EUR": 5000.0,
    "JPY": 500000.0,
}


def clone_portfolio(portfolio: dict[str, float] | None = None) -> dict[str, float]:
    return {k: float(v) for k, v in deepcopy(portfolio or DEFAULT_PORTFOLIO).items()}


def clone_floors(floors: dict[str, float] | None = None) -> dict[str, float]:
    return {k: float(v) for k, v in deepcopy(floors or DEFAULT_LIQUIDITY_FLOORS).items()}


def liquidity_floor_status(portfolio: dict[str, float], floors: dict[str, float]) -> dict[str, bool]:
    return {ccy: float(portfolio.get(ccy, 0.0)) >= float(floor) for ccy, floor in floors.items()}


def min_headroom_ratio(portfolio: dict[str, float], floors: dict[str, float]) -> float:
    ratios = []
    for ccy, floor in floors.items():
        if floor <= 0:
            continue
        ratios.append((float(portfolio.get(ccy, 0.0)) - float(floor)) / float(floor))
    return min(ratios) if ratios else 0.0


def apply_conversion(
    portfolio: dict[str, float],
    *,
    sell_currency: str,
    buy_currency: str,
    sell_amount: float,
    buy_amount: float,
) -> dict[str, float]:
    out = dict(portfolio)
    out[sell_currency] = float(out.get(sell_currency, 0.0)) - float(sell_amount)
    out[buy_currency] = float(out.get(buy_currency, 0.0)) + float(buy_amount)
    return out

