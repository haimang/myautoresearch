"""Evaluate one spot FX route candidate against current quotes."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from portfolio import apply_conversion, liquidity_floor_status, min_headroom_ratio
from quote_graph import route_for_candidate


def portfolio_value(portfolio: dict[str, float], provider, anchor_currency: str) -> float:
    total = 0.0
    for ccy, amount in portfolio.items():
        total += float(amount) if ccy == anchor_currency else float(amount) * provider.mid_rate(ccy, anchor_currency)
    return total


def evaluate_route(
    *,
    candidate: dict,
    portfolio: dict[str, float],
    floors: dict[str, float],
    anchor_currency: str,
    provider,
) -> dict:
    route = route_for_candidate(candidate, anchor_currency=anchor_currency)
    max_legs = int(candidate.get("max_legs", len(route) - 1))
    if len(route) - 1 > max_legs:
        raise ValueError(f"route has {len(route) - 1} legs but max_legs={max_legs}")

    before_value = portfolio_value(portfolio, provider, anchor_currency)
    sell_currency = route[0]
    surplus = max(0.0, float(portfolio.get(sell_currency, 0.0)) - float(floors.get(sell_currency, 0.0)))
    fraction = float(candidate.get("rebalance_fraction", 0.5))
    first_sell_amount = surplus * max(0.0, min(1.0, fraction))

    current = dict(portfolio)
    legs = []
    quotes = []
    sell_amount = first_sell_amount
    embedded_spread = []
    validity_remaining = []
    settlement_lag = []

    for idx, (sell, buy) in enumerate(zip(route[:-1], route[1:])):
        state_before = dict(current)
        quote = provider.quote(sell, buy, sell_amount)
        quotes.append(quote)
        current = apply_conversion(
            current,
            sell_currency=sell,
            buy_currency=buy,
            sell_amount=quote["sell_amount"],
            buy_amount=quote["buy_amount"],
        )
        spread_bps = (1.0 - (quote["client_rate"] / quote["mid_rate"])) * 10000.0
        embedded_spread.append(spread_bps)
        valid_to = datetime.fromisoformat(quote["valid_to_at"])
        validity_remaining.append(max(0.0, (valid_to - datetime.now(timezone.utc)).total_seconds()))
        settlement_lag.append(float(quote.get("settlement_lag_s", 60)))
        legs.append({
            "leg_index": idx,
            "sell_currency": sell,
            "buy_currency": buy,
            "sell_amount": quote["sell_amount"],
            "buy_amount": quote["buy_amount"],
            "quote": quote,
            "route_state_before_json": json.dumps(state_before, ensure_ascii=False, sort_keys=True),
            "route_state_after_json": json.dumps(current, ensure_ascii=False, sort_keys=True),
        })
        sell_amount = quote["buy_amount"]

    after_value = portfolio_value(current, provider, anchor_currency)
    floors_ok = liquidity_floor_status(current, floors)
    locked_value = first_sell_amount * provider.mid_rate(sell_currency, anchor_currency)
    metrics = {
        "liquidity_floor_ok": 1.0 if all(floors_ok.values()) else 0.0,
        "liquidity_headroom_ratio": min_headroom_ratio(current, floors),
        "preservation_ratio": after_value / before_value if before_value else 0.0,
        "spot_uplift_bps": ((after_value / before_value) - 1.0) * 10000.0 if before_value else 0.0,
        "quote_validity_remaining_s": min(validity_remaining) if validity_remaining else 0.0,
        "embedded_spread_bps": sum(embedded_spread) / len(embedded_spread) if embedded_spread else 0.0,
        "route_leg_count": float(len(legs)),
        "settlement_lag_s": sum(settlement_lag) if settlement_lag else 0.0,
        "locked_funds_ratio": locked_value / before_value if before_value else 0.0,
    }
    return {
        "route": route,
        "quotes": quotes,
        "legs": legs,
        "portfolio_before": portfolio,
        "portfolio_after": current,
        "portfolio_value_before": before_value,
        "portfolio_value_after": after_value,
        "liquidity_floor_status": floors_ok,
        "metrics": metrics,
    }
