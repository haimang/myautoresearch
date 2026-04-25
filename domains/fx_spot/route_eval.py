"""Evaluate one spot FX route candidate against current quotes."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from portfolio import apply_conversion, liquidity_floor_status, min_headroom_ratio
from quote_graph import route_family, route_for_candidate, route_signature


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
    holding = float(portfolio.get(sell_currency, 0.0))
    floor = float(floors.get(sell_currency, 0.0))
    surplus = max(0.0, holding - floor)
    fraction = float(candidate.get("rebalance_fraction", 0.5))
    amount_mode = candidate.get("sell_amount_mode", "surplus_fraction")
    if amount_mode == "explicit_ratio":
        ratio = float(candidate.get("sell_amount_ratio", fraction))
        first_sell_amount = holding * max(0.0, ratio)
    elif amount_mode == "floor_probe":
        target_buffer = float(candidate.get("floor_buffer_target", 0.0))
        first_sell_amount = max(0.0, holding - floor * (1.0 + target_buffer))
    else:
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
    headroom_after = {
        ccy: (float(current.get(ccy, 0.0)) - float(floor_value)) / float(floor_value)
        for ccy, floor_value in floors.items()
        if float(floor_value) > 0
    }
    breach_reasons = []
    if first_sell_amount > holding:
        breach_reasons.append("insufficient_balance")
    breached = [ccy for ccy, ok in floors_ok.items() if not ok]
    if breached:
        breach_reasons.append("liquidity_floor_breach")
    expired_quotes = sum(1 for seconds in validity_remaining if seconds <= 0)
    if expired_quotes:
        breach_reasons.append("quote_expired")
    min_breach_margin = min(headroom_after.values()) if headroom_after else 0.0
    sell_amount_ratio = first_sell_amount / holding if holding else 0.0
    route_sig = route_signature(route)
    metrics = {
        "liquidity_floor_ok": 1.0 if all(floors_ok.values()) else 0.0,
        "liquidity_breach_count": float(len(breached)),
        "insufficient_balance_count": 1.0 if first_sell_amount > holding else 0.0,
        "expired_quote_count": float(expired_quotes),
        "liquidity_headroom_ratio": min_headroom_ratio(current, floors),
        "preservation_ratio": after_value / before_value if before_value else 0.0,
        "spot_uplift_bps": ((after_value / before_value) - 1.0) * 10000.0 if before_value else 0.0,
        "quote_validity_remaining_s": min(validity_remaining) if validity_remaining else 0.0,
        "embedded_spread_bps": sum(embedded_spread) / len(embedded_spread) if embedded_spread else 0.0,
        "route_leg_count": float(len(legs)),
        "effective_leg_count": float(len(legs)),
        "settlement_lag_s": sum(settlement_lag) if settlement_lag else 0.0,
        "locked_funds_ratio": locked_value / before_value if before_value else 0.0,
        "sell_amount_ratio": sell_amount_ratio,
        "breach_margin_ratio": min_breach_margin,
    }
    return {
        "route": route,
        "route_signature": route_sig,
        "route_family": route_family(route),
        "breach_reasons": breach_reasons,
        "quotes": quotes,
        "legs": legs,
        "portfolio_before": portfolio,
        "portfolio_after": current,
        "portfolio_value_before": before_value,
        "portfolio_value_after": after_value,
        "liquidity_floor_status": floors_ok,
        "liquidity_headroom_after": headroom_after,
        "metrics": metrics,
    }
