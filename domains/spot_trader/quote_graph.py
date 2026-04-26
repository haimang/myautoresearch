"""Route construction helpers for spot FX candidates."""

from __future__ import annotations


def route_for_candidate(candidate: dict, *, anchor_currency: str) -> list[str]:
    sell = candidate.get("sell_currency", "EUR")
    buy = candidate.get("buy_currency", anchor_currency)
    template = candidate.get("route_template", "direct")
    if sell == buy:
        raise ValueError("sell_currency and buy_currency must differ")
    if template == "direct":
        return [sell, buy]
    if template.startswith("via_"):
        bridge = template[4:].upper()
        if bridge in (sell, buy):
            if candidate.get("reject_degenerate_bridge"):
                raise ValueError(f"degenerate bridge route: {sell}->{bridge}->{buy}")
            return [sell, buy]
        return [sell, bridge, buy]
    return [sell, buy]


def route_signature(route: list[str]) -> str:
    return "->".join(route)


def route_family(route: list[str]) -> str:
    if len(route) <= 2:
        return "direct"
    return f"via_{route[1].lower()}"
