"""Provider interface for spot FX quotes."""

from __future__ import annotations

from typing import Protocol


class QuoteProvider(Protocol):
    provider_name: str
    environment: str

    def quote(self, sell_currency: str, buy_currency: str, sell_amount: float) -> dict:
        """Return one guaranteed or mock spot quote."""

    def mid_rate(self, sell_currency: str, buy_currency: str) -> float:
        """Return a mid-market buy-per-sell reference rate."""

