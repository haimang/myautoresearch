"""Deterministic mock quote provider for local spot FX tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import uuid

from scenario_presets import BASE_CNY_PER_UNIT, SCENARIO_PRESETS


class MockQuoteProvider:
    provider_name = "mock"
    environment = "local_mock"

    def __init__(self, *, scenario: str = "base", quote_validity_seconds: int = 1800):
        self.scenario = scenario
        self.quote_validity_seconds = quote_validity_seconds
        if scenario not in SCENARIO_PRESETS:
            raise ValueError(f"unknown quote scenario: {scenario}")
        self.preset = SCENARIO_PRESETS[scenario]

    def mid_rate(self, sell_currency: str, buy_currency: str) -> float:
        if sell_currency not in BASE_CNY_PER_UNIT or buy_currency not in BASE_CNY_PER_UNIT:
            raise ValueError(f"unsupported currency pair: {sell_currency}/{buy_currency}")
        return BASE_CNY_PER_UNIT[sell_currency] / BASE_CNY_PER_UNIT[buy_currency]

    def _spread_bps(self, sell_currency: str, buy_currency: str) -> float:
        base = float(self.preset.get("spread_base_bps", 12.0))
        pair_hash = int(hashlib.sha256(f"{sell_currency}/{buy_currency}".encode()).hexdigest()[:6], 16)
        skew = (pair_hash % 9) - 4
        spread = base + skew
        spread += float(self.preset.get("currency_spread_adjust_bps", {}).get(sell_currency, 0.0))
        spread += float(self.preset.get("currency_spread_adjust_bps", {}).get(buy_currency, 0.0))
        spread += float(self.preset.get("pair_spread_adjust_bps", {}).get(f"{sell_currency}/{buy_currency}", 0.0))
        for ccy, adjust in self.preset.get("pair_contains_adjust_bps", {}).items():
            if sell_currency == ccy or buy_currency == ccy:
                spread += float(adjust)
        return max(1.0, spread)

    def _validity_seconds(self, sell_currency: str, buy_currency: str) -> int:
        pair = f"{sell_currency}/{buy_currency}"
        if pair in self.preset.get("pair_validity_s", {}):
            return int(self.preset["pair_validity_s"][pair])
        for ccy in (sell_currency, buy_currency):
            if ccy in self.preset.get("currency_validity_s", {}):
                return int(self.preset["currency_validity_s"][ccy])
        return int(self.preset.get("default_validity_s", self.quote_validity_seconds))

    def quote(self, sell_currency: str, buy_currency: str, sell_amount: float) -> dict:
        now = datetime.now(timezone.utc)
        validity = self._validity_seconds(sell_currency, buy_currency)
        mid = self.mid_rate(sell_currency, buy_currency)
        spread_bps = self._spread_bps(sell_currency, buy_currency)
        client_rate = mid * (1.0 - spread_bps / 10000.0)
        buy_amount = float(sell_amount) * client_rate
        quote_id = f"mock-{uuid.uuid4().hex[:16]}"
        return {
            "provider": self.provider_name,
            "environment": self.environment,
            "quote_source": self.scenario,
            "sell_currency": sell_currency,
            "buy_currency": buy_currency,
            "sell_amount": float(sell_amount),
            "buy_amount": buy_amount,
            "client_rate": client_rate,
            "mid_rate": mid,
            "awx_rate": client_rate,
            "quote_id": quote_id,
            "valid_from_at": now.isoformat(),
            "valid_to_at": (now + timedelta(seconds=validity)).isoformat(),
            "conversion_date": now.date().isoformat(),
            "quote_latency_ms": float(self.preset.get("quote_latency_ms", 1.0)),
            "settlement_lag_s": int(self.preset.get("settlement_lag_s", 60)),
            "rate_details": {"embedded_spread_bps": spread_bps},
        }
