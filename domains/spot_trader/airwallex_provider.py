"""Airwallex quote adapter boundary for spot FX.

This module intentionally exposes quote retrieval only. It does not create
conversions or move funds; local v22 tests use MockQuoteProvider instead.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class AirwallexQuoteProvider:
    provider_name = "airwallex"

    def __init__(self, *, base_url: str | None = None, api_key: str | None = None, client_id: str | None = None):
        self.base_url = (base_url or os.environ.get("AIRWALLEX_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("AIRWALLEX_API_KEY")
        self.client_id = client_id or os.environ.get("AIRWALLEX_CLIENT_ID")
        self.environment = os.environ.get("AIRWALLEX_ENV", "sandbox")
        if not self.base_url:
            raise ValueError("AIRWALLEX_BASE_URL is required")
        if not self.api_key:
            raise ValueError("AIRWALLEX_API_KEY is required")

    def quote(self, sell_currency: str, buy_currency: str, sell_amount: float) -> dict:
        payload = {
            "sell_currency": sell_currency,
            "buy_currency": buy_currency,
            "sell_amount": str(float(sell_amount)),
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/v1/fx/quotes/current",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                **({"x-client-id": self.client_id} if self.client_id else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Airwallex quote request failed: HTTP {exc.code}: {body}") from exc

    def mid_rate(self, sell_currency: str, buy_currency: str) -> float:
        raise NotImplementedError("Airwallex mid-rate lookup must be derived from quote payload fields")

