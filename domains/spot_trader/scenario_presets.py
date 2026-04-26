"""Scenario presets for richer spot FX smoke experiments."""

from __future__ import annotations

BASE_CNY_PER_UNIT = {
    "CNY": 1.0,
    "USD": 7.20,
    "EUR": 7.85,
    "JPY": 0.048,
    "HKD": 0.92,
    "SGD": 5.33,
    "GBP": 9.15,
    "AUD": 4.68,
    "MXN": 0.42,
}


SCENARIO_PRESETS = {
    # Backward-compatible v22 baseline presets
    "base": {
        "spread_base_bps": 12.0,
        "default_validity_s": 1800,
        "quote_latency_ms": 1.0,
        "settlement_lag_s": 60,
    },
    "wide_spread": {
        "spread_base_bps": 35.0,
        "default_validity_s": 1800,
        "quote_latency_ms": 1.3,
        "settlement_lag_s": 90,
    },
    "short_validity": {
        "spread_base_bps": 12.0,
        "default_validity_s": 120,
        "quote_latency_ms": 1.0,
        "settlement_lag_s": 60,
    },
    # Richer treasury-style regimes
    "cn_exporter": {
        "spread_base_bps": 11.0,
        "default_validity_s": 1500,
        "quote_latency_ms": 1.2,
        "settlement_lag_s": 60,
        "currency_spread_adjust_bps": {"HKD": -2.0, "USD": -1.0, "MXN": 8.0},
        "pair_spread_adjust_bps": {
            "USD/CNY": -3.0,
            "EUR/CNY": -1.0,
            "HKD/CNY": -4.0,
            "SGD/CNY": -2.0,
            "JPY/CNY": 1.0,
            "GBP/CNY": 4.0,
            "AUD/CNY": 3.0,
            "MXN/CNY": 10.0,
        },
    },
    "asia_hub": {
        "spread_base_bps": 13.0,
        "default_validity_s": 1200,
        "quote_latency_ms": 1.4,
        "settlement_lag_s": 75,
        "currency_spread_adjust_bps": {"HKD": -3.0, "SGD": -2.0, "GBP": 3.0, "MXN": 6.0},
        "pair_spread_adjust_bps": {
            "JPY/HKD": -4.0,
            "HKD/CNY": -5.0,
            "SGD/CNY": -3.0,
            "JPY/SGD": -2.0,
            "EUR/HKD": -1.0,
        },
        "currency_validity_s": {"JPY": 900, "MXN": 720},
    },
    "usd_liquidity": {
        "spread_base_bps": 10.0,
        "default_validity_s": 900,
        "quote_latency_ms": 1.1,
        "settlement_lag_s": 60,
        "currency_spread_adjust_bps": {"USD": -2.0, "MXN": 4.0},
        "pair_contains_adjust_bps": {"USD": -3.0},
        "currency_validity_s": {"MXN": 480, "JPY": 720},
    },
    "europe_corridor": {
        "spread_base_bps": 14.0,
        "default_validity_s": 1500,
        "quote_latency_ms": 1.4,
        "settlement_lag_s": 90,
        "currency_spread_adjust_bps": {"EUR": -2.0, "GBP": -1.0, "HKD": 2.0},
        "pair_spread_adjust_bps": {
            "EUR/USD": -3.0,
            "GBP/USD": -2.0,
            "EUR/CNY": -2.0,
            "GBP/EUR": -3.0,
            "GBP/CNY": -1.0,
        },
    },
    "uplift_corridor": {
        "spread_base_bps": 9.0,
        "default_validity_s": 1200,
        "quote_latency_ms": 1.5,
        "settlement_lag_s": 105,
        "currency_spread_adjust_bps": {"USD": -2.0, "HKD": -2.0, "SGD": -1.5, "MXN": 5.0},
        "pair_spread_adjust_bps": {
            "EUR/USD": -3.0,
            "USD/HKD": -4.0,
            "HKD/CNY": -5.0,
            "SGD/USD": -2.0,
            "USD/CNY": -1.0,
        },
        "pair_rate_edge_bps": {
            "EUR/USD": 5.0,
            "USD/HKD": 4.0,
            "HKD/CNY": 5.5,
            "SGD/USD": 3.5,
            "USD/CNY": 2.5,
        },
        "amount_tiers": [
            {"min_amount": 0.0, "rate_edge_bps": 0.0},
            {"min_amount": 100000.0, "rate_edge_bps": 1.5},
            {"min_amount": 500000.0, "rate_edge_bps": 3.0},
        ],
    },
    "constraint_stress": {
        "spread_base_bps": 16.0,
        "default_validity_s": 600,
        "quote_latency_ms": 1.8,
        "settlement_lag_s": 120,
        "currency_spread_adjust_bps": {"MXN": 10.0, "JPY": 4.0, "AUD": 3.0},
        "pair_contains_adjust_bps": {"USD": -1.0},
        "pair_rate_edge_bps": {
            "USD/CNY": 1.0,
            "HKD/CNY": 1.0,
            "EUR/USD": 1.5,
        },
    },
    "latam_volatility": {
        "spread_base_bps": 18.0,
        "default_validity_s": 600,
        "quote_latency_ms": 1.7,
        "settlement_lag_s": 120,
        "currency_spread_adjust_bps": {"MXN": 12.0, "USD": -1.0, "HKD": 1.0},
        "pair_spread_adjust_bps": {
            "MXN/USD": 2.0,
            "MXN/CNY": 10.0,
            "MXN/EUR": 12.0,
            "USD/CNY": -1.0,
        },
        "currency_validity_s": {"MXN": 180, "JPY": 420},
        "pair_validity_s": {"MXN/CNY": 120, "MXN/EUR": 150},
    },
}
