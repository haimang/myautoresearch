"""Treasury portfolio scenarios for spot FX smoke experiments."""

from __future__ import annotations

from copy import deepcopy


TREASURY_SCENARIOS = {
    "cn_exporter_core": {
        "description": "China-based exporter with CNY anchor and large USD/EUR/JPY receivables.",
        "anchor_currency": "CNY",
        "portfolio": {
            "CNY": 8_000_000.0,
            "USD": 1_200_000.0,
            "EUR": 350_000.0,
            "JPY": 120_000_000.0,
            "HKD": 2_500_000.0,
            "SGD": 600_000.0,
        },
        "liquidity_floors": {
            "CNY": 5_000_000.0,
            "USD": 400_000.0,
            "EUR": 120_000.0,
            "JPY": 45_000_000.0,
            "HKD": 900_000.0,
            "SGD": 180_000.0,
        },
    },
    "usd_importer_mix": {
        "description": "Importer funding purchases in USD with residual CNY, AUD, SGD, and MXN balances.",
        "anchor_currency": "USD",
        "portfolio": {
            "USD": 2_400_000.0,
            "CNY": 6_000_000.0,
            "EUR": 220_000.0,
            "AUD": 480_000.0,
            "SGD": 300_000.0,
            "MXN": 8_000_000.0,
        },
        "liquidity_floors": {
            "USD": 1_200_000.0,
            "CNY": 2_200_000.0,
            "EUR": 80_000.0,
            "AUD": 160_000.0,
            "SGD": 100_000.0,
            "MXN": 2_500_000.0,
        },
    },
    "global_diversified": {
        "description": "Diversified treasury with Europe, APAC, and LatAm balances under a CNY anchor.",
        "anchor_currency": "CNY",
        "portfolio": {
            "CNY": 5_500_000.0,
            "USD": 900_000.0,
            "EUR": 420_000.0,
            "GBP": 160_000.0,
            "AUD": 300_000.0,
            "JPY": 40_000_000.0,
            "MXN": 5_000_000.0,
        },
        "liquidity_floors": {
            "CNY": 3_200_000.0,
            "USD": 280_000.0,
            "EUR": 150_000.0,
            "GBP": 60_000.0,
            "AUD": 120_000.0,
            "JPY": 15_000_000.0,
            "MXN": 1_500_000.0,
        },
    },
    "asia_procurement_hub": {
        "description": "Regional procurement treasury centered on USD with HKD/SGD/JPY operational liquidity.",
        "anchor_currency": "USD",
        "portfolio": {
            "USD": 1_800_000.0,
            "CNY": 2_800_000.0,
            "HKD": 4_200_000.0,
            "SGD": 950_000.0,
            "JPY": 55_000_000.0,
            "AUD": 250_000.0,
        },
        "liquidity_floors": {
            "USD": 850_000.0,
            "CNY": 1_000_000.0,
            "HKD": 1_600_000.0,
            "SGD": 300_000.0,
            "JPY": 20_000_000.0,
            "AUD": 90_000.0,
        },
    },
}


def get_treasury_scenario(name: str) -> dict:
    if name not in TREASURY_SCENARIOS:
        raise ValueError(f"unknown treasury scenario: {name}")
    scenario = deepcopy(TREASURY_SCENARIOS[name])
    scenario["name"] = name
    return scenario


def apply_treasury_scenario(candidate: dict) -> dict:
    name = candidate.get("treasury_scenario")
    if not name:
        return dict(candidate)
    scenario = get_treasury_scenario(name)
    materialized = dict(candidate)
    materialized.setdefault("anchor_currency", scenario["anchor_currency"])
    materialized.setdefault("portfolio", scenario["portfolio"])
    materialized.setdefault("liquidity_floors", scenario["liquidity_floors"])
    return materialized

