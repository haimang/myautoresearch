"""FX and run-metric repositories."""

from __future__ import annotations

import sqlite3
import uuid

from .common import stable_json, utc_now_iso


def save_run_metric(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    metric_name: str,
    metric_value: float,
    metric_unit: str | None = None,
    metric_role: str = "objective",
    direction: str = "none",
    source: str | None = "domain",
) -> None:
    conn.execute(
        """INSERT INTO run_metrics
           (run_id, metric_name, metric_value, metric_unit, metric_role,
            direction, source, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(run_id, metric_name) DO UPDATE SET
               metric_value = excluded.metric_value,
               metric_unit = excluded.metric_unit,
               metric_role = excluded.metric_role,
               direction = excluded.direction,
               source = excluded.source,
               created_at = excluded.created_at""",
        (run_id, metric_name, float(metric_value), metric_unit, metric_role, direction, source, utc_now_iso()),
    )
    conn.commit()


def save_run_metrics(conn: sqlite3.Connection, run_id: str, metrics: list[dict]) -> None:
    now = utc_now_iso()
    for metric in metrics:
        conn.execute(
            """INSERT INTO run_metrics
               (run_id, metric_name, metric_value, metric_unit, metric_role,
                direction, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id, metric_name) DO UPDATE SET
                   metric_value = excluded.metric_value,
                   metric_unit = excluded.metric_unit,
                   metric_role = excluded.metric_role,
                   direction = excluded.direction,
                   source = excluded.source,
                   created_at = excluded.created_at""",
            (
                run_id,
                metric["metric_name"],
                float(metric["metric_value"]),
                metric.get("metric_unit"),
                metric.get("metric_role", "objective"),
                metric.get("direction", "none"),
                metric.get("source", "domain"),
                now,
            ),
        )
    conn.commit()


def list_run_metrics(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM run_metrics WHERE run_id = ? ORDER BY metric_name",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_run_metrics_map(conn: sqlite3.Connection, run_id: str) -> dict[str, float]:
    return {m["metric_name"]: m["metric_value"] for m in list_run_metrics(conn, run_id)}


def save_quote_window(
    conn: sqlite3.Connection,
    *,
    window_id: str,
    campaign_id: str,
    anchor_currency: str,
    started_at: str,
    expires_at: str,
    max_quote_age_seconds: int,
    portfolio_snapshot_json: str,
    liquidity_floor_json: str,
    provider_config_json: str,
    status: str = "open",
) -> None:
    conn.execute(
        """INSERT INTO quote_windows
           (id, campaign_id, anchor_currency, started_at, expires_at,
            max_quote_age_seconds, portfolio_snapshot_json, liquidity_floor_json,
            provider_config_json, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               campaign_id = excluded.campaign_id,
               anchor_currency = excluded.anchor_currency,
               started_at = excluded.started_at,
               expires_at = excluded.expires_at,
               max_quote_age_seconds = excluded.max_quote_age_seconds,
               portfolio_snapshot_json = excluded.portfolio_snapshot_json,
               liquidity_floor_json = excluded.liquidity_floor_json,
               provider_config_json = excluded.provider_config_json,
               status = excluded.status""",
        (
            window_id,
            campaign_id,
            anchor_currency,
            started_at,
            expires_at,
            max_quote_age_seconds,
            portfolio_snapshot_json,
            liquidity_floor_json,
            provider_config_json,
            status,
        ),
    )
    conn.commit()


def save_fx_quote(conn: sqlite3.Connection, quote: dict) -> str:
    quote_id = quote.get("id") or str(uuid.uuid4())
    conn.execute(
        """INSERT INTO fx_quotes
           (id, quote_window_id, provider, environment, quote_source,
            sell_currency, buy_currency, sell_amount, buy_amount, client_rate,
            mid_rate, awx_rate, quote_id, valid_from_at, valid_to_at,
            conversion_date, quote_latency_ms, raw_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               raw_json = excluded.raw_json,
               quote_latency_ms = excluded.quote_latency_ms""",
        (
            quote_id,
            quote["quote_window_id"],
            quote["provider"],
            quote["environment"],
            quote["quote_source"],
            quote["sell_currency"],
            quote["buy_currency"],
            quote.get("sell_amount"),
            quote.get("buy_amount"),
            quote.get("client_rate"),
            quote.get("mid_rate"),
            quote.get("awx_rate"),
            quote.get("quote_id"),
            quote.get("valid_from_at"),
            quote.get("valid_to_at"),
            quote.get("conversion_date"),
            quote.get("quote_latency_ms"),
            quote.get("raw_json", stable_json(quote)),
            quote.get("created_at", utc_now_iso()),
        ),
    )
    conn.commit()
    return quote_id


def save_fx_route_leg(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    leg_index: int,
    sell_currency: str,
    buy_currency: str,
    sell_amount: float | None,
    buy_amount: float | None,
    quote_ref: str | None,
    route_state_before_json: str,
    route_state_after_json: str,
) -> None:
    conn.execute(
        """INSERT INTO fx_route_legs
           (run_id, leg_index, sell_currency, buy_currency, sell_amount,
            buy_amount, quote_ref, route_state_before_json, route_state_after_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            leg_index,
            sell_currency,
            buy_currency,
            sell_amount,
            buy_amount,
            quote_ref,
            route_state_before_json,
            route_state_after_json,
        ),
    )
    conn.commit()
