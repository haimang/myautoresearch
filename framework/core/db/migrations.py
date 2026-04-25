"""Schema migrations for the db package."""

from __future__ import annotations

import sqlite3


def apply_migrations(conn: sqlite3.Connection) -> None:
    for col, typ in [("resumed_from", "TEXT"), ("output_dir", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    for col, typ in [("is_benchmark", "INTEGER DEFAULT 0"), ("eval_opponent", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    for col, typ in [("num_res_blocks", "INTEGER"), ("num_filters", "INTEGER")]:
        try:
            conn.execute(f"ALTER TABLE opponents ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    for col, typ in [("sweep_tag", "TEXT"), ("seed", "INTEGER")]:
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    for col, typ in [("policy_loss", "REAL"), ("value_loss", "REAL")]:
        try:
            conn.execute(f"ALTER TABLE cycle_metrics ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    for col, typ in [("eval_unique_openings", "INTEGER")]:
        try:
            conn.execute(f"ALTER TABLE checkpoints ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    for col, typ in [("promotion_eligible", "INTEGER"), ("promotion_reason", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE checkpoints ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_breakdown (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint_id   INTEGER NOT NULL REFERENCES checkpoints(id),
            opening_index   INTEGER NOT NULL,
            opening_moves   TEXT,
            wins            INTEGER NOT NULL,
            losses          INTEGER NOT NULL,
            draws           INTEGER NOT NULL,
            avg_length      REAL,
            unique_games    INTEGER
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_breakdown_ckpt ON eval_breakdown(checkpoint_id)")
    for col, typ in [("prev_alias", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE opponents ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    for col, typ in [("eval_submitted_cycle", "INTEGER")]:
        try:
            conn.execute(f"ALTER TABLE cycle_metrics ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS frontier_snapshots (
            id              TEXT PRIMARY KEY,
            created_at      TEXT NOT NULL,
            maximize_axes   TEXT NOT NULL,
            minimize_axes   TEXT NOT NULL,
            front_run_ids   TEXT NOT NULL,
            dominated_count INTEGER NOT NULL,
            total_runs      INTEGER NOT NULL,
            eval_level      INTEGER,
            sweep_tag       TEXT
        )
        """
    )
    for col, typ in [("campaign_id", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE frontier_snapshots ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS search_spaces (
            id            TEXT PRIMARY KEY,
            created_at    TEXT NOT NULL,
            domain        TEXT NOT NULL,
            name          TEXT NOT NULL,
            version       TEXT NOT NULL,
            profile_json  TEXT NOT NULL,
            profile_hash  TEXT NOT NULL UNIQUE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS objective_profiles (
            id            TEXT PRIMARY KEY,
            created_at    TEXT NOT NULL,
            domain        TEXT NOT NULL,
            name          TEXT NOT NULL,
            version       TEXT NOT NULL,
            profile_json  TEXT NOT NULL,
            profile_hash  TEXT NOT NULL UNIQUE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS campaigns (
            id              TEXT PRIMARY KEY,
            created_at      TEXT NOT NULL,
            name            TEXT NOT NULL UNIQUE,
            domain          TEXT NOT NULL,
            train_script    TEXT NOT NULL,
            search_space_id TEXT NOT NULL REFERENCES search_spaces(id),
            protocol_json   TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'active',
            notes           TEXT
        )
        """
    )
    for col, typ in [("objective_profile_id", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE campaigns ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiment_runs (
            id                   TEXT PRIMARY KEY,
            domain               TEXT NOT NULL,
            created_at           TEXT NOT NULL,
            objective_profile_id TEXT,
            output_root          TEXT NOT NULL,
            manifest_json        TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_experiment_runs_domain ON experiment_runs(domain)")
    for col, typ in [("experiment_run_id", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE campaigns ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS campaign_runs (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id       TEXT NOT NULL REFERENCES campaigns(id),
            run_id            TEXT NOT NULL REFERENCES runs(id),
            stage             TEXT,
            sweep_tag         TEXT,
            seed              INTEGER,
            axis_values_json  TEXT NOT NULL,
            status            TEXT NOT NULL DEFAULT 'linked',
            created_at        TEXT NOT NULL,
            UNIQUE(campaign_id, run_id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_campaign_runs_campaign ON campaign_runs(campaign_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_campaign_runs_run ON campaign_runs(run_id)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS campaign_stages (
            id              TEXT PRIMARY KEY,
            campaign_id     TEXT NOT NULL REFERENCES campaigns(id),
            stage           TEXT NOT NULL,
            policy_json     TEXT NOT NULL,
            budget_json     TEXT NOT NULL,
            seed_target     INTEGER NOT NULL,
            status          TEXT NOT NULL DEFAULT 'open',
            created_at      TEXT NOT NULL,
            closed_at       TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_campaign_stages_campaign ON campaign_stages(campaign_id)")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_campaign_stages_unique ON campaign_stages(campaign_id, stage)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS promotion_decisions (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id             TEXT NOT NULL REFERENCES campaigns(id),
            from_stage              TEXT NOT NULL,
            to_stage                TEXT NOT NULL,
            candidate_key           TEXT NOT NULL,
            axis_values_json        TEXT NOT NULL,
            aggregated_metrics_json TEXT NOT NULL,
            seed_count              INTEGER NOT NULL,
            decision                TEXT NOT NULL,
            decision_rank           INTEGER,
            reason                  TEXT NOT NULL,
            created_at              TEXT NOT NULL,
            UNIQUE(campaign_id, from_stage, to_stage, candidate_key)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_promotion_decisions_campaign ON promotion_decisions(campaign_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_promotion_decisions_stage ON promotion_decisions(from_stage, to_stage)")
    for col, typ in [("candidate_key", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE campaign_runs ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_branches (
            id                  TEXT PRIMARY KEY,
            campaign_id         TEXT NOT NULL REFERENCES campaigns(id),
            parent_run_id       TEXT NOT NULL REFERENCES runs(id),
            parent_checkpoint_id INTEGER REFERENCES checkpoints(id),
            child_run_id        TEXT REFERENCES runs(id),
            from_stage          TEXT NOT NULL,
            branch_reason       TEXT NOT NULL,
            branch_params_json  TEXT NOT NULL,
            delta_json          TEXT NOT NULL,
            status              TEXT NOT NULL DEFAULT 'planned',
            result_summary_json TEXT,
            created_at          TEXT NOT NULL,
            started_at          TEXT,
            finished_at         TEXT,
            UNIQUE(campaign_id, parent_run_id, parent_checkpoint_id, branch_reason, delta_json)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run_branches_campaign ON run_branches(campaign_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run_branches_parent ON run_branches(parent_run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run_branches_child ON run_branches(child_run_id)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendation_batches (
            id                  TEXT PRIMARY KEY,
            campaign_id         TEXT NOT NULL REFERENCES campaigns(id),
            selector_name       TEXT NOT NULL,
            selector_version    TEXT NOT NULL,
            selector_hash       TEXT NOT NULL,
            frontier_snapshot_id TEXT,
            acquisition_name    TEXT,
            acquisition_version TEXT,
            surrogate_snapshot_id TEXT,
            created_at          TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rec_batches_campaign ON recommendation_batches(campaign_id)")
    for col, typ in [
        ("acquisition_name", "TEXT"),
        ("acquisition_version", "TEXT"),
        ("surrogate_snapshot_id", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE recommendation_batches ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendations (
            id                  TEXT PRIMARY KEY,
            batch_id            TEXT NOT NULL REFERENCES recommendation_batches(id),
            candidate_type      TEXT NOT NULL,
            candidate_key       TEXT,
            rank                INTEGER NOT NULL,
            score_total         REAL NOT NULL,
            score_breakdown_json TEXT NOT NULL,
            rationale_json      TEXT NOT NULL,
            axis_values_json    TEXT,
            branch_reason       TEXT,
            delta_json          TEXT,
            selector_score_total REAL,
            acquisition_score  REAL,
            parent_run_id      TEXT REFERENCES runs(id),
            parent_checkpoint_id INTEGER REFERENCES checkpoints(id),
            status              TEXT NOT NULL DEFAULT 'planned',
            created_at          TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_batch ON recommendations(batch_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_campaign ON recommendations(batch_id, candidate_type)")
    for col, typ in [
        ("selector_score_total", "REAL"),
        ("acquisition_score", "REAL"),
        ("parent_run_id", "TEXT"),
        ("parent_checkpoint_id", "INTEGER"),
    ]:
        try:
            conn.execute(f"ALTER TABLE recommendations ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    for col, typ in [("candidate_payload_json", "TEXT"), ("objective_metrics_json", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE recommendations ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendation_outcomes (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            recommendation_id   TEXT NOT NULL REFERENCES recommendations(id),
            run_id              TEXT REFERENCES runs(id),
            branch_id           TEXT REFERENCES run_branches(id),
            observed_metrics_json TEXT NOT NULL,
            frontier_delta_json TEXT,
            outcome_label       TEXT NOT NULL,
            evaluated_at        TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rec_outcomes_rec ON recommendation_outcomes(recommendation_id)")
    for col, typ in [("constraint_status_json", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE recommendation_outcomes ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS surrogate_snapshots (
            id                   TEXT PRIMARY KEY,
            campaign_id          TEXT NOT NULL REFERENCES campaigns(id),
            frontier_snapshot_id TEXT REFERENCES frontier_snapshots(id),
            acquisition_name     TEXT NOT NULL,
            acquisition_version  TEXT NOT NULL,
            policy_hash          TEXT NOT NULL,
            objectives_json      TEXT NOT NULL,
            feature_schema_json  TEXT NOT NULL,
            summary_json         TEXT NOT NULL,
            candidate_count      INTEGER NOT NULL,
            created_at           TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_surrogate_snapshots_campaign ON surrogate_snapshots(campaign_id)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL REFERENCES runs(id),
            metric_name     TEXT NOT NULL,
            metric_value    REAL NOT NULL,
            metric_unit     TEXT,
            metric_role     TEXT NOT NULL,
            direction       TEXT NOT NULL,
            source          TEXT,
            created_at      TEXT NOT NULL,
            UNIQUE(run_id, metric_name)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run_metrics_run ON run_metrics(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run_metrics_name ON run_metrics(metric_name)")
    for col, typ in [
        ("objective_profile_id", "TEXT"),
        ("metric_source", "TEXT"),
        ("constraints_json", "TEXT"),
        ("knee_run_id", "TEXT"),
        ("knee_rationale_json", "TEXT"),
        ("artifact_dir", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE frontier_snapshots ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    for col, typ in [("artifact_dir", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS quote_windows (
            id                         TEXT PRIMARY KEY,
            campaign_id                TEXT NOT NULL REFERENCES campaigns(id),
            anchor_currency            TEXT NOT NULL,
            started_at                 TEXT NOT NULL,
            expires_at                 TEXT NOT NULL,
            max_quote_age_seconds      INTEGER NOT NULL,
            portfolio_snapshot_json    TEXT NOT NULL,
            liquidity_floor_json       TEXT NOT NULL,
            provider_config_json       TEXT NOT NULL,
            status                     TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quote_windows_campaign ON quote_windows(campaign_id)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fx_quotes (
            id                  TEXT PRIMARY KEY,
            quote_window_id     TEXT NOT NULL REFERENCES quote_windows(id),
            provider            TEXT NOT NULL,
            environment         TEXT NOT NULL,
            quote_source        TEXT NOT NULL,
            sell_currency       TEXT NOT NULL,
            buy_currency        TEXT NOT NULL,
            sell_amount         REAL,
            buy_amount          REAL,
            client_rate         REAL,
            mid_rate            REAL,
            awx_rate            REAL,
            quote_id            TEXT,
            valid_from_at       TEXT,
            valid_to_at         TEXT,
            conversion_date     TEXT,
            quote_latency_ms    REAL,
            raw_json            TEXT NOT NULL,
            created_at          TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fx_quotes_window ON fx_quotes(quote_window_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fx_quotes_pair ON fx_quotes(sell_currency, buy_currency)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fx_route_legs (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id                      TEXT NOT NULL REFERENCES runs(id),
            leg_index                   INTEGER NOT NULL,
            sell_currency               TEXT NOT NULL,
            buy_currency                TEXT NOT NULL,
            sell_amount                 REAL,
            buy_amount                  REAL,
            quote_ref                   TEXT REFERENCES fx_quotes(id),
            route_state_before_json     TEXT NOT NULL,
            route_state_after_json      TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fx_route_legs_run ON fx_route_legs(run_id)")
    conn.commit()
