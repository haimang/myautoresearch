"""Base schema and checkpoint rules for the db package."""

from __future__ import annotations


CHECKPOINT_THRESHOLDS = [0.85, 0.90, 0.95, 1.00]

PROMOTION_WR_THRESHOLD_BY_LEVEL = {
    0: 0.95,
    1: 0.80,
    2: 0.60,
    3: 0.50,
}


def can_promote(ckpt: dict, recent_smoothed_wr: list[float]) -> tuple[bool, str]:
    """Return whether a checkpoint is eligible for opponent promotion."""
    level = ckpt.get("eval_level")
    if level is None:
        return False, "missing eval_level"

    threshold = PROMOTION_WR_THRESHOLD_BY_LEVEL.get(level, 0.80)
    wr = ckpt.get("win_rate")
    if wr is None or wr < threshold:
        return False, f"WR {wr:.2%} < {threshold:.0%} threshold for L{level}"

    uniq = ckpt.get("eval_unique_openings")
    if uniq is None or uniq < 16:
        return False, f"unique_openings {uniq} < 16 (stats collapse risk)"

    avg_len = ckpt.get("avg_game_length")
    if avg_len is None:
        return False, "missing avg_game_length"
    if avg_len < 12.0 or avg_len > 60.0:
        return False, f"avg_len {avg_len:.1f} outside sane range [12, 60]"

    if recent_smoothed_wr and len(recent_smoothed_wr) >= 3:
        recent = recent_smoothed_wr[-5:]
        spread = max(recent) - min(recent)
        if spread > 0.15:
            return False, f"recent smoothed WR unstable (range {spread:.0%} > 15%)"

    return True, "OK"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id               TEXT PRIMARY KEY,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    status           TEXT DEFAULT 'running',

    -- hardware
    chip             TEXT,
    cpu_cores        INTEGER,
    gpu_cores        INTEGER,
    memory_gb        INTEGER,
    mlx_version      TEXT,

    -- hyperparameters snapshot
    num_res_blocks      INTEGER,
    num_filters         INTEGER,
    learning_rate       REAL,
    weight_decay        REAL,
    batch_size          INTEGER,
    parallel_games      INTEGER,
    mcts_simulations    INTEGER,
    temperature         REAL,
    temp_threshold      INTEGER,
    replay_buffer_size  INTEGER,
    train_steps_per_cycle INTEGER,
    time_budget         INTEGER,
    target_win_rate     REAL,
    target_games        INTEGER,
    eval_level          INTEGER,

    -- final summary (filled on completion)
    total_cycles     INTEGER,
    total_games      INTEGER,
    total_steps      INTEGER,
    final_loss       REAL,
    final_win_rate   REAL,
    num_params       INTEGER,
    num_checkpoints  INTEGER,
    wall_time_s      REAL,
    peak_memory_mb   REAL
);

CREATE TABLE IF NOT EXISTS cycle_metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES runs(id),
    cycle         INTEGER NOT NULL,
    timestamp_s   REAL NOT NULL,
    loss          REAL,
    total_games   INTEGER,
    total_steps   INTEGER,
    buffer_size   INTEGER,
    win_rate      REAL,
    eval_type     TEXT,
    eval_games    INTEGER,
    eval_level    INTEGER
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL REFERENCES runs(id),
    tag              TEXT NOT NULL,
    cycle            INTEGER NOT NULL,
    step             INTEGER NOT NULL,
    loss             REAL NOT NULL,

    win_rate         REAL NOT NULL,
    eval_level       INTEGER NOT NULL,
    eval_games       INTEGER NOT NULL,
    wins             INTEGER,
    losses           INTEGER,
    draws            INTEGER,
    avg_game_length  REAL,

    num_params       INTEGER,
    model_path       TEXT NOT NULL,
    model_size_bytes INTEGER,

    created_at       TEXT NOT NULL,
    train_elapsed_s  REAL,
    eval_elapsed_s   REAL,
    UNIQUE(run_id, tag)
);

CREATE TABLE IF NOT EXISTS recordings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id    INTEGER NOT NULL REFERENCES checkpoints(id),
    run_id           TEXT NOT NULL REFERENCES runs(id),
    game_index       INTEGER NOT NULL,
    game_file        TEXT NOT NULL,
    result           TEXT NOT NULL,
    total_moves      INTEGER NOT NULL,
    black            TEXT NOT NULL,
    white            TEXT NOT NULL,
    nn_side          TEXT NOT NULL,
    nn_won           INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cycle_metrics_run ON cycle_metrics(run_id, cycle);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id);
CREATE INDEX IF NOT EXISTS idx_recordings_checkpoint ON recordings(checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_recordings_run ON recordings(run_id);

CREATE TABLE IF NOT EXISTS opponents (
    alias        TEXT PRIMARY KEY,
    source_run   TEXT,
    source_tag   TEXT,
    model_path   TEXT NOT NULL,
    win_rate     REAL,
    eval_level   INTEGER,
    description  TEXT,
    created_at   TEXT NOT NULL
);
"""
