"""
Experiment tracker for MAG-Gomoku — SQLite-based.

Manages: runs, cycle_metrics, checkpoints, recordings.
All paths stored as relative to project root (e.g. "output/checkpoints/xxx.safetensors").
"""

import os
import platform
import sqlite3
import subprocess
from datetime import datetime, timezone
from typing import Optional

DB_PATH = "output/tracker.db"

# Win-rate thresholds for checkpoint export
# <80%: every 5%, 80-90%: every 2%, >90%: every 1%
CHECKPOINT_THRESHOLDS = [
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.82, 0.84, 0.86, 0.88,
    0.90, 0.91, 0.92, 0.93, 0.94, 0.95,
    0.96, 0.97, 0.98, 0.99, 1.00,
]

_SCHEMA_SQL = """
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
    tag              TEXT NOT NULL UNIQUE,
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
    eval_elapsed_s   REAL
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
"""


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create tables if they don't exist and return a connection."""
    conn = _connect(db_path)
    conn.executescript(_SCHEMA_SQL)
    # v3 migration: add columns for UUID output dirs and resume support
    for col, typ in [("resumed_from", "TEXT"), ("output_dir", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

def collect_hardware_info() -> dict:
    """Gather chip, cores, memory, MLX version from the current machine."""
    info: dict = {}
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        info["chip"] = out
    except Exception:
        info["chip"] = platform.processor() or "unknown"

    try:
        import plistlib
        raw = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "-xml"],
            stderr=subprocess.DEVNULL,
        )
        hw = plistlib.loads(raw)[0]["_items"][0]
        info["chip"] = hw.get("chip_type", info.get("chip", "unknown"))
        # number_processors is e.g. "proc 16:12:4" → total is 16
        proc_str = hw.get("number_processors", "0")
        if isinstance(proc_str, str) and ":" in proc_str:
            info["cpu_cores"] = int(proc_str.split()[1].split(":")[0])
        elif isinstance(proc_str, int):
            info["cpu_cores"] = proc_str
        else:
            info["cpu_cores"] = int(proc_str) if proc_str.isdigit() else 0
        # Memory comes as string like "128 GB"
        mem_str = hw.get("physical_memory", "0 GB")
        info["memory_gb"] = int(mem_str.split()[0]) if isinstance(mem_str, str) else 0
        # GPU cores: try Metal device query, fall back to known chip specs
        gpu_str = hw.get("platform_number_gpu_cores", "")
        if gpu_str:
            info["gpu_cores"] = int(gpu_str)
        else:
            # Infer from chip name
            chip = info.get("chip", "")
            if "M3 Max" in chip:
                info["gpu_cores"] = 40
            elif "M3 Pro" in chip:
                info["gpu_cores"] = 18
            elif "M3 Ultra" in chip:
                info["gpu_cores"] = 80
            else:
                info["gpu_cores"] = 0
    except Exception:
        info.setdefault("cpu_cores", os.cpu_count() or 0)
        info.setdefault("gpu_cores", 0)
        info.setdefault("memory_gb", 0)

    try:
        import mlx.core as mx
        info["mlx_version"] = mx.__version__
    except ImportError:
        info["mlx_version"] = "unknown"

    return info


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------

def create_run(conn: sqlite3.Connection, run_id: str,
               hyperparams: dict, hardware: Optional[dict] = None,
               resumed_from: Optional[str] = None,
               output_dir: Optional[str] = None) -> None:
    """Insert a new training run."""
    hw = hardware or {}
    conn.execute(
        """INSERT INTO runs (
            id, started_at, status,
            chip, cpu_cores, gpu_cores, memory_gb, mlx_version,
            num_res_blocks, num_filters, learning_rate, weight_decay,
            batch_size, parallel_games, mcts_simulations, temperature,
            temp_threshold, replay_buffer_size, train_steps_per_cycle,
            time_budget, target_win_rate, target_games, eval_level,
            resumed_from, output_dir
        ) VALUES (
            ?, ?, 'running',
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?
        )""",
        (
            run_id,
            datetime.now(timezone.utc).isoformat(),
            hw.get("chip"), hw.get("cpu_cores"), hw.get("gpu_cores"),
            hw.get("memory_gb"), hw.get("mlx_version"),
            hyperparams.get("num_res_blocks"),
            hyperparams.get("num_filters"),
            hyperparams.get("learning_rate"),
            hyperparams.get("weight_decay"),
            hyperparams.get("batch_size"),
            hyperparams.get("parallel_games"),
            hyperparams.get("mcts_simulations"),
            hyperparams.get("temperature"),
            hyperparams.get("temp_threshold"),
            hyperparams.get("replay_buffer_size"),
            hyperparams.get("train_steps_per_cycle"),
            hyperparams.get("time_budget"),
            hyperparams.get("target_win_rate"),
            hyperparams.get("target_games"),
            hyperparams.get("eval_level"),
            resumed_from,
            output_dir,
        ),
    )
    conn.commit()


def finish_run(conn: sqlite3.Connection, run_id: str, summary: dict) -> None:
    """Mark a run as completed and fill in summary fields."""
    conn.execute(
        """UPDATE runs SET
            finished_at = ?, status = ?,
            total_cycles = ?, total_games = ?, total_steps = ?,
            final_loss = ?, final_win_rate = ?,
            num_params = ?, num_checkpoints = ?,
            wall_time_s = ?, peak_memory_mb = ?
        WHERE id = ?""",
        (
            datetime.now(timezone.utc).isoformat(),
            summary.get("status", "completed"),
            summary.get("total_cycles"),
            summary.get("total_games"),
            summary.get("total_steps"),
            summary.get("final_loss"),
            summary.get("final_win_rate"),
            summary.get("num_params"),
            summary.get("num_checkpoints"),
            summary.get("wall_time_s"),
            summary.get("peak_memory_mb"),
            run_id,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Cycle metrics
# ---------------------------------------------------------------------------

def save_cycle_metric(conn: sqlite3.Connection, run_id: str,
                      metric: dict) -> None:
    conn.execute(
        """INSERT INTO cycle_metrics (
            run_id, cycle, timestamp_s, loss,
            total_games, total_steps, buffer_size,
            win_rate, eval_type, eval_games, eval_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            metric["cycle"],
            metric["timestamp_s"],
            metric.get("loss"),
            metric.get("total_games"),
            metric.get("total_steps"),
            metric.get("buffer_size"),
            metric.get("win_rate"),
            metric.get("eval_type"),
            metric.get("eval_games"),
            metric.get("eval_level"),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(conn: sqlite3.Connection, run_id: str,
                    data: dict) -> int:
    """Insert a checkpoint record. Returns the checkpoint id."""
    cur = conn.execute(
        """INSERT INTO checkpoints (
            run_id, tag, cycle, step, loss,
            win_rate, eval_level, eval_games,
            wins, losses, draws, avg_game_length,
            num_params, model_path, model_size_bytes,
            created_at, train_elapsed_s, eval_elapsed_s
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            data["tag"],
            data["cycle"],
            data["step"],
            data["loss"],
            data["win_rate"],
            data["eval_level"],
            data["eval_games"],
            data.get("wins"),
            data.get("losses"),
            data.get("draws"),
            data.get("avg_game_length"),
            data.get("num_params"),
            data["model_path"],
            data.get("model_size_bytes"),
            datetime.now(timezone.utc).isoformat(),
            data.get("train_elapsed_s"),
            data.get("eval_elapsed_s"),
        ),
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------

def save_recording(conn: sqlite3.Connection, checkpoint_id: int,
                   run_id: str, data: dict) -> None:
    conn.execute(
        """INSERT INTO recordings (
            checkpoint_id, run_id, game_index, game_file,
            result, total_moves, black, white, nn_side, nn_won
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            checkpoint_id,
            run_id,
            data["game_index"],
            data["game_file"],
            data["result"],
            data["total_moves"],
            data["black"],
            data["white"],
            data["nn_side"],
            data.get("nn_won", 0),
        ),
    )
    # Caller should commit in batch for performance


def save_recordings_batch(conn: sqlite3.Connection, checkpoint_id: int,
                          run_id: str, records: list[dict]) -> None:
    """Bulk insert recordings and commit once."""
    for rec in records:
        save_recording(conn, checkpoint_id, run_id, rec)
    conn.commit()


# ---------------------------------------------------------------------------
# Threshold logic
# ---------------------------------------------------------------------------

def should_checkpoint(current_wr: float, last_ckpt_wr: float) -> bool:
    """Return True if current_wr crosses the next threshold above last_ckpt_wr."""
    for t in CHECKPOINT_THRESHOLDS:
        if last_ckpt_wr < t <= current_wr:
            return True
    return False


def next_threshold(last_ckpt_wr: float) -> Optional[float]:
    """Return the next threshold above last_ckpt_wr, or None."""
    for t in CHECKPOINT_THRESHOLDS:
        if t > last_ckpt_wr:
            return t
    return None


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_run(conn: sqlite3.Connection, run_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_checkpoints(conn: sqlite3.Connection,
                    run_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY cycle",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_recordings(conn: sqlite3.Connection,
                   checkpoint_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM recordings WHERE checkpoint_id = ? ORDER BY game_index",
        (checkpoint_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_cycle_metrics(conn: sqlite3.Connection,
                      run_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM cycle_metrics WHERE run_id = ? ORDER BY cycle",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def count_checkpoints(conn: sqlite3.Connection, run_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE run_id = ?", (run_id,),
    ).fetchone()
    return row[0] if row else 0


def get_latest_checkpoint(conn: sqlite3.Connection,
                          run_id: str) -> Optional[dict]:
    """Return the most recent checkpoint for a run, or None."""
    row = conn.execute(
        "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY cycle DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    return dict(row) if row else None


def list_all_checkpoints(conn: sqlite3.Connection,
                         limit: int = 50) -> list[dict]:
    """List checkpoints across all runs, newest first."""
    rows = conn.execute(
        """SELECT c.*, r.chip, r.output_dir
           FROM checkpoints c
           JOIN runs r ON c.run_id = r.id
           ORDER BY c.created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def find_checkpoint_by_tag(conn: sqlite3.Connection,
                           tag: str) -> Optional[dict]:
    """Find a checkpoint by exact or partial tag match."""
    row = conn.execute(
        "SELECT * FROM checkpoints WHERE tag = ?", (tag,),
    ).fetchone()
    if row:
        return dict(row)
    # Partial match
    row = conn.execute(
        "SELECT * FROM checkpoints WHERE tag LIKE ? ORDER BY created_at DESC LIMIT 1",
        (f"%{tag}%",),
    ).fetchone()
    return dict(row) if row else None
