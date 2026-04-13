"""
autoresearch 数据库层 — 基于 SQLite。

框架的单一数据源：schema 定义、连接管理、迁移、全部 CRUD 操作。
管理: runs（训练运行）、cycle_metrics（周期指标）、checkpoints（检查点）、
recordings（对局记录）、opponents（注册对手）。

所有路径以项目根目录为基准存储（如 "output/<uuid>/checkpoints/xxx.safetensors"）。
"""

import os
import platform
import sqlite3
import subprocess
from datetime import datetime, timezone
from typing import Optional

DB_PATH = "output/tracker.db"

# v15 A2: 精简到 4 档 + Ctrl+C final 快照。
#
# 过去 22 个阈值对于 mcts_10/11 这种 3h 规模的 run 会产生 6-8 个
# 中间 checkpoint，绝大部分都只是"刚越过阈值就存"的冗余记录。v15 起只在
# 真正有晋升价值的四个点保存 checkpoint —— 其余的进度由 cycle_metrics
# 的 probe 点体现，不需要复制一份 weights 文件。Ctrl+C 时的 final 快照
# 仍然由 train.py 的 try/except KeyboardInterrupt 路径保证。
CHECKPOINT_THRESHOLDS = [0.85, 0.90, 0.95, 1.00]

# v15 E1: per-level WR thresholds for promotion eligibility.
# Tighter for weaker opponents, looser for stronger ones.
PROMOTION_WR_THRESHOLD_BY_LEVEL = {
    0: 0.95,   # vs random — should be near-perfect to count as S0
    1: 0.80,   # vs minimax depth 2
    2: 0.60,   # vs minimax depth 4
    3: 0.50,   # vs minimax depth 6 — "持续优化" stage
}


def can_promote(ckpt: dict, recent_smoothed_wr: list[float]) -> tuple[bool, str]:
    """v15 E1: stage-promotion gate built from mcts_9th/10/11 hard lessons.

    Returns (eligible, reason). Eligible only if ALL four checks pass:

      1. WR ≥ level-specific threshold (PROMOTION_WR_THRESHOLD_BY_LEVEL)
      2. unique_openings ≥ 16 (statistical-power gate; mcts_9th had ≈2)
      3. avg_game_length ∈ [12, 60] (no speed-replay collapse)
      4. recent smoothed WR range ≤ 0.15 over last 5 probes (stability)

    If any check fails, returns (False, descriptive_reason).
    """
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


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    """建立数据库连接，自动创建目录。"""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """初始化数据库（建表 + 迁移），返回连接。"""
    conn = _connect(db_path)
    conn.executescript(_SCHEMA_SQL)
    # v3 迁移: UUID 输出目录 + 续训支持
    for col, typ in [("resumed_from", "TEXT"), ("output_dir", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass  # 列已存在
    # v6 迁移: benchmark 标记 + eval_opponent
    for col, typ in [("is_benchmark", "INTEGER DEFAULT 0"), ("eval_opponent", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    # v8 迁移: 对手架构元数据
    for col, typ in [("num_res_blocks", "INTEGER"), ("num_filters", "INTEGER")]:
        try:
            conn.execute(f"ALTER TABLE opponents ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    # v9 迁移: sweep 标签和种子
    for col, typ in [("sweep_tag", "TEXT"), ("seed", "INTEGER")]:
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    # v15 迁移: 拆分 policy / value loss（总 loss 信号不够诊断，
    # 有时只有策略在学或只有价值在学，单一 loss 数字会遮蔽这个现象）
    for col, typ in [("policy_loss", "REAL"), ("value_loss", "REAL")]:
        try:
            conn.execute(f"ALTER TABLE cycle_metrics ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    # v15 迁移: 评估时实际出现的不同开局数 —
    # 确定性评估会让 "N games" 坍缩成 "2 games"，这一列让该坍缩可观测。
    for col, typ in [("eval_unique_openings", "INTEGER")]:
        try:
            conn.execute(f"ALTER TABLE checkpoints ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    # v15 E2: promotion gate result — written by save_checkpoint via can_promote()
    for col, typ in [("promotion_eligible", "INTEGER"),
                     ("promotion_reason", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE checkpoints ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    # v15 E3: eval_breakdown table — per-opening WR for diagnosing model blind spots
    conn.execute("""
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
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_breakdown_ckpt "
                 "ON eval_breakdown(checkpoint_id)")
    # v15 E5: opponent promotion chain (S0 → S1 → S2 → ...)
    for col, typ in [("prev_alias", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE opponents ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    # v15 B7: async eval — record submit-time cycle so analyze.py can show latency
    for col, typ in [("eval_submitted_cycle", "INTEGER")]:
        try:
            conn.execute(f"ALTER TABLE cycle_metrics ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# 硬件信息
# ---------------------------------------------------------------------------

def collect_hardware_info() -> dict:
    """收集当前机器的芯片、核心数、内存、MLX 版本。"""
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
# 运行记录 CRUD
# ---------------------------------------------------------------------------

def create_run(conn: sqlite3.Connection, run_id: str,
               hyperparams: dict, hardware: Optional[dict] = None,
               resumed_from: Optional[str] = None,
               output_dir: Optional[str] = None,
               is_benchmark: bool = False,
               eval_opponent: Optional[str] = None) -> None:
    """插入新的训练运行记录。"""
    hw = hardware or {}
    conn.execute(
        """INSERT INTO runs (
            id, started_at, status,
            chip, cpu_cores, gpu_cores, memory_gb, mlx_version,
            num_res_blocks, num_filters, learning_rate, weight_decay,
            batch_size, parallel_games, mcts_simulations, temperature,
            temp_threshold, replay_buffer_size, train_steps_per_cycle,
            time_budget, target_win_rate, target_games, eval_level,
            resumed_from, output_dir, is_benchmark, eval_opponent,
            sweep_tag, seed
        ) VALUES (
            ?, ?, 'running',
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
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
            1 if is_benchmark else 0,
            eval_opponent,
            hyperparams.get("sweep_tag"),
            hyperparams.get("seed"),
        ),
    )
    conn.commit()


def finish_run(conn: sqlite3.Connection, run_id: str, summary: dict) -> None:
    """标记运行完成，填充汇总字段。"""
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
# 周期指标
# ---------------------------------------------------------------------------

def save_cycle_metric(conn: sqlite3.Connection, run_id: str,
                      metric: dict) -> None:
    """v15.2: now also persists eval_submitted_cycle (the bug v15 introduced
    by adding the column to the schema migration without updating INSERT)."""
    conn.execute(
        """INSERT INTO cycle_metrics (
            run_id, cycle, timestamp_s, loss,
            total_games, total_steps, buffer_size,
            win_rate, eval_type, eval_games, eval_level,
            policy_loss, value_loss, eval_submitted_cycle
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            metric.get("policy_loss"),
            metric.get("value_loss"),
            metric.get("eval_submitted_cycle"),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# 检查点
# ---------------------------------------------------------------------------

def save_checkpoint(conn: sqlite3.Connection, run_id: str,
                    data: dict) -> int:
    """插入检查点记录，返回 checkpoint id。

    v15 E2: also computes promotion_eligible via can_promote() if the caller
    provides `recent_smoothed_wr` in `data`. Otherwise leaves it NULL.
    """
    eligible_int = None
    eligible_reason = None
    if "recent_smoothed_wr" in data:
        ok, reason = can_promote(data, data.get("recent_smoothed_wr") or [])
        eligible_int = 1 if ok else 0
        eligible_reason = reason

    cur = conn.execute(
        """INSERT INTO checkpoints (
            run_id, tag, cycle, step, loss,
            win_rate, eval_level, eval_games,
            wins, losses, draws, avg_game_length,
            num_params, model_path, model_size_bytes,
            created_at, train_elapsed_s, eval_elapsed_s,
            eval_unique_openings, promotion_eligible, promotion_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            data.get("eval_unique_openings"),
            eligible_int,
            eligible_reason,
        ),
    )
    conn.commit()
    return cur.lastrowid


def save_eval_breakdown(conn: sqlite3.Connection, checkpoint_id: int,
                        breakdown: list[dict]) -> None:
    """v15 E3: persist per-opening WR breakdown for a checkpoint."""
    for b in breakdown:
        conn.execute(
            """INSERT INTO eval_breakdown (
                checkpoint_id, opening_index, opening_moves,
                wins, losses, draws, avg_length, unique_games
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                checkpoint_id,
                b["opening_index"],
                b.get("opening_moves"),
                b["wins"],
                b["losses"],
                b["draws"],
                b.get("avg_length"),
                b.get("unique_games"),
            ),
        )
    conn.commit()


def get_eval_breakdown(conn: sqlite3.Connection, checkpoint_id: int) -> list[dict]:
    cur = conn.execute(
        "SELECT * FROM eval_breakdown WHERE checkpoint_id = ? ORDER BY opening_index",
        (checkpoint_id,),
    )
    return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# 对局记录
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
    """批量插入对局记录并提交。"""
    for rec in records:
        save_recording(conn, checkpoint_id, run_id, rec)
    conn.commit()


# ---------------------------------------------------------------------------
# 阈值逻辑
# ---------------------------------------------------------------------------

def should_checkpoint(current_wr: float, last_ckpt_wr: float) -> bool:
    """如果 current_wr 越过 last_ckpt_wr 之上的下一个阈值，返回 True。"""
    for t in CHECKPOINT_THRESHOLDS:
        if last_ckpt_wr < t <= current_wr:
            return True
    return False


def crossed_threshold(current_wr: float, last_ckpt_wr: float) -> Optional[float]:
    """返回 last_ckpt_wr 到 current_wr 之间越过的最高阈值，无则返回 None。"""
    crossed = None
    for t in CHECKPOINT_THRESHOLDS:
        if last_ckpt_wr < t <= current_wr:
            crossed = t
    return crossed


def next_threshold(last_ckpt_wr: float) -> Optional[float]:
    """返回 last_ckpt_wr 之上的下一个阈值，无则返回 None。"""
    for t in CHECKPOINT_THRESHOLDS:
        if t > last_ckpt_wr:
            return t
    return None


# ---------------------------------------------------------------------------
# 查询辅助
# ---------------------------------------------------------------------------

def get_run(conn: sqlite3.Connection, run_id: str) -> Optional[dict]:
    """通过完整 ID 或短前缀查找运行记录。"""
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row:
        return dict(row)
    # 前缀匹配
    rows = conn.execute(
        "SELECT * FROM runs WHERE id LIKE ?", (run_id + "%",)
    ).fetchall()
    if len(rows) == 1:
        return dict(rows[0])
    return None


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
    """返回某运行的最新检查点，无则返回 None。"""
    row = conn.execute(
        "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY cycle DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    return dict(row) if row else None


def list_all_checkpoints(conn: sqlite3.Connection,
                         limit: int = 50) -> list[dict]:
    """列出所有运行的检查点，最新优先。"""
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
    """通过完整或部分标签匹配查找检查点。"""
    row = conn.execute(
        "SELECT * FROM checkpoints WHERE tag = ?", (tag,),
    ).fetchone()
    if row:
        return dict(row)
    # 部分匹配
    row = conn.execute(
        "SELECT * FROM checkpoints WHERE tag LIKE ? ORDER BY created_at DESC LIMIT 1",
        (f"%{tag}%",),
    ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# 对手注册
# ---------------------------------------------------------------------------

def register_opponent(conn: sqlite3.Connection, alias: str,
                      model_path: str, source_run: Optional[str] = None,
                      source_tag: Optional[str] = None,
                      win_rate: Optional[float] = None,
                      eval_level: Optional[int] = None,
                      description: Optional[str] = None,
                      num_res_blocks: Optional[int] = None,
                      num_filters: Optional[int] = None,
                      prev_alias: Optional[str] = None) -> None:
    """注册一个模型检查点作为命名对手。

    v15 E5: `prev_alias` records the promotion chain (S0 → S1 → S2 → …).
    """
    conn.execute(
        """INSERT OR REPLACE INTO opponents (
            alias, source_run, source_tag, model_path,
            win_rate, eval_level, description, created_at,
            num_res_blocks, num_filters, prev_alias
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (alias, source_run, source_tag, model_path,
         win_rate, eval_level, description,
         datetime.now(timezone.utc).isoformat(),
         num_res_blocks, num_filters, prev_alias),
    )
    conn.commit()


def get_opponent(conn: sqlite3.Connection, alias: str) -> Optional[dict]:
    """按别名查找注册对手。"""
    row = conn.execute(
        "SELECT * FROM opponents WHERE alias = ?", (alias,),
    ).fetchone()
    return dict(row) if row else None


def list_opponents(conn: sqlite3.Connection) -> list[dict]:
    """列出所有注册对手。"""
    rows = conn.execute(
        "SELECT * FROM opponents ORDER BY created_at",
    ).fetchall()
    return [dict(r) for r in rows]
