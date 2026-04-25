"""
autoresearch 数据库层 — 基于 SQLite。

框架的单一数据源：schema 定义、连接管理、迁移、全部 CRUD 操作。
管理: runs（训练运行）、cycle_metrics（周期指标）、checkpoints（检查点）、
recordings（对局记录）、opponents（注册对手）。

所有路径以项目根目录为基准存储（如 "output/<uuid>/checkpoints/xxx.safetensors"）。
"""

import json as _json
import os
import platform
import sqlite3
import subprocess
import uuid
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
    # v20: frontier_snapshots — Pareto frontier version history
    conn.execute("""
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
    """)
    for col, typ in [("campaign_id", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE frontier_snapshots ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    # v20.1: campaign + search-space governance
    conn.execute("""
        CREATE TABLE IF NOT EXISTS search_spaces (
            id            TEXT PRIMARY KEY,
            created_at    TEXT NOT NULL,
            domain        TEXT NOT NULL,
            name          TEXT NOT NULL,
            version       TEXT NOT NULL,
            profile_json  TEXT NOT NULL,
            profile_hash  TEXT NOT NULL UNIQUE
        )
    """)
    conn.execute("""
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
    """)
    conn.execute("""
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
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_campaign_runs_campaign "
                 "ON campaign_runs(campaign_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_campaign_runs_run "
                 "ON campaign_runs(run_id)")

    # v20.2: multi-fidelity promotion engine
    conn.execute("""
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
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_campaign_stages_campaign "
                 "ON campaign_stages(campaign_id)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_campaign_stages_unique "
                 "ON campaign_stages(campaign_id, stage)")

    conn.execute("""
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
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_promotion_decisions_campaign "
                 "ON promotion_decisions(campaign_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_promotion_decisions_stage "
                 "ON promotion_decisions(from_stage, to_stage)")

    # v20.2: candidate_key on campaign_runs for aggregation
    for col, typ in [("candidate_key", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE campaign_runs ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass

    # v20.3: continuation / trajectory explorer
    conn.execute("""
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
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run_branches_campaign "
                 "ON run_branches(campaign_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run_branches_parent "
                 "ON run_branches(parent_run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_run_branches_child "
                 "ON run_branches(child_run_id)")

    # v21: recommendation ledger
    conn.execute("""
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
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rec_batches_campaign "
                 "ON recommendation_batches(campaign_id)")
    for col, typ in [
        ("acquisition_name", "TEXT"),
        ("acquisition_version", "TEXT"),
        ("surrogate_snapshot_id", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE recommendation_batches ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass

    conn.execute("""
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
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_batch "
                 "ON recommendations(batch_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_campaign "
                 "ON recommendations(batch_id, candidate_type)")
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

    conn.execute("""
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
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rec_outcomes_rec "
                 "ON recommendation_outcomes(recommendation_id)")

    # v21.1: surrogate / acquisition evidence lineage
    conn.execute("""
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
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_surrogate_snapshots_campaign "
                 "ON surrogate_snapshots(campaign_id)")

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


def _stable_json(value: object) -> str:
    return _json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def get_search_space(conn: sqlite3.Connection, space_id: str) -> Optional[sqlite3.Row]:
    """Return a search space by id, or None."""
    row = conn.execute(
        "SELECT * FROM search_spaces WHERE id = ?",
        (space_id,),
    ).fetchone()
    return row


def save_search_space(conn: sqlite3.Connection, profile: dict) -> str:
    """Persist a normalized search-space profile; return its id."""
    profile_json = _stable_json(profile)
    profile_hash = profile.get("profile_hash")
    if not profile_hash:
        raise ValueError("profile missing profile_hash")

    row = conn.execute(
        "SELECT id FROM search_spaces WHERE profile_hash = ?",
        (profile_hash,),
    ).fetchone()
    if row:
        return row["id"]

    space_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO search_spaces
           (id, created_at, domain, name, version, profile_json, profile_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            space_id,
            datetime.now(timezone.utc).isoformat(),
            profile["domain"],
            profile["name"],
            profile["version"],
            profile_json,
            profile_hash,
        ),
    )
    conn.commit()
    return space_id


def get_campaign(conn: sqlite3.Connection, campaign: str) -> Optional[sqlite3.Row]:
    """Resolve a campaign by exact id or exact name."""
    row = conn.execute(
        "SELECT * FROM campaigns WHERE id = ? OR name = ? ORDER BY created_at DESC LIMIT 1",
        (campaign, campaign),
    ).fetchone()
    return row


def get_or_create_campaign(conn: sqlite3.Connection, *,
                           name: str,
                           domain: str,
                           train_script: str,
                           search_space_id: str,
                           protocol: dict,
                           notes: Optional[str] = None) -> dict:
    """Create a campaign if missing, otherwise verify compatibility."""
    protocol_json = _stable_json(protocol)
    row = get_campaign(conn, name)
    if row:
        mismatches = []
        if row["domain"] != domain:
            mismatches.append(f"domain={row['domain']} != {domain}")
        if row["train_script"] != train_script:
            mismatches.append(f"train_script={row['train_script']} != {train_script}")
        if row["search_space_id"] != search_space_id:
            mismatches.append("search_space profile differs")
        if row["protocol_json"] != protocol_json:
            mismatches.append("protocol differs")
        if mismatches:
            raise ValueError(
                f"campaign '{name}' already exists with incompatible config: " + "; ".join(mismatches)
            )
        return dict(row)

    campaign_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO campaigns
           (id, created_at, name, domain, train_script, search_space_id,
            protocol_json, status, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)""",
        (
            campaign_id,
            datetime.now(timezone.utc).isoformat(),
            name,
            domain,
            train_script,
            search_space_id,
            protocol_json,
            notes,
        ),
    )
    conn.commit()
    row = get_campaign(conn, campaign_id)
    return dict(row)


def link_run_to_campaign(conn: sqlite3.Connection, *,
                         campaign_id: str,
                         run_id: str,
                         stage: Optional[str],
                         sweep_tag: Optional[str],
                         seed: Optional[int],
                         axis_values: dict,
                         status: str = "linked") -> None:
    """Attach a run to a campaign with structured axis metadata."""
    conn.execute(
        """INSERT INTO campaign_runs
           (campaign_id, run_id, stage, sweep_tag, seed, axis_values_json, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(campaign_id, run_id) DO UPDATE SET
               stage = excluded.stage,
               sweep_tag = excluded.sweep_tag,
               seed = excluded.seed,
               axis_values_json = excluded.axis_values_json,
               status = excluded.status""",
        (
            campaign_id,
            run_id,
            stage,
            sweep_tag,
            seed,
            _stable_json(axis_values),
            status,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def find_run_by_sweep_tag(conn: sqlite3.Connection, sweep_tag: str) -> Optional[str]:
    """Return the most recent run id for an exact sweep tag."""
    row = conn.execute(
        "SELECT id FROM runs WHERE sweep_tag = ? ORDER BY started_at DESC LIMIT 1",
        (sweep_tag,),
    ).fetchone()
    return row["id"] if row else None


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


# ---------------------------------------------------------------------------
# v20.2: candidate key + stage ledger + promotion decisions
# ---------------------------------------------------------------------------

def _candidate_key(axis_values: dict) -> str:
    """Stable candidate key from axis values (excluding seed)."""
    filtered = {k: v for k, v in axis_values.items() if k != "seed"}
    return _stable_json(filtered)


def link_run_to_campaign_v20(
    conn: sqlite3.Connection, *,
    campaign_id: str,
    run_id: str,
    stage: Optional[str],
    sweep_tag: Optional[str],
    seed: Optional[int],
    axis_values: dict,
    status: str = "linked",
) -> None:
    """Attach a run to a campaign with candidate_key generation."""
    candidate_key = _candidate_key(axis_values)
    conn.execute(
        """INSERT INTO campaign_runs
           (campaign_id, run_id, stage, sweep_tag, seed, axis_values_json, candidate_key, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(campaign_id, run_id) DO UPDATE SET
               stage = excluded.stage,
               sweep_tag = excluded.sweep_tag,
               seed = excluded.seed,
               axis_values_json = excluded.axis_values_json,
               candidate_key = excluded.candidate_key,
               status = excluded.status""",
        (
            campaign_id,
            run_id,
            stage,
            sweep_tag,
            seed,
            _stable_json(axis_values),
            candidate_key,
            status,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def save_campaign_stage(
    conn: sqlite3.Connection, *,
    campaign_id: str,
    stage: str,
    policy_json: str,
    budget_json: str,
    seed_target: int,
    status: str = "open",
) -> None:
    """Open or update a campaign stage record."""
    stage_id = f"{campaign_id}_{stage}"
    conn.execute(
        """INSERT INTO campaign_stages
           (id, campaign_id, stage, policy_json, budget_json, seed_target, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               policy_json = excluded.policy_json,
               budget_json = excluded.budget_json,
               seed_target = excluded.seed_target,
               status = excluded.status""",
        (
            stage_id,
            campaign_id,
            stage,
            policy_json,
            budget_json,
            seed_target,
            status,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def close_campaign_stage(conn: sqlite3.Connection,
                         campaign_id: str, stage: str) -> None:
    """Mark a campaign stage as closed."""
    stage_id = f"{campaign_id}_{stage}"
    conn.execute(
        """UPDATE campaign_stages SET status = 'closed', closed_at = ?
           WHERE id = ?""",
        (datetime.now(timezone.utc).isoformat(), stage_id),
    )
    conn.commit()


def get_campaign_stages(conn: sqlite3.Connection,
                        campaign_id: str) -> list[dict]:
    """Return all stages for a campaign, ordered by stage name."""
    rows = conn.execute(
        "SELECT * FROM campaign_stages WHERE campaign_id = ? ORDER BY stage",
        (campaign_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_campaign_stage(conn: sqlite3.Connection,
                       campaign_id: str, stage: str) -> Optional[dict]:
    """Return a single campaign stage, or None."""
    row = conn.execute(
        "SELECT * FROM campaign_stages WHERE campaign_id = ? AND stage = ?",
        (campaign_id, stage),
    ).fetchone()
    return dict(row) if row else None


def save_promotion_decision(
    conn: sqlite3.Connection, *,
    campaign_id: str,
    from_stage: str,
    to_stage: str,
    candidate_key: str,
    axis_values: dict,
    aggregated_metrics: dict,
    seed_count: int,
    decision: str,
    decision_rank: Optional[int] = None,
    reason: str = "",
) -> None:
    """Persist a promotion decision (promote / reject / hold)."""
    conn.execute(
        """INSERT INTO promotion_decisions
           (campaign_id, from_stage, to_stage, candidate_key, axis_values_json,
            aggregated_metrics_json, seed_count, decision, decision_rank, reason, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(campaign_id, from_stage, to_stage, candidate_key) DO UPDATE SET
               axis_values_json = excluded.axis_values_json,
               aggregated_metrics_json = excluded.aggregated_metrics_json,
               seed_count = excluded.seed_count,
               decision = excluded.decision,
               decision_rank = excluded.decision_rank,
               reason = excluded.reason""",
        (
            campaign_id,
            from_stage,
            to_stage,
            candidate_key,
            _stable_json(axis_values),
            _stable_json(aggregated_metrics),
            seed_count,
            decision,
            decision_rank,
            reason,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def get_promotion_decisions(conn: sqlite3.Connection,
                            campaign_id: str,
                            from_stage: Optional[str] = None,
                            to_stage: Optional[str] = None) -> list[dict]:
    """Return promotion decisions for a campaign, optionally filtered by stage transition."""
    sql = "SELECT * FROM promotion_decisions WHERE campaign_id = ?"
    params: list = [campaign_id]
    if from_stage is not None:
        sql += " AND from_stage = ?"
        params.append(from_stage)
    if to_stage is not None:
        sql += " AND to_stage = ?"
        params.append(to_stage)
    sql += " ORDER BY from_stage, decision_rank, created_at"
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_campaign_runs_by_stage(conn: sqlite3.Connection,
                               campaign_id: str,
                               stage: str) -> list[dict]:
    """Return campaign_runs rows for a specific campaign + stage."""
    rows = conn.execute(
        """SELECT cr.*, r.status AS run_status, r.final_win_rate, r.wall_time_s,
                  r.total_games, r.total_steps, r.num_params, r.final_loss
           FROM campaign_runs cr
           JOIN runs r ON r.id = cr.run_id
           WHERE cr.campaign_id = ? AND cr.stage = ?""",
        (campaign_id, stage),
    ).fetchall()
    return [dict(r) for r in rows]


def aggregate_candidates_by_stage(conn: sqlite3.Connection,
                                  campaign_id: str,
                                  stage: str) -> list[dict]:
    """Aggregate campaign_runs by candidate_key for a given stage.

    Returns one row per candidate with mean/std of key metrics.
    """
    rows = conn.execute(
        """SELECT
               cr.candidate_key,
               cr.axis_values_json,
               COUNT(*) AS run_count,
               COUNT(DISTINCT cr.seed) AS seed_count,
               AVG(r.final_win_rate) AS mean_wr,
               MAX(r.final_win_rate) - MIN(r.final_win_rate) AS range_wr,
               CASE WHEN COUNT(*) > 1
                    THEN SQRT(((COUNT(*)*SUM(r.final_win_rate*r.final_win_rate) -
                               SUM(r.final_win_rate)*SUM(r.final_win_rate)) /
                               (COUNT(*)*(COUNT(*)-1))))
                    ELSE 0.0
               END AS std_wr,
               AVG(r.wall_time_s) AS mean_wall_s,
               SUM(r.total_games) AS games_total,
               AVG(r.num_params) AS mean_params,
               MIN(r.final_win_rate) AS min_wr,
               MAX(r.final_win_rate) AS max_wr
           FROM campaign_runs cr
           JOIN runs r ON r.id = cr.run_id
           WHERE cr.campaign_id = ? AND cr.stage = ?
             AND r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')
           GROUP BY cr.candidate_key
           ORDER BY mean_wr DESC""",
        (campaign_id, stage),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# v20.3: run_branches / lineage helpers
# ---------------------------------------------------------------------------

def save_run_branch(
    conn: sqlite3.Connection, *,
    branch_id: str,
    campaign_id: str,
    parent_run_id: str,
    parent_checkpoint_id: int | None,
    from_stage: str,
    branch_reason: str,
    branch_params_json: str,
    delta_json: str,
    status: str = "planned",
    result_summary_json: str = "{}",
) -> None:
    """Persist a planned branch record."""
    # Application-layer guard: SQLite treats NULL != NULL in UNIQUE constraints,
    # so duplicates can sneak through when parent_checkpoint_id IS NULL.
    if parent_checkpoint_id is None:
        existing = conn.execute(
            """SELECT id FROM run_branches
               WHERE campaign_id = ? AND parent_run_id = ?
                 AND branch_reason = ? AND delta_json = ?
                 AND parent_checkpoint_id IS NULL""",
            (campaign_id, parent_run_id, branch_reason, delta_json),
        ).fetchone()
        if existing and existing["id"] != branch_id:
            return  # Semantic duplicate, skip silent insert
    conn.execute(
        """INSERT INTO run_branches
           (id, campaign_id, parent_run_id, parent_checkpoint_id, child_run_id,
            from_stage, branch_reason, branch_params_json, delta_json,
            status, result_summary_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id)
           DO UPDATE SET
               campaign_id = excluded.campaign_id,
               parent_run_id = excluded.parent_run_id,
               parent_checkpoint_id = excluded.parent_checkpoint_id,
               child_run_id = excluded.child_run_id,
               from_stage = excluded.from_stage,
               branch_reason = excluded.branch_reason,
               branch_params_json = excluded.branch_params_json,
               delta_json = excluded.delta_json,
               status = excluded.status,
               result_summary_json = excluded.result_summary_json""",
        (
            branch_id,
            campaign_id,
            parent_run_id,
            parent_checkpoint_id,
            None,
            from_stage,
            branch_reason,
            branch_params_json,
            delta_json,
            status,
            result_summary_json,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def bind_branch_child_run(
    conn: sqlite3.Connection, *,
    branch_id: str,
    child_run_id: str,
    status: str = "running",
) -> None:
    """Link a child run to an existing branch and update status."""
    conn.execute(
        """UPDATE run_branches
           SET child_run_id = ?, status = ?, started_at = ?
           WHERE id = ?""",
        (child_run_id, status, datetime.now(timezone.utc).isoformat(), branch_id),
    )
    conn.commit()


def update_branch_status(
    conn: sqlite3.Connection, *,
    branch_id: str,
    status: str,
    result_summary_json: str | None = None,
) -> None:
    """Update branch status and optional result summary."""
    if result_summary_json is not None:
        conn.execute(
            """UPDATE run_branches
               SET status = ?, finished_at = ?, result_summary_json = ?
               WHERE id = ?""",
            (status, datetime.now(timezone.utc).isoformat(), result_summary_json, branch_id),
        )
    else:
        conn.execute(
            """UPDATE run_branches
               SET status = ?, finished_at = ?
               WHERE id = ?""",
            (status, datetime.now(timezone.utc).isoformat(), branch_id),
        )
    conn.commit()


def list_branches_for_campaign(conn: sqlite3.Connection,
                                campaign_id: str) -> list[dict]:
    """Return all branches for a campaign, ordered by created_at."""
    rows = conn.execute(
        """SELECT rb.*,
               p.sweep_tag AS parent_sweep_tag,
               c.sweep_tag AS child_sweep_tag
         FROM run_branches rb
         LEFT JOIN runs p ON p.id = rb.parent_run_id
         LEFT JOIN runs c ON c.id = rb.child_run_id
         WHERE rb.campaign_id = ?
         ORDER BY rb.created_at""",
        (campaign_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_branches_for_checkpoint(conn: sqlite3.Connection,
                                  checkpoint_id: int) -> list[dict]:
    """Return all branches originating from a specific checkpoint."""
    rows = conn.execute(
        """SELECT rb.*,
               p.sweep_tag AS parent_sweep_tag,
               c.sweep_tag AS child_sweep_tag
         FROM run_branches rb
         LEFT JOIN runs p ON p.id = rb.parent_run_id
         LEFT JOIN runs c ON c.id = rb.child_run_id
         WHERE rb.parent_checkpoint_id = ?
         ORDER BY rb.created_at""",
        (checkpoint_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_branch_tree(conn: sqlite3.Connection,
                    campaign_id: str) -> list[dict]:
    """Return a flat list of all branches with parent/child context for tree building."""
    rows = conn.execute(
        """SELECT rb.id, rb.parent_run_id, rb.child_run_id,
               rb.parent_checkpoint_id, rb.branch_reason, rb.delta_json,
               rb.status, rb.result_summary_json,
               p.sweep_tag AS parent_tag, p.final_win_rate AS parent_wr,
               c.sweep_tag AS child_tag, c.final_win_rate AS child_wr,
               c.wall_time_s AS child_wall_s, c.num_params AS child_params
         FROM run_branches rb
         LEFT JOIN runs p ON p.id = rb.parent_run_id
         LEFT JOIN runs c ON c.id = rb.child_run_id
         WHERE rb.campaign_id = ?
         ORDER BY rb.created_at""",
        (campaign_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_branch_by_id(conn: sqlite3.Connection,
                     branch_id: str) -> dict | None:
    """Return a single branch by id, or None."""
    row = conn.execute(
        """SELECT rb.*,
               p.sweep_tag AS parent_sweep_tag,
               c.sweep_tag AS child_sweep_tag
         FROM run_branches rb
         LEFT JOIN runs p ON p.id = rb.parent_run_id
         LEFT JOIN runs c ON c.id = rb.child_run_id
         WHERE rb.id = ?""",
        (branch_id,),
    ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# v21: recommendation ledger helpers
# ---------------------------------------------------------------------------

def save_recommendation_batch(
    conn: sqlite3.Connection, *,
    batch_id: str,
    campaign_id: str,
    selector_name: str,
    selector_version: str,
    selector_hash: str,
    frontier_snapshot_id: str | None = None,
    acquisition_name: str | None = None,
    acquisition_version: str | None = None,
    surrogate_snapshot_id: str | None = None,
) -> None:
    """Persist a recommendation batch."""
    conn.execute(
        """INSERT INTO recommendation_batches
           (id, campaign_id, selector_name, selector_version, selector_hash,
            frontier_snapshot_id, acquisition_name, acquisition_version,
            surrogate_snapshot_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               campaign_id = excluded.campaign_id,
               selector_name = excluded.selector_name,
               selector_version = excluded.selector_version,
               selector_hash = excluded.selector_hash,
               frontier_snapshot_id = excluded.frontier_snapshot_id,
               acquisition_name = excluded.acquisition_name,
               acquisition_version = excluded.acquisition_version,
               surrogate_snapshot_id = excluded.surrogate_snapshot_id""",
        (batch_id, campaign_id, selector_name, selector_version, selector_hash,
         frontier_snapshot_id, acquisition_name, acquisition_version,
         surrogate_snapshot_id, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def save_recommendation(
    conn: sqlite3.Connection, *,
    recommendation_id: str,
    batch_id: str,
    candidate_type: str,
    candidate_key: str | None,
    rank: int,
    score_total: float,
    score_breakdown_json: str,
    rationale_json: str,
    axis_values_json: str | None = None,
    branch_reason: str | None = None,
    delta_json: str | None = None,
    selector_score_total: float | None = None,
    acquisition_score: float | None = None,
    parent_run_id: str | None = None,
    parent_checkpoint_id: int | None = None,
    status: str = "planned",
) -> None:
    """Persist a single recommendation."""
    conn.execute(
        """INSERT INTO recommendations
           (id, batch_id, candidate_type, candidate_key, rank, score_total,
             score_breakdown_json, rationale_json, axis_values_json,
             branch_reason, delta_json, selector_score_total, acquisition_score,
             parent_run_id, parent_checkpoint_id, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                batch_id = excluded.batch_id,
                candidate_type = excluded.candidate_type,
                candidate_key = excluded.candidate_key,
                rank = excluded.rank,
               score_total = excluded.score_total,
                score_breakdown_json = excluded.score_breakdown_json,
                rationale_json = excluded.rationale_json,
                axis_values_json = excluded.axis_values_json,
                branch_reason = excluded.branch_reason,
                delta_json = excluded.delta_json,
                selector_score_total = excluded.selector_score_total,
                acquisition_score = excluded.acquisition_score,
                parent_run_id = excluded.parent_run_id,
                parent_checkpoint_id = excluded.parent_checkpoint_id,
                status = excluded.status""",
        (recommendation_id, batch_id, candidate_type, candidate_key, rank,
         score_total, score_breakdown_json, rationale_json, axis_values_json,
         branch_reason, delta_json, selector_score_total, acquisition_score,
         parent_run_id, parent_checkpoint_id, status, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def save_surrogate_snapshot(
    conn: sqlite3.Connection, *,
    snapshot_id: str,
    campaign_id: str,
    frontier_snapshot_id: str | None,
    acquisition_name: str,
    acquisition_version: str,
    policy_hash: str,
    objectives_json: str,
    feature_schema_json: str,
    summary_json: str,
    candidate_count: int,
) -> None:
    """Persist a surrogate/acquisition snapshot for recommendation replay."""
    conn.execute(
        """INSERT INTO surrogate_snapshots
           (id, campaign_id, frontier_snapshot_id, acquisition_name,
            acquisition_version, policy_hash, objectives_json,
            feature_schema_json, summary_json, candidate_count, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               campaign_id = excluded.campaign_id,
               frontier_snapshot_id = excluded.frontier_snapshot_id,
               acquisition_name = excluded.acquisition_name,
               acquisition_version = excluded.acquisition_version,
               policy_hash = excluded.policy_hash,
               objectives_json = excluded.objectives_json,
               feature_schema_json = excluded.feature_schema_json,
               summary_json = excluded.summary_json,
               candidate_count = excluded.candidate_count""",
        (
            snapshot_id,
            campaign_id,
            frontier_snapshot_id,
            acquisition_name,
            acquisition_version,
            policy_hash,
            objectives_json,
            feature_schema_json,
            summary_json,
            candidate_count,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def get_surrogate_snapshot(conn: sqlite3.Connection, snapshot_id: str) -> dict | None:
    """Return a single surrogate snapshot by id, or None."""
    row = conn.execute(
        "SELECT * FROM surrogate_snapshots WHERE id = ?",
        (snapshot_id,),
    ).fetchone()
    return dict(row) if row else None


def list_surrogate_snapshots(conn: sqlite3.Connection, campaign_id: str) -> list[dict]:
    """Return surrogate snapshots for a campaign, newest first."""
    rows = conn.execute(
        "SELECT * FROM surrogate_snapshots WHERE campaign_id = ? ORDER BY created_at DESC",
        (campaign_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def update_recommendation_status(
    conn: sqlite3.Connection, *,
    recommendation_id: str,
    status: str,
) -> None:
    """Update recommendation status (planned / accepted / executed / rejected / invalidated)."""
    conn.execute(
        "UPDATE recommendations SET status = ? WHERE id = ?",
        (status, recommendation_id),
    )
    conn.commit()


def save_recommendation_outcome(
    conn: sqlite3.Connection, *,
    recommendation_id: str,
    run_id: str | None = None,
    branch_id: str | None = None,
    observed_metrics_json: str = "{}",
    frontier_delta_json: str | None = None,
    outcome_label: str = "unknown",
) -> None:
    """Persist an outcome record for a recommendation."""
    conn.execute(
        """INSERT INTO recommendation_outcomes
           (recommendation_id, run_id, branch_id, observed_metrics_json,
            frontier_delta_json, outcome_label, evaluated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (recommendation_id, run_id, branch_id, observed_metrics_json,
         frontier_delta_json, outcome_label, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def list_recommendation_batches(conn: sqlite3.Connection,
                                 campaign_id: str) -> list[dict]:
    """Return all recommendation batches for a campaign, newest first."""
    rows = conn.execute(
        "SELECT * FROM recommendation_batches WHERE campaign_id = ? ORDER BY created_at DESC",
        (campaign_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_recommendations_for_batch(conn: sqlite3.Connection,
                                    batch_id: str) -> list[dict]:
    """Return all recommendations in a batch, ordered by rank."""
    rows = conn.execute(
        """SELECT r.*, b.campaign_id
           FROM recommendations r
           JOIN recommendation_batches b ON b.id = r.batch_id
           WHERE r.batch_id = ?
           ORDER BY r.rank""",
        (batch_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_recommendation_outcomes(conn: sqlite3.Connection,
                                  recommendation_id: str) -> list[dict]:
    """Return all outcome records for a recommendation."""
    rows = conn.execute(
        "SELECT * FROM recommendation_outcomes WHERE recommendation_id = ? ORDER BY evaluated_at",
        (recommendation_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_latest_recommendation_batch(conn: sqlite3.Connection,
                                     campaign_id: str) -> dict | None:
    """Return the most recent batch for a campaign, or None."""
    row = conn.execute(
        "SELECT * FROM recommendation_batches WHERE campaign_id = ? ORDER BY created_at DESC LIMIT 1",
        (campaign_id,),
    ).fetchone()
    return dict(row) if row else None


def get_recommendation_by_id(conn: sqlite3.Connection,
                              recommendation_id: str) -> dict | None:
    """Return a single recommendation by id, or None."""
    row = conn.execute(
        """SELECT r.*, b.campaign_id, b.frontier_snapshot_id,
                  b.acquisition_name, b.acquisition_version, b.surrogate_snapshot_id
           FROM recommendations r
           JOIN recommendation_batches b ON b.id = r.batch_id
           WHERE r.id = ?""",
        (recommendation_id,),
    ).fetchone()
    return dict(row) if row else None
