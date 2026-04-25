"""Run, checkpoint, recording, and opponent repositories."""

from __future__ import annotations

import os
import platform
import sqlite3
import subprocess
from typing import Optional

from .common import utc_now_iso
from .schema_base import CHECKPOINT_THRESHOLDS, can_promote


def collect_hardware_info() -> dict:
    info: dict = {}
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
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
        proc_str = hw.get("number_processors", "0")
        if isinstance(proc_str, str) and ":" in proc_str:
            info["cpu_cores"] = int(proc_str.split()[1].split(":")[0])
        elif isinstance(proc_str, int):
            info["cpu_cores"] = proc_str
        else:
            info["cpu_cores"] = int(proc_str) if proc_str.isdigit() else 0
        mem_str = hw.get("physical_memory", "0 GB")
        info["memory_gb"] = int(mem_str.split()[0]) if isinstance(mem_str, str) else 0
        gpu_str = hw.get("platform_number_gpu_cores", "")
        if gpu_str:
            info["gpu_cores"] = int(gpu_str)
        else:
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


def create_run(
    conn: sqlite3.Connection,
    run_id: str,
    hyperparams: dict,
    hardware: Optional[dict] = None,
    resumed_from: Optional[str] = None,
    output_dir: Optional[str] = None,
    is_benchmark: bool = False,
    eval_opponent: Optional[str] = None,
) -> None:
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
            utc_now_iso(),
            hw.get("chip"),
            hw.get("cpu_cores"),
            hw.get("gpu_cores"),
            hw.get("memory_gb"),
            hw.get("mlx_version"),
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
    artifact_dir = hyperparams.get("artifact_dir") or output_dir
    if artifact_dir:
        try:
            conn.execute("UPDATE runs SET artifact_dir = ? WHERE id = ?", (artifact_dir, run_id))
        except sqlite3.OperationalError:
            pass
    conn.commit()


def finish_run(conn: sqlite3.Connection, run_id: str, summary: dict) -> None:
    conn.execute(
        """UPDATE runs SET
            finished_at = ?, status = ?,
            total_cycles = ?, total_games = ?, total_steps = ?,
            final_loss = ?, final_win_rate = ?,
            num_params = ?, num_checkpoints = ?,
            wall_time_s = ?, peak_memory_mb = ?
        WHERE id = ?""",
        (
            utc_now_iso(),
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


def save_cycle_metric(conn: sqlite3.Connection, run_id: str, metric: dict) -> None:
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


def save_checkpoint(conn: sqlite3.Connection, run_id: str, data: dict) -> int:
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
            utc_now_iso(),
            data.get("train_elapsed_s"),
            data.get("eval_elapsed_s"),
            data.get("eval_unique_openings"),
            eligible_int,
            eligible_reason,
        ),
    )
    conn.commit()
    return cur.lastrowid


def save_eval_breakdown(conn: sqlite3.Connection, checkpoint_id: int, breakdown: list[dict]) -> None:
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


def save_recording(conn: sqlite3.Connection, checkpoint_id: int, run_id: str, data: dict) -> None:
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


def save_recordings_batch(conn: sqlite3.Connection, checkpoint_id: int, run_id: str, records: list[dict]) -> None:
    for rec in records:
        save_recording(conn, checkpoint_id, run_id, rec)
    conn.commit()


def should_checkpoint(current_wr: float, last_ckpt_wr: float) -> bool:
    return any(last_ckpt_wr < t <= current_wr for t in CHECKPOINT_THRESHOLDS)


def crossed_threshold(current_wr: float, last_ckpt_wr: float) -> Optional[float]:
    crossed = None
    for t in CHECKPOINT_THRESHOLDS:
        if last_ckpt_wr < t <= current_wr:
            crossed = t
    return crossed


def next_threshold(last_ckpt_wr: float) -> Optional[float]:
    for t in CHECKPOINT_THRESHOLDS:
        if t > last_ckpt_wr:
            return t
    return None


def get_run(conn: sqlite3.Connection, run_id: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row:
        return dict(row)
    rows = conn.execute("SELECT * FROM runs WHERE id LIKE ?", (run_id + "%",)).fetchall()
    if len(rows) == 1:
        return dict(rows[0])
    return None


def get_checkpoints(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = conn.execute("SELECT * FROM checkpoints WHERE run_id = ? ORDER BY cycle", (run_id,)).fetchall()
    return [dict(r) for r in rows]


def get_recordings(conn: sqlite3.Connection, checkpoint_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM recordings WHERE checkpoint_id = ? ORDER BY game_index",
        (checkpoint_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_cycle_metrics(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    rows = conn.execute("SELECT * FROM cycle_metrics WHERE run_id = ? ORDER BY cycle", (run_id,)).fetchall()
    return [dict(r) for r in rows]


def count_checkpoints(conn: sqlite3.Connection, run_id: str) -> int:
    row = conn.execute("SELECT COUNT(*) FROM checkpoints WHERE run_id = ?", (run_id,)).fetchone()
    return row[0] if row else 0


def get_latest_checkpoint(conn: sqlite3.Connection, run_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY cycle DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    return dict(row) if row else None


def list_all_checkpoints(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        """SELECT c.*, r.chip, r.output_dir
           FROM checkpoints c
           JOIN runs r ON c.run_id = r.id
           ORDER BY c.created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def find_checkpoint_by_tag(conn: sqlite3.Connection, tag: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM checkpoints WHERE tag = ?", (tag,)).fetchone()
    if row:
        return dict(row)
    row = conn.execute(
        "SELECT * FROM checkpoints WHERE tag LIKE ? ORDER BY created_at DESC LIMIT 1",
        (f"%{tag}%",),
    ).fetchone()
    return dict(row) if row else None


def register_opponent(
    conn: sqlite3.Connection,
    alias: str,
    model_path: str,
    source_run: Optional[str] = None,
    source_tag: Optional[str] = None,
    win_rate: Optional[float] = None,
    eval_level: Optional[int] = None,
    description: Optional[str] = None,
    num_res_blocks: Optional[int] = None,
    num_filters: Optional[int] = None,
    prev_alias: Optional[str] = None,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO opponents (
            alias, source_run, source_tag, model_path,
            win_rate, eval_level, description, created_at,
            num_res_blocks, num_filters, prev_alias
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            alias,
            source_run,
            source_tag,
            model_path,
            win_rate,
            eval_level,
            description,
            utc_now_iso(),
            num_res_blocks,
            num_filters,
            prev_alias,
        ),
    )
    conn.commit()


def get_opponent(conn: sqlite3.Connection, alias: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM opponents WHERE alias = ?", (alias,)).fetchone()
    return dict(row) if row else None


def list_opponents(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM opponents ORDER BY created_at").fetchall()
    return [dict(r) for r in rows]
