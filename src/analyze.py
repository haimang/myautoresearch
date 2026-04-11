#!/usr/bin/env python3
"""MAG-Gomoku tracker analysis tool (read-only).

Queries the SQLite tracker database to report on training runs,
checkpoints, opponents, and win-rate progression.

Usage:
    uv run python src/analyze.py --best
    uv run python src/analyze.py --frontier
    uv run python src/analyze.py --compare RUN_A RUN_B
    uv run python src/analyze.py --lineage RUN_ID
    uv run python src/analyze.py --opponents
    uv run python src/analyze.py --runs
    uv run python src/analyze.py --report
    uv run python src/analyze.py --report --format json
"""

import argparse
import json as _json
import math
import sqlite3
import sys
import os
from datetime import datetime, timezone

DB_PATH = os.path.join("output", "tracker.db")


def _connect() -> sqlite3.Connection:
    if not os.path.exists(DB_PATH):
        print(f"Error: database not found at {DB_PATH}")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Ensure schema migrations are applied
    for col, typ in [("is_benchmark", "INTEGER DEFAULT 0"), ("eval_opponent", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass
    return conn


def _col(text: str, width: int) -> str:
    """Left-pad/truncate text to fixed width."""
    return str(text)[:width].ljust(width)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_runs(conn: sqlite3.Connection) -> None:
    """List all training runs with summary stats."""
    rows = conn.execute(
        """SELECT id, status, started_at, total_cycles, total_games,
                  final_loss, final_win_rate, wall_time_s, eval_level,
                  is_benchmark, eval_opponent, resumed_from,
                  num_res_blocks, num_filters, num_params
           FROM runs ORDER BY started_at DESC"""
    ).fetchall()
    if not rows:
        print("No runs found.")
        return
    print(f"{'Run':>10}  {'Status':>11}  {'Cycles':>6}  {'Games':>6}  "
          f"{'Loss':>7}  {'WR':>6}  {'Time':>7}  {'Opp':>6}  {'Model':>8}  {'Type':>12}")
    print("-" * 105)
    for r in rows:
        rid = r["id"][:8]
        st = r["status"] or "?"
        cyc = r["total_cycles"] if r["total_cycles"] else "-"
        gm = r["total_games"] if r["total_games"] else "-"
        loss = f"{r['final_loss']:.3f}" if r["final_loss"] else "-"
        wr = f"{r['final_win_rate']:.1%}" if r["final_win_rate"] is not None else "-"
        wt = f"{r['wall_time_s']:.0f}s" if r["wall_time_s"] else "-"
        opp = r["eval_opponent"] if r["eval_opponent"] else f"L{r['eval_level'] or 0}"
        bm = "benchmark" if r["is_benchmark"] else "exploratory"
        resumed = " (resumed)" if r["resumed_from"] else ""
        nb = r["num_res_blocks"] or "?"
        nf = r["num_filters"] or "?"
        model_str = f"{nb}x{nf}"
        print(f"{rid:>10}  {st:>11}  {str(cyc):>6}  {str(gm):>6}  "
              f"{loss:>7}  {wr:>6}  {wt:>7}  {opp:>6}  {model_str:>8}  {bm}{resumed}")


def cmd_best(conn: sqlite3.Connection) -> None:
    """Show the best checkpoint per completed run."""
    rows = conn.execute(
        """SELECT c.run_id, c.tag, c.win_rate, c.eval_games, c.cycle,
                  r.eval_level, r.eval_opponent
           FROM checkpoints c
           JOIN runs r ON c.run_id = r.id
           WHERE c.win_rate = (
               SELECT MAX(c2.win_rate) FROM checkpoints c2 WHERE c2.run_id = c.run_id
           )
           ORDER BY c.win_rate DESC"""
    ).fetchall()
    if not rows:
        print("No checkpoints found.")
        return
    print(f"{'Run':>10}  {'Tag':>16}  {'WR':>6}  {'Games':>6}  {'Cycle':>6}  {'Opponent':>8}")
    print("-" * 62)
    for r in rows:
        rid = r["run_id"][:8]
        opp = r["eval_opponent"] if r["eval_opponent"] else f"L{r['eval_level'] or 0}"
        print(f"{rid:>10}  {r['tag']:>16}  {r['win_rate']:>5.1%}  "
              f"{r['eval_games'] or '-':>6}  {r['cycle']:>6}  {opp:>8}")


def cmd_frontier(conn: sqlite3.Connection) -> None:
    """Show WR progression across all runs (monotonically improving checkpoints)."""
    rows = conn.execute(
        """SELECT c.run_id, c.tag, c.win_rate, c.created_at, c.cycle,
                  r.eval_level, r.eval_opponent
           FROM checkpoints c
           JOIN runs r ON c.run_id = r.id
           ORDER BY c.win_rate ASC, c.created_at ASC"""
    ).fetchall()
    if not rows:
        print("No checkpoints found.")
        return
    print(f"{'Run':>10}  {'Tag':>16}  {'WR':>6}  {'Cycle':>6}  {'Opponent':>8}  {'Date':>20}")
    print("-" * 78)
    best_wr = -1.0
    for r in rows:
        if r["win_rate"] > best_wr:
            best_wr = r["win_rate"]
            rid = r["run_id"][:8]
            opp = r["eval_opponent"] if r["eval_opponent"] else f"L{r['eval_level'] or 0}"
            dt = r["created_at"][:19] if r["created_at"] else "-"
            print(f"{rid:>10}  {r['tag']:>16}  {r['win_rate']:>5.1%}  "
                  f"{r['cycle']:>6}  {opp:>8}  {dt:>20}")


def cmd_compare(conn: sqlite3.Connection, run_a: str, run_b: str) -> None:
    """Compare two runs side-by-side."""
    def _get_run(rid: str) -> dict:
        row = conn.execute(
            "SELECT * FROM runs WHERE id LIKE ?", (rid + "%",)
        ).fetchone()
        if not row:
            print(f"Run not found: {rid}")
            sys.exit(1)
        return dict(row)

    a, b = _get_run(run_a), _get_run(run_b)

    fields = [
        ("Status", "status"),
        ("Model", None),  # special handling
        ("Cycles", "total_cycles"),
        ("Games", "total_games"),
        ("Steps", "total_steps"),
        ("Final Loss", "final_loss"),
        ("Final WR", "final_win_rate"),
        ("Checkpoints", "num_checkpoints"),
        ("Wall Time", "wall_time_s"),
        ("Eval Level", "eval_level"),
        ("Opponent", "eval_opponent"),
        ("Benchmark", "is_benchmark"),
        ("Buffer Size", "replay_buffer_size"),
        ("Batch Size", "batch_size"),
        ("Parallel", "parallel_games"),
    ]

    print(f"{'':>16}  {a['id'][:8]:>14}  {b['id'][:8]:>14}")
    print("-" * 50)
    for label, key in fields:
        if key is None and label == "Model":
            va = f"{a.get('num_res_blocks') or '?'}x{a.get('num_filters') or '?'} ({a.get('num_params') or 0:.0f})"
            vb = f"{b.get('num_res_blocks') or '?'}x{b.get('num_filters') or '?'} ({b.get('num_params') or 0:.0f})"
            print(f"{label:>16}  {va:>14}  {vb:>14}")
            continue
        va = a.get(key, "-")
        vb = b.get(key, "-")
        if key == "final_win_rate":
            va = f"{va:.1%}" if va is not None else "-"
            vb = f"{vb:.1%}" if vb is not None else "-"
        elif key == "final_loss":
            va = f"{va:.4f}" if va is not None else "-"
            vb = f"{vb:.4f}" if vb is not None else "-"
        elif key == "wall_time_s":
            va = f"{va:.0f}s" if va is not None else "-"
            vb = f"{vb:.0f}s" if vb is not None else "-"
        elif key == "is_benchmark":
            va = "yes" if va else "no"
            vb = "yes" if vb else "no"
        else:
            va = str(va) if va is not None else "-"
            vb = str(vb) if vb is not None else "-"
        print(f"{label:>16}  {va:>14}  {vb:>14}")


def cmd_lineage(conn: sqlite3.Connection, run_id: str) -> None:
    """Trace the resume chain for a run."""
    chain = []
    current = run_id
    while current:
        row = conn.execute(
            "SELECT id, resumed_from, status, total_cycles, final_win_rate "
            "FROM runs WHERE id LIKE ?", (current + "%",)
        ).fetchone()
        if not row:
            break
        chain.append(dict(row))
        current = row["resumed_from"]

    if not chain:
        print(f"Run not found: {run_id}")
        return

    chain.reverse()
    print("Resume lineage (oldest first):")
    print("-" * 60)
    for i, r in enumerate(chain):
        prefix = "  " * i + ("└─ " if i > 0 else "")
        wr = f"{r['final_win_rate']:.1%}" if r["final_win_rate"] is not None else "-"
        cyc = r["total_cycles"] if r["total_cycles"] else "-"
        print(f"{prefix}{r['id'][:8]}  {r['status']}  cycles={cyc}  WR={wr}")


def cmd_opponents(conn: sqlite3.Connection) -> None:
    """List all registered opponents."""
    try:
        rows = conn.execute(
            """SELECT alias, source_run, source_tag, win_rate,
                      created_at, description
               FROM opponents ORDER BY created_at"""
        ).fetchall()
    except sqlite3.OperationalError:
        print("No opponents table found.")
        return
    if not rows:
        print("No opponents registered.")
        return
    print(f"{'Alias':>10}  {'Source':>10}  {'Tag':>16}  {'WR':>6}  {'Date':>12}  Description")
    print("-" * 80)
    for r in rows:
        rid = r["source_run"][:8] if r["source_run"] else "-"
        wr = f"{r['win_rate']:.1%}" if r["win_rate"] is not None else "-"
        dt = r["created_at"][:10] if r["created_at"] else "-"
        desc = r["description"] or ""
        print(f"{r['alias']:>10}  {rid:>10}  {r['source_tag'] or '-':>16}  "
              f"{wr:>6}  {dt:>12}  {desc}")


def cmd_matrix(conn: sqlite3.Connection, tag_prefix: str) -> None:
    """Show sweep results grouped by tag prefix with aggregated metrics."""

    rows = conn.execute(
        "SELECT id, status, sweep_tag, total_cycles, total_games, "
        "final_loss, final_win_rate, wall_time_s, num_params, "
        "learning_rate, train_steps_per_cycle, num_res_blocks, num_filters, "
        "replay_buffer_size, seed "
        "FROM runs WHERE sweep_tag IS NOT NULL ORDER BY started_at"
    ).fetchall()

    # Filter to runs with matching sweep_tag prefix
    groups: dict[str, list[dict]] = {}
    for r in rows:
        sweep_tag = r["sweep_tag"] or ""
        if not sweep_tag.startswith(tag_prefix):
            continue
        # Strip seed suffix to group by config
        parts = sweep_tag.rsplit("_sd", 1)
        base_tag = parts[0] if len(parts) == 2 else sweep_tag
        seed = parts[1] if len(parts) == 2 else "?"

        entry = {
            "id": r["id"], "status": r["status"], "seed": seed,
            "cycles": r["total_cycles"], "games": r["total_games"],
            "loss": r["final_loss"], "wr": r["final_win_rate"],
            "wall_s": r["wall_time_s"], "params": r["num_params"],
            "lr": r["learning_rate"], "spc": r["train_steps_per_cycle"],
            "nb": r["num_res_blocks"], "nf": r["num_filters"],
            "buf": r["replay_buffer_size"],
        }
        groups.setdefault(base_tag, []).append(entry)

    if not groups:
        print(f"No runs found with sweep_tag prefix '{tag_prefix}'")
        return

    # Extract varying axes from hyperparams across groups
    print(f"Sweep Matrix: {tag_prefix}  ({sum(len(v) for v in groups.values())} runs in {len(groups)} configs)")
    print("=" * 95)
    print(f"  {'Config':<35} {'Seeds':>5}  {'WR Mean':>7}  {'WR Std':>6}  "
          f"{'Loss':>7}  {'Games/s':>7}  {'Params':>8}")
    print("-" * 95)

    best_wr = -1.0
    best_tag = ""

    for base_tag in sorted(groups.keys()):
        runs = groups[base_tag]
        n = len(runs)
        wrs = [r["wr"] for r in runs if r["wr"] is not None]
        losses = [r["loss"] for r in runs if r["loss"] is not None]
        throughputs = []
        for r in runs:
            if r["games"] and r["wall_s"] and r["wall_s"] > 0:
                throughputs.append(r["games"] / r["wall_s"])

        mean_wr = sum(wrs) / len(wrs) if wrs else 0
        std_wr = math.sqrt(sum((w - mean_wr)**2 for w in wrs) / len(wrs)) if len(wrs) > 1 else 0
        mean_loss = sum(losses) / len(losses) if losses else 0
        mean_thr = sum(throughputs) / len(throughputs) if throughputs else 0
        params = runs[0]["params"] or 0

        # Short config label: strip the tag prefix
        label = base_tag[len(tag_prefix) + 1:] if base_tag.startswith(tag_prefix + "_") else base_tag

        marker = ""
        if mean_wr > best_wr:
            best_wr = mean_wr
            best_tag = label

        print(f"  {label:<35} {n:>5}  {mean_wr:>6.1%}  {std_wr:>5.1%}  "
              f"{mean_loss:>7.3f}  {mean_thr:>6.1f}/s  {params:>8.0f}")

    print("-" * 95)
    if best_tag:
        print(f"  Best mean WR: {best_tag}  ({best_wr:.1%})")
    print()


def cmd_stability(conn: sqlite3.Connection, run_id: str) -> None:
    """Show training stability metrics for a run."""

    row = conn.execute(
        "SELECT id, total_cycles, total_games, final_win_rate, final_loss "
        "FROM runs WHERE id LIKE ?", (run_id + "%",)
    ).fetchone()
    if not row:
        print(f"Run not found: {run_id}")
        return

    full_id = row["id"]
    print(f"Stability report for run {full_id[:8]}")
    print("=" * 60)

    # WR metrics
    wr_rows = conn.execute(
        "SELECT cycle, win_rate FROM cycle_metrics "
        "WHERE run_id = ? AND win_rate IS NOT NULL ORDER BY cycle",
        (full_id,)
    ).fetchall()
    if wr_rows:
        wrs = [r["win_rate"] for r in wr_rows]
        n = len(wrs)
        mean_wr = sum(wrs) / n
        var_wr = sum((w - mean_wr) ** 2 for w in wrs) / n if n > 1 else 0
        std_wr = math.sqrt(var_wr)
        min_wr, max_wr = min(wrs), max(wrs)
        # Max consecutive swing
        max_swing = 0.0
        for i in range(1, len(wrs)):
            max_swing = max(max_swing, abs(wrs[i] - wrs[i - 1]))

        print(f"\nWin Rate ({n} probes)")
        print(f"  Mean:       {mean_wr:.1%}")
        print(f"  Std Dev:    {std_wr:.1%}")
        print(f"  Range:      {min_wr:.1%} — {max_wr:.1%}")
        print(f"  Max Swing:  {max_swing:.1%} (between consecutive probes)")
        if n >= 5:
            last5 = wrs[-5:]
            trend = last5[-1] - last5[0]
            print(f"  Last 5 WR:  {' → '.join(f'{w:.0%}' for w in last5)}  (trend: {trend:+.1%})")
    else:
        print("\nNo win-rate data.")

    # Loss metrics
    loss_rows = conn.execute(
        "SELECT cycle, loss FROM cycle_metrics "
        "WHERE run_id = ? AND loss IS NOT NULL ORDER BY cycle",
        (full_id,)
    ).fetchall()
    if loss_rows:
        losses = [r["loss"] for r in loss_rows]
        n = len(losses)
        mean_loss = sum(losses) / n
        var_loss = sum((l - mean_loss) ** 2 for l in losses) / n if n > 1 else 0
        std_loss = math.sqrt(var_loss)
        min_loss, max_loss = min(losses), max(losses)
        # Loss reduction rate (first vs last quarter)
        q = max(n // 4, 1)
        first_q = sum(losses[:q]) / q
        last_q = sum(losses[-q:]) / q
        reduction = (first_q - last_q) / first_q * 100 if first_q > 0 else 0

        print(f"\nLoss ({n} cycles)")
        print(f"  Mean:       {mean_loss:.4f}")
        print(f"  Std Dev:    {std_loss:.4f}")
        print(f"  Range:      {min_loss:.4f} — {max_loss:.4f}")
        print(f"  Reduction:  {reduction:.0f}% (first quarter → last quarter)")
        if n >= 5:
            last5 = losses[-5:]
            print(f"  Last 5:     {' → '.join(f'{l:.3f}' for l in last5)}")
    else:
        print("\nNo loss data.")

    # Checkpoint distribution
    ckpt_rows = conn.execute(
        "SELECT tag, cycle, win_rate FROM checkpoints "
        "WHERE run_id = ? ORDER BY cycle", (full_id,)
    ).fetchall()
    if ckpt_rows:
        print(f"\nCheckpoints ({len(ckpt_rows)})")
        for c in ckpt_rows:
            wr = f"{c['win_rate']:.1%}" if c["win_rate"] is not None else "-"
            print(f"  cycle {c['cycle']:>4}  {c['tag']:<18}  WR={wr}")

    print()


# ---------------------------------------------------------------------------
# Report generation (dual format: markdown / JSON)
# ---------------------------------------------------------------------------

# Stage promotion ladder
_STAGES = [
    {"level": 0, "opponent": "L0 (random)",        "threshold": 0.95, "next": "L1 (minimax depth 2)"},
    {"level": 1, "opponent": "L1 (minimax depth 2)","threshold": 0.80, "next": "L2 (minimax depth 4)"},
    {"level": 2, "opponent": "L2 (minimax depth 4)","threshold": 0.60, "next": "L3 (minimax depth 6)"},
    {"level": 3, "opponent": "L3 (minimax depth 6)","threshold": None,  "next": None},
]


def _gather_report_data(conn: sqlite3.Connection, n_recent: int = 5) -> dict:
    """Gather all data needed for report generation."""

    # --- Section 1: Recent runs ---
    recent = conn.execute(
        "SELECT id, status, started_at, total_cycles, total_games, total_steps,"
        " final_loss, final_win_rate, wall_time_s, eval_level, eval_opponent,"
        " is_benchmark, num_res_blocks, num_filters, num_params,"
        " learning_rate, train_steps_per_cycle, replay_buffer_size,"
        " parallel_games, seed, sweep_tag"
        " FROM runs WHERE status='completed'"
        " ORDER BY started_at DESC LIMIT ?", (n_recent,)
    ).fetchall()

    # --- Section 2: Best checkpoint (highest WR across all runs) ---
    best_ckpt = conn.execute(
        "SELECT c.run_id, c.tag, c.win_rate, c.eval_games, c.cycle,"
        " c.num_params, r.eval_level, r.eval_opponent"
        " FROM checkpoints c JOIN runs r ON c.run_id = r.id"
        " ORDER BY c.win_rate DESC LIMIT 1"
    ).fetchone()

    # Best benchmark WR
    best_bm = conn.execute(
        "SELECT id, final_win_rate, num_res_blocks, num_filters, wall_time_s"
        " FROM runs WHERE status='completed' AND is_benchmark=1"
        " ORDER BY final_win_rate DESC LIMIT 1"
    ).fetchone()

    # --- Section 3: Frontier (monotonically improving checkpoints) ---
    all_ckpts = conn.execute(
        "SELECT c.run_id, c.tag, c.win_rate, c.created_at, c.cycle,"
        " r.eval_level, r.eval_opponent"
        " FROM checkpoints c JOIN runs r ON c.run_id = r.id"
        " ORDER BY c.created_at ASC"
    ).fetchall()
    frontier = []
    best_wr = -1.0
    for c in all_ckpts:
        if c["win_rate"] is not None and c["win_rate"] > best_wr:
            best_wr = c["win_rate"]
            frontier.append(dict(c))

    # --- Section 4: Stability of latest completed run ---
    latest_run = conn.execute(
        "SELECT id, total_cycles FROM runs WHERE status='completed'"
        " ORDER BY started_at DESC LIMIT 1"
    ).fetchone()

    stability = {"run_id": None, "wr": None, "loss": None}
    if latest_run:
        lid = latest_run["id"]
        stability["run_id"] = lid

        wr_rows = conn.execute(
            "SELECT cycle, win_rate FROM cycle_metrics"
            " WHERE run_id=? AND win_rate IS NOT NULL ORDER BY cycle", (lid,)
        ).fetchall()
        if len(wr_rows) >= 3:
            wrs = [r["win_rate"] for r in wr_rows]
            mean_wr = sum(wrs) / len(wrs)
            std_wr = math.sqrt(sum((w - mean_wr)**2 for w in wrs) / len(wrs))
            stability["wr"] = {
                "n": len(wrs), "mean": mean_wr, "std": std_wr,
                "min": min(wrs), "max": max(wrs),
                "last_5": wrs[-5:],
            }

        loss_rows = conn.execute(
            "SELECT cycle, loss FROM cycle_metrics"
            " WHERE run_id=? AND loss IS NOT NULL ORDER BY cycle", (lid,)
        ).fetchall()
        if loss_rows:
            losses = [r["loss"] for r in loss_rows]
            n = len(losses)
            mean_l = sum(losses) / n
            std_l = math.sqrt(sum((l - mean_l)**2 for l in losses) / n) if n > 1 else 0
            q = max(n // 4, 1)
            first_q = sum(losses[:q]) / q
            last_q = sum(losses[-q:]) / q
            reduction = (first_q - last_q) / first_q * 100 if first_q > 0 else 0
            stability["loss"] = {
                "n": n, "mean": mean_l, "std": std_l,
                "min": min(losses), "max": max(losses),
                "reduction_pct": reduction,
                "last_5": losses[-5:],
            }

    # --- Section 5: Opponents ---
    try:
        opponents = conn.execute(
            "SELECT alias, source_run, source_tag, win_rate, description"
            " FROM opponents ORDER BY created_at"
        ).fetchall()
    except sqlite3.OperationalError:
        opponents = []

    # --- Section 6: Stage assessment ---
    # Determine current stage from most recent run's eval_level
    cur_level = 0
    if recent:
        cur_level = recent[0]["eval_level"] or 0

    stage_info = _STAGES[min(cur_level, len(_STAGES) - 1)]
    best_bm_wr = best_bm["final_win_rate"] if best_bm else None
    gap = (stage_info["threshold"] - best_bm_wr) if (stage_info["threshold"] and best_bm_wr) else None

    stage = {
        "current": cur_level,
        "opponent": stage_info["opponent"],
        "best_benchmark_wr": best_bm_wr,
        "promotion_threshold": stage_info["threshold"],
        "promotion_target": stage_info["next"],
        "gap_to_promotion": gap,
    }

    # --- Section 7: Signals ---
    signals = _generate_signals(conn, recent, best_bm, stage, stability)

    # --- Hyperparams summary ---
    all_completed = conn.execute(
        "SELECT num_res_blocks, num_filters, learning_rate FROM runs"
        " WHERE status='completed'"
    ).fetchall()
    archs = sorted(set(f"{r['num_res_blocks']}x{r['num_filters']}" for r in all_completed
                       if r["num_res_blocks"] and r["num_filters"]))
    lrs = sorted(set(r["learning_rate"] for r in all_completed if r["learning_rate"]))

    hp_summary = {
        "architectures_tested": archs,
        "learning_rates_tested": lrs,
    }

    return {
        "recent": [dict(r) for r in recent],
        "best_checkpoint": dict(best_ckpt) if best_ckpt else None,
        "best_benchmark": dict(best_bm) if best_bm else None,
        "frontier": frontier,
        "stability": stability,
        "opponents": [dict(o) for o in opponents],
        "stage": stage,
        "signals": signals,
        "hyperparams_summary": hp_summary,
    }


def _generate_signals(conn, recent, best_bm, stage, stability) -> list[dict]:
    """Rule-based signal generation."""
    signals = []

    # CLOSE_TO_PROMOTION
    if stage["gap_to_promotion"] is not None and stage["gap_to_promotion"] <= 0.05:
        signals.append({
            "type": "CLOSE_TO_PROMOTION", "severity": "high",
            "message": f"最佳 benchmark 胜率 {stage['best_benchmark_wr']:.1%}，"
                       f"距晋级阈值（{stage['promotion_threshold']:.0%}）仅差 {stage['gap_to_promotion']:.1%}",
            "suggestion": "专注可靠性提升，不冒险",
        })

    # WR_PLATEAU (last 3 completed runs, WR not improving)
    if len(recent) >= 3:
        last3_wr = [r["final_win_rate"] for r in recent[:3] if r["final_win_rate"] is not None]
        if len(last3_wr) >= 3:
            if max(last3_wr) - min(last3_wr) < 0.03:
                signals.append({
                    "type": "WR_PLATEAU", "severity": "medium",
                    "message": f"最近 3 次运行胜率无明显提升（{min(last3_wr):.1%} — {max(last3_wr):.1%}）",
                    "suggestion": "考虑更激进的架构变更，而非仅调超参",
                })

    # LOSS_DIVERGENCE
    if len(recent) >= 2:
        r0, r1 = recent[0], recent[1]
        if (r0["final_loss"] and r1["final_loss"] and
                r0["final_loss"] > r1["final_loss"] * 2):
            signals.append({
                "type": "LOSS_DIVERGENCE", "severity": "high",
                "message": f"最新运行 loss {r0['final_loss']:.3f} 是前次 {r1['final_loss']:.3f} 的 "
                           f"{r0['final_loss']/r1['final_loss']:.1f} 倍",
                "suggestion": "降低学习率或增大 buffer",
            })

    # INSUFFICIENT_BENCHMARKS
    bm_count = conn.execute(
        "SELECT COUNT(*) AS n FROM runs WHERE status='completed' AND is_benchmark=1"
    ).fetchone()["n"]
    if bm_count < 3:
        signals.append({
            "type": "INSUFFICIENT_BENCHMARKS", "severity": "medium",
            "message": f"仅 {bm_count} 次 benchmark 运行，前沿追踪不够可靠",
            "suggestion": "运行更多 benchmark 以建立可靠的前沿数据",
        })

    # ARCHITECTURE_PATTERN — check if one arch consistently outperforms
    arch_rows = conn.execute(
        "SELECT num_res_blocks, num_filters, final_win_rate FROM runs"
        " WHERE status='completed' AND final_win_rate IS NOT NULL"
    ).fetchall()
    arch_wrs: dict[str, list[float]] = {}
    for r in arch_rows:
        if r["num_res_blocks"] and r["num_filters"]:
            key = f"{r['num_res_blocks']}x{r['num_filters']}"
            arch_wrs.setdefault(key, []).append(r["final_win_rate"])
    if len(arch_wrs) >= 2:
        best_arch = max(arch_wrs, key=lambda k: sum(arch_wrs[k]) / len(arch_wrs[k]))
        best_mean = sum(arch_wrs[best_arch]) / len(arch_wrs[best_arch])
        others = {k: sum(v)/len(v) for k, v in arch_wrs.items() if k != best_arch}
        if others:
            second_best = max(others.values())
            if best_mean - second_best > 0.05:
                signals.append({
                    "type": "ARCHITECTURE_PATTERN", "severity": "info",
                    "message": f"{best_arch} 架构平均胜率 {best_mean:.1%} 明显优于其他架构",
                    "suggestion": f"在 {best_arch} 基础上增加容量（更深或更宽）",
                })

    # REGRESSION_WARNING
    if recent and best_bm:
        latest_wr = recent[0]["final_win_rate"]
        best_wr = best_bm["final_win_rate"]
        if latest_wr is not None and best_wr is not None and best_wr - latest_wr > 0.20:
            signals.append({
                "type": "REGRESSION_WARNING", "severity": "high",
                "message": f"最新运行胜率 {latest_wr:.1%} 远低于最佳 {best_wr:.1%}",
                "suggestion": "检查最近的代码变更，考虑回退",
            })

    return signals


def _format_report_md(data: dict) -> str:
    """Format report as Chinese markdown for human consumption."""
    lines = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    lines.append("# MAG-Gomoku 实验报告")
    lines.append(f"\n> 生成命令: `uv run python src/analyze.py --report`")
    lines.append(f"> 时间戳: {ts}")

    # Section 1: Recent Runs
    recent = data["recent"]
    lines.append("\n## 1. 近期训练运行")
    if recent:
        lines.append(f"\n| {'运行':>8} | {'模型':>5} | {'周期':>4} | {'对局':>5} | "
                     f"{'Loss':>7} | {'胜率':>6} | {'耗时':>6} | {'对手':>6} | {'类型':>10} |")
        lines.append(f"|{'-'*10}|{'-'*7}|{'-'*6}|{'-'*7}|{'-'*9}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*12}|")
        for r in recent:
            rid = r["id"][:8]
            model = f"{r['num_res_blocks'] or '?'}x{r['num_filters'] or '?'}"
            cyc = r["total_cycles"] or "-"
            gm = r["total_games"] or "-"
            loss = f"{r['final_loss']:.3f}" if r["final_loss"] else "-"
            wr = f"{r['final_win_rate']:.1%}" if r["final_win_rate"] is not None else "-"
            wt = f"{r['wall_time_s']:.0f}s" if r["wall_time_s"] else "-"
            opp = r["eval_opponent"] or f"L{r['eval_level'] or 0}"
            typ = "benchmark" if r["is_benchmark"] else "exploratory"
            lines.append(f"| {rid:>8} | {model:>5} | {str(cyc):>4} | {str(gm):>5} | "
                         f"{loss:>7} | {wr:>6} | {wt:>6} | {opp:>6} | {typ:>10} |")
    else:
        lines.append("\n无已完成的运行。")

    # Section 2: Best Checkpoint
    lines.append("\n## 2. 当前最佳检查点")
    bc = data["best_checkpoint"]
    if bc:
        opp = bc.get("eval_opponent") or f"L{bc.get('eval_level', 0)}"
        lines.append(f"\n- 运行: {bc['run_id'][:8]}, 标签: {bc['tag']}, "
                     f"胜率: {bc['win_rate']:.1%}, 周期: {bc['cycle']}, 对手: {opp}")
    bm = data["best_benchmark"]
    if bm:
        lines.append(f"- 历史最佳 benchmark: **{bm['final_win_rate']:.1%}**"
                     f"（运行 {bm['id'][:8]}，"
                     f"{bm['num_res_blocks']}x{bm['num_filters']}，"
                     f"{bm['wall_time_s']:.0f}s）")

    # Section 3: Frontier
    lines.append("\n## 3. 胜率前沿")
    if data["frontier"]:
        lines.append(f"\n| {'运行':>8} | {'标签':>16} | {'胜率':>6} | {'周期':>4} | {'对手':>6} |")
        lines.append(f"|{'-'*10}|{'-'*18}|{'-'*8}|{'-'*6}|{'-'*8}|")
        for f in data["frontier"]:
            rid = f["run_id"][:8]
            opp = f.get("eval_opponent") or f"L{f.get('eval_level', 0)}"
            lines.append(f"| {rid:>8} | {f['tag']:>16} | {f['win_rate']:>5.1%} | "
                         f"{f['cycle']:>4} | {opp:>6} |")

    # Section 4: Stability
    lines.append("\n## 4. 最新运行稳定性")
    stab = data["stability"]
    if stab["run_id"]:
        lines.append(f"\n运行: {stab['run_id'][:8]}")
        if stab["wr"]:
            w = stab["wr"]
            lines.append(f"\n胜率（{w['n']} 次探测）: 均值 {w['mean']:.1%}, "
                         f"标准差 {w['std']:.1%}, 范围 {w['min']:.1%}–{w['max']:.1%}")
        else:
            lines.append("\n胜率: 探测数据不足")
        if stab["loss"]:
            lo = stab["loss"]
            lines.append(f"Loss（{lo['n']} 周期）: 均值 {lo['mean']:.3f}, "
                         f"标准差 {lo['std']:.3f}, 下降 {lo['reduction_pct']:.0f}%")
    else:
        lines.append("\n无已完成的运行。")

    # Section 5: Opponents
    lines.append("\n## 5. 对手注册表")
    if data["opponents"]:
        lines.append(f"\n| {'别名':>6} | {'来源':>8} | {'标签':>16} | {'胜率':>6} | {'说明'} |")
        lines.append(f"|{'-'*8}|{'-'*10}|{'-'*18}|{'-'*8}|{'-'*20}|")
        for o in data["opponents"]:
            src = o["source_run"][:8] if o["source_run"] else "-"
            wr = f"{o['win_rate']:.1%}" if o["win_rate"] is not None else "-"
            lines.append(f"| {o['alias']:>6} | {src:>8} | {o['source_tag'] or '-':>16} | "
                         f"{wr:>6} | {o['description'] or ''} |")
    else:
        lines.append("\n无注册对手。")

    # Section 6: Stage
    lines.append("\n## 6. 阶段评估")
    s = data["stage"]
    lines.append(f"\n- 当前阶段: **Stage {s['current']}**（{s['opponent']}）")
    if s["best_benchmark_wr"] is not None:
        lines.append(f"- 最佳 benchmark 胜率: {s['best_benchmark_wr']:.1%}")
    if s["promotion_threshold"]:
        lines.append(f"- 晋级阈值: {s['promotion_threshold']:.0%} → {s['promotion_target']}")
    if s["gap_to_promotion"] is not None:
        lines.append(f"- 距晋级差距: **{s['gap_to_promotion']:.1%}**")

    # Section 7: Signals
    lines.append("\n## 7. 信号与观察")
    if data["signals"]:
        icons = {"high": "🔴", "medium": "🟡", "info": "🔵"}
        for sig in data["signals"]:
            icon = icons.get(sig["severity"], "⚪")
            lines.append(f"\n- {icon} **{sig['type']}**: {sig['message']}")
            lines.append(f"  → {sig['suggestion']}")
    else:
        lines.append("\n无特殊信号。一切正常。")

    return "\n".join(lines) + "\n"


def _format_report_json(data: dict) -> str:
    """Format report as structured JSON for agent consumption."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    recent_runs = []
    for r in data["recent"]:
        recent_runs.append({
            "run_id": r["id"][:8],
            "status": r["status"],
            "type": "benchmark" if r["is_benchmark"] else "exploratory",
            "hyperparams": {
                "num_res_blocks": r["num_res_blocks"],
                "num_filters": r["num_filters"],
                "num_params": r["num_params"],
                "learning_rate": r["learning_rate"],
                "steps_per_cycle": r["train_steps_per_cycle"],
                "replay_buffer_size": r["replay_buffer_size"],
                "parallel_games": r["parallel_games"],
                "seed": r["seed"],
            },
            "results": {
                "total_cycles": r["total_cycles"],
                "total_games": r["total_games"],
                "final_loss": round(r["final_loss"], 4) if r["final_loss"] else None,
                "final_win_rate": round(r["final_win_rate"], 4) if r["final_win_rate"] is not None else None,
                "wall_time_s": round(r["wall_time_s"], 1) if r["wall_time_s"] else None,
            },
            "eval": {
                "level": r["eval_level"],
                "opponent": r["eval_opponent"],
            },
            "sweep_tag": r["sweep_tag"],
        })

    bc = data["best_checkpoint"]
    best_checkpoint = None
    if bc:
        best_checkpoint = {
            "run_id": bc["run_id"][:8],
            "tag": bc["tag"],
            "win_rate": round(bc["win_rate"], 4) if bc["win_rate"] is not None else None,
            "eval_games": bc["eval_games"],
            "cycle": bc["cycle"],
            "num_params": bc["num_params"],
        }

    frontier_list = []
    for f in data["frontier"]:
        frontier_list.append({
            "run_id": f["run_id"][:8],
            "tag": f["tag"],
            "win_rate": round(f["win_rate"], 4),
            "cycle": f["cycle"],
        })

    stab = data["stability"]
    stability_out = {"run_id": stab["run_id"][:8] if stab["run_id"] else None}
    if stab["wr"]:
        w = stab["wr"]
        stability_out["win_rate"] = {
            "n": w["n"], "mean": round(w["mean"], 4), "std": round(w["std"], 4),
            "min": round(w["min"], 4), "max": round(w["max"], 4),
            "last_5": [round(x, 4) for x in w["last_5"]],
        }
    else:
        stability_out["win_rate"] = {"status": "insufficient_data"}
    if stab["loss"]:
        lo = stab["loss"]
        stability_out["loss"] = {
            "n": lo["n"], "mean": round(lo["mean"], 4), "std": round(lo["std"], 4),
            "min": round(lo["min"], 4), "max": round(lo["max"], 4),
            "reduction_pct": round(lo["reduction_pct"], 1),
            "last_5": [round(x, 4) for x in lo["last_5"]],
        }

    opponents_out = []
    for o in data["opponents"]:
        opponents_out.append({
            "alias": o["alias"],
            "source_run": o["source_run"][:8] if o["source_run"] else None,
            "source_tag": o["source_tag"],
            "win_rate": round(o["win_rate"], 4) if o["win_rate"] is not None else None,
            "description": o["description"],
        })

    report = {
        "report_version": "1.0",
        "generated_by": "analyze.py --report --format json",
        "timestamp": ts,
        "stage": data["stage"],
        "recent_runs": recent_runs,
        "best_checkpoint": best_checkpoint,
        "frontier": frontier_list,
        "stability": stability_out,
        "opponents": opponents_out,
        "hyperparams_summary": data["hyperparams_summary"],
        "signals": data["signals"],
    }

    # Round floats in stage
    if report["stage"]["best_benchmark_wr"] is not None:
        report["stage"]["best_benchmark_wr"] = round(report["stage"]["best_benchmark_wr"], 4)
    if report["stage"]["gap_to_promotion"] is not None:
        report["stage"]["gap_to_promotion"] = round(report["stage"]["gap_to_promotion"], 4)

    return _json.dumps(report, indent=2, ensure_ascii=False) + "\n"


def cmd_report(conn: sqlite3.Connection, n_recent: int = 5, fmt: str = "md") -> None:
    """Generate structured experiment report (dual format)."""
    data = _gather_report_data(conn, n_recent)
    if fmt == "json":
        print(_format_report_json(data), end="")
    else:
        print(_format_report_md(data), end="")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MAG-Gomoku training analysis (read-only)")
    parser.add_argument("--db", default=DB_PATH, help="Path to tracker.db")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs", action="store_true",
                       help="List all training runs")
    group.add_argument("--best", action="store_true",
                       help="Best checkpoint per run")
    group.add_argument("--frontier", action="store_true",
                       help="WR frontier across all runs")
    group.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                       help="Compare two runs side-by-side")
    group.add_argument("--lineage", metavar="RUN_ID",
                       help="Trace resume chain for a run")
    group.add_argument("--opponents", action="store_true",
                       help="List registered opponents")
    group.add_argument("--stability", metavar="RUN_ID",
                       help="Training stability report for a run")
    group.add_argument("--matrix", metavar="TAG_PREFIX",
                       help="Sweep matrix results grouped by tag prefix")
    group.add_argument("--report", action="store_true",
                       help="Generate structured experiment report for agent/human")

    parser.add_argument("--format", choices=["md", "json"], default="md",
                        help="Report format: md (Chinese markdown) or json (structured)")
    parser.add_argument("--recent", type=int, default=5,
                        help="Number of recent runs in report (default: 5)")

    args = parser.parse_args()
    conn = _connect()

    if args.runs:
        cmd_runs(conn)
    elif args.best:
        cmd_best(conn)
    elif args.frontier:
        cmd_frontier(conn)
    elif args.compare:
        cmd_compare(conn, args.compare[0], args.compare[1])
    elif args.lineage:
        cmd_lineage(conn, args.lineage)
    elif args.opponents:
        cmd_opponents(conn)
    elif args.stability:
        cmd_stability(conn, args.stability)
    elif args.matrix:
        cmd_matrix(conn, args.matrix)
    elif args.report:
        cmd_report(conn, n_recent=args.recent, fmt=args.format)

    conn.close()


if __name__ == "__main__":
    main()
