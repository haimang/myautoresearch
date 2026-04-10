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
"""

import argparse
import sqlite3
import sys
import os

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
                  is_benchmark, eval_opponent, resumed_from
           FROM runs ORDER BY started_at DESC"""
    ).fetchall()
    if not rows:
        print("No runs found.")
        return
    print(f"{'Run':>10}  {'Status':>11}  {'Cycles':>6}  {'Games':>6}  "
          f"{'Loss':>7}  {'WR':>6}  {'Time':>7}  {'Opp':>6}  {'Type':>12}")
    print("-" * 90)
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
        print(f"{rid:>10}  {st:>11}  {str(cyc):>6}  {str(gm):>6}  "
              f"{loss:>7}  {wr:>6}  {wt:>7}  {opp:>6}  {bm}{resumed}")


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
        ("Batch Size", "batch_size"),
        ("Parallel", "parallel_games"),
    ]

    print(f"{'':>16}  {a['id'][:8]:>14}  {b['id'][:8]:>14}")
    print("-" * 50)
    for label, key in fields:
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


def cmd_stability(conn: sqlite3.Connection, run_id: str) -> None:
    """Show training stability metrics for a run."""
    import math

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

    conn.close()


if __name__ == "__main__":
    main()
