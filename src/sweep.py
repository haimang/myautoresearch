#!/usr/bin/env python3
"""MAG-Gomoku sweep tool — batch hyperparameter experiments.

Runs a matrix of training configurations sequentially, tagging each run
for easy comparison with `analyze.py --matrix`.

Usage:
    # Vary one axis at a time (others use defaults)
    uv run python src/sweep.py --num-filters 32,48,64 --time-budget 120 --seeds 42,137

    # Full matrix (cartesian product of all axes)
    uv run python src/sweep.py \
        --num-blocks 6,8 --num-filters 32,64 \
        --learning-rate 3e-4,5e-4 \
        --seeds 42,137 --time-budget 120 --tag v9-screen

    # Resume interrupted sweep (skips completed tag+seed combos)
    uv run python src/sweep.py ... --resume
"""

import argparse
import itertools
import json
import os
import sqlite3
import subprocess
import sys
import time


DB_PATH = "output/tracker.db"


def parse_args():
    p = argparse.ArgumentParser(description="MAG-Gomoku Sweep Tool")

    # Sweep axes — comma-separated values form the cartesian product
    p.add_argument("--num-blocks", type=str, default=None,
                   help="Comma-separated num_blocks values (e.g. 6,8)")
    p.add_argument("--num-filters", type=str, default=None,
                   help="Comma-separated num_filters values (e.g. 32,48,64)")
    p.add_argument("--learning-rate", type=str, default=None,
                   help="Comma-separated learning rates (e.g. 3e-4,5e-4,7e-4)")
    p.add_argument("--steps-per-cycle", type=str, default=None,
                   help="Comma-separated steps per cycle (e.g. 20,30,40)")
    p.add_argument("--buffer-size", type=str, default=None,
                   help="Comma-separated buffer sizes (e.g. 50000,100000)")

    # Fixed params applied to every run
    p.add_argument("--time-budget", type=int, required=True,
                   help="Time budget per run (seconds)")
    p.add_argument("--seeds", type=str, default="42",
                   help="Comma-separated seeds (default: 42)")
    p.add_argument("--tag", type=str, default="sweep",
                   help="Tag prefix for grouping runs (default: sweep)")

    # Extra fixed args passed through to train.py
    p.add_argument("--eval-opponent", type=str, default=None)
    p.add_argument("--parallel-games", type=int, default=None)
    p.add_argument("--target-win-rate", type=float, default=None)

    # Control
    p.add_argument("--resume", action="store_true",
                   help="Skip configs whose tag+seed already exist in tracker.db")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the matrix without running")

    return p.parse_args()


def parse_csv(val, dtype):
    """Parse comma-separated string to list of typed values."""
    if val is None:
        return [None]
    return [dtype(v.strip()) for v in val.split(",")]


def get_completed_tags(db_path):
    """Get set of sweep_tag values already completed in tracker.db."""
    if not os.path.exists(db_path):
        return set()
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT sweep_tag FROM runs WHERE status IN ('completed', 'time_budget', 'target_win_rate', 'target_games') AND sweep_tag IS NOT NULL"
        ).fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()


def build_matrix(args):
    """Build list of config dicts from sweep axes."""
    axes = {}

    blocks = parse_csv(args.num_blocks, int)
    filters_ = parse_csv(args.num_filters, int)
    lrs = parse_csv(args.learning_rate, float)
    steps = parse_csv(args.steps_per_cycle, int)
    bufs = parse_csv(args.buffer_size, int)
    seeds = parse_csv(args.seeds, int)

    # Only include axes that were explicitly specified
    axis_names = []
    axis_values = []

    if args.num_blocks is not None:
        axis_names.append("num_blocks")
        axis_values.append(blocks)
    if args.num_filters is not None:
        axis_names.append("num_filters")
        axis_values.append(filters_)
    if args.learning_rate is not None:
        axis_names.append("learning_rate")
        axis_values.append(lrs)
    if args.steps_per_cycle is not None:
        axis_names.append("steps_per_cycle")
        axis_values.append(steps)
    if args.buffer_size is not None:
        axis_names.append("buffer_size")
        axis_values.append(bufs)

    if not axis_names:
        print("Error: specify at least one sweep axis (--num-blocks, --num-filters, --learning-rate, --steps-per-cycle, --buffer-size)")
        sys.exit(1)

    configs = []
    for combo in itertools.product(*axis_values):
        cfg = dict(zip(axis_names, combo))
        for seed in seeds:
            # Build a descriptive tag
            parts = [args.tag]
            for name, val in cfg.items():
                short = {"num_blocks": "b", "num_filters": "f",
                         "learning_rate": "lr", "steps_per_cycle": "s",
                         "buffer_size": "buf"}[name]
                parts.append(f"{short}{val}")
            parts.append(f"sd{seed}")
            tag = "_".join(parts)

            configs.append({**cfg, "seed": seed, "sweep_tag": tag})

    return configs


def run_one(cfg, args, idx, total):
    """Run a single training config. Returns (tag, success, elapsed)."""
    tag = cfg["sweep_tag"]
    seed = cfg["seed"]

    cmd = [sys.executable, "src/train.py",
           "--time-budget", str(args.time_budget),
           "--seed", str(seed),
           "--sweep-tag", tag]

    # Sweep axis params
    if "num_blocks" in cfg and cfg["num_blocks"] is not None:
        cmd += ["--num-blocks", str(cfg["num_blocks"])]
    if "num_filters" in cfg and cfg["num_filters"] is not None:
        cmd += ["--num-filters", str(cfg["num_filters"])]
    if "learning_rate" in cfg and cfg["learning_rate"] is not None:
        cmd += ["--learning-rate", str(cfg["learning_rate"])]
    if "steps_per_cycle" in cfg and cfg["steps_per_cycle"] is not None:
        cmd += ["--steps-per-cycle", str(cfg["steps_per_cycle"])]
    if "buffer_size" in cfg and cfg["buffer_size"] is not None:
        cmd += ["--buffer-size", str(cfg["buffer_size"])]

    # Fixed passthrough params
    if args.eval_opponent:
        cmd += ["--eval-opponent", args.eval_opponent]
    if args.parallel_games:
        cmd += ["--parallel-games", str(args.parallel_games)]
    if args.target_win_rate:
        cmd += ["--target-win-rate", str(args.target_win_rate)]

    # Header
    axis_desc = "  ".join(f"{k}={v}" for k, v in cfg.items()
                          if k not in ("seed", "sweep_tag"))
    print(f"\n{'='*60}")
    print(f"[{idx}/{total}] {tag}")
    print(f"  {axis_desc}  seed={seed}")
    print(f"{'='*60}")

    t0 = time.time()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode})")
        if proc.stderr:
            # Show last few lines of stderr
            lines = proc.stderr.strip().split("\n")
            for line in lines[-5:]:
                print(f"  {line}")
        return tag, False, elapsed

    # Extract final win rate from stdout
    wr_line = ""
    for line in proc.stdout.splitlines():
        if "Win rate:" in line or "win_rate" in line.lower():
            wr_line = line.strip()
    if wr_line:
        print(f"  {wr_line}")
    print(f"  Completed in {elapsed:.0f}s")

    return tag, True, elapsed


def main():
    args = parse_args()
    configs = build_matrix(args)

    # Resume: skip completed tags
    if args.resume:
        completed = get_completed_tags(DB_PATH)
        before = len(configs)
        configs = [c for c in configs if c["sweep_tag"] not in completed]
        skipped = before - len(configs)
        if skipped:
            print(f"Resume: skipping {skipped} already-completed configs")

    total = len(configs)
    if total == 0:
        print("Nothing to run (all configs already completed or no configs generated).")
        return

    print(f"Sweep: {total} configs × {args.time_budget}s each = ~{total * args.time_budget / 60:.0f} min total")
    print(f"Tag prefix: {args.tag}")

    if args.dry_run:
        print(f"\n{'Tag':<55} {'Params'}")
        print("-" * 80)
        for c in configs:
            axis_desc = "  ".join(f"{k}={v}" for k, v in c.items()
                                  if k not in ("seed", "sweep_tag"))
            print(f"{c['sweep_tag']:<55} {axis_desc}")
        return

    results = []
    sweep_start = time.time()

    for i, cfg in enumerate(configs, 1):
        tag, ok, elapsed = run_one(cfg, args, i, total)
        results.append((tag, ok, elapsed))

    # Summary
    sweep_elapsed = time.time() - sweep_start
    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = total - n_ok

    print(f"\n{'='*60}")
    print(f"Sweep complete: {n_ok} succeeded, {n_fail} failed, {sweep_elapsed:.0f}s total")
    print(f"{'='*60}")

    if n_fail > 0:
        print("\nFailed runs:")
        for tag, ok, _ in results:
            if not ok:
                print(f"  {tag}")

    print(f"\nView results: uv run python src/analyze.py --matrix {args.tag}")


if __name__ == "__main__":
    main()
