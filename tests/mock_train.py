#!/usr/bin/env python3
"""Mock train script for branch execute integration tests."""
import argparse
import json
import os
import sys
import tempfile
import uuid

# Find project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from framework.core.db import create_run, finish_run, init_db


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--resume-checkpoint-tag", type=str, default=None)
    p.add_argument("--sweep-tag", type=str, default=None)
    p.add_argument("--time-budget", type=int, default=60)
    p.add_argument("--db", type=str, default="output/tracker.db")
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--num-filters", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--steps-per-cycle", type=int, default=100)
    p.add_argument("--buffer-size", type=int, default=10000)
    p.add_argument("--mcts-sims", type=int, default=100)
    p.add_argument("--eval-level", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--fail", action="store_true", help="Simulate failure")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fail:
        print("Simulated train failure", file=sys.stderr)
        sys.exit(1)

    run_id = str(uuid.uuid4()).replace("-", "")[:16]
    db_path = args.db or "output/tracker.db"
    conn = init_db(db_path)
    create_run(conn, run_id, {
        "sweep_tag": args.sweep_tag,
        "eval_level": args.eval_level,
        "learning_rate": args.learning_rate,
        "num_res_blocks": args.num_blocks,
        "num_filters": args.num_filters,
        "seed": args.seed,
    }, is_benchmark=True)
    finish_run(conn, run_id, {
        "status": "completed",
        "final_win_rate": 0.75,
        "wall_time_s": float(args.time_budget),
        "num_params": 100000,
        "total_games": 500,
    })
    conn.close()
    print(f"Win rate: 75.0%")
    print(f"Run completed: {run_id}")


if __name__ == "__main__":
    main()
