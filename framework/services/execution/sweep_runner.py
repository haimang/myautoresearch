"""Sweep execution helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from framework.profiles.search_space import describe_profile


def run_one(cfg, args, idx, total, campaign=None):
    """执行单个训练配置。返回 (tag, 是否成功, 耗时)。"""
    tag = cfg["sweep_tag"]
    seed = cfg["seed"]
    cmd = [sys.executable, args.train_script, "--time-budget", str(args.time_budget), "--seed", str(seed), "--sweep-tag", tag]
    if args.db:
        cmd += ["--db", args.db]
    if campaign and campaign.get("objective_profile_id"):
        cmd += ["--campaign-id", campaign["id"]]
    if getattr(args, "run_id", None):
        cmd += ["--run-id", args.run_id]
    if getattr(args, "_artifact_root", None):
        cmd += ["--artifact-root", args._artifact_root]
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
    if cfg.get("__candidate_payload") is not None:
        cmd += ["--candidate-json", json.dumps(cfg["__candidate_payload"], ensure_ascii=False, sort_keys=True)]
    if args.eval_level is not None:
        cmd += ["--eval-level", str(args.eval_level)]
    if args.eval_opponent:
        cmd += ["--eval-opponent", args.eval_opponent]
    if args.parallel_games:
        cmd += ["--parallel-games", str(args.parallel_games)]
    if args.target_win_rate:
        cmd += ["--target-win-rate", str(args.target_win_rate)]

    axis_desc = "  ".join(f"{k}={v}" for k, v in cfg.items() if not k.startswith("__") and k not in ("seed", "sweep_tag"))
    print(f"\n{'='*60}")
    print(f"[{idx}/{total}] {tag}")
    print(f"  {axis_desc}  seed={seed}")
    print(f"{'='*60}")

    t0 = time.time()
    proc = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"}, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  失败 (退出码 {proc.returncode})")
        if proc.stderr:
            for line in proc.stderr.strip().split("\n")[-5:]:
                print(f"  {line}")
        return tag, False, elapsed
    for line in proc.stdout.splitlines():
        if "Win rate:" in line or "win_rate" in line.lower():
            print(f"  {line.strip()}")
    print(f"  完成，耗时 {elapsed:.0f}s")
    return tag, True, elapsed


def print_matrix_preview(configs, profile: dict | None, args, protocol: dict):
    if profile is not None:
        print(describe_profile(profile))
        print()
    if args.campaign:
        print(f"Campaign: {args.campaign}")
        print(f"Protocol: {json.dumps(protocol, ensure_ascii=False, sort_keys=True)}")
        print()
    print(f"{'Tag':<55} {'Params'}")
    print("-" * 96)
    for cfg in configs:
        axis_desc = "  ".join(f"{k}={v}" for k, v in cfg.items() if not k.startswith("__") and k not in ("seed", "sweep_tag"))
        print(f"{cfg['sweep_tag']:<55} {axis_desc}")
