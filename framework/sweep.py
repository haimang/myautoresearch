#!/usr/bin/env python3
"""autoresearch 超参数扫描工具 — 批量实验。

顺序执行一组训练配置的笛卡尔积，每次运行带唯一标签，
便于用 `analyze.py --matrix` 进行对比。

用法:
    # 单轴扫描（其他轴使用默认值）
    uv run python framework/sweep.py --train-script domains/gomoku/train.py \
        --num-filters 32,48,64 --time-budget 120 --seeds 42,137

    # 全矩阵（所有轴的笛卡尔积）
    uv run python framework/sweep.py --train-script domains/gomoku/train.py \
        --num-blocks 6,8 --num-filters 32,64 \
        --learning-rate 3e-4,5e-4 \
        --seeds 42,137 --time-budget 120 --tag v9-screen

    # 恢复中断的扫描（跳过已完成的 tag+seed 组合）
    uv run python framework/sweep.py --train-script domains/gomoku/train.py ... --resume
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
    p = argparse.ArgumentParser(description="autoresearch 超参数扫描工具")

    # 训练脚本路径 — 领域无关
    p.add_argument("--train-script", type=str, required=True,
                   help="领域训练脚本路径 (e.g. domains/gomoku/train.py)")

    # 扫描轴 — 逗号分隔值的笛卡尔积
    p.add_argument("--num-blocks", type=str, default=None,
                   help="逗号分隔的 num_blocks 值 (e.g. 6,8)")
    p.add_argument("--num-filters", type=str, default=None,
                   help="逗号分隔的 num_filters 值 (e.g. 32,48,64)")
    p.add_argument("--learning-rate", type=str, default=None,
                   help="逗号分隔的学习率 (e.g. 3e-4,5e-4,7e-4)")
    p.add_argument("--steps-per-cycle", type=str, default=None,
                   help="逗号分隔的每周期步数 (e.g. 20,30,40)")
    p.add_argument("--buffer-size", type=str, default=None,
                   help="逗号分隔的缓冲区大小 (e.g. 50000,100000)")

    # 固定参数，应用于每次运行
    p.add_argument("--time-budget", type=int, required=True,
                   help="每次运行的时间预算（秒）")
    p.add_argument("--seeds", type=str, default="42",
                   help="逗号分隔的随机种子 (默认: 42)")
    p.add_argument("--tag", type=str, default="sweep",
                   help="运行分组的标签前缀 (默认: sweep)")

    # 额外固定参数，透传至 train.py
    p.add_argument("--eval-opponent", type=str, default=None)
    p.add_argument("--parallel-games", type=int, default=None)
    p.add_argument("--target-win-rate", type=float, default=None)

    # 控制
    p.add_argument("--resume", action="store_true",
                   help="跳过 tracker.db 中已存在的 tag+seed 配置")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印矩阵，不实际运行")

    return p.parse_args()


def parse_csv(val, dtype):
    """解析逗号分隔字符串为指定类型的值列表。"""
    if val is None:
        return [None]
    return [dtype(v.strip()) for v in val.split(",")]


def get_completed_tags(db_path):
    """获取 tracker.db 中已完成的 sweep_tag 集合。"""
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
    """从扫描轴生成配置字典列表。"""
    axes = {}

    blocks = parse_csv(args.num_blocks, int)
    filters_ = parse_csv(args.num_filters, int)
    lrs = parse_csv(args.learning_rate, float)
    steps = parse_csv(args.steps_per_cycle, int)
    bufs = parse_csv(args.buffer_size, int)
    seeds = parse_csv(args.seeds, int)

    # 只包含显式指定的轴
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
        print("错误: 至少指定一个扫描轴 (--num-blocks, --num-filters, --learning-rate, --steps-per-cycle, --buffer-size)")
        sys.exit(1)

    configs = []
    for combo in itertools.product(*axis_values):
        cfg = dict(zip(axis_names, combo))
        for seed in seeds:
            # 生成描述性标签
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
    """执行单个训练配置。返回 (tag, 是否成功, 耗时)。"""
    tag = cfg["sweep_tag"]
    seed = cfg["seed"]

    cmd = [sys.executable, args.train_script,
           "--time-budget", str(args.time_budget),
           "--seed", str(seed),
           "--sweep-tag", tag]

    # 扫描轴参数
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

    # 固定透传参数
    if args.eval_opponent:
        cmd += ["--eval-opponent", args.eval_opponent]
    if args.parallel_games:
        cmd += ["--parallel-games", str(args.parallel_games)]
    if args.target_win_rate:
        cmd += ["--target-win-rate", str(args.target_win_rate)]

    # 运行头部信息
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
        print(f"  失败 (退出码 {proc.returncode})")
        if proc.stderr:
            # 显示 stderr 最后几行
            lines = proc.stderr.strip().split("\n")
            for line in lines[-5:]:
                print(f"  {line}")
        return tag, False, elapsed

    # 从 stdout 提取最终胜率
    wr_line = ""
    for line in proc.stdout.splitlines():
        if "Win rate:" in line or "win_rate" in line.lower():
            wr_line = line.strip()
    if wr_line:
        print(f"  {wr_line}")
    print(f"  完成，耗时 {elapsed:.0f}s")

    return tag, True, elapsed


def main():
    args = parse_args()
    configs = build_matrix(args)

    # 恢复: 跳过已完成的标签
    if args.resume:
        completed = get_completed_tags(DB_PATH)
        before = len(configs)
        configs = [c for c in configs if c["sweep_tag"] not in completed]
        skipped = before - len(configs)
        if skipped:
            print(f"恢复模式: 跳过 {skipped} 个已完成的配置")

    total = len(configs)
    if total == 0:
        print("无可运行配置（所有配置已完成或未生成配置）。")
        return

    print(f"扫描: {total} 个配置 × {args.time_budget}s = ~{total * args.time_budget / 60:.0f} 分钟")
    print(f"标签前缀: {args.tag}")

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

    # 汇总
    sweep_elapsed = time.time() - sweep_start
    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = total - n_ok

    print(f"\n{'='*60}")
    print(f"扫描完成: {n_ok} 成功, {n_fail} 失败, 总耗时 {sweep_elapsed:.0f}s")
    print(f"{'='*60}")

    if n_fail > 0:
        print("\n失败的运行:")
        for tag, ok, _ in results:
            if not ok:
                print(f"  {tag}")

    print(f"\n查看结果: uv run python framework/analyze.py --matrix {args.tag}")


if __name__ == "__main__":
    main()
