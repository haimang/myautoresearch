"""Experiment report assembly and rendering."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone

STAGES = [
    {"level": 0, "opponent": "L0 (random)", "threshold": 0.95, "next": "L1 (minimax depth 2)"},
    {"level": 1, "opponent": "L1 (minimax depth 2)", "threshold": 0.80, "next": "L2 (minimax depth 4)"},
    {"level": 2, "opponent": "L2 (minimax depth 4)", "threshold": 0.60, "next": "L3 (minimax depth 6)"},
    {"level": 3, "opponent": "L3 (minimax depth 6)", "threshold": None, "next": None},
]


def generate_signals(conn, recent, best_bm, stage, stability) -> list[dict]:
    signals = []
    if stage["gap_to_promotion"] is not None and stage["gap_to_promotion"] <= 0.05:
        signals.append({
            "type": "CLOSE_TO_PROMOTION", "severity": "high",
            "message": f"最佳 benchmark 胜率 {stage['best_benchmark_wr']:.1%}，距晋级阈值（{stage['promotion_threshold']:.0%}）仅差 {stage['gap_to_promotion']:.1%}",
            "suggestion": "专注可靠性提升，不冒险",
        })
    if len(recent) >= 3:
        last3_wr = [r["final_win_rate"] for r in recent[:3] if r["final_win_rate"] is not None]
        if len(last3_wr) >= 3 and max(last3_wr) - min(last3_wr) < 0.03:
            signals.append({
                "type": "WR_PLATEAU", "severity": "medium",
                "message": f"最近 3 次运行胜率无明显提升（{min(last3_wr):.1%} — {max(last3_wr):.1%}）",
                "suggestion": "考虑更激进的架构变更，而非仅调超参",
            })
    if len(recent) >= 2:
        r0, r1 = recent[0], recent[1]
        if r0["final_loss"] and r1["final_loss"] and r0["final_loss"] > r1["final_loss"] * 2:
            signals.append({
                "type": "LOSS_DIVERGENCE", "severity": "high",
                "message": f"最新运行 loss {r0['final_loss']:.3f} 是前次 {r1['final_loss']:.3f} 的 {r0['final_loss']/r1['final_loss']:.1f} 倍",
                "suggestion": "降低学习率或增大 buffer",
            })
    bm_count = conn.execute(
        "SELECT COUNT(*) AS n FROM runs WHERE status='completed' AND is_benchmark=1"
    ).fetchone()["n"]
    if bm_count < 3:
        signals.append({
            "type": "INSUFFICIENT_BENCHMARKS", "severity": "medium",
            "message": f"仅 {bm_count} 次 benchmark 运行，前沿追踪不够可靠",
            "suggestion": "运行更多 benchmark 以建立可靠的前沿数据",
        })
    arch_rows = conn.execute(
        "SELECT num_res_blocks, num_filters, final_win_rate FROM runs WHERE status='completed' AND final_win_rate IS NOT NULL"
    ).fetchall()
    arch_wrs: dict[str, list[float]] = {}
    for row in arch_rows:
        if row["num_res_blocks"] and row["num_filters"]:
            key = f"{row['num_res_blocks']}x{row['num_filters']}"
            arch_wrs.setdefault(key, []).append(row["final_win_rate"])
    if len(arch_wrs) >= 2:
        best_arch = max(arch_wrs, key=lambda k: sum(arch_wrs[k]) / len(arch_wrs[k]))
        best_mean = sum(arch_wrs[best_arch]) / len(arch_wrs[best_arch])
        others = {k: sum(v) / len(v) for k, v in arch_wrs.items() if k != best_arch}
        if others and best_mean - max(others.values()) > 0.05:
            signals.append({
                "type": "ARCHITECTURE_PATTERN", "severity": "info",
                "message": f"{best_arch} 架构平均胜率 {best_mean:.1%} 明显优于其他架构",
                "suggestion": f"在 {best_arch} 基础上增加容量（更深或更宽）",
            })
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


def gather_report_data(conn: sqlite3.Connection, n_recent: int = 5) -> dict:
    recent = conn.execute(
        "SELECT id, status, started_at, total_cycles, total_games, total_steps,"
        " final_loss, final_win_rate, wall_time_s, eval_level, eval_opponent,"
        " is_benchmark, num_res_blocks, num_filters, num_params,"
        " learning_rate, train_steps_per_cycle, replay_buffer_size,"
        " parallel_games, seed, sweep_tag"
        " FROM runs WHERE status='completed'"
        " ORDER BY started_at DESC LIMIT ?", (n_recent,)
    ).fetchall()
    best_ckpt = conn.execute(
        "SELECT c.run_id, c.tag, c.win_rate, c.eval_games, c.cycle,"
        " c.num_params, r.eval_level, r.eval_opponent"
        " FROM checkpoints c JOIN runs r ON c.run_id = r.id"
        " ORDER BY c.win_rate DESC LIMIT 1"
    ).fetchone()
    best_bm = conn.execute(
        "SELECT id, final_win_rate, num_res_blocks, num_filters, wall_time_s"
        " FROM runs WHERE status='completed' AND is_benchmark=1"
        " ORDER BY final_win_rate DESC LIMIT 1"
    ).fetchone()
    all_ckpts = conn.execute(
        "SELECT c.run_id, c.tag, c.win_rate, c.created_at, c.cycle,"
        " r.eval_level, r.eval_opponent"
        " FROM checkpoints c JOIN runs r ON c.run_id = r.id"
        " ORDER BY c.created_at ASC"
    ).fetchall()
    frontier = []
    best_wr = -1.0
    for ckpt in all_ckpts:
        if ckpt["win_rate"] is not None and ckpt["win_rate"] > best_wr:
            best_wr = ckpt["win_rate"]
            frontier.append(dict(ckpt))

    latest_run = conn.execute(
        "SELECT id, total_cycles FROM runs WHERE status='completed' ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    stability = {"run_id": None, "wr": None, "loss": None}
    if latest_run:
        lid = latest_run["id"]
        stability["run_id"] = lid
        wr_rows = conn.execute(
            "SELECT cycle, win_rate FROM cycle_metrics WHERE run_id=? AND win_rate IS NOT NULL ORDER BY cycle", (lid,)
        ).fetchall()
        if len(wr_rows) >= 3:
            wrs = [row["win_rate"] for row in wr_rows]
            mean_wr = sum(wrs) / len(wrs)
            stability["wr"] = {
                "n": len(wrs),
                "mean": mean_wr,
                "std": math.sqrt(sum((value - mean_wr) ** 2 for value in wrs) / len(wrs)),
                "min": min(wrs),
                "max": max(wrs),
                "last_5": wrs[-5:],
            }
        loss_rows = conn.execute(
            "SELECT cycle, loss FROM cycle_metrics WHERE run_id=? AND loss IS NOT NULL ORDER BY cycle", (lid,)
        ).fetchall()
        if loss_rows:
            losses = [row["loss"] for row in loss_rows]
            n = len(losses)
            mean_l = sum(losses) / n
            std_l = math.sqrt(sum((value - mean_l) ** 2 for value in losses) / n) if n > 1 else 0
            q = max(n // 4, 1)
            first_q = sum(losses[:q]) / q
            last_q = sum(losses[-q:]) / q
            stability["loss"] = {
                "n": n,
                "mean": mean_l,
                "std": std_l,
                "min": min(losses),
                "max": max(losses),
                "reduction_pct": (first_q - last_q) / first_q * 100 if first_q > 0 else 0,
                "last_5": losses[-5:],
            }

    try:
        opponents = conn.execute(
            "SELECT alias, source_run, source_tag, win_rate, description FROM opponents ORDER BY created_at"
        ).fetchall()
    except sqlite3.OperationalError:
        opponents = []

    cur_level = recent[0]["eval_level"] or 0 if recent else 0
    stage_info = STAGES[min(cur_level, len(STAGES) - 1)]
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
    signals = generate_signals(conn, recent, best_bm, stage, stability)

    all_completed = conn.execute(
        "SELECT num_res_blocks, num_filters, learning_rate FROM runs WHERE status='completed'"
    ).fetchall()
    hp_summary = {
        "architectures_tested": sorted(set(
            f"{row['num_res_blocks']}x{row['num_filters']}"
            for row in all_completed if row["num_res_blocks"] and row["num_filters"]
        )),
        "learning_rates_tested": sorted(set(row["learning_rate"] for row in all_completed if row["learning_rate"])),
    }
    return {
        "recent": [dict(row) for row in recent],
        "best_checkpoint": dict(best_ckpt) if best_ckpt else None,
        "best_benchmark": dict(best_bm) if best_bm else None,
        "frontier": frontier,
        "stability": stability,
        "opponents": [dict(row) for row in opponents],
        "stage": stage,
        "signals": signals,
        "hyperparams_summary": hp_summary,
    }


def format_report_md(data: dict) -> str:
    lines = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    lines.append("# 实验报告")
    lines.append(f"\n> 生成命令: `uv run python framework/analyze.py --report`")
    lines.append(f"> 时间戳: {ts}")
    recent = data["recent"]
    lines.append("\n## 1. 近期训练运行")
    if recent:
        lines.append(f"\n| {'运行':>8} | {'模型':>5} | {'周期':>4} | {'对局':>5} | {'Loss':>7} | {'胜率':>6} | {'耗时':>6} | {'对手':>6} | {'类型':>10} |")
        lines.append(f"|{'-'*10}|{'-'*7}|{'-'*6}|{'-'*7}|{'-'*9}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*12}|")
        for row in recent:
            rid = row["id"][:8]
            model = f"{row['num_res_blocks'] or '?'}x{row['num_filters'] or '?'}"
            cyc = row["total_cycles"] or "-"
            gm = row["total_games"] or "-"
            loss = f"{row['final_loss']:.3f}" if row["final_loss"] else "-"
            wr = f"{row['final_win_rate']:.1%}" if row["final_win_rate"] is not None else "-"
            wt = f"{row['wall_time_s']:.0f}s" if row["wall_time_s"] else "-"
            opp = row["eval_opponent"] or f"L{row['eval_level'] or 0}"
            typ = "benchmark" if row["is_benchmark"] else "exploratory"
            lines.append(f"| {rid:>8} | {model:>5} | {str(cyc):>4} | {str(gm):>5} | {loss:>7} | {wr:>6} | {wt:>6} | {opp:>6} | {typ:>10} |")
    else:
        lines.append("\n无已完成的运行。")
    lines.append("\n## 2. 当前最佳检查点")
    best_checkpoint = data["best_checkpoint"]
    if best_checkpoint:
        opp = best_checkpoint.get("eval_opponent") or f"L{best_checkpoint.get('eval_level', 0)}"
        lines.append(f"\n- 运行: {best_checkpoint['run_id'][:8]}, 标签: {best_checkpoint['tag']}, 胜率: {best_checkpoint['win_rate']:.1%}, 周期: {best_checkpoint['cycle']}, 对手: {opp}")
    best_benchmark = data["best_benchmark"]
    if best_benchmark:
        lines.append(f"- 历史最佳 benchmark: **{best_benchmark['final_win_rate']:.1%}**（运行 {best_benchmark['id'][:8]}，{best_benchmark['num_res_blocks']}x{best_benchmark['num_filters']}，{best_benchmark['wall_time_s']:.0f}s）")
    lines.append("\n## 3. 胜率前沿")
    if data["frontier"]:
        lines.append(f"\n| {'运行':>8} | {'标签':>16} | {'胜率':>6} | {'周期':>4} | {'对手':>6} |")
        lines.append(f"|{'-'*10}|{'-'*18}|{'-'*8}|{'-'*6}|{'-'*8}|")
        for frontier in data["frontier"]:
            rid = frontier["run_id"][:8]
            opp = frontier.get("eval_opponent") or f"L{frontier.get('eval_level', 0)}"
            lines.append(f"| {rid:>8} | {frontier['tag']:>16} | {frontier['win_rate']:>5.1%} | {frontier['cycle']:>4} | {opp:>6} |")
    lines.append("\n## 4. 最新运行稳定性")
    stability = data["stability"]
    if stability["run_id"]:
        lines.append(f"\n运行: {stability['run_id'][:8]}")
        if stability["wr"]:
            wr = stability["wr"]
            lines.append(f"\n胜率（{wr['n']} 次探测）: 均值 {wr['mean']:.1%}, 标准差 {wr['std']:.1%}, 范围 {wr['min']:.1%}–{wr['max']:.1%}")
        else:
            lines.append("\n胜率: 探测数据不足")
        if stability["loss"]:
            loss = stability["loss"]
            lines.append(f"Loss（{loss['n']} 周期）: 均值 {loss['mean']:.3f}, 标准差 {loss['std']:.3f}, 下降 {loss['reduction_pct']:.0f}%")
    else:
        lines.append("\n无已完成的运行。")
    lines.append("\n## 5. 对手注册表")
    if data["opponents"]:
        lines.append(f"\n| {'别名':>6} | {'来源':>8} | {'标签':>16} | {'胜率':>6} | {'说明'} |")
        lines.append(f"|{'-'*8}|{'-'*10}|{'-'*18}|{'-'*8}|{'-'*20}|")
        for row in data["opponents"]:
            src = row["source_run"][:8] if row["source_run"] else "-"
            wr = f"{row['win_rate']:.1%}" if row["win_rate"] is not None else "-"
            lines.append(f"| {row['alias']:>6} | {src:>8} | {row['source_tag'] or '-':>16} | {wr:>6} | {row['description'] or ''} |")
    else:
        lines.append("\n无注册对手。")
    lines.append("\n## 6. 阶段评估")
    stage = data["stage"]
    lines.append(f"\n- 当前阶段: **Stage {stage['current']}**（{stage['opponent']}）")
    if stage["best_benchmark_wr"] is not None:
        lines.append(f"- 最佳 benchmark 胜率: {stage['best_benchmark_wr']:.1%}")
    if stage["promotion_threshold"]:
        lines.append(f"- 晋级阈值: {stage['promotion_threshold']:.0%} → {stage['promotion_target']}")
    if stage["gap_to_promotion"] is not None:
        lines.append(f"- 距晋级差距: **{stage['gap_to_promotion']:.1%}**")
    lines.append("\n## 7. 信号与观察")
    if data["signals"]:
        icons = {"high": "🔴", "medium": "🟡", "info": "🔵"}
        for sig in data["signals"]:
            lines.append(f"\n- {icons.get(sig['severity'], '⚪')} **{sig['type']}**: {sig['message']}")
            lines.append(f"  → {sig['suggestion']}")
    else:
        lines.append("\n无特殊信号。一切正常。")
    return "\n".join(lines) + "\n"


def format_report_json(data: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    recent_runs = []
    for row in data["recent"]:
        recent_runs.append({
            "run_id": row["id"][:8],
            "status": row["status"],
            "type": "benchmark" if row["is_benchmark"] else "exploratory",
            "hyperparams": {
                "num_res_blocks": row["num_res_blocks"],
                "num_filters": row["num_filters"],
                "num_params": row["num_params"],
                "learning_rate": row["learning_rate"],
                "steps_per_cycle": row["train_steps_per_cycle"],
                "replay_buffer_size": row["replay_buffer_size"],
                "parallel_games": row["parallel_games"],
                "seed": row["seed"],
            },
            "results": {
                "total_cycles": row["total_cycles"],
                "total_games": row["total_games"],
                "final_loss": round(row["final_loss"], 4) if row["final_loss"] else None,
                "final_win_rate": round(row["final_win_rate"], 4) if row["final_win_rate"] is not None else None,
                "wall_time_s": round(row["wall_time_s"], 1) if row["wall_time_s"] else None,
            },
            "eval": {"level": row["eval_level"], "opponent": row["eval_opponent"]},
            "sweep_tag": row["sweep_tag"],
        })
    best_checkpoint = None
    if data["best_checkpoint"]:
        best_checkpoint = {
            "run_id": data["best_checkpoint"]["run_id"][:8],
            "tag": data["best_checkpoint"]["tag"],
            "win_rate": round(data["best_checkpoint"]["win_rate"], 4) if data["best_checkpoint"]["win_rate"] is not None else None,
            "eval_games": data["best_checkpoint"]["eval_games"],
            "cycle": data["best_checkpoint"]["cycle"],
            "num_params": data["best_checkpoint"]["num_params"],
        }
    stability_out = {"run_id": data["stability"]["run_id"][:8] if data["stability"]["run_id"] else None}
    if data["stability"]["wr"]:
        wr = data["stability"]["wr"]
        stability_out["win_rate"] = {
            "n": wr["n"], "mean": round(wr["mean"], 4), "std": round(wr["std"], 4),
            "min": round(wr["min"], 4), "max": round(wr["max"], 4),
            "last_5": [round(x, 4) for x in wr["last_5"]],
        }
    else:
        stability_out["win_rate"] = {"status": "insufficient_data"}
    if data["stability"]["loss"]:
        loss = data["stability"]["loss"]
        stability_out["loss"] = {
            "n": loss["n"], "mean": round(loss["mean"], 4), "std": round(loss["std"], 4),
            "min": round(loss["min"], 4), "max": round(loss["max"], 4),
            "reduction_pct": round(loss["reduction_pct"], 1),
            "last_5": [round(x, 4) for x in loss["last_5"]],
        }
    report = {
        "report_version": "1.0",
        "generated_by": "framework/analyze.py --report --format json",
        "timestamp": ts,
        "stage": data["stage"],
        "recent_runs": recent_runs,
        "best_checkpoint": best_checkpoint,
        "frontier": [
            {"run_id": row["run_id"][:8], "tag": row["tag"], "win_rate": round(row["win_rate"], 4), "cycle": row["cycle"]}
            for row in data["frontier"]
        ],
        "stability": stability_out,
        "opponents": [
            {
                "alias": row["alias"],
                "source_run": row["source_run"][:8] if row["source_run"] else None,
                "source_tag": row["source_tag"],
                "win_rate": round(row["win_rate"], 4) if row["win_rate"] is not None else None,
                "description": row["description"],
            }
            for row in data["opponents"]
        ],
        "hyperparams_summary": data["hyperparams_summary"],
        "signals": data["signals"],
    }
    if report["stage"]["best_benchmark_wr"] is not None:
        report["stage"]["best_benchmark_wr"] = round(report["stage"]["best_benchmark_wr"], 4)
    if report["stage"]["gap_to_promotion"] is not None:
        report["stage"]["gap_to_promotion"] = round(report["stage"]["gap_to_promotion"], 4)
    return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
