#!/usr/bin/env python3
"""autoresearch 训练分析工具（只读）。

查询 SQLite 追踪数据库，报告训练运行、检查点、对手和胜率进展。

用法:
    uv run python framework/index.py analyze --runs
    uv run python framework/index.py analyze --best
    uv run python framework/index.py analyze --frontier
    uv run python framework/index.py analyze --compare RUN_A RUN_B
    uv run python framework/index.py analyze --report
    uv run python framework/index.py analyze --report --format json
"""

import argparse
import json as _json
import math
import os
import sqlite3
import sys
import uuid

from framework.core.db import DB_PATH, get_campaign, init_db
from framework.services.frontier.pareto import compute_knee_point as service_compute_knee_point, pareto_front as service_pareto_front
from framework.services.frontier.snapshots import save_frontier_snapshot as service_save_frontier_snapshot
from framework.services.reporting.experiment_report import (
    format_report_json as service_format_report_json,
    format_report_md as service_format_report_md,
    gather_report_data as service_gather_report_data,
)


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open tracker.db via core.db (shared connection + migration logic)."""
    if not os.path.exists(db_path):
        print(f"Error: database not found at {db_path}")
        sys.exit(1)
    return init_db(db_path)


def _resolve_campaign_or_exit(conn: sqlite3.Connection, campaign: str) -> dict:
    row = get_campaign(conn, campaign)
    if not row:
        print(f"Campaign not found: {campaign}")
        sys.exit(1)
    data = dict(row)
    data["protocol"] = _json.loads(data["protocol_json"])
    return data


def _campaign_run_rows(conn: sqlite3.Connection, campaign_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT r.*, cr.stage, cr.axis_values_json
           FROM campaign_runs cr
           JOIN runs r ON r.id = cr.run_id
           WHERE cr.campaign_id = ?
           ORDER BY r.started_at""",
        (campaign_id,),
    ).fetchall()


def _campaign_drift(rows: list[sqlite3.Row], protocol: dict) -> list[dict]:
    drift = []
    for row in rows:
        reasons = []
        if "eval_level" in protocol and protocol["eval_level"] != row["eval_level"]:
            reasons.append(f"eval_level={row['eval_level']} != {protocol['eval_level']}")
        if "eval_opponent" in protocol and protocol["eval_opponent"] != row["eval_opponent"]:
            reasons.append(f"eval_opponent={row['eval_opponent']!r} != {protocol['eval_opponent']!r}")
        if "is_benchmark" in protocol and bool(protocol["is_benchmark"]) != bool(row["is_benchmark"]):
            reasons.append(
                f"is_benchmark={bool(row['is_benchmark'])} != {bool(protocol['is_benchmark'])}"
            )
        if reasons:
            drift.append({"run_id": row["id"], "reasons": reasons})
    return drift


def _col(text: str, width: int) -> str:
    """左对齐/截断文本到固定宽度。"""
    return str(text)[:width].ljust(width)


def _capture_frontier_snapshot_for_campaign(
    conn: sqlite3.Connection,
    campaign: dict,
    *,
    maximize: list[str] | None = None,
    minimize: list[str] | None = None,
) -> str | None:
    """Compute and persist a fresh frontier snapshot for a campaign, returning its id."""
    maximize = maximize or ["wr"]
    minimize = minimize or ["params", "wall_s"]
    rows = conn.execute(
        """SELECT r.id, r.num_res_blocks, r.num_filters, r.num_params,
                  r.final_win_rate, r.wall_time_s, r.total_games, r.total_cycles,
                  r.total_steps, r.eval_level, r.eval_opponent, r.is_benchmark,
                  r.learning_rate, r.train_steps_per_cycle, r.parallel_games,
                  r.sweep_tag
           FROM runs r
           JOIN campaign_runs cr ON cr.run_id = r.id
           WHERE cr.campaign_id = ?
             AND r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')
             AND r.final_win_rate IS NOT NULL
           ORDER BY r.final_win_rate DESC""",
        (campaign["id"],),
    ).fetchall()
    if not rows:
        return None

    points = []
    for r in rows:
        wall_s = r["wall_time_s"]
        games = r["total_games"]
        throughput = (games / wall_s) if (games and wall_s and wall_s > 0) else None
        points.append({
            "run": r["id"][:8],
            "run_full": r["id"],
            "arch": f"{r['num_res_blocks'] or '?'}x{r['num_filters'] or '?'}",
            "params": r["num_params"],
            "wr": r["final_win_rate"],
            "wall_s": wall_s,
            "games": games,
            "cycles": r["total_cycles"],
            "steps": r["total_steps"],
            "lr": r["learning_rate"],
            "throughput": throughput,
            "eval_level": r["eval_level"],
            "eval_opponent": r["eval_opponent"],
            "is_benchmark": r["is_benchmark"],
            "sweep_tag": r["sweep_tag"],
        })

    front, dominated = _pareto_front(points, maximize=maximize, minimize=minimize)
    return _save_frontier_snapshot(
        conn,
        front,
        dominated,
        maximize,
        minimize,
        campaign.get("protocol", {}).get("eval_level"),
        None,
        campaign["id"],
    )


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------

def cmd_runs(conn: sqlite3.Connection) -> None:
    """列出所有训练运行及其汇总统计。"""
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
    """显示每个已完成运行的最佳检查点。"""
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
    """显示跨运行的胜率前沿（单调递增的检查点）。"""
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
    """并排对比两个运行。"""
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
    """追踪某运行的续训链。"""
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
    """列出所有注册对手。"""
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


def cmd_list_campaigns(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """SELECT c.id, c.name, c.domain, c.status, c.created_at, COUNT(cr.run_id) AS run_count
           FROM campaigns c
           LEFT JOIN campaign_runs cr ON cr.campaign_id = c.id
           GROUP BY c.id
           ORDER BY c.created_at DESC"""
    ).fetchall()
    if not rows:
        print("No campaigns found.")
        return
    print(f"{'Campaign':>18}  {'ID':>8}  {'Domain':>8}  {'Status':>10}  {'Runs':>4}  {'Date':>12}")
    print("-" * 74)
    for row in rows:
        print(f"{row['name']:>18}  {row['id'][:8]:>8}  {row['domain']:>8}  "
              f"{row['status']:>10}  {row['run_count']:>4}  {(row['created_at'] or '')[:10]:>12}")


def cmd_campaign_summary(conn: sqlite3.Connection, campaign: str) -> None:
    c = _resolve_campaign_or_exit(conn, campaign)
    search_space = conn.execute(
        "SELECT * FROM search_spaces WHERE id = ?",
        (c["search_space_id"],),
    ).fetchone()
    rows = _campaign_run_rows(conn, c["id"])
    status_counts: dict[str, int] = {}
    axis_coverage: dict[str, set] = {}
    for row in rows:
        status = row["status"] or "?"
        status_counts[status] = status_counts.get(status, 0) + 1
        axis_values = _json.loads(row["axis_values_json"])
        for key, value in axis_values.items():
            axis_coverage.setdefault(key, set()).add(str(value))
    drift = _campaign_drift(rows, c["protocol"])
    frontier_count_row = conn.execute(
        "SELECT COUNT(*) AS n FROM frontier_snapshots WHERE campaign_id = ?",
        (c["id"],),
    ).fetchone()

    print(f"Campaign Summary: {c['name']} ({c['id'][:8]})")
    print("=" * 72)
    print(f"  Domain:        {c['domain']}")
    print(f"  Train script:  {c['train_script']}")
    print(f"  Status:        {c['status']}")
    print(f"  Created:       {(c['created_at'] or '')[:19]}")
    print(f"  Search space:  {search_space['name']} v{search_space['version']} ({search_space['profile_hash'][:12]})")
    print(f"  Protocol:      {c['protocol']}")
    print(f"  Runs:          {len(rows)}")
    print(f"  Frontier snapshots: {frontier_count_row['n']}")
    print()
    print("Status counts:")
    for key in sorted(status_counts):
        print(f"  - {key}: {status_counts[key]}")
    print()
    print("Axis coverage:")
    for key in sorted(axis_coverage):
        values = ", ".join(sorted(axis_coverage[key], key=lambda v: (len(v), v)))
        print(f"  - {key}: {values}")
    print()
    if drift:
        print(f"Protocol drift: {len(drift)} run(s)")
        for item in drift[:8]:
            print(f"  - {item['run_id'][:8]}: {'; '.join(item['reasons'])}")
    else:
        print("Protocol drift: none")
    print()


def cmd_stage_summary(conn: sqlite3.Connection, campaign: str) -> None:
    """Show stage-by-stage summary for a campaign."""
    c = _resolve_campaign_or_exit(conn, campaign)
    stages = conn.execute(
        "SELECT * FROM campaign_stages WHERE campaign_id = ? ORDER BY stage",
        (c["id"],),
    ).fetchall()

    print(f"Stage Summary: {c['name']} ({c['id'][:8]})")
    print("=" * 72)
    if not stages:
        print("  (No stage records found.)")
        return

    for st in stages:
        policy = _json.loads(st["policy_json"] or "{}")
        budget = _json.loads(st["budget_json"] or "{}")
        # Count runs linked to this stage
        run_count = conn.execute(
            "SELECT COUNT(*) AS n FROM campaign_runs WHERE campaign_id = ? AND stage = ?",
            (c["id"], st["stage"]),
        ).fetchone()["n"]
        # Count candidates (distinct candidate_key)
        cand_count = conn.execute(
            "SELECT COUNT(DISTINCT candidate_key) AS n FROM campaign_runs WHERE campaign_id = ? AND stage = ?",
            (c["id"], st["stage"]),
        ).fetchone()["n"]
        # Count promotion decisions
        promo_count = conn.execute(
            "SELECT COUNT(*) AS n FROM promotion_decisions WHERE campaign_id = ? AND from_stage = ?",
            (c["id"], st["stage"]),
        ).fetchone()["n"]
        status_icon = "🔒" if st["status"] == "closed" else "🟢"
        print(f"  {status_icon} Stage {st['stage']}: {st['status']}")
        print(f"      budget={budget.get('time_budget', '?')}s  seed_target={st['seed_target']}")
        print(f"      runs={run_count}  candidates={cand_count}  promotions={promo_count}")
        if st["closed_at"]:
            print(f"      closed_at={st['closed_at'][:19]}")
    print()


def cmd_promotion_log(conn: sqlite3.Connection, campaign: str) -> None:
    """Show promotion decisions for a campaign."""
    c = _resolve_campaign_or_exit(conn, campaign)
    rows = conn.execute(
        """SELECT * FROM promotion_decisions
           WHERE campaign_id = ?
           ORDER BY from_stage, decision_rank""",
        (c["id"],),
    ).fetchall()

    print(f"Promotion Log: {c['name']} ({c['id'][:8]})")
    print("=" * 72)
    if not rows:
        print("  (No promotion decisions recorded.)")
        return

    current_stage = None
    for row in rows:
        if row["from_stage"] != current_stage:
            current_stage = row["from_stage"]
            print(f"\n  Stage {current_stage} → {row['to_stage']}:")
            print(f"  {'Rank':>6}  {'Decision':>10}  {'Candidate':>20}  {'Metric':>10}  {'Seeds':>6}  {'Reason'}")
            print(f"  {'─' * 80}")
        icon = "⬆" if row["decision"] == "promote" else ("⏸" if row["decision"] == "hold" else "⬇")
        ck = row["candidate_key"][:18]
        metrics = _json.loads(row["aggregated_metrics_json"] or "{}")
        metric_val = metrics.get("mean_wr", metrics.get("mean_metric", "?"))
        if isinstance(metric_val, (int, float)):
            metric_str = f"{metric_val:>10.1%}"
        else:
            metric_str = f"{metric_val:>10}"
        print(
            f"  {row['decision_rank'] or '-':>6}  "
            f"{icon} {row['decision']:<8}  "
            f"{ck:>20}  "
            f"{metric_str}  "
            f"{row['seed_count'] or '-':>6}  "
            f"{row['reason']}"
        )
    print()


def cmd_branch_tree(conn: sqlite3.Connection, campaign: str) -> None:
    """Show parent/child branch tree for a campaign."""
    c = _resolve_campaign_or_exit(conn, campaign)
    rows = conn.execute(
        """SELECT rb.id, rb.parent_run_id, rb.child_run_id,
                  rb.branch_reason, rb.delta_json, rb.status,
                  p.sweep_tag AS parent_tag, p.final_win_rate AS parent_wr,
                  c.sweep_tag AS child_tag, c.final_win_rate AS child_wr
           FROM run_branches rb
           LEFT JOIN runs p ON p.id = rb.parent_run_id
           LEFT JOIN runs c ON c.id = rb.child_run_id
           WHERE rb.campaign_id = ?
           ORDER BY rb.created_at""",
        (c["id"],),
    ).fetchall()

    print(f"Branch Tree: {c['name']} ({c['id'][:8]})")
    print("=" * 72)
    if not rows:
        print("  (No branches recorded.)")
        return

    # Group by parent
    from collections import defaultdict
    parent_groups = defaultdict(list)
    for row in rows:
        parent_groups[row["parent_run_id"]].append(row)

    for parent_id, branches in parent_groups.items():
        parent_tag = branches[0]["parent_tag"] or parent_id[:8]
        parent_wr = branches[0]["parent_wr"]
        wr_str = f"{parent_wr:.1%}" if parent_wr is not None else "?"
        print(f"\n  Parent: {parent_tag}  WR={wr_str}")
        print(f"  {'Branch':>8}  {'Reason':>18}  {'Status':>10}  {'Child':>20}  {'WR':>7}")
        print(f"  {'─' * 68}")
        for b in branches:
            child_wr = b["child_wr"]
            cwr_str = f"{child_wr:.1%}" if child_wr is not None else "?"
            child_tag = b["child_tag"] or "(not yet linked)"
            print(
                f"  {b['id'][:8]:>8}  "
                f"{b['branch_reason']:>18}  "
                f"{b['status']:>10}  "
                f"{child_tag[:20]:>20}  "
                f"{cwr_str:>7}"
            )
    print()


def cmd_trajectory_report(conn: sqlite3.Connection, campaign: str) -> None:
    """Show trajectory report: reason / delta / result for all branches."""
    c = _resolve_campaign_or_exit(conn, campaign)
    rows = conn.execute(
        """SELECT rb.*,
                  p.sweep_tag AS parent_tag, p.final_win_rate AS parent_wr, p.wall_time_s AS parent_wall,
                  c.sweep_tag AS child_tag, c.final_win_rate AS child_wr, c.wall_time_s AS child_wall,
                  c.num_params AS child_params
           FROM run_branches rb
           LEFT JOIN runs p ON p.id = rb.parent_run_id
           LEFT JOIN runs c ON c.id = rb.child_run_id
           WHERE rb.campaign_id = ?
           ORDER BY rb.created_at""",
        (c["id"],),
    ).fetchall()

    print(f"Trajectory Report: {c['name']} ({c['id'][:8]})")
    print("=" * 72)
    if not rows:
        print("  (No branches recorded.)")
        return

    for row in rows:
        delta = _json.loads(row["delta_json"] or "{}")
        delta_str = ", ".join(f"{k}={v}" for k, v in delta.items()) if delta else "(no change)"
        parent_wr = row["parent_wr"]
        child_wr = row["child_wr"]
        improvement = ""
        if parent_wr is not None and child_wr is not None:
            diff = child_wr - parent_wr
            icon = "📈" if diff > 0 else ("📉" if diff < 0 else "➡")
            improvement = f"  {icon} ΔWR={diff:+.1%}"

        parent_wr_str = f"{parent_wr:.1%}" if parent_wr is not None else "?"
        child_wr_str = f"{child_wr:.1%}" if child_wr is not None else "?"
        print(f"\n  Branch {row['id'][:8]} | {row['branch_reason']} | {row['status']}")
        print(f"    Parent: {row['parent_tag'] or row['parent_run_id'][:8]}  WR={parent_wr_str}")
        print(f"    Child:  {row['child_tag'] or '(pending)'}  WR={child_wr_str}{improvement}")
        print(f"    Delta:  {delta_str}")
    print()


def cmd_compare_parent_child(conn: sqlite3.Connection, branch_id: str) -> None:
    """Compare parent and child for a specific branch."""
    row = conn.execute(
        """SELECT rb.*,
                  p.sweep_tag AS parent_tag, p.final_win_rate AS parent_wr,
                  p.wall_time_s AS parent_wall, p.num_params AS parent_params,
                  p.total_games AS parent_games,
                  c.sweep_tag AS child_tag, c.final_win_rate AS child_wr,
                  c.wall_time_s AS child_wall, c.num_params AS child_params,
                  c.total_games AS child_games
           FROM run_branches rb
           LEFT JOIN runs p ON p.id = rb.parent_run_id
           LEFT JOIN runs c ON c.id = rb.child_run_id
           WHERE rb.id = ?""",
        (branch_id,),
    ).fetchone()

    if not row:
        print(f"Branch not found: {branch_id}")
        return

    delta = _json.loads(row["delta_json"] or "{}")

    print(f"Parent-Child Compare: Branch {branch_id[:8]}")
    print("=" * 72)
    print(f"  Reason: {row['branch_reason']}")
    print(f"  Delta:  {_json.dumps(delta, ensure_ascii=False)}")
    print()
    print(f"  {'Metric':>14}  {'Parent':>14}  {'Child':>14}  {'Δ':>10}")
    print(f"  {'─' * 56}")

    metrics = [
        ("WR", row["parent_wr"], row["child_wr"], lambda x: f"{x:.1%}" if x is not None else "?"),
        ("Wall(s)", row["parent_wall"], row["child_wall"], lambda x: f"{x:.0f}" if x is not None else "?"),
        ("Params", row["parent_params"], row["child_params"], lambda x: f"{x/1e3:.0f}K" if x else "?"),
        ("Games", row["parent_games"], row["child_games"], lambda x: f"{x}" if x is not None else "?"),
    ]
    for name, pval, cval, fmt in metrics:
        pstr = fmt(pval)
        cstr = fmt(cval)
        if pval is not None and cval is not None:
            diff = cval - pval
            dstr = f"{diff:+.1%}" if name == "WR" else f"{diff:+.0f}"
        else:
            dstr = "?"
        print(f"  {name:>14}  {pstr:>14}  {cstr:>14}  {dstr:>10}")
    print()


# ---------------------------------------------------------------------------
# v21: Surrogate-Guided Selector
# ---------------------------------------------------------------------------

def cmd_recommend_next(conn: sqlite3.Connection, campaign: str,
                       selector_policy: str | None = None,
                       acquisition_policy: str | None = None,
                       candidate_type: str | None = None,
                       limit: int = 5,
                       fmt: str = "md") -> None:
    """Generate and display recommendations for a campaign."""
    from framework.policies.selector_policy import load_selector_policy, policy_hash
    from framework.services.research.selector_service import recommend_for_campaign, build_recommendation_id
    from framework.policies.acquisition_policy import load_acquisition_policy, policy_hash as acquisition_policy_hash
    from framework.services.research.acquisition_service import rerank_candidates, snapshot_payload
    from framework.core.db import (
        get_campaign,
        save_recommendation_batch,
        save_recommendation,
        save_surrogate_snapshot,
    )

    c = get_campaign(conn, campaign)
    if not c:
        print(f"Campaign not found: {campaign}")
        return

    # Load selector policy
    if selector_policy:
        policy = load_selector_policy(selector_policy)
    else:
        # Default: look for domain selector policy
        default_path = f"domains/{c['domain']}/selector_policy.json"
        if os.path.isfile(default_path):
            policy = load_selector_policy(default_path)
        else:
            print(f"No selector policy found for domain '{c['domain']}'")
            return

    acquisition = None
    if acquisition_policy:
        acquisition = load_acquisition_policy(acquisition_policy)
    else:
        default_path = f"domains/{c['domain']}/acquisition_policy.json"
        if os.path.isfile(default_path):
            acquisition = load_acquisition_policy(default_path)

    # Verify domain match
    if policy["domain"] != c["domain"]:
        print(f"Error: policy domain '{policy['domain']}' != campaign domain '{c['domain']}'")
        return
    if acquisition and acquisition["domain"] != c["domain"]:
        print(f"Error: acquisition domain '{acquisition['domain']}' != campaign domain '{c['domain']}'")
        return

    # Protocol drift guard: check campaign runs against protocol
    drift_rows = conn.execute(
        """SELECT r.id, r.eval_level, r.is_benchmark, r.eval_opponent
           FROM campaign_runs cr
           JOIN runs r ON r.id = cr.run_id
           WHERE cr.campaign_id = ? AND r.status IN ('completed','time_budget','target_win_rate','target_games')""",
        (c["id"],),
    ).fetchall()
    campaign_protocol = {}
    if c and c["protocol_json"]:
        try:
            campaign_protocol = _json.loads(c["protocol_json"])
        except Exception:
            campaign_protocol = {}
    drift = _campaign_drift(drift_rows, campaign_protocol)
    if drift:
        print(f"Error: protocol drift detected in {len(drift)} run(s). Cannot recommend until resolved.")
        for d in drift[:3]:
            print(f"  Run {d['run_id']}: {', '.join(d['reasons'])}")
        return

    campaign_data = dict(c)
    if campaign_data.get("protocol_json"):
        try:
            campaign_data["protocol"] = _json.loads(campaign_data["protocol_json"])
        except Exception:
            campaign_data["protocol"] = {}
    else:
        campaign_data["protocol"] = {}

    frontier_snapshot_id = _capture_frontier_snapshot_for_campaign(conn, campaign_data)
    pool = recommend_for_campaign(conn, campaign_data, policy,
                                  candidate_type=candidate_type,
                                  limit=None)

    surrogate_snapshot_id = None
    if acquisition and pool:
        pool, surrogate_summary = rerank_candidates(pool, acquisition)
        surrogate_snapshot_id = f"sur-{uuid.uuid4().hex[:16]}"
        save_surrogate_snapshot(
            conn,
            snapshot_id=surrogate_snapshot_id,
            campaign_id=campaign_data["id"],
            frontier_snapshot_id=frontier_snapshot_id,
            acquisition_name=acquisition["name"],
            acquisition_version=acquisition["version"],
            policy_hash=acquisition_policy_hash(acquisition),
            objectives_json=_json.dumps(acquisition["objectives"], ensure_ascii=False, sort_keys=True),
            feature_schema_json=_json.dumps(surrogate_summary["feature_schema"], ensure_ascii=False, sort_keys=True),
            summary_json=_json.dumps(snapshot_payload(acquisition, surrogate_summary), ensure_ascii=False, sort_keys=True),
            candidate_count=len(pool),
        )

    recommendations = pool[:limit]

    if not recommendations:
        print(f"No recommendations for campaign '{campaign}'")
        return

    # Invalidate stale planned recommendations before creating new batch
    conn.execute(
        """UPDATE recommendations
           SET status = 'invalidated'
           WHERE batch_id IN (
               SELECT id FROM recommendation_batches WHERE campaign_id = ?
           ) AND status = 'planned'""",
        (c["id"],),
    )
    conn.commit()

    # Persist batch + recommendations
    batch_id = f"batch-{uuid.uuid4().hex[:16]}"
    save_recommendation_batch(
        conn,
        batch_id=batch_id,
        campaign_id=c["id"],
        selector_name=policy["name"],
        selector_version=policy["version"],
        selector_hash=policy_hash(policy),
        frontier_snapshot_id=frontier_snapshot_id,
        acquisition_name=acquisition["name"] if acquisition else None,
        acquisition_version=acquisition["version"] if acquisition else None,
        surrogate_snapshot_id=surrogate_snapshot_id,
    )
    for rank, rec in enumerate(recommendations, 1):
        rec_id = build_recommendation_id(batch_id, rec)
        save_recommendation(
            conn,
            recommendation_id=rec_id,
            batch_id=batch_id,
            candidate_type=rec["candidate_type"],
            candidate_key=rec.get("candidate_key"),
            rank=rank,
            score_total=rec["score_total"],
            score_breakdown_json=_json.dumps(rec["score_breakdown"], ensure_ascii=False, sort_keys=True),
            rationale_json=_json.dumps(rec["rationale"], ensure_ascii=False, sort_keys=True),
            axis_values_json=_json.dumps(rec.get("axis_values", {}), ensure_ascii=False, sort_keys=True) if rec.get("axis_values") else None,
            branch_reason=rec.get("branch_reason"),
            delta_json=rec.get("delta_json"),
            selector_score_total=rec.get("selector_score_total", rec["score_total"]),
            acquisition_score=rec.get("acquisition_score"),
            parent_run_id=rec.get("parent_run_id"),
            parent_checkpoint_id=rec.get("parent_checkpoint_id"),
            candidate_payload_json=_json.dumps(rec.get("candidate_payload", rec.get("axis_values", {})), ensure_ascii=False, sort_keys=True) if (rec.get("candidate_payload") or rec.get("axis_values")) else None,
            objective_metrics_json=_json.dumps(rec.get("score_signals", {}).get("objective_metrics", {}), ensure_ascii=False, sort_keys=True) if rec.get("score_signals", {}).get("objective_metrics") else None,
        )

    if fmt == "json":
        output = {
            "campaign": campaign,
            "batch_id": batch_id,
            "selector": policy["name"],
            "acquisition": acquisition["name"] if acquisition else None,
            "frontier_snapshot_id": frontier_snapshot_id,
            "surrogate_snapshot_id": surrogate_snapshot_id,
            "recommendations": recommendations,
        }
        print(_json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(f"Recommendations for: {campaign}")
        print(f"Selector: {policy['name']} v{policy['version']}")
        if acquisition:
            print(f"Acquisition: {acquisition['name']} v{acquisition['version']}")
        print(f"Batch: {batch_id}")
        if frontier_snapshot_id:
            print(f"Frontier snapshot: {frontier_snapshot_id[:16]}")
        if surrogate_snapshot_id:
            print(f"Surrogate snapshot: {surrogate_snapshot_id[:16]}")
        print("=" * 80)
        print(f"  {'#':>3}  {'Type':>16}  {'Score':>8}  {'Key':>20}  {'Rationale'}")
        print("-" * 80)
        for i, rec in enumerate(recommendations, 1):
            key = (rec.get("candidate_key") or "")[:18]
            rationale = rec["rationale"].get("summary", "")
            print(f"  {i:>3}  {rec['candidate_type']:>16}  {rec['score_total']:>8.3f}  {key:>20}  {rationale}")
        print()
        print(f"Total: {len(recommendations)} recommendations")
        print(f"To accept: update status to 'accepted', then execute via sweep.py or branch.py")


def cmd_recommendation_log(conn: sqlite3.Connection, campaign: str, fmt: str = "md") -> None:
    """Show recommendation history for a campaign."""
    from framework.core.db import get_campaign, list_recommendation_batches, list_recommendations_for_batch

    c = get_campaign(conn, campaign)
    if not c:
        print(f"Campaign not found: {campaign}")
        return

    batches = list_recommendation_batches(conn, c["id"])
    if not batches:
        print(f"No recommendation batches for campaign '{campaign}'")
        return

    if fmt == "json":
        output = {"campaign": campaign, "batches": []}
        for b in batches:
            recs = list_recommendations_for_batch(conn, b["id"])
            output["batches"].append({
                "batch_id": b["id"],
                "created_at": b["created_at"],
                "selector": f"{b['selector_name']} v{b['selector_version']}",
                "recommendations": recs,
            })
        print(_json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(f"Recommendation Log: {campaign}")
        print("=" * 80)
        for b in batches:
            recs = list_recommendations_for_batch(conn, b["id"])
            status_counts = {}
            for r in recs:
                status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
            acq = ""
            if b.get("acquisition_name"):
                acq = f" | {b['acquisition_name']} v{b.get('acquisition_version') or '?'}"
            print(f"\n  Batch: {b['id'][:16]} | {b['selector_name']} v{b['selector_version']}{acq}")
            print(f"  Created: {b['created_at']}")
            if b.get("frontier_snapshot_id"):
                print(f"  Frontier snapshot: {b['frontier_snapshot_id'][:16]}")
            if b.get("surrogate_snapshot_id"):
                print(f"  Surrogate snapshot: {b['surrogate_snapshot_id'][:16]}")
            print(f"  Status: {', '.join(f'{k}={v}' for k, v in status_counts.items())}")
            print(f"  {'#':>3}  {'Type':>16}  {'Score':>8}  {'Status':>10}  {'Key'}")
            print(f"  {'─' * 60}")
            for r in recs:
                key = (r.get("candidate_key") or "")[:30]
                print(f"  {r['rank']:>3}  {r['candidate_type']:>16}  {r['score_total']:>8.3f}  {r['status']:>10}  {key}")
        print()


def cmd_recommendation_outcomes(conn: sqlite3.Connection, campaign: str, fmt: str = "md") -> None:
    """Show recommendation outcome summary for a campaign."""
    from framework.core.db import (
        get_campaign,
        list_recommendation_batches,
        list_recommendations_for_batch,
        list_recommendation_outcomes,
    )

    c = get_campaign(conn, campaign)
    if not c:
        print(f"Campaign not found: {campaign}")
        return

    batches = list_recommendation_batches(conn, c["id"])
    if not batches:
        print(f"No recommendation batches for campaign '{campaign}'")
        return

    all_outcomes = []
    for b in batches:
        recs = list_recommendations_for_batch(conn, b["id"])
        for r in recs:
            outcomes = list_recommendation_outcomes(conn, r["id"])
            for o in outcomes:
                all_outcomes.append({
                    "batch_id": b["id"],
                    "frontier_snapshot_id": b.get("frontier_snapshot_id"),
                    "surrogate_snapshot_id": b.get("surrogate_snapshot_id"),
                    "recommendation_id": r["id"],
                    "candidate_type": r["candidate_type"],
                    "rank": r["rank"],
                    "outcome_label": o["outcome_label"],
                    "observed_metrics": o["observed_metrics_json"],
                })

    if not all_outcomes:
        print(f"No outcomes recorded for campaign '{campaign}'")
        return

    if fmt == "json":
        print(_json.dumps({"campaign": campaign, "outcomes": all_outcomes}, indent=2, ensure_ascii=False))
    else:
        print(f"Recommendation Outcomes: {campaign}")
        print("=" * 80)
        print(f"  {'Rank':>4}  {'Type':>16}  {'Outcome':>12}  {'Metrics'}")
        print("-" * 80)
        for o in all_outcomes:
            metrics = o.get("observed_metrics", "{}")[:40]
            print(f"  {o['rank']:>4}  {o['candidate_type']:>16}  {o['outcome_label']:>12}  {metrics}")
        print()
        labels = {}
        for o in all_outcomes:
            labels[o["outcome_label"]] = labels.get(o["outcome_label"], 0) + 1
        print(f"Summary: {', '.join(f'{k}={v}' for k, v in labels.items())}")
        print()


def cmd_acquisition_summary(conn: sqlite3.Connection, campaign: str, fmt: str = "md") -> None:
    """Show v21.1 acquisition lineage for recommendation batches."""
    from framework.core.db import (
        get_campaign,
        list_recommendation_batches,
        list_recommendations_for_batch,
        list_recommendation_outcomes,
        get_surrogate_snapshot,
    )

    c = get_campaign(conn, campaign)
    if not c:
        print(f"Campaign not found: {campaign}")
        return

    batches = list_recommendation_batches(conn, c["id"])
    if not batches:
        print(f"No recommendation batches for campaign '{campaign}'")
        return

    rows = []
    for b in batches:
        recs = list_recommendations_for_batch(conn, b["id"])
        outcomes = []
        for r in recs:
            outcomes.extend(list_recommendation_outcomes(conn, r["id"]))
        snapshot = get_surrogate_snapshot(conn, b["surrogate_snapshot_id"]) if b.get("surrogate_snapshot_id") else None
        rows.append({
            "batch_id": b["id"],
            "selector": f"{b['selector_name']} v{b['selector_version']}",
            "acquisition": f"{b['acquisition_name']} v{b['acquisition_version']}" if b.get("acquisition_name") else None,
            "frontier_snapshot_id": b.get("frontier_snapshot_id"),
            "surrogate_snapshot_id": b.get("surrogate_snapshot_id"),
            "candidate_count": snapshot.get("candidate_count") if snapshot else len(recs),
            "outcome_count": len(outcomes),
        })

    if fmt == "json":
        print(_json.dumps({"campaign": campaign, "batches": rows}, indent=2, ensure_ascii=False))
        return

    print(f"Acquisition Summary: {campaign}")
    print("=" * 80)
    print(f"  {'Batch':>16}  {'Candidates':>10}  {'Outcomes':>8}  {'Acquisition'}")
    print("-" * 80)
    for row in rows:
        print(
            f"  {row['batch_id'][:16]:>16}  {row['candidate_count']:>10}  "
            f"{row['outcome_count']:>8}  {(row['acquisition'] or 'heuristic'):>20}"
        )
    print()


def cmd_matrix(conn: sqlite3.Connection, tag_prefix: str, campaign: str | None = None,
               allow_drift: bool = False) -> None:
    """按标签前缀分组展示 sweep 结果。"""
    filter_info = tag_prefix
    if campaign:
        c = _resolve_campaign_or_exit(conn, campaign)
        rows = _campaign_run_rows(conn, c["id"])
        drift = _campaign_drift(rows, c["protocol"])
        if drift and not allow_drift:
            print(f"Campaign '{c['name']}' has {len(drift)} drift run(s); refusing matrix output.")
            print("Use --allow-drift to override.")
            return
        filter_info = f"{tag_prefix} (campaign={c['name']})"
    else:
        rows = conn.execute(
            "SELECT id, status, sweep_tag, total_cycles, total_games, "
            "final_loss, final_win_rate, wall_time_s, num_params, "
            "learning_rate, train_steps_per_cycle, num_res_blocks, num_filters, "
            "replay_buffer_size, seed "
            "FROM runs WHERE sweep_tag IS NOT NULL ORDER BY started_at"
        ).fetchall()

    # 过滤匹配 sweep_tag 前缀的运行
    groups: dict[str, list[dict]] = {}
    for r in rows:
        sweep_tag = r["sweep_tag"] or ""
        if not sweep_tag.startswith(tag_prefix):
            continue
        # 去除种子后缀以按配置分组
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
        print(f"No runs found with sweep_tag prefix '{filter_info}'")
        return

    # 从超参中提取变化轴
    print(f"Sweep Matrix: {filter_info}  ({sum(len(v) for v in groups.values())} runs in {len(groups)} configs)")
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
        std_wr = math.sqrt(sum((w - mean_wr)**2 for w in wrs) / (len(wrs) - 1)) if len(wrs) > 1 else 0
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
    """显示某运行的训练稳定性指标。"""

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
        # 最大连续波动
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

    # 检查点分布
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
# 停滞检测（domain-agnostic）
# ---------------------------------------------------------------------------

def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Simple linear regression. Returns (slope, intercept, r_squared)."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_xx = sum(x * x for x in xs)
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0, sum_y / n, 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    # R-squared
    mean_y = sum_y / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r_sq = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return slope, intercept, r_sq


def cmd_stagnation(conn: sqlite3.Connection, run_id: str) -> None:
    """检测训练停滞：WR 时间序列在 eval 点上无趋势改善。

    domain-agnostic: 只读 cycle_metrics 中的 (cycle, win_rate) 做线性回归。
    Uses the second half of the run for detection (avoids early ramp-up noise).
    """
    row = conn.execute(
        "SELECT id, total_cycles, final_win_rate FROM runs WHERE id LIKE ?",
        (run_id + "%",)
    ).fetchone()
    if not row:
        print(f"Run not found: {run_id}")
        return

    full_id = row["id"]
    wr_rows = conn.execute(
        "SELECT cycle, win_rate FROM cycle_metrics "
        "WHERE run_id = ? AND win_rate IS NOT NULL ORDER BY cycle",
        (full_id,)
    ).fetchall()

    if not wr_rows:
        print(f"No win-rate data for run {full_id[:8]}")
        return

    cycles = [float(r["cycle"]) for r in wr_rows]
    wrs = [r["win_rate"] for r in wr_rows]
    n = len(wrs)

    print(f"Stagnation analysis for run {full_id[:8]}")
    print(f"  Total eval points: {n}")
    print(f"  WR range: {min(wrs):.1%} — {max(wrs):.1%}")
    print()

    # Full-run regression
    slope_all, _, r2_all = _linear_regression(cycles, wrs)
    print(f"  Full-run trend:  slope={slope_all:.6f}/cycle  R²={r2_all:.3f}")

    if n < 10:
        print(f"  Not enough data for stagnation analysis (need >= 10 probes, have {n})")
        print()
        return

    # Analyze second half of the run (skip early ramp-up)
    half = n // 2
    half_c = cycles[half:]
    half_w = wrs[half:]
    nh = len(half_w)
    slope_half, _, r2_half = _linear_regression(half_c, half_w)
    mean_w = sum(half_w) / nh
    wr_std = (sum((w - mean_w) ** 2 for w in half_w) / nh) ** 0.5

    print(f"  Second half ({nh} probes from cycle {int(half_c[0])}):")
    print(f"    slope={slope_half:.6f}/cycle  R²={r2_half:.3f}  WR_std={wr_std:.1%}")

    # Stagnation criteria: near-zero slope AND high variance relative to trend
    # A meaningful slope should produce WR change >> std over the window
    cycle_span = half_c[-1] - half_c[0]
    expected_change = abs(slope_half * cycle_span)
    is_stagnant = (r2_half < 0.15 and expected_change < wr_std) or abs(slope_half) < 0.0001

    if is_stagnant:
        # Find onset: scan forward with sliding window of 10 points
        stag_start_cycle = int(half_c[0])
        window = min(10, n // 3)
        for i in range(window, n):
            seg_c = cycles[i - window:i]
            seg_w = wrs[i - window:i]
            s, _, r2 = _linear_regression(seg_c, seg_w)
            span = seg_c[-1] - seg_c[0]
            seg_mean = sum(seg_w) / window
            seg_std = (sum((w - seg_mean) ** 2 for w in seg_w) / window) ** 0.5
            if (r2 < 0.15 and abs(s * span) < seg_std) or abs(s) < 0.0001:
                stag_start_cycle = int(seg_c[0])
                break

        wasted_cycles = int(cycles[-1]) - stag_start_cycle
        print()
        print(f"  ⚠ STAGNATION DETECTED")
        print(f"    Estimated onset: cycle {stag_start_cycle}")
        print(f"    Wasted cycles: ~{wasted_cycles} ({wasted_cycles * 100 // max(1, int(cycles[-1]))}% of total)")
        print(f"    WR has not shown meaningful improvement in the second half of training.")
        print(f"    Consider: stopping this run, changing hyperparams, or switching opponent.")
    else:
        print()
        print(f"  ✓ No stagnation detected (WR still trending upward)")

    print()


# ---------------------------------------------------------------------------
# Pareto 非支配排序（domain-agnostic）
# ---------------------------------------------------------------------------

def _pareto_front(points: list[dict], maximize: list[str],
                  minimize: list[str]) -> tuple[list[dict], list[dict]]:
    return service_pareto_front(points, maximize=maximize, minimize=minimize)


def _load_objective_profile_for_pareto(
    conn: sqlite3.Connection,
    profile_ref: str | None,
    campaign_row: dict | None,
) -> tuple[dict | None, str | None]:
    """Resolve an objective profile from a path, id, or campaign binding."""
    if profile_ref:
        if os.path.isfile(profile_ref):
            from framework.profiles.objective_profile import load_objective_profile
            profile = load_objective_profile(profile_ref)
            return profile, None
        row = conn.execute(
            "SELECT * FROM objective_profiles WHERE id = ?",
            (profile_ref,),
        ).fetchone()
        if not row:
            print(f"Objective profile not found: {profile_ref}")
            return None, None
        return _json.loads(row["profile_json"]), row["id"]

    if campaign_row and campaign_row.get("objective_profile_id"):
        row = conn.execute(
            "SELECT * FROM objective_profiles WHERE id = ?",
            (campaign_row["objective_profile_id"],),
        ).fetchone()
        if row:
            return _json.loads(row["profile_json"]), row["id"]
    return None, None


def _constraint_satisfied(value: float | None, op: str, expected) -> bool:
    if value is None:
        return False
    expected = float(expected)
    if op == "eq":
        return value == expected
    if op == "le":
        return value <= expected
    if op == "ge":
        return value >= expected
    if op == "lt":
        return value < expected
    if op == "gt":
        return value > expected
    return False


def _objective_axis_meta(profile: dict | None) -> dict:
    if not profile:
        return {}
    display = profile.get("display", {})
    return {
        metric: {
            "label": meta.get("label", metric),
            "format": meta.get("format", "number"),
        }
        for metric, meta in display.items()
    }


def _build_metric_points(
    conn: sqlite3.Connection,
    *,
    campaign_id: str | None,
    stage: str | None,
    profile: dict,
) -> tuple[list[dict], list[dict]]:
    """Build Pareto points from run_metrics, returning (feasible, infeasible)."""
    required = set(profile.get("maximize", [])) | set(profile.get("minimize", []))
    required |= {c["metric"] for c in profile.get("hard_constraints", [])}
    if not required:
        return [], []

    params: list = []
    if campaign_id:
        sql = """SELECT r.id, r.sweep_tag, cr.stage, cr.axis_values_json
                 FROM campaign_runs cr
                 JOIN runs r ON r.id = cr.run_id
                 WHERE cr.campaign_id = ?
                   AND r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')"""
        params.append(campaign_id)
        if stage:
            sql += " AND cr.stage = ?"
            params.append(stage)
    else:
        sql = """SELECT r.id, r.sweep_tag, NULL AS stage, '{}' AS axis_values_json
                 FROM runs r
                 WHERE r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')"""

    rows = conn.execute(sql, params).fetchall()
    feasible = []
    infeasible = []
    for row in rows:
        metric_rows = conn.execute(
            "SELECT metric_name, metric_value FROM run_metrics WHERE run_id = ?",
            (row["id"],),
        ).fetchall()
        metrics = {m["metric_name"]: m["metric_value"] for m in metric_rows}
        if not required.issubset(metrics):
            continue
        point = {
            "run": row["id"][:8],
            "run_full": row["id"],
            "label": row["sweep_tag"] or row["id"][:8],
            "arch": row["sweep_tag"] or row["id"][:8],
            "stage": row["stage"],
            "axis_values": _json.loads(row["axis_values_json"] or "{}"),
            "constraint_status": {},
        }
        point.update(metrics)
        failed = []
        for constraint in profile.get("hard_constraints", []):
            metric = constraint["metric"]
            ok = _constraint_satisfied(metrics.get(metric), constraint["op"], constraint["value"])
            point["constraint_status"][metric] = ok
            if not ok:
                failed.append(constraint)
        if failed:
            point["failed_constraints"] = failed
            infeasible.append(point)
        else:
            feasible.append(point)
    return feasible, infeasible


def _compute_knee_point(front: list[dict], maximize: list[str], minimize: list[str]) -> tuple[dict | None, dict | None]:
    best = service_compute_knee_point(front, maximize=maximize, minimize=minimize)
    if not best:
        return None, None
    rationale = {"method": "utopia_distance", "axes": maximize + minimize}
    return best, rationale


def cmd_pareto(conn: sqlite3.Connection, fmt: str = "md",
               maximize: list[str] | None = None,
               minimize: list[str] | None = None,
               eval_level: int | None = None,
                sweep_tag: str | None = None,
               campaign: str | None = None,
               allow_drift: bool = False,
               plot: bool = False,
               output_path: str | None = None,
               stage: str | None = None,
               objective_profile: str | None = None,
               metric_source: str = "legacy",
               show_knee: bool = False) -> None:
    """对 completed runs 执行非支配排序，输出 Pareto 前沿。

    默认轴: maximize WR, minimize params 和 wall_time.
    支持 --eval-level 和 --sweep-tag 过滤，确保同 benchmark 条件比较。
    支持 --stage 按 campaign stage 过滤。
    支持 --plot 生成 PNG 散点图。
    """
    campaign_row = None
    campaign_id = None
    if campaign is not None:
        campaign_row = _resolve_campaign_or_exit(conn, campaign)
        campaign_id = campaign_row["id"]

    profile, profile_id = _load_objective_profile_for_pareto(conn, objective_profile, campaign_row)
    if profile:
        maximize = maximize or profile.get("maximize", [])
        minimize = minimize or profile.get("minimize", [])
        metric_source = "run_metrics"
    else:
        maximize = maximize or ["wr"]
        minimize = minimize or ["params", "wall_s"]

    if metric_source == "run_metrics":
        if campaign_row:
            drift = _campaign_drift(_campaign_run_rows(conn, campaign_id), campaign_row["protocol"])
            if drift and not allow_drift:
                print(f"Campaign '{campaign_row['name']}' has {len(drift)} drift run(s); refusing Pareto output.")
                print("Use --allow-drift to override.")
                return
        if not profile:
            profile = {
                "maximize": maximize,
                "minimize": minimize,
                "hard_constraints": [],
                "display": {},
                "knee": {"method": "utopia_distance"},
            }
        points, infeasible = _build_metric_points(conn, campaign_id=campaign_id, stage=stage, profile=profile)
        if not points:
            print("No completed runs with run_metrics matching objective profile.")
            return
        front, dominated = _pareto_front(points, maximize=maximize, minimize=minimize)
        sort_key = maximize[0] if maximize else (minimize[0] if minimize else None)
        if sort_key:
            front.sort(key=lambda p: -(p.get(sort_key) or 0))
            dominated.sort(key=lambda p: -(p.get(sort_key) or 0))
        knee, knee_rationale = _compute_knee_point(front, maximize, minimize) if show_knee or profile.get("knee", {}).get("method") == "utopia_distance" else (None, None)
        _save_frontier_snapshot(
            conn, front, dominated, maximize, minimize, eval_level, sweep_tag, campaign_id,
            objective_profile_id=profile_id,
            metric_source="run_metrics",
            constraints_json=_json.dumps(profile.get("hard_constraints", []), ensure_ascii=False, sort_keys=True),
            knee_run_id=knee.get("run_full") if knee else None,
            knee_rationale_json=_json.dumps(knee_rationale, ensure_ascii=False, sort_keys=True) if knee_rationale else None,
        )
        if fmt == "json":
            output = {
                "pareto_front": front,
                "dominated": dominated,
                "infeasible": infeasible,
                "knee": knee,
                "knee_rationale": knee_rationale,
                "axes": {"maximize": maximize, "minimize": minimize},
                "metric_source": "run_metrics",
                "campaign": campaign_row["name"] if campaign_row is not None else None,
            }
            print(_json.dumps(output, indent=2, ensure_ascii=False))
        else:
            print(f"Pareto Front (metric_source=run_metrics, maximize {', '.join(maximize)}, minimize {', '.join(minimize)})")
            print(f"{'─' * 96}")
            print(f"  Feasible: {len(points)}  |  Infeasible: {len(infeasible)}  |  Front: {len(front)}  |  Dominated: {len(dominated)}")
            if knee:
                print(f"  Knee: {knee['run']} ({knee.get('label', '-')})")
            print()
            cols = ["run", "label"] + maximize[:2] + minimize[:2]
            print("  " + "  ".join(f"{c:>18}" for c in cols))
            print("  " + "─" * (20 * len(cols)))
            for p in front:
                print("  " + "  ".join(f"{str(round(p.get(c), 6)) if isinstance(p.get(c), float) else str(p.get(c, '-')):>18}" for c in cols))
        if plot:
            from framework.services.frontier.plotting import plot_pareto_artifacts
            y_axis = maximize[0] if maximize else None
            x_axis = minimize[0] if minimize else None
            if y_axis and x_axis:
                artifacts = plot_pareto_artifacts(
                    front, dominated,
                    x_key=x_axis, y_key=y_axis,
                    label_key="label",
                    output_path=output_path or "output/pareto_front.png",
                    axis_meta=_objective_axis_meta(profile),
                    knee_point=knee,
                    metrics=maximize + minimize,
                )
                print(f"  📊 Plot saved: {artifacts['overview']}")
                print(f"  📄 Front table: {artifacts['front_csv']}")
        return

    select_sql = """SELECT id, num_res_blocks, num_filters, num_params,
                           final_win_rate, wall_time_s, total_games, total_cycles,
                           total_steps, eval_level, eval_opponent, is_benchmark,
                           learning_rate, train_steps_per_cycle, parallel_games,
                           sweep_tag
                    FROM runs"""

    status_col = "status"
    wr_col = "final_win_rate"

    # Build WHERE clause with optional filters
    where_clauses = [f"{status_col} IN ('completed', 'time_budget', 'target_win_rate', 'target_games')",
                     f"{wr_col} IS NOT NULL"]
    params_sql: list = []
    if campaign is not None:
        drift = _campaign_drift(_campaign_run_rows(conn, campaign_id), campaign_row["protocol"])
        if drift and not allow_drift:
            print(f"Campaign '{campaign_row['name']}' has {len(drift)} drift run(s); refusing Pareto output.")
            print("Use --allow-drift to override.")
            return
        select_sql = """SELECT r.id, r.num_res_blocks, r.num_filters, r.num_params,
                               r.final_win_rate, r.wall_time_s, r.total_games, r.total_cycles,
                               r.total_steps, r.eval_level, r.eval_opponent, r.is_benchmark,
                                r.learning_rate, r.train_steps_per_cycle, r.parallel_games,
                                r.sweep_tag
                         FROM runs r
                         JOIN campaign_runs cr ON cr.run_id = r.id"""
        status_col = "r.status"
        wr_col = "r.final_win_rate"
        where_clauses = [f"{status_col} IN ('completed', 'time_budget', 'target_win_rate', 'target_games')",
                         f"{wr_col} IS NOT NULL"]
        where_clauses.append("cr.campaign_id = ?")
        params_sql.append(campaign_id)
    if eval_level is not None:
        where_clauses.append("eval_level = ?")
        params_sql.append(eval_level)
    if sweep_tag is not None:
        where_clauses.append("sweep_tag LIKE ?")
        params_sql.append(sweep_tag + "%")
    if stage is not None:
        if campaign is None:
            print("--stage requires --campaign")
            return
        where_clauses.append("cr.stage = ?")
        params_sql.append(stage)

    where_str = " AND ".join(where_clauses)
    rows = conn.execute(
        f"""{select_sql}
           WHERE {where_str}
           ORDER BY final_win_rate DESC""",
        params_sql
    ).fetchall()

    if not rows:
        print("No completed runs with win-rate data matching filters.")
        if eval_level is not None:
            print(f"  (filter: eval_level={eval_level})")
        if sweep_tag is not None:
            print(f"  (filter: sweep_tag={sweep_tag}*)")
        if campaign_row is not None:
            print(f"  (filter: campaign={campaign_row['name']})")
        if stage is not None:
            print(f"  (filter: stage={stage})")
        return

    # Auto eval_level grouping: warn if data crosses levels and no filter set
    levels = set(r["eval_level"] for r in rows if r["eval_level"] is not None)
    if len(levels) > 1 and eval_level is None:
        from collections import Counter
        level_counts = Counter(r["eval_level"] for r in rows if r["eval_level"] is not None)
        most_common_level = level_counts.most_common(1)[0][0]
        print(f"⚠  Data spans eval_levels {sorted(levels)}. "
              f"Auto-selecting level {most_common_level} (most data: {level_counts[most_common_level]} runs).")
        print(f"   Use --eval-level N to override.\n")
        rows = [r for r in rows if r["eval_level"] == most_common_level]
        eval_level = most_common_level

    points = []
    for r in rows:
        wall_s = r["wall_time_s"]
        games = r["total_games"]
        throughput = (games / wall_s) if (games and wall_s and wall_s > 0) else None
        points.append({
            "run": r["id"][:8],
            "run_full": r["id"],
            "arch": f"{r['num_res_blocks'] or '?'}x{r['num_filters'] or '?'}",
            "params": r["num_params"],
            "wr": r["final_win_rate"],
            "wall_s": wall_s,
            "games": games,
            "cycles": r["total_cycles"],
            "steps": r["total_steps"],
            "lr": r["learning_rate"],
            "throughput": throughput,
            "eval_level": r["eval_level"],
            "eval_opponent": r["eval_opponent"],
            "is_benchmark": r["is_benchmark"],
            "sweep_tag": r["sweep_tag"],
        })

    front, dominated = _pareto_front(points, maximize=maximize, minimize=minimize)

    # Sort front by first maximize axis descending
    sort_key = maximize[0] if maximize else "wr"
    front.sort(key=lambda p: -(p.get(sort_key) or 0))
    dominated.sort(key=lambda p: -(p.get(sort_key) or 0))

    # --- Save frontier snapshot ---
    _save_frontier_snapshot(conn, front, dominated, maximize, minimize,
                           eval_level, sweep_tag, campaign_id)

    if fmt == "json":
        import json as _j
        output = {
            "pareto_front": front,
            "dominated": dominated,
            "axes": {"maximize": maximize, "minimize": minimize},
            "total_runs": len(points),
            "eval_level": eval_level,
            "sweep_tag": sweep_tag,
            "campaign": campaign_row["name"] if campaign_row is not None else None,
        }
        print(_j.dumps(output, indent=2, ensure_ascii=False))
    else:
        # Markdown text output (backward compatible)
        max_str = ", ".join(maximize)
        min_str = ", ".join(minimize)
        print(f"Pareto Front (maximize {max_str}, minimize {min_str})")
        print(f"{'─' * 80}")
        filter_info = ""
        if eval_level is not None:
            filter_info += f"  eval_level={eval_level}"
        if sweep_tag is not None:
            filter_info += f"  sweep_tag={sweep_tag}*"
        if campaign_row is not None:
            filter_info += f"  campaign={campaign_row['name']}"
        if stage is not None:
            filter_info += f"  stage={stage}"
        print(f"  Total runs analyzed: {len(points)}{filter_info}")
        print(f"  Front points: {len(front)}  |  Dominated: {len(dominated)}")
        print()

        print("  ★ PARETO FRONT:")
        print(f"  {'Run':>10}  {'Arch':>8}  {'Params':>8}  {'WR':>7}  {'Wall':>7}  {'Games':>7}  {'Opp':>6}")
        print(f"  {'─' * 68}")
        for p in front:
            opp = p["eval_opponent"] if p["eval_opponent"] else f"L{p['eval_level'] or 0}"
            params_s = f"{p['params']/1000:.0f}K" if p["params"] else "?"
            wall = f"{p['wall_s']:.0f}s" if p["wall_s"] else "?"
            print(f"  {p['run']:>10}  {p['arch']:>8}  {params_s:>8}  {p['wr']:.1%}  {wall:>7}  {str(p['games'] or '-'):>7}  {opp:>6}")

        if dominated:
            print()
            print("  ○ DOMINATED:")
            print(f"  {'Run':>10}  {'Arch':>8}  {'Params':>8}  {'WR':>7}  {'Wall':>7}  {'Games':>7}  {'Opp':>6}")
            print(f"  {'─' * 68}")
            for p in dominated:
                opp = p["eval_opponent"] if p["eval_opponent"] else f"L{p['eval_level'] or 0}"
                params_s = f"{p['params']/1000:.0f}K" if p["params"] else "?"
                wall = f"{p['wall_s']:.0f}s" if p["wall_s"] else "?"
                print(f"  {p['run']:>10}  {p['arch']:>8}  {params_s:>8}  {p['wr']:.1%}  {wall:>7}  {str(p['games'] or '-'):>7}  {opp:>6}")

        print()

    # --- Plot if requested ---
    if plot:
        from framework.services.frontier.plotting import plot_pareto
        # Determine x/y axes for plot: y = first maximize, x = first minimize
        y_axis = maximize[0] if maximize else "wr"
        x_axis = minimize[0] if minimize else "params"
        size_axis = minimize[1] if len(minimize) > 1 else None

        if output_path is None:
            output_path = "output/pareto_front.png"

        path = plot_pareto(
            front, dominated,
            x_key=x_axis, y_key=y_axis,
            size_key=size_axis,
            output_path=output_path,
            eval_level=eval_level,
            sweep_tag=sweep_tag,
        )
        print(f"  📊 Plot saved: {path}")


def _save_frontier_snapshot(conn: sqlite3.Connection,
                            front: list[dict], dominated: list[dict],
                            maximize: list[str], minimize: list[str],
                            eval_level: int | None, sweep_tag: str | None,
                            campaign_id: str | None = None,
                            objective_profile_id: str | None = None,
                            metric_source: str | None = None,
                            constraints_json: str | None = None,
                            knee_run_id: str | None = None,
                            knee_rationale_json: str | None = None) -> str | None:
    return service_save_frontier_snapshot(
        conn,
        front=front,
        dominated=dominated,
        maximize=maximize,
        minimize=minimize,
        eval_level=eval_level,
        sweep_tag=sweep_tag,
        campaign_id=campaign_id,
        objective_profile_id=objective_profile_id,
        metric_source=metric_source,
        constraints_json=constraints_json,
        knee_run_id=knee_run_id,
        knee_rationale_json=knee_rationale_json,
    )


# ---------------------------------------------------------------------------
# v15 E7: 对手晋升链
# ---------------------------------------------------------------------------

def cmd_promotion_chain(conn: sqlite3.Connection) -> None:
    """Print the opponent promotion chain (S0 → S1 → S2 → ...).

    Reads the `prev_alias` column added in v15 E5. Each opponent is shown
    with its source run, source checkpoint, eval level, and registered
    win rate. The chain is reconstructed by walking `prev_alias` pointers.
    """
    rows = conn.execute(
        "SELECT alias, source_run, source_tag, win_rate, eval_level, "
        "       num_res_blocks, num_filters, description, created_at, prev_alias "
        "FROM opponents ORDER BY created_at"
    ).fetchall()

    if not rows:
        print("No registered opponents.")
        return

    by_alias = {r["alias"]: dict(r) for r in rows}

    # Build chains from leaf nodes (those nobody references as prev_alias)
    referenced = {r["prev_alias"] for r in rows if r["prev_alias"]}
    leaves = [r["alias"] for r in rows if r["alias"] not in referenced]

    print(f"Promotion chains ({len(rows)} opponents)")
    print("─" * 78)
    for leaf in leaves:
        chain = []
        cur = leaf
        seen = set()
        while cur and cur in by_alias and cur not in seen:
            seen.add(cur)
            chain.append(by_alias[cur])
            cur = by_alias[cur].get("prev_alias")
        chain.reverse()  # root → ... → leaf

        for i, opp in enumerate(chain):
            arrow = "  " if i == 0 else "→ "
            arch = ""
            if opp.get("num_res_blocks") and opp.get("num_filters"):
                arch = f" {opp['num_res_blocks']}x{opp['num_filters']}"
            wr = f"{opp['win_rate']:.0%}" if opp.get("win_rate") is not None else "?"
            lvl = f"L{opp['eval_level']}" if opp.get("eval_level") is not None else "L?"
            print(f"  {arrow}{opp['alias']:<8}  WR {wr:>5} vs {lvl}{arch}  "
                  f"src={opp['source_run'][:8] if opp['source_run'] else '?'}/"
                  f"{opp['source_tag'] or '?'}")
            if opp.get("description"):
                print(f"        {opp['description']}")
        print()


# ---------------------------------------------------------------------------
# v15 E3: per-opening WR breakdown
# ---------------------------------------------------------------------------

def cmd_opening_breakdown(conn: sqlite3.Connection, run_id: str) -> None:
    """Show per-opening WR breakdown for all checkpoints of a run.

    Reads from the eval_breakdown table (v15 E3).
    """
    ckpts = conn.execute(
        "SELECT id, tag, cycle, win_rate, eval_unique_openings "
        "FROM checkpoints WHERE run_id LIKE ? ORDER BY cycle",
        (run_id + "%",),
    ).fetchall()

    if not ckpts:
        print(f"No checkpoints for run '{run_id}'")
        return

    print(f"Per-opening WR breakdown for run {run_id[:8]}*")
    print("─" * 78)
    for ck in ckpts:
        breakdown = conn.execute(
            "SELECT * FROM eval_breakdown WHERE checkpoint_id = ? "
            "ORDER BY opening_index",
            (ck["id"],),
        ).fetchall()
        if not breakdown:
            print(f"\n  {ck['tag']} (cycle {ck['cycle']}, WR {ck['win_rate']:.1%}): "
                  f"no breakdown stored")
            continue

        print(f"\n  {ck['tag']} (cycle {ck['cycle']}, total WR {ck['win_rate']:.1%}, "
              f"{ck['eval_unique_openings'] or '?'} unique games)")
        print(f"    {'idx':>3}  {'WR':>7}  {'W':>4} {'L':>4} {'D':>4}  "
              f"{'avgL':>5}  {'uniq':>4}  opening")
        for b in breakdown:
            n = b["wins"] + b["losses"] + b["draws"]
            wr = b["wins"] / n if n > 0 else 0.0
            print(f"    {b['opening_index']:>3}  {wr:>6.0%}  "
                  f"{b['wins']:>4} {b['losses']:>4} {b['draws']:>4}  "
                  f"{b['avg_length'] or 0:>5.1f}  {b['unique_games'] or 0:>4}  "
                  f"{b['opening_moves']}")


# ---------------------------------------------------------------------------
# 步数归一化对比
# ---------------------------------------------------------------------------

def cmd_compare_by_steps(conn: sqlite3.Connection, run_a: str, run_b: str) -> None:
    """按训练步数（而非时间）对齐两个 run 的 WR 曲线，用于公平对比不同算法。

    解决的问题：MCTS 每步耗时 19-35x，按时间对比不公平。按步数对齐后，
    可以判断相同梯度更新量下哪种算法产生了更好的 WR。
    """
    def _resolve(run_id):
        row = conn.execute(
            "SELECT id, mcts_simulations, num_res_blocks, num_filters, "
            "total_cycles, total_games, total_steps, final_win_rate, wall_time_s "
            "FROM runs WHERE id LIKE ?", (run_id + "%",)
        ).fetchone()
        return row

    ra = _resolve(run_a)
    rb = _resolve(run_b)
    if not ra:
        print(f"Run not found: {run_a}")
        return
    if not rb:
        print(f"Run not found: {run_b}")
        return

    def _get_wr_by_steps(full_id):
        rows = conn.execute(
            "SELECT total_steps, win_rate FROM cycle_metrics "
            "WHERE run_id = ? AND win_rate IS NOT NULL ORDER BY cycle",
            (full_id,)
        ).fetchall()
        return [(r["total_steps"], r["win_rate"]) for r in rows]

    points_a = _get_wr_by_steps(ra["id"])
    points_b = _get_wr_by_steps(rb["id"])

    if not points_a or not points_b:
        print("Insufficient WR data for comparison.")
        return

    mcts_a = ra["mcts_simulations"] or 0
    mcts_b = rb["mcts_simulations"] or 0
    label_a = f"{ra['id'][:8]} ({'MCTS-'+str(mcts_a) if mcts_a else 'Pure'})"
    label_b = f"{rb['id'][:8]} ({'MCTS-'+str(mcts_b) if mcts_b else 'Pure'})"

    print(f"Step-Normalized WR Comparison")
    print(f"{'─' * 72}")
    print(f"  A: {label_a}  ({ra['total_steps'] or 0} steps, {ra['total_games'] or 0} games, {ra['wall_time_s'] or 0:.0f}s)")
    print(f"  B: {label_b}  ({rb['total_steps'] or 0} steps, {rb['total_games'] or 0} games, {rb['wall_time_s'] or 0:.0f}s)")
    print()

    # Find common step range
    max_steps = min(
        points_a[-1][0] if points_a else 0,
        points_b[-1][0] if points_b else 0,
    )
    if max_steps == 0:
        print("  No overlapping step range.")
        return

    print(f"  Comparable range: 0 — {max_steps} steps")
    print()
    print(f"  {'Steps':>8}  {'WR_A':>7}  {'WR_B':>7}  {'Diff':>7}  {'Better':>8}")
    print(f"  {'─' * 48}")

    # Interpolate WR at common step checkpoints
    def _interp(points, target_step):
        """Linear interpolation of WR at a given step count."""
        if not points:
            return None
        if target_step <= points[0][0]:
            return points[0][1]
        if target_step >= points[-1][0]:
            return points[-1][1]
        for i in range(1, len(points)):
            if points[i][0] >= target_step:
                s0, w0 = points[i-1]
                s1, w1 = points[i]
                if s1 == s0:
                    return w1
                frac = (target_step - s0) / (s1 - s0)
                return w0 + frac * (w1 - w0)
        return points[-1][1]

    # Sample at regular step intervals
    n_samples = min(20, max(5, max_steps // 50))
    step_interval = max(1, max_steps // n_samples)
    a_wins = 0
    b_wins = 0
    for step in range(step_interval, max_steps + 1, step_interval):
        wa = _interp(points_a, step)
        wb = _interp(points_b, step)
        if wa is None or wb is None:
            continue
        diff = wa - wb
        if abs(diff) < 0.01:
            better = "="
        elif diff > 0:
            better = "A"
            a_wins += 1
        else:
            better = "B"
            b_wins += 1
        print(f"  {step:>8d}  {wa:>6.1%}  {wb:>6.1%}  {diff:>+6.1%}   {better:>6}")

    print()
    total = a_wins + b_wins
    if total > 0:
        print(f"  Summary: A better at {a_wins}/{total} checkpoints, B better at {b_wins}/{total}")
    print()


def cmd_report(conn: sqlite3.Connection, n_recent: int = 5, fmt: str = "md") -> None:
    """生成结构化实验报告（双格式）。"""
    data = service_gather_report_data(conn, n_recent)
    if fmt == "json":
        print(service_format_report_json(data), end="")
    else:
        print(service_format_report_md(data), end="")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="autoresearch 训练分析工具（只读）")
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
    group.add_argument("--list-campaigns", action="store_true",
                       help="List campaign ledgers")
    group.add_argument("--campaign-summary", metavar="CAMPAIGN",
                       help="Show one campaign summary")
    group.add_argument("--stage-summary", metavar="CAMPAIGN",
                       help="Show stage-by-stage summary for a campaign (v20.2)")
    group.add_argument("--promotion-log", metavar="CAMPAIGN",
                       help="Show promotion decisions for a campaign (v20.2)")
    group.add_argument("--stagnation", metavar="RUN_ID",
                       help="Detect training stagnation (WR plateau) for a run")
    group.add_argument("--pareto", action="store_true",
                       help="Pareto non-dominated sort across completed runs")
    group.add_argument("--compare-by-steps", nargs=2, metavar=("RUN_A", "RUN_B"),
                       help="Compare two runs by training steps (not wall time)")
    group.add_argument("--promotion-chain", action="store_true",
                       help="Print the S0 → S1 → S2 → ... opponent promotion chain")
    group.add_argument("--opening-breakdown", metavar="RUN_ID",
                       help="Per-opening WR breakdown for a run's checkpoints")
    group.add_argument("--report", action="store_true",
                       help="Generate structured experiment report for agent/human")
    group.add_argument("--branch-tree", metavar="CAMPAIGN",
                       help="Show parent/child branch tree for a campaign (v20.3)")
    group.add_argument("--trajectory-report", metavar="CAMPAIGN",
                       help="Show trajectory report for a campaign (v20.3)")
    group.add_argument("--compare-parent-child", metavar="BRANCH_ID",
                       help="Compare parent and child for a branch (v20.3)")
    group.add_argument("--recommend-next", metavar="CAMPAIGN",
                       help="Recommend next point / branch candidates for a campaign (v21)")
    group.add_argument("--recommendation-log", metavar="CAMPAIGN",
                       help="Show recommendation history for a campaign (v21)")
    group.add_argument("--recommendation-outcomes", metavar="CAMPAIGN",
                       help="Show recommendation outcome summary for a campaign (v21)")
    group.add_argument("--acquisition-summary", metavar="CAMPAIGN",
                       help="Show acquisition lineage summary for a campaign (v21.1)")

    parser.add_argument("--format", choices=["md", "json"], default="md",
                        help="Report format: md (Chinese markdown) or json (structured)")
    parser.add_argument("--recent", type=int, default=5,
                        help="Number of recent runs in report (default: 5)")

    # v20: Pareto configuration
    parser.add_argument("--plot", action="store_true",
                        help="Generate PNG scatter plot (used with --pareto)")
    parser.add_argument("--maximize", type=str, default=None,
                        help="Comma-separated axes to maximize (default: wr)")
    parser.add_argument("--minimize", type=str, default=None,
                        help="Comma-separated axes to minimize (default: params,wall_s)")
    parser.add_argument("--eval-level", type=int, default=None,
                        help="Filter runs by eval_level for Pareto analysis")
    parser.add_argument("--sweep-tag", type=str, default=None,
                        help="Filter runs by sweep_tag prefix for Pareto analysis")
    parser.add_argument("--campaign", type=str, default=None,
                        help="Filter Pareto/matrix to a specific campaign")
    parser.add_argument("--stage", type=str, default=None,
                        help="Filter Pareto/matrix to a campaign stage (requires --campaign)")
    parser.add_argument("--allow-drift", action="store_true",
                        help="Allow protocol-drift runs inside campaign-scoped analyses")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for plot PNG (default: output/pareto_front.png)")
    parser.add_argument("--objective-profile", type=str, default=None,
                        help="Objective profile path/id for domain-generic Pareto analysis")
    parser.add_argument("--metric-source", choices=["legacy", "run_metrics"], default="legacy",
                        help="Metric source for Pareto analysis (default: legacy)")
    parser.add_argument("--knee", action="store_true",
                        help="Report a normalized utopia-distance knee point for --pareto")

    # v21: selector configuration
    parser.add_argument("--selector-policy", type=str, default=None,
                        help="Path to selector policy JSON (used with --recommend-next)")
    parser.add_argument("--acquisition-policy", type=str, default=None,
                        help="Path to acquisition policy JSON (used with --recommend-next)")
    parser.add_argument("--candidate-type", type=str, default=None,
                        help="Filter recommendations by candidate type (new_point, seed_recheck, continue_branch, eval_upgrade)")
    parser.add_argument("--limit", type=int, default=5,
                        help="Maximum number of recommendations to output (default: 5)")

    args = parser.parse_args()
    conn = _connect(args.db)

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
        cmd_matrix(conn, args.matrix, campaign=args.campaign, allow_drift=args.allow_drift)
    elif args.list_campaigns:
        cmd_list_campaigns(conn)
    elif args.campaign_summary:
        cmd_campaign_summary(conn, args.campaign_summary)
    elif args.stage_summary:
        cmd_stage_summary(conn, args.stage_summary)
    elif args.promotion_log:
        cmd_promotion_log(conn, args.promotion_log)
    elif args.stagnation:
        cmd_stagnation(conn, args.stagnation)
    elif args.pareto:
        max_axes = args.maximize.split(",") if args.maximize else None
        min_axes = args.minimize.split(",") if args.minimize else None
        cmd_pareto(conn, fmt=args.format,
                   maximize=max_axes, minimize=min_axes,
                   eval_level=args.eval_level, sweep_tag=args.sweep_tag,
                   campaign=args.campaign, allow_drift=args.allow_drift,
                   plot=args.plot, output_path=args.output,
                   stage=args.stage,
                   objective_profile=args.objective_profile,
                   metric_source=args.metric_source,
                   show_knee=args.knee)
    elif args.compare_by_steps:
        cmd_compare_by_steps(conn, args.compare_by_steps[0], args.compare_by_steps[1])
    elif args.promotion_chain:
        cmd_promotion_chain(conn)
    elif args.opening_breakdown:
        cmd_opening_breakdown(conn, args.opening_breakdown)
    elif args.report:
        cmd_report(conn, n_recent=args.recent, fmt=args.format)
    elif args.branch_tree:
        cmd_branch_tree(conn, args.branch_tree)
    elif args.trajectory_report:
        cmd_trajectory_report(conn, args.trajectory_report)
    elif args.compare_parent_child:
        cmd_compare_parent_child(conn, args.compare_parent_child)
    elif args.recommend_next:
        cmd_recommend_next(conn, args.recommend_next,
                           selector_policy=args.selector_policy,
                           acquisition_policy=args.acquisition_policy,
                           candidate_type=args.candidate_type,
                           limit=args.limit,
                           fmt=args.format)
    elif args.recommendation_log:
        cmd_recommendation_log(conn, args.recommendation_log, fmt=args.format)
    elif args.recommendation_outcomes:
        cmd_recommendation_outcomes(conn, args.recommendation_outcomes, fmt=args.format)
    elif args.acquisition_summary:
        cmd_acquisition_summary(conn, args.acquisition_summary, fmt=args.format)

    conn.close()


if __name__ == "__main__":
    main()
