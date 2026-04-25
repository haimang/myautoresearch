#!/usr/bin/env python3
"""autoresearch 超参数扫描工具 — 批量实验。"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))

from core.db import (
    DB_PATH,
    find_run_by_sweep_tag,
    get_campaign,
    get_or_create_campaign,
    get_campaign_stages,
    get_recommendation_by_id,
    init_db,
    link_run_to_campaign,
    link_run_to_campaign_v20,
    save_recommendation_outcome,
    save_campaign_stage,
    save_objective_profile,
    save_experiment_run,
    save_search_space,
    update_recommendation_status,
)
from objective_profile import load_objective_profile
from search_space import describe_profile, load_profile, validate_selected_axes
from stage_policy import load_stage_policy, get_stage_by_name
from selector_policy import load_selector_policy, get_candidate_kind_config


def parse_args():
    p = argparse.ArgumentParser(description="autoresearch 超参数扫描工具")
    p.add_argument("--db", type=str, default=DB_PATH,
                   help="tracker.db 路径（默认: output/tracker.db）")

    # 训练脚本路径 — 领域无关
    p.add_argument("--train-script", type=str, required=False,
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
    p.add_argument("--axis", action="append", default=None,
                   help="Generic dynamic axis as name=v1,v2 (v22)")
    p.add_argument("--candidate-json", type=str, default=None,
                   help="Inline JSON object or path to a candidate JSON file (v22)")

    # 固定参数，应用于每次运行
    p.add_argument("--time-budget", type=int, default=None,
                   help="每次运行的时间预算（秒）；若指定 --stage，则优先取 stage policy")
    p.add_argument("--seeds", type=str, default="42",
                   help="逗号分隔的随机种子 (默认: 42)")
    p.add_argument("--tag", type=str, default="sweep",
                   help="运行分组的标签前缀（默认: sweep；若传 --campaign 且未改 tag，则自动取 campaign 名）")
    p.add_argument("--campaign", type=str, default=None,
                   help="本轮研究的 campaign 名称")
    p.add_argument("--search-space", type=str, default=None,
                    help="JSON search-space profile 路径（v20.1）")
    p.add_argument("--objective-profile", type=str, default=None,
                   help="JSON objective profile path for domain-generic metrics (v22)")
    p.add_argument("--run-id", type=str, default=None,
                   help="Filesystem-level experiment run id for run-scoped artifacts (v23)")
    p.add_argument("--output-root", type=str, default="output",
                   help="Root directory for run-scoped artifacts (default: output)")
    p.add_argument("--stage-policy", type=str, default=None,
                   help="JSON stage-policy 路径（v20.2）")
    p.add_argument("--stage", type=str, default=None,
                   help="当前 campaign stage: A / B / C / D（v20.2）")

    # 额外固定参数，透传至 train.py
    p.add_argument("--eval-level", type=int, default=None)
    p.add_argument("--eval-opponent", type=str, default=None)
    p.add_argument("--parallel-games", type=int, default=None)
    p.add_argument("--target-win-rate", type=float, default=None)

    # 控制
    p.add_argument("--resume", action="store_true",
                   help="跳过 tracker.db 中已存在的已完成配置")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印矩阵，不实际运行")
    p.add_argument("--no-auto-pareto", action="store_true",
                   help="完成后不自动生成 Pareto 分析和图表")
    p.add_argument("--execute-recommendation", type=str, default=None,
                   help="Execute an accepted point recommendation by id (v21.1)")

    return p.parse_args()


def parse_csv(val, dtype):
    if val is None:
        return [None]
    return [dtype(v.strip()) for v in val.split(",")]


def get_completed_tags(db_path: str) -> set[str]:
    if not os.path.exists(db_path):
        return set()
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT sweep_tag FROM runs "
            "WHERE status IN ('completed', 'time_budget', 'target_win_rate', 'target_games') "
            "AND sweep_tag IS NOT NULL"
        ).fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()


def get_completed_campaign_tags(db_path: str, campaign_id: str) -> set[str]:
    if not os.path.exists(db_path):
        return set()
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """SELECT cr.sweep_tag
               FROM campaign_runs cr
               JOIN runs r ON r.id = cr.run_id
               WHERE cr.campaign_id = ?
                 AND r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')
                 AND cr.sweep_tag IS NOT NULL""",
            (campaign_id,),
        ).fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()


def _parse_axis_values(name: str, raw: str, profile: dict | None):
    axis = (profile or {}).get("axes", {}).get(name, {})
    axis_type = axis.get("type")
    values = []
    for part in raw.split(","):
        text = part.strip()
        if axis_type == "int":
            values.append(int(text))
        elif axis_type == "float":
            values.append(float(text))
        else:
            if text.lower() == "true":
                values.append(True)
            elif text.lower() == "false":
                values.append(False)
            else:
                try:
                    values.append(json.loads(text))
                except json.JSONDecodeError:
                    values.append(text)
    return values


def _load_candidate_json_arg(raw: str) -> dict:
    if os.path.isfile(raw):
        with open(raw, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--candidate-json must resolve to a JSON object")
    return data


def _stable_candidate_key(payload: dict) -> str:
    volatile = {"quote_id", "valid_from_at", "valid_to_at", "created_at", "latency", "quote_latency_ms"}
    clean = {k: v for k, v in payload.items() if k not in volatile}
    return json.dumps(clean, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def build_matrix(args, profile: dict | None = None):
    """从扫描轴生成配置字典列表。"""
    if args.candidate_json:
        payload = _load_candidate_json_arg(args.candidate_json)
        seed = int(payload.get("seed", parse_csv(args.seeds, int)[0]))
        tag = payload.get("sweep_tag") or f"{args.tag}_candidate_sd{seed}"
        return [{
            **payload,
            "seed": seed,
            "sweep_tag": tag,
            "__candidate_payload": payload,
            "__candidate_key": _stable_candidate_key(payload),
        }]

    blocks = parse_csv(args.num_blocks, int)
    filters_ = parse_csv(args.num_filters, int)
    lrs = parse_csv(args.learning_rate, float)
    steps = parse_csv(args.steps_per_cycle, int)
    bufs = parse_csv(args.buffer_size, int)
    seeds = parse_csv(args.seeds, int)

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
    if args.axis:
        for spec in args.axis:
            if "=" not in spec:
                raise ValueError(f"--axis must use name=v1,v2 syntax, got: {spec}")
            name, raw_values = spec.split("=", 1)
            name = name.strip()
            if not name:
                raise ValueError("--axis name cannot be empty")
            if name in axis_names:
                raise ValueError(f"duplicate axis specified: {name}")
            axis_names.append(name)
            axis_values.append(_parse_axis_values(name, raw_values, profile))

    if not axis_names:
        print("错误: 至少指定一个扫描轴 (--num-blocks, --num-filters, --learning-rate, --steps-per-cycle, --buffer-size)")
        sys.exit(1)

    configs = []
    for combo in itertools.product(*axis_values):
        cfg = dict(zip(axis_names, combo))
        for seed in seeds:
            parts = [args.tag]
            for name, val in cfg.items():
                short = {
                    "num_blocks": "b",
                    "num_filters": "f",
                    "learning_rate": "lr",
                    "steps_per_cycle": "s",
                    "buffer_size": "buf",
                }.get(name, name)
                parts.append(f"{short}{val}")
            parts.append(f"sd{seed}")
            tag = "_".join(parts)
            candidate_payload = {k: v for k, v in cfg.items()}
            configs.append({
                **cfg,
                "seed": seed,
                "sweep_tag": tag,
                "__candidate_payload": candidate_payload if args.axis else None,
                "__candidate_key": _stable_candidate_key(candidate_payload) if args.axis else None,
            })
    return configs


def _filter_fx_degenerate_routes(configs: list[dict], profile: dict | None) -> list[dict]:
    if not profile or profile.get("domain") != "fx_spot":
        return configs
    out = []
    for cfg in configs:
        template = cfg.get("route_template")
        if isinstance(template, str) and template.startswith("via_"):
            bridge = template[4:].upper()
            sell = str(cfg.get("sell_currency", "")).upper()
            buy = str(cfg.get("buy_currency", "")).upper()
            if bridge in (sell, buy):
                continue
        out.append(cfg)
    return out


def _maybe_setup_run_workspace(args, profile: dict | None, objective_profile: dict | None) -> dict | None:
    """Create a v23 run-scoped workspace for fx_spot sweeps."""
    if not profile or profile.get("domain") != "fx_spot":
        return None
    run_id = args.run_id or datetime.now(timezone.utc).strftime("fx-%Y%m%d-%H%M%S")
    workspace = os.path.join(args.output_root, "fx_spot", run_id)
    os.makedirs(workspace, exist_ok=True)
    os.makedirs(os.path.join(workspace, "logs"), exist_ok=True)
    os.makedirs(os.path.join(workspace, "campaigns"), exist_ok=True)
    if args.db == DB_PATH:
        args.db = os.path.join(workspace, "tracker.db")
    manifest = {
        "fx_run_id": run_id,
        "domain": "fx_spot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign": args.campaign,
        "search_space": args.search_space,
        "objective_profile": args.objective_profile,
        "search_space_hash": profile.get("profile_hash"),
        "objective_profile_hash": (objective_profile or {}).get("profile_hash"),
        "db": args.db,
    }
    with open(os.path.join(workspace, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    args.run_id = run_id
    args._artifact_root = workspace
    return manifest


def _derive_protocol(args, profile: dict | None) -> dict:
    profile_protocol = (profile or {}).get("protocol", {})
    eval_level = args.eval_level if args.eval_level is not None else profile_protocol.get("eval_level")
    eval_opponent = args.eval_opponent if args.eval_opponent is not None else profile_protocol.get("eval_opponent")
    is_benchmark = bool(profile_protocol.get("is_benchmark", False))
    if args.eval_opponent is None and eval_opponent is not None:
        # profile-defined NN opponent keeps exploratory semantics
        is_benchmark = False
    if eval_opponent is None and eval_level is not None:
        is_benchmark = True
    return {
        "eval_level": eval_level,
        "eval_opponent": eval_opponent,
        "is_benchmark": is_benchmark,
        "train_script": args.train_script,
    }


def _ensure_profile_protocol(profile: dict, protocol: dict) -> None:
    expected = profile["protocol"]
    mismatches = []
    for key in ("eval_level", "eval_opponent", "is_benchmark"):
        if expected.get(key) != protocol.get(key):
            mismatches.append(f"{key}={protocol.get(key)!r} != profile {expected.get(key)!r}")
    if mismatches:
        raise ValueError("protocol does not match search-space profile: " + "; ".join(mismatches))


def _ensure_campaign(conn, args, profile: dict | None, protocol: dict, objective_profile: dict | None = None):
    if not args.campaign:
        return None, None
    if profile is None:
        raise ValueError("--campaign requires --search-space")

    search_space_id = save_search_space(conn, profile)
    objective_profile_id = save_objective_profile(conn, objective_profile) if objective_profile else None
    campaign = get_or_create_campaign(
        conn,
        name=args.campaign,
        domain=profile["domain"],
        train_script=args.train_script,
        search_space_id=search_space_id,
        protocol=protocol,
        objective_profile_id=objective_profile_id,
    )
    if getattr(args, "run_id", None):
        conn.execute(
            "UPDATE campaigns SET experiment_run_id = ? WHERE id = ?",
            (args.run_id, campaign["id"]),
        )
        conn.commit()
        campaign = dict(get_campaign(conn, campaign["id"]))
    return campaign, search_space_id


def run_one(cfg, args, idx, total, campaign=None):
    """执行单个训练配置。返回 (tag, 是否成功, 耗时)。"""
    tag = cfg["sweep_tag"]
    seed = cfg["seed"]

    cmd = [sys.executable, args.train_script,
           "--time-budget", str(args.time_budget),
           "--seed", str(seed),
           "--sweep-tag", tag]
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
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  失败 (退出码 {proc.returncode})")
        if proc.stderr:
            lines = proc.stderr.strip().split("\n")
            for line in lines[-5:]:
                print(f"  {line}")
        return tag, False, elapsed

    wr_line = ""
    for line in proc.stdout.splitlines():
        if "Win rate:" in line or "win_rate" in line.lower():
            wr_line = line.strip()
    if wr_line:
        print(f"  {wr_line}")
    print(f"  完成，耗时 {elapsed:.0f}s")

    return tag, True, elapsed


def _print_matrix_preview(configs, profile: dict | None, args, protocol: dict):
    if profile is not None:
        print(describe_profile(profile))
        print()
    if args.campaign:
        print(f"Campaign: {args.campaign}")
        print(f"Protocol: {json.dumps(protocol, ensure_ascii=False, sort_keys=True)}")
        print()
    print(f"{'Tag':<55} {'Params'}")
    print("-" * 96)
    for c in configs:
        axis_desc = "  ".join(f"{k}={v}" for k, v in c.items() if not k.startswith("__") and k not in ("seed", "sweep_tag"))
        print(f"{c['sweep_tag']:<55} {axis_desc}")


def _infer_recommendation_stage(conn, campaign_id: str) -> str | None:
    stages = get_campaign_stages(conn, campaign_id)
    if stages:
        open_stages = [s for s in stages if s.get("status") == "open"]
        source = open_stages if open_stages else stages
        source = sorted(source, key=lambda s: s["stage"], reverse=True)
        return source[0]["stage"]
    row = conn.execute(
        "SELECT stage FROM campaign_runs WHERE campaign_id = ? AND stage IS NOT NULL ORDER BY stage DESC LIMIT 1",
        (campaign_id,),
    ).fetchone()
    return row["stage"] if row else None


def _next_seed_for_candidate(conn, campaign_id: str, candidate_key: str | None) -> int:
    if not candidate_key:
        return 1
    rows = conn.execute(
        """SELECT DISTINCT seed FROM campaign_runs
           WHERE campaign_id = ? AND candidate_key = ? AND seed IS NOT NULL
           ORDER BY seed""",
        (campaign_id, candidate_key),
    ).fetchall()
    used = {r["seed"] for r in rows if r["seed"] is not None}
    seed = 1
    while seed in used:
        seed += 1
    return seed


def _default_budget_for_recommendation(campaign: dict, candidate_type: str) -> int:
    selector_path = os.path.join(_PROJECT_ROOT, "domains", campaign["domain"], "selector_policy.json")
    if os.path.isfile(selector_path):
        policy = load_selector_policy(selector_path)
        cfg = get_candidate_kind_config(policy, candidate_type)
        if cfg and cfg.get("default_budget_s"):
            return int(cfg["default_budget_s"])
    return 60


def _build_frontier_delta(conn, campaign_id: str, previous_best_wr: float | None, run_id: str) -> tuple[str, str, str]:
    row = conn.execute(
        """SELECT final_win_rate, wall_time_s, num_params, total_games, status
           FROM runs WHERE id = ?""",
        (run_id,),
    ).fetchone()
    observed = {
        "final_win_rate": row["final_win_rate"],
        "wall_time_s": row["wall_time_s"],
        "num_params": row["num_params"],
        "total_games": row["total_games"],
        "status": row["status"],
    }
    latest_best = conn.execute(
        """SELECT MAX(r.final_win_rate) AS best_wr
           FROM campaign_runs cr
           JOIN runs r ON r.id = cr.run_id
           WHERE cr.campaign_id = ?
             AND r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')""",
        (campaign_id,),
    ).fetchone()
    new_best_wr = latest_best["best_wr"] if latest_best else None
    old_best_wr = previous_best_wr or 0.0
    current_wr = observed["final_win_rate"] or 0.0
    if current_wr > old_best_wr:
        outcome_label = "new_front"
    elif current_wr >= max(0.0, old_best_wr - 0.02):
        outcome_label = "near_front"
    else:
        outcome_label = "no_gain"
    frontier_delta = {
        "old_best_wr": old_best_wr,
        "new_best_wr": new_best_wr,
        "delta": round((new_best_wr or 0.0) - old_best_wr, 6),
    }
    return json.dumps(observed, ensure_ascii=False, sort_keys=True), json.dumps(frontier_delta, ensure_ascii=False, sort_keys=True), outcome_label


def _execute_point_recommendation(args) -> None:
    conn = init_db(args.db)
    rec = get_recommendation_by_id(conn, args.execute_recommendation)
    if not rec:
        print(f"Recommendation not found: {args.execute_recommendation}")
        conn.close()
        sys.exit(1)
    if rec["status"] != "accepted":
        print(f"Recommendation {rec['id']} status is '{rec['status']}', expected 'accepted'")
        conn.close()
        sys.exit(1)
    if rec["candidate_type"] not in ("new_point", "seed_recheck"):
        print(f"Recommendation {rec['id']} is '{rec['candidate_type']}', use branch.py for branch execution")
        conn.close()
        sys.exit(1)

    campaign_row = get_campaign(conn, rec["campaign_id"])
    if not campaign_row:
        print(f"Campaign not found for recommendation {rec['id']}")
        conn.close()
        sys.exit(1)
    campaign = dict(campaign_row)

    if not args.train_script:
        args.train_script = campaign["train_script"]
    if not args.train_script:
        print("Error: train_script is required for recommendation execution")
        conn.close()
        sys.exit(1)

    args.time_budget = args.time_budget or _default_budget_for_recommendation(campaign, rec["candidate_type"])
    stage = args.stage or _infer_recommendation_stage(conn, campaign["id"])

    axis_values = json.loads(rec.get("axis_values_json") or "{}")
    if not axis_values and rec.get("candidate_key"):
        try:
            axis_values = json.loads(rec["candidate_key"])
        except json.JSONDecodeError:
            axis_values = {}

    seed = _next_seed_for_candidate(conn, campaign["id"], rec.get("candidate_key"))
    tag = f"{campaign['name']}_rec_{rec['id'][:8]}_sd{seed}"
    cfg = {
        **axis_values,
        "seed": seed,
        "sweep_tag": tag,
    }
    if campaign.get("objective_profile_id"):
        cfg["__candidate_payload"] = axis_values
        cfg["__candidate_key"] = _stable_candidate_key(axis_values)

    previous_best = conn.execute(
        """SELECT MAX(r.final_win_rate) AS best_wr
           FROM campaign_runs cr
           JOIN runs r ON r.id = cr.run_id
           WHERE cr.campaign_id = ?
             AND r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')""",
        (campaign["id"],),
    ).fetchone()
    previous_best_wr = previous_best["best_wr"] if previous_best else None

    _, ok, _ = run_one(cfg, args, 1, 1, campaign=campaign)
    run_id = find_run_by_sweep_tag(conn, tag)
    if run_id:
        axis_identity = {k: v for k, v in cfg.items() if k not in ("seed", "sweep_tag")}
        link_run_to_campaign_v20(
            conn,
            campaign_id=campaign["id"],
            run_id=run_id,
            stage=stage,
            sweep_tag=tag,
            seed=seed,
            axis_values=axis_identity,
            status="linked" if ok else "failed",
        )

    if ok and run_id:
        update_recommendation_status(conn, recommendation_id=rec["id"], status="executed")
        observed_json, frontier_delta_json, outcome_label = _build_frontier_delta(
            conn, campaign["id"], previous_best_wr, run_id
        )
        save_recommendation_outcome(
            conn,
            recommendation_id=rec["id"],
            run_id=run_id,
            observed_metrics_json=observed_json,
            frontier_delta_json=frontier_delta_json,
            outcome_label=outcome_label,
        )
        print(f"Recommendation executed: {rec['id']} -> run {run_id[:8]} ({outcome_label})")
    else:
        print(f"Recommendation execution failed: {rec['id']}")

    conn.close()


def main():
    args = parse_args()

    if args.execute_recommendation:
        _execute_point_recommendation(args)
        return

    if args.campaign and args.tag == "sweep":
        args.tag = args.campaign

    if not args.train_script:
        print("Error: --train-script is required unless --execute-recommendation is used")
        sys.exit(1)

    stage_policy = None
    stage_cfg = None
    try:
        profile = load_profile(args.search_space) if args.search_space else None
        objective_profile = load_objective_profile(args.objective_profile) if args.objective_profile else None
        if objective_profile and profile and objective_profile["domain"] != profile["domain"]:
            raise ValueError(
                f"objective profile domain '{objective_profile['domain']}' != search-space domain '{profile['domain']}'"
            )
        configs = build_matrix(args, profile)
        configs = _filter_fx_degenerate_routes(configs, profile)

        selected_axes = {}
        for key in ("num_blocks", "num_filters", "learning_rate", "steps_per_cycle", "buffer_size"):
            raw = getattr(args, key)
            if raw is not None:
                dtype = float if key == "learning_rate" else int
                selected_axes[key] = parse_csv(raw, dtype)
        if args.axis:
            for spec in args.axis:
                name, raw_values = spec.split("=", 1)
                selected_axes[name.strip()] = _parse_axis_values(name.strip(), raw_values, profile)
        if profile is not None:
            validate_selected_axes(profile, selected_axes)

        protocol = _derive_protocol(args, profile)
        if profile is not None:
            _ensure_profile_protocol(profile, protocol)

        # v20.2: stage policy
        if args.stage_policy:
            stage_policy = load_stage_policy(args.stage_policy)
            if args.stage:
                stage_cfg = get_stage_by_name(stage_policy, args.stage)
                if stage_cfg is None:
                    raise ValueError(f"stage '{args.stage}' not found in policy")
                # Override time_budget from stage policy
                args.time_budget = stage_cfg["time_budget"]
                print(f"[v20.2] Stage {args.stage} policy applied: budget={args.time_budget}s")
            else:
                print(f"[v20.2] Stage policy loaded; use --stage to select A/B/C/D")

        # Validate time_budget is resolved
        if args.time_budget is None:
            raise ValueError("--time-budget is required when --stage is not specified")

        workspace_manifest = _maybe_setup_run_workspace(args, profile, objective_profile)
        db_conn = init_db(args.db)
        campaign = None
        if args.campaign:
            campaign, _ = _ensure_campaign(db_conn, args, profile, protocol, objective_profile)
            # v20.2: write campaign stage record
            if stage_cfg:
                save_campaign_stage(
                    db_conn,
                    campaign_id=campaign["id"],
                    stage=args.stage,
                    policy_json=json.dumps(stage_cfg, ensure_ascii=False, sort_keys=True),
                    budget_json=json.dumps({"time_budget": stage_cfg["time_budget"]}, ensure_ascii=False, sort_keys=True),
                    seed_target=stage_cfg["seed_count"],
                    status="open",
                )
        if workspace_manifest and args.run_id:
            save_experiment_run(
                db_conn,
                run_id=args.run_id,
                domain=profile["domain"],
                output_root=getattr(args, "_artifact_root"),
                manifest=workspace_manifest,
                objective_profile_id=campaign.get("objective_profile_id") if campaign else None,
            )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if args.resume:
        if campaign:
            completed = get_completed_campaign_tags(args.db, campaign["id"])
        else:
            completed = get_completed_tags(args.db)
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
    if campaign:
        print(f"Campaign: {campaign['name']} ({campaign['id'][:8]})")
        if args.stage:
            print(f"Stage: {args.stage}")

    if args.dry_run:
        _print_matrix_preview(configs, profile, args, protocol)
        if stage_cfg:
            print()
            print(f"Stage policy: {stage_cfg['name']} | budget={stage_cfg['time_budget']}s | seeds={stage_cfg['seed_count']} | promote_top_k={stage_cfg['promote_top_k']}")
        return

    results = []
    sweep_start = time.time()
    for i, cfg in enumerate(configs, 1):
        tag, ok, elapsed = run_one(cfg, args, i, total, campaign=campaign)
        results.append((tag, ok, elapsed))
        if campaign:
            run_id = find_run_by_sweep_tag(db_conn, tag)
            if run_id:
                axis_values = {k: v for k, v in cfg.items() if not k.startswith("__") and k not in ("seed", "sweep_tag")}
                link_run_to_campaign_v20(
                    db_conn,
                    campaign_id=campaign["id"],
                    run_id=run_id,
                    stage=args.stage,
                    sweep_tag=tag,
                    seed=cfg["seed"],
                    axis_values=axis_values,
                    status="linked" if ok else "failed",
                )

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
    if campaign:
        print(f"Campaign 汇总: uv run python framework/analyze.py --campaign-summary {campaign['name']}")
        if args.stage:
            print(f"Stage 汇总: uv run python framework/analyze.py --stage-summary {campaign['name']}")

    if n_ok > 0 and not args.no_auto_pareto:
        print(f"\n{'─'*60}")
        print("自动 Pareto 分析...")
        print(f"{'─'*60}\n")
        plot_key = campaign["name"] if campaign else args.tag
        if getattr(args, "_artifact_root", None) and campaign:
            pareto_dir = os.path.join(args._artifact_root, "campaigns", campaign["name"], "pareto")
            os.makedirs(pareto_dir, exist_ok=True)
            pareto_plot_path = os.path.join(pareto_dir, "overview.png")
        else:
            pareto_plot_path = f"output/pareto_{plot_key}.png"
        pareto_cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), "analyze.py"),
            "--pareto", "--plot", "--output", pareto_plot_path,
            "--db", args.db,
        ]
        if campaign:
            pareto_cmd += ["--campaign", campaign["name"]]
            if args.stage:
                pareto_cmd += ["--stage", args.stage]
            if campaign.get("objective_profile_id"):
                pareto_cmd += ["--metric-source", "run_metrics"]
        else:
            pareto_cmd += ["--sweep-tag", args.tag]
        proc = subprocess.run(pareto_cmd, capture_output=False, text=True)
        if proc.returncode == 0:
            print(f"\n查看 Pareto 图: open {pareto_plot_path}")
        else:
            print(f"\n⚠  Pareto 分析执行失败 (退出码 {proc.returncode})")

    db_conn.close()


if __name__ == "__main__":
    main()
