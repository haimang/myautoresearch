#!/usr/bin/env python3
"""autoresearch 超参数扫描工具 — 批量实验。"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FRAMEWORK_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _FRAMEWORK_DIR not in sys.path:
    sys.path.insert(0, _FRAMEWORK_DIR)

from framework.core.db import (
    DB_PATH,
    find_run_by_sweep_tag,
    get_or_create_campaign,
    init_db,
    link_run_to_campaign,
    link_run_to_campaign_v20,
    save_campaign_stage,
    save_objective_profile,
    save_experiment_run,
    save_search_space,
)
from framework.profiles.objective_profile import load_objective_profile
from framework.profiles.search_space import describe_profile, load_profile, validate_selected_axes
from framework.services.execution.matrix import (
    build_matrix as service_build_matrix,
    get_completed_campaign_tags as service_get_completed_campaign_tags,
    get_completed_tags as service_get_completed_tags,
    load_candidate_json_arg as service_load_candidate_json_arg,
    parse_axis_values as service_parse_axis_values,
    parse_csv as service_parse_csv,
    stable_candidate_key as service_stable_candidate_key,
)
from framework.services.execution.recommendation_execution import (
    execute_point_recommendation as service_execute_point_recommendation,
)
from framework.services.execution.sweep_runner import (
    print_matrix_preview as service_print_matrix_preview,
    run_one as service_run_one,
)
from framework.services.execution.workspace import (
    derive_protocol as service_derive_protocol,
    ensure_campaign as service_ensure_campaign,
    ensure_profile_protocol as service_ensure_profile_protocol,
    filter_fx_degenerate_routes as service_filter_fx_degenerate_routes,
    maybe_setup_run_workspace as service_maybe_setup_run_workspace,
)
from framework.policies.stage_policy import load_stage_policy, get_stage_by_name


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
    return service_parse_csv(val, dtype)


def get_completed_tags(db_path: str) -> set[str]:
    return service_get_completed_tags(db_path)


def get_completed_campaign_tags(db_path: str, campaign_id: str) -> set[str]:
    return service_get_completed_campaign_tags(db_path, campaign_id)


def _parse_axis_values(name: str, raw: str, profile: dict | None):
    return service_parse_axis_values(name, raw, profile)


def _load_candidate_json_arg(raw: str) -> dict:
    return service_load_candidate_json_arg(raw)


def _stable_candidate_key(payload: dict) -> str:
    return service_stable_candidate_key(payload)


def build_matrix(args, profile: dict | None = None):
    return service_build_matrix(args, profile)


def _filter_fx_degenerate_routes(configs: list[dict], profile: dict | None) -> list[dict]:
    return service_filter_fx_degenerate_routes(configs, profile)


def _maybe_setup_run_workspace(args, profile: dict | None, objective_profile: dict | None) -> dict | None:
    return service_maybe_setup_run_workspace(args, profile, objective_profile)


def _derive_protocol(args, profile: dict | None) -> dict:
    return service_derive_protocol(args, profile)


def _ensure_profile_protocol(profile: dict, protocol: dict) -> None:
    service_ensure_profile_protocol(profile, protocol)


def _ensure_campaign(conn, args, profile: dict | None, protocol: dict, objective_profile: dict | None = None):
    return service_ensure_campaign(conn, args, profile, protocol, objective_profile)


def run_one(cfg, args, idx, total, campaign=None):
    return service_run_one(cfg, args, idx, total, campaign=campaign)


def _print_matrix_preview(configs, profile: dict | None, args, protocol: dict):
    service_print_matrix_preview(configs, profile, args, protocol)


def _execute_point_recommendation(args) -> None:
    conn = init_db(args.db)
    try:
        service_execute_point_recommendation(conn, args, project_root=_PROJECT_ROOT)
    finally:
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

    print(f"\n查看结果: uv run python framework/index.py analyze --matrix {args.tag}")
    if campaign:
        print(f"Campaign 汇总: uv run python framework/index.py analyze --campaign-summary {campaign['name']}")
        if args.stage:
            print(f"Stage 汇总: uv run python framework/index.py analyze --stage-summary {campaign['name']}")

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
