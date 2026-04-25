"""Matrix-building helpers extracted from the sweep CLI."""

from __future__ import annotations

import itertools
import json
import os
import sqlite3
import sys


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


def parse_axis_values(name: str, raw: str, profile: dict | None):
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


def load_candidate_json_arg(raw: str) -> dict:
    if os.path.isfile(raw):
        with open(raw, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--candidate-json must resolve to a JSON object")
    return data


def stable_candidate_key(payload: dict) -> str:
    volatile = {"quote_id", "valid_from_at", "valid_to_at", "created_at", "latency", "quote_latency_ms"}
    clean = {k: v for k, v in payload.items() if k not in volatile}
    return json.dumps(clean, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def build_matrix(args, profile: dict | None = None):
    if args.candidate_json:
        payload = load_candidate_json_arg(args.candidate_json)
        seed = int(payload.get("seed", parse_csv(args.seeds, int)[0]))
        tag = payload.get("sweep_tag") or f"{args.tag}_candidate_sd{seed}"
        return [{
            **payload,
            "seed": seed,
            "sweep_tag": tag,
            "__candidate_payload": payload,
            "__candidate_key": stable_candidate_key(payload),
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
            axis_values.append(parse_axis_values(name, raw_values, profile))

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
            tag = "_".join([*parts, f"sd{seed}"])
            candidate_payload = {k: v for k, v in cfg.items()}
            configs.append({
                **cfg,
                "seed": seed,
                "sweep_tag": tag,
                "__candidate_payload": candidate_payload if args.axis else None,
                "__candidate_key": stable_candidate_key(candidate_payload) if args.axis else None,
            })
    return configs
