"""Workspace and campaign helpers extracted from the sweep CLI."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os

from core.db import get_campaign, get_or_create_campaign, save_objective_profile, save_search_space
from core.db import DB_PATH


def filter_fx_degenerate_routes(configs: list[dict], profile: dict | None) -> list[dict]:
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


def maybe_setup_run_workspace(args, profile: dict | None, objective_profile: dict | None) -> dict | None:
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
    with open(os.path.join(workspace, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
    args.run_id = run_id
    args._artifact_root = workspace
    return manifest


def derive_protocol(args, profile: dict | None) -> dict:
    profile_protocol = (profile or {}).get("protocol", {})
    eval_level = args.eval_level if args.eval_level is not None else profile_protocol.get("eval_level")
    eval_opponent = args.eval_opponent if args.eval_opponent is not None else profile_protocol.get("eval_opponent")
    is_benchmark = bool(profile_protocol.get("is_benchmark", False))
    if args.eval_opponent is None and eval_opponent is not None:
        is_benchmark = False
    if eval_opponent is None and eval_level is not None:
        is_benchmark = True
    return {
        "eval_level": eval_level,
        "eval_opponent": eval_opponent,
        "is_benchmark": is_benchmark,
        "train_script": args.train_script,
    }


def ensure_profile_protocol(profile: dict, protocol: dict) -> None:
    expected = profile["protocol"]
    mismatches = []
    for key in ("eval_level", "eval_opponent", "is_benchmark"):
        if expected.get(key) != protocol.get(key):
            mismatches.append(f"{key}={protocol.get(key)!r} != profile {expected.get(key)!r}")
    if mismatches:
        raise ValueError("protocol does not match search-space profile: " + "; ".join(mismatches))


def ensure_campaign(conn, args, profile: dict | None, protocol: dict, objective_profile: dict | None = None):
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
        conn.execute("UPDATE campaigns SET experiment_run_id = ? WHERE id = ?", (args.run_id, campaign["id"]))
        conn.commit()
        campaign = dict(get_campaign(conn, campaign["id"]))
    return campaign, search_space_id
