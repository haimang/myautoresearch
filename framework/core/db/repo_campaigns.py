"""Campaign, search-space, experiment-run, and stage repositories."""

from __future__ import annotations

import sqlite3
import uuid
from typing import Optional

from .common import stable_json, utc_now_iso


def get_search_space(conn: sqlite3.Connection, space_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM search_spaces WHERE id = ?", (space_id,)).fetchone()


def save_search_space(conn: sqlite3.Connection, profile: dict) -> str:
    profile_json = stable_json(profile)
    profile_hash = profile.get("profile_hash")
    if not profile_hash:
        raise ValueError("profile missing profile_hash")
    row = conn.execute("SELECT id FROM search_spaces WHERE profile_hash = ?", (profile_hash,)).fetchone()
    if row:
        return row["id"]
    space_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO search_spaces
           (id, created_at, domain, name, version, profile_json, profile_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (space_id, utc_now_iso(), profile["domain"], profile["name"], profile["version"], profile_json, profile_hash),
    )
    conn.commit()
    return space_id


def save_objective_profile(conn: sqlite3.Connection, profile: dict) -> str:
    profile_json = stable_json(profile)
    profile_hash = profile.get("profile_hash")
    if not profile_hash:
        raise ValueError("objective profile missing profile_hash")
    row = conn.execute("SELECT id FROM objective_profiles WHERE profile_hash = ?", (profile_hash,)).fetchone()
    if row:
        return row["id"]
    profile_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO objective_profiles
           (id, created_at, domain, name, version, profile_json, profile_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (profile_id, utc_now_iso(), profile["domain"], profile["name"], profile["version"], profile_json, profile_hash),
    )
    conn.commit()
    return profile_id


def get_objective_profile(conn: sqlite3.Connection, profile_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM objective_profiles WHERE id = ?", (profile_id,)).fetchone()


def save_experiment_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    domain: str,
    output_root: str,
    manifest: dict,
    objective_profile_id: Optional[str] = None,
) -> None:
    conn.execute(
        """INSERT INTO experiment_runs
           (id, domain, created_at, objective_profile_id, output_root, manifest_json)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               objective_profile_id = excluded.objective_profile_id,
               output_root = excluded.output_root,
               manifest_json = excluded.manifest_json""",
        (run_id, domain, utc_now_iso(), objective_profile_id, output_root, stable_json(manifest)),
    )
    conn.commit()


def get_campaign(conn: sqlite3.Connection, campaign: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM campaigns WHERE id = ? OR name = ? ORDER BY created_at DESC LIMIT 1",
        (campaign, campaign),
    ).fetchone()


def get_or_create_campaign(
    conn: sqlite3.Connection,
    *,
    name: str,
    domain: str,
    train_script: str,
    search_space_id: str,
    protocol: dict,
    objective_profile_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict:
    protocol_json = stable_json(protocol)
    row = get_campaign(conn, name)
    if row:
        mismatches = []
        if row["domain"] != domain:
            mismatches.append(f"domain={row['domain']} != {domain}")
        if row["train_script"] != train_script:
            mismatches.append(f"train_script={row['train_script']} != {train_script}")
        if row["search_space_id"] != search_space_id:
            mismatches.append("search_space profile differs")
        if row["protocol_json"] != protocol_json:
            mismatches.append("protocol differs")
        if objective_profile_id and row["objective_profile_id"] not in (None, objective_profile_id):
            mismatches.append("objective profile differs")
        if mismatches:
            raise ValueError(
                f"campaign '{name}' already exists with incompatible config: " + "; ".join(mismatches)
            )
        if objective_profile_id and row["objective_profile_id"] is None:
            conn.execute(
                "UPDATE campaigns SET objective_profile_id = ? WHERE id = ?",
                (objective_profile_id, row["id"]),
            )
            conn.commit()
            row = get_campaign(conn, name)
        return dict(row)

    campaign_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO campaigns
           (id, created_at, name, domain, train_script, search_space_id,
             protocol_json, objective_profile_id, status, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)""",
        (
            campaign_id,
            utc_now_iso(),
            name,
            domain,
            train_script,
            search_space_id,
            protocol_json,
            objective_profile_id,
            notes,
        ),
    )
    conn.commit()
    row = get_campaign(conn, campaign_id)
    return dict(row)


def link_run_to_campaign(
    conn: sqlite3.Connection,
    *,
    campaign_id: str,
    run_id: str,
    stage: Optional[str],
    sweep_tag: Optional[str],
    seed: Optional[int],
    axis_values: dict,
    status: str = "linked",
) -> None:
    conn.execute(
        """INSERT INTO campaign_runs
           (campaign_id, run_id, stage, sweep_tag, seed, axis_values_json, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(campaign_id, run_id) DO UPDATE SET
               stage = excluded.stage,
               sweep_tag = excluded.sweep_tag,
               seed = excluded.seed,
               axis_values_json = excluded.axis_values_json,
               status = excluded.status""",
        (campaign_id, run_id, stage, sweep_tag, seed, stable_json(axis_values), status, utc_now_iso()),
    )
    conn.commit()


def find_run_by_sweep_tag(conn: sqlite3.Connection, sweep_tag: str) -> Optional[str]:
    row = conn.execute(
        "SELECT id FROM runs WHERE sweep_tag = ? ORDER BY started_at DESC LIMIT 1",
        (sweep_tag,),
    ).fetchone()
    return row["id"] if row else None


def candidate_key(axis_values: dict) -> str:
    return stable_json({k: v for k, v in axis_values.items() if k != "seed"})


def link_run_to_campaign_v20(
    conn: sqlite3.Connection,
    *,
    campaign_id: str,
    run_id: str,
    stage: Optional[str],
    sweep_tag: Optional[str],
    seed: Optional[int],
    axis_values: dict,
    status: str = "linked",
) -> None:
    generated_key = candidate_key(axis_values)
    conn.execute(
        """INSERT INTO campaign_runs
           (campaign_id, run_id, stage, sweep_tag, seed, axis_values_json, candidate_key, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(campaign_id, run_id) DO UPDATE SET
               stage = excluded.stage,
               sweep_tag = excluded.sweep_tag,
               seed = excluded.seed,
               axis_values_json = excluded.axis_values_json,
               candidate_key = excluded.candidate_key,
               status = excluded.status""",
        (
            campaign_id,
            run_id,
            stage,
            sweep_tag,
            seed,
            stable_json(axis_values),
            generated_key,
            status,
            utc_now_iso(),
        ),
    )
    conn.commit()


def save_campaign_stage(
    conn: sqlite3.Connection,
    *,
    campaign_id: str,
    stage: str,
    policy_json: str,
    budget_json: str,
    seed_target: int,
    status: str = "open",
) -> None:
    stage_id = f"{campaign_id}_{stage}"
    conn.execute(
        """INSERT INTO campaign_stages
           (id, campaign_id, stage, policy_json, budget_json, seed_target, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               policy_json = excluded.policy_json,
               budget_json = excluded.budget_json,
               seed_target = excluded.seed_target,
               status = excluded.status""",
        (stage_id, campaign_id, stage, policy_json, budget_json, seed_target, status, utc_now_iso()),
    )
    conn.commit()


def close_campaign_stage(conn: sqlite3.Connection, campaign_id: str, stage: str) -> None:
    stage_id = f"{campaign_id}_{stage}"
    conn.execute(
        "UPDATE campaign_stages SET status = 'closed', closed_at = ? WHERE id = ?",
        (utc_now_iso(), stage_id),
    )
    conn.commit()


def get_campaign_stages(conn: sqlite3.Connection, campaign_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM campaign_stages WHERE campaign_id = ? ORDER BY stage",
        (campaign_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_campaign_stage(conn: sqlite3.Connection, campaign_id: str, stage: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM campaign_stages WHERE campaign_id = ? AND stage = ?",
        (campaign_id, stage),
    ).fetchone()
    return dict(row) if row else None


def get_campaign_runs_by_stage(conn: sqlite3.Connection, campaign_id: str, stage: str) -> list[dict]:
    rows = conn.execute(
        """SELECT cr.*, r.status AS run_status, r.final_win_rate, r.wall_time_s,
                  r.total_games, r.total_steps, r.num_params, r.final_loss
           FROM campaign_runs cr
           JOIN runs r ON r.id = cr.run_id
           WHERE cr.campaign_id = ? AND cr.stage = ?""",
        (campaign_id, stage),
    ).fetchall()
    return [dict(r) for r in rows]


def aggregate_candidates_by_stage(conn: sqlite3.Connection, campaign_id: str, stage: str) -> list[dict]:
    rows = conn.execute(
        """SELECT
               cr.candidate_key,
               cr.axis_values_json,
               COUNT(*) AS run_count,
               COUNT(DISTINCT cr.seed) AS seed_count,
               AVG(r.final_win_rate) AS mean_wr,
               MAX(r.final_win_rate) - MIN(r.final_win_rate) AS range_wr,
               CASE WHEN COUNT(*) > 1
                    THEN SQRT(((COUNT(*)*SUM(r.final_win_rate*r.final_win_rate) -
                               SUM(r.final_win_rate)*SUM(r.final_win_rate)) /
                               (COUNT(*)*(COUNT(*)-1))))
                    ELSE 0.0
               END AS std_wr,
               AVG(r.wall_time_s) AS mean_wall_s,
               SUM(r.total_games) AS games_total,
               AVG(r.num_params) AS mean_params,
               MIN(r.final_win_rate) AS min_wr,
               MAX(r.final_win_rate) AS max_wr
           FROM campaign_runs cr
           JOIN runs r ON r.id = cr.run_id
           WHERE cr.campaign_id = ? AND cr.stage = ?
             AND r.status IN ('completed', 'time_budget', 'target_win_rate', 'target_games')
           GROUP BY cr.candidate_key
           ORDER BY mean_wr DESC""",
        (campaign_id, stage),
    ).fetchall()
    return [dict(r) for r in rows]
