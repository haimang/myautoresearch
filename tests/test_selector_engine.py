"""Tests for framework/selector.py — v21 candidate generation and scoring engine."""

import json
import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.core.db import (
    create_run,
    finish_run,
    get_or_create_campaign,
    init_db,
    save_checkpoint,
    save_recommendation,
    save_recommendation_batch,
    save_campaign_stage,
    save_search_space,
)
from framework.profiles.search_space import load_profile
from framework.services.research.selector_service import (
    generate_point_candidates,
    generate_branch_candidates,
    recommend_for_campaign,
    _score_candidate,
    _is_dominated,
    build_recommendation_id,
)
from framework.policies.selector_policy import load_selector_policy

PROFILE_PATH = ROOT / "domains" / "gomoku" / "search_space.json"
SELECTOR_POLICY_PATH = ROOT / "domains" / "gomoku" / "selector_policy.json"


class TestSelectorEngine(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "tracker.db")
        self.conn = init_db(self.db_path)
        self.profile = load_profile(str(PROFILE_PATH))
        self.space_id = save_search_space(self.conn, self.profile)
        self.campaign = get_or_create_campaign(
            self.conn,
            name="sel-engine-test",
            domain="gomoku",
            train_script="t.py",
            search_space_id=self.space_id,
            protocol={"eval_level": 0},
        )
        self.policy = load_selector_policy(str(SELECTOR_POLICY_PATH))

        # Seed 3 runs with different candidate_keys
        for i, (ck, axis, wr) in enumerate([
            ("ck_a", '{"lr":0.01}', 0.85),
            ("ck_b", '{"lr":0.005}', 0.60),
            ("ck_c", '{"lr":0.001}', 0.75),
        ]):
            rid = f"run-{i}"
            create_run(self.conn, rid, {"sweep_tag": f"r{i}", "eval_level": 0, "learning_rate": 0.01}, is_benchmark=True)
            finish_run(self.conn, rid, {
                "status": "completed", "final_win_rate": wr,
                "wall_time_s": 100.0, "num_params": 100000, "total_games": 500,
            })
            self.conn.execute(
                """INSERT INTO campaign_runs (campaign_id, run_id, stage, sweep_tag, seed, axis_values_json, status, created_at)
                   VALUES (?, ?, 'C', ?, ?, ?, 'linked', '2024-01-01')""",
                (self.campaign["id"], rid, f"r{i}", i, axis),
            )
        self.conn.commit()
        self.conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_generate_point_candidates_not_empty(self):
        conn = init_db(self.db_path)
        candidates = generate_point_candidates(conn, self.campaign, self.policy)
        conn.close()
        self.assertTrue(len(candidates) > 0)

    def test_generate_point_candidates_has_frontier_signals(self):
        conn = init_db(self.db_path)
        candidates = generate_point_candidates(conn, self.campaign, self.policy)
        conn.close()
        for c in candidates:
            self.assertIn("score_signals", c)
            signals = c["score_signals"] or {}
            self.assertTrue("mean_wr" in signals or "parent_wr" in signals)

    def test_generate_branch_candidates_not_empty(self):
        conn = init_db(self.db_path)
        # Add checkpoint for best run
        self.conn = conn
        conn.execute(
            """INSERT INTO checkpoints (run_id, tag, cycle, step, loss, win_rate, eval_level, eval_games, model_path, created_at)
               VALUES ('run-0', 'ckpt_final', 10, 1000, 0.5, 0.85, 0, 200, 'model.safetensors', '2024-01-01')"""
        )
        conn.commit()
        candidates = generate_branch_candidates(conn, self.campaign, self.policy)
        conn.close()
        self.assertTrue(len(candidates) > 0)
        for c in candidates:
            self.assertIn("branch_reason", c)

    def test_score_candidate_has_breakdown(self):
        candidate = {
            "candidate_type": "new_point",
            "score_signals": {"mean_wr": 0.8, "std_wr": 0.03, "seed_count": 2, "mean_params": 50000},
        }
        weights = {"frontier_gap": 1.0, "uncertainty": 0.8, "cost_penalty": 0.5, "dominance_penalty": 1.2}
        score, breakdown, rationale = _score_candidate(candidate, weights, [candidate])
        self.assertIsInstance(score, float)
        self.assertIn("frontier_gap", breakdown)
        self.assertIn("uncertainty", breakdown)
        self.assertIn("cost_penalty", breakdown)
        self.assertIn("summary", rationale)

    def test_score_breakdown_consistent_with_total(self):
        candidate = {
            "candidate_type": "new_point",
            "score_signals": {"mean_wr": 0.8, "std_wr": 0.03, "seed_count": 2, "mean_params": 50000},
        }
        weights = {"frontier_gap": 1.0, "uncertainty": 0.8, "cost_penalty": 0.5, "dominance_penalty": 1.2}
        score, breakdown, _ = _score_candidate(candidate, weights, [candidate])
        total_from_breakdown = sum(breakdown.values())
        self.assertAlmostEqual(score, round(total_from_breakdown, 4), places=3)

    def test_dominated_candidate_gets_penalty(self):
        dominated = {"score_signals": {"mean_wr": 0.5, "mean_params": 200000}}
        dominant = {"score_signals": {"mean_wr": 0.8, "mean_params": 100000}}
        self.assertTrue(_is_dominated(dominated, [dominated, dominant]))
        self.assertFalse(_is_dominated(dominant, [dominated, dominant]))

    def test_recommend_for_campaign_returns_sorted(self):
        conn = init_db(self.db_path)
        recs = recommend_for_campaign(conn, self.campaign, self.policy, limit=5)
        conn.close()
        self.assertTrue(len(recs) <= 5)
        for i in range(len(recs) - 1):
            self.assertGreaterEqual(recs[i]["score_total"], recs[i + 1]["score_total"])

    def test_recommend_for_campaign_has_rationale(self):
        conn = init_db(self.db_path)
        recs = recommend_for_campaign(conn, self.campaign, self.policy, limit=3)
        conn.close()
        for r in recs:
            self.assertIn("rationale", r)
            self.assertIn("score_breakdown", r)

    def test_recommend_for_campaign_filter_by_type(self):
        conn = init_db(self.db_path)
        recs_point = recommend_for_campaign(conn, self.campaign, self.policy,
                                            candidate_type="new_point", limit=5)
        conn.close()
        for r in recs_point:
            self.assertEqual(r["candidate_type"], "new_point")

    def test_build_recommendation_id_stable(self):
        batch_id = "batch-1"
        cand = {"candidate_type": "new_point", "candidate_key": "ck1"}
        id1 = build_recommendation_id(batch_id, cand)
        id2 = build_recommendation_id(batch_id, cand)
        self.assertEqual(id1, id2)
        self.assertTrue(id1.startswith("rec-"))

    def test_high_variance_gets_uncertainty_bonus(self):
        low_var = {"candidate_type": "new_point", "score_signals": {"mean_wr": 0.7, "std_wr": 0.01, "seed_count": 3, "mean_params": 50000}}
        high_var = {"candidate_type": "new_point", "score_signals": {"mean_wr": 0.7, "std_wr": 0.15, "seed_count": 1, "mean_params": 50000}}
        weights = {"frontier_gap": 1.0, "uncertainty": 0.8, "cost_penalty": 0.5, "dominance_penalty": 1.2}
        score_low, _, _ = _score_candidate(low_var, weights, [low_var, high_var])
        score_high, _, _ = _score_candidate(high_var, weights, [low_var, high_var])
        self.assertGreater(score_high, score_low)

    def test_empty_campaign_returns_empty(self):
        conn = init_db(self.db_path)
        empty_campaign = get_or_create_campaign(
            conn, name="empty-sel", domain="gomoku", train_script="t.py",
            search_space_id=self.space_id, protocol={"eval_level": 0},
        )
        recs = recommend_for_campaign(conn, empty_campaign, self.policy)
        conn.close()
        self.assertEqual(recs, [])

    def test_recommend_for_campaign_dedupes_by_full_identity(self):
        conn = init_db(self.db_path)
        campaign = get_or_create_campaign(
            conn,
            name="sel-dedupe-test",
            domain="gomoku",
            train_script="t.py",
            search_space_id=self.space_id,
            protocol={"eval_level": 0},
        )
        axis = {"lr": 0.005, "blocks": 10, "filters": 128}
        candidate_key = json.dumps(axis, sort_keys=True, separators=(",", ":"))
        run_id = "run-dedupe"
        create_run(conn, run_id, {
            "sweep_tag": "dedupe",
            "eval_level": 0,
            "learning_rate": axis["lr"],
            "num_res_blocks": axis["blocks"],
            "num_filters": axis["filters"],
        }, is_benchmark=True)
        finish_run(conn, run_id, {
            "status": "completed",
            "final_win_rate": 0.88,
            "wall_time_s": 120.0,
            "num_params": 200000,
            "total_games": 500,
        })
        conn.execute(
            """INSERT INTO campaign_runs
               (campaign_id, run_id, stage, sweep_tag, seed, axis_values_json, status, created_at, candidate_key)
               VALUES (?, ?, 'C', 'dedupe', 1, ?, 'linked', '2024-01-01', ?)""",
            (campaign["id"], run_id, candidate_key, candidate_key),
        )
        save_checkpoint(conn, run_id, {
            "tag": "final",
            "cycle": 10,
            "step": 100,
            "loss": 0.5,
            "win_rate": 0.88,
            "eval_level": 0,
            "eval_games": 100,
            "model_path": "/tmp/model.npz",
        })
        conn.commit()

        recs_before = recommend_for_campaign(conn, campaign, self.policy, limit=10)
        accepted = next(r for r in recs_before if r["candidate_type"] == "seed_recheck")
        save_recommendation_batch(
            conn,
            batch_id="batch-dedupe",
            campaign_id=campaign["id"],
            selector_name=self.policy["name"],
            selector_version=self.policy["version"],
            selector_hash="test-hash",
        )
        save_recommendation(
            conn,
            recommendation_id="rec-dedupe-seed",
            batch_id="batch-dedupe",
            candidate_type=accepted["candidate_type"],
            candidate_key=accepted["candidate_key"],
            rank=1,
            score_total=accepted["score_total"],
            score_breakdown_json="{}",
            rationale_json="{}",
            axis_values_json=json.dumps(accepted.get("axis_values", {}), sort_keys=True),
            branch_reason=accepted.get("branch_reason"),
            delta_json=accepted.get("delta_json"),
            status="accepted",
        )
        conn.commit()

        recs_after = recommend_for_campaign(conn, campaign, self.policy, limit=10)
        conn.close()

        self.assertNotIn(
            ("seed_recheck", None),
            {(r["candidate_type"], r.get("branch_reason")) for r in recs_after},
        )
        self.assertIn(
            ("continue_branch", "lr_decay"),
            {(r["candidate_type"], r.get("branch_reason")) for r in recs_after},
        )
        self.assertIn(
            ("continue_branch", "seed_recheck"),
            {(r["candidate_type"], r.get("branch_reason")) for r in recs_after},
        )


class TestNewPointBoundsAndAxisValues(unittest.TestCase):
    """T-4: new_point candidates respect search space discrete bounds.
    T-6: Stage D axis_values contain actual hyperparams (not branch metadata).
    """

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "tracker.db")
        self.conn = init_db(self.db_path)
        self.profile = load_profile(str(PROFILE_PATH))
        self.space_id = save_search_space(self.conn, self.profile)
        self.campaign = get_or_create_campaign(
            self.conn,
            name="bounds-test",
            domain="gomoku",
            train_script="t.py",
            search_space_id=self.space_id,
            protocol={"eval_level": 0},
        )
        self.policy = load_selector_policy(str(SELECTOR_POLICY_PATH))

    def tearDown(self):
        self.tmp.cleanup()

    def test_new_point_axis_values_within_allowed_values(self):
        """T-4: perturbed axis values must appear in search space allowed values."""
        # Insert run first (FK constraint), then link into campaign_runs
        create_run(self.conn, "run-b1", {"sweep_tag": "rb1", "eval_level": 0, "learning_rate": 0.0003}, is_benchmark=True)
        finish_run(self.conn, "run-b1", {
            "status": "completed", "final_win_rate": 0.7,
            "wall_time_s": 100.0, "num_params": 100000, "total_games": 500,
        })
        self.conn.execute(
            """INSERT INTO campaign_runs (campaign_id, run_id, stage, sweep_tag, seed,
               axis_values_json, status, created_at, candidate_key)
               VALUES (?, 'run-b1', 'A', 'rb1', 1,
               '{"learning_rate":0.0003,"num_blocks":6,"num_filters":64}',
               'linked', '2024-01-01',
               '{"learning_rate":0.0003,"num_blocks":6,"num_filters":64}')""",
            (self.campaign["id"],),
        )
        self.conn.commit()

        candidates = generate_point_candidates(self.conn, self.campaign, self.policy)

        # Extract learning_rate values from new_point candidates
        new_point_cands = [c for c in candidates if c["candidate_type"] == "new_point"]
        for c in new_point_cands:
            av = c["axis_values"]
            lr = av.get("learning_rate")
            if lr is not None:
                allowed_lr = [0.0003, 0.0005, 0.0007]
                self.assertIn(
                    lr, allowed_lr,
                    f"learning_rate={lr} not in allowed values {allowed_lr}"
                )

    def test_stage_d_axis_values_contain_hyperparams(self):
        """T-6: campaign_runs for Stage D must contain actual hyperparams, not branch metadata."""
        from framework.core.db import link_run_to_campaign_v20

        # Simulate what execute_branches now does: store child hyperparam axis_values
        child_params = {
            "num_blocks": 8,
            "num_filters": 64,
            "learning_rate": 0.0001,
            "steps_per_cycle": 30,
            "buffer_size": 100000,
            "mcts_simulations": 400,
            "seed": 42,
            "time_budget": 1800,
        }
        child_axis_values = {k: v for k, v in child_params.items() if k not in {"seed", "time_budget"}}

        create_run(self.conn, "child-d", {"sweep_tag": "d-child", "eval_level": 0}, is_benchmark=True)
        finish_run(self.conn, "child-d", {
            "status": "completed", "final_win_rate": 0.88,
            "wall_time_s": 1800.0, "num_params": 200000, "total_games": 1000,
        })
        link_run_to_campaign_v20(
            self.conn,
            campaign_id=self.campaign["id"],
            run_id="child-d",
            stage="D",
            sweep_tag="d-child",
            seed=42,
            axis_values=child_axis_values,
            status="linked",
        )
        self.conn.commit()

        row = self.conn.execute(
            "SELECT axis_values_json, candidate_key FROM campaign_runs "
            "WHERE run_id = 'child-d'",
        ).fetchone()
        self.assertIsNotNone(row)
        av = json.loads(row["axis_values_json"])

        # axis_values must contain hyperparams, not branch metadata
        self.assertIn("num_blocks", av)
        self.assertIn("learning_rate", av)
        self.assertNotIn("branch_reason", av)
        self.assertNotIn("parent_run_id", av)

        # candidate_key must be non-trivial (JSON of hyperparams)
        ck = row["candidate_key"]
        self.assertIsNotNone(ck)
        ck_obj = json.loads(ck)
        self.assertIn("num_blocks", ck_obj)


if __name__ == "__main__":
    unittest.main()
