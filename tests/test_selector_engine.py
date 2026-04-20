"""Tests for framework/selector.py — v21 candidate generation and scoring engine."""

import json
import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from core.db import (
    create_run,
    finish_run,
    get_or_create_campaign,
    init_db,
    save_campaign_stage,
    save_search_space,
)
from search_space import load_profile
from selector import (
    generate_point_candidates,
    generate_branch_candidates,
    recommend_for_campaign,
    _score_candidate,
    _is_dominated,
    build_recommendation_id,
)
from selector_policy import load_selector_policy

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


if __name__ == "__main__":
    unittest.main()
