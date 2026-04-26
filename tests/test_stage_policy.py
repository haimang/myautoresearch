"""Tests for framework/stage_policy.py — v20.2 Multi-Fidelity Promotion Engine."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.policies.stage_policy import (
    aggregate_stage_metrics,
    describe_stage_policy,
    get_next_stage,
    get_stage_by_name,
    load_stage_policy,
    plan_promotions,
    validate_stage_policy,
)


class TestStagePolicyLoad(unittest.TestCase):
    def test_load_valid_policy(self):
        path = ROOT / "domains" / "gomoku" / "manifest" / "stage_policy.json"
        policy = load_stage_policy(str(path))
        self.assertEqual(policy["domain"], "gomoku")
        self.assertEqual(len(policy["stages"]), 4)

    def test_load_missing_file_raises(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            load_stage_policy("/nonexistent/stage_policy.json")


class TestValidateMetricWhitelist(unittest.TestCase):
    """T-2: validate that disallowed metric names are rejected (SQL injection guard)."""

    def _make_policy(self, metric):
        return {
            "domain": "test",
            "name": "test-policy",
            "version": "1.0",
            "search_space_ref": {"domain": "test", "name": "ss"},
            "stages": [
                {"name": "A", "time_budget": 60, "seed_count": 1,
                 "promote_top_k": 1, "metric": metric, "min_runs": 1},
                {"name": "B", "time_budget": 180, "seed_count": 1,
                 "promote_top_k": 1, "metric": "win_rate", "min_runs": 1},
                {"name": "C", "time_budget": 600, "seed_count": 1,
                 "promote_top_k": 1, "metric": "win_rate", "min_runs": 1},
                {"name": "D", "time_budget": 1800, "seed_count": 1,
                 "promote_top_k": 0, "metric": "win_rate", "min_runs": 1},
            ],
        }

    def test_unknown_metric_rejected_in_validate(self):
        with self.assertRaisesRegex(ValueError, "metric"):
            validate_stage_policy(self._make_policy("wr"))

    def test_sql_injection_metric_rejected_in_validate(self):
        with self.assertRaisesRegex(ValueError, "metric"):
            validate_stage_policy(
                self._make_policy("final_win_rate; DROP TABLE runs--")
            )

    def test_aggregate_stage_metrics_rejects_bad_col(self):
        from framework.core.db import init_db
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = init_db(db_path)
            with self.assertRaisesRegex(ValueError, "metric_col"):
                aggregate_stage_metrics(conn, "some-id", "A", "bad_col")
            with self.assertRaisesRegex(ValueError, "metric_col"):
                aggregate_stage_metrics(conn, "some-id", "A",
                                        "final_win_rate; DROP TABLE runs--")
            conn.close()
        finally:
            import os; os.remove(db_path)


class TestStagePolicyValidate(unittest.TestCase):
    def test_valid_policy_passes(self):
        policy = {
            "domain": "test",
            "name": "test-policy",
            "version": "1.0",
            "search_space_ref": {"domain": "test", "name": "ss", "version": "1.0"},
            "stages": [
                {"name": "A", "time_budget": 60, "seed_count": 1, "promote_top_k": 2, "metric": "win_rate", "min_runs": 1},
                {"name": "B", "time_budget": 180, "seed_count": 2, "promote_top_k": 1, "metric": "win_rate", "min_runs": 1},
                {"name": "C", "time_budget": 600, "seed_count": 3, "promote_top_k": 1, "metric": "win_rate", "min_runs": 1},
                {"name": "D", "time_budget": 1800, "seed_count": 3, "promote_top_k": 0, "metric": "win_rate", "min_runs": 1},
            ],
        }
        validate_stage_policy(policy)  # should not raise

    def test_missing_top_keys_raises(self):
        with self.assertRaisesRegex(ValueError, "Missing top-level keys"):
            validate_stage_policy({"domain": "test"})

    def test_duplicate_stage_name_raises(self):
        policy = {
            "domain": "test",
            "name": "test",
            "version": "1",
            "search_space_ref": {"domain": "test", "name": "ss"},
            "stages": [
                {"name": "A", "time_budget": 60, "seed_count": 1, "promote_top_k": 2, "metric": "win_rate", "min_runs": 1},
                {"name": "A", "time_budget": 60, "seed_count": 1, "promote_top_k": 2, "metric": "win_rate", "min_runs": 1},
            ],
        }
        with self.assertRaisesRegex(ValueError, "Duplicate stage name"):
            validate_stage_policy(policy)

    def test_negative_promote_top_k_raises(self):
        policy = {
            "domain": "test",
            "name": "test",
            "version": "1",
            "search_space_ref": {"domain": "test", "name": "ss"},
            "stages": [
                {"name": "A", "time_budget": 60, "seed_count": 1, "promote_top_k": -1, "metric": "win_rate", "min_runs": 1},
            ],
        }
        with self.assertRaisesRegex(ValueError, "promote_top_k must be >= 0"):
            validate_stage_policy(policy)

    def test_out_of_order_stages_raises(self):
        policy = {
            "domain": "test",
            "name": "test",
            "version": "1",
            "search_space_ref": {"domain": "test", "name": "ss"},
            "stages": [
                {"name": "B", "time_budget": 60, "seed_count": 1, "promote_top_k": 2, "metric": "win_rate", "min_runs": 1},
            ],
        }
        with self.assertRaisesRegex(ValueError, "Stage order must be"):
            validate_stage_policy(policy)

    def test_missing_stage_d_raises(self):
        policy = {
            "domain": "test",
            "name": "test",
            "version": "1",
            "search_space_ref": {"domain": "test", "name": "ss"},
            "stages": [
                {"name": "A", "time_budget": 60, "seed_count": 1, "promote_top_k": 2, "metric": "win_rate", "min_runs": 1},
                {"name": "B", "time_budget": 180, "seed_count": 2, "promote_top_k": 1, "metric": "win_rate", "min_runs": 1},
                {"name": "C", "time_budget": 600, "seed_count": 3, "promote_top_k": 1, "metric": "win_rate", "min_runs": 1},
            ],
        }
        with self.assertRaisesRegex(ValueError, "Stage D must be present"):
            validate_stage_policy(policy)

    def test_insufficient_min_runs_gets_hold(self):
        policy = {
            "stages": [
                {"name": "A", "time_budget": 60, "seed_count": 1, "promote_top_k": 1, "metric": "win_rate", "min_runs": 3},
            ]
        }
        aggregated = [
            {"candidate_key": "ck1", "axis_values": {"lr": 0.01}, "mean_metric": 0.9, "seed_count": 3, "run_count": 1},
        ]
        decisions = plan_promotions(policy, "A", aggregated)
        self.assertEqual(decisions[0]["decision"], "hold")
        self.assertIn("runs=1/3", decisions[0]["reason"])


class TestStagePolicyHelpers(unittest.TestCase):
    def test_get_stage_by_name_found(self):
        policy = {
            "stages": [
                {"name": "A", "time_budget": 60},
                {"name": "B", "time_budget": 180},
            ]
        }
        self.assertEqual(get_stage_by_name(policy, "A")["time_budget"], 60)
        self.assertEqual(get_stage_by_name(policy, "B")["time_budget"], 180)

    def test_get_stage_by_name_missing(self):
        self.assertIsNone(get_stage_by_name({"stages": []}, "A"))

    def test_get_next_stage(self):
        policy = {
            "stages": [
                {"name": "A"}, {"name": "B"}, {"name": "C"},
            ]
        }
        self.assertEqual(get_next_stage(policy, "A"), "B")
        self.assertEqual(get_next_stage(policy, "B"), "C")
        self.assertIsNone(get_next_stage(policy, "C"))
        self.assertIsNone(get_next_stage(policy, "Z"))

    def test_describe_stage_policy(self):
        policy = {
            "domain": "test",
            "name": "test-policy",
            "version": "1.0",
            "search_space_ref": {"name": "ss", "version": "1.0"},
            "stages": [
                {"name": "A", "time_budget": 60, "seed_count": 1, "promote_top_k": 2, "metric": "win_rate", "min_runs": 1},
            ],
        }
        desc = describe_stage_policy(policy)
        self.assertIn("Stage Policy: test-policy v1.0", desc)
        self.assertIn("budget=60s", desc)


class TestPlanPromotions(unittest.TestCase):
    def test_top_k_gets_promoted(self):
        policy = {
            "stages": [
                {"name": "A", "time_budget": 60, "seed_count": 1, "promote_top_k": 2, "metric": "win_rate", "min_runs": 1},
            ]
        }
        aggregated = [
            {"candidate_key": "ck1", "axis_values": {"lr": 0.01}, "mean_metric": 0.9, "seed_count": 2, "run_count": 2},
            {"candidate_key": "ck2", "axis_values": {"lr": 0.02}, "mean_metric": 0.8, "seed_count": 1, "run_count": 1},
            {"candidate_key": "ck3", "axis_values": {"lr": 0.03}, "mean_metric": 0.7, "seed_count": 1, "run_count": 1},
        ]
        decisions = plan_promotions(policy, "A", aggregated)
        self.assertEqual(len(decisions), 3)
        self.assertEqual(decisions[0]["decision"], "promote")
        self.assertEqual(decisions[1]["decision"], "promote")
        self.assertEqual(decisions[2]["decision"], "reject")

    def test_insufficient_seeds_gets_hold(self):
        policy = {
            "stages": [
                {"name": "A", "time_budget": 60, "seed_count": 3, "promote_top_k": 1, "metric": "win_rate", "min_runs": 1},
            ]
        }
        aggregated = [
            {"candidate_key": "ck1", "axis_values": {"lr": 0.01}, "mean_metric": 0.9, "seed_count": 1, "run_count": 1},
        ]
        decisions = plan_promotions(policy, "A", aggregated)
        self.assertEqual(decisions[0]["decision"], "hold")
        self.assertIn("insufficient data", decisions[0]["reason"])


if __name__ == "__main__":
    unittest.main()
