"""Tests for framework/selector_policy.py — v21 Surrogate-Guided Selector."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from selector_policy import (
    describe_selector_policy,
    get_candidate_kind_config,
    get_score_weights,
    list_candidate_kinds,
    load_selector_policy,
    policy_hash,
    validate_selector_policy,
)


class TestSelectorPolicyLoad(unittest.TestCase):
    def test_load_valid_policy(self):
        path = ROOT / "domains" / "gomoku" / "selector_policy.json"
        policy = load_selector_policy(str(path))
        self.assertEqual(policy["domain"], "gomoku")
        self.assertIn("new_point", policy["candidate_kinds"])

    def test_load_missing_file_raises(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            load_selector_policy("/nonexistent/selector_policy.json")


class TestSelectorPolicyValidate(unittest.TestCase):
    def _valid_policy(self):
        return {
            "domain": "test",
            "name": "test-selector",
            "version": "1.0",
            "search_space_ref": {"domain": "test", "name": "ss", "version": "1.0"},
            "stage_policy_ref": {"domain": "test", "name": "sp", "version": "1.0"},
            "branch_policy_ref": {"domain": "test", "name": "bp", "version": "1.0"},
            "candidate_kinds": {
                "new_point": {"description": "x", "max_per_batch": 3, "default_budget_s": 60},
                "seed_recheck": {"description": "x", "max_per_batch": 2, "default_budget_s": 60},
                "continue_branch": {"description": "x", "max_per_batch": 2, "default_budget_s": 120},
                "eval_upgrade": {"description": "x", "max_per_batch": 1, "default_budget_s": 120},
                "skip_dominated": {"description": "x", "max_per_batch": 0, "default_budget_s": 0},
            },
            "score_weights": {
                "frontier_gap": 1.0,
                "uncertainty": 0.8,
                "cost_penalty": 0.5,
                "dominance_penalty": 1.2,
            },
        }

    def test_valid_policy_passes(self):
        validate_selector_policy(self._valid_policy())

    def test_missing_top_keys_raises(self):
        with self.assertRaisesRegex(ValueError, "Missing top-level keys"):
            validate_selector_policy({"domain": "test"})

    def test_missing_candidate_kinds_raises(self):
        p = self._valid_policy()
        del p["candidate_kinds"]
        with self.assertRaisesRegex(ValueError, "Missing top-level keys"):
            validate_selector_policy(p)

    def test_unknown_candidate_kind_raises(self):
        p = self._valid_policy()
        p["candidate_kinds"]["bogus_kind"] = {"description": "x", "max_per_batch": 1, "default_budget_s": 1}
        with self.assertRaisesRegex(ValueError, "Unknown candidate kind"):
            validate_selector_policy(p)

    def test_missing_score_weight_raises(self):
        p = self._valid_policy()
        del p["score_weights"]["frontier_gap"]
        with self.assertRaisesRegex(ValueError, "score_weights missing keys"):
            validate_selector_policy(p)

    def test_ref_domain_mismatch_raises(self):
        p = self._valid_policy()
        p["search_space_ref"]["domain"] = "chess"
        with self.assertRaisesRegex(ValueError, "does not match policy domain"):
            validate_selector_policy(p)

    def test_stage_policy_ref_domain_mismatch_raises(self):
        p = self._valid_policy()
        p["stage_policy_ref"]["domain"] = "chess"
        with self.assertRaisesRegex(ValueError, "does not match policy domain"):
            validate_selector_policy(p)

    def test_negative_weight_raises(self):
        p = self._valid_policy()
        p["score_weights"]["dominance_penalty"] = -1.0
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_selector_policy(p)

    def test_describe_selector_policy(self):
        policy = self._valid_policy()
        desc = describe_selector_policy(policy)
        self.assertIn("test-selector v1.0", desc)
        self.assertIn("new_point", desc)
        self.assertIn("frontier_gap", desc)


class TestSelectorPolicyHelpers(unittest.TestCase):
    def test_list_candidate_kinds(self):
        policy = {"candidate_kinds": {"a": {}, "b": {}}}
        self.assertEqual(sorted(list_candidate_kinds(policy)), ["a", "b"])

    def test_get_candidate_kind_config_found(self):
        policy = {"candidate_kinds": {"new_point": {"max_per_batch": 3}}}
        self.assertEqual(get_candidate_kind_config(policy, "new_point")["max_per_batch"], 3)

    def test_get_candidate_kind_config_missing(self):
        self.assertIsNone(get_candidate_kind_config({"candidate_kinds": {}}, "x"))

    def test_get_score_weights(self):
        policy = {"score_weights": {"a": 1.0}}
        self.assertEqual(get_score_weights(policy)["a"], 1.0)

    def test_policy_hash_stable(self):
        p = {"a": 1, "b": [2, 3]}
        h1 = policy_hash(p)
        h2 = policy_hash(p)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)


if __name__ == "__main__":
    unittest.main()
