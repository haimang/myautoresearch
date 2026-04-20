"""Tests for framework/branch_policy.py — v20.3 Continuation / Trajectory Explorer."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from branch_policy import (
    apply_delta,
    describe_branch_policy,
    get_reason_config,
    load_branch_policy,
    list_reasons,
    reason_preserves_protocol,
    validate_branch_policy,
    validate_delta,
)


class TestBranchPolicyLoad(unittest.TestCase):
    def test_load_valid_policy(self):
        path = ROOT / "domains" / "gomoku" / "branch_policy.json"
        policy = load_branch_policy(str(path))
        self.assertEqual(policy["domain"], "gomoku")
        self.assertIn("lr_decay", policy["branch_reasons"])

    def test_load_missing_file_raises(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            load_branch_policy("/nonexistent/branch_policy.json")


class TestBranchPolicyValidate(unittest.TestCase):
    def _valid_policy(self):
        return {
            "domain": "test",
            "name": "test-branch",
            "version": "1.0",
            "search_space_ref": {"domain": "test", "name": "ss", "version": "1.0"},
            "stage_policy_ref": {"domain": "test", "name": "sp", "version": "1.0"},
            "branch_reasons": {
                "lr_decay": {
                    "description": "x",
                    "allowed_deltas": {"learning_rate": {"type": "multiply", "default_factor": 0.1}},
                    "preserves_protocol": True,
                },
                "mcts_upshift": {
                    "description": "x",
                    "allowed_deltas": {"mcts_simulations": {"type": "add", "default_delta": 100}},
                    "preserves_protocol": True,
                },
                "eval_upgrade": {
                    "description": "x",
                    "allowed_deltas": {"eval_level": {"type": "add", "default_delta": 1}},
                    "preserves_protocol": False,
                    "allowed_protocol_changes": ["eval_level"],
                },
                "seed_recheck": {
                    "description": "x",
                    "allowed_deltas": {"seed": {"type": "set", "default_value": 1}},
                    "preserves_protocol": True,
                },
                "buffer_or_spc_adjust": {
                    "description": "x",
                    "allowed_deltas": {
                        "buffer_size": {"type": "multiply", "default_factor": 2.0},
                    },
                    "preserves_protocol": True,
                },
            },
        }

    def test_valid_policy_passes(self):
        validate_branch_policy(self._valid_policy())

    def test_missing_top_keys_raises(self):
        with self.assertRaisesRegex(ValueError, "Missing top-level keys"):
            validate_branch_policy({"domain": "test"})

    def test_missing_branch_reasons_raises(self):
        p = self._valid_policy()
        del p["branch_reasons"]
        with self.assertRaisesRegex(ValueError, "Missing top-level keys"):
            validate_branch_policy(p)

    def test_missing_expected_reason_raises(self):
        p = self._valid_policy()
        del p["branch_reasons"]["lr_decay"]
        with self.assertRaisesRegex(ValueError, "Missing expected branch reasons"):
            validate_branch_policy(p)

    def test_unknown_reason_delta_type_raises(self):
        p = self._valid_policy()
        p["branch_reasons"]["lr_decay"]["allowed_deltas"]["learning_rate"]["type"] = "bogus"
        with self.assertRaisesRegex(ValueError, "invalid"):
            validate_branch_policy(p)

    def test_eval_upgrade_missing_allowed_protocol_changes_raises(self):
        p = self._valid_policy()
        del p["branch_reasons"]["eval_upgrade"]["allowed_protocol_changes"]
        with self.assertRaisesRegex(ValueError, "must define allowed_protocol_changes"):
            validate_branch_policy(p)

    def test_search_space_ref_incomplete_raises(self):
        p = self._valid_policy()
        p["search_space_ref"] = {"domain": "test"}
        with self.assertRaisesRegex(ValueError, "search_space_ref must be an object with domain and name"):
            validate_branch_policy(p)


class TestBranchPolicyHelpers(unittest.TestCase):
    def test_list_reasons(self):
        policy = {"branch_reasons": {"a": {}, "b": {}}}
        self.assertEqual(sorted(list_reasons(policy)), ["a", "b"])

    def test_get_reason_config_found(self):
        policy = {"branch_reasons": {"lr_decay": {"preserves_protocol": True}}}
        self.assertTrue(get_reason_config(policy, "lr_decay")["preserves_protocol"])

    def test_get_reason_config_missing(self):
        self.assertIsNone(get_reason_config({"branch_reasons": {}}, "x"))

    def test_reason_preserves_protocol(self):
        policy = {"branch_reasons": {"lr_decay": {"preserves_protocol": True}}}
        self.assertTrue(reason_preserves_protocol(policy, "lr_decay"))

    def test_describe_branch_policy(self):
        policy = {
            "domain": "test",
            "name": "bp",
            "version": "1.0",
            "search_space_ref": {"name": "ss", "version": "1.0"},
            "stage_policy_ref": {"name": "sp", "version": "1.0"},
            "branch_reasons": {
                "lr_decay": {
                    "description": "Reduce LR",
                    "allowed_deltas": {"learning_rate": {"type": "multiply"}},
                    "preserves_protocol": True,
                    "example": "lr *= 0.1",
                }
            },
        }
        desc = describe_branch_policy(policy)
        self.assertIn("Branch Policy: bp v1.0", desc)
        self.assertIn("lr_decay", desc)


class TestDeltaApplication(unittest.TestCase):
    def test_apply_delta_multiply(self):
        policy = {
            "branch_reasons": {
                "lr_decay": {
                    "allowed_deltas": {"learning_rate": {"type": "multiply", "default_factor": 0.1}}
                }
            }
        }
        child = apply_delta({"learning_rate": 0.01}, "lr_decay", policy)
        self.assertAlmostEqual(child["learning_rate"], 0.001)

    def test_apply_delta_add(self):
        policy = {
            "branch_reasons": {
                "mcts_upshift": {
                    "allowed_deltas": {"mcts_simulations": {"type": "add", "default_delta": 200}}
                }
            }
        }
        child = apply_delta({"mcts_simulations": 400}, "mcts_upshift", policy)
        self.assertEqual(child["mcts_simulations"], 600)

    def test_apply_delta_set(self):
        policy = {
            "branch_reasons": {
                "seed_recheck": {
                    "allowed_deltas": {"seed": {"type": "set", "default_value": 99}}
                }
            }
        }
        child = apply_delta({"seed": 42}, "seed_recheck", policy)
        self.assertEqual(child["seed"], 99)

    def test_apply_delta_override(self):
        policy = {
            "branch_reasons": {
                "lr_decay": {
                    "allowed_deltas": {"learning_rate": {"type": "multiply", "default_factor": 0.1}}
                }
            }
        }
        child = apply_delta({"learning_rate": 0.01}, "lr_decay", policy, override={"learning_rate": 0.5})
        self.assertAlmostEqual(child["learning_rate"], 0.5)

    def test_validate_delta_in_bounds(self):
        policy = {
            "branch_reasons": {
                "lr_decay": {
                    "allowed_deltas": {"learning_rate": {"type": "multiply", "min_factor": 0.01, "max_factor": 0.5}}
                }
            }
        }
        validate_delta("lr_decay", {"learning_rate": 0.1}, policy)  # should not raise

    def test_validate_delta_out_of_bounds(self):
        policy = {
            "branch_reasons": {
                "lr_decay": {
                    "allowed_deltas": {"learning_rate": {"type": "multiply", "min_factor": 0.1, "max_factor": 0.5}}
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "max"):
            validate_delta("lr_decay", {"learning_rate": 0.9}, policy)


if __name__ == "__main__":
    unittest.main()
