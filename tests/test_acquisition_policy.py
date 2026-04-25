"""Tests for framework/acquisition_policy.py."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.policies.acquisition_policy import describe_acquisition_policy, load_acquisition_policy, validate_acquisition_policy

POLICY_PATH = ROOT / "domains" / "gomoku" / "acquisition_policy.json"


class TestAcquisitionPolicy(unittest.TestCase):
    def test_load_valid_policy(self):
        policy = load_acquisition_policy(str(POLICY_PATH))
        self.assertEqual(policy["domain"], "gomoku")
        self.assertEqual(policy["name"], "candidate-pool-ucb")

    def test_invalid_missing_keys_raises(self):
        with self.assertRaises(ValueError):
            validate_acquisition_policy({"domain": "gomoku"})

    def test_invalid_target_seed_count_raises(self):
        data = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        data["priors"]["target_seed_count"] = 0
        with self.assertRaises(ValueError):
            validate_acquisition_policy(data)

    def test_describe_mentions_objectives(self):
        policy = load_acquisition_policy(str(POLICY_PATH))
        text = describe_acquisition_policy(policy)
        self.assertIn("Objectives", text)
        self.assertIn("candidate-pool-ucb", text)


if __name__ == "__main__":
    unittest.main()
