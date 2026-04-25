"""Tests for v22 objective profiles."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from objective_profile import load_objective_profile, validate_objective_profile


class TestObjectiveProfile(unittest.TestCase):
    def test_loads_fx_profile(self):
        profile = load_objective_profile(str(ROOT / "domains" / "fx_spot" / "objective_profile.json"))
        self.assertEqual(profile["domain"], "fx_spot")
        self.assertIn("preservation_ratio", profile["maximize"])
        self.assertIn("embedded_spread_bps", profile["minimize"])
        self.assertIn("profile_hash", profile)

    def test_rejects_duplicate_objective_metric(self):
        with self.assertRaisesRegex(ValueError, "appears more than once"):
            validate_objective_profile({
                "domain": "x",
                "name": "bad",
                "version": "1",
                "maximize": ["m"],
                "minimize": ["m"],
            })


if __name__ == "__main__":
    unittest.main()

