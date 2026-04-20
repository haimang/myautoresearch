import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from search_space import (
    describe_profile,
    load_profile,
    normalize_profile,
    profile_hash,
    validate_selected_axes,
)


PROFILE_PATH = ROOT / "domains" / "gomoku" / "search_space.json"


class TestSearchSpace(unittest.TestCase):
    def test_load_gomoku_profile(self):
        profile = load_profile(str(PROFILE_PATH))
        self.assertEqual(profile["domain"], "gomoku")
        self.assertIn("num_blocks", profile["axes"])
        self.assertIn("profile_hash", profile)

    def test_missing_domain_raises(self):
        with self.assertRaisesRegex(ValueError, "missing required field 'domain'"):
            normalize_profile({"name": "x", "version": "1", "protocol": {}, "axes": {"a": {"type": "int", "values": [1], "role": "training"}}})

    def test_missing_axes_raises(self):
        with self.assertRaisesRegex(ValueError, "missing required field 'axes'"):
            normalize_profile({"domain": "gomoku", "name": "x", "version": "1", "protocol": {}})

    def test_invalid_axis_type_raises(self):
        profile = {
            "domain": "gomoku",
            "name": "x",
            "version": "1",
            "protocol": {},
            "axes": {"lr": {"type": "bogus", "values": [0.1], "role": "training"}},
        }
        with self.assertRaisesRegex(ValueError, "invalid type"):
            normalize_profile(profile)

    def test_log_scale_requires_positive_values(self):
        profile = {
            "domain": "gomoku",
            "name": "x",
            "version": "1",
            "protocol": {},
            "axes": {"lr": {"type": "float", "values": [0.0, 0.1], "role": "training", "scale": "log"}},
        }
        with self.assertRaisesRegex(ValueError, "scale=log"):
            normalize_profile(profile)

    def test_invalid_role_raises(self):
        profile = {
            "domain": "gomoku",
            "name": "x",
            "version": "1",
            "protocol": {},
            "axes": {"lr": {"type": "float", "values": [0.1], "role": "bad"}},
        }
        with self.assertRaisesRegex(ValueError, "invalid role"):
            normalize_profile(profile)

    def test_allow_continuation_defaults_false(self):
        profile = normalize_profile({
            "domain": "gomoku",
            "name": "x",
            "version": "1",
            "protocol": {},
            "axes": {"lr": {"type": "float", "values": [0.1], "role": "training"}},
        })
        self.assertFalse(profile["axes"]["lr"]["allow_continuation"])

    def test_describe_profile_contains_domain_protocol_and_roles(self):
        profile = load_profile(str(PROFILE_PATH))
        text = describe_profile(profile)
        self.assertIn("gomoku/cold-start-core", text)
        self.assertIn("Protocol:", text)
        self.assertIn("structure:", text)
        self.assertIn("training:", text)

    def test_profile_hash_is_stable(self):
        a = load_profile(str(PROFILE_PATH))
        b = load_profile(str(PROFILE_PATH))
        self.assertEqual(profile_hash(a), profile_hash(b))

    def test_validate_selected_axes_rejects_out_of_profile_values(self):
        profile = load_profile(str(PROFILE_PATH))
        with self.assertRaisesRegex(ValueError, "outside profile definition"):
            validate_selected_axes(profile, {"num_blocks": [999]})


if __name__ == "__main__":
    unittest.main()
