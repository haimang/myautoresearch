"""Tests for framework/acquisition.py."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.services.research.acquisition_service import rerank_candidates, replay_recommendation_history
from framework.policies.acquisition_policy import load_acquisition_policy

POLICY_PATH = ROOT / "domains" / "gomoku" / "manifest" / "acquisition_policy.json"


class TestAcquisitionEngine(unittest.TestCase):
    def setUp(self):
        self.policy = load_acquisition_policy(str(POLICY_PATH))

    def test_rerank_candidates_adds_acquisition_fields(self):
        candidates = [
            {
                "candidate_type": "new_point",
                "score_total": 0.7,
                "score_signals": {"mean_wr": 0.72, "std_wr": 0.01, "seed_count": 2, "mean_params": 100000, "mean_wall_s": 120},
                "rationale": {},
            },
            {
                "candidate_type": "continue_branch",
                "score_total": 0.68,
                "score_signals": {"parent_wr": 0.74, "std_wr": 0.04, "seed_count": 1, "mean_params": 120000, "mean_wall_s": 140},
                "rationale": {},
            },
        ]
        reranked, summary = rerank_candidates(candidates, self.policy)
        self.assertEqual(len(reranked), 2)
        self.assertIn("acquisition_score", reranked[0])
        self.assertIn("posterior_sigma", reranked[0])
        self.assertEqual(summary["candidate_count"], 2)

    def test_replay_history_reports_hit_rates(self):
        rows = [
            {"batch_id": "b1", "rank": 1, "score_total": 0.7, "selector_score_total": 0.9, "acquisition_score": 0.6, "outcome_label": "no_gain"},
            {"batch_id": "b1", "rank": 2, "score_total": 0.6, "selector_score_total": 0.4, "acquisition_score": 0.95, "outcome_label": "new_front"},
            {"batch_id": "b2", "rank": 1, "score_total": 0.8, "selector_score_total": 0.85, "acquisition_score": 0.8, "outcome_label": "near_front"},
            {"batch_id": "b2", "rank": 2, "score_total": 0.5, "selector_score_total": 0.5, "acquisition_score": 0.3, "outcome_label": "no_gain"},
        ]
        summary = replay_recommendation_history(
            rows,
            top_k=1,
            positive_outcomes=["new_front", "near_front"],
        )
        self.assertEqual(summary["evaluated_batches"], 2)
        self.assertEqual(summary["selector_hits"], 1)
        self.assertEqual(summary["acquisition_hits"], 2)
        self.assertGreater(summary["acquisition_hit_rate"], summary["selector_hit_rate"])


if __name__ == "__main__":
    unittest.main()
