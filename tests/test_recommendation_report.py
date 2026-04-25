"""Tests for v21 recommendation report commands in analyze.py."""

import json
import subprocess
import sys
import tempfile
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
    save_recommendation,
    save_recommendation_batch,
    save_recommendation_outcome,
    save_search_space,
)
from framework.profiles.search_space import load_profile

INDEX = ROOT / "framework" / "index.py"
PROFILE_PATH = ROOT / "domains" / "gomoku" / "search_space.json"


class TestRecommendationReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "tracker.db")
        self.conn = init_db(self.db_path)
        self.profile = load_profile(str(PROFILE_PATH))
        self.space_id = save_search_space(self.conn, self.profile)
        self.campaign = get_or_create_campaign(
            self.conn,
            name="rec-report-test",
            domain="gomoku",
            train_script="t.py",
            search_space_id=self.space_id,
            protocol={"eval_level": 0},
        )
        # Seed a run for outcome linkage
        create_run(self.conn, "run-1", {"sweep_tag": "r1", "eval_level": 0}, is_benchmark=True)
        finish_run(self.conn, "run-1", {
            "status": "completed", "final_win_rate": 0.80,
            "wall_time_s": 100.0, "num_params": 100000, "total_games": 500,
        })
        # Seed batch + recommendations + outcomes
        save_recommendation_batch(
            self.conn,
            batch_id="batch-1",
            campaign_id=self.campaign["id"],
            selector_name="sel",
            selector_version="1.0",
            selector_hash="abc",
        )
        save_recommendation(
            self.conn,
            recommendation_id="rec-1",
            batch_id="batch-1",
            candidate_type="new_point",
            candidate_key="ck1",
            rank=1,
            score_total=0.95,
            score_breakdown_json='{"frontier_gap": 0.8}',
            rationale_json='{"summary": "top candidate"}',
            axis_values_json='{"lr": 0.01}',
        )
        save_recommendation(
            self.conn,
            recommendation_id="rec-2",
            batch_id="batch-1",
            candidate_type="continue_branch",
            candidate_key="ck2",
            rank=2,
            score_total=0.85,
            score_breakdown_json='{"frontier_gap": 0.7}',
            rationale_json='{"summary": "good branch"}',
            branch_reason="lr_decay",
            delta_json='{"learning_rate": 0.0001}',
        )
        save_recommendation_outcome(
            self.conn,
            recommendation_id="rec-1",
            run_id="run-1",
            observed_metrics_json='{"wr": 0.82}',
            outcome_label="near_front",
        )
        self.conn.commit()
        self.conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_recommendation_log_shows_top_ranked(self):
        proc = self._run(
            str(INDEX), "analyze",
            "--db", self.db_path,
            "--recommendation-log", "rec-report-test",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("rec-report-test", proc.stdout)
        self.assertIn("new_point", proc.stdout)
        self.assertIn("continue_branch", proc.stdout)

    def test_recommendation_outcomes_shows_metrics(self):
        proc = self._run(
            str(INDEX), "analyze",
            "--db", self.db_path,
            "--recommendation-outcomes", "rec-report-test",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("near_front", proc.stdout)
        self.assertIn("new_point", proc.stdout)

    def test_recommendation_log_missing_campaign_friendly(self):
        proc = self._run(
            str(INDEX), "analyze",
            "--db", self.db_path,
            "--recommendation-log", "nonexistent",
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("not found", proc.stdout.lower())

    def test_recommendation_outcomes_missing_campaign_friendly(self):
        proc = self._run(
            str(INDEX), "analyze",
            "--db", self.db_path,
            "--recommendation-outcomes", "nonexistent",
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("not found", proc.stdout.lower())

    def test_recommendation_log_does_not_mix_campaigns(self):
        # Create another campaign with its own batch
        conn = init_db(self.db_path)
        c2 = get_or_create_campaign(
            conn, name="other-campaign", domain="gomoku", train_script="t.py",
            search_space_id=self.space_id, protocol={},
        )
        save_recommendation_batch(
            conn, batch_id="batch-2", campaign_id=c2["id"],
            selector_name="s", selector_version="1", selector_hash="h",
        )
        save_recommendation(
            conn, recommendation_id="rec-3", batch_id="batch-2",
            candidate_type="new_point", candidate_key="ck3", rank=1,
            score_total=0.5, score_breakdown_json='{}', rationale_json='{}',
        )
        conn.commit()
        conn.close()
        proc = self._run(
            str(INDEX), "analyze",
            "--db", self.db_path,
            "--recommendation-log", "rec-report-test",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("ck3", proc.stdout)

    def test_recommendation_log_json_has_batch_struct(self):
        proc = self._run(
            str(INDEX), "analyze",
            "--db", self.db_path,
            "--recommendation-log", "rec-report-test",
            "--format", "json",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertIn("batches", data)
        self.assertEqual(len(data["batches"]), 1)
        self.assertEqual(len(data["batches"][0]["recommendations"]), 2)

    def test_recommendation_outcomes_json_has_outcomes(self):
        proc = self._run(
            str(INDEX), "analyze",
            "--db", self.db_path,
            "--recommendation-outcomes", "rec-report-test",
            "--format", "json",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertIn("outcomes", data)
        self.assertTrue(len(data["outcomes"]) > 0)

    def test_recommendation_log_shows_status_counts(self):
        proc = self._run(
            str(INDEX), "analyze",
            "--db", self.db_path,
            "--recommendation-log", "rec-report-test",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("planned", proc.stdout)


if __name__ == "__main__":
    unittest.main()
