"""Tests for v21.1 acquisition summary and replay benchmark."""

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
    get_or_create_campaign,
    init_db,
    save_recommendation,
    save_recommendation_batch,
    save_recommendation_outcome,
    save_search_space,
    save_surrogate_snapshot,
)
from framework.profiles.search_space import load_profile

INDEX = ROOT / "framework" / "index.py"
REPLAY = ROOT / "scripts" / "v21_1_replay_benchmark.py"
PROFILE_PATH = ROOT / "domains" / "gomoku" / "manifest" / "search_space.json"


class TestAcquisitionReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "tracker.db")
        conn = init_db(self.db_path)
        profile = load_profile(str(PROFILE_PATH))
        space_id = save_search_space(conn, profile)
        self.campaign = get_or_create_campaign(
            conn,
            name="acq-report-test",
            domain="gomoku",
            train_script="t.py",
            search_space_id=space_id,
            protocol={"eval_level": 0},
        )
        conn.execute(
            """INSERT INTO frontier_snapshots
               (id, created_at, maximize_axes, minimize_axes, front_run_ids,
                dominated_count, total_runs, eval_level, sweep_tag, campaign_id)
               VALUES ('front-1', '2026-04-25T00:00:00+00:00', '["wr"]',
                       '["params","wall_s"]', '[]', 0, 0, 0, NULL, ?)""",
            (self.campaign["id"],),
        )
        save_surrogate_snapshot(
            conn,
            snapshot_id="sur-1",
            campaign_id=self.campaign["id"],
            frontier_snapshot_id="front-1",
            acquisition_name="candidate-pool-ucb",
            acquisition_version="1.0",
            policy_hash="hash",
            objectives_json='{"maximize":["win_rate"],"minimize":["num_params","wall_time_s"]}',
            feature_schema_json='{"predicted_wr":"float"}',
            summary_json='{"candidate_count":2}',
            candidate_count=2,
        )
        save_recommendation_batch(
            conn,
            batch_id="batch-1",
            campaign_id=self.campaign["id"],
            selector_name="sel",
            selector_version="1.0",
            selector_hash="hash",
            frontier_snapshot_id="front-1",
            acquisition_name="candidate-pool-ucb",
            acquisition_version="1.0",
            surrogate_snapshot_id="sur-1",
        )
        save_recommendation(
            conn,
            recommendation_id="rec-1",
            batch_id="batch-1",
            candidate_type="new_point",
            candidate_key="ck1",
            rank=1,
            score_total=0.88,
            score_breakdown_json='{}',
            rationale_json='{}',
            selector_score_total=0.50,
            acquisition_score=0.88,
        )
        save_recommendation(
            conn,
            recommendation_id="rec-2",
            batch_id="batch-1",
            candidate_type="continue_branch",
            candidate_key="ck2",
            rank=2,
            score_total=0.62,
            score_breakdown_json='{}',
            rationale_json='{}',
            selector_score_total=0.90,
            acquisition_score=0.62,
        )
        save_recommendation_outcome(
            conn,
            recommendation_id="rec-1",
            observed_metrics_json='{"final_win_rate":0.91}',
            outcome_label="new_front",
        )
        save_recommendation_outcome(
            conn,
            recommendation_id="rec-2",
            observed_metrics_json='{"final_win_rate":0.70}',
            outcome_label="no_gain",
        )
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_acquisition_summary_cli(self):
        proc = self._run(
            str(INDEX), "analyze",
            "--db", self.db_path,
            "--acquisition-summary", "acq-report-test",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("candidate-pool-ucb", proc.stdout)
        self.assertIn("Acquisition Summary", proc.stdout)

    def test_replay_benchmark_script_json(self):
        proc = self._run(
            str(REPLAY),
            "--db", self.db_path,
            "--campaign", "acq-report-test",
            "--format", "json",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual(data["summary"]["evaluated_batches"], 1)
        self.assertEqual(data["summary"]["acquisition_hits"], 1)


if __name__ == "__main__":
    unittest.main()
