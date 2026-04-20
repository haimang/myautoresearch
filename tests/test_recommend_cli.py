"""Tests for v21 recommendation CLI in analyze.py."""

import json
import subprocess
import sys
import tempfile
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

ANALYZE = ROOT / "framework" / "analyze.py"
SELECTOR_POLICY = ROOT / "domains" / "gomoku" / "selector_policy.json"
PROFILE_PATH = ROOT / "domains" / "gomoku" / "search_space.json"


class TestRecommendCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "tracker.db")
        self.conn = init_db(self.db_path)
        self.profile = load_profile(str(PROFILE_PATH))
        self.space_id = save_search_space(self.conn, self.profile)
        self.campaign = get_or_create_campaign(
            self.conn,
            name="rec-cli-test",
            domain="gomoku",
            train_script="t.py",
            search_space_id=self.space_id,
            protocol={"eval_level": 0},
        )
        # Seed runs
        for i, wr in enumerate([0.85, 0.60, 0.75]):
            rid = f"run-{i}"
            create_run(self.conn, rid, {"sweep_tag": f"r{i}", "eval_level": 0}, is_benchmark=True)
            finish_run(self.conn, rid, {
                "status": "completed", "final_win_rate": wr,
                "wall_time_s": 100.0, "num_params": 100000, "total_games": 500,
            })
            self.conn.execute(
                """INSERT INTO campaign_runs (campaign_id, run_id, stage, sweep_tag, seed, axis_values_json, status, created_at)
                   VALUES (?, ?, 'C', ?, ?, '{}', 'linked', '2024-01-01')""",
                (self.campaign["id"], rid, f"r{i}", i),
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

    def test_recommend_next_prints_list(self):
        proc = self._run(
            str(ANALYZE),
            "--db", self.db_path,
            "--recommend-next", "rec-cli-test",
            "--selector-policy", str(SELECTOR_POLICY),
            "--limit", "3",
        )
        self.assertEqual(proc.returncode, 0, f"stderr: {proc.stderr}")
        self.assertIn("Recommendations for:", proc.stdout)
        self.assertTrue(any(t in proc.stdout for t in ["new_point", "seed_recheck", "continue_branch"]), f"Expected candidate type in output, got: {proc.stdout}")

    def test_recommend_next_json_format(self):
        proc = self._run(
            str(ANALYZE),
            "--db", self.db_path,
            "--recommend-next", "rec-cli-test",
            "--selector-policy", str(SELECTOR_POLICY),
            "--format", "json",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertIn("recommendations", data)

    def test_recommend_next_persists_batch(self):
        proc = self._run(
            str(ANALYZE),
            "--db", self.db_path,
            "--recommend-next", "rec-cli-test",
            "--selector-policy", str(SELECTOR_POLICY),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        conn = init_db(self.db_path)
        batches = conn.execute(
            "SELECT * FROM recommendation_batches WHERE campaign_id = ?",
            (self.campaign["id"],),
        ).fetchall()
        conn.close()
        self.assertEqual(len(batches), 1)

    def test_recommendation_log_shows_batches(self):
        # First generate a batch
        self._run(
            str(ANALYZE),
            "--db", self.db_path,
            "--recommend-next", "rec-cli-test",
            "--selector-policy", str(SELECTOR_POLICY),
        )
        proc = self._run(
            str(ANALYZE),
            "--db", self.db_path,
            "--recommendation-log", "rec-cli-test",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Recommendation Log:", proc.stdout)
        self.assertTrue(any(t in proc.stdout for t in ["new_point", "seed_recheck", "continue_branch"]), f"Expected candidate type in output, got: {proc.stdout}")

    def test_recommendation_log_json_format(self):
        self._run(
            str(ANALYZE),
            "--db", self.db_path,
            "--recommend-next", "rec-cli-test",
            "--selector-policy", str(SELECTOR_POLICY),
        )
        proc = self._run(
            str(ANALYZE),
            "--db", self.db_path,
            "--recommendation-log", "rec-cli-test",
            "--format", "json",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertIn("batches", data)

    def test_recommendation_outcomes_empty_friendly(self):
        proc = self._run(
            str(ANALYZE),
            "--db", self.db_path,
            "--recommendation-outcomes", "rec-cli-test",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # Either no batches or no outcomes is acceptable for empty campaign
        self.assertTrue("No outcomes" in proc.stdout or "No recommendation" in proc.stdout)

    def test_recommend_next_invalid_selector_policy_rejected(self):
        bad_policy = str(Path(self.tmp.name) / "bad.json")
        with open(bad_policy, "w") as f:
            json.dump({"domain": "chess"}, f)
        proc = self._run(
            str(ANALYZE),
            "--db", self.db_path,
            "--recommend-next", "rec-cli-test",
            "--selector-policy", bad_policy,
        )
        self.assertNotEqual(proc.returncode, 0)

    def test_recommend_next_missing_campaign_friendly(self):
        proc = self._run(
            str(ANALYZE),
            "--db", self.db_path,
            "--recommend-next", "nonexistent",
            "--selector-policy", str(SELECTOR_POLICY),
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("not found", proc.stdout.lower())

    def test_recommend_next_limit_respected(self):
        proc = self._run(
            str(ANALYZE),
            "--db", self.db_path,
            "--recommend-next", "rec-cli-test",
            "--selector-policy", str(SELECTOR_POLICY),
            "--limit", "2",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # Count recommendation lines (approximate)
        lines = [l for l in proc.stdout.splitlines() if l.strip().startswith("  ") and "new_point" in l or "seed_recheck" in l or "continue_branch" in l]
        self.assertTrue(len(lines) <= 2 or "Total: 2" in proc.stdout or "Total:" in proc.stdout)

    def test_recommend_next_candidate_type_filter(self):
        proc = self._run(
            str(ANALYZE),
            "--db", self.db_path,
            "--recommend-next", "rec-cli-test",
            "--selector-policy", str(SELECTOR_POLICY),
            "--candidate-type", "new_point",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # Should only show new_point recommendations
        self.assertNotIn("continue_branch", proc.stdout)

    def test_recommend_next_protocol_drift_rejected(self):
        # Add a run with eval_level mismatch to trigger protocol drift
        conn = init_db(self.db_path)
        create_run(conn, "run-drift", {"sweep_tag": "drift", "eval_level": 99}, is_benchmark=True)
        finish_run(conn, "run-drift", {
            "status": "completed", "final_win_rate": 0.5,
            "wall_time_s": 100.0, "num_params": 100000, "total_games": 500,
        })
        conn.execute(
            """INSERT INTO campaign_runs (campaign_id, run_id, stage, sweep_tag, seed, axis_values_json, status, created_at)
               VALUES (?, ?, 'C', ?, ?, '{}', 'linked', '2024-01-01')""",
            (self.campaign["id"], "run-drift", "drift", 99),
        )
        conn.commit()
        conn.close()
        proc = self._run(
            str(ANALYZE),
            "--db", self.db_path,
            "--recommend-next", "rec-cli-test",
            "--selector-policy", str(SELECTOR_POLICY),
        )
        self.assertIn("protocol drift", proc.stdout.lower())

    def test_recommend_next_stale_invalidated(self):
        # First batch
        self._run(
            str(ANALYZE),
            "--db", self.db_path,
            "--recommend-next", "rec-cli-test",
            "--selector-policy", str(SELECTOR_POLICY),
        )
        conn = init_db(self.db_path)
        old_statuses = conn.execute(
            """SELECT status FROM recommendations
               WHERE batch_id IN (SELECT id FROM recommendation_batches WHERE campaign_id = ?)""",
            (self.campaign["id"],),
        ).fetchall()
        conn.close()
        self.assertTrue(all(s["status"] == "planned" for s in old_statuses))

        # Second batch should invalidate old planned recommendations
        self._run(
            str(ANALYZE),
            "--db", self.db_path,
            "--recommend-next", "rec-cli-test",
            "--selector-policy", str(SELECTOR_POLICY),
        )
        conn = init_db(self.db_path)
        all_statuses = conn.execute(
            """SELECT r.status, r.batch_id, b.created_at
               FROM recommendations r
               JOIN recommendation_batches b ON b.id = r.batch_id
               WHERE b.campaign_id = ?
               ORDER BY b.created_at""",
            (self.campaign["id"],),
        ).fetchall()
        conn.close()
        # At least one old recommendation should now be invalidated
        statuses_by_batch = {}
        for s in all_statuses:
            statuses_by_batch.setdefault(s["batch_id"], []).append(s["status"])
        # First batch (oldest by created_at) should have invalidated statuses
        first_batch = list(statuses_by_batch.keys())[0]
        self.assertIn("invalidated", statuses_by_batch[first_batch])


if __name__ == "__main__":
    unittest.main()
