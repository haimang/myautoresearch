import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.core.db import create_run, finish_run, get_or_create_campaign, init_db, link_run_to_campaign, save_search_space
from framework.profiles.search_space import load_profile


INDEX = ROOT / "framework" / "index.py"
PROFILE_PATH = ROOT / "domains" / "gomoku" / "search_space.json"


class TestCampaignCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "tracker.db")
        self.conn = init_db(self.db_path)
        self.profile = load_profile(str(PROFILE_PATH))
        self.space_id = save_search_space(self.conn, self.profile)
        self.protocol = {
            "eval_level": 0,
            "eval_opponent": None,
            "is_benchmark": True,
            "train_script": "domains/gomoku/train.py",
        }

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def _create_run(self, run_id: str, sweep_tag: str, wr: float, eval_level: int = 0):
        hyperparams = {
            "num_res_blocks": 4,
            "num_filters": 32,
            "learning_rate": 5e-4,
            "train_steps_per_cycle": 20,
            "replay_buffer_size": 50000,
            "time_budget": 30,
            "eval_level": eval_level,
            "sweep_tag": sweep_tag,
            "seed": 42,
        }
        create_run(self.conn, run_id, hyperparams, is_benchmark=(eval_level == 0))
        finish_run(self.conn, run_id, {
            "status": "completed",
            "total_cycles": 5,
            "total_games": 40,
            "total_steps": 100,
            "final_loss": 1.23,
            "final_win_rate": wr,
            "num_params": 12345,
            "num_checkpoints": 0,
            "wall_time_s": 20.0,
            "peak_memory_mb": 100.0,
        })

    def _create_campaign_with_runs(self, name="gomoku-smoke", include_drift=False):
        campaign = get_or_create_campaign(
            self.conn,
            name=name,
            domain="gomoku",
            train_script="domains/gomoku/train.py",
            search_space_id=self.space_id,
            protocol=self.protocol,
        )
        self._create_run("run-a", "gomoku-smoke_b4_f32_sd42", 0.55, eval_level=0)
        link_run_to_campaign(
            self.conn,
            campaign_id=campaign["id"],
            run_id="run-a",
            stage=None,
            sweep_tag="gomoku-smoke_b4_f32_sd42",
            seed=42,
            axis_values={"num_blocks": 4, "num_filters": 32},
        )
        if include_drift:
            self._create_run("run-b", "gomoku-smoke_b6_f64_sd42", 0.65, eval_level=1)
            link_run_to_campaign(
                self.conn,
                campaign_id=campaign["id"],
                run_id="run-b",
                stage=None,
                sweep_tag="gomoku-smoke_b6_f64_sd42",
                seed=42,
                axis_values={"num_blocks": 6, "num_filters": 64},
            )
        return campaign

    def test_sweep_dry_run_prints_profile_and_campaign(self):
        proc = self._run(
            str(INDEX), "sweep",
            "--db", self.db_path,
            "--train-script", "domains/gomoku/train.py",
            "--campaign", "gomoku-smoke",
            "--search-space", str(PROFILE_PATH),
            "--num-blocks", "4,6",
            "--num-filters", "32,64",
            "--time-budget", "10",
            "--seeds", "42",
            "--eval-level", "0",
            "--dry-run",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Search Space: gomoku/cold-start-core", proc.stdout)
        self.assertIn("Campaign: gomoku-smoke", proc.stdout)

    def test_same_protocol_campaign_can_reuse_name(self):
        self._create_campaign_with_runs()
        proc = self._run(
            str(INDEX), "sweep",
            "--db", self.db_path,
            "--train-script", "domains/gomoku/train.py",
            "--campaign", "gomoku-smoke",
            "--search-space", str(PROFILE_PATH),
            "--num-blocks", "4",
            "--num-filters", "32",
            "--time-budget", "10",
            "--seeds", "42",
            "--eval-level", "0",
            "--dry-run",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_different_protocol_campaign_is_rejected(self):
        self._create_campaign_with_runs()
        proc = self._run(
            str(INDEX), "sweep",
            "--db", self.db_path,
            "--train-script", "domains/gomoku/train.py",
            "--campaign", "gomoku-smoke",
            "--search-space", str(PROFILE_PATH),
            "--num-blocks", "4",
            "--num-filters", "32",
            "--time-budget", "10",
            "--seeds", "42",
            "--eval-level", "1",
            "--dry-run",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("protocol", proc.stdout)

    def test_list_campaigns_outputs_campaign_row(self):
        self._create_campaign_with_runs()
        proc = self._run(str(INDEX), "analyze", "--db", self.db_path, "--list-campaigns")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("gomoku-smoke", proc.stdout)

    def test_campaign_summary_outputs_protocol_and_run_count(self):
        self._create_campaign_with_runs()
        proc = self._run(str(INDEX), "analyze", "--db", self.db_path, "--campaign-summary", "gomoku-smoke")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Campaign Summary: gomoku-smoke", proc.stdout)
        self.assertIn("Runs:          1", proc.stdout)
        self.assertIn("Protocol drift: none", proc.stdout)

    def test_pareto_campaign_filters_outside_runs(self):
        self._create_campaign_with_runs()
        self._create_run("run-outside", "outside_sd42", 0.99, eval_level=0)
        proc = self._run(str(INDEX), "analyze", "--db", self.db_path, "--pareto", "--campaign", "gomoku-smoke")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("run-a", proc.stdout)
        self.assertNotIn("run-outsid", proc.stdout)

    def test_pareto_campaign_rejects_protocol_drift_by_default(self):
        self._create_campaign_with_runs(include_drift=True)
        proc = self._run(str(INDEX), "analyze", "--db", self.db_path, "--pareto", "--campaign", "gomoku-smoke")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("refusing Pareto output", proc.stdout)

    def test_campaign_summary_missing_campaign_is_friendly(self):
        proc = self._run(str(INDEX), "analyze", "--db", self.db_path, "--campaign-summary", "missing-campaign")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Campaign not found", proc.stdout)


if __name__ == "__main__":
    unittest.main()
