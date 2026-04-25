"""Tests for v20.2 promotion CLI commands (--stage-summary, --promotion-log)."""

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
    link_run_to_campaign_v20,
    save_campaign_stage,
    save_promotion_decision,
    save_search_space,
)
from framework.profiles.search_space import load_profile

INDEX = ROOT / "framework" / "index.py"
PROFILE_PATH = ROOT / "domains" / "gomoku" / "search_space.json"


class TestPromotionCLI(unittest.TestCase):
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

    def _create_campaign_with_stages(self, name="gomoku-stage"):
        campaign = get_or_create_campaign(
            self.conn,
            name=name,
            domain="gomoku",
            train_script="domains/gomoku/train.py",
            search_space_id=self.space_id,
            protocol=self.protocol,
        )
        # Stage A
        save_campaign_stage(
            self.conn,
            campaign_id=campaign["id"],
            stage="A",
            policy_json='{"time_budget": 60}',
            budget_json='{"time_budget": 60}',
            seed_target=1,
            status="closed",
        )
        # Stage B (open)
        save_campaign_stage(
            self.conn,
            campaign_id=campaign["id"],
            stage="B",
            policy_json='{"time_budget": 180}',
            budget_json='{"time_budget": 180}',
            seed_target=2,
            status="open",
        )
        # Run linked to Stage A
        self._create_run("run-a", "gomoku-stage_A_b4_f32_sd42", 0.55)
        link_run_to_campaign_v20(
            self.conn,
            campaign_id=campaign["id"],
            run_id="run-a",
            stage="A",
            sweep_tag="gomoku-stage_A_b4_f32_sd42",
            seed=42,
            axis_values={"num_blocks": 4, "num_filters": 32},
        )
        # Promotion decision
        save_promotion_decision(
            self.conn,
            campaign_id=campaign["id"],
            from_stage="A",
            to_stage="B",
            candidate_key="b4_f32_lr0.0005_s20_buf50000",
            axis_values={"num_blocks": 4, "num_filters": 32},
            aggregated_metrics={"mean_metric": 0.55},
            seed_count=1,
            decision="promote",
            decision_rank=1,
            reason="top-1 in stage A",
        )
        return campaign

    def test_stage_summary_shows_stages(self):
        self._create_campaign_with_stages()
        proc = self._run(str(INDEX), "analyze", "--db", self.db_path, "--stage-summary", "gomoku-stage")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Stage Summary: gomoku-stage", proc.stdout)
        self.assertIn("Stage A:", proc.stdout)
        self.assertIn("Stage B:", proc.stdout)
        self.assertIn("closed", proc.stdout)
        self.assertIn("open", proc.stdout)

    def test_promotion_log_shows_decisions(self):
        self._create_campaign_with_stages()
        proc = self._run(str(INDEX), "analyze", "--db", self.db_path, "--promotion-log", "gomoku-stage")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Promotion Log: gomoku-stage", proc.stdout)
        self.assertIn("promote", proc.stdout)
        self.assertIn("top-1 in stage A", proc.stdout)

    def test_stage_summary_missing_campaign_is_friendly(self):
        proc = self._run(str(INDEX), "analyze", "--db", self.db_path, "--stage-summary", "nonexistent")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not found", proc.stdout.lower())


class TestPromoteCLI(unittest.TestCase):
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
        # Create a valid stage policy file for promote tests
        self.policy_path = str(Path(self.tmp.name) / "stage_policy.json")
        with open(self.policy_path, "w") as f:
            json.dump({
                "domain": "gomoku",
                "name": "test-policy",
                "version": "1.0",
                "search_space_ref": {"domain": "gomoku", "name": "cold-start-core", "version": "1.0"},
                "stages": [
                    {"name": "A", "time_budget": 60, "seed_count": 1, "promote_top_k": 4, "metric": "win_rate", "min_runs": 1},
                    {"name": "B", "time_budget": 180, "seed_count": 2, "promote_top_k": 2, "metric": "win_rate", "min_runs": 1},
                    {"name": "C", "time_budget": 600, "seed_count": 3, "promote_top_k": 1, "metric": "win_rate", "min_runs": 1},
                    {"name": "D", "time_budget": 1800, "seed_count": 3, "promote_top_k": 0, "metric": "win_rate", "min_runs": 1},
                ]
            }, f)
        self.promote = ROOT / "framework" / "index.py"

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

    def _create_campaign_with_runs(self, name="promote-test"):
        campaign = get_or_create_campaign(
            self.conn,
            name=name,
            domain="gomoku",
            train_script="domains/gomoku/train.py",
            search_space_id=self.space_id,
            protocol=self.protocol,
        )
        # Seed 4 completed runs in Stage A
        for i, (nb, nf, wr) in enumerate([(4, 32, 0.75), (4, 64, 0.90), (6, 32, 0.82), (6, 64, 0.665)]):
            run_id = f"run-{i}"
            hyperparams = {
                "num_res_blocks": nb,
                "num_filters": nf,
                "learning_rate": 5e-4,
                "train_steps_per_cycle": 20,
                "replay_buffer_size": 50000,
                "time_budget": 30,
                "eval_level": 0,
                "sweep_tag": f"{name}_A_b{nb}_f{nf}_sd42",
                "seed": 42,
            }
            create_run(self.conn, run_id, hyperparams, is_benchmark=True)
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
            link_run_to_campaign_v20(
                self.conn,
                campaign_id=campaign["id"],
                run_id=run_id,
                stage="A",
                sweep_tag=f"{name}_A_b{nb}_f{nf}_sd42",
                seed=42,
                axis_values={"num_blocks": nb, "num_filters": nf},
            )
        save_campaign_stage(
            self.conn,
            campaign_id=campaign["id"],
            stage="A",
            policy_json='{"time_budget": 60}',
            budget_json='{"time_budget": 60}',
            seed_target=1,
            status="closed",
        )
        return campaign

    def test_promote_plan_outputs_decisions(self):
        self._create_campaign_with_runs()
        proc = self._run(
            str(self.promote), "promote",
            "--db", self.db_path,
            "--campaign", "promote-test",
            "--stage-policy", self.policy_path,
            "--from-stage", "A",
            "--to-stage", "B",
            "--plan",
        )
        self.assertEqual(proc.returncode, 0, f"stderr: {proc.stderr}")
        self.assertIn("Promotion Plan: A → B", proc.stdout)
        self.assertIn("promote", proc.stdout)
        self.assertIn("Summary:", proc.stdout)

    def test_promote_plan_domain_mismatch_rejected(self):
        self._create_campaign_with_runs()
        bad_policy = str(Path(self.tmp.name) / "bad_domain.json")
        with open(bad_policy, "w") as f:
            json.dump({
                "domain": "chess",
                "name": "bad",
                "version": "1.0",
                "search_space_ref": {"domain": "chess", "name": "ss", "version": "1.0"},
                "stages": [
                    {"name": "A", "time_budget": 60, "seed_count": 1, "promote_top_k": 4, "metric": "win_rate", "min_runs": 1},
                    {"name": "B", "time_budget": 180, "seed_count": 2, "promote_top_k": 2, "metric": "win_rate", "min_runs": 1},
                    {"name": "C", "time_budget": 600, "seed_count": 3, "promote_top_k": 1, "metric": "win_rate", "min_runs": 1},
                    {"name": "D", "time_budget": 1800, "seed_count": 3, "promote_top_k": 0, "metric": "win_rate", "min_runs": 1},
                ]
            }, f)
        proc = self._run(
            str(self.promote), "promote",
            "--db", self.db_path,
            "--campaign", "promote-test",
            "--stage-policy", bad_policy,
            "--from-stage", "A",
            "--to-stage", "B",
            "--plan",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("domain", proc.stdout)

    def test_promote_plan_search_space_mismatch_rejected(self):
        self._create_campaign_with_runs()
        bad_policy = str(Path(self.tmp.name) / "bad_ss.json")
        with open(bad_policy, "w") as f:
            json.dump({
                "domain": "gomoku",
                "name": "bad",
                "version": "1.0",
                "search_space_ref": {"domain": "gomoku", "name": "wrong-space", "version": "9.9"},
                "stages": [
                    {"name": "A", "time_budget": 60, "seed_count": 1, "promote_top_k": 4, "metric": "win_rate", "min_runs": 1},
                    {"name": "B", "time_budget": 180, "seed_count": 2, "promote_top_k": 2, "metric": "win_rate", "min_runs": 1},
                    {"name": "C", "time_budget": 600, "seed_count": 3, "promote_top_k": 1, "metric": "win_rate", "min_runs": 1},
                    {"name": "D", "time_budget": 1800, "seed_count": 3, "promote_top_k": 0, "metric": "win_rate", "min_runs": 1},
                ]
            }, f)
        proc = self._run(
            str(self.promote), "promote",
            "--db", self.db_path,
            "--campaign", "promote-test",
            "--stage-policy", bad_policy,
            "--from-stage", "A",
            "--to-stage", "B",
            "--plan",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("search_space_ref", proc.stdout)

    def test_promote_execute_stage_d_blocked(self):
        self._create_campaign_with_runs()
        proc = self._run(
            str(self.promote), "promote",
            "--db", self.db_path,
            "--campaign", "promote-test",
            "--stage-policy", self.policy_path,
            "--from-stage", "C",
            "--to-stage", "D",
            "--execute",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("blocked", proc.stdout.lower())


if __name__ == "__main__":
    unittest.main()
