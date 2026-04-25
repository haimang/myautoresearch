"""Tests for v20.3 branch.py CLI — plan / execute / guard behavior."""

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
    get_branch_by_id,
    get_or_create_campaign,
    init_db,
    save_campaign_stage,
    save_search_space,
)
from framework.profiles.search_space import load_profile

INDEX = ROOT / "framework" / "index.py"
PROFILE_PATH = ROOT / "domains" / "gomoku" / "search_space.json"


class TestBranchCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "tracker.db")
        self.conn = init_db(self.db_path)
        self.profile = load_profile(str(PROFILE_PATH))
        self.space_id = save_search_space(self.conn, self.profile)
        self.campaign = get_or_create_campaign(
            self.conn,
            name="branch-cli-test",
            domain="gomoku",
            train_script="domains/gomoku/train.py",
            search_space_id=self.space_id,
            protocol={"eval_level": 0, "is_benchmark": True, "train_script": "domains/gomoku/train.py"},
        )
        # Create a valid branch policy file
        self.policy_path = str(Path(self.tmp.name) / "branch_policy.json")
        with open(self.policy_path, "w") as f:
            json.dump({
                "domain": "gomoku",
                "name": "test-branch-policy",
                "version": "1.0",
                "search_space_ref": {"domain": "gomoku", "name": "cold-start-core", "version": "1.0"},
                "stage_policy_ref": {"domain": "gomoku", "name": "cold-start-promotion", "version": "1.0"},
                "branch_reasons": {
                    "lr_decay": {
                        "description": "Reduce LR",
                        "allowed_deltas": {"learning_rate": {"type": "multiply", "default_factor": 0.1, "min_factor": 0.01, "max_factor": 0.5}},
                        "preserves_protocol": True,
                    },
                    "seed_recheck": {
                        "description": "Change seed",
                        "allowed_deltas": {"seed": {"type": "set", "default_value": 99}},
                        "preserves_protocol": True,
                    },
                    "eval_upgrade": {
                        "description": "Upgrade eval",
                        "allowed_deltas": {"eval_level": {"type": "add", "default_delta": 1, "min_delta": 1, "max_delta": 2}},
                        "preserves_protocol": False,
                        "allowed_protocol_changes": ["eval_level"],
                    },
                    "mcts_upshift": {
                        "description": "MCTS",
                        "allowed_deltas": {"mcts_simulations": {"type": "add", "default_delta": 200}},
                        "preserves_protocol": True,
                    },
                    "buffer_or_spc_adjust": {
                        "description": "Buffer",
                        "allowed_deltas": {"buffer_size": {"type": "multiply", "default_factor": 2.0}},
                        "preserves_protocol": True,
                    },
                }
            }, f)

        # Seed a parent run with checkpoint
        create_run(self.conn, "parent-run", {
            "sweep_tag": "parent_tag", "eval_level": 0, "learning_rate": 0.01,
            "num_res_blocks": 4, "num_filters": 32,
        }, is_benchmark=True)
        finish_run(self.conn, "parent-run", {
            "status": "completed", "final_win_rate": 0.80,
            "wall_time_s": 100.0, "num_params": 100000, "total_games": 500,
        })
        self.conn.execute(
            """INSERT INTO checkpoints (run_id, tag, cycle, step, loss, win_rate, eval_level, eval_games, model_path, created_at)
               VALUES ('parent-run', 'ckpt_final', 10, 1000, 0.5, 0.8, 0, 200, 'model.safetensors', '2024-01-01')"""
        )
        # Link to campaign stage C
        self.conn.execute(
            """INSERT INTO campaign_runs (campaign_id, run_id, stage, sweep_tag, seed, axis_values_json, status, created_at)
               VALUES (?, 'parent-run', 'C', 'parent_tag', 42, '{}', 'linked', '2024-01-01')""",
            (self.campaign["id"],),
        )
        save_campaign_stage(
            self.conn,
            campaign_id=self.campaign["id"],
            stage="C",
            policy_json='{"time_budget": 60}',
            budget_json='{"time_budget": 60}',
            seed_target=1,
            status="closed",
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

    def test_branch_plan_dry_run_shows_branches(self):
        proc = self._run(
            str(INDEX), "branch",
            "--db", self.db_path,
            "--campaign", "branch-cli-test",
            "--branch-policy", self.policy_path,
            "--parent-checkpoint", "ckpt_final",
            "--reason", "lr_decay",
            "--reason", "seed_recheck",
            "--plan", "--dry-run",
        )
        self.assertEqual(proc.returncode, 0, f"stderr: {proc.stderr}")
        self.assertIn("Branch Plan:", proc.stdout)
        self.assertIn("lr_decay", proc.stdout)
        self.assertIn("seed_recheck", proc.stdout)
        self.assertIn("dry-run", proc.stdout)

    def test_branch_plan_persists_branches(self):
        proc = self._run(
            str(INDEX), "branch",
            "--db", self.db_path,
            "--campaign", "branch-cli-test",
            "--branch-policy", self.policy_path,
            "--parent-checkpoint", "ckpt_final",
            "--reason", "lr_decay",
            "--plan",
        )
        self.assertEqual(proc.returncode, 0, f"stderr: {proc.stderr}")
        self.assertIn("Persisted", proc.stdout)
        # Verify in DB
        conn = init_db(self.db_path)
        rows = conn.execute("SELECT * FROM run_branches WHERE campaign_id = ?", (self.campaign["id"],)).fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["branch_reason"], "lr_decay")

    def test_branch_plan_domain_mismatch_rejected(self):
        bad_policy = str(Path(self.tmp.name) / "bad_domain.json")
        with open(bad_policy, "w") as f:
            json.dump({
                "domain": "chess",
                "name": "bad",
                "version": "1.0",
                "search_space_ref": {"domain": "chess", "name": "ss"},
                "stage_policy_ref": {"domain": "chess", "name": "sp"},
                "branch_reasons": {
                    "lr_decay": {"description": "x", "allowed_deltas": {"lr": {"type": "multiply"}}, "preserves_protocol": True},
                    "mcts_upshift": {"description": "x", "allowed_deltas": {"mcts": {"type": "add"}}, "preserves_protocol": True},
                    "eval_upgrade": {"description": "x", "allowed_deltas": {"eval": {"type": "add"}}, "preserves_protocol": False, "allowed_protocol_changes": ["eval"]},
                    "seed_recheck": {"description": "x", "allowed_deltas": {"seed": {"type": "set"}}, "preserves_protocol": True},
                    "buffer_or_spc_adjust": {"description": "x", "allowed_deltas": {"buf": {"type": "multiply"}}, "preserves_protocol": True},
                }
            }, f)
        proc = self._run(
            str(INDEX), "branch",
            "--db", self.db_path,
            "--campaign", "branch-cli-test",
            "--branch-policy", bad_policy,
            "--parent-checkpoint", "ckpt_final",
            "--reason", "lr_decay",
            "--plan",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("domain", proc.stdout.lower())

    def test_branch_plan_search_space_mismatch_rejected(self):
        bad_policy = str(Path(self.tmp.name) / "bad_ss.json")
        with open(bad_policy, "w") as f:
            json.dump({
                "domain": "gomoku",
                "name": "bad",
                "version": "1.0",
                "search_space_ref": {"domain": "gomoku", "name": "wrong-space", "version": "9.9"},
                "stage_policy_ref": {"domain": "gomoku", "name": "sp"},
                "branch_reasons": {
                    "lr_decay": {"description": "x", "allowed_deltas": {"lr": {"type": "multiply"}}, "preserves_protocol": True},
                    "mcts_upshift": {"description": "x", "allowed_deltas": {"mcts": {"type": "add"}}, "preserves_protocol": True},
                    "eval_upgrade": {"description": "x", "allowed_deltas": {"eval": {"type": "add"}}, "preserves_protocol": False, "allowed_protocol_changes": ["eval"]},
                    "seed_recheck": {"description": "x", "allowed_deltas": {"seed": {"type": "set"}}, "preserves_protocol": True},
                    "buffer_or_spc_adjust": {"description": "x", "allowed_deltas": {"buf": {"type": "multiply"}}, "preserves_protocol": True},
                }
            }, f)
        proc = self._run(
            str(INDEX), "branch",
            "--db", self.db_path,
            "--campaign", "branch-cli-test",
            "--branch-policy", bad_policy,
            "--parent-checkpoint", "ckpt_final",
            "--reason", "lr_decay",
            "--plan",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("search_space_ref", proc.stdout)

    def test_branch_plan_invalid_reason_rejected(self):
        proc = self._run(
            str(INDEX), "branch",
            "--db", self.db_path,
            "--campaign", "branch-cli-test",
            "--branch-policy", self.policy_path,
            "--parent-checkpoint", "ckpt_final",
            "--reason", "not_a_real_reason",
            "--plan",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("unknown", proc.stdout.lower())

    def test_branch_plan_invalid_delta_rejected(self):
        proc = self._run(
            str(INDEX), "branch",
            "--db", self.db_path,
            "--campaign", "branch-cli-test",
            "--branch-policy", self.policy_path,
            "--parent-checkpoint", "ckpt_final",
            "--reason", "lr_decay",
            "--delta", '{"learning_rate": 0.99}',
            "--plan",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("max", proc.stdout.lower())

    def test_branch_plan_missing_checkpoint_friendly(self):
        proc = self._run(
            str(INDEX), "branch",
            "--db", self.db_path,
            "--campaign", "branch-cli-test",
            "--branch-policy", self.policy_path,
            "--parent-checkpoint", "nonexistent_ckpt",
            "--reason", "lr_decay",
            "--plan",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("not found", proc.stdout.lower())

    def test_branch_plan_duplicate_not_duplicated(self):
        # First plan
        self._run(
            str(INDEX), "branch",
            "--db", self.db_path,
            "--campaign", "branch-cli-test",
            "--branch-policy", self.policy_path,
            "--parent-checkpoint", "ckpt_final",
            "--reason", "lr_decay",
            "--plan",
        )
        # Second plan — same params
        proc = self._run(
            str(INDEX), "branch",
            "--db", self.db_path,
            "--campaign", "branch-cli-test",
            "--branch-policy", self.policy_path,
            "--parent-checkpoint", "ckpt_final",
            "--reason", "lr_decay",
            "--plan",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        conn = init_db(self.db_path)
        rows = conn.execute("SELECT * FROM run_branches WHERE campaign_id = ?", (self.campaign["id"],)).fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)  # upsert, not duplicate

    def test_branch_plan_no_stage_c_runs_rejected(self):
        # Create campaign with no Stage C runs
        conn = init_db(self.db_path)
        c = get_or_create_campaign(
            conn, name="empty-branch", domain="gomoku", train_script="t.py",
            search_space_id=self.space_id, protocol={"eval_level": 0},
        )
        conn.close()
        proc = self._run(
            str(INDEX), "branch",
            "--db", self.db_path,
            "--campaign", "empty-branch",
            "--branch-policy", self.policy_path,
            "--reason", "lr_decay",
            "--plan",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("no completed runs", proc.stdout.lower())

    def test_branch_execute_mock_train_creates_child_and_links_campaign(self):
        """Execute with mock train script: child run created, linked to branch and campaign."""
        mock_train = str(ROOT / "tests" / "mock_train.py")
        proc = self._run(
            str(INDEX), "branch",
            "--db", self.db_path,
            "--campaign", "branch-cli-test",
            "--branch-policy", self.policy_path,
            "--parent-checkpoint", "ckpt_final",
            "--reason", "lr_decay",
            "--execute",
            "--train-script", mock_train,
            "--time-budget", "10",
        )
        self.assertEqual(proc.returncode, 0, f"stderr: {proc.stderr}")
        conn = init_db(self.db_path)
        rows = conn.execute(
            "SELECT * FROM run_branches WHERE campaign_id = ?",
            (self.campaign["id"],),
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertIsNotNone(rows[0]["child_run_id"])
        self.assertEqual(rows[0]["status"], "completed")
        # Verify child also in campaign_runs (stage D)
        cr = conn.execute(
            "SELECT * FROM campaign_runs WHERE campaign_id = ? AND stage = 'D'",
            (self.campaign["id"],),
        ).fetchall()
        conn.close()
        self.assertEqual(len(cr), 1)
        self.assertEqual(cr[0]["run_id"], rows[0]["child_run_id"])

    def test_branch_protocol_guard_rejects_drift(self):
        """A reason declaring preserves_protocol=true must not change protocol fields."""
        bad_policy = str(Path(self.tmp.name) / "bad_proto.json")
        with open(bad_policy, "w") as f:
            json.dump({
                "domain": "gomoku",
                "name": "bad",
                "version": "1.0",
                "search_space_ref": {"domain": "gomoku", "name": "cold-start-core", "version": "1.0"},
                "stage_policy_ref": {"domain": "gomoku", "name": "cold-start-promotion", "version": "1.0"},
                "branch_reasons": {
                    "lr_decay": {
                        "description": "x",
                        "allowed_deltas": {
                            "learning_rate": {"type": "multiply", "default_factor": 0.1},
                            "eval_level": {"type": "set", "default_value": 99},
                        },
                        "preserves_protocol": True,
                    },
                    "mcts_upshift": {"description": "x", "allowed_deltas": {"mcts_simulations": {"type": "add"}}, "preserves_protocol": True},
                    "eval_upgrade": {"description": "x", "allowed_deltas": {"eval_level": {"type": "add"}}, "preserves_protocol": False, "allowed_protocol_changes": ["eval_level"]},
                    "seed_recheck": {"description": "x", "allowed_deltas": {"seed": {"type": "set"}}, "preserves_protocol": True},
                    "buffer_or_spc_adjust": {"description": "x", "allowed_deltas": {"buffer_size": {"type": "multiply"}}, "preserves_protocol": True},
                }
            }, f)
        proc = self._run(
            str(INDEX), "branch",
            "--db", self.db_path,
            "--campaign", "branch-cli-test",
            "--branch-policy", bad_policy,
            "--parent-checkpoint", "ckpt_final",
            "--reason", "lr_decay",
            "--plan",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("preserves_protocol", proc.stdout)

    def test_branch_execute_child_failure_updates_status(self):
        """When child train script fails, branch status must be updated to 'failed'."""
        fail_script = str(Path(self.tmp.name) / "fail_train.py")
        with open(fail_script, "w") as f:
            f.write("raise SystemExit(1)\n")
        proc = self._run(
            str(INDEX), "branch",
            "--db", self.db_path,
            "--campaign", "branch-cli-test",
            "--branch-policy", self.policy_path,
            "--parent-checkpoint", "ckpt_final",
            "--reason", "lr_decay",
            "--execute",
            "--train-script", fail_script,
            "--time-budget", "10",
        )
        self.assertEqual(proc.returncode, 0, f"Unexpected crash: {proc.stderr}")
        conn = init_db(self.db_path)
        rows = conn.execute(
            "SELECT * FROM run_branches WHERE campaign_id = ?",
            (self.campaign["id"],),
        ).fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "failed")

    def test_branch_stage_policy_ref_domain_mismatch_rejected(self):
        """Branch policy stage_policy_ref domain must match campaign domain."""
        conn = init_db(self.db_path)
        c = get_or_create_campaign(
            conn, name="stage-mismatch-test", domain="gomoku", train_script="t.py",
            search_space_id=self.space_id, protocol={"eval_level": 0},
        )
        # Seed parent run + checkpoint + stage C
        create_run(conn, "parent-run-2", {
            "sweep_tag": "parent2", "eval_level": 0, "learning_rate": 0.01,
            "num_res_blocks": 4, "num_filters": 32,
        }, is_benchmark=True)
        finish_run(conn, "parent-run-2", {
            "status": "completed", "final_win_rate": 0.80,
            "wall_time_s": 100.0, "num_params": 100000, "total_games": 500,
        })
        conn.execute(
            """INSERT INTO checkpoints (run_id, tag, cycle, step, loss, win_rate, eval_level, eval_games, model_path, created_at)
               VALUES ('parent-run-2', 'ckpt_final2', 10, 1000, 0.5, 0.8, 0, 200, 'model.safetensors', '2024-01-01')"""
        )
        conn.execute(
            """INSERT INTO campaign_runs (campaign_id, run_id, stage, sweep_tag, seed, axis_values_json, status, created_at)
               VALUES (?, 'parent-run-2', 'C', 'parent2', 42, '{}', 'linked', '2024-01-01')""",
            (c["id"],),
        )
        save_campaign_stage(
            conn,
            campaign_id=c["id"],
            stage="C",
            policy_json='{"time_budget": 60}',
            budget_json='{"time_budget": 60}',
            seed_target=1,
            status="closed",
        )
        conn.commit()
        conn.close()
        # Create a branch policy with mismatched stage_policy_ref domain
        bad_policy = str(Path(self.tmp.name) / "bad_stage_domain.json")
        with open(bad_policy, "w") as f:
            json.dump({
                "domain": "gomoku",
                "name": "test",
                "version": "1.0",
                "search_space_ref": {"domain": "gomoku", "name": "cold-start-core", "version": "1.0"},
                "stage_policy_ref": {"domain": "chess", "name": "cold-start-promotion", "version": "1.0"},
                "branch_reasons": {
                    "lr_decay": {"description": "x", "allowed_deltas": {"learning_rate": {"type": "multiply"}}, "preserves_protocol": True},
                    "mcts_upshift": {"description": "x", "allowed_deltas": {"mcts_simulations": {"type": "add"}}, "preserves_protocol": True},
                    "eval_upgrade": {"description": "x", "allowed_deltas": {"eval_level": {"type": "add"}}, "preserves_protocol": False, "allowed_protocol_changes": ["eval_level"]},
                    "seed_recheck": {"description": "x", "allowed_deltas": {"seed": {"type": "set"}}, "preserves_protocol": True},
                    "buffer_or_spc_adjust": {"description": "x", "allowed_deltas": {"buffer_size": {"type": "multiply"}}, "preserves_protocol": True},
                }
            }, f)
        proc = self._run(
            str(INDEX), "branch",
            "--db", self.db_path,
            "--campaign", "stage-mismatch-test",
            "--branch-policy", bad_policy,
            "--parent-checkpoint", "ckpt_final2",
            "--reason", "lr_decay",
            "--plan",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("stage_policy_ref", proc.stdout)


if __name__ == "__main__":
    unittest.main()
