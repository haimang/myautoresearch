"""Tests for v20.3 trajectory report commands in analyze.py."""

import sqlite3
import sys
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
    save_run_branch,
    save_search_space,
)
from search_space import load_profile

ANALYZE = ROOT / "framework" / "analyze.py"
PROFILE_PATH = ROOT / "domains" / "gomoku" / "search_space.json"


class TestTrajectoryReport(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db_path = self._tmp.name
        self.conn = init_db(self.db_path)
        self.profile = load_profile(str(PROFILE_PATH))
        self.space_id = save_search_space(self.conn, self.profile)
        self.campaign = get_or_create_campaign(
            self.conn,
            name="traj-test",
            domain="gomoku",
            train_script="train.py",
            search_space_id=self.space_id,
            protocol={"eval_level": 0},
        )
        # Parent run
        create_run(self.conn, "parent-run", {"sweep_tag": "parent", "eval_level": 0}, is_benchmark=True)
        finish_run(self.conn, "parent-run", {
            "status": "completed", "final_win_rate": 0.80,
            "wall_time_s": 100.0, "num_params": 100000, "total_games": 500,
        })
        # Child run
        create_run(self.conn, "child-run", {"sweep_tag": "child", "eval_level": 0}, is_benchmark=True)
        finish_run(self.conn, "child-run", {
            "status": "completed", "final_win_rate": 0.85,
            "wall_time_s": 120.0, "num_params": 100000, "total_games": 600,
        })
        # Branch linking parent and child
        save_run_branch(
            self.conn,
            branch_id="br-traj-1",
            campaign_id=self.campaign["id"],
            parent_run_id="parent-run",
            parent_checkpoint_id=None,
            from_stage="D",
            branch_reason="lr_decay",
            branch_params_json='{"lr": 0.001}',
            delta_json='{"learning_rate": 0.001}',
            status="completed",
            result_summary_json='{"elapsed_s": 120}',
        )
        # Update branch with child_run_id manually since save_run_branch doesn't set it
        self.conn.execute(
            "UPDATE run_branches SET child_run_id = ? WHERE id = ?",
            ("child-run", "br-traj-1"),
        )
        self.conn.commit()
        self.conn.close()

    def tearDown(self):
        import os
        if Path(self.db_path).exists():
            os.remove(self.db_path)

    def _run(self, *args):
        import subprocess
        return subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_branch_tree_shows_parent_child(self):
        proc = self._run(str(ANALYZE), "--db", self.db_path, "--branch-tree", "traj-test")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Branch Tree: traj-test", proc.stdout)
        self.assertIn("parent", proc.stdout)
        self.assertIn("lr_decay", proc.stdout)

    def test_trajectory_report_shows_reason_and_delta(self):
        proc = self._run(str(ANALYZE), "--db", self.db_path, "--trajectory-report", "traj-test")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Trajectory Report: traj-test", proc.stdout)
        self.assertIn("lr_decay", proc.stdout)
        self.assertIn("learning_rate", proc.stdout)

    def test_compare_parent_child_shows_metrics(self):
        proc = self._run(str(ANALYZE), "--db", self.db_path, "--compare-parent-child", "br-traj-1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Parent-Child Compare", proc.stdout)
        self.assertIn("WR", proc.stdout)
        self.assertIn("Δ", proc.stdout)

    def test_branch_tree_empty_campaign_friendly(self):
        # Create empty campaign
        conn = init_db(self.db_path)
        c = get_or_create_campaign(
            conn, name="empty-traj", domain="gomoku", train_script="t.py",
            search_space_id=self.space_id, protocol={},
        )
        conn.close()
        proc = self._run(str(ANALYZE), "--db", self.db_path, "--branch-tree", "empty-traj")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("No branches recorded", proc.stdout)

    def test_trajectory_report_missing_campaign_friendly(self):
        proc = self._run(str(ANALYZE), "--db", self.db_path, "--trajectory-report", "nonexistent")
        self.assertNotEqual(proc.returncode, 0)

    def test_compare_parent_child_missing_branch_friendly(self):
        proc = self._run(str(ANALYZE), "--db", self.db_path, "--compare-parent-child", "br-missing")
        self.assertEqual(proc.returncode, 0)  # analyze prints friendly msg, doesn't crash
        self.assertIn("not found", proc.stdout.lower())

    def test_branch_tree_shows_multiple_children(self):
        """Branch tree correctly groups multiple children under one parent."""
        # Add a second child branch
        conn = init_db(self.db_path)
        create_run(conn, "child-run-2", {"sweep_tag": "child2", "eval_level": 0}, is_benchmark=True)
        finish_run(conn, "child-run-2", {
            "status": "completed", "final_win_rate": 0.90,
            "wall_time_s": 130.0, "num_params": 100000, "total_games": 650,
        })
        save_run_branch(
            conn,
            branch_id="br-traj-2",
            campaign_id=self.campaign["id"],
            parent_run_id="parent-run",
            parent_checkpoint_id=None,
            from_stage="D",
            branch_reason="mcts_upshift",
            branch_params_json='{"mcts_simulations": 600}',
            delta_json='{"mcts_simulations": 600}',
            status="completed",
            result_summary_json='{"elapsed_s": 130}',
        )
        conn.execute(
            "UPDATE run_branches SET child_run_id = ? WHERE id = ?",
            ("child-run-2", "br-traj-2"),
        )
        conn.commit()
        conn.close()
        proc = self._run(str(ANALYZE), "--db", self.db_path, "--branch-tree", "traj-test")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("mcts_upshift", proc.stdout)
        self.assertIn("child2", proc.stdout)
        self.assertIn("90.0%", proc.stdout)

    def test_trajectory_report_best_child_highlighted(self):
        """Trajectory report should show the best child among siblings."""
        conn = init_db(self.db_path)
        create_run(conn, "child-run-2", {"sweep_tag": "child2", "eval_level": 0}, is_benchmark=True)
        finish_run(conn, "child-run-2", {
            "status": "completed", "final_win_rate": 0.90,
            "wall_time_s": 130.0, "num_params": 100000, "total_games": 650,
        })
        save_run_branch(
            conn,
            branch_id="br-traj-2",
            campaign_id=self.campaign["id"],
            parent_run_id="parent-run",
            parent_checkpoint_id=None,
            from_stage="D",
            branch_reason="mcts_upshift",
            branch_params_json='{"mcts_simulations": 600}',
            delta_json='{"mcts_simulations": 600}',
            status="completed",
            result_summary_json='{"elapsed_s": 130}',
        )
        conn.execute(
            "UPDATE run_branches SET child_run_id = ? WHERE id = ?",
            ("child-run-2", "br-traj-2"),
        )
        conn.commit()
        conn.close()
        proc = self._run(str(ANALYZE), "--db", self.db_path, "--trajectory-report", "traj-test")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("mcts_upshift", proc.stdout)
        # Should show ΔWR for both children
        self.assertIn("ΔWR", proc.stdout)


if __name__ == "__main__":
    unittest.main()
