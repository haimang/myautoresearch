"""Tests for v20.3 branch DB helpers (run_branches, lineage)."""

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from core.db import (
    bind_branch_child_run,
    create_run,
    finish_run,
    get_branch_by_id,
    get_branch_tree,
    get_or_create_campaign,
    init_db,
    list_branches_for_campaign,
    list_branches_for_checkpoint,
    save_campaign_stage,
    save_run_branch,
    save_search_space,
    update_branch_status,
)
from search_space import load_profile


class TestBranchDB(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")
        self.profile = load_profile(str(ROOT / "domains" / "gomoku" / "search_space.json"))
        self.space_id = save_search_space(self.conn, self.profile)
        self.campaign = get_or_create_campaign(
            self.conn,
            name="branch-test",
            domain="gomoku",
            train_script="train.py",
            search_space_id=self.space_id,
            protocol={"eval_level": 0},
        )
        # Seed parent run + checkpoint
        create_run(self.conn, "parent-run", {"sweep_tag": "parent", "eval_level": 0}, is_benchmark=True)
        finish_run(self.conn, "parent-run", {
            "status": "completed",
            "final_win_rate": 0.8,
            "wall_time_s": 100.0,
            "num_params": 12345,
            "total_games": 100,
        })
        self.conn.execute(
            """INSERT INTO checkpoints (run_id, tag, cycle, step, loss, win_rate, eval_level, eval_games, model_path, created_at)
               VALUES ('parent-run', 'ckpt_final', 10, 1000, 0.5, 0.8, 0, 200, 'model.safetensors', '2024-01-01')"""
        )
        self.ckpt_id = self.conn.execute("SELECT id FROM checkpoints WHERE run_id = ?", ("parent-run",)).fetchone()[0]

    def tearDown(self):
        self.conn.close()

    def test_save_and_get_branch(self):
        save_run_branch(
            self.conn,
            branch_id="br-1",
            campaign_id=self.campaign["id"],
            parent_run_id="parent-run",
            parent_checkpoint_id=self.ckpt_id,
            from_stage="D",
            branch_reason="lr_decay",
            branch_params_json='{"lr": 0.001}',
            delta_json='{"lr": 0.1}',
            status="planned",
        )
        b = get_branch_by_id(self.conn, "br-1")
        self.assertIsNotNone(b)
        self.assertEqual(b["branch_reason"], "lr_decay")
        self.assertEqual(b["status"], "planned")

    def test_branch_upsert(self):
        save_run_branch(
            self.conn, branch_id="br-2", campaign_id=self.campaign["id"],
            parent_run_id="parent-run", parent_checkpoint_id=self.ckpt_id,
            from_stage="D", branch_reason="lr_decay",
            branch_params_json='{"lr": 0.001}', delta_json='{"lr": 0.1}', status="planned",
        )
        save_run_branch(
            self.conn, branch_id="br-2", campaign_id=self.campaign["id"],
            parent_run_id="parent-run", parent_checkpoint_id=self.ckpt_id,
            from_stage="D", branch_reason="lr_decay",
            branch_params_json='{"lr": 0.0001}', delta_json='{"lr": 0.01}', status="planned",
        )
        b = get_branch_by_id(self.conn, "br-2")
        self.assertEqual(b["delta_json"], '{"lr": 0.01}')

    def test_list_branches_for_campaign(self):
        save_run_branch(
            self.conn, branch_id="br-a", campaign_id=self.campaign["id"],
            parent_run_id="parent-run", parent_checkpoint_id=self.ckpt_id,
            from_stage="D", branch_reason="lr_decay",
            branch_params_json='{}', delta_json='{}', status="planned",
        )
        save_run_branch(
            self.conn, branch_id="br-b", campaign_id=self.campaign["id"],
            parent_run_id="parent-run", parent_checkpoint_id=self.ckpt_id,
            from_stage="D", branch_reason="seed_recheck",
            branch_params_json='{}', delta_json='{}', status="planned",
        )
        rows = list_branches_for_campaign(self.conn, self.campaign["id"])
        self.assertEqual(len(rows), 2)

    def test_list_branches_for_checkpoint(self):
        save_run_branch(
            self.conn, branch_id="br-c", campaign_id=self.campaign["id"],
            parent_run_id="parent-run", parent_checkpoint_id=self.ckpt_id,
            from_stage="D", branch_reason="lr_decay",
            branch_params_json='{}', delta_json='{}', status="planned",
        )
        rows = list_branches_for_checkpoint(self.conn, self.ckpt_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["branch_reason"], "lr_decay")

    def test_bind_child_run(self):
        save_run_branch(
            self.conn, branch_id="br-d", campaign_id=self.campaign["id"],
            parent_run_id="parent-run", parent_checkpoint_id=self.ckpt_id,
            from_stage="D", branch_reason="lr_decay",
            branch_params_json='{}', delta_json='{}', status="planned",
        )
        create_run(self.conn, "child-run", {"sweep_tag": "child", "eval_level": 0}, is_benchmark=True)
        bind_branch_child_run(self.conn, branch_id="br-d", child_run_id="child-run", status="running")
        b = get_branch_by_id(self.conn, "br-d")
        self.assertEqual(b["child_run_id"], "child-run")
        self.assertEqual(b["status"], "running")

    def test_update_branch_status(self):
        save_run_branch(
            self.conn, branch_id="br-e", campaign_id=self.campaign["id"],
            parent_run_id="parent-run", parent_checkpoint_id=self.ckpt_id,
            from_stage="D", branch_reason="lr_decay",
            branch_params_json='{}', delta_json='{}', status="planned",
        )
        update_branch_status(self.conn, branch_id="br-e", status="completed",
                             result_summary_json='{"wr": 0.85}')
        b = get_branch_by_id(self.conn, "br-e")
        self.assertEqual(b["status"], "completed")
        self.assertIn("0.85", b["result_summary_json"])
        self.assertIsNotNone(b["finished_at"])

    def test_get_branch_tree(self):
        save_run_branch(
            self.conn, branch_id="br-f", campaign_id=self.campaign["id"],
            parent_run_id="parent-run", parent_checkpoint_id=self.ckpt_id,
            from_stage="D", branch_reason="lr_decay",
            branch_params_json='{}', delta_json='{}', status="planned",
        )
        tree = get_branch_tree(self.conn, self.campaign["id"])
        self.assertEqual(len(tree), 1)
        self.assertEqual(tree[0]["branch_reason"], "lr_decay")
        self.assertEqual(tree[0]["parent_tag"], "parent")

    def test_db_init_idempotent(self):
        conn2 = init_db(":memory:")
        rows = conn2.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='run_branches'").fetchall()
        self.assertEqual(len(rows), 1)
        conn2.close()


if __name__ == "__main__":
    unittest.main()
