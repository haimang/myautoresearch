"""Tests for v20.2 promotion DB helpers (campaign_stages, promotion_decisions)."""

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.core.db import (
    close_campaign_stage,
    get_campaign_stage,
    get_campaign_stages,
    init_db,
    save_campaign_stage,
    save_promotion_decision,
)


class TestPromotionDB(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")
        # Seed a campaign
        self.conn.execute(
            """INSERT INTO search_spaces (id, domain, name, version, profile_hash, profile_json, created_at)
               VALUES ('ss1', 'gomoku', 'cold-start-core', '1.0', 'abcd', '{}', '2024-01-01T00:00:00')"""
        )
        self.conn.execute(
            """INSERT INTO campaigns (id, name, domain, search_space_id, train_script, protocol_json, status, created_at)
               VALUES ('camp1', 'test-campaign', 'gomoku', 'ss1', 'train.py', '{}', 'open', '2024-01-01T00:00:00')"""
        )
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    def test_save_and_get_campaign_stage(self):
        save_campaign_stage(
            self.conn,
            campaign_id="camp1",
            stage="A",
            policy_json='{"time_budget": 60}',
            budget_json='{"time_budget": 60}',
            seed_target=1,
            status="open",
        )
        st = get_campaign_stage(self.conn, "camp1", "A")
        self.assertIsNotNone(st)
        self.assertEqual(st["stage"], "A")
        self.assertEqual(st["status"], "open")
        self.assertEqual(st["seed_target"], 1)

    def test_get_campaign_stages_ordered(self):
        save_campaign_stage(self.conn, campaign_id="camp1", stage="B", policy_json="{}", budget_json="{}", seed_target=2, status="open")
        save_campaign_stage(self.conn, campaign_id="camp1", stage="A", policy_json="{}", budget_json="{}", seed_target=1, status="open")
        stages = get_campaign_stages(self.conn, "camp1")
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0]["stage"], "A")
        self.assertEqual(stages[1]["stage"], "B")

    def test_close_campaign_stage(self):
        save_campaign_stage(self.conn, campaign_id="camp1", stage="A", policy_json="{}", budget_json="{}", seed_target=1, status="open")
        close_campaign_stage(self.conn, "camp1", "A")
        st = get_campaign_stage(self.conn, "camp1", "A")
        self.assertEqual(st["status"], "closed")
        self.assertIsNotNone(st["closed_at"])

    def test_save_and_query_promotion_decision(self):
        save_promotion_decision(
            self.conn,
            campaign_id="camp1",
            from_stage="A",
            to_stage="B",
            candidate_key="ck1",
            axis_values={"lr": 0.01},
            aggregated_metrics={"mean_wr": 0.85},
            seed_count=2,
            decision="promote",
            decision_rank=1,
            reason="top-1",
        )
        rows = self.conn.execute(
            "SELECT * FROM promotion_decisions WHERE campaign_id = ?",
            ("camp1",),
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision"], "promote")
        self.assertEqual(rows[0]["candidate_key"], "ck1")

    def test_promotion_decision_upsert(self):
        save_promotion_decision(
            self.conn,
            campaign_id="camp1",
            from_stage="A",
            to_stage="B",
            candidate_key="ck1",
            axis_values={"lr": 0.01},
            aggregated_metrics={"mean_wr": 0.80},
            seed_count=1,
            decision="hold",
            decision_rank=1,
            reason="initial",
        )
        save_promotion_decision(
            self.conn,
            campaign_id="camp1",
            from_stage="A",
            to_stage="B",
            candidate_key="ck1",
            axis_values={"lr": 0.01},
            aggregated_metrics={"mean_wr": 0.85},
            seed_count=2,
            decision="promote",
            decision_rank=1,
            reason="updated",
        )
        rows = self.conn.execute(
            "SELECT * FROM promotion_decisions WHERE campaign_id = ?",
            ("camp1",),
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision"], "promote")
        self.assertEqual(rows[0]["reason"], "updated")


if __name__ == "__main__":
    unittest.main()
