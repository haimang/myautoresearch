"""Tests for v21 recommendation ledger in framework/core/db.py."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.core.db import (
    create_run,
    finish_run,
    get_campaign,
    get_or_create_campaign,
    get_recommendation_by_id,
    get_latest_recommendation_batch,
    init_db,
    list_recommendation_batches,
    list_recommendations_for_batch,
    list_recommendation_outcomes,
    save_recommendation_batch,
    save_recommendation,
    save_recommendation_outcome,
    update_recommendation_status,
    save_search_space,
)
from framework.profiles.search_space import load_profile

PROFILE_PATH = ROOT / "domains" / "gomoku" / "search_space.json"


class TestRecommendationDB(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "tracker.db")
        self.conn = init_db(self.db_path)
        self.profile = load_profile(str(PROFILE_PATH))
        self.space_id = save_search_space(self.conn, self.profile)
        self.campaign = get_or_create_campaign(
            self.conn,
            name="rec-db-test",
            domain="gomoku",
            train_script="t.py",
            search_space_id=self.space_id,
            protocol={"eval_level": 0},
        )
        # Seed a run for outcome linkage
        create_run(self.conn, "run-1", {"sweep_tag": "r1", "eval_level": 0}, is_benchmark=True)
        finish_run(self.conn, "run-1", {
            "status": "completed", "final_win_rate": 0.75,
            "wall_time_s": 100.0, "num_params": 100000, "total_games": 500,
        })
        self.conn.commit()
        self.conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_and_get_batch(self):
        conn = init_db(self.db_path)
        save_recommendation_batch(
            conn,
            batch_id="batch-1",
            campaign_id=self.campaign["id"],
            selector_name="sel",
            selector_version="1.0",
            selector_hash="abc",
        )
        batches = list_recommendation_batches(conn, self.campaign["id"])
        conn.close()
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["selector_name"], "sel")

    def test_get_latest_batch(self):
        conn = init_db(self.db_path)
        save_recommendation_batch(conn, batch_id="batch-a", campaign_id=self.campaign["id"],
                                  selector_name="s", selector_version="1", selector_hash="h")
        save_recommendation_batch(conn, batch_id="batch-b", campaign_id=self.campaign["id"],
                                  selector_name="s", selector_version="1", selector_hash="h")
        latest = get_latest_recommendation_batch(conn, self.campaign["id"])
        conn.close()
        self.assertIsNotNone(latest)
        self.assertEqual(latest["id"], "batch-b")

    def test_save_and_get_recommendation(self):
        conn = init_db(self.db_path)
        save_recommendation_batch(conn, batch_id="batch-1", campaign_id=self.campaign["id"],
                                  selector_name="s", selector_version="1", selector_hash="h")
        save_recommendation(
            conn,
            recommendation_id="rec-1",
            batch_id="batch-1",
            candidate_type="new_point",
            candidate_key="ck1",
            rank=1,
            score_total=0.9,
            score_breakdown_json='{"a": 1}',
            rationale_json='{"r": "x"}',
        )
        rec = get_recommendation_by_id(conn, "rec-1")
        conn.close()
        self.assertIsNotNone(rec)
        self.assertEqual(rec["candidate_type"], "new_point")
        self.assertEqual(rec["status"], "planned")

    def test_list_recommendations_for_batch(self):
        conn = init_db(self.db_path)
        save_recommendation_batch(conn, batch_id="batch-1", campaign_id=self.campaign["id"],
                                  selector_name="s", selector_version="1", selector_hash="h")
        save_recommendation(conn, recommendation_id="rec-1", batch_id="batch-1",
                            candidate_type="new_point", candidate_key="ck1", rank=1,
                            score_total=0.9, score_breakdown_json='{}', rationale_json='{}')
        save_recommendation(conn, recommendation_id="rec-2", batch_id="batch-1",
                            candidate_type="seed_recheck", candidate_key="ck2", rank=2,
                            score_total=0.8, score_breakdown_json='{}', rationale_json='{}')
        recs = list_recommendations_for_batch(conn, "batch-1")
        conn.close()
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["rank"], 1)
        self.assertEqual(recs[1]["rank"], 2)

    def test_update_recommendation_status(self):
        conn = init_db(self.db_path)
        save_recommendation_batch(conn, batch_id="batch-1", campaign_id=self.campaign["id"],
                                  selector_name="s", selector_version="1", selector_hash="h")
        save_recommendation(conn, recommendation_id="rec-1", batch_id="batch-1",
                            candidate_type="new_point", candidate_key="ck1", rank=1,
                            score_total=0.9, score_breakdown_json='{}', rationale_json='{}')
        update_recommendation_status(conn, recommendation_id="rec-1", status="accepted")
        rec = get_recommendation_by_id(conn, "rec-1")
        conn.close()
        self.assertEqual(rec["status"], "accepted")

    def test_save_recommendation_outcome(self):
        conn = init_db(self.db_path)
        save_recommendation_batch(conn, batch_id="batch-1", campaign_id=self.campaign["id"],
                                  selector_name="s", selector_version="1", selector_hash="h")
        save_recommendation(conn, recommendation_id="rec-1", batch_id="batch-1",
                            candidate_type="new_point", candidate_key="ck1", rank=1,
                            score_total=0.9, score_breakdown_json='{}', rationale_json='{}')
        save_recommendation_outcome(
            conn,
            recommendation_id="rec-1",
            run_id="run-1",
            observed_metrics_json='{"wr": 0.8}',
            outcome_label="hit_front",
        )
        outcomes = list_recommendation_outcomes(conn, "rec-1")
        conn.close()
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["outcome_label"], "hit_front")

    def test_batch_upsert(self):
        conn = init_db(self.db_path)
        save_recommendation_batch(conn, batch_id="batch-1", campaign_id=self.campaign["id"],
                                  selector_name="s", selector_version="1", selector_hash="h")
        save_recommendation_batch(conn, batch_id="batch-1", campaign_id=self.campaign["id"],
                                  selector_name="s2", selector_version="2", selector_hash="h2")
        batches = list_recommendation_batches(conn, self.campaign["id"])
        conn.close()
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["selector_name"], "s2")

    def test_recommendation_upsert(self):
        conn = init_db(self.db_path)
        save_recommendation_batch(conn, batch_id="batch-1", campaign_id=self.campaign["id"],
                                  selector_name="s", selector_version="1", selector_hash="h")
        save_recommendation(conn, recommendation_id="rec-1", batch_id="batch-1",
                            candidate_type="new_point", candidate_key="ck1", rank=1,
                            score_total=0.9, score_breakdown_json='{}', rationale_json='{}')
        save_recommendation(conn, recommendation_id="rec-1", batch_id="batch-1",
                            candidate_type="new_point", candidate_key="ck1", rank=1,
                            score_total=1.0, score_breakdown_json='{}', rationale_json='{}')
        rec = get_recommendation_by_id(conn, "rec-1")
        conn.close()
        self.assertEqual(rec["score_total"], 1.0)

    def test_no_batches_returns_empty(self):
        conn = init_db(self.db_path)
        batches = list_recommendation_batches(conn, self.campaign["id"])
        conn.close()
        self.assertEqual(batches, [])

    def test_get_latest_batch_none(self):
        conn = init_db(self.db_path)
        latest = get_latest_recommendation_batch(conn, self.campaign["id"])
        conn.close()
        self.assertIsNone(latest)


if __name__ == "__main__":
    unittest.main()
