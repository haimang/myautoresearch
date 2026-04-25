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
    find_run_by_sweep_tag,
    get_campaign,
    get_or_create_campaign,
    init_db,
    link_run_to_campaign,
    save_search_space,
)
from framework.profiles.search_space import load_profile


PROFILE_PATH = ROOT / "domains" / "gomoku" / "search_space.json"


class TestCampaignDB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "tracker.db")
        self.conn = init_db(self.db_path)
        self.profile = load_profile(str(PROFILE_PATH))
        self.protocol = {
            "eval_level": 0,
            "eval_opponent": None,
            "is_benchmark": True,
            "train_script": "domains/gomoku/train.py",
        }

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _create_run(self, sweep_tag="camp_b4_f32_sd42", seed=42):
        hyperparams = {
            "num_res_blocks": 4,
            "num_filters": 32,
            "learning_rate": 5e-4,
            "train_steps_per_cycle": 20,
            "replay_buffer_size": 50000,
            "time_budget": 30,
            "eval_level": 0,
            "sweep_tag": sweep_tag,
            "seed": seed,
        }
        run_id = f"run-{seed:02d}"
        create_run(self.conn, run_id, hyperparams, is_benchmark=True)
        return run_id

    def test_tables_exist_after_init(self):
        for table in ("campaigns", "campaign_runs", "search_spaces"):
            row = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            self.assertIsNotNone(row, table)

    def test_init_db_is_idempotent(self):
        conn2 = init_db(self.db_path)
        conn2.close()

    def test_save_search_space_deduplicates_by_hash(self):
        a = save_search_space(self.conn, self.profile)
        b = save_search_space(self.conn, self.profile)
        self.assertEqual(a, b)
        count = self.conn.execute("SELECT COUNT(*) AS n FROM search_spaces").fetchone()["n"]
        self.assertEqual(count, 1)

    def test_create_campaign_persists_search_space_link(self):
        space_id = save_search_space(self.conn, self.profile)
        campaign = get_or_create_campaign(
            self.conn,
            name="gomoku-smoke",
            domain="gomoku",
            train_script="domains/gomoku/train.py",
            search_space_id=space_id,
            protocol=self.protocol,
        )
        self.assertEqual(campaign["search_space_id"], space_id)

    def test_get_or_create_campaign_reuses_same_name_when_compatible(self):
        space_id = save_search_space(self.conn, self.profile)
        a = get_or_create_campaign(
            self.conn,
            name="gomoku-smoke",
            domain="gomoku",
            train_script="domains/gomoku/train.py",
            search_space_id=space_id,
            protocol=self.protocol,
        )
        b = get_or_create_campaign(
            self.conn,
            name="gomoku-smoke",
            domain="gomoku",
            train_script="domains/gomoku/train.py",
            search_space_id=space_id,
            protocol=self.protocol,
        )
        self.assertEqual(a["id"], b["id"])

    def test_campaign_runs_round_trip_axis_json(self):
        space_id = save_search_space(self.conn, self.profile)
        campaign = get_or_create_campaign(
            self.conn,
            name="gomoku-smoke",
            domain="gomoku",
            train_script="domains/gomoku/train.py",
            search_space_id=space_id,
            protocol=self.protocol,
        )
        run_id = self._create_run()
        link_run_to_campaign(
            self.conn,
            campaign_id=campaign["id"],
            run_id=run_id,
            stage=None,
            sweep_tag="camp_b4_f32_sd42",
            seed=42,
            axis_values={"num_blocks": 4, "num_filters": 32},
        )
        row = self.conn.execute("SELECT axis_values_json FROM campaign_runs").fetchone()
        self.assertEqual(row["axis_values_json"], "{\"num_blocks\":4,\"num_filters\":32}")

    def test_duplicate_campaign_run_is_upserted(self):
        space_id = save_search_space(self.conn, self.profile)
        campaign = get_or_create_campaign(
            self.conn,
            name="gomoku-smoke",
            domain="gomoku",
            train_script="domains/gomoku/train.py",
            search_space_id=space_id,
            protocol=self.protocol,
        )
        run_id = self._create_run()
        link_run_to_campaign(
            self.conn, campaign_id=campaign["id"], run_id=run_id, stage=None,
            sweep_tag="camp_b4_f32_sd42", seed=42, axis_values={"num_blocks": 4}, status="linked"
        )
        link_run_to_campaign(
            self.conn, campaign_id=campaign["id"], run_id=run_id, stage="A",
            sweep_tag="camp_b4_f32_sd42", seed=42, axis_values={"num_blocks": 4}, status="done"
        )
        row = self.conn.execute("SELECT COUNT(*) AS n, status, stage FROM campaign_runs").fetchone()
        self.assertEqual(row["n"], 1)
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["stage"], "A")

    def test_protocol_json_round_trip(self):
        space_id = save_search_space(self.conn, self.profile)
        campaign = get_or_create_campaign(
            self.conn,
            name="gomoku-smoke",
            domain="gomoku",
            train_script="domains/gomoku/train.py",
            search_space_id=space_id,
            protocol=self.protocol,
        )
        row = get_campaign(self.conn, campaign["id"])
        self.assertIn("\"eval_level\":0", row["protocol_json"])
        self.assertIn("\"train_script\":\"domains/gomoku/train.py\"", row["protocol_json"])

    def test_find_run_by_sweep_tag_returns_exact_match(self):
        run_id = self._create_run("camp_b6_f64_sd42", 42)
        self.assertEqual(find_run_by_sweep_tag(self.conn, "camp_b6_f64_sd42"), run_id)


if __name__ == "__main__":
    unittest.main()
