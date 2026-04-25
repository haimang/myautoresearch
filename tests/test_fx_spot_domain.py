"""Smoke tests for the v22 fx_spot domain."""

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

from core.db import get_or_create_campaign, init_db, save_objective_profile, save_search_space
from objective_profile import load_objective_profile
from search_space import load_profile


class TestFxSpotDomain(unittest.TestCase):
    def test_mock_train_writes_metrics_and_quote_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tracker.db")
            conn = init_db(db_path)
            search = load_profile(str(ROOT / "domains" / "fx_spot" / "search_space.json"))
            objective = load_objective_profile(str(ROOT / "domains" / "fx_spot" / "objective_profile.json"))
            space_id = save_search_space(conn, search)
            objective_id = save_objective_profile(conn, objective)
            campaign = get_or_create_campaign(
                conn,
                name="fx-smoke",
                domain="fx_spot",
                train_script=str(ROOT / "domains" / "fx_spot" / "train.py"),
                search_space_id=space_id,
                protocol={"eval_level": None, "eval_opponent": None, "is_benchmark": False},
                objective_profile_id=objective_id,
            )
            conn.close()

            candidate = {
                "sell_currency": "EUR",
                "buy_currency": "CNY",
                "route_template": "direct",
                "rebalance_fraction": 0.5,
                "max_legs": 2,
                "quote_scenario": "base",
            }
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "domains" / "fx_spot" / "train.py"),
                    "--db", db_path,
                    "--campaign-id", campaign["id"],
                    "--sweep-tag", "fx-smoke",
                    "--candidate-json", json.dumps(candidate),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            conn = init_db(db_path)
            run = conn.execute("SELECT id, status FROM runs WHERE sweep_tag = 'fx-smoke'").fetchone()
            metrics = conn.execute("SELECT metric_name FROM run_metrics WHERE run_id = ?", (run["id"],)).fetchall()
            windows = conn.execute("SELECT COUNT(*) AS n FROM quote_windows").fetchone()
            quotes = conn.execute("SELECT COUNT(*) AS n FROM fx_quotes").fetchone()
            legs = conn.execute("SELECT COUNT(*) AS n FROM fx_route_legs WHERE run_id = ?", (run["id"],)).fetchone()
            conn.close()
            self.assertEqual(run["status"], "completed")
            self.assertEqual(len(metrics), 9)
            self.assertEqual(windows["n"], 1)
            self.assertEqual(quotes["n"], 1)
            self.assertEqual(legs["n"], 1)

    def test_treasury_scenario_materializes_portfolio(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tracker.db")
            conn = init_db(db_path)
            search = load_profile(str(ROOT / "domains" / "fx_spot" / "search_space.json"))
            objective = load_objective_profile(str(ROOT / "domains" / "fx_spot" / "objective_profile.json"))
            space_id = save_search_space(conn, search)
            objective_id = save_objective_profile(conn, objective)
            campaign = get_or_create_campaign(
                conn,
                name="fx-scenario",
                domain="fx_spot",
                train_script=str(ROOT / "domains" / "fx_spot" / "train.py"),
                search_space_id=space_id,
                protocol={"eval_level": None, "eval_opponent": None, "is_benchmark": False},
                objective_profile_id=objective_id,
            )
            conn.close()

            candidate = {
                "treasury_scenario": "asia_procurement_hub",
                "sell_currency": "JPY",
                "buy_currency": "USD",
                "route_template": "via_hkd",
                "rebalance_fraction": 0.25,
                "max_legs": 2,
                "quote_scenario": "asia_hub",
            }
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "domains" / "fx_spot" / "train.py"),
                    "--db", db_path,
                    "--campaign-id", campaign["id"],
                    "--sweep-tag", "fx-scenario",
                    "--candidate-json", json.dumps(candidate),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            conn = init_db(db_path)
            window = conn.execute("SELECT portfolio_snapshot_json, anchor_currency FROM quote_windows").fetchone()
            quotes = conn.execute("SELECT COUNT(*) AS n FROM fx_quotes").fetchone()
            conn.close()
            self.assertEqual(window["anchor_currency"], "USD")
            self.assertIn("HKD", json.loads(window["portfolio_snapshot_json"]))
            self.assertEqual(quotes["n"], 2)


if __name__ == "__main__":
    unittest.main()
