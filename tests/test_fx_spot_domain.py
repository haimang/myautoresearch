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
FX_SPOT = ROOT / "domains" / "fx_spot"
if str(FX_SPOT) not in sys.path:
    sys.path.insert(0, str(FX_SPOT))

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
            metric_names = {m["metric_name"] for m in metrics}
            self.assertIn("liquidity_breach_count", metric_names)
            self.assertIn("effective_leg_count", metric_names)
            self.assertIn("breach_margin_ratio", metric_names)
            self.assertGreaterEqual(len(metrics), 15)
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

    def test_floor_probe_can_create_infeasible_constraint_evidence(self):
        from mock_provider import MockQuoteProvider
        from portfolio import clone_floors, clone_portfolio
        from route_eval import evaluate_route
        from treasury_scenarios import apply_treasury_scenario

        candidate = apply_treasury_scenario({
            "treasury_scenario": "cn_exporter_core",
            "sell_currency": "USD",
            "buy_currency": "CNY",
            "route_template": "direct",
            "rebalance_fraction": 0.5,
            "sell_amount_mode": "floor_probe",
            "floor_buffer_target": -0.05,
            "max_legs": 2,
            "quote_scenario": "constraint_stress",
        })
        result = evaluate_route(
            candidate=candidate,
            portfolio=clone_portfolio(candidate["portfolio"]),
            floors=clone_floors(candidate["liquidity_floors"]),
            anchor_currency=candidate["anchor_currency"],
            provider=MockQuoteProvider(scenario="constraint_stress"),
        )
        self.assertEqual(result["metrics"]["liquidity_floor_ok"], 0.0)
        self.assertGreater(result["metrics"]["liquidity_breach_count"], 0.0)
        self.assertIn("liquidity_floor_breach", result["breach_reasons"])

    def test_degenerate_bridge_can_be_rejected(self):
        from quote_graph import route_for_candidate

        with self.assertRaises(ValueError):
            route_for_candidate(
                {
                    "sell_currency": "USD",
                    "buy_currency": "CNY",
                    "route_template": "via_usd",
                    "reject_degenerate_bridge": True,
                },
                anchor_currency="CNY",
            )


if __name__ == "__main__":
    unittest.main()
