"""Tests for v22 run_metrics-backed Pareto analysis."""

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

from framework.core.db import create_run, finish_run, init_db, save_run_metrics


class TestRunMetricsPareto(unittest.TestCase):
    def test_analyze_pareto_uses_objective_profile_and_run_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "tracker.db")
            conn = init_db(db_path)
            for run_id, preservation, spread, floor_ok in [
                ("run-a", 0.9990, 8.0, 1.0),
                ("run-b", 0.9980, 6.0, 1.0),
                ("run-c", 1.0010, 20.0, 0.0),
            ]:
                create_run(conn, run_id, {"sweep_tag": run_id})
                finish_run(conn, run_id, {"status": "completed", "wall_time_s": 1.0})
                save_run_metrics(conn, run_id, [
                    {"metric_name": "liquidity_floor_ok", "metric_value": floor_ok, "metric_role": "constraint"},
                    {"metric_name": "liquidity_headroom_ratio", "metric_value": 0.1, "direction": "maximize"},
                    {"metric_name": "preservation_ratio", "metric_value": preservation, "direction": "maximize"},
                    {"metric_name": "spot_uplift_bps", "metric_value": (preservation - 1.0) * 10000, "direction": "maximize"},
                    {"metric_name": "quote_validity_remaining_s", "metric_value": 1200, "direction": "maximize"},
                    {"metric_name": "embedded_spread_bps", "metric_value": spread, "direction": "minimize"},
                    {"metric_name": "route_leg_count", "metric_value": 1, "direction": "minimize"},
                    {"metric_name": "settlement_lag_s", "metric_value": 60, "direction": "minimize"},
                    {"metric_name": "locked_funds_ratio", "metric_value": 0.03, "direction": "minimize"},
                ])
            conn.close()

            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "framework" / "index.py"), "analyze",
                    "--db", db_path,
                    "--pareto",
                    "--metric-source", "run_metrics",
                    "--objective-profile", str(ROOT / "domains" / "spot_trader" / "manifest" / "objective_profile.json"),
                    "--format", "json",
                    "--knee",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["metric_source"], "run_metrics")
            self.assertEqual(len(payload["infeasible"]), 1)
            self.assertEqual(payload["infeasible"][0]["run_full"], "run-c")
            self.assertTrue(payload["pareto_front"])
            self.assertIsNotNone(payload["knee"])


if __name__ == "__main__":
    unittest.main()

