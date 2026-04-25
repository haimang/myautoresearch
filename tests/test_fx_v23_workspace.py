"""Tests for v23 fx_spot run-scoped workspace and BO smoke."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "framework" / "index.py"
BO = ROOT / "domains" / "fx_spot" / "bayesian_refine.py"


class TestFxV23Workspace(unittest.TestCase):
    def test_sweep_run_id_creates_fx_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(INDEX), "sweep",
                    "--train-script", str(ROOT / "domains" / "fx_spot" / "train.py"),
                    "--search-space", str(ROOT / "domains" / "fx_spot" / "search_space.json"),
                    "--objective-profile", str(ROOT / "domains" / "fx_spot" / "objective_profile.json"),
                    "--campaign", "fx-v23-workspace",
                    "--run-id", "fx-test-run",
                    "--output-root", str(output_root),
                    "--time-budget", "1",
                    "--axis", "treasury_scenario=cn_exporter_core",
                    "--axis", "sell_currency=USD",
                    "--axis", "buy_currency=CNY",
                    "--axis", "route_template=direct",
                    "--axis", "rebalance_fraction=0.15",
                    "--axis", "sell_amount_mode=surplus_fraction",
                    "--axis", "sell_amount_ratio=0.0",
                    "--axis", "floor_buffer_target=0.0",
                    "--axis", "max_legs=2",
                    "--axis", "provider=mock",
                    "--axis", "quote_scenario=base",
                    "--seeds", "1",
                    "--no-auto-pareto",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            workspace = output_root / "fx_spot" / "fx-test-run"
            self.assertTrue((workspace / "manifest.json").exists())
            self.assertTrue((workspace / "tracker.db").exists())
            self.assertTrue((workspace / "runs").exists())

    def test_bayesian_refine_smoke_outputs_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(BO),
                    "--run-id", "fx-bo-test",
                    "--output-root", str(output_root),
                    "--scenarios", "cn_exporter_core",
                    "--budget", "12",
                    "--seed-observations", "4",
                    "--batch-size", "4",
                    "--max-universe", "48",
                    "--seed", "3",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            workspace = output_root / "fx_spot" / "fx-bo-test"
            self.assertTrue((workspace / "benchmarks" / "benchmark_summary.csv").exists())
            self.assertTrue((workspace / "benchmarks" / "benchmark_summary.md").exists())
            self.assertTrue(any(workspace.glob("campaigns/*/pareto/overview.png")))


if __name__ == "__main__":
    unittest.main()
