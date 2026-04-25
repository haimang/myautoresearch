"""Tests for v22 dynamic sweep candidate payloads."""

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "framework" / "index.py"


class TestDynamicSweep(unittest.TestCase):
    def test_dynamic_axes_pass_candidate_json_to_train_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "tracker.db"
            payload_path = tmp_path / "candidate.json"
            fake_train = tmp_path / "fake_train.py"
            fake_train.write_text(
                textwrap.dedent(
                    f"""
                    import argparse
                    import json
                    import sys
                    from pathlib import Path
                    ROOT = Path(r"{ROOT}")
                    FRAMEWORK = ROOT / "framework"
                    if str(ROOT) not in sys.path:
                        sys.path.insert(0, str(ROOT))
                    from framework.core.db import create_run, finish_run, init_db

                    p = argparse.ArgumentParser()
                    p.add_argument("--db", required=True)
                    p.add_argument("--sweep-tag", required=True)
                    p.add_argument("--time-budget", type=int, required=True)
                    p.add_argument("--seed", type=int, required=True)
                    p.add_argument("--candidate-json", required=True)
                    p.add_argument("--campaign-id", default=None)
                    args = p.parse_args()
                    data = json.loads(args.candidate_json)
                    Path(r"{payload_path}").write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
                    conn = init_db(args.db)
                    run_id = "run-" + args.sweep_tag
                    create_run(conn, run_id, {{"sweep_tag": args.sweep_tag, "seed": args.seed, "time_budget": args.time_budget}})
                    finish_run(conn, run_id, {{"status": "completed", "wall_time_s": 1.0}})
                    conn.close()
                    """
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(INDEX), "sweep",
                    "--db", str(db_path),
                    "--train-script", str(fake_train),
                    "--time-budget", "1",
                    "--tag", "fx",
                    "--axis", "sell_currency=EUR",
                    "--axis", "rebalance_fraction=0.5",
                    "--seeds", "7",
                    "--no-auto-pareto",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["sell_currency"], "EUR")
            self.assertEqual(payload["rebalance_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()

