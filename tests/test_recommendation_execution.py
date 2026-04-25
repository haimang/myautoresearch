"""Tests for v21.1 recommendation execution via sweep.py and branch.py."""

import json
import subprocess
import sys
import tempfile
import textwrap
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
    save_checkpoint,
    save_recommendation,
    save_recommendation_batch,
    save_search_space,
)
from search_space import load_profile

SWEEP = ROOT / "framework" / "sweep.py"
BRANCH = ROOT / "framework" / "branch.py"
PROFILE_PATH = ROOT / "domains" / "gomoku" / "search_space.json"


class TestRecommendationExecution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "tracker.db")
        self.fake_train = Path(self.tmp.name) / "fake_train.py"
        self.fake_train.write_text(
            textwrap.dedent(
                f"""
                import argparse
                import sys
                from pathlib import Path
                ROOT = Path(r"{ROOT}")
                FRAMEWORK = ROOT / "framework"
                if str(FRAMEWORK) not in sys.path:
                    sys.path.insert(0, str(FRAMEWORK))
                from core.db import create_run, finish_run, init_db

                p = argparse.ArgumentParser()
                p.add_argument("--db", required=True)
                p.add_argument("--sweep-tag", required=True)
                p.add_argument("--time-budget", type=int, default=60)
                p.add_argument("--seed", type=int, default=None)
                p.add_argument("--num-blocks", type=int, default=6)
                p.add_argument("--num-filters", type=int, default=64)
                p.add_argument("--learning-rate", type=float, default=0.001)
                p.add_argument("--steps-per-cycle", type=int, default=30)
                p.add_argument("--buffer-size", type=int, default=100000)
                p.add_argument("--eval-level", type=int, default=0)
                p.add_argument("--mcts-sims", type=int, default=400)
                p.add_argument("--resume", type=str, default=None)
                p.add_argument("--resume-checkpoint-tag", type=str, default=None)
                args = p.parse_args()

                conn = init_db(args.db)
                run_id = f"run-{{args.sweep_tag}}"
                create_run(conn, run_id, {{
                    "sweep_tag": args.sweep_tag,
                    "eval_level": args.eval_level,
                    "seed": args.seed,
                    "learning_rate": args.learning_rate,
                    "num_res_blocks": args.num_blocks,
                    "num_filters": args.num_filters,
                    "train_steps_per_cycle": args.steps_per_cycle,
                    "replay_buffer_size": args.buffer_size,
                    "mcts_simulations": args.mcts_sims,
                    "time_budget": args.time_budget,
                }}, is_benchmark=True)
                wr = 0.93 if args.resume else 0.91
                finish_run(conn, run_id, {{
                    "status": "completed",
                    "final_win_rate": wr,
                    "wall_time_s": float(args.time_budget),
                    "num_params": 123456,
                    "total_games": 500,
                }})
                print(f"Win rate: {{wr:.1%}}")
                """
            ),
            encoding="utf-8",
        )

        conn = init_db(self.db_path)
        profile = load_profile(str(PROFILE_PATH))
        space_id = save_search_space(conn, profile)
        self.campaign = get_or_create_campaign(
            conn,
            name="exec-test",
            domain="gomoku",
            train_script=str(self.fake_train),
            search_space_id=space_id,
            protocol={"eval_level": 0},
        )
        create_run(conn, "run-parent", {
            "sweep_tag": "parent",
            "eval_level": 0,
            "seed": 42,
            "learning_rate": 0.001,
            "num_res_blocks": 6,
            "num_filters": 64,
            "train_steps_per_cycle": 30,
            "replay_buffer_size": 100000,
            "mcts_simulations": 400,
            "time_budget": 60,
        }, is_benchmark=True)
        finish_run(conn, "run-parent", {
            "status": "completed",
            "final_win_rate": 0.88,
            "wall_time_s": 60.0,
            "num_params": 120000,
            "total_games": 500,
        })
        conn.execute(
            """INSERT INTO campaign_runs
               (campaign_id, run_id, stage, sweep_tag, seed, axis_values_json, status, created_at, candidate_key)
               VALUES (?, 'run-parent', 'C', 'parent', 42, ?, 'linked', '2024-01-01', ?)""",
            (
                self.campaign["id"],
                '{"num_blocks":6,"num_filters":64,"learning_rate":0.001,"steps_per_cycle":30,"buffer_size":100000,"mcts_simulations":400}',
                '{"buffer_size":100000,"learning_rate":0.001,"mcts_simulations":400,"num_blocks":6,"num_filters":64,"steps_per_cycle":30}',
            ),
        )
        ckpt_id = save_checkpoint(conn, "run-parent", {
            "tag": "final",
            "cycle": 10,
            "step": 100,
            "loss": 0.5,
            "win_rate": 0.88,
            "eval_level": 0,
            "eval_games": 100,
            "model_path": "parent.safetensors",
        })
        save_recommendation_batch(
            conn,
            batch_id="batch-exec",
            campaign_id=self.campaign["id"],
            selector_name="sel",
            selector_version="1.0",
            selector_hash="hash",
        )
        save_recommendation(
            conn,
            recommendation_id="rec-point",
            batch_id="batch-exec",
            candidate_type="new_point",
            candidate_key='{"buffer_size":100000,"learning_rate":0.0007,"num_blocks":6,"num_filters":64,"steps_per_cycle":30}',
            rank=1,
            score_total=1.0,
            score_breakdown_json='{}',
            rationale_json='{}',
            axis_values_json='{"num_blocks":6,"num_filters":64,"learning_rate":0.0007,"steps_per_cycle":30,"buffer_size":100000}',
            status="accepted",
        )
        save_recommendation(
            conn,
            recommendation_id="rec-branch",
            batch_id="batch-exec",
            candidate_type="continue_branch",
            candidate_key='{"buffer_size":100000,"learning_rate":0.001,"mcts_simulations":400,"num_blocks":6,"num_filters":64,"steps_per_cycle":30}',
            rank=2,
            score_total=0.9,
            score_breakdown_json='{}',
            rationale_json='{}',
            branch_reason="lr_decay",
            delta_json='{"learning_rate":0.1}',
            parent_run_id="run-parent",
            parent_checkpoint_id=ckpt_id,
            status="accepted",
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_sweep_executes_accepted_point_recommendation(self):
        proc = self._run(
            str(SWEEP),
            "--db", self.db_path,
            "--execute-recommendation", "rec-point",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        conn = init_db(self.db_path)
        rec = conn.execute("SELECT status FROM recommendations WHERE id = 'rec-point'").fetchone()
        outcomes = conn.execute("SELECT * FROM recommendation_outcomes WHERE recommendation_id = 'rec-point'").fetchall()
        linked = conn.execute(
            """SELECT COUNT(*) AS n FROM campaign_runs
               WHERE campaign_id = ? AND run_id LIKE 'run-exec-test_rec_%'""",
            (self.campaign["id"],),
        ).fetchone()
        conn.close()
        self.assertEqual(rec["status"], "executed")
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(linked["n"], 1)

    def test_branch_executes_accepted_branch_recommendation(self):
        proc = self._run(
            str(BRANCH),
            "--db", self.db_path,
            "--execute-recommendation", "rec-branch",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        conn = init_db(self.db_path)
        rec = conn.execute("SELECT status FROM recommendations WHERE id = 'rec-branch'").fetchone()
        outcomes = conn.execute("SELECT * FROM recommendation_outcomes WHERE recommendation_id = 'rec-branch'").fetchall()
        branches = conn.execute("SELECT * FROM run_branches").fetchall()
        stage_d = conn.execute(
            """SELECT COUNT(*) AS n FROM campaign_runs
               WHERE campaign_id = ? AND stage = 'D'""",
            (self.campaign["id"],),
        ).fetchone()
        conn.close()
        self.assertEqual(rec["status"], "executed")
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(len(branches), 1)
        self.assertEqual(stage_d["n"], 1)


if __name__ == "__main__":
    unittest.main()
