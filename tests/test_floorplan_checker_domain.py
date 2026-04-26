"""Smoke tests for the floorplan_checker domain and manifest policies."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.core.db import init_db
from framework.policies.acquisition_policy import load_acquisition_policy
from framework.policies.branch_policy import load_branch_policy
from framework.policies.selector_policy import load_selector_policy
from framework.policies.stage_policy import load_stage_policy
from framework.profiles.objective_profile import load_objective_profile
from framework.profiles.search_space import load_profile
from domains.floorplan_checker.dataset_contract import DatasetContractError, inspect_dataset_contract


class TestFloorplanCheckerPolicies(unittest.TestCase):
    def test_floorplan_manifest_files_load(self):
        manifest_dir = ROOT / "domains" / "floorplan_checker" / "manifest"
        search = load_profile(str(manifest_dir / "search_space.json"))
        objective = load_objective_profile(str(manifest_dir / "objective_profile.json"))
        stage = load_stage_policy(str(manifest_dir / "stage_policy.json"))
        branch = load_branch_policy(str(manifest_dir / "branch_policy.json"))
        selector = load_selector_policy(str(manifest_dir / "selector_policy.json"))
        acquisition = load_acquisition_policy(str(manifest_dir / "acquisition_policy.json"))

        self.assertEqual(search["domain"], "floorplan_checker")
        self.assertEqual(objective["domain"], "floorplan_checker")
        self.assertEqual(stage["domain"], "floorplan_checker")
        self.assertEqual(branch["domain"], "floorplan_checker")
        self.assertEqual(selector["domain"], "floorplan_checker")
        self.assertEqual(acquisition["domain"], "floorplan_checker")


class TestFloorplanCheckerTrain(unittest.TestCase):
    def _write_sample_dataset(self, dataset_dir: Path, *, leak_test_listing: bool = False) -> None:
        manifests = dataset_dir / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)

        split_counts = {"train": 2, "eval": 1, "test": 1}
        for split, count in split_counts.items():
            (dataset_dir / split / "b2_ba1_p1").mkdir(parents=True, exist_ok=True)
            for idx in range(count):
                image_rel = f"{split}/b2_ba1_p1/{split}-{idx}.png"
                image_path = dataset_dir / image_rel
                Image.new("RGB", (32, 32), color=(255, 255 - idx, 255 - idx)).save(image_path)
                listing_id = f"{split}-listing-{idx}"
                if leak_test_listing and split == "test":
                    listing_id = "eval-listing-0"
                record = {
                    "output_path": image_rel,
                    "bedroom_head": "2",
                    "bathroom_head": "1",
                    "parking_head": "1",
                    "listingId": listing_id,
                }
                with open(manifests / f"{split}.jsonl", "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
        summary = {split: {"count": count} for split, count in split_counts.items()}
        with open(manifests / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle)

    def test_dataset_contract_detects_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            self._write_sample_dataset(dataset_dir, leak_test_listing=True)

            with self.assertRaises(DatasetContractError):
                inspect_dataset_contract(str(dataset_dir), path_check_mode="full", path_check_limit=10).assert_valid()

    def test_dataset_check_cli_outputs_contract_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            self._write_sample_dataset(dataset_dir)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "domains" / "floorplan_checker" / "train.py"),
                    "--dataset-check",
                    "--dataset-check-mode",
                    "full",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT), "FLOORPLAN_DATASET_DIR": str(dataset_dir)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["split_counts"]["train"], 2)
            self.assertEqual(payload["missing_file_count"], 0)
            self.assertEqual(payload["leakage_count"], 0)

    def test_train_writes_run_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_dir = tmp_path / "dataset"
            self._write_sample_dataset(dataset_dir)
            db_path = tmp_path / "tracker.db"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "domains" / "floorplan_checker" / "train.py"),
                    "--db",
                    str(db_path),
                    "--time-budget",
                    "1",
                    "--candidate-json",
                    json.dumps(
                        {
                            "num_res_blocks": 2,
                            "num_filters": 8,
                            "learning_rate": 5e-4,
                            "batch_size": 1,
                            "image_resolution": 64,
                        }
                    ),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT), "FLOORPLAN_DATASET_DIR": str(dataset_dir)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

            conn = init_db(str(db_path))
            run = conn.execute(
                "SELECT id, status, total_steps, final_win_rate FROM runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            metrics = conn.execute(
                "SELECT metric_name FROM run_metrics WHERE run_id = ?",
                (run["id"],),
            ).fetchall()
            conn.close()

            self.assertEqual(run["status"], "completed")
            self.assertGreaterEqual(run["total_steps"], 1)
            metric_names = {row["metric_name"] for row in metrics}
            self.assertIn("val_acc_bedroom", metric_names)
            self.assertIn("val_acc_bathroom", metric_names)
            self.assertIn("val_acc_parking", metric_names)
            self.assertIn("val_acc_macro", metric_names)
            self.assertIn("test_acc_bedroom", metric_names)
            self.assertIn("dataset_leakage_count", metric_names)
            self.assertIn("samples_per_second", metric_names)
            self.assertIn("peak_memory_mb", metric_names)

    def test_workspace_run_id_does_not_collide_in_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_dir = tmp_path / "dataset"
            artifact_root = tmp_path / "artifacts"
            self._write_sample_dataset(dataset_dir)
            db_path = tmp_path / "tracker.db"

            env = {**os.environ, "PYTHONPATH": str(ROOT), "FLOORPLAN_DATASET_DIR": str(dataset_dir)}
            base_cmd = [
                sys.executable,
                str(ROOT / "domains" / "floorplan_checker" / "train.py"),
                "--db",
                str(db_path),
                "--time-budget",
                "1",
                "--run-id",
                "workspace-run",
                "--artifact-root",
                str(artifact_root),
                "--eval-level",
                "0",
                "--candidate-json",
                json.dumps(
                    {
                        "num_res_blocks": 2,
                        "num_filters": 8,
                        "learning_rate": 5e-4,
                        "batch_size": 1,
                    }
                ),
            ]

            first = subprocess.run(base_cmd, cwd=ROOT, env=env, capture_output=True, text=True)
            second = subprocess.run(base_cmd, cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)

            conn = init_db(str(db_path))
            rows = conn.execute("SELECT id, is_benchmark, eval_level FROM runs ORDER BY started_at").fetchall()
            conn.close()

            self.assertEqual(len(rows), 2)
            self.assertNotEqual(rows[0]["id"], rows[1]["id"])
            self.assertEqual(rows[0]["is_benchmark"], 1)
            self.assertEqual(rows[0]["eval_level"], 0)
