#!/usr/bin/env python3
"""Multi-head ResNet Floorplan-Checker training entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(1, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from data_loader import FloorplanDataset
from dataset_contract import DatasetContractError, inspect_dataset_contract
from metrics import metric_rows
from models import FloorplanNet
from utils.memory_monitor import clear_cache, get_peak_memory_mb

from framework.core.db import (
    DB_PATH,
    collect_hardware_info,
    create_run,
    finish_run,
    init_db,
    save_run_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", type=str, default="{}")
    parser.add_argument("--campaign-id", type=str)
    parser.add_argument("--time-budget", type=int, default=300)
    parser.add_argument("--sweep-tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", type=str, default=DB_PATH)
    parser.add_argument("--run-id", type=str, default=str(uuid.uuid4()))
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--eval-level", type=int, default=None)
    parser.add_argument("--eval-opponent", type=str, default=None)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--dataset-check", action="store_true")
    parser.add_argument("--dataset-check-mode", choices=("none", "sample", "full"), default="sample")
    parser.add_argument("--dataset-enforcement", choices=("warn", "strict"), default="warn")
    parser.add_argument("--dataset-path-check-limit", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    # Legacy flags injected by sweep.py
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--num-filters", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--steps-per-cycle", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    return parser.parse_args()


def _candidate_from_args(args: argparse.Namespace) -> dict:
    if not args.candidate_json:
        return {}
    if os.path.exists(args.candidate_json):
        with open(args.candidate_json, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(args.candidate_json)


def _classification_loss(logits: mx.array, targets: mx.array) -> mx.array:
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def _loss_components_from_logits(
    logits_bed: mx.array,
    logits_bath: mx.array,
    logits_park: mx.array,
    y_bed: mx.array,
    y_bath: mx.array,
    y_park: mx.array,
    weights: dict[str, float],
) -> tuple[mx.array, dict[str, mx.array]]:
    losses = {
        "bedroom": _classification_loss(logits_bed, y_bed),
        "bathroom": _classification_loss(logits_bath, y_bath),
        "parking": _classification_loss(logits_park, y_park),
    }
    total = (
        weights["bed"] * losses["bedroom"]
        + weights["bath"] * losses["bathroom"]
        + weights["park"] * losses["parking"]
    )
    return total, losses


def loss_fn(
    model: FloorplanNet,
    x: mx.array,
    y_bed: mx.array,
    y_bath: mx.array,
    y_park: mx.array,
    weights: dict[str, float],
) -> mx.array:
    logits_bed, logits_bath, logits_park = model(x)
    total, _ = _loss_components_from_logits(logits_bed, logits_bath, logits_park, y_bed, y_bath, y_park, weights)
    return total


def count_parameters(model: FloorplanNet) -> int:
    leaves = nn.utils.tree_flatten(model.parameters())
    return sum(value.size for _, value in leaves)


def evaluate_split(
    model: FloorplanNet,
    dataset: FloorplanDataset,
    *,
    weights: dict[str, float],
    prefix: str,
) -> dict[str, float]:
    model.eval()
    total_samples = 0
    inference_time_s = 0.0
    correct = {"bedroom": 0, "bathroom": 0, "parking": 0}
    loss_sums = {"bedroom": 0.0, "bathroom": 0.0, "parking": 0.0, "total": 0.0}

    for x, y_bed, y_bath, y_park in dataset.get_batches():
        batch_size = int(y_bed.shape[0])
        batch_started = time.perf_counter()
        logits_bed, logits_bath, logits_park = model(x)
        mx.eval(logits_bed, logits_bath, logits_park)
        inference_time_s += time.perf_counter() - batch_started

        total_loss, loss_parts = _loss_components_from_logits(
            logits_bed, logits_bath, logits_park, y_bed, y_bath, y_park, weights
        )
        loss_sums["bedroom"] += loss_parts["bedroom"].item() * batch_size
        loss_sums["bathroom"] += loss_parts["bathroom"].item() * batch_size
        loss_sums["parking"] += loss_parts["parking"].item() * batch_size
        loss_sums["total"] += total_loss.item() * batch_size

        correct["bedroom"] += int(mx.sum(mx.argmax(logits_bed, axis=1) == y_bed).item())
        correct["bathroom"] += int(mx.sum(mx.argmax(logits_bath, axis=1) == y_bath).item())
        correct["parking"] += int(mx.sum(mx.argmax(logits_park, axis=1) == y_park).item())
        total_samples += batch_size

    if total_samples == 0:
        return {
            f"{prefix}_sample_count": 0.0,
            f"{prefix}_acc_bedroom": 0.0,
            f"{prefix}_acc_bathroom": 0.0,
            f"{prefix}_acc_parking": 0.0,
            f"{prefix}_acc_macro": 0.0,
            f"{prefix}_acc_min_head": 0.0,
            f"{prefix}_head_acc_gap": 0.0,
            f"{prefix}_loss_bedroom": 0.0,
            f"{prefix}_loss_bathroom": 0.0,
            f"{prefix}_loss_parking": 0.0,
            f"{prefix}_loss_total": 0.0,
            f"{prefix}_inference_latency_ms": 0.0,
        }

    acc_bedroom = correct["bedroom"] / total_samples
    acc_bathroom = correct["bathroom"] / total_samples
    acc_parking = correct["parking"] / total_samples
    acc_values = [acc_bedroom, acc_bathroom, acc_parking]

    return {
        f"{prefix}_sample_count": float(total_samples),
        f"{prefix}_acc_bedroom": acc_bedroom,
        f"{prefix}_acc_bathroom": acc_bathroom,
        f"{prefix}_acc_parking": acc_parking,
        f"{prefix}_acc_macro": sum(acc_values) / len(acc_values),
        f"{prefix}_acc_min_head": min(acc_values),
        f"{prefix}_head_acc_gap": max(acc_values) - min(acc_values),
        f"{prefix}_loss_bedroom": loss_sums["bedroom"] / total_samples,
        f"{prefix}_loss_bathroom": loss_sums["bathroom"] / total_samples,
        f"{prefix}_loss_parking": loss_sums["parking"] / total_samples,
        f"{prefix}_loss_total": loss_sums["total"] / total_samples,
        f"{prefix}_inference_latency_ms": (inference_time_s / total_samples) * 1000.0,
    }


def _finish_failed_run(conn, run_id: str, start_time: float, num_params: int | None = None) -> None:
    finish_run(
        conn,
        run_id,
        {
            "status": "failed",
            "total_cycles": 1,
            "total_games": 0,
            "total_steps": 0,
            "wall_time_s": time.time() - start_time,
            "final_win_rate": 0.0,
            "num_params": num_params,
            "peak_memory_mb": get_peak_memory_mb(),
        },
    )


def _emit_contract_warning(report) -> None:
    problems: list[str] = []
    if report.missing_field_count:
        problems.append(f"missing_fields={report.missing_field_count}")
    if report.missing_file_count:
        problems.append(f"missing_files={report.missing_file_count}")
    if report.leakage_count:
        problems.append(f"leakage={report.leakage_count}")
    if report.summary_mismatch_count:
        problems.append(f"summary_mismatches={report.summary_mismatch_count}")
    if problems:
        print(f"Dataset contract warning: {', '.join(problems)}")


def _resolve_db_run_id(args: argparse.Namespace) -> str:
    if args.artifact_root or args.campaign_id:
        return f"{args.run_id}-{uuid.uuid4().hex[:8]}"
    return args.run_id


def main() -> None:
    args = parse_args()
    start_time = time.time()
    candidate = _candidate_from_args(args)
    if args.num_blocks is not None:
        candidate["num_res_blocks"] = args.num_blocks
    if args.num_filters is not None:
        candidate["num_filters"] = args.num_filters
    if args.learning_rate is not None:
        candidate["learning_rate"] = args.learning_rate

    num_blocks = candidate.get("num_res_blocks", 4)
    num_filters = candidate.get("num_filters", 32)
    lr = candidate.get("learning_rate", 1e-4)
    batch_size = candidate.get("batch_size", 16)
    image_resolution = candidate.get("image_resolution", 256)
    image_size = (image_resolution, image_resolution)
    weights = {
        "bed": candidate.get("loss_weight_bed", 1.0),
        "bath": candidate.get("loss_weight_bath", 1.0),
        "park": candidate.get("loss_weight_park", 1.0),
    }

    dataset_dir = os.environ.get("FLOORPLAN_DATASET_DIR", os.path.join(_THIS_DIR, "dataset"))
    contract_report = inspect_dataset_contract(
        dataset_dir,
        path_check_mode=args.dataset_check_mode,
        path_check_limit=args.dataset_path_check_limit,
    )
    if args.dataset_check:
        print(json.dumps(contract_report.to_dict(), ensure_ascii=False, indent=2))
        if args.dataset_enforcement == "strict":
            contract_report.assert_valid()
        return
    if args.dataset_enforcement == "strict":
        contract_report.assert_valid()
    else:
        _emit_contract_warning(contract_report)

    run_id = _resolve_db_run_id(args)
    conn = init_db(args.db)
    artifact_dir = None
    campaign_bucket = args.campaign_id or "standalone"
    if args.artifact_root:
        artifact_dir = os.path.join(args.artifact_root, "runs", campaign_bucket, run_id)
        os.makedirs(artifact_dir, exist_ok=True)
    is_benchmark = args.eval_opponent is None and args.eval_level is not None

    create_run(
        conn,
        run_id,
        {
            "sweep_tag": args.sweep_tag,
            "seed": args.seed,
            "time_budget": args.time_budget,
            "artifact_dir": artifact_dir,
            "num_res_blocks": num_blocks,
            "num_filters": num_filters,
            "learning_rate": lr,
            "batch_size": batch_size,
            "image_resolution": image_resolution,
            "loader_workers": args.loader_workers,
            "eval_level": args.eval_level,
        },
        hardware=collect_hardware_info(),
        output_dir=artifact_dir,
        is_benchmark=is_benchmark,
        eval_opponent=args.eval_opponent,
    )

    from framework.core.db import link_run_to_campaign_v20

    if args.campaign_id:
        link_run_to_campaign_v20(
            conn,
            campaign_id=args.campaign_id,
            run_id=run_id,
            stage=None,
            sweep_tag=args.sweep_tag,
            seed=args.seed,
            axis_values={k: v for k, v in candidate.items() if not k.startswith("_")},
        )

    max_train_samples = args.max_train_samples
    max_eval_samples = args.max_eval_samples
    max_test_samples = args.max_test_samples
    if args.time_budget < 600:
        max_train_samples = max_train_samples or batch_size * 20
        max_eval_samples = max_eval_samples or batch_size * 10
        max_test_samples = max_test_samples or batch_size * 10

    train_dataset = None
    eval_dataset = None
    test_dataset = None
    model = FloorplanNet(num_blocks, num_filters)
    mx.eval(model.parameters())
    num_params = count_parameters(model)
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    try:
        train_dataset = FloorplanDataset(
            dataset_dir,
            split="train",
            batch_size=batch_size,
            max_samples=max_train_samples,
            image_size=image_size,
            shuffle=True,
            num_workers=args.loader_workers,
        )
        eval_dataset = FloorplanDataset(
            dataset_dir,
            split="eval",
            batch_size=batch_size,
            max_samples=max_eval_samples,
            image_size=image_size,
            shuffle=False,
            num_workers=args.loader_workers,
        )
        test_dataset = FloorplanDataset(
            dataset_dir,
            split="test",
            batch_size=batch_size,
            max_samples=max_test_samples,
            image_size=image_size,
            shuffle=False,
            num_workers=args.loader_workers,
        )

        model.train()
        nan_loss_count = 0
        total_steps = 0
        train_sample_count = 0
        train_loss_sum = 0.0
        loader_wait_s = 0.0
        train_compute_s = 0.0
        print(f"Starting MLX training loop for run {run_id}...")

        batch_iter = iter(train_dataset.get_batches())
        while True:
            load_started = time.perf_counter()
            try:
                x, y_bed, y_bath, y_park = next(batch_iter)
            except StopIteration:
                break
            loader_wait_s += time.perf_counter() - load_started

            if time.time() - start_time > args.time_budget and total_steps > 0:
                print("Time budget exceeded, truncating early.")
                break

            compute_started = time.perf_counter()
            loss, grads = loss_and_grad_fn(model, x, y_bed, y_bath, y_park, weights)
            loss_value = float(loss.item())
            if mx.isnan(loss).item():
                raise RuntimeError("NaN loss detected during training.")

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            train_compute_s += time.perf_counter() - compute_started

            batch_samples = int(y_bed.shape[0])
            train_sample_count += batch_samples
            train_loss_sum += loss_value * batch_samples
            total_steps += 1

            if total_steps % 10 == 0:
                clear_cache()

        val_metrics = evaluate_split(model, eval_dataset, weights=weights, prefix="val")
        test_metrics = evaluate_split(model, test_dataset, weights=weights, prefix="test")
        wall_time_s = time.time() - start_time
        peak_memory_mb = get_peak_memory_mb()
        avg_acc = val_metrics["val_acc_macro"]

        metrics = {
            **contract_report.to_metrics(),
            **val_metrics,
            **test_metrics,
            "train_loss_total": (train_loss_sum / train_sample_count) if train_sample_count else 0.0,
            "train_sample_count": float(train_sample_count),
            "samples_per_second": (train_sample_count / wall_time_s) if wall_time_s > 0 else 0.0,
            "loader_wait_ratio": (
                loader_wait_s / (loader_wait_s + train_compute_s)
                if (loader_wait_s + train_compute_s) > 0
                else 0.0
            ),
            "train_step_time_ms": (train_compute_s / total_steps) * 1000.0 if total_steps else 0.0,
            "inference_latency_ms": val_metrics["val_inference_latency_ms"],
            "wall_time_s": wall_time_s,
            "peak_memory_mb": peak_memory_mb,
            "nan_loss_count": 0.0,
        }
        print(f"Completed! Metrics: {json.dumps(metrics)}")

        save_run_metrics(conn, run_id, metric_rows(metrics))
        finish_run(
            conn,
            run_id,
            {
                "status": "completed",
                "total_cycles": 1,
                "total_games": 0,
                "total_steps": total_steps,
                "wall_time_s": wall_time_s,
                "final_loss": metrics["train_loss_total"],
                "final_win_rate": avg_acc,
                "num_params": num_params,
                "peak_memory_mb": peak_memory_mb,
            },
        )

        if artifact_dir:
            summary_path = os.path.join(artifact_dir, "evaluation_summary.json")
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "run_id": run_id,
                        "workspace_run_id": args.run_id,
                        "candidate": candidate,
                        "dataset_contract": contract_report.to_dict(),
                        "metrics": metrics,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
    except DatasetContractError:
        _finish_failed_run(conn, run_id, start_time, num_params=num_params)
        raise
    except Exception:
        _finish_failed_run(conn, run_id, start_time, num_params=num_params)
        raise
    finally:
        if train_dataset is not None:
            train_dataset.close()
        if eval_dataset is not None:
            eval_dataset.close()
        if test_dataset is not None:
            test_dataset.close()
        conn.close()


if __name__ == "__main__":
    main()
