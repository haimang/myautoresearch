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
from models import FloorplanNet
from utils.memory_monitor import get_peak_memory_mb, clear_cache
from metrics import metric_rows

from framework.core.db import (
    DB_PATH,
    collect_hardware_info,
    create_run,
    finish_run,
    init_db,
    save_run_metrics,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-json", type=str, default="{}")
    p.add_argument("--campaign-id", type=str)
    p.add_argument("--time-budget", type=int, default=300)
    p.add_argument("--sweep-tag", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--db", type=str, default=DB_PATH)
    p.add_argument("--run-id", type=str, default=str(uuid.uuid4()))
    p.add_argument("--artifact-root", default=None)
    # Legacy flags injected by sweep.py
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--num-filters", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--steps-per-cycle", type=int, default=None)
    p.add_argument("--buffer-size", type=int, default=None)
    return p.parse_args()

def loss_fn(model, x, y_bed, y_bath, y_park, weights):
    logits_bed, logits_bath, logits_park = model(x)
    loss_bed = nn.losses.cross_entropy(logits_bed, y_bed, reduction="mean")
    loss_bath = nn.losses.cross_entropy(logits_bath, y_bath, reduction="mean")
    loss_park = nn.losses.cross_entropy(logits_park, y_park, reduction="mean")
    return weights['bed'] * loss_bed + weights['bath'] * loss_bath + weights['park'] * loss_park

def _candidate_from_args(args: argparse.Namespace) -> dict:
    if args.candidate_json:
        try:
            if os.path.exists(args.candidate_json):
                with open(args.candidate_json, 'r') as f:
                    return json.load(f)
            return json.loads(args.candidate_json)
        except Exception:
            return {}
    return {}

def main() -> None:
    args = parse_args()
    start_time = time.time()
    run_id = args.run_id
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
    weights = {
        'bed': candidate.get("loss_weight_bed", 1.0),
        'bath': candidate.get("loss_weight_bath", 1.0),
        'park': candidate.get("loss_weight_park", 1.0)
    }

    # If sweep runner sends candidate json, we shouldn't hardcode run_id from it unless necessary.
    # Framework passes run_id via arg. For independent runs within a workspace, we must use a unique ID.
    workspace_id = args.run_id
    run_id = str(uuid.uuid4())
    # Set up DB and Run with Framework Integration
    conn = init_db(args.db)
    artifact_dir = None
    if args.artifact_root:
        safe_campaign = args.campaign_id or "standalone"
        artifact_dir = os.path.join(args.artifact_root, "runs", run_id)
        os.makedirs(artifact_dir, exist_ok=True)
        
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
        },
        hardware=collect_hardware_info(),
        output_dir=artifact_dir,
        is_benchmark=False,
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
            axis_values={k: v for k, v in candidate.items() if not k.startswith("_")}
        )

    dataset_dir = os.path.join(_THIS_DIR, "dataset")
    max_train_samples = batch_size * 20 if args.time_budget < 600 else None
    max_eval_samples = batch_size * 10 if args.time_budget < 600 else None

    train_dataset = FloorplanDataset(dataset_dir, split="train", batch_size=batch_size, max_samples=max_train_samples)
    eval_dataset = FloorplanDataset(dataset_dir, split="eval", batch_size=batch_size, max_samples=max_eval_samples)

    model = FloorplanNet(num_blocks, num_filters)
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    nan_loss_count = 0
    total_steps = 0
    print("Starting MLX training loop...")
    
    try:
        for batch_idx, (x, y_bed, y_bath, y_park) in enumerate(train_dataset.get_batches()):
            if time.time() - start_time > args.time_budget:
                print("Time budget exceeded, truncating early.")
                break
                
            loss, grads = loss_and_grad_fn(model, x, y_bed, y_bath, y_park, weights)
            if mx.isnan(loss).item():
                print("NaN loss detected. Marking as constraint breach.")
                nan_loss_count += 1
                break
                
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total_steps += 1
            
            if batch_idx % 10 == 0:
                clear_cache()
                
    except Exception as e:
        print(f"Training crashed (OOM or IO Error): {e}")
        nan_loss_count += 1

    print("Evaluating...")
    val_acc_bed, val_acc_bath, val_acc_park = 0.0, 0.0, 0.0
    eval_batches = 0
    if nan_loss_count == 0:
        for x, y_bed, y_bath, y_park in eval_dataset.get_batches():
            logits_bed, logits_bath, logits_park = model(x)
            val_acc_bed += mx.mean(mx.argmax(logits_bed, axis=1) == y_bed).item()
            val_acc_bath += mx.mean(mx.argmax(logits_bath, axis=1) == y_bath).item()
            val_acc_park += mx.mean(mx.argmax(logits_park, axis=1) == y_park).item()
            eval_batches += 1
        if eval_batches > 0:
            val_acc_bed /= eval_batches
            val_acc_bath /= eval_batches
            val_acc_park /= eval_batches

    wall_time_s = time.time() - start_time
    peak_memory_mb = get_peak_memory_mb()
    
    metrics = {
        "val_acc_bedroom": val_acc_bed,
        "val_acc_bathroom": val_acc_bath,
        "val_acc_parking": val_acc_park,
        "wall_time_s": wall_time_s,
        "peak_memory_mb": peak_memory_mb,
        "nan_loss_count": nan_loss_count
    }
    print(f"Completed! Metrics: {json.dumps(metrics)}")
    
    try:
        # Use framework's generic metrics insertion
        save_run_metrics(conn, run_id, metric_rows(metrics))
        
        status = "failed" if nan_loss_count > 0 else "completed"
        # Average Accuracy acts as a legacy surrogate for win_rate if needed by old plots
        avg_acc = (val_acc_bed + val_acc_bath + val_acc_park) / 3 if nan_loss_count == 0 else 0.0
        
        # Complete the run lifecycle managed by autoresearch
        finish_run(
            conn,
            run_id,
            {
                "status": status,
                "total_cycles": 1,
                "total_games": 0,
                "total_steps": total_steps,
                "wall_time_s": wall_time_s,
                "final_win_rate": avg_acc,
            },
        )
    finally:
        conn.close()

if __name__ == "__main__":
    main()
