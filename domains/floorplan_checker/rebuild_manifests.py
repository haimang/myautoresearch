#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SPLITS = ("train", "eval", "test")
DEFAULT_RATIOS = {"train": 0.86, "eval": 0.09, "test": 0.05}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild floorplan manifests with listingId-exclusive splits.")
    parser.add_argument("--dataset-dir", default=Path(__file__).resolve().parent / "dataset")
    parser.add_argument("--backup-name", default="backup-v24_1-pre-clean-split")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def choose_split(listing_id: str) -> str:
    digest = hashlib.sha1(listing_id.encode("utf-8")).hexdigest()[:12]
    value = int(digest, 16) / float(16**12)
    if value < DEFAULT_RATIOS["train"]:
        return "train"
    if value < DEFAULT_RATIOS["train"] + DEFAULT_RATIOS["eval"]:
        return "eval"
    return "test"


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    manifests_dir = dataset_dir / "manifests"
    backup_dir = manifests_dir / args.backup_name
    if not manifests_dir.is_dir():
        raise SystemExit(f"Manifest directory not found: {manifests_dir}")
    if backup_dir.exists() and not args.force:
        raise SystemExit(f"Backup already exists: {backup_dir}")
    backup_dir.mkdir(parents=True, exist_ok=True)

    for name in ("summary.json", "train.jsonl", "eval.jsonl", "test.jsonl"):
        src = manifests_dir / name
        if not src.exists():
            raise SystemExit(f"Missing manifest file: {src}")
        dst = backup_dir / name
        if not dst.exists():
            shutil.copy2(src, dst)

    source_records = defaultdict(list)
    original_split_counts = Counter()
    overlap_listing_ids: set[str] = set()
    listing_seen_splits: dict[str, set[str]] = defaultdict(set)

    for split in SPLITS:
        with open(manifests_dir / f"{split}.jsonl", "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                listing_id = str(record.get("listingId", record["output_path"]))
                source_records[listing_id].append(record)
                original_split_counts[split] += 1
                listing_seen_splits[listing_id].add(split)

    for listing_id, seen in listing_seen_splits.items():
        if len(seen) > 1:
            overlap_listing_ids.add(listing_id)

    rebuilt_records: dict[str, list[dict]] = {split: [] for split in SPLITS}
    rebuilt_counts = Counter()
    listing_assignment_counts = Counter()
    moved_records = 0

    for listing_id, records in source_records.items():
        target_split = choose_split(listing_id)
        listing_assignment_counts[target_split] += 1
        for record in records:
            current_output_path = str(record["output_path"])
            original_split = current_output_path.split("/", 1)[0] if "/" in current_output_path else "unknown"
            if original_split != target_split:
                moved_records += 1
            rebuilt_records[target_split].append(record)
            rebuilt_counts[target_split] += 1

    for split in SPLITS:
        with open(manifests_dir / f"{split}.jsonl", "w", encoding="utf-8") as handle:
            for record in rebuilt_records[split]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "package_root": str(dataset_dir),
        "package_type": "floorplan-unified-v1-clean-split",
        "self_contained": True,
        "copy_mode": "physical_copy",
        "split_counts": {split: rebuilt_counts[split] for split in SPLITS},
        "total_files": sum(rebuilt_counts.values()),
        "rebuilt_at": datetime.now(timezone.utc).isoformat(),
        "split_method": {
            "type": "stable_hash_listing_id",
            "field": "listingId",
            "ratios": DEFAULT_RATIOS,
            "note": "Physical image files were left in place; benchmark membership now follows manifest assignment.",
        },
        "repair_summary": {
            "backup_dir": str(backup_dir),
            "original_split_counts": {split: original_split_counts[split] for split in SPLITS},
            "listing_assignment_counts": {split: listing_assignment_counts[split] for split in SPLITS},
            "overlap_listing_count_before": len(overlap_listing_ids),
            "moved_record_count": moved_records,
        },
    }
    with open(manifests_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
