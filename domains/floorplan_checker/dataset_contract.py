from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import combinations

REQUIRED_SPLITS = ("train", "eval", "test")
REQUIRED_LABEL_FIELDS = ("bedroom_head", "bathroom_head", "parking_head")
REQUIRED_FIELDS = ("output_path", *REQUIRED_LABEL_FIELDS)


class DatasetContractError(RuntimeError):
    """Raised when the dataset layout or manifest contract is invalid."""


@dataclass(frozen=True)
class DatasetContractReport:
    dataset_dir: str
    split_counts: dict[str, int]
    checked_path_count: int
    missing_file_count: int
    missing_field_count: int
    leakage_count: int
    summary_mismatch_count: int
    path_check_mode: str
    path_check_limit: int | None
    leakage_pairs: dict[str, int]

    @property
    def total_samples(self) -> int:
        return sum(self.split_counts.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_dir": self.dataset_dir,
            "split_counts": dict(self.split_counts),
            "total_samples": self.total_samples,
            "checked_path_count": self.checked_path_count,
            "missing_file_count": self.missing_file_count,
            "missing_field_count": self.missing_field_count,
            "leakage_count": self.leakage_count,
            "summary_mismatch_count": self.summary_mismatch_count,
            "path_check_mode": self.path_check_mode,
            "path_check_limit": self.path_check_limit,
            "leakage_pairs": dict(self.leakage_pairs),
        }

    def to_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "dataset_total_samples": float(self.total_samples),
            "dataset_checked_path_count": float(self.checked_path_count),
            "dataset_missing_file_count": float(self.missing_file_count),
            "dataset_missing_field_count": float(self.missing_field_count),
            "dataset_leakage_count": float(self.leakage_count),
            "dataset_summary_mismatch_count": float(self.summary_mismatch_count),
        }
        for split, count in self.split_counts.items():
            metrics[f"dataset_{split}_manifest_count"] = float(count)
        for pair, count in self.leakage_pairs.items():
            metrics[f"dataset_leakage_{pair}_count"] = float(count)
        return metrics

    def assert_valid(self) -> None:
        problems: list[str] = []
        if self.missing_field_count:
            problems.append(f"missing fields: {self.missing_field_count}")
        if self.missing_file_count:
            problems.append(f"missing files: {self.missing_file_count}")
        if self.leakage_count:
            problems.append(f"split leakage: {self.leakage_count}")
        if self.summary_mismatch_count:
            problems.append(f"summary mismatches: {self.summary_mismatch_count}")
        if problems:
            raise DatasetContractError(", ".join(problems))


def _summary_count(summary: dict, split: str) -> int | None:
    value = summary.get(split)
    if isinstance(value, dict):
        count = value.get("count")
        if isinstance(count, int):
            return count
    split_counts = summary.get("split_counts")
    if isinstance(split_counts, dict):
        count = split_counts.get(split)
        if isinstance(count, int):
            return count
    return None


def inspect_dataset_contract(
    dataset_dir: str,
    *,
    path_check_mode: str = "sample",
    path_check_limit: int = 256,
) -> DatasetContractReport:
    if path_check_mode not in {"none", "sample", "full"}:
        raise ValueError(f"Unsupported path_check_mode: {path_check_mode}")
    if path_check_limit < 0:
        raise ValueError("path_check_limit must be >= 0")

    manifests_dir = os.path.join(dataset_dir, "manifests")
    if not os.path.isdir(dataset_dir):
        raise DatasetContractError(f"Dataset directory not found: {dataset_dir}")
    if not os.path.isdir(manifests_dir):
        raise DatasetContractError(f"Manifest directory not found: {manifests_dir}")

    summary_path = os.path.join(manifests_dir, "summary.json")
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)

    split_counts: dict[str, int] = {}
    listing_sets: dict[str, set[str]] = {split: set() for split in REQUIRED_SPLITS}
    checked_path_count = 0
    missing_file_count = 0
    missing_field_count = 0
    summary_mismatch_count = 0

    for split in REQUIRED_SPLITS:
        manifest_path = os.path.join(manifests_dir, f"{split}.jsonl")
        if not os.path.exists(manifest_path):
            raise DatasetContractError(f"Missing manifest: {manifest_path}")

        split_count = 0
        path_budget = None
        if path_check_mode == "full":
            path_budget = -1
        elif path_check_mode == "sample":
            path_budget = path_check_limit

        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                split_count += 1

                missing = [field for field in REQUIRED_FIELDS if field not in record]
                if missing:
                    missing_field_count += len(missing)
                    raise DatasetContractError(
                        f"{manifest_path}:{line_number} missing required fields: {', '.join(missing)}"
                    )

                listing_id = record.get("listingId")
                if listing_id is not None:
                    listing_sets[split].add(str(listing_id))

                should_check_path = path_budget is not None and path_budget != 0
                if should_check_path:
                    image_path = os.path.join(dataset_dir, record["output_path"])
                    checked_path_count += 1
                    if not os.path.exists(image_path):
                        missing_file_count += 1
                        raise DatasetContractError(
                            f"{manifest_path}:{line_number} references missing file: {image_path}"
                        )
                    if path_budget > 0:
                        path_budget -= 1

        split_counts[split] = split_count
        if summary is not None:
            expected = _summary_count(summary, split)
            if expected is not None and expected != split_count:
                summary_mismatch_count += 1

    leakage_pairs: dict[str, int] = {}
    leakage_count = 0
    for left, right in combinations(REQUIRED_SPLITS, 2):
        overlap_count = len(listing_sets[left].intersection(listing_sets[right]))
        leakage_pairs[f"{left}_{right}"] = overlap_count
        leakage_count += overlap_count

    return DatasetContractReport(
        dataset_dir=dataset_dir,
        split_counts=split_counts,
        checked_path_count=checked_path_count,
        missing_file_count=missing_file_count,
        missing_field_count=missing_field_count,
        leakage_count=leakage_count,
        summary_mismatch_count=summary_mismatch_count,
        path_check_mode=path_check_mode,
        path_check_limit=path_check_limit if path_check_mode == "sample" else None,
        leakage_pairs=leakage_pairs,
    )
