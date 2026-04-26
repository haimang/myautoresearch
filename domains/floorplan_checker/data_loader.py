from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import mlx.core as mx
import numpy as np

try:
    from .utils.transforms import load_and_transform_image
except ImportError:  # pragma: no cover - direct script execution path
    from utils.transforms import load_and_transform_image

BEDROOM_LABEL_MAP = {"1": 0, "2": 1, "3": 2, "4": 3, "5+": 4}
BATHROOM_LABEL_MAP = {"1": 0, "2": 1, "3": 2, "4+": 3}
PARKING_LABEL_MAP = {"0": 0, "1": 1, "2": 2, "3": 3, "4+": 4}


class FloorplanDataset:
    def __init__(
        self,
        dataset_dir: str,
        *,
        split: str = "train",
        batch_size: int = 32,
        max_samples: int | None = None,
        image_size: tuple[int, int] = (256, 256),
        shuffle: bool = True,
        num_workers: int = 8,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.split = split
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.num_workers = max(1, num_workers)
        self.manifest_path = os.path.join(dataset_dir, "manifests", f"{split}.jsonl")
        self.samples: list[dict[str, str]] = []
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self._load_manifest(max_samples)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "FloorplanDataset":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _load_manifest(self, max_samples: int | None = None) -> None:
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    output_path = record["output_path"]
                    bed = str(record["bedroom_head"])
                    bath = str(record["bathroom_head"])
                    park = str(record["parking_head"])
                except KeyError as exc:
                    raise ValueError(f"{self.manifest_path}:{line_number} missing field: {exc.args[0]}") from exc

                if bed not in BEDROOM_LABEL_MAP:
                    raise ValueError(f"{self.manifest_path}:{line_number} unknown bedroom label: {bed}")
                if bath not in BATHROOM_LABEL_MAP:
                    raise ValueError(f"{self.manifest_path}:{line_number} unknown bathroom label: {bath}")
                if park not in PARKING_LABEL_MAP:
                    raise ValueError(f"{self.manifest_path}:{line_number} unknown parking label: {park}")

                self.samples.append(
                    {
                        "path": os.path.join(self.dataset_dir, output_path),
                        "bed": bed,
                        "bath": bath,
                        "park": park,
                    }
                )
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample(self, idx: int) -> tuple[mx.array, int, int, int]:
        sample = self.samples[idx]
        image = load_and_transform_image(sample["path"], size=self.image_size)
        return (
            image,
            BEDROOM_LABEL_MAP[sample["bed"]],
            BATHROOM_LABEL_MAP[sample["bath"]],
            PARKING_LABEL_MAP[sample["park"]],
        )

    def get_batches(self) -> Iterator[tuple[mx.array, mx.array, mx.array, mx.array]]:
        if not self.samples:
            return

        indices = np.arange(len(self.samples))
        if self.shuffle:
            indices = np.random.permutation(indices)

        results = self._executor.map(self._load_sample, indices.tolist(), chunksize=max(1, self.batch_size // 2))
        batch: list[tuple[mx.array, int, int, int]] = []
        for item in results:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield self._stack_batch(batch)
                batch = []

        if batch:
            yield self._stack_batch(batch)

    def _stack_batch(
        self, batch: list[tuple[mx.array, int, int, int]]
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        images = [item[0] for item in batch]
        beds = [item[1] for item in batch]
        baths = [item[2] for item in batch]
        parks = [item[3] for item in batch]
        return (
            mx.stack(images),
            mx.array(beds, dtype=mx.int32),
            mx.array(baths, dtype=mx.int32),
            mx.array(parks, dtype=mx.int32),
        )
