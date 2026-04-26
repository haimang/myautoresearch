import json
import os
import mlx.core as mx
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Tuple

from utils.transforms import load_and_transform_image

class FloorplanDataset:
    def __init__(self, dataset_dir: str, split: str = "train", batch_size: int = 32, max_samples: int = None):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.manifest_path = os.path.join(dataset_dir, "manifests", f"{split}.jsonl")
        
        self.samples = []
        # Label mappings based on manifest inspection
        self.bed_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5+": 4}
        self.bath_map = {"1": 0, "2": 1, "3": 2, "4+": 3}
        self.park_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4+": 4}
        
        self._load_manifest(max_samples)
        
    def _load_manifest(self, max_samples: int = None):
        if not os.path.exists(self.manifest_path):
            print(f"Warning: Manifest not found at {self.manifest_path}")
            return
            
        with open(self.manifest_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.samples.append({
                    "path": os.path.join(self.dataset_dir, data["output_path"]),
                    "bed": data.get("bedroom_head", "0"),
                    "bath": data.get("bathroom_head", "1"),
                    "park": data.get("parking_head", "0")
                })
                if max_samples and len(self.samples) >= max_samples:
                    break
                    
    def __len__(self):
        return len(self.samples)
        
    def get_batches(self) -> Iterator[Tuple[mx.array, mx.array, mx.array, mx.array]]:
        if not self.samples:
            return
            
        indices = np.random.permutation(len(self.samples))
        
        def load_sample(idx):
            s = self.samples[idx]
            img = load_and_transform_image(s["path"])
            
            bed_lbl = self.bed_map.get(str(s["bed"]), 0)
            bath_lbl = self.bath_map.get(str(s["bath"]), 0)
            park_lbl = self.park_map.get(str(s["park"]), 0)
            
            return img, bed_lbl, bath_lbl, park_lbl

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            # Asynchronous I/O to avoid blocking GPU
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(load_sample, batch_indices))
            
            imgs = [r[0] for r in results]
            beds = [r[1] for r in results]
            baths = [r[2] for r in results]
            parks = [r[3] for r in results]
            
            batch_imgs = mx.stack(imgs)
            batch_beds = mx.array(beds, dtype=mx.int32)
            batch_baths = mx.array(baths, dtype=mx.int32)
            batch_parks = mx.array(parks, dtype=mx.int32)
            
            yield batch_imgs, batch_beds, batch_baths, batch_parks
