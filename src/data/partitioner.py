"""3-way source text partitioner with SHA-256 manifest."""

import hashlib
import json
import random
from pathlib import Path

import pandas as pd


def partition_source_texts(df, output_dir, seed=42, train_ratio=0.70, val_ratio=0.15):
    """Partition all source texts into non-overlapping train/val/test pools.

    Args:
        df: DataFrame with 'text' and 'label' columns.
        output_dir: Directory to write manifest JSON.
        seed: Random seed.
        train_ratio: Fraction for training pool.
        val_ratio: Fraction for validation pool (test gets remainder).

    Returns:
        Dict manifest with pool assignments and SHA-256 hashes.
    """
    random.seed(seed)
    test_ratio = 1.0 - train_ratio - val_ratio

    manifest = {
        "injection_pools": {"train": [], "val": [], "test": []},
        "benign_pools": {"train": [], "val": [], "test": []},
        "injection_hashes": {"train": [], "val": [], "test": []},
        "benign_hashes": {"train": [], "val": [], "test": []},
    }

    for label, pool_key in [(1, "injection"), (0, "benign")]:
        texts = df[df["label"] == label]["text"].tolist()
        random.shuffle(texts)

        n = len(texts)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {
            "train": texts[:train_end],
            "val": texts[train_end:val_end],
            "test": texts[val_end:],
        }

        for split_name, split_texts in splits.items():
            manifest[f"{pool_key}_pools"][split_name] = split_texts
            manifest[f"{pool_key}_hashes"][split_name] = [
                hashlib.sha256(t.encode()).hexdigest() for t in split_texts
            ]

    out_path = Path(output_dir) / "partition_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_manifest(path):
    """Load partition manifest from JSON."""
    with open(path) as f:
        return json.load(f)
