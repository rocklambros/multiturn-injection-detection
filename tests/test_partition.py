import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
from src.data.partitioner import partition_source_texts, load_manifest


def test_partition_zero_overlap():
    """No text appears in more than one pool."""
    injection_texts = [f"inject command {i}" for i in range(100)]
    benign_texts = [f"hello how are you {i}" for i in range(200)]

    df = pd.DataFrame({
        "text": injection_texts + benign_texts,
        "label": [1] * 100 + [0] * 200,
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = partition_source_texts(df, output_dir=tmpdir, seed=42)

        inj_pools = [set(manifest["injection_pools"][k]) for k in ["train", "val", "test"]]
        assert len(inj_pools[0] & inj_pools[1]) == 0, "train/val injection overlap"
        assert len(inj_pools[0] & inj_pools[2]) == 0, "train/test injection overlap"
        assert len(inj_pools[1] & inj_pools[2]) == 0, "val/test injection overlap"

        ben_pools = [set(manifest["benign_pools"][k]) for k in ["train", "val", "test"]]
        assert len(ben_pools[0] & ben_pools[1]) == 0, "train/val benign overlap"
        assert len(ben_pools[0] & ben_pools[2]) == 0, "train/test benign overlap"
        assert len(ben_pools[1] & ben_pools[2]) == 0, "val/test benign overlap"

        all_inj = inj_pools[0] | inj_pools[1] | inj_pools[2]
        assert len(all_inj) == 100

        all_ben = ben_pools[0] | ben_pools[1] | ben_pools[2]
        assert len(all_ben) == 200


def test_partition_manifest_has_hashes():
    """Every text in the manifest has a SHA-256 hash."""
    injection_texts = [f"inject {i}" for i in range(50)]
    benign_texts = [f"benign {i}" for i in range(100)]

    df = pd.DataFrame({
        "text": injection_texts + benign_texts,
        "label": [1] * 50 + [0] * 100,
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = partition_source_texts(df, output_dir=tmpdir, seed=42)

        for pool_name, hashes in manifest["injection_hashes"].items():
            for text_hash in hashes:
                assert len(text_hash) == 64  # SHA-256 hex length


def test_partition_ratios():
    """Pool sizes approximate 70/15/15 split."""
    texts = [f"text {i}" for i in range(1000)]
    df = pd.DataFrame({"text": texts, "label": [1] * 500 + [0] * 500})

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = partition_source_texts(df, output_dir=tmpdir, seed=42)

        train_size = len(manifest["injection_pools"]["train"])
        val_size = len(manifest["injection_pools"]["val"])
        test_size = len(manifest["injection_pools"]["test"])
        total = train_size + val_size + test_size

        assert 0.65 <= train_size / total <= 0.75
        assert 0.10 <= val_size / total <= 0.20
        assert 0.10 <= test_size / total <= 0.20
