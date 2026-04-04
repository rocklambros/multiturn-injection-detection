"""Download prompt injection datasets from HuggingFace.

Downloads three datasets, saves as parquet, generates manifest.json.
No Kaggle datasets or APIs.
"""

from src.utils.seed import set_global_seed
set_global_seed(42)

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset


RAW_DIR = Path("data/raw")

SOURCES = {
    "deepset": {
        "path": "deepset/prompt-injections",
        "text_col": "text",
        "label_col": "label",
    },
    "safeguard": {
        "path": "xTRam1/safe-guard-prompt-injection",
        "text_col": "prompt",
        "label_col": "label",
    },
    "neuralchemy": {
        "path": "neuralchemy/Prompt-injection-dataset",
        "text_col": "text",
        "label_col": "label",
    },
}


def download_dataset(name, config, max_retries=3):
    """Download a single dataset from HuggingFace with retry logic.

    Args:
        name: Short name for the dataset (e.g., 'deepset').
        config: Dict with 'path', 'text_col', 'label_col'.
        max_retries: Number of retry attempts with exponential backoff.

    Returns:
        Dict with download metadata (rows, columns, timestamp).

    Side effects:
        Saves parquet file to data/raw/{name}/data.parquet.
    """
    out_dir = RAW_DIR / name
    os.makedirs(out_dir, exist_ok=True)

    for attempt in range(max_retries):
        try:
            print(f"\n{'='*60}")
            print(f"Downloading {name}: {config['path']} (attempt {attempt+1})")
            ds = load_dataset(config["path"])

            # Combine all splits into one DataFrame
            frames = []
            for split_name, split_ds in ds.items():
                df = split_ds.to_pandas()
                df["_split"] = split_name
                frames.append(df)
                print(f"  Split '{split_name}': {len(df)} rows, columns: {list(df.columns)}")

            combined = pd.concat(frames, ignore_index=True)
            print(f"  Combined: {len(combined)} rows")

            # Print label distribution
            label_col = config["label_col"]
            if label_col in combined.columns:
                print(f"  Label distribution:")
                dist = combined[label_col].value_counts()
                for val, count in dist.items():
                    print(f"    {val}: {count} ({count/len(combined)*100:.1f}%)")

            # Save as parquet
            parquet_path = out_dir / "data.parquet"
            combined.to_parquet(parquet_path, index=False)
            print(f"  Saved to {parquet_path}")

            return {
                "source": name,
                "hf_path": config["path"],
                "rows": len(combined),
                "columns": list(combined.columns),
                "text_col": config["text_col"],
                "label_col": config["label_col"],
                "splits": {s: len(d) for s, d in zip(ds.keys(), frames)},
                "download_timestamp": datetime.now(timezone.utc).isoformat(),
                "parquet_path": str(parquet_path),
            }

        except Exception as e:
            wait = 2 ** attempt
            print(f"  ERROR: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Failed to download {name} after {max_retries} attempts: {e}")


def download_all():
    """Download all three datasets and save manifest.

    Returns:
        Dict: manifest with metadata for all datasets.

    Side effects:
        Creates data/raw/{source}/ directories with parquet files.
        Saves data/raw/manifest.json.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    manifest = {"datasets": [], "download_date": datetime.now(timezone.utc).isoformat()}

    for name, config in SOURCES.items():
        meta = download_dataset(name, config)
        manifest["datasets"].append(meta)

    # Save manifest
    manifest_path = RAW_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    total = 0
    for ds_meta in manifest["datasets"]:
        print(f"  {ds_meta['source']}: {ds_meta['rows']} rows")
        total += ds_meta["rows"]
    print(f"  Total: {total} rows")

    return manifest


if __name__ == "__main__":
    download_all()
