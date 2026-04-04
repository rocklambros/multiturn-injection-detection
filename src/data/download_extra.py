"""Download additional prompt injection datasets from HuggingFace.

Expands the training data from ~16K to ~60K+ samples to push
the Chollet ratio above 1,500 (enabling transformer competitiveness).
"""

from src.utils.seed import set_global_seed
set_global_seed(42)

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset


RAW_DIR = Path("data/raw")

EXTRA_SOURCES = {
    "imoxto": {
        "path": "imoxto/prompt_injection_cleaned_dataset-v2",
        "text_col": "text",
        "label_col": "labels",
        "description": "535K binary prompt injection samples with system prompt context",
        "subsample": 40000,  # Subsample to keep manageable
    },
    "spml": {
        "path": "reshabhs/SPML_Chatbot_Prompt_Injection",
        "text_col": "User Prompt",
        "label_col": "Prompt injection",
        "description": "16K prompts from Gandalf CTF with injection degree labels",
        "subsample": None,
    },
    "trustailab_jailbreak": {
        "path": "TrustAIRLab/in-the-wild-jailbreak-prompts",
        "config": "jailbreak_2023_12_25",
        "text_col": "prompt",
        "label_val": 1,  # All jailbreak = injection
        "description": "1.4K real-world jailbreak prompts from the wild",
        "subsample": None,
    },
    "trustailab_regular": {
        "path": "TrustAIRLab/in-the-wild-jailbreak-prompts",
        "config": "regular_2023_12_25",
        "text_col": "prompt",
        "label_val": 0,  # All regular = benign
        "description": "13.7K real-world regular prompts from the wild",
        "subsample": None,
    },
    "jackhhao": {
        "path": "jackhhao/jailbreak-classification",
        "text_col": "prompt",
        "label_col": "type",
        "label_map": {"jailbreak": 1, "benign": 0},
        "description": "1.3K curated jailbreak/benign classification dataset",
        "subsample": None,
    },
}


def download_extra_dataset(name, config):
    """Download a single extra dataset from HuggingFace.

    Args:
        name: Short name for the dataset.
        config: Dict with dataset configuration.

    Returns:
        pd.DataFrame with columns: text, label, source.

    Side effects:
        Saves parquet to data/raw/{name}/data.parquet.
    """
    out_dir = RAW_DIR / name
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading {name}: {config['path']}")

    # Load from HuggingFace
    hf_config = config.get("config", None)
    if hf_config:
        ds = load_dataset(config["path"], hf_config)
    else:
        ds = load_dataset(config["path"])

    # Combine all splits
    frames = []
    for split_name in ds:
        split_ds = ds[split_name]
        # Convert without to_pandas for problematic datasets
        records = []
        text_col = config["text_col"]
        for i in range(len(split_ds)):
            row = split_ds[i]
            text = row[text_col]

            # Determine label
            if "label_val" in config:
                label = config["label_val"]
            elif "label_map" in config:
                label = config["label_map"][row[config["label_col"]]]
            else:
                label = int(row[config["label_col"]])

            records.append({"text": text, "label": label, "source": name})

        df = pd.DataFrame(records)
        frames.append(df)
        print(f"  Split '{split_name}': {len(df)} rows")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined: {len(combined)} rows")

    # Subsample if needed
    subsample = config.get("subsample")
    if subsample and len(combined) > subsample:
        # Stratified subsample
        from sklearn.model_selection import train_test_split
        combined, _ = train_test_split(
            combined, train_size=subsample, stratify=combined["label"], random_state=42
        )
        print(f"  Subsampled to {len(combined)} rows")

    # Print distribution
    dist = combined["label"].value_counts()
    print(f"  Labels: benign={dist.get(0, 0)}, injection={dist.get(1, 0)}")

    # Save
    parquet_path = out_dir / "data.parquet"
    combined.to_parquet(parquet_path, index=False)
    print(f"  Saved to {parquet_path}")

    return combined


def download_all_extra():
    """Download all extra datasets and update manifest.

    Returns:
        Dict: updated manifest.

    Side effects:
        Creates data/raw/{source}/ directories with parquet files.
        Updates data/raw/manifest.json.
    """
    # Load existing manifest
    manifest_path = RAW_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_extra = []
    for name, config in EXTRA_SOURCES.items():
        try:
            df = download_extra_dataset(name, config)
            all_extra.append(df)

            manifest["datasets"].append({
                "source": name,
                "hf_path": config["path"],
                "rows": len(df),
                "text_col": "text",
                "label_col": "label",
                "download_timestamp": datetime.now(timezone.utc).isoformat(),
                "parquet_path": str(RAW_DIR / name / "data.parquet"),
            })
        except Exception as e:
            print(f"  ERROR downloading {name}: {e}")

    # Save updated manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest updated: {manifest_path}")

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY (EXTRA)")
    print(f"{'='*60}")
    total_extra = 0
    for ds_meta in manifest["datasets"]:
        print(f"  {ds_meta['source']}: {ds_meta['rows']} rows")
        total_extra += ds_meta["rows"]
    print(f"  Total (all sources): {total_extra} rows")

    return manifest


if __name__ == "__main__":
    download_all_extra()
