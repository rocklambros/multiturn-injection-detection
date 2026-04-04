"""Clean, merge, deduplicate, and split prompt injection datasets.

Implements all 9 cleaning steps from PRD Section 3.2.
Produces train/val/test CSVs and a bias report.
"""

from src.utils.seed import set_global_seed
set_global_seed(42)

import json
import os
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_raw_datasets():
    """Load raw parquet files and normalize columns to (text, label, source).

    Returns:
        pd.DataFrame with columns: text, label, source.

    Side effects:
        Prints row counts and schemas per source.
    """
    with open(RAW_DIR / "manifest.json") as f:
        manifest = json.load(f)

    frames = []
    for ds_meta in manifest["datasets"]:
        source = ds_meta["source"]
        parquet_path = ds_meta["parquet_path"]
        if not os.path.exists(parquet_path):
            print(f"\nSkipping {source}: {parquet_path} not found")
            continue

        df = pd.read_parquet(parquet_path)
        print(f"\nLoaded {source}: {len(df)} rows, columns: {list(df.columns)}")

        # Step 1: Normalize columns to text and label
        text_col = ds_meta["text_col"] if ds_meta["text_col"] in df.columns else "text"
        label_col = ds_meta["label_col"] if ds_meta["label_col"] in df.columns else "label"

        normalized = pd.DataFrame({
            "text": df[text_col],
            "label": df[label_col],
            "source": source,
        })

        # Step 2: Normalize labels to 0=benign, 1=injection
        unique_labels = normalized["label"].unique()
        print(f"  Raw labels: {sorted(unique_labels)}")

        normalized["label"] = normalized["label"].astype(int)

        dist = normalized["label"].value_counts()
        print(f"  Label dist: benign={dist.get(0, 0)}, injection={dist.get(1, 0)}")

        frames.append(normalized)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {len(combined)} rows")
    return combined


def clean_dataset(df):
    """Apply all 9 cleaning steps from PRD Section 3.2 in order.

    Args:
        df: DataFrame with columns text, label, source.

    Returns:
        Cleaned DataFrame.

    Side effects:
        Prints removal counts per step.
    """
    removals = {}
    initial = len(df)
    print(f"\nStarting cleaning: {initial} rows")

    # Step 3: Strip leading/trailing whitespace
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()

    # Step 4: Collapse internal whitespace, normalize newlines to spaces
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)

    # Step 5: Remove exact duplicates on text (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    removals["duplicates"] = before - len(df)
    print(f"  Exact duplicates removed: {removals['duplicates']}")

    # Step 6: Remove near-duplicates (identical after lowercasing + stripping punctuation)
    translator = str.maketrans("", "", string.punctuation)
    df["_normalized"] = df["text"].str.lower().str.translate(translator).str.strip()
    before = len(df)
    df = df.drop_duplicates(subset=["_normalized"], keep="first")
    removals["near_duplicates"] = before - len(df)
    df = df.drop(columns=["_normalized"])
    print(f"  Near-duplicates removed: {removals['near_duplicates']}")

    # Step 7: Remove rows where text is empty or fewer than 3 tokens
    before = len(df)
    df = df[df["text"].str.split().str.len() >= 3]
    removals["empty"] = before - len(df)
    print(f"  Empty/short removed: {removals['empty']}")

    # Step 8: Remove rows where text exceeds 2048 characters
    before = len(df)
    df = df[df["text"].str.len() <= 2048]
    removals["too_long"] = before - len(df)
    print(f"  Too long removed: {removals['too_long']}")

    # Step 9: Log removals
    print(f"\nRemoval log: {json.dumps(removals)}")
    print(f"Rows remaining: {len(df)} (removed {initial - len(df)} total)")

    return df.reset_index(drop=True)


def split_dataset(df):
    """Split 70/15/15 stratified on label with seed=42.

    Args:
        df: Cleaned DataFrame with columns text, label, source.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Side effects:
        Prints split sizes and class distributions.
    """
    # First split: 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )

    # Second split: 50/50 of temp -> 15% val, 15% test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df["label"].value_counts()
        benign = dist.get(0, 0)
        injection = dist.get(1, 0)
        total = len(split_df)
        print(f"  {name}: {total} rows (benign: {benign} [{benign/total*100:.1f}%], "
              f"injection: {injection} [{injection/total*100:.1f}%])")

    return train_df, val_df, test_df


def generate_bias_report(train_df, val_df, test_df):
    """Generate bias report per PRD Section 3.2.

    Args:
        train_df, val_df, test_df: Split DataFrames.

    Returns:
        str: Bias report text.

    Side effects:
        Saves report to data/processed/bias_report.txt.
    """
    lines = ["BIAS REPORT", "=" * 60, ""]

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        lines.append(f"--- {name} Split ({len(df)} samples) ---")

        # Class distribution
        dist = df["label"].value_counts()
        for label, count in sorted(dist.items()):
            label_name = "benign" if label == 0 else "injection"
            lines.append(f"  Class {label} ({label_name}): {count} ({count/len(df)*100:.1f}%)")

        # Source distribution
        src_dist = df["source"].value_counts()
        for src, count in src_dist.items():
            lines.append(f"  Source '{src}': {count} ({count/len(df)*100:.1f}%)")

        # Text length distribution per class
        for label in [0, 1]:
            label_name = "benign" if label == 0 else "injection"
            subset = df[df["label"] == label]
            lengths = subset["text"].str.len()
            lines.append(f"  Text length ({label_name}): "
                        f"mean={lengths.mean():.0f}, std={lengths.std():.0f}, "
                        f"min={lengths.min()}, max={lengths.max()}")
        lines.append("")

    lines.append("KNOWN LIMITATIONS")
    lines.append("-" * 40)
    lines.append("- All datasets are English-only. Non-English injection patterns are not represented.")
    lines.append("- Datasets skew toward known attack patterns. Novel social engineering approaches are underrepresented.")
    lines.append("")

    report = "\n".join(lines)
    return report


def run_pipeline():
    """Execute the full cleaning and splitting pipeline.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Side effects:
        Creates data/processed/ directory and saves CSVs and bias report.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load and merge
    df = load_raw_datasets()

    # Clean
    df = clean_dataset(df)

    # Split
    print(f"\nSplitting 70/15/15 stratified:")
    train_df, val_df, test_df = split_dataset(df)

    # Save CSVs
    train_df.to_csv(PROCESSED_DIR / "single_turn_train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "single_turn_val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "single_turn_test.csv", index=False)
    print(f"\nSaved to {PROCESSED_DIR}/single_turn_{{train,val,test}}.csv")

    # Bias report
    report = generate_bias_report(train_df, val_df, test_df)
    report_path = PROCESSED_DIR / "bias_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Bias report saved to {report_path}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    run_pipeline()
