"""Upload regenerated data + model artifacts to WandB for RunPod download.

Usage: python scripts/upload_wandb_artifact.py
"""

import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dotenv
dotenv.load_dotenv()

import wandb

PROJECT = "multiturn-injection-detection-v2"
ARTIFACT_NAME = "synthetic_v2_data"


def main():
    required_files = [
        "data/synthetic_v2/multiturn_train.json",
        "data/synthetic_v2/multiturn_val.json",
        "data/synthetic_v2/multiturn_test.json",
        "data/processed/single_turn_train.csv",
        "data/processed/single_turn_val.csv",
        "data/processed/single_turn_test.csv",
        "models/vocab.json",
        "models/v2_gru_retrain.pt",
        "results/encoder_decision.json",
    ]

    print("Checking required files...")
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"ERROR: Missing files: {missing}")
        sys.exit(1)
    print("  All required files present")

    run = wandb.init(
        project=PROJECT,
        job_type="data-upload",
        name="upload_v3_clean_data",
    )

    art = wandb.Artifact(
        ARTIFACT_NAME,
        type="dataset",
        description="Clean regenerated data after fixing 4 leakage vectors",
    )

    print("\nAdding files to artifact...")

    for f in required_files:
        art.add_file(f)
        size_mb = os.path.getsize(f) / 1e6
        print(f"  {f} ({size_mb:.1f} MB)")

    for pattern in [
        "data/synthetic_v2/llm_*_stripped.jsonl",
        "data/synthetic_v2/template_*.json",
        "data/synthetic_v2/partition_manifest.json",
        "data/synthetic_v2/intents.json",
    ]:
        for f in sorted(glob.glob(pattern)):
            art.add_file(f)
            size_mb = os.path.getsize(f) / 1e6
            print(f"  {f} ({size_mb:.1f} MB)")

    total_size = sum(
        e.size for e in art.manifest.entries.values()
    ) / 1e6
    print(f"\nTotal artifact size: {total_size:.1f} MB")
    print(f"Files: {len(art.manifest.entries)}")

    print("\nUploading...")
    run.log_artifact(art)
    run.finish()
    print(f"Artifact uploaded: {PROJECT}/{ARTIFACT_NAME}:latest")


if __name__ == "__main__":
    main()
