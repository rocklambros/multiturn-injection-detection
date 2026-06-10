"""Download all training results from WandB artifacts after RunPod training.

Usage: python scripts/collect_runpod_results.py [--check-only]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Run: pip install wandb")
    sys.exit(1)

TASKS = ["gru_retrain", "iter5", "iter6", "distilbert_hier", "distilbert_concat"]
PROJECT = "REDACTED/multiturn-injection-detection-v2"


def check_artifacts():
    api = wandb.Api()
    status = {}
    for task in TASKS:
        try:
            art = api.artifact(f"{PROJECT}/{task}_results:latest")
            status[task] = {
                "found": True,
                "version": art.version,
                "created": str(art.created_at),
                "size": art.size,
            }
        except wandb.errors.CommError:
            status[task] = {"found": False}
    return status


def download_all():
    api = wandb.Api()
    downloaded = []
    missing = []

    for task in TASKS:
        try:
            art = api.artifact(f"{PROJECT}/{task}_results:latest")
            print(f"Downloading {task} (v{art.version}, {art.size / 1e6:.1f} MB)...")
            art.download(".")
            downloaded.append(task)
        except wandb.errors.CommError:
            print(f"WARNING: {task}_results not found — task may not have completed")
            missing.append(task)

    return downloaded, missing


def verify_results():
    expected = {
        "gru_retrain": ["models/v2_gru_retrain.pt", "results/v2_gru_retrain/training_history.json"],
        "iter5": ["models/v2_iter5_multiturn.pt", "results/v2_iter5_multiturn/training_history.json"],
        "iter6": ["models/v2_iter6_attention.pt", "results/v2_iter6_attention/training_history.json"],
        "distilbert_hier": ["models/v2_distilbert_hier.pt", "results/v2_distilbert_hier/training_history.json"],
        "distilbert_concat": ["models/v2_distilbert_concat.pt", "results/v2_distilbert_concat/training_history.json"],
    }

    all_ok = True
    for task, files in expected.items():
        for f in files:
            if os.path.exists(f):
                size = os.path.getsize(f)
                print(f"  OK  {f} ({size / 1e6:.1f} MB)")
            else:
                print(f"  MISSING  {f}")
                all_ok = False

    return all_ok


def print_training_summary():
    print("\n=== Training Summary ===\n")
    for task in TASKS:
        history_path = f"results/v2_{task}/training_history.json"
        if not os.path.exists(history_path):
            print(f"{task}: NO RESULTS")
            continue

        with open(history_path) as f:
            history = json.load(f)

        best_epoch = min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i])
        best_val_loss = history["val_loss"][best_epoch]
        best_val_acc = history["val_acc"][best_epoch]
        total_epochs = len(history["val_loss"])

        print(f"{task}:")
        print(f"  Epochs: {total_epochs}")
        print(f"  Best val loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
        print(f"  Best val acc:  {best_val_acc:.4f}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="Only check artifact availability")
    args = parser.parse_args()

    if args.check_only:
        print("Checking WandB artifacts...\n")
        status = check_artifacts()
        for task, info in status.items():
            if info["found"]:
                print(f"  FOUND  {task}_results (v{info['version']}, {info['size'] / 1e6:.1f} MB)")
            else:
                print(f"  MISSING  {task}_results")
        found = sum(1 for s in status.values() if s["found"])
        print(f"\n{found}/{len(TASKS)} artifacts found")
        sys.exit(0 if found == len(TASKS) else 1)

    print("=== Collecting RunPod Training Results ===\n")

    downloaded, missing = download_all()

    print(f"\nDownloaded: {len(downloaded)}/{len(TASKS)}")
    if missing:
        print(f"Missing: {', '.join(missing)}")

    print("\nVerifying files...")
    all_ok = verify_results()

    print_training_summary()

    if not all_ok:
        print("WARNING: Some files are missing. Check RunPod logs and WandB.")
        sys.exit(1)

    print("All results collected successfully.")
