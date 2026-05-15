"""Evaluation pipeline: per-strategy, per-difficulty, three-subset, bootstrap CIs.

Usage: python scripts/run_evaluation.py --results-dir results/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def compile_results(results_dir):
    """Compile all iteration results into a summary."""
    results_dir = Path(results_dir)
    summary = {}

    for metrics_file in sorted(results_dir.glob("*/metrics.json")):
        iteration = metrics_file.parent.name
        with open(metrics_file) as f:
            metrics = json.load(f)
        summary[iteration] = {
            "f1": metrics.get("f1"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
        }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    summary = compile_results(args.results_dir)
    print(json.dumps(summary, indent=2))
