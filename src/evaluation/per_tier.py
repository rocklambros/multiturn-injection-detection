"""Per-tier evaluation breakdown for multi-turn classification.

Reports accuracy, F1, precision, recall, and AUC per difficulty tier
(easy, medium, hard, adversarial, template) and overall.
"""

import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_per_tier(predictions, labels, tier_labels, output_path=None):
    """Evaluate classification performance broken down by difficulty tier.

    Args:
        predictions: np.array of predicted probabilities, shape (N,).
        labels: np.array of ground truth binary labels, shape (N,).
        tier_labels: list of tier strings (e.g. 'easy', 'medium'), length N.
        output_path: Optional path to save JSON results.

    Returns:
        dict: Nested dict with 'overall' and per-tier metrics.
    """
    preds_binary = (predictions >= 0.5).astype(int)

    results = {}

    results["overall"] = {
        "accuracy": float(accuracy_score(labels, preds_binary)),
        "f1": float(f1_score(labels, preds_binary)),
        "precision": float(precision_score(labels, preds_binary)),
        "recall": float(recall_score(labels, preds_binary)),
        "auc": float(roc_auc_score(labels, predictions)),
        "n": int(len(labels)),
    }

    tiers = sorted(set(tier_labels))
    results["per_tier"] = {}

    for tier in tiers:
        mask = np.array([t == tier for t in tier_labels])
        tier_labels_sub = labels[mask]
        tier_preds = predictions[mask]
        tier_preds_binary = preds_binary[mask]

        if len(tier_labels_sub) == 0:
            continue

        tier_results = {
            "accuracy": float(accuracy_score(tier_labels_sub, tier_preds_binary)),
            "f1": float(f1_score(tier_labels_sub, tier_preds_binary, zero_division=0)),
            "precision": float(precision_score(tier_labels_sub, tier_preds_binary, zero_division=0)),
            "recall": float(recall_score(tier_labels_sub, tier_preds_binary, zero_division=0)),
            "n": int(mask.sum()),
            "n_attack": int(tier_labels_sub.sum()),
            "n_benign": int((1 - tier_labels_sub).sum()),
        }

        if len(set(tier_labels_sub)) > 1:
            tier_results["auc"] = float(roc_auc_score(tier_labels_sub, tier_preds))

        results["per_tier"][tier] = tier_results

    print("\n" + "=" * 70)
    print("PER-TIER EVALUATION")
    print("=" * 70)
    print(f"{'Tier':<15} {'N':>6} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'AUC':>8}")
    print("-" * 70)

    for tier in tiers:
        if tier not in results["per_tier"]:
            continue
        r = results["per_tier"][tier]
        auc_str = f"{r.get('auc', 0):.4f}" if "auc" in r else "N/A"
        print(f"  {tier:<13} {r['n']:>6} {r['accuracy']:>8.4f} {r['f1']:>8.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} {auc_str:>8}")

    r = results["overall"]
    print("-" * 70)
    print(f"  {'OVERALL':<13} {r['n']:>6} {r['accuracy']:>8.4f} {r['f1']:>8.4f} "
          f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['auc']:>8.4f}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {output_path}")

    return results


def load_tier_labels(data_path):
    """Extract tier labels from a multiturn JSON file.

    Args:
        data_path: Path to multiturn_{split}.json.

    Returns:
        list of tier strings, one per sequence.
    """
    with open(data_path) as f:
        data = json.load(f)

    tiers = []
    for seq in data:
        tier = seq.get("difficulty", seq.get("tier", "unknown"))
        if "template" in seq.get("id", ""):
            tier = "template"
        tiers.append(tier)

    return tiers
