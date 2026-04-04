"""Classification metrics for prompt injection detection."""

from src.utils.seed import set_global_seed
set_global_seed(42)

import json
import os
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute all classification metrics.

    Args:
        y_true: Ground truth labels (numpy array).
        y_pred: Predicted binary labels (numpy array).
        y_prob: Predicted probabilities (numpy array).

    Returns:
        Dict with keys: f1 (PRIMARY), precision, recall, roc_auc, pr_auc,
        confusion_matrix (as list of lists), accuracy.
    """
    print(f"[compute_metrics] y_true shape: {np.asarray(y_true).shape}, "
          f"y_pred shape: {np.asarray(y_pred).shape}, "
          f"y_prob shape: {np.asarray(y_prob).shape}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    metrics = {
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    print(f"[compute_metrics] F1 (primary): {metrics['f1']:.4f}, "
          f"Accuracy: {metrics['accuracy']:.4f}")
    return metrics


def save_metrics(metrics_dict, iteration_name):
    """Save metrics to results/{iteration_name}/metrics.json.

    Args:
        metrics_dict: Dictionary of metric name -> value.
        iteration_name: Name of the current iteration/experiment.
    """
    out_dir = os.path.join("results", iteration_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.json")

    with open(out_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"[save_metrics] Saved metrics to {out_path}")
