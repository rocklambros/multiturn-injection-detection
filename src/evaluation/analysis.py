"""Error analysis utilities for prompt injection detection."""

from src.utils.seed import set_global_seed
set_global_seed(42)

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, iteration_name):
    """Plot and save a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels (numpy array).
        y_pred: Predicted binary labels (numpy array).
        iteration_name: Name of the current iteration for saving.
    """
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    print(f"[plot_confusion_matrix] y_true shape: {y_true.shape}, "
          f"y_pred shape: {y_pred.shape}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"[plot_confusion_matrix] Confusion matrix shape: {cm.shape}")

    out_dir = os.path.join("results", iteration_name)
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Injection"],
                yticklabels=["Benign", "Injection"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {iteration_name}")

    out_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_confusion_matrix] Saved to {out_path}")


def get_top_errors(texts, y_true, y_pred, y_prob, n=10):
    """Return the top false-positive and false-negative predictions.

    Args:
        texts: List of input text strings.
        y_true: Ground truth labels (numpy array).
        y_pred: Predicted binary labels (numpy array).
        y_prob: Predicted probabilities (numpy array).
        n: Number of top errors to return per category.

    Returns:
        Dict with 'false_positives' and 'false_negatives', each a list of
        dicts with keys: text, true_label, pred_label, confidence.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    print(f"[get_top_errors] Analysing {len(texts)} samples, "
          f"y_true shape: {y_true.shape}")

    # False positives: predicted 1, actually 0 — sorted by descending confidence
    fp_mask = (y_pred == 1) & (y_true == 0)
    fp_indices = np.where(fp_mask)[0]
    fp_sorted = fp_indices[np.argsort(-y_prob[fp_indices])][:n]

    # False negatives: predicted 0, actually 1 — sorted by ascending confidence
    fn_mask = (y_pred == 0) & (y_true == 1)
    fn_indices = np.where(fn_mask)[0]
    fn_sorted = fn_indices[np.argsort(y_prob[fn_indices])][:n]

    def _build_records(indices):
        """Build error record dicts for given indices."""
        return [
            {
                "text": texts[i],
                "true_label": int(y_true[i]),
                "pred_label": int(y_pred[i]),
                "confidence": float(y_prob[i]),
            }
            for i in indices
        ]

    result = {
        "false_positives": _build_records(fp_sorted),
        "false_negatives": _build_records(fn_sorted),
    }
    print(f"[get_top_errors] Found {len(result['false_positives'])} FP, "
          f"{len(result['false_negatives'])} FN (top {n})")
    return result


def plot_confidence_histogram(y_prob, y_true, iteration_name):
    """Plot confidence distribution for correct vs incorrect predictions.

    Args:
        y_prob: Predicted probabilities (numpy array).
        y_true: Ground truth labels (numpy array).
        iteration_name: Name of the current iteration for saving.
    """
    y_prob = np.asarray(y_prob)
    y_true = np.asarray(y_true)
    print(f"[plot_confidence_histogram] y_prob shape: {y_prob.shape}, "
          f"y_true shape: {y_true.shape}")

    out_dir = os.path.join("results", iteration_name)
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_prob[y_true == 0], bins=30, alpha=0.6, label="Benign", color="steelblue")
    ax.hist(y_prob[y_true == 1], bins=30, alpha=0.6, label="Injection", color="salmon")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence Distribution — {iteration_name}")
    ax.legend()

    out_path = os.path.join(out_dir, "confidence_histogram.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_confidence_histogram] Saved to {out_path}")


def plot_attention_heatmap(attention_weights, turn_texts, iteration_name, sample_idx):
    """Plot an attention heatmap for a multi-turn conversation sample.

    Args:
        attention_weights: 2-D numpy array of shape (num_heads_or_layers, num_turns).
        turn_texts: List of strings, one per conversation turn.
        iteration_name: Name of the current iteration for saving.
        sample_idx: Index of the sample being visualised.
    """
    attention_weights = np.asarray(attention_weights)
    if attention_weights.ndim == 1:
        attention_weights = attention_weights.reshape(1, -1)
    print(f"[plot_attention_heatmap] attention_weights shape: "
          f"{attention_weights.shape}, turns: {len(turn_texts)}")

    out_dir = os.path.join("results", iteration_name)
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, len(turn_texts)), 6))
    sns.heatmap(attention_weights, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=turn_texts, ax=ax)
    ax.set_xlabel("Turns")
    ax.set_ylabel("Heads / Layers")
    ax.set_title(f"Attention Heatmap — Sample {sample_idx} — {iteration_name}")
    plt.xticks(rotation=45, ha="right")

    out_path = os.path.join(out_dir, f"attention_heatmap_sample_{sample_idx}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_attention_heatmap] Saved to {out_path}")
