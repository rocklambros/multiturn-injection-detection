"""Cross-iteration visualization utilities for prompt injection detection."""

from src.utils.seed import set_global_seed
set_global_seed(42)

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def plot_training_curves(history, iteration_name):
    """Plot training and validation loss/accuracy curves.

    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                 each a list of per-epoch values.
        iteration_name: Name of the current iteration for saving.
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))
    print(f"[plot_training_curves] Plotting {len(epochs)} epochs for {iteration_name}")

    out_dir = os.path.join("results", iteration_name)
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss — {iteration_name}")
    ax1.legend()

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    ax2.plot(epochs, history["val_acc"], label="Val Acc", marker="o", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy — {iteration_name}")
    ax2.legend()

    out_path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_training_curves] Saved to {out_path}")


def plot_roc_curve(y_true, y_prob, iteration_name):
    """Plot ROC curve with AUC score.

    Args:
        y_true: Ground truth labels (numpy array).
        y_prob: Predicted probabilities (numpy array).
        iteration_name: Name of the current iteration for saving.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    print(f"[plot_roc_curve] y_true shape: {y_true.shape}, "
          f"y_prob shape: {y_prob.shape}")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"[plot_roc_curve] AUC: {roc_auc:.4f}")

    out_dir = os.path.join("results", iteration_name)
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {iteration_name}")
    ax.legend(loc="lower right")

    out_path = os.path.join(out_dir, "roc_curve.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_roc_curve] Saved to {out_path}")


def plot_pr_curve(y_true, y_prob, iteration_name):
    """Plot Precision-Recall curve with average precision.

    Args:
        y_true: Ground truth labels (numpy array).
        y_prob: Predicted probabilities (numpy array).
        iteration_name: Name of the current iteration for saving.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    print(f"[plot_pr_curve] y_true shape: {y_true.shape}, "
          f"y_prob shape: {y_prob.shape}")

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    print(f"[plot_pr_curve] AP: {ap:.4f}")

    out_dir = os.path.join("results", iteration_name)
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="mediumblue", lw=2, label=f"PR (AP = {ap:.4f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {iteration_name}")
    ax.legend(loc="lower left")

    out_path = os.path.join(out_dir, "pr_curve.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_pr_curve] Saved to {out_path}")


def plot_iteration_comparison(iteration_metrics: dict):
    """Plot bar chart comparing F1 scores across iterations.

    Args:
        iteration_metrics: Dict mapping iteration_name -> metrics dict
                          (each must have an 'f1' key).
    """
    names = list(iteration_metrics.keys())
    f1_scores = [iteration_metrics[n]["f1"] for n in names]
    print(f"[plot_iteration_comparison] Comparing {len(names)} iterations: {names}")
    print(f"[plot_iteration_comparison] F1 scores: {f1_scores}")

    out_dir = os.path.join("results", "comparison")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 6))
    bars = ax.bar(names, f1_scores, color="teal", edgecolor="black")
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Comparison Across Iterations")
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=30, ha="right")

    out_path = os.path.join(out_dir, "iteration_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_iteration_comparison] Saved to {out_path}")


def plot_attention_heatmap(weights, labels, iteration_name, idx):
    """Plot a seaborn attention heatmap for a given sample.

    Args:
        weights: 2-D numpy array of attention weights.
        labels: List of labels for the x-axis (e.g., token or turn labels).
        iteration_name: Name of the current iteration for saving.
        idx: Sample index used in the filename.
    """
    weights = np.asarray(weights)
    print(f"[plot_attention_heatmap] weights shape: {weights.shape}, "
          f"labels count: {len(labels)}")

    out_dir = os.path.join("results", iteration_name)
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8, len(labels)), 6))
    sns.heatmap(weights, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=labels, ax=ax)
    ax.set_xlabel("Tokens / Turns")
    ax.set_ylabel("Heads / Layers")
    ax.set_title(f"Attention Heatmap — Sample {idx} — {iteration_name}")
    plt.xticks(rotation=45, ha="right")

    out_path = os.path.join(out_dir, f"attention_heatmap_{idx}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_attention_heatmap] Saved to {out_path}")
