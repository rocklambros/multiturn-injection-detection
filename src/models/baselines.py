"""Iteration 0: sklearn baselines for prompt injection detection.

TF-IDF + Logistic Regression and TF-IDF + Random Forest.
These have no temporal awareness and will fail on multi-turn sequences.
"""

from src.utils.seed import set_global_seed
set_global_seed(42)

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import compute_metrics, save_metrics
from src.evaluation.analysis import plot_confusion_matrix, plot_confidence_histogram
from src.evaluation.visualization import plot_roc_curve, plot_pr_curve


def create_pipelines():
    """Create TF-IDF + LR and TF-IDF + RF pipelines.

    Returns:
        Dict of pipeline_name -> sklearn Pipeline.
    """
    pipeline_lr = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    pipeline_rf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    return {"iter0_baseline_lr": pipeline_lr, "iter0_baseline_rf": pipeline_rf}


def evaluate_single_turn(pipeline, name, train_df, test_df):
    """Train and evaluate a pipeline on single-turn data.

    Args:
        pipeline: sklearn Pipeline.
        name: Iteration name for saving results.
        train_df: Training DataFrame with text, label columns.
        test_df: Test DataFrame with text, label columns.

    Returns:
        Dict of metrics.

    Side effects:
        Saves metrics, plots to results/{name}/.
    """
    print(f"\n{'='*60}")
    print(f"Training {name} on single-turn data")
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    X_train = train_df["text"].tolist()
    y_train = train_df["label"].values
    X_test = test_df["text"].tolist()
    y_test = test_df["label"].values

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"  y_pred shape: {y_pred.shape}, y_prob shape: {y_prob.shape}")

    metrics = compute_metrics(y_test, y_pred, y_prob)
    print(f"  F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    save_metrics(metrics, name)
    plot_confusion_matrix(y_test, y_pred, name)
    plot_confidence_histogram(y_prob, y_test, name)
    plot_roc_curve(y_test, y_prob, name)
    plot_pr_curve(y_test, y_prob, name)

    return metrics


def evaluate_multiturn(pipeline, name, test_json_path):
    """Evaluate a trained pipeline on multi-turn data (concatenated turns).

    Args:
        pipeline: Trained sklearn Pipeline.
        name: Iteration name (will append '_multiturn').
        test_json_path: Path to multiturn_test.json.

    Returns:
        Dict of metrics.

    Side effects:
        Prints metrics showing baseline failure on multi-turn data.
    """
    with open(test_json_path) as f:
        data = json.load(f)

    # Concatenate all turns into single string (no temporal awareness)
    texts = []
    labels = []
    for seq in data:
        combined_text = " ".join(turn["text"] for turn in seq["turns"])
        texts.append(combined_text)
        labels.append(seq["label"])

    y_test = np.array(labels)
    y_pred = pipeline.predict(texts)
    y_prob = pipeline.predict_proba(texts)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    mt_name = f"{name}_multiturn"
    print(f"\n  Multi-turn evaluation ({mt_name}):")
    print(f"  F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, "
          f"Recall: {metrics['recall']:.4f}")
    print(f"  NOTE: Baselines fail on multi-turn because TF-IDF has no temporal awareness.")
    print(f"  Concatenating turns loses the sequential relationship between them.")

    save_metrics(metrics, mt_name)

    return metrics


def run_baselines():
    """Execute full baseline evaluation pipeline.

    Side effects:
        Trains LR and RF pipelines, evaluates on single-turn and multi-turn,
        saves all metrics and plots.
    """
    train_df = pd.read_csv("data/processed/single_turn_train.csv")
    test_df = pd.read_csv("data/processed/single_turn_test.csv")

    pipelines = create_pipelines()
    results = {}

    for name, pipeline in pipelines.items():
        # Single-turn evaluation
        st_metrics = evaluate_single_turn(pipeline, name, train_df, test_df)
        results[f"{name}_single"] = st_metrics

        # Multi-turn evaluation (concatenated turns)
        mt_metrics = evaluate_multiturn(pipeline, name, "data/synthetic/multiturn_test.json")
        results[f"{name}_multiturn"] = mt_metrics

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    for name, metrics in results.items():
        print(f"  {name}: F1={metrics['f1']:.4f}")

    return results


if __name__ == "__main__":
    run_baselines()
