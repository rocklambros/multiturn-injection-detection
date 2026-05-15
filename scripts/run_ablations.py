"""Ablation runner for RunPod GPU instances.

Usage:
    python scripts/run_ablations.py --ablation a10_voting
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_global_seed
from src.utils.tokenizer import load_vocab
from src.models.run_multi_turn import load_encoder_decision, load_turn_encoder, load_multiturn_data
from src.models.ablations import (
    MeanPoolClassifier, MaxPoolClassifier, LearnedWeightedMeanClassifier,
    TurnLevelVoting,
)
from src.training.train import train_model
from src.evaluation.metrics import compute_metrics, save_metrics
from src.evaluation.bootstrap import bootstrap_ci


def run_a10_voting():
    """A10: Turn-level voting baselines — HIGHEST PRIORITY."""
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = load_vocab("models/vocab.json")
    decision = load_encoder_decision()
    turn_encoder = load_turn_encoder(decision, vocab, device)

    _, val_loader, test_loader, _ = load_multiturn_data(vocab, batch_size=32, data_dir="data/synthetic_v2")

    voting = TurnLevelVoting(turn_encoder, device)

    # Sweep thresholds on validation set
    for method_name, predict_fn in [
        ("a10_max_vote", voting.predict_max_vote),
        ("a10_mean_vote", voting.predict_mean_vote),
        ("a10_top3_mean", lambda x, m, t: voting.predict_top_k_mean(x, m, k=3, threshold=t)),
    ]:
        best_f1 = 0
        best_thresh = 0.5

        # Collect val predictions at each threshold
        for thresh in np.arange(0.05, 0.95, 0.05):
            all_preds, all_labels = [], []
            for inputs, mask, labels in val_loader:
                inputs, mask = inputs.to(device), mask.to(device)
                preds, _ = predict_fn(inputs, mask, thresh)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
            from sklearn.metrics import f1_score
            f1 = f1_score(all_labels, all_preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        # Evaluate on test with best threshold
        all_preds, all_scores, all_labels = [], [], []
        for inputs, mask, labels in test_loader:
            inputs, mask = inputs.to(device), mask.to(device)
            preds, scores = predict_fn(inputs, mask, best_thresh)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_scores)

        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics["best_threshold"] = float(best_thresh)
        metrics["bootstrap_ci"] = bootstrap_ci(y_true, y_pred)
        save_metrics(metrics, method_name)

        print(f"{method_name}: F1={metrics['f1']:.4f} (thresh={best_thresh:.2f})")


ABLATIONS = {
    "a10_voting": run_a10_voting,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True, choices=list(ABLATIONS.keys()))
    args = parser.parse_args()
    ABLATIONS[args.ablation]()
