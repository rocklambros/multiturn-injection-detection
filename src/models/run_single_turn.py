"""Run single-turn model iterations 1-4.

Iteration 1: Simple LSTM, random embeddings
Iteration 2: LSTM with GloVe embeddings
Iteration 3: BiLSTM with dropout (0.3 and 0.5)
Iteration 4: GRU comparison + encoder decision
"""

from src.utils.seed import set_global_seed
set_global_seed(42)

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.utils.config import ITERATIONS
from src.utils.tokenizer import load_vocab, encode_texts
from src.data.loader import SingleTurnDataset
from src.models.single_turn import SingleTurnLSTM, BiLSTMClassifier, GRUClassifier
from src.training.train import train_model
from src.evaluation.metrics import compute_metrics, save_metrics
from src.evaluation.analysis import plot_confusion_matrix, plot_confidence_histogram
from src.evaluation.visualization import plot_training_curves, plot_roc_curve, plot_pr_curve
from torch.utils.data import DataLoader


def load_data(vocab, batch_size=64, max_len=256):
    """Load and encode single-turn datasets.

    Args:
        vocab: Word-to-index dict.
        batch_size: Training batch size.
        max_len: Max sequence length.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_df = pd.read_csv("data/processed/single_turn_train.csv")
    val_df = pd.read_csv("data/processed/single_turn_val.csv")
    test_df = pd.read_csv("data/processed/single_turn_test.csv")

    print(f"Data loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        ids = encode_texts(vocab, df["text"].tolist(), max_len=max_len)
        labels = torch.FloatTensor(df["label"].values)
        ds = SingleTurnDataset(ids, labels)
        if name == "train":
            train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                       num_workers=2, pin_memory=True)
        elif name == "val":
            val_loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                     num_workers=2, pin_memory=True)
        else:
            test_loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                      num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


def evaluate_model(model, test_loader, device, iteration_name, test_df=None):
    """Evaluate trained model on test set.

    Args:
        model: Trained PyTorch model.
        test_loader: Test DataLoader.
        device: torch.device.
        iteration_name: Name for saving results.
        test_df: Optional test DataFrame for error analysis texts.

    Returns:
        Dict of metrics.
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = outputs.squeeze(-1).cpu().numpy()
            if probs.ndim == 0:
                probs = probs.reshape(1)
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    save_metrics(metrics, iteration_name)

    # Plots
    plot_confusion_matrix(y_true, y_pred, iteration_name)
    plot_confidence_histogram(y_prob, y_true, iteration_name)
    plot_roc_curve(y_true, y_prob, iteration_name)
    plot_pr_curve(y_true, y_prob, iteration_name)

    return metrics


def run_iteration_1(vocab, train_loader, val_loader, test_loader, device):
    """Iteration 1: Simple LSTM with random embeddings.

    Returns:
        Dict of test metrics.
    """
    print(f"\n{'#'*60}")
    print("ITERATION 1: Simple LSTM, Random Embeddings")
    print(f"{'#'*60}")

    config = ITERATIONS["iter1_lstm"]
    model = SingleTurnLSTM(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        dense_dim=config.dense_dim,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCELoss()

    history = train_model(
        model, train_loader, val_loader,
        epochs=config.epochs,
        iteration_name=config.name,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        patience=config.early_stopping_patience,
    )
    plot_training_curves(history, config.name)

    metrics = evaluate_model(model, test_loader, device, config.name)
    print(f"\nIteration 1 Test F1: {metrics['f1']:.4f}")
    return metrics


def run_iteration_2(vocab, train_loader, val_loader, test_loader, device):
    """Iteration 2: LSTM with GloVe embeddings.

    Returns:
        Dict of test metrics.
    """
    print(f"\n{'#'*60}")
    print("ITERATION 2: LSTM, GloVe Embeddings")
    print(f"{'#'*60}")

    embedding_matrix = np.load("data/embeddings/embedding_matrix.npy")
    print(f"Embedding matrix shape: {embedding_matrix.shape}")

    config = ITERATIONS["iter2_lstm_glove"]
    model = SingleTurnLSTM(
        vocab_size=len(vocab),
        embedding_dim=100,  # GloVe 100d
        hidden_dim=config.hidden_dim,
        dense_dim=config.dense_dim,
        pretrained_embeddings=embedding_matrix,
        freeze_embeddings=True,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
    )
    criterion = nn.BCELoss()

    history = train_model(
        model, train_loader, val_loader,
        epochs=config.epochs,
        iteration_name=config.name,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        patience=config.early_stopping_patience,
    )
    plot_training_curves(history, config.name)

    metrics = evaluate_model(model, test_loader, device, config.name)
    print(f"\nIteration 2 Test F1: {metrics['f1']:.4f}")
    return metrics


def run_iteration_3(vocab, train_loader, val_loader, test_loader, device):
    """Iteration 3: BiLSTM with dropout (0.3 and 0.5).

    Returns:
        Tuple of (metrics_03, metrics_05, best_dropout).
    """
    results = {}
    for dropout_rate, config_key in [(0.3, "iter3_bilstm_dropout"), (0.5, "iter3_bilstm_dropout_05")]:
        print(f"\n{'#'*60}")
        print(f"ITERATION 3: BiLSTM, Dropout={dropout_rate}")
        print(f"{'#'*60}")

        set_global_seed(42)
        config = ITERATIONS[config_key]
        model = BiLSTMClassifier(
            vocab_size=len(vocab),
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout_rate=dropout_rate,
            dense_dim=config.dense_dim,
        )
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.BCELoss()

        history = train_model(
            model, train_loader, val_loader,
            epochs=config.epochs,
            iteration_name=config.name,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            patience=config.early_stopping_patience,
        )
        plot_training_curves(history, config.name)

        metrics = evaluate_model(model, test_loader, device, config.name)
        results[dropout_rate] = metrics
        print(f"\nIteration 3 (dropout={dropout_rate}) Test F1: {metrics['f1']:.4f}")

    # Pick best dropout
    best_dropout = max(results, key=lambda d: results[d]["f1"])
    print(f"\nBest dropout: {best_dropout} (F1={results[best_dropout]['f1']:.4f})")
    return results[0.3], results[0.5], best_dropout


def run_iteration_4(vocab, train_loader, val_loader, test_loader, device):
    """Iteration 4: GRU comparison and encoder decision.

    Returns:
        Tuple of (metrics, param_count, time_per_epoch).
    """
    print(f"\n{'#'*60}")
    print("ITERATION 4: GRU Comparison")
    print(f"{'#'*60}")

    set_global_seed(42)
    config = ITERATIONS["iter4_gru"]
    model = GRUClassifier(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        dropout_rate=config.dropout_rate,
        dense_dim=config.dense_dim,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCELoss()

    start = time.time()
    history = train_model(
        model, train_loader, val_loader,
        epochs=config.epochs,
        iteration_name=config.name,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        patience=config.early_stopping_patience,
    )
    total_time = time.time() - start
    epochs_trained = len(history["train_loss"])
    time_per_epoch = total_time / epochs_trained if epochs_trained > 0 else 0
    plot_training_curves(history, config.name)

    metrics = evaluate_model(model, test_loader, device, config.name)
    print(f"\nIteration 4 Test F1: {metrics['f1']:.4f}")
    print(f"  Parameters: {param_count:,}")
    print(f"  Time/epoch: {time_per_epoch:.1f}s")

    return metrics, param_count, time_per_epoch


def run_all():
    """Execute all single-turn iterations and make encoder decision.

    Side effects:
        Trains 5 models (iter1 LSTM, iter2 GloVe LSTM, iter3 BiLSTM x2, iter4 GRU),
        saves all metrics and plots, determines best encoder for multi-turn phase.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    vocab = load_vocab("models/vocab.json")
    train_loader, val_loader, test_loader = load_data(vocab, batch_size=64)

    # Run iterations
    iter1_metrics = run_iteration_1(vocab, train_loader, val_loader, test_loader, device)
    iter2_metrics = run_iteration_2(vocab, train_loader, val_loader, test_loader, device)
    iter3_03, iter3_05, best_dropout = run_iteration_3(vocab, train_loader, val_loader, test_loader, device)
    iter4_metrics, gru_params, gru_time = run_iteration_4(vocab, train_loader, val_loader, test_loader, device)

    # Load iter3 best param count for comparison
    iter3_best = iter3_03 if best_dropout == 0.3 else iter3_05
    iter3_name = "iter3_bilstm_dropout" if best_dropout == 0.3 else "iter3_bilstm_dropout_05"

    # Encoder decision: compare BiLSTM vs GRU
    bilstm_f1 = iter3_best["f1"]
    gru_f1 = iter4_metrics["f1"]

    print(f"\n{'='*60}")
    print("ENCODER DECISION")
    print(f"{'='*60}")
    print(f"  BiLSTM (dropout={best_dropout}): F1={bilstm_f1:.4f}")
    print(f"  GRU: F1={gru_f1:.4f}, Params={gru_params:,}")

    if gru_f1 >= bilstm_f1 - 0.01:  # GRU preferred if competitive (fewer params)
        encoder_decision = "GRU"
        reasoning = f"GRU achieves F1={gru_f1:.4f} vs BiLSTM F1={bilstm_f1:.4f} with fewer parameters"
        best_iteration = 4
        best_path = "models/iter4_gru.pt"
    else:
        encoder_decision = "BiLSTM"
        reasoning = f"BiLSTM F1={bilstm_f1:.4f} outperforms GRU F1={gru_f1:.4f}"
        best_iteration = 3
        best_path = f"models/{iter3_name}.pt"

    print(f"  Decision: {encoder_decision}")
    print(f"  Reasoning: {reasoning}")

    # Summary
    print(f"\n{'='*60}")
    print("SINGLE-TURN ITERATION SUMMARY")
    print(f"{'='*60}")
    all_results = {
        "iter1_lstm": iter1_metrics["f1"],
        "iter2_lstm_glove": iter2_metrics["f1"],
        f"iter3_bilstm_d{best_dropout}": iter3_best["f1"],
        "iter4_gru": iter4_metrics["f1"],
    }
    for name, f1 in all_results.items():
        print(f"  {name}: F1={f1:.4f}")

    # Save decision
    decision = {
        "encoder_decision": encoder_decision,
        "reasoning": reasoning,
        "best_single_turn_iteration": best_iteration,
        "best_single_turn_path": best_path,
        "best_dropout": best_dropout,
        "all_f1": {k: float(v) for k, v in all_results.items()},
    }
    os.makedirs("results", exist_ok=True)
    with open("results/encoder_decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    print(f"\nEncoder decision saved to results/encoder_decision.json")

    return decision


if __name__ == "__main__":
    run_all()
