"""Run transformer model iterations for comparison with LSTM/GRU.

Iteration 4b: Custom small Transformer encoder (controlled comparison)
Iteration 4c: DistilBERT fine-tuned (transfer learning comparison)

Implements Chollet heuristic analysis to explain when each architecture wins.
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
from torch.utils.data import DataLoader, Dataset

from src.utils.tokenizer import load_vocab, encode_texts
from src.data.loader import SingleTurnDataset
from src.models.transformer import TransformerClassifier, DistilBERTClassifier
from src.training.train import train_model
from src.evaluation.metrics import compute_metrics, save_metrics
from src.evaluation.analysis import plot_confusion_matrix, plot_confidence_histogram
from src.evaluation.visualization import plot_training_curves, plot_roc_curve, plot_pr_curve


class BERTDataset(Dataset):
    """Dataset for DistilBERT with tokenizer-produced inputs.

    Args:
        input_ids: Tensor of token IDs.
        attention_mask: Tensor of attention masks.
        labels: Tensor of labels.
    """

    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


def evaluate_model(model, test_loader, device, iteration_name, is_bert=False):
    """Evaluate a trained model on test set.

    Args:
        model: Trained PyTorch model.
        test_loader: Test DataLoader.
        device: torch.device.
        iteration_name: Name for saving results.
        is_bert: If True, handle 3-element batches (input_ids, attention_mask, labels).

    Returns:
        Dict of metrics.
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if is_bert:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids, attention_mask)
            else:
                inputs, labels = batch
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

    plot_confusion_matrix(y_true, y_pred, iteration_name)
    plot_confidence_histogram(y_prob, y_true, iteration_name)
    plot_roc_curve(y_true, y_prob, iteration_name)
    plot_pr_curve(y_true, y_prob, iteration_name)

    return metrics


def run_custom_transformer(vocab, train_loader, val_loader, test_loader, device):
    """Iteration 4b: Custom small Transformer encoder.

    Controlled comparison with LSTM/GRU using same vocab and embeddings.

    Returns:
        Dict of test metrics.
    """
    print(f"\n{'#'*60}")
    print("ITERATION 4b: Custom Transformer Encoder")
    print(f"{'#'*60}")

    set_global_seed(42)
    model = TransformerClassifier(
        vocab_size=len(vocab),
        embedding_dim=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.3,
        max_len=256,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    start = time.time()
    history = train_model(
        model, train_loader, val_loader,
        epochs=20,
        iteration_name="iter4b_transformer",
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        patience=3,
    )
    total_time = time.time() - start
    epochs_trained = len(history["train_loss"])
    print(f"Training time: {total_time:.0f}s ({total_time/epochs_trained:.1f}s/epoch)")

    plot_training_curves(history, "iter4b_transformer")

    metrics = evaluate_model(model, test_loader, device, "iter4b_transformer")
    print(f"\nIteration 4b Test F1: {metrics['f1']:.4f}")

    return metrics, param_count


def run_distilbert(device):
    """Iteration 4c: DistilBERT fine-tuned classifier.

    Transfer learning comparison — pretrained language model applied to security domain.

    Returns:
        Dict of test metrics.
    """
    print(f"\n{'#'*60}")
    print("ITERATION 4c: DistilBERT Fine-Tuned")
    print(f"{'#'*60}")

    from transformers import DistilBertTokenizer

    set_global_seed(42)

    # Load data
    train_df = pd.read_csv("data/processed/single_turn_train.csv")
    val_df = pd.read_csv("data/processed/single_turn_val.csv")
    test_df = pd.read_csv("data/processed/single_turn_test.csv")

    # Tokenize with DistilBERT tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def encode_df(df, max_len=128):
        """Encode DataFrame texts using DistilBERT tokenizer."""
        encoded = tokenizer(
            df["text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    print("Tokenizing with DistilBERT tokenizer...")
    train_ids, train_mask = encode_df(train_df)
    val_ids, val_mask = encode_df(val_df)
    test_ids, test_mask = encode_df(test_df)
    print(f"  Train: {train_ids.shape}, Val: {val_ids.shape}, Test: {test_ids.shape}")

    # Create datasets and loaders
    train_ds = BERTDataset(train_ids, train_mask, torch.FloatTensor(train_df["label"].values))
    val_ds = BERTDataset(val_ids, val_mask, torch.FloatTensor(val_df["label"].values))
    test_ds = BERTDataset(test_ids, test_mask, torch.FloatTensor(test_df["label"].values))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # Create model (frozen body)
    model = DistilBERTClassifier(freeze_body=True, dropout=0.3)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    model = model.to(device)

    # Custom training loop for BERT (3-element batches)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4,
    )
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    os.makedirs("results/iter4c_distilbert", exist_ok=True)

    for epoch in range(10):
        print(f"\n  Epoch {epoch+1}/10")

        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * input_ids.size(0)
            preds = (outputs >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += input_ids.size(0)

            if batch_idx == 0:
                print(f"    [Shape] input_ids: {input_ids.shape}, output: {outputs.shape}")

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * input_ids.size(0)
                preds = (outputs >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += input_ids.size(0)

        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"    Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"    Val   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/iter4c_distilbert.pt")
            print(f"    [CHECKPOINT] Saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    [EARLY STOP] after {epoch+1} epochs")
                break

    # Restore best
    model.load_state_dict(torch.load("models/iter4c_distilbert.pt", weights_only=True))

    # Save training history
    with open("results/iter4c_distilbert/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, "iter4c_distilbert")

    metrics = evaluate_model(model, test_loader, device, "iter4c_distilbert", is_bert=True)
    print(f"\nIteration 4c Test F1: {metrics['f1']:.4f}")

    return metrics, trainable_params


def compute_chollet_analysis():
    """Compute and save Chollet heuristic analysis.

    Returns:
        Dict with ratio, predictions, and empirical results.
    """
    train_df = pd.read_csv("data/processed/single_turn_train.csv")

    n_samples = len(train_df)
    mean_words = train_df["text"].str.split().str.len().mean()
    ratio = n_samples / mean_words

    analysis = {
        "n_training_samples": n_samples,
        "mean_words_per_sample": round(mean_words, 1),
        "chollet_ratio": round(ratio, 0),
        "threshold": 1500,
        "prediction": "bag-of-bigrams" if ratio < 1500 else "sequence/transformer",
        "explanation": (
            f"Chollet heuristic (Chapter 11/15): ratio = {n_samples} / {mean_words:.1f} = {ratio:.0f}. "
            f"Threshold is 1,500. At ratio {ratio:.0f}, the heuristic predicts "
            f"{'bag-of-bigrams models win' if ratio < 1500 else 'sequence/transformer models win'}. "
            f"This is {'confirmed' if ratio < 1500 else 'to be validated'} by our empirical results."
        ),
    }

    return analysis


def run_all():
    """Execute all transformer iterations and produce comparison analysis.

    Side effects:
        Trains custom transformer and DistilBERT, saves metrics and Chollet analysis.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data for custom transformer (uses same vocab as LSTM/GRU)
    vocab = load_vocab("models/vocab.json")
    train_df = pd.read_csv("data/processed/single_turn_train.csv")
    val_df = pd.read_csv("data/processed/single_turn_val.csv")
    test_df = pd.read_csv("data/processed/single_turn_test.csv")

    print(f"Data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Encode for custom transformer (same as LSTM/GRU)
    train_ids = encode_texts(vocab, train_df["text"].tolist(), max_len=256)
    val_ids = encode_texts(vocab, val_df["text"].tolist(), max_len=256)
    test_ids = encode_texts(vocab, test_df["text"].tolist(), max_len=256)

    train_ds = SingleTurnDataset(train_ids, torch.FloatTensor(train_df["label"].values))
    val_ds = SingleTurnDataset(val_ids, torch.FloatTensor(val_df["label"].values))
    test_ds = SingleTurnDataset(test_ids, torch.FloatTensor(test_df["label"].values))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # Run custom transformer
    transformer_metrics, transformer_params = run_custom_transformer(
        vocab, train_loader, val_loader, test_loader, device
    )

    # Run DistilBERT
    distilbert_metrics, distilbert_trainable = run_distilbert(device)

    # Chollet analysis
    chollet = compute_chollet_analysis()

    # Load all single-turn results for comparison
    all_results = {}
    for name in ["iter0_baseline_lr", "iter0_baseline_rf", "iter1_lstm", "iter2_lstm_glove",
                  "iter3_bilstm_dropout", "iter4_gru"]:
        try:
            with open(f"results/{name}/metrics.json") as f:
                all_results[name] = json.load(f)["f1"]
        except FileNotFoundError:
            pass

    all_results["iter4b_transformer"] = transformer_metrics["f1"]
    all_results["iter4c_distilbert"] = distilbert_metrics["f1"]

    # Comparison summary
    chollet["empirical_results"] = all_results
    chollet["transformer_params"] = transformer_params
    chollet["distilbert_trainable_params"] = distilbert_trainable

    # Best bag-of-words vs best sequence vs best transformer
    bow_best = max(all_results.get("iter0_baseline_lr", 0), all_results.get("iter0_baseline_rf", 0))
    seq_best = max(all_results.get("iter1_lstm", 0), all_results.get("iter4_gru", 0))
    trans_best = max(all_results.get("iter4b_transformer", 0), all_results.get("iter4c_distilbert", 0))

    chollet["bow_best_f1"] = bow_best
    chollet["sequence_best_f1"] = seq_best
    chollet["transformer_best_f1"] = trans_best
    chollet["heuristic_confirmed"] = bow_best >= seq_best if chollet["chollet_ratio"] < 1500 else seq_best >= bow_best

    os.makedirs("results", exist_ok=True)
    with open("results/chollet_analysis.json", "w") as f:
        json.dump(chollet, f, indent=2)
    print(f"\nChollet analysis saved to results/chollet_analysis.json")

    # Print summary
    print(f"\n{'='*60}")
    print("TRANSFORMER COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Chollet ratio: {chollet['chollet_ratio']} (threshold: 1,500)")
    print(f"Prediction: {chollet['prediction']}")
    print(f"\nAll F1 scores:")
    for name, f1 in sorted(all_results.items(), key=lambda x: -x[1]):
        print(f"  {name}: {f1:.4f}")
    print(f"\nBest bag-of-words:  {bow_best:.4f}")
    print(f"Best sequence:      {seq_best:.4f}")
    print(f"Best transformer:   {trans_best:.4f}")
    print(f"Heuristic confirmed: {chollet['heuristic_confirmed']}")

    return chollet


if __name__ == "__main__":
    run_all()
