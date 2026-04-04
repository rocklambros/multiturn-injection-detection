"""Training loop with early stopping, LR scheduling, and checkpointing."""

from src.utils.seed import set_global_seed
set_global_seed(42)

import copy
import json
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm


def compute_accuracy(outputs, targets):
    """Compute binary classification accuracy.

    Args:
        outputs: Model predictions, shape (batch, 1) or (batch,).
        targets: Ground truth labels, shape (batch, 1) or (batch,).

    Returns:
        float: Accuracy as a fraction in [0, 1].
    """
    preds = (outputs >= 0.5).float()
    targets_flat = targets.view(-1)
    preds_flat = preds.view(-1)
    correct = (preds_flat == targets_flat).sum().item()
    return correct / targets_flat.size(0)


def save_model_summary(model, path):
    """Save a text summary of the model architecture and parameter count.

    Args:
        model: PyTorch nn.Module.
        path: File path to write the summary.

    Side effects:
        Writes model_summary.txt to the given path.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines = [
        "Model Summary",
        "=" * 60,
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
        f"Non-trainable parameters: {total_params - trainable_params:,}",
        "=" * 60,
        "",
        "Architecture:",
        str(model),
        "",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Model summary saved to {path}")


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Run one training epoch.

    Args:
        model: PyTorch nn.Module.
        train_loader: DataLoader yielding (inputs, labels) batches.
        optimizer: PyTorch optimizer.
        criterion: Loss function.
        device: torch.device for computation.

    Returns:
        tuple: (average_loss, average_accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="  Training", leave=False)):
        # Handle both 2-element (inputs, labels) and 3-element (inputs, mask, labels) batches
        if len(batch) == 3:
            inputs, mask, labels = batch
            inputs = inputs.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
        else:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask = None

        if batch_idx == 0:
            print(f"    [Shape] Train batch inputs: {inputs.shape}, labels: {labels.shape}")
            if mask is not None:
                print(f"    [Shape] Train batch mask: {mask.shape}")

        optimizer.zero_grad()
        if mask is not None:
            outputs = model(inputs, mask)
        else:
            outputs = model(inputs)

        # Handle models that output (batch,) instead of (batch, 1)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        if batch_idx == 0:
            print(f"    [Shape] Model output: {outputs.shape}, labels after reshape: {labels.shape}")

        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_correct += ((outputs >= 0.5).float() == labels).sum().item()
        total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples
    avg_acc = running_correct / total_samples
    return avg_loss, avg_acc


def validate(model, val_loader, criterion, device):
    """Run validation pass.

    Args:
        model: PyTorch nn.Module.
        val_loader: DataLoader yielding (inputs, labels) batches.
        criterion: Loss function.
        device: torch.device for computation.

    Returns:
        tuple: (average_loss, average_accuracy) for the validation set.
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="  Validating", leave=False)):
            if len(batch) == 3:
                inputs, mask, labels = batch
                inputs = inputs.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                mask = None

            if batch_idx == 0:
                print(f"    [Shape] Val batch inputs: {inputs.shape}, labels: {labels.shape}")

            if mask is not None:
                outputs = model(inputs, mask)
            else:
                outputs = model(inputs)

            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_correct += ((outputs >= 0.5).float() == labels).sum().item()
            total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples
    avg_acc = running_correct / total_samples
    return avg_loss, avg_acc


def train_model(model, train_loader, val_loader, epochs, iteration_name,
                optimizer, criterion, device, patience=3):
    """Train a PyTorch model with early stopping, LR scheduling, and checkpointing.

    Args:
        model: PyTorch nn.Module.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Maximum training epochs.
        iteration_name: String identifier for saving results (e.g., 'iter1_lstm').
        optimizer: PyTorch optimizer (typically Adam).
        criterion: Loss function (typically BCELoss).
        device: torch.device ('cuda' or 'cpu').
        patience: Early stopping patience (default 3).

    Returns:
        dict: Training history with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
              (lists of floats, one entry per epoch).

    Side effects:
        - Saves training_history.json to results/{iteration_name}/
        - Saves model_summary.txt to results/{iteration_name}/
        - Saves best model weights to models/{iteration_name}.pt
    """
    # Setup directories
    results_dir = os.path.join("results", iteration_name)
    models_dir = "models"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    model_save_path = os.path.join(models_dir, f"{iteration_name}.pt")
    history_path = os.path.join(results_dir, "training_history.json")
    summary_path = os.path.join(results_dir, "model_summary.txt")

    # Move model to device
    model = model.to(device)
    print(f"[INFO] Training on device: {device}")
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Save model summary
    save_model_summary(model, summary_path)

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, min_lr=1e-6
    )

    # Early stopping state
    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - epoch_start
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  [CHECKPOINT] Best model saved to {model_save_path}")
        else:
            epochs_without_improvement += 1
            print(f"  [EARLY STOP] No improvement for {epochs_without_improvement}/{patience} epochs")

            if epochs_without_improvement >= patience:
                print(f"\n[INFO] Early stopping triggered after {epoch} epochs.")
                model.load_state_dict(best_model_weights)
                break

    # Restore best weights
    model.load_state_dict(best_model_weights)
    print(f"\n[INFO] Restored best model weights (val_loss={best_val_loss:.4f})")

    # Save training history
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] Training history saved to {history_path}")

    return history
