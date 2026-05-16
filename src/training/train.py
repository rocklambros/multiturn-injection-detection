"""Training loop with early stopping, LR scheduling, and checkpointing."""

import copy
import json
import os
import time

import torch
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def compute_accuracy(outputs, targets):
    """Compute binary classification accuracy.

    Args:
        outputs: Model predictions, shape (batch, 1) or (batch,).
        targets: Ground truth labels, shape (batch, 1) or (batch,).

    Returns:
        float: Accuracy as a fraction in [0, 1].
    """
    preds = (outputs >= 0.0).float()
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


def train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    """Run one training epoch.

    Args:
        model: PyTorch nn.Module.
        train_loader: DataLoader yielding (inputs, labels) batches.
        optimizer: PyTorch optimizer.
        criterion: Loss function.
        device: torch.device for computation.
        scheduler: Optional per-step LR scheduler (e.g. warmup).

    Returns:
        tuple: (average_loss, average_accuracy, nan_detected).
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    nan_detected = False

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="  Training", leave=False)):
        turn_mask = None
        if len(batch) == 4:
            inputs, mask, turn_mask, labels = batch
            inputs = inputs.to(device)
            mask = mask.to(device)
            turn_mask = turn_mask.to(device)
            labels = labels.to(device)
        elif len(batch) == 3:
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
        if turn_mask is not None:
            outputs = model(inputs, mask, turn_mask)
        elif mask is not None:
            outputs = model(inputs, mask)
        else:
            outputs = model(inputs)

        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        if batch_idx == 0:
            print(f"    [Shape] Model output: {outputs.shape}, labels after reshape: {labels.shape}")

        loss = criterion(outputs, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    [NaN] Loss is NaN/Inf at batch {batch_idx}, skipping batch")
            nan_detected = True
            optimizer.zero_grad()
            continue

        loss.backward()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        if HAS_WANDB and wandb.run is not None and batch_idx % 50 == 0:
            wandb.log({"batch_grad_norm": grad_norm.item()})

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        running_correct += ((outputs >= 0.0).float() == labels).sum().item()
        total_samples += inputs.size(0)

    if total_samples == 0:
        return float('nan'), 0.0, True
    avg_loss = running_loss / total_samples
    avg_acc = running_correct / total_samples
    return avg_loss, avg_acc, nan_detected


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
            turn_mask = None
            if len(batch) == 4:
                inputs, mask, turn_mask, labels = batch
                inputs = inputs.to(device)
                mask = mask.to(device)
                turn_mask = turn_mask.to(device)
                labels = labels.to(device)
            elif len(batch) == 3:
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

            if turn_mask is not None:
                outputs = model(inputs, mask, turn_mask)
            elif mask is not None:
                outputs = model(inputs, mask)
            else:
                outputs = model(inputs)

            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_correct += ((outputs >= 0.0).float() == labels).sum().item()
            total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples
    avg_acc = running_correct / total_samples
    return avg_loss, avg_acc


def train_model(model, train_loader, val_loader, epochs, iteration_name,
                optimizer, criterion, device, patience=3, wandb_config=None,
                warmup_steps=0, max_nan_rollbacks=3):
    """Train a PyTorch model with early stopping, NaN rollback, and optional warmup.

    Args:
        model: PyTorch nn.Module.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Maximum training epochs.
        iteration_name: String identifier for saving results.
        optimizer: PyTorch optimizer.
        criterion: Loss function.
        device: torch.device.
        patience: Early stopping patience (default 3).
        wandb_config: Optional dict for WandB logging.
        warmup_steps: Linear warmup steps (0 = no warmup).
        max_nan_rollbacks: Max NaN rollbacks before aborting.

    Returns:
        dict: Training history with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
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

    # LR schedulers
    reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, min_lr=1e-6
    )

    warmup_scheduler = None
    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps,
        )

    # WandB initialization
    if HAS_WANDB and wandb_config is not None:
        wandb.init(
            project=wandb_config.get("project", "multiturn-injection-detection-v2"),
            name=iteration_name,
            group=wandb_config.get("group", "training"),
            config={
                "iteration": iteration_name,
                "epochs": epochs,
                "patience": patience,
                "lr": optimizer.param_groups[0]["lr"],
                "model_params": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
            tags=wandb_config.get("tags", []),
        )

    # Early stopping state
    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    epochs_without_improvement = 0
    nan_rollback_count = 0

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

        train_loss, train_acc, nan_detected = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=warmup_scheduler,
        )

        if nan_detected or (isinstance(train_loss, float) and (train_loss != train_loss)):
            nan_rollback_count += 1
            print(f"  [NaN ROLLBACK {nan_rollback_count}/{max_nan_rollbacks}] Restoring last good state")
            if nan_rollback_count >= max_nan_rollbacks:
                print(f"  [ABORT] Too many NaN rollbacks, stopping training")
                model.load_state_dict(best_model_weights)
                break
            model.load_state_dict(best_model_weights)
            optimizer.load_state_dict(best_optimizer_state)
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.5
            print(f"  [NaN ROLLBACK] LR halved to {optimizer.param_groups[0]['lr']:.2e}")
            continue

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        reduce_scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if HAS_WANDB and wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })

        elapsed = time.time() - epoch_start
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
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

    if HAS_WANDB and wandb.run is not None:
        wandb.finish()

    return history
