"""Training orchestrator for RunPod GPU instances.

Usage:
    python scripts/run_training.py --task gru_retrain
    python scripts/run_training.py --task iter5
    python scripts/run_training.py --task iter6
    python scripts/run_training.py --task distilbert_hier
    python scripts/run_training.py --task distilbert_concat
    python scripts/run_training.py --task ablation_shuffled
    python scripts/run_training.py --task ablation_reversed
    python scripts/run_training.py --task ablation_prefix
    python scripts/run_training.py --task ablation_continuation
    python scripts/run_training.py --task ablation_autoencoder
    python scripts/run_training.py --task ablation_mean_pool
    python scripts/run_training.py --task ablation_max_pool
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_global_seed

DATA_DIR = "data/synthetic_v3"


def train_gru_retrain():
    """Retrain single-turn GRU with BCEWithLogitsLoss on full data."""
    import torch
    import torch.nn as nn
    from src.data.loader import create_single_turn_loaders
    from src.models.single_turn import GRUClassifier
    from src.training.train import train_model

    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, vocab = create_single_turn_loaders(batch_size=64)

    model = GRUClassifier(
        vocab_size=len(vocab), embedding_dim=128, hidden_dim=64,
        dropout_rate=0.3, dense_dim=32,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    train_model(
        model, train_loader, val_loader,
        epochs=30, iteration_name="v3_gru_retrain",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=5,
        wandb_config={"group": "training", "tags": ["v3", "gru", "retrain"]},
    )

    import json
    decision_path = "results/encoder_decision.json"
    with open(decision_path) as f:
        decision = json.load(f)
    decision["best_single_turn_path"] = "models/v3_gru_retrain.pt"
    decision["v3_retrained"] = True
    with open(decision_path, "w") as f:
        json.dump(decision, f, indent=2)
    print(f"Updated {decision_path} to point to v3 retrained encoder")


def train_iter5():
    """Retrain iter5 multi-turn LSTM on v3 data."""
    import torch
    import torch.nn as nn
    from src.utils.tokenizer import load_vocab
    from src.models.run_multi_turn import load_encoder_decision, load_turn_encoder, load_multiturn_data
    from src.models.multi_turn import MultiTurnClassifier
    from src.training.train import train_model

    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = load_vocab("models/vocab.json")
    decision = load_encoder_decision()
    turn_encoder = load_turn_encoder(decision, vocab, device)

    train_loader, val_loader, test_loader, _ = load_multiturn_data(
        vocab, batch_size=32, data_dir=DATA_DIR,
    )

    model = MultiTurnClassifier(
        turn_encoder=turn_encoder, turn_encoding_dim=32,
        hidden_dim=64, dropout_rate=0.3,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=0.01,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_model(
        model, train_loader, val_loader,
        epochs=30, iteration_name="v3_iter5_multiturn",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=5,
        wandb_config={"group": "training", "tags": ["v3", "iter5", "multiturn"]},
    )


def train_iter6():
    """Retrain iter6 attention model on v3 data."""
    import torch
    import torch.nn as nn
    from src.utils.tokenizer import load_vocab
    from src.models.run_multi_turn import load_encoder_decision, load_turn_encoder, load_multiturn_data
    from src.models.attention import MultiTurnAttentionClassifier
    from src.training.train import train_model

    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = load_vocab("models/vocab.json")
    decision = load_encoder_decision()
    turn_encoder = load_turn_encoder(decision, vocab, device)

    train_loader, val_loader, test_loader, _ = load_multiturn_data(
        vocab, batch_size=32, data_dir=DATA_DIR,
    )

    model = MultiTurnAttentionClassifier(
        turn_encoder=turn_encoder, turn_encoding_dim=32,
        hidden_dim=64, dropout_rate=0.3,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=0.01,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_model(
        model, train_loader, val_loader,
        epochs=30, iteration_name="v3_iter6_attention",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=5,
        wandb_config={"group": "training", "tags": ["v3", "iter6", "attention"]},
    )


def train_distilbert_hier():
    """Train hierarchical DistilBERT with positional encoding on v3 data."""
    import torch
    import torch.nn as nn
    from src.models.transformer_multiturn import HierarchicalDistilBERT
    from src.data.loader import create_distilbert_multiturn_loaders
    from src.training.train import train_model

    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = create_distilbert_multiturn_loaders(
        data_dir=DATA_DIR, batch_size=16, max_turns=10, max_len=128,
    )

    model = HierarchicalDistilBERT(
        num_attention_heads=4, cross_turn_layers=2,
        max_turns=10, dropout_rate=0.3, freeze_bert=True,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6, weight_decay=0.01,
    )
    criterion = nn.BCEWithLogitsLoss()

    warmup_steps = len(train_loader) * 3

    train_model(
        model, train_loader, val_loader,
        epochs=20, iteration_name="v3_distilbert_hier",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=5,
        wandb_config={"group": "training", "tags": ["v3", "distilbert", "hierarchical"]},
        warmup_steps=warmup_steps,
        max_nan_rollbacks=5,
    )


def train_distilbert_concat():
    """Train concatenated DistilBERT on v3 data."""
    import torch
    import torch.nn as nn
    from src.models.concat_distilbert import ConcatenatedDistilBERT
    from src.data.loader import create_concat_distilbert_loaders
    from src.training.train import train_model

    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = create_concat_distilbert_loaders(
        data_dir=DATA_DIR, batch_size=16, max_length=512,
    )

    model = ConcatenatedDistilBERT(
        max_length=512, dropout_rate=0.3, freeze_bert=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    warmup_steps = len(train_loader) * 2

    train_model(
        model, train_loader, val_loader,
        epochs=10, iteration_name="v3_distilbert_concat",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=3,
        wandb_config={"group": "training", "tags": ["v3", "distilbert", "concat"]},
        warmup_steps=warmup_steps,
    )


def _load_turn_encoder_and_data(batch_size=32):
    """Shared helper for ablation tasks that use the frozen GRU turn encoder."""
    import torch
    from src.utils.tokenizer import load_vocab
    from src.models.run_multi_turn import load_encoder_decision, load_turn_encoder, load_multiturn_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab("models/vocab.json")
    decision = load_encoder_decision()
    turn_encoder = load_turn_encoder(decision, vocab, device)

    train_loader, val_loader, test_loader, _ = load_multiturn_data(
        vocab, batch_size=batch_size, data_dir=DATA_DIR,
    )
    return turn_encoder, train_loader, val_loader, device


def train_ablation_shuffled():
    """A2a: Shuffled turns ablation."""
    import torch.nn as nn
    from src.models.ablations import ShuffledTurnsClassifier
    from src.training.train import train_model
    import torch

    set_global_seed(42)
    turn_encoder, train_loader, val_loader, device = _load_turn_encoder_and_data()

    model = ShuffledTurnsClassifier(
        turn_encoder=turn_encoder, turn_encoding_dim=32,
        hidden_dim=64, dropout_rate=0.3,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=0.01,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_model(
        model, train_loader, val_loader,
        epochs=30, iteration_name="v3_ablation_shuffled",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=5,
        wandb_config={"group": "ablations", "tags": ["v3", "A2a", "shuffled"]},
    )


def train_ablation_reversed():
    """A2b: Reversed turns ablation."""
    import torch.nn as nn
    from src.models.ablations import ReversedTurnsClassifier
    from src.training.train import train_model
    import torch

    set_global_seed(42)
    turn_encoder, train_loader, val_loader, device = _load_turn_encoder_and_data()

    model = ReversedTurnsClassifier(
        turn_encoder=turn_encoder, turn_encoding_dim=32,
        hidden_dim=64, dropout_rate=0.3,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=0.01,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_model(
        model, train_loader, val_loader,
        epochs=30, iteration_name="v3_ablation_reversed",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=5,
        wandb_config={"group": "ablations", "tags": ["v3", "A2b", "reversed"]},
    )


def train_ablation_prefix():
    """A12: Prefix-only ablation."""
    import torch.nn as nn
    from src.models.ablations import PrefixOnlyClassifier
    from src.training.train import train_model
    import torch

    set_global_seed(42)
    turn_encoder, train_loader, val_loader, device = _load_turn_encoder_and_data()

    model = PrefixOnlyClassifier(
        turn_encoder=turn_encoder, turn_encoding_dim=32,
        hidden_dim=64, dropout_rate=0.3,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=0.01,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_model(
        model, train_loader, val_loader,
        epochs=30, iteration_name="v3_ablation_prefix",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=5,
        wandb_config={"group": "ablations", "tags": ["v3", "A12", "prefix-only"]},
    )


def train_ablation_continuation():
    """A13: Continuation-only ablation."""
    import torch.nn as nn
    from src.models.ablations import ContinuationOnlyClassifier
    from src.training.train import train_model
    import torch

    set_global_seed(42)
    turn_encoder, train_loader, val_loader, device = _load_turn_encoder_and_data()

    model = ContinuationOnlyClassifier(
        turn_encoder=turn_encoder, turn_encoding_dim=32,
        hidden_dim=64, dropout_rate=0.3,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=0.01,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_model(
        model, train_loader, val_loader,
        epochs=30, iteration_name="v3_ablation_continuation",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=5,
        wandb_config={"group": "ablations", "tags": ["v3", "A13", "continuation-only"]},
    )


def train_ablation_autoencoder():
    """A14: Train autoencoder on turn encodings, then use as alternative encoder."""
    import torch
    import torch.nn as nn
    from src.utils.tokenizer import load_vocab
    from src.models.run_multi_turn import load_encoder_decision, load_turn_encoder, load_multiturn_data
    from src.models.ablations import TurnAutoencoder, AutoencoderMultiTurnClassifier
    from src.training.train import train_model

    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab("models/vocab.json")
    decision = load_encoder_decision()
    turn_encoder = load_turn_encoder(decision, vocab, device)

    train_loader, val_loader, test_loader, _ = load_multiturn_data(
        vocab, batch_size=32, data_dir=DATA_DIR,
    )

    # Phase 1: Train autoencoder on turn encodings (reconstruction objective)
    print("=== Phase 1: Training autoencoder ===")
    autoencoder = TurnAutoencoder(input_dim=32, bottleneck_dim=32).to(device)
    ae_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3, weight_decay=0.01)
    ae_criterion = nn.MSELoss()

    turn_encoder.eval()
    autoencoder.train()
    for ae_epoch in range(20):
        ae_loss_total = 0.0
        ae_count = 0
        for batch in train_loader:
            if len(batch) == 3:
                inputs, mask, labels = batch
                inputs, mask = inputs.to(device), mask.to(device)
            else:
                continue

            batch_size, max_turns, seq_len = inputs.shape
            with torch.no_grad():
                encs = []
                for t in range(max_turns):
                    enc = turn_encoder.encode(inputs[:, t, :])
                    encs.append(enc)
                encs = torch.stack(encs, dim=1)

            flat_encs = encs.view(-1, 32)
            flat_mask = mask.view(-1)
            valid = flat_encs[flat_mask > 0]

            if valid.shape[0] == 0:
                continue

            ae_optimizer.zero_grad()
            recon, _ = autoencoder(valid)
            loss = ae_criterion(recon, valid)
            loss.backward()
            ae_optimizer.step()
            ae_loss_total += loss.item() * valid.shape[0]
            ae_count += valid.shape[0]

        if ae_count > 0:
            print(f"  AE Epoch {ae_epoch+1}/20 loss: {ae_loss_total/ae_count:.6f}")

    torch.save(autoencoder.state_dict(), "models/v3_turn_autoencoder.pt")
    print("Autoencoder saved to models/v3_turn_autoencoder.pt")

    # Phase 2: Train sequence classifier with frozen autoencoder encoder
    print("\n=== Phase 2: Training sequence classifier with AE encoder ===")
    autoencoder.eval()
    model = AutoencoderMultiTurnClassifier(
        base_turn_encoder=turn_encoder, autoencoder=autoencoder,
        turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=0.01,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_model(
        model, train_loader, val_loader,
        epochs=30, iteration_name="v3_ablation_autoencoder",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=5,
        wandb_config={"group": "ablations", "tags": ["v3", "A14", "autoencoder"]},
    )


def train_ablation_mean_pool():
    """A1a: Mean pool ablation."""
    import torch.nn as nn
    from src.models.ablations import MeanPoolClassifier
    from src.training.train import train_model
    import torch

    set_global_seed(42)
    turn_encoder, train_loader, val_loader, device = _load_turn_encoder_and_data()

    model = MeanPoolClassifier(
        turn_encoder=turn_encoder, turn_encoding_dim=32,
        hidden_dim=64, dropout_rate=0.3,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=0.01,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_model(
        model, train_loader, val_loader,
        epochs=30, iteration_name="v3_ablation_mean_pool",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=5,
        wandb_config={"group": "ablations", "tags": ["v3", "A1a", "mean-pool"]},
    )


def train_ablation_max_pool():
    """A1b: Max pool ablation."""
    import torch.nn as nn
    from src.models.ablations import MaxPoolClassifier
    from src.training.train import train_model
    import torch

    set_global_seed(42)
    turn_encoder, train_loader, val_loader, device = _load_turn_encoder_and_data()

    model = MaxPoolClassifier(
        turn_encoder=turn_encoder, turn_encoding_dim=32,
        hidden_dim=64, dropout_rate=0.3,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=0.01,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_model(
        model, train_loader, val_loader,
        epochs=30, iteration_name="v3_ablation_max_pool",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=5,
        wandb_config={"group": "ablations", "tags": ["v3", "A1b", "max-pool"]},
    )


TASKS = {
    "gru_retrain": train_gru_retrain,
    "iter5": train_iter5,
    "iter6": train_iter6,
    "distilbert_hier": train_distilbert_hier,
    "distilbert_concat": train_distilbert_concat,
    "ablation_shuffled": train_ablation_shuffled,
    "ablation_reversed": train_ablation_reversed,
    "ablation_prefix": train_ablation_prefix,
    "ablation_continuation": train_ablation_continuation,
    "ablation_autoencoder": train_ablation_autoencoder,
    "ablation_mean_pool": train_ablation_mean_pool,
    "ablation_max_pool": train_ablation_max_pool,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS.keys()))
    args = parser.parse_args()

    TASKS[args.task]()
