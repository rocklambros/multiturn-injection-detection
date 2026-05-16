"""Training orchestrator for RunPod GPU instances.

Usage:
    python scripts/run_training.py --task gru_retrain
    python scripts/run_training.py --task iter5
    python scripts/run_training.py --task iter6
    python scripts/run_training.py --task distilbert_hier
    python scripts/run_training.py --task distilbert_concat
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_global_seed


def train_gru_retrain():
    """T3.2: Retrain single-turn GRU with BCEWithLogitsLoss on full data."""
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    train_model(
        model, train_loader, val_loader,
        epochs=30, iteration_name="v2_gru_retrain",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=5,
        wandb_config={"group": "training", "tags": ["v2", "gru", "retrain"]},
    )

    import json
    decision_path = "models/encoder_decision.json"
    with open(decision_path) as f:
        decision = json.load(f)
    decision["best_single_turn_path"] = "models/v2_gru_retrain_best.pt"
    decision["v2_retrained"] = True
    with open(decision_path, "w") as f:
        json.dump(decision, f, indent=2)
    print(f"Updated {decision_path} to point to v2 retrained encoder")


def train_iter5():
    """T3.3: Retrain iter5 multi-turn (mask-fixed, new data)."""
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
        vocab, batch_size=32, data_dir="data/synthetic_v2",
    )

    model = MultiTurnClassifier(
        turn_encoder=turn_encoder, turn_encoding_dim=32,
        hidden_dim=64, dropout_rate=0.3,
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_model(
        model, train_loader, val_loader,
        epochs=30, iteration_name="v2_iter5_multiturn",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=5,
        wandb_config={"group": "training", "tags": ["v2", "iter5", "mask-fixed"]},
    )


def train_iter6():
    """T3.4: Retrain iter6 attention model (mask-fixed, new data)."""
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
        vocab, batch_size=32, data_dir="data/synthetic_v2",
    )

    model = MultiTurnAttentionClassifier(
        turn_encoder=turn_encoder, turn_encoding_dim=32,
        hidden_dim=64, dropout_rate=0.3,
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_model(
        model, train_loader, val_loader,
        epochs=30, iteration_name="v2_iter6_attention",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=5,
        wandb_config={"group": "training", "tags": ["v2", "iter6", "attention"]},
    )


def train_distilbert_hier():
    """T3.5: Train hierarchical DistilBERT baseline (PM-1a)."""
    import torch
    import torch.nn as nn
    from src.models.transformer_multiturn import HierarchicalDistilBERT
    from src.data.loader import create_distilbert_multiturn_loaders
    from src.training.train import train_model

    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = create_distilbert_multiturn_loaders(
        data_dir="data/synthetic_v2", batch_size=16, max_turns=10, max_len=128,
    )

    model = HierarchicalDistilBERT(
        num_attention_heads=4, cross_turn_layers=2,
        max_turns=10, dropout_rate=0.3, freeze_bert=True,
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_model(
        model, train_loader, val_loader,
        epochs=20, iteration_name="v2_distilbert_hier",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=5,
        wandb_config={"group": "training", "tags": ["v2", "distilbert", "hierarchical"]},
    )


def train_distilbert_concat():
    """T3.6: Train concatenated DistilBERT baseline (PM-1b)."""
    import torch
    import torch.nn as nn
    from src.models.concat_distilbert import ConcatenatedDistilBERT
    from src.data.loader import create_concat_distilbert_loaders
    from src.training.train import train_model

    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = create_concat_distilbert_loaders(
        data_dir="data/synthetic_v2", batch_size=16, max_length=512,
    )

    model = ConcatenatedDistilBERT(
        max_length=512, dropout_rate=0.3, freeze_bert=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    train_model(
        model, train_loader, val_loader,
        epochs=10, iteration_name="v2_distilbert_concat",
        optimizer=optimizer, criterion=criterion,
        device=device, patience=3,
        wandb_config={"group": "training", "tags": ["v2", "distilbert", "concat"]},
    )


TASKS = {
    "gru_retrain": train_gru_retrain,
    "iter5": train_iter5,
    "iter6": train_iter6,
    "distilbert_hier": train_distilbert_hier,
    "distilbert_concat": train_distilbert_concat,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS.keys()))
    args = parser.parse_args()

    TASKS[args.task]()
