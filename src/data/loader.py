"""PyTorch Dataset and DataLoader classes for single-turn and multi-turn data.

Single-turn shape: (batch_size, 256)
Multi-turn shape: (batch_size, 10, 256) with mask (batch_size, 10)
"""

from src.utils.seed import set_global_seed
set_global_seed(42)

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.tokenizer import load_vocab, encode_texts, encode_multiturn


class SingleTurnDataset(Dataset):
    """Dataset for single-turn prompt injection classification.

    Args:
        token_ids: torch.LongTensor of shape (N, max_seq_len).
        labels: torch.LongTensor or torch.FloatTensor of shape (N,).

    Returns per __getitem__:
        Tuple of (token_ids[i], label[i]).
    """

    def __init__(self, token_ids, labels):
        self.token_ids = token_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.labels[idx]


class MultiTurnDataset(Dataset):
    """Dataset for multi-turn conversation classification.

    Args:
        token_ids: torch.LongTensor of shape (N, max_turns, max_seq_len).
        masks: torch.FloatTensor of shape (N, max_turns).
        labels: torch.LongTensor or torch.FloatTensor of shape (N,).

    Returns per __getitem__:
        Tuple of (token_ids[i], mask[i], label[i]).
    """

    def __init__(self, token_ids, masks, labels):
        self.token_ids = token_ids
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.masks[idx], self.labels[idx]


def create_single_turn_loaders(vocab_path="models/vocab.json",
                                data_dir="data/processed",
                                batch_size=64,
                                max_len=256,
                                num_workers=2):
    """Create DataLoaders for single-turn train/val/test sets.

    Args:
        vocab_path: Path to vocab JSON.
        data_dir: Directory with single_turn_{train,val,test}.csv.
        batch_size: Batch size.
        max_len: Max sequence length.
        num_workers: DataLoader workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, vocab).

    Side effects:
        Prints dataset sizes and batch shapes.
    """
    vocab = load_vocab(vocab_path)

    loaders = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"{data_dir}/single_turn_{split}.csv")
        print(f"\n{split}: {len(df)} samples")

        token_ids = encode_texts(vocab, df["text"].tolist(), max_len=max_len)
        labels = torch.FloatTensor(df["label"].values)
        print(f"  token_ids shape: {token_ids.shape}")
        print(f"  labels shape: {labels.shape}")

        dataset = SingleTurnDataset(token_ids, labels)
        shuffle = (split == "train")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders[split] = loader

    # Print first batch shape
    batch = next(iter(loaders["train"]))
    print(f"\nFirst train batch: token_ids={batch[0].shape}, labels={batch[1].shape}")

    return loaders["train"], loaders["val"], loaders["test"], vocab


def create_multi_turn_loaders(vocab_path="models/vocab.json",
                               data_dir="data/synthetic",
                               batch_size=32,
                               max_turns=10,
                               max_len=256,
                               num_workers=2):
    """Create DataLoaders for multi-turn train/val/test sets.

    Args:
        vocab_path: Path to vocab JSON.
        data_dir: Directory with multiturn_{train,val,test}.json.
        batch_size: Batch size.
        max_turns: Max conversation turns.
        max_len: Max tokens per turn.
        num_workers: DataLoader workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, vocab).

    Side effects:
        Prints dataset sizes and batch shapes.
    """
    vocab = load_vocab(vocab_path)

    loaders = {}
    for split in ["train", "val", "test"]:
        with open(f"{data_dir}/multiturn_{split}.json") as f:
            data = json.load(f)
        print(f"\n{split}: {len(data)} sequences")

        # Extract turns and labels
        turns_list = [[turn["text"] for turn in seq["turns"]] for seq in data]
        labels_list = [seq["label"] for seq in data]

        token_ids, masks = encode_multiturn(vocab, turns_list, max_turns=max_turns, max_len=max_len)
        labels = torch.FloatTensor(labels_list)
        print(f"  token_ids shape: {token_ids.shape}")
        print(f"  masks shape: {masks.shape}")
        print(f"  labels shape: {labels.shape}")

        dataset = MultiTurnDataset(token_ids, masks, labels)
        shuffle = (split == "train")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders[split] = loader

    # Print first batch shape
    batch = next(iter(loaders["train"]))
    print(f"\nFirst train batch: token_ids={batch[0].shape}, mask={batch[1].shape}, labels={batch[2].shape}")

    return loaders["train"], loaders["val"], loaders["test"], vocab


if __name__ == "__main__":
    print("Testing SingleTurnDataset:")
    ds = SingleTurnDataset(torch.randint(0, 100, (50, 256)), torch.randint(0, 2, (50,)))
    print(f"  Sample: {ds[0][0].shape}, {ds[0][1].shape}")

    print("\nTesting MultiTurnDataset:")
    ds = MultiTurnDataset(
        torch.randint(0, 100, (20, 10, 256)),
        torch.ones(20, 10),
        torch.randint(0, 2, (20,)),
    )
    print(f"  Sample: {ds[0][0].shape}, {ds[0][1].shape}, {ds[0][2].shape}")
