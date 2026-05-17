"""PyTorch Dataset and DataLoader classes for single-turn and multi-turn data.

Single-turn shape: (batch_size, 256)
Multi-turn shape: (batch_size, 10, 256) with mask (batch_size, 10)
"""

import json
from pathlib import Path


def load_sequences(filepath, exclude_method=None, only_method=None):
    """Load sequences from JSON with optional generation-method filtering.

    Args:
        filepath: Path to multiturn_{split}.json.
        exclude_method: If set, drop sequences with this generation_method.
        only_method: If set, keep only sequences with this generation_method.

    Returns:
        List of sequence dicts.
    """
    with open(filepath) as f:
        seqs = json.load(f)
    if only_method:
        seqs = [s for s in seqs if s.get("generation_method") == only_method]
    elif exclude_method:
        seqs = [s for s in seqs if s.get("generation_method") != exclude_method]
    return seqs

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertTokenizer

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
        turns_list = [[turn["text"] for turn in seq["turns"] if turn.get("role", "user") == "user"] for seq in data]
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


class DistilBertMultiTurnDataset(Dataset):
    """Dataset for hierarchical DistilBERT: tokenizes each turn independently.

    Args:
        sequences: List of dicts with 'turns' (list of {'text': str}) and 'label' (int).
        tokenizer: DistilBertTokenizer instance.
        max_turns: Maximum number of turns to keep per conversation.
        max_len: Maximum token length per turn.

    Returns per __getitem__:
        Tuple of (input_ids, attention_mask, turn_mask, label).
    """

    def __init__(self, sequences, tokenizer, max_turns=10, max_len=128):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        turns = [t["text"] for t in seq["turns"] if t.get("role", "user") == "user"][:self.max_turns]
        label = seq["label"]

        input_ids = torch.zeros(self.max_turns, self.max_len, dtype=torch.long)
        attention_mask = torch.zeros(self.max_turns, self.max_len, dtype=torch.long)
        turn_mask = torch.zeros(self.max_turns, dtype=torch.float)

        for i, turn_text in enumerate(turns):
            enc = self.tokenizer(
                turn_text, max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            input_ids[i] = enc["input_ids"].squeeze(0)
            attention_mask[i] = enc["attention_mask"].squeeze(0)
            turn_mask[i] = 1.0

        return input_ids, attention_mask, turn_mask, torch.tensor(label, dtype=torch.float)


class ConcatDistilBertDataset(Dataset):
    """Dataset for concatenated DistilBERT: joins all turns with [SEP].

    Args:
        sequences: List of dicts with 'turns' (list of {'text': str}) and 'label' (int).
        tokenizer: DistilBertTokenizer instance.
        max_length: Maximum total token length for the concatenated input.

    Returns per __getitem__:
        Tuple of (input_ids, attention_mask, label).
    """

    def __init__(self, sequences, tokenizer, max_length=512):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        turns = [t["text"] for t in seq["turns"] if t.get("role", "user") == "user"]
        label = seq["label"]
        combined = " [SEP] ".join(turns)
        enc = self.tokenizer(
            combined, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(label, dtype=torch.float),
        )


def create_distilbert_multiturn_loaders(data_dir="data/synthetic_v2",
                                         batch_size=16, max_turns=10,
                                         max_len=128, num_workers=2):
    """Create DataLoaders for hierarchical DistilBERT.

    Args:
        data_dir: Directory with multiturn_{train,val,test}.json.
        batch_size: Batch size.
        max_turns: Maximum conversation turns.
        max_len: Maximum token length per turn.
        num_workers: DataLoader workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Side effects:
        Downloads distilbert-base-uncased tokenizer on first call.
        Prints dataset sizes per split.
    """
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    loaders = {}
    for split in ["train", "val", "test"]:
        with open(f"{data_dir}/multiturn_{split}.json") as f:
            data = json.load(f)
        dataset = DistilBertMultiTurnDataset(data, tokenizer, max_turns, max_len)
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=True,
        )
        print(f"DistilBERT hierarchical {split}: {len(data)} sequences")
    return loaders["train"], loaders["val"], loaders["test"]


def create_concat_distilbert_loaders(data_dir="data/synthetic_v2",
                                      batch_size=16, max_length=512,
                                      num_workers=2):
    """Create DataLoaders for concatenated DistilBERT.

    Args:
        data_dir: Directory with multiturn_{train,val,test}.json.
        batch_size: Batch size.
        max_length: Maximum total token length for concatenated input.
        num_workers: DataLoader workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Side effects:
        Downloads distilbert-base-uncased tokenizer on first call.
        Prints dataset sizes per split.
    """
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    loaders = {}
    for split in ["train", "val", "test"]:
        with open(f"{data_dir}/multiturn_{split}.json") as f:
            data = json.load(f)
        dataset = ConcatDistilBertDataset(data, tokenizer, max_length)
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=True,
        )
        print(f"DistilBERT concat {split}: {len(data)} sequences")
    return loaders["train"], loaders["val"], loaders["test"]


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
