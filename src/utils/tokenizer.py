"""Custom vocabulary and tokenization for prompt injection detection.

Builds vocabulary from training data only (no data leakage).
Supports single-turn and multi-turn encoding.
"""

from src.utils.seed import set_global_seed
set_global_seed(42)

import json
import re
from collections import Counter
from pathlib import Path

import torch


TOKENIZER_CONFIG = {
    "max_vocab_size": 20000,
    "max_sequence_length": 256,
    "oov_token": "<OOV>",
    "pad_token": "<PAD>",
    "padding": "post",
    "truncating": "post",
}


def tokenize_text(text):
    """Simple whitespace + punctuation tokenizer.

    Args:
        text: Input string.

    Returns:
        List of lowercase tokens.
    """
    text = text.lower().strip()
    tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
    return tokens


def build_vocab(train_texts, max_vocab_size=20000):
    """Build vocabulary from training texts.

    Args:
        train_texts: List of training text strings.
        max_vocab_size: Maximum vocabulary size including special tokens.

    Returns:
        Dict mapping word -> integer index. PAD=0, OOV=1.

    Side effects:
        Prints vocab statistics.
    """
    # Count all tokens
    counter = Counter()
    for text in train_texts:
        tokens = tokenize_text(str(text))
        counter.update(tokens)

    print(f"Total unique tokens: {len(counter)}")

    # Reserve indices 0 and 1 for special tokens
    vocab = {"<PAD>": 0, "<OOV>": 1}

    # Add most common tokens up to max_vocab_size
    for word, count in counter.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)

    print(f"Vocab size (with special tokens): {len(vocab)}")
    print(f"Top 10 tokens: {counter.most_common(10)}")

    return vocab


def encode_texts(vocab, texts, max_len=256):
    """Encode texts to padded integer sequences.

    Args:
        vocab: Word-to-index dict from build_vocab.
        texts: List of text strings.
        max_len: Maximum sequence length (post-truncation/padding).

    Returns:
        torch.LongTensor of shape (len(texts), max_len).

    Side effects:
        Prints output shape.
    """
    oov_idx = vocab.get("<OOV>", 1)
    encoded = []

    for text in texts:
        tokens = tokenize_text(str(text))
        ids = [vocab.get(t, oov_idx) for t in tokens]

        # Truncate
        if len(ids) > max_len:
            ids = ids[:max_len]

        # Pad
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))

        encoded.append(ids)

    result = torch.LongTensor(encoded)
    print(f"Encoded shape: {result.shape}")
    return result


def encode_multiturn(vocab, turns_list, max_turns=10, max_len=256):
    """Encode multi-turn conversations to 3D padded tensor.

    Args:
        vocab: Word-to-index dict from build_vocab.
        turns_list: List of conversations, each a list of turn strings.
        max_turns: Maximum number of turns per conversation.
        max_len: Maximum tokens per turn.

    Returns:
        Tuple of (token_ids, mask) where:
            token_ids: torch.LongTensor of shape (len(turns_list), max_turns, max_len)
            mask: torch.FloatTensor of shape (len(turns_list), max_turns), 1=real, 0=pad

    Side effects:
        Prints output shapes.
    """
    oov_idx = vocab.get("<OOV>", 1)
    all_ids = []
    all_masks = []

    for conversation in turns_list:
        conv_ids = []
        mask = []

        for t in range(max_turns):
            if t < len(conversation):
                tokens = tokenize_text(str(conversation[t]))
                ids = [vocab.get(tok, oov_idx) for tok in tokens]
                # Truncate
                if len(ids) > max_len:
                    ids = ids[:max_len]
                # Pad
                if len(ids) < max_len:
                    ids = ids + [0] * (max_len - len(ids))
                conv_ids.append(ids)
                mask.append(1.0)
            else:
                conv_ids.append([0] * max_len)
                mask.append(0.0)

        all_ids.append(conv_ids)
        all_masks.append(mask)

    token_ids = torch.LongTensor(all_ids)
    masks = torch.FloatTensor(all_masks)
    print(f"Multi-turn token_ids shape: {token_ids.shape}")
    print(f"Multi-turn mask shape: {masks.shape}")
    return token_ids, masks


def save_vocab(vocab, path="models/vocab.json"):
    """Save vocabulary to JSON file.

    Args:
        path: Output path for vocab JSON.

    Side effects:
        Creates file at path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocab saved to {path}")


def load_vocab(path="models/vocab.json"):
    """Load vocabulary from JSON file.

    Args:
        path: Path to vocab JSON.

    Returns:
        Dict mapping word -> integer index.
    """
    with open(path) as f:
        vocab = json.load(f)
    print(f"Vocab loaded from {path}: {len(vocab)} tokens")
    return vocab


def compute_oov_rate(vocab, texts):
    """Compute OOV rate for a set of texts.

    Args:
        vocab: Word-to-index dict.
        texts: List of text strings.

    Returns:
        Float: fraction of tokens that are OOV.

    Side effects:
        Prints OOV statistics.
    """
    total = 0
    oov = 0
    for text in texts:
        tokens = tokenize_text(str(text))
        for t in tokens:
            total += 1
            if t not in vocab:
                oov += 1

    rate = oov / total if total > 0 else 0
    print(f"OOV rate: {oov}/{total} = {rate*100:.2f}%")
    return rate


if __name__ == "__main__":
    import pandas as pd

    train = pd.read_csv("data/processed/single_turn_train.csv")
    val = pd.read_csv("data/processed/single_turn_val.csv")

    vocab = build_vocab(train["text"].tolist())
    save_vocab(vocab)

    # Test encoding
    sample = encode_texts(vocab, train["text"][:5].tolist())
    print(f"Sample encoded: {sample[0][:20]}")

    # OOV rates
    print("\nTraining set:")
    compute_oov_rate(vocab, train["text"].tolist())
    print("Validation set:")
    compute_oov_rate(vocab, val["text"].tolist())
