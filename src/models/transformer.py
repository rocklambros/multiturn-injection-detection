"""Transformer-based models for prompt injection detection.

Implements two approaches per instructor feedback:
1. Custom small Transformer encoder (controlled comparison with LSTM/GRU)
2. DistilBERT fine-tuned classifier (transfer learning comparison)

Uses Chollet heuristic (Chapter 11/15) to frame when each approach is appropriate:
- Ratio = training_samples / mean_words_per_sample
- Below 1,500: bag-of-bigrams tends to win
- Above 1,500: sequence/transformer models tend to win
"""

from src.utils.seed import set_global_seed
set_global_seed(42)

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input embeddings.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model) with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """Custom small Transformer encoder for binary classification.

    Controlled comparison with LSTM/GRU: same vocab, same embeddings,
    similar parameter count. Uses multi-head self-attention instead of
    recurrent gates.

    Args:
        vocab_size: Size of vocabulary.
        embedding_dim: Embedding dimension (also d_model for transformer).
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: FFN hidden dimension in each layer.
        dropout: Dropout rate.
        max_len: Maximum sequence length.
    """

    def __init__(self, vocab_size=20000, embedding_dim=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.3, max_len=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, seq_len) token IDs.

        Returns:
            (batch, 1) probabilities.
        """
        # Create padding mask (True = ignore)
        padding_mask = (x == 0)

        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Global average pooling over non-padded positions
        mask = (~padding_mask).float().unsqueeze(-1)  # (batch, seq, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.classifier(x)


class DistilBERTClassifier(nn.Module):
    """DistilBERT fine-tuned for binary prompt injection classification.

    Uses HuggingFace DistilBERT with frozen body and trainable classification head.
    This demonstrates transfer learning: pretrained language understanding
    applied to the security domain.

    Args:
        freeze_body: If True, freeze DistilBERT body (only train head).
        dropout: Dropout rate for classification head.
    """

    def __init__(self, freeze_body=True, dropout=0.3):
        super().__init__()
        from transformers import DistilBertModel
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        if freeze_body:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask=None):
        """Forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs from DistilBERT tokenizer.
            attention_mask: (batch, seq_len) attention mask.

        Returns:
            (batch, 1) probabilities.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token representation
        cls_output = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_output)
