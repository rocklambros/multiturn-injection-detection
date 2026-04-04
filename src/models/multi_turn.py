"""Iteration 5: Multi-turn sequence classifier for distributed prompt injection.

Two-level architecture: turn encoder (frozen) + sequence-level LSTM.
The turn encoder processes each turn independently.
The sequence LSTM processes the sequence of turn encodings over time.
"""

from src.utils.seed import set_global_seed
set_global_seed(42)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTurnClassifier(nn.Module):
    """Multi-turn conversation classifier with dual-encoder architecture.

    The single-turn encoder encodes each turn into a fixed-length vector.
    A second LSTM processes the sequence of turn vectors, carrying forward
    accumulated context to classify the full conversation.

    Args:
        turn_encoder: Trained single-turn model with encode() method.
        turn_encoding_dim: Output dim of turn encoder's encode() (default 32).
        hidden_dim: Sequence LSTM hidden dimension.
        max_turns: Maximum number of conversation turns.
        dropout_rate: Dropout probability.
    """

    def __init__(self, turn_encoder, turn_encoding_dim=32, hidden_dim=64,
                 max_turns=10, dropout_rate=0.3):
        super().__init__()
        self.turn_encoder = turn_encoder
        self.max_turns = max_turns

        # Freeze the turn encoder
        for param in self.turn_encoder.parameters():
            param.requires_grad = False

        self.sequence_lstm = nn.LSTM(turn_encoding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, mask):
        """Forward pass for multi-turn classification.

        Args:
            x: Token IDs, shape (batch, max_turns, seq_len).
            mask: Turn mask, shape (batch, max_turns), 1=real turn, 0=padding.

        Returns:
            Sigmoid probability, shape (batch, 1).
        """
        batch_size, max_turns, seq_len = x.shape

        # Encode each turn independently
        turn_encodings = []
        for t in range(max_turns):
            turn_input = x[:, t, :]  # (batch, seq_len)
            with torch.no_grad():
                encoding = self.turn_encoder.encode(turn_input)  # (batch, encoding_dim)
            turn_encodings.append(encoding)

        turn_encodings = torch.stack(turn_encodings, dim=1)  # (batch, max_turns, encoding_dim)
        print(f"    [Shape] Turn encodings: {turn_encodings.shape}") if not hasattr(self, '_logged') else None
        self._logged = True

        # Sequence-level LSTM
        lstm_out, (hidden, _) = self.sequence_lstm(turn_encodings)
        out = self.dropout(F.relu(self.fc1(hidden.squeeze(0))))
        return torch.sigmoid(self.fc2(self.dropout(out)))
