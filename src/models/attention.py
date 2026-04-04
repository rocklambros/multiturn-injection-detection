"""Iteration 6: Multi-turn classifier with turn-level attention mechanism.

Adds attention over LSTM hidden states to identify which turns
contribute most to the classification decision.
"""

from src.utils.seed import set_global_seed
set_global_seed(42)

import torch
import torch.nn as nn
import torch.nn.functional as F


class TurnAttention(nn.Module):
    """Additive attention over turn-level LSTM outputs.

    Args:
        hidden_dim: LSTM hidden state dimension.
        attention_dim: Internal attention projection dimension.
    """

    def __init__(self, hidden_dim, attention_dim=32):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.V = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, lstm_outputs, mask=None):
        """Compute attention weights over turns.

        Args:
            lstm_outputs: LSTM hidden states, shape (batch, max_turns, hidden_dim).
            mask: Turn mask, shape (batch, max_turns), 1=real, 0=pad.

        Returns:
            Tuple of (context, attention_weights) where:
                context: Weighted sum, shape (batch, hidden_dim).
                attention_weights: Softmax weights, shape (batch, max_turns).
        """
        scores = self.V(torch.tanh(self.W(lstm_outputs)))  # (batch, max_turns, 1)
        scores = scores.squeeze(-1)  # (batch, max_turns)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)  # (batch, max_turns)
        context = torch.bmm(
            attention_weights.unsqueeze(1), lstm_outputs
        ).squeeze(1)  # (batch, hidden_dim)

        return context, attention_weights


class MultiTurnAttentionClassifier(nn.Module):
    """Multi-turn classifier with attention over turn encodings.

    Uses full LSTM output sequence (not just final hidden) and applies
    attention to create a weighted context vector for classification.
    Returns attention weights for visualization.

    Args:
        turn_encoder: Trained single-turn model with encode() method.
        turn_encoding_dim: Output dim of turn encoder's encode() (default 32).
        hidden_dim: Sequence LSTM hidden dimension.
        attention_dim: Attention projection dimension.
        max_turns: Maximum number of conversation turns.
        dropout_rate: Dropout probability.
    """

    def __init__(self, turn_encoder, turn_encoding_dim=32, hidden_dim=64,
                 attention_dim=32, max_turns=10, dropout_rate=0.3):
        super().__init__()
        self.turn_encoder = turn_encoder
        self.max_turns = max_turns
        self._return_attention = False

        # Freeze the turn encoder
        for param in self.turn_encoder.parameters():
            param.requires_grad = False

        self.sequence_lstm = nn.LSTM(turn_encoding_dim, hidden_dim, batch_first=True)
        self.attention = TurnAttention(hidden_dim, attention_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, mask):
        """Forward pass for multi-turn classification with attention.

        Args:
            x: Token IDs, shape (batch, max_turns, seq_len).
            mask: Turn mask, shape (batch, max_turns), 1=real, 0=pad.

        Returns:
            If self._return_attention is False:
                Sigmoid probability, shape (batch, 1).
            If self._return_attention is True:
                Tuple of (probability, attention_weights).
        """
        batch_size, max_turns, seq_len = x.shape

        # Encode each turn
        turn_encodings = []
        for t in range(max_turns):
            turn_input = x[:, t, :]
            with torch.no_grad():
                encoding = self.turn_encoder.encode(turn_input)
            turn_encodings.append(encoding)

        turn_encodings = torch.stack(turn_encodings, dim=1)  # (batch, max_turns, encoding_dim)

        # Sequence LSTM — use full output for attention
        lstm_out, _ = self.sequence_lstm(turn_encodings)  # (batch, max_turns, hidden_dim)

        # Attention
        context, attention_weights = self.attention(lstm_out, mask)  # (batch, hidden_dim)

        out = self.dropout(F.relu(self.fc1(context)))
        prob = torch.sigmoid(self.fc2(self.dropout(out)))

        if self._return_attention:
            return prob, attention_weights
        return prob
