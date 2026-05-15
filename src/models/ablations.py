"""Ablation model variants for isolating temporal reasoning contribution.

A1: Matched-capacity pooling (mean, max, learned weighted-mean)
A2: Shuffled/reverse turn order
A3: Per-turn score aggregation (best-single, top-k-mean)
A4: Encoder quality gradient (random projection, early checkpoint)
A10: Turn-level voting baselines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─── A1: Matched-Capacity Pooling ───────────────────────────────

class MeanPoolClassifier(nn.Module):
    """A1a: Mean pooling over turn encodings (no LSTM).
    Matched classifier capacity to full model."""

    def __init__(self, turn_encoder, turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        self.turn_encoder = turn_encoder
        for param in self.turn_encoder.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(turn_encoding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x, mask):
        batch_size, max_turns, seq_len = x.shape
        encodings = []
        for t in range(max_turns):
            with torch.no_grad():
                enc = self.turn_encoder.encode(x[:, t, :])
            encodings.append(enc)
        encodings = torch.stack(encodings, dim=1)  # (batch, turns, dim)
        encodings = encodings * mask.unsqueeze(-1)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = encodings.sum(dim=1) / counts  # mean pool
        out = self.dropout(F.relu(self.fc1(pooled)))
        out = self.dropout(F.relu(self.fc2(out)))
        return self.fc3(out)


class MaxPoolClassifier(nn.Module):
    """A1b: Max pooling over turn encodings."""

    def __init__(self, turn_encoder, turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        self.turn_encoder = turn_encoder
        for param in self.turn_encoder.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(turn_encoding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x, mask):
        batch_size, max_turns, seq_len = x.shape
        encodings = []
        for t in range(max_turns):
            with torch.no_grad():
                enc = self.turn_encoder.encode(x[:, t, :])
            encodings.append(enc)
        encodings = torch.stack(encodings, dim=1)
        encodings = encodings * mask.unsqueeze(-1)
        encodings[mask == 0] = float('-inf')
        pooled = encodings.max(dim=1)[0]
        out = self.dropout(F.relu(self.fc1(pooled)))
        out = self.dropout(F.relu(self.fc2(out)))
        return self.fc3(out)


class LearnedWeightedMeanClassifier(nn.Module):
    """A1c: Learned per-turn weights (no cross-turn conditioning)."""

    def __init__(self, turn_encoder, turn_encoding_dim=32, hidden_dim=64,
                 max_turns=10, dropout_rate=0.3):
        super().__init__()
        self.turn_encoder = turn_encoder
        for param in self.turn_encoder.parameters():
            param.requires_grad = False
        self.turn_weights = nn.Parameter(torch.ones(max_turns) / max_turns)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(turn_encoding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x, mask):
        batch_size, max_turns, seq_len = x.shape
        encodings = []
        for t in range(max_turns):
            with torch.no_grad():
                enc = self.turn_encoder.encode(x[:, t, :])
            encodings.append(enc)
        encodings = torch.stack(encodings, dim=1)
        encodings = encodings * mask.unsqueeze(-1)
        weights = F.softmax(self.turn_weights[:max_turns] * mask + (1 - mask) * -1e9, dim=-1)
        pooled = (encodings * weights.unsqueeze(-1)).sum(dim=1)
        out = self.dropout(F.relu(self.fc1(pooled)))
        out = self.dropout(F.relu(self.fc2(out)))
        return self.fc3(out)


# ─── A4: Encoder Quality Gradient ───────────────────────────────

class RandomProjectionEncoder(nn.Module):
    """A4a: TF-IDF -> random projection to 32 dims.
    Tests whether text features matter for temporal reasoning."""

    def __init__(self, input_dim=20000, output_dim=32):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.normal_(self.projection.weight, std=1.0 / (input_dim ** 0.5))
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x):
        """Encode token IDs via bag-of-words + random projection."""
        batch_size, seq_len = x.shape
        bow = torch.zeros(batch_size, self.projection.in_features, device=x.device)
        for i in range(batch_size):
            ids = x[i][x[i] > 0]
            valid = ids[ids < self.projection.in_features]
            if len(valid) > 0:
                bow[i].scatter_add_(0, valid.long(), torch.ones_like(valid, dtype=torch.float))
        return self.projection(bow)

    def forward(self, x):
        return self.encode(x)


# ─── A10: Turn-Level Voting Baselines ────────────────────────────

class TurnLevelVoting:
    """A10: Score each turn independently, aggregate via voting.

    Not a trainable model — uses the frozen turn encoder directly.
    Threshold swept on validation set.
    """

    def __init__(self, turn_encoder, device):
        self.turn_encoder = turn_encoder
        self.device = device
        self.turn_encoder.eval()

    def score_turns(self, x, mask):
        """Score each turn independently.

        Args:
            x: (batch, max_turns, seq_len)
            mask: (batch, max_turns)

        Returns:
            (batch, max_turns) per-turn probabilities.
        """
        batch_size, max_turns, seq_len = x.shape
        scores = []
        with torch.no_grad():
            for t in range(max_turns):
                logits = self.turn_encoder(x[:, t, :])
                probs = torch.sigmoid(logits).squeeze(-1)
                scores.append(probs)
        scores = torch.stack(scores, dim=1)  # (batch, max_turns)
        return scores * mask  # zero out padding

    def predict_max_vote(self, x, mask, threshold=0.5):
        """A10a: Classify as attack if max(per-turn scores) > threshold."""
        scores = self.score_turns(x, mask)
        max_scores = scores.max(dim=1)[0]
        return (max_scores >= threshold).long(), max_scores

    def predict_mean_vote(self, x, mask, threshold=0.5):
        """A10b: Classify as attack if mean(per-turn scores) > threshold."""
        scores = self.score_turns(x, mask)
        counts = mask.sum(dim=1).clamp(min=1)
        mean_scores = scores.sum(dim=1) / counts
        return (mean_scores >= threshold).long(), mean_scores

    def predict_top_k_mean(self, x, mask, k=3, threshold=0.5):
        """A10c: Classify as attack if mean(top-k per-turn scores) > threshold."""
        scores = self.score_turns(x, mask)
        # Replace padding with -inf for topk
        scores_masked = scores.clone()
        scores_masked[mask == 0] = float('-inf')
        topk_scores = scores_masked.topk(min(k, scores.shape[1]), dim=1)[0]
        topk_scores[topk_scores == float('-inf')] = 0
        valid_k = (topk_scores > 0).sum(dim=1).clamp(min=1).float()
        mean_topk = topk_scores.sum(dim=1) / valid_k
        return (mean_topk >= threshold).long(), mean_topk
