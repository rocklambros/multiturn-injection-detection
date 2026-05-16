"""Ablation model variants for isolating temporal reasoning contribution.

A1: Matched-capacity pooling (mean, max, learned weighted-mean)
A2: Shuffled/reverse turn order
A3: Per-turn score aggregation (best-single, top-k-mean)
A4: Encoder quality gradient (random projection, early checkpoint)
A10: Turn-level voting baselines
A12: Prefix-only (turns 1..K)
A13: Continuation-only (turns K+1..N)
A14: Autoencoder encoder control
B6: Cosine-similarity discontinuity baseline
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
        scores_masked = scores.clone()
        scores_masked[mask == 0] = float('-inf')
        topk_scores = scores_masked.topk(min(k, scores.shape[1]), dim=1)[0]
        valid_k = (topk_scores > float('-inf')).sum(dim=1).clamp(min=1).float()
        topk_scores = topk_scores.clamp(min=0)
        mean_topk = topk_scores.sum(dim=1) / valid_k
        return (mean_topk >= threshold).long(), mean_topk


# ─── A2: Shuffled/Reversed Turn Order ──────────────────────────────

class ShuffledTurnsClassifier(nn.Module):
    """A2a: Randomly permute valid turns before LSTM. If temporal order matters,
    this should degrade performance vs the full model."""

    def __init__(self, turn_encoder, turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        self.turn_encoder = turn_encoder
        for param in self.turn_encoder.parameters():
            param.requires_grad = False
        self.sequence_lstm = nn.LSTM(turn_encoding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, mask):
        batch_size, max_turns, seq_len = x.shape
        encodings = []
        for t in range(max_turns):
            with torch.no_grad():
                enc = self.turn_encoder.encode(x[:, t, :])
            encodings.append(enc)
        encodings = torch.stack(encodings, dim=1)
        encodings = encodings * mask.unsqueeze(-1)

        # Shuffle valid turns independently per sample
        shuffled = encodings.clone()
        for i in range(batch_size):
            n_valid = int(mask[i].sum().item())
            if n_valid > 1:
                perm = torch.randperm(n_valid, device=x.device)
                shuffled[i, :n_valid] = encodings[i, perm]

        lstm_out, (hidden, _) = self.sequence_lstm(shuffled)
        out = self.dropout(F.relu(self.fc1(hidden.squeeze(0))))
        return self.fc2(self.dropout(out))


class ReversedTurnsClassifier(nn.Module):
    """A2b: Reverse valid turn order before LSTM."""

    def __init__(self, turn_encoder, turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        self.turn_encoder = turn_encoder
        for param in self.turn_encoder.parameters():
            param.requires_grad = False
        self.sequence_lstm = nn.LSTM(turn_encoding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, mask):
        batch_size, max_turns, seq_len = x.shape
        encodings = []
        for t in range(max_turns):
            with torch.no_grad():
                enc = self.turn_encoder.encode(x[:, t, :])
            encodings.append(enc)
        encodings = torch.stack(encodings, dim=1)
        encodings = encodings * mask.unsqueeze(-1)

        reversed_enc = encodings.clone()
        for i in range(batch_size):
            n_valid = int(mask[i].sum().item())
            if n_valid > 1:
                reversed_enc[i, :n_valid] = encodings[i, :n_valid].flip(0)

        lstm_out, (hidden, _) = self.sequence_lstm(reversed_enc)
        out = self.dropout(F.relu(self.fc1(hidden.squeeze(0))))
        return self.fc2(self.dropout(out))


# ─── A12: Prefix-Only (turns 1..K) ─────────────────────────────────

class PrefixOnlyClassifier(nn.Module):
    """A12: Feed only the shared-prefix turns (1..K) to LSTM.
    If the model relies on continuation turns for detection, this should fail.
    Requires k_values tensor indicating split point per sample."""

    def __init__(self, turn_encoder, turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        self.turn_encoder = turn_encoder
        for param in self.turn_encoder.parameters():
            param.requires_grad = False
        self.sequence_lstm = nn.LSTM(turn_encoding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, mask, k_values=None):
        batch_size, max_turns, seq_len = x.shape
        encodings = []
        for t in range(max_turns):
            with torch.no_grad():
                enc = self.turn_encoder.encode(x[:, t, :])
            encodings.append(enc)
        encodings = torch.stack(encodings, dim=1)

        # Zero out turns after K (prefix only)
        prefix_mask = torch.zeros_like(mask)
        if k_values is not None:
            for i in range(batch_size):
                k = int(k_values[i].item())
                prefix_mask[i, :k] = mask[i, :k]
        else:
            n_valid = mask.sum(dim=1)
            for i in range(batch_size):
                k = max(1, int(n_valid[i].item()) // 2)
                prefix_mask[i, :k] = mask[i, :k]

        encodings = encodings * prefix_mask.unsqueeze(-1)
        lstm_out, (hidden, _) = self.sequence_lstm(encodings)
        out = self.dropout(F.relu(self.fc1(hidden.squeeze(0))))
        return self.fc2(self.dropout(out))


# ─── A13: Continuation-Only (turns K+1..N) ─────────────────────────

class ContinuationOnlyClassifier(nn.Module):
    """A13: Feed only continuation turns (K+1..N) to LSTM.
    Tests whether detection signal lives entirely in the continuation."""

    def __init__(self, turn_encoder, turn_encoding_dim=32, hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        self.turn_encoder = turn_encoder
        for param in self.turn_encoder.parameters():
            param.requires_grad = False
        self.sequence_lstm = nn.LSTM(turn_encoding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, mask, k_values=None):
        batch_size, max_turns, seq_len = x.shape
        encodings = []
        for t in range(max_turns):
            with torch.no_grad():
                enc = self.turn_encoder.encode(x[:, t, :])
            encodings.append(enc)
        encodings = torch.stack(encodings, dim=1)

        cont_mask = torch.zeros_like(mask)
        if k_values is not None:
            for i in range(batch_size):
                k = int(k_values[i].item())
                cont_mask[i, k:] = mask[i, k:]
        else:
            n_valid = mask.sum(dim=1)
            for i in range(batch_size):
                k = max(1, int(n_valid[i].item()) // 2)
                cont_mask[i, k:] = mask[i, k:]

        encodings = encodings * cont_mask.unsqueeze(-1)
        lstm_out, (hidden, _) = self.sequence_lstm(encodings)
        out = self.dropout(F.relu(self.fc1(hidden.squeeze(0))))
        return self.fc2(self.dropout(out))


# ─── A14: Autoencoder Encoder Control ──────────────────────────────

class TurnAutoencoder(nn.Module):
    """A14: Autoencoder trained to reconstruct turn encodings.
    The bottleneck captures general text features without classification bias.
    Use the encoder portion as an alternative turn encoder for the sequence LSTM."""

    def __init__(self, input_dim=32, bottleneck_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


class AutoencoderMultiTurnClassifier(nn.Module):
    """Sequence LSTM using autoencoder bottleneck instead of GRU's encode().
    Distinguishes temporal reasoning from injection-score aggregation."""

    def __init__(self, base_turn_encoder, autoencoder, turn_encoding_dim=32,
                 hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        self.base_turn_encoder = base_turn_encoder
        self.autoencoder = autoencoder
        for param in self.base_turn_encoder.parameters():
            param.requires_grad = False
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.sequence_lstm = nn.LSTM(turn_encoding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, mask):
        batch_size, max_turns, seq_len = x.shape
        encodings = []
        for t in range(max_turns):
            with torch.no_grad():
                raw_enc = self.base_turn_encoder.encode(x[:, t, :])
                ae_enc = self.autoencoder.encode(raw_enc)
            encodings.append(ae_enc)
        encodings = torch.stack(encodings, dim=1)
        encodings = encodings * mask.unsqueeze(-1)

        lstm_out, (hidden, _) = self.sequence_lstm(encodings)
        out = self.dropout(F.relu(self.fc1(hidden.squeeze(0))))
        return self.fc2(self.dropout(out))


# ─── B6: Cosine-Similarity Discontinuity Baseline ──────────────────

class CosineSimilarityBaseline:
    """B6: Detect attacks via cosine-similarity discontinuity between consecutive turns.
    Non-neural baseline using TF-IDF features + logistic regression on
    similarity features (min, mean, max discontinuity)."""

    def __init__(self):
        self.vectorizer = None
        self.clf = None

    def _extract_features(self, sequences):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            all_turns = []
            for seq in sequences:
                turns = seq.get("turns", [])
                for t in turns:
                    text = t.get("text", "") if isinstance(t, dict) else str(t)
                    all_turns.append(text)
            self.vectorizer.fit(all_turns)

        features = []
        for seq in sequences:
            turns = seq.get("turns", [])
            texts = []
            for t in turns:
                text = t.get("text", "") if isinstance(t, dict) else str(t)
                if text.strip():
                    texts.append(text)
            if len(texts) < 2:
                features.append([0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            vecs = self.vectorizer.transform(texts)
            sims = []
            for i in range(len(texts) - 1):
                sim = cos_sim(vecs[i:i+1], vecs[i+1:i+2])[0, 0]
                sims.append(sim)

            sims = np.array(sims)
            diffs = np.abs(np.diff(sims)) if len(sims) > 1 else np.array([0.0])
            features.append([
                float(np.min(sims)),
                float(np.mean(sims)),
                float(np.max(diffs)) if len(diffs) > 0 else 0.0,
                float(np.std(sims)),
                float(sims[-1] - sims[0]) if len(sims) > 1 else 0.0,
            ])

        return np.array(features)

    def fit(self, train_sequences):
        from sklearn.linear_model import LogisticRegression

        X = self._extract_features(train_sequences)
        y = np.array([s["label"] for s in train_sequences])
        self.clf = LogisticRegression(max_iter=1000, random_state=42)
        self.clf.fit(X, y)

    def predict(self, sequences):
        X = self._extract_features(sequences)
        return self.clf.predict(X), self.clf.predict_proba(X)[:, 1]
