"""Single-turn classifier gate for per-turn scoring of generated sequences.

Scores each turn independently using the retrained GRU. Rejects
sequences where any turn exceeds the threshold.
"""

import torch
from src.utils.tokenizer import load_vocab, encode_texts


class ValidationGate:
    """Gate that scores individual turns and rejects overtly malicious ones.

    Args:
        encoder: Trained single-turn model (returns logits).
        threshold: Maximum per-turn probability. Reject if any turn >= this.
        device: torch.device.
        vocab: Vocabulary dict. If None, loads from default path.
        max_len: Max token sequence length per turn.
    """

    def __init__(self, encoder=None, threshold=0.5, device=None,
                 vocab=None, max_len=256, model_path=None, vocab_path=None):
        self.threshold = threshold
        self.device = device or torch.device("cpu")
        self.vocab = vocab
        self.max_len = max_len

        if encoder is not None:
            self.encoder = encoder
        elif model_path is not None:
            self.encoder = self._load_model(model_path, vocab_path)
        else:
            raise ValueError("Either encoder or model_path must be provided")

        self.encoder.eval()

    def _load_model(self, model_path, vocab_path):
        from src.models.single_turn import GRUClassifier
        self.vocab = load_vocab(vocab_path or "models/vocab.json")
        model = GRUClassifier(
            vocab_size=len(self.vocab), embedding_dim=128, hidden_dim=64,
            dropout_rate=0.3, dense_dim=32,
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model

    def score_sequence(self, turns, return_per_turn=True):
        """Score each turn in a conversation sequence.

        Args:
            turns: List of turn text strings.
            return_per_turn: Whether to include per-turn scores.

        Returns:
            Dict with:
                passed: bool - True if all turns below threshold.
                max_score: float - highest per-turn probability.
                flagged_turn_idx: int or None - index of first flagged turn.
                per_turn_scores: list of floats (if return_per_turn).
        """
        scores = []

        with torch.no_grad():
            for i, turn_text in enumerate(turns):
                if self.vocab is not None:
                    token_ids = encode_texts(self.vocab, [turn_text], max_len=self.max_len)
                    token_ids = token_ids.to(self.device)
                    logits = self.encoder(token_ids)
                    prob = torch.sigmoid(logits).item()
                else:
                    dummy_input = torch.zeros(1, self.max_len, dtype=torch.long).to(self.device)
                    logits = self.encoder(dummy_input)
                    prob = torch.sigmoid(logits).item()

                scores.append(prob)

        max_score = max(scores)
        flagged_idx = None
        for i, s in enumerate(scores):
            if s >= self.threshold:
                flagged_idx = i
                break

        result = {
            "passed": max_score < self.threshold,
            "max_score": max_score,
            "flagged_turn_idx": flagged_idx,
        }
        if return_per_turn:
            result["per_turn_scores"] = scores

        return result

    def filter_sequences(self, sequences, threshold=None):
        """Filter a batch of sequences, returning passed and failed lists.

        Args:
            sequences: List of sequence dicts with "turns" list.
            threshold: Override threshold for this batch.

        Returns:
            Tuple of (passed_sequences, failed_sequences).
        """
        orig_threshold = self.threshold
        if threshold is not None:
            self.threshold = threshold

        passed, failed = [], []
        for seq in sequences:
            turns = [t["text"] for t in seq["turns"] if t.get("role", "user") == "user"]
            result = self.score_sequence(turns)
            seq["gate_result"] = result
            if result["passed"]:
                passed.append(seq)
            else:
                failed.append(seq)

        self.threshold = orig_threshold
        return passed, failed
