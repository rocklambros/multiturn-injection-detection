import torch
from src.data.validation_gate import ValidationGate


def test_gate_rejects_high_scoring_turn():
    """Gate rejects a sequence where one turn scores above threshold."""

    class MockEncoder(torch.nn.Module):
        """Returns fixed probabilities per turn for testing."""
        def __init__(self, scores):
            super().__init__()
            self.scores = scores
            self.call_idx = 0

        def forward(self, x):
            score = self.scores[self.call_idx % len(self.scores)]
            self.call_idx += 1
            batch_size = x.shape[0]
            return torch.full((batch_size, 1), score)

    encoder = MockEncoder(scores=[-2.0, -1.5, 2.0, -2.0, -1.0])
    gate = ValidationGate(encoder=encoder, threshold=0.5, device=torch.device("cpu"))

    turns = ["hello", "how are you", "INJECT HERE", "thanks", "bye"]
    result = gate.score_sequence(turns)

    assert result["passed"] is False
    assert result["max_score"] >= 0.5
    assert result["flagged_turn_idx"] == 2


def test_gate_accepts_clean_sequence():
    """Gate accepts sequence where all turns score below threshold."""

    class MockEncoder(torch.nn.Module):
        def __init__(self, scores):
            super().__init__()
            self.scores = scores
            self.call_idx = 0

        def forward(self, x):
            score = self.scores[self.call_idx % len(self.scores)]
            self.call_idx += 1
            batch_size = x.shape[0]
            return torch.full((batch_size, 1), score)

    encoder = MockEncoder(scores=[-2.0, -1.5, -1.0, -2.0, -1.5])
    gate = ValidationGate(encoder=encoder, threshold=0.5, device=torch.device("cpu"))

    turns = ["hello", "how are you", "what is python", "thanks", "bye"]
    result = gate.score_sequence(turns)

    assert result["passed"] is True
    assert result["max_score"] < 0.5
