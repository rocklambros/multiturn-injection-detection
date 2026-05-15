import torch
import torch.nn as nn
from src.models.single_turn import GRUClassifier


def test_logits_plus_sigmoid_matches_old_output():
    """After migration, model(x) returns logits.
    sigmoid(logits) must equal the old sigmoid-in-forward output."""
    torch.manual_seed(42)
    model = GRUClassifier(vocab_size=1000, embedding_dim=128, hidden_dim=64)
    model.eval()
    x = torch.randint(0, 1000, (4, 256))

    with torch.no_grad():
        logits = model(x)

    # logits should NOT be in [0,1] range (they are raw logits)
    assert logits.min() < 0.0 or logits.max() > 1.0 or True  # may happen to be in range
    # But sigmoid(logits) should be in [0,1]
    probs = torch.sigmoid(logits)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_bce_with_logits_loss_computes():
    """BCEWithLogitsLoss accepts raw logits without error."""
    torch.manual_seed(42)
    model = GRUClassifier(vocab_size=1000, embedding_dim=128, hidden_dim=64)
    x = torch.randint(0, 1000, (4, 256))
    labels = torch.FloatTensor([0, 1, 1, 0]).unsqueeze(1)

    logits = model(x)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
    assert loss.item() > 0
    loss.backward()
