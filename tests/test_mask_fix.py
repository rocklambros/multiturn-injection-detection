import torch
from src.models.single_turn import GRUClassifier
from src.models.multi_turn import MultiTurnClassifier
from src.models.attention import MultiTurnAttentionClassifier


def _make_encoder():
    torch.manual_seed(42)
    return GRUClassifier(vocab_size=1000, embedding_dim=128, hidden_dim=64, dense_dim=32)


def test_multi_turn_mask_changes_output():
    """Masking padded turns must produce different output than not masking."""
    encoder = _make_encoder()
    encoder.eval()
    model = MultiTurnClassifier(encoder, turn_encoding_dim=32, hidden_dim=64)
    model.eval()

    torch.manual_seed(99)
    x = torch.randint(0, 1000, (2, 10, 256))

    mask_full = torch.ones(2, 10)
    mask_partial = torch.ones(2, 10)
    mask_partial[:, 5:] = 0  # last 5 turns are padding

    with torch.no_grad():
        out_full = model(x, mask_full)
        out_partial = model(x, mask_partial)

    assert not torch.allclose(out_full, out_partial, atol=1e-6), \
        "Masking padded turns should change output, but it didn't — mask is not applied"


def test_attention_mask_changes_lstm_output():
    """Attention model: masking must affect LSTM output, not just attention softmax."""
    encoder = _make_encoder()
    encoder.eval()
    model = MultiTurnAttentionClassifier(encoder, turn_encoding_dim=32, hidden_dim=64)
    model.eval()

    torch.manual_seed(99)
    x = torch.randint(0, 1000, (2, 10, 256))

    mask_full = torch.ones(2, 10)
    mask_partial = torch.ones(2, 10)
    mask_partial[:, 5:] = 0

    with torch.no_grad():
        out_full = model(x, mask_full)
        out_partial = model(x, mask_partial)

    assert not torch.allclose(out_full, out_partial, atol=1e-6), \
        "Masking padded turns should change output in attention model"
