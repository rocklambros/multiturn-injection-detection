"""Configuration system for all model iterations."""

from dataclasses import dataclass


@dataclass
class IterationConfig:
    """Configuration for a single model iteration.

    Args:
        name: Unique iteration identifier (e.g., 'iter1_lstm').
        model_type: One of 'baseline_lr', 'baseline_rf', 'lstm', 'bilstm',
                     'gru', 'multiturn', 'multiturn_attn'.
        embedding_dim: Dimension of word embeddings.
        embedding_type: 'random' or 'glove'.
        hidden_dim: LSTM/GRU hidden state dimension.
        bidirectional: Whether to use bidirectional encoder.
        dropout_rate: Dropout probability.
        dense_dim: Dense layer dimension before output.
        batch_size: Training batch size.
        epochs: Maximum training epochs.
        learning_rate: Initial learning rate for Adam.
        early_stopping_patience: Epochs without improvement before stopping.
        max_sequence_length: Maximum token sequence length per turn.
        max_turns: Maximum conversation turns (multi-turn only).
        freeze_encoder: Whether to freeze turn encoder (multi-turn only).
        threshold: Classification threshold for binary prediction.
    """
    name: str
    model_type: str
    embedding_dim: int = 128
    embedding_type: str = "random"
    hidden_dim: int = 64
    bidirectional: bool = False
    dropout_rate: float = 0.0
    dense_dim: int = 32
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.001
    early_stopping_patience: int = 3
    max_sequence_length: int = 256
    max_turns: int = 10
    freeze_encoder: bool = True
    threshold: float = 0.5


ITERATIONS = {
    "iter0_baseline_lr": IterationConfig(
        name="iter0_baseline_lr",
        model_type="baseline_lr",
    ),
    "iter0_baseline_rf": IterationConfig(
        name="iter0_baseline_rf",
        model_type="baseline_rf",
    ),
    "iter1_lstm": IterationConfig(
        name="iter1_lstm",
        model_type="lstm",
        embedding_dim=128,
        hidden_dim=64,
        epochs=20,
        early_stopping_patience=3,
    ),
    "iter2_lstm_glove": IterationConfig(
        name="iter2_lstm_glove",
        model_type="lstm",
        embedding_dim=100,
        embedding_type="glove",
        hidden_dim=64,
        epochs=20,
        early_stopping_patience=3,
    ),
    "iter3_bilstm_dropout": IterationConfig(
        name="iter3_bilstm_dropout",
        model_type="bilstm",
        embedding_dim=128,
        hidden_dim=64,
        bidirectional=True,
        dropout_rate=0.3,
        epochs=30,
        early_stopping_patience=5,
    ),
    "iter4_gru": IterationConfig(
        name="iter4_gru",
        model_type="gru",
        embedding_dim=128,
        hidden_dim=64,
        bidirectional=True,
        dropout_rate=0.3,
        epochs=30,
        early_stopping_patience=5,
    ),
    "iter5_multiturn": IterationConfig(
        name="iter5_multiturn",
        model_type="multiturn",
        hidden_dim=64,
        dropout_rate=0.3,
        batch_size=32,
        epochs=30,
        early_stopping_patience=5,
    ),
    "iter6_attention": IterationConfig(
        name="iter6_attention",
        model_type="multiturn_attn",
        hidden_dim=64,
        dropout_rate=0.3,
        batch_size=32,
        epochs=30,
        early_stopping_patience=5,
    ),
    "iter7_threshold": IterationConfig(
        name="iter7_threshold",
        model_type="multiturn_attn",
        hidden_dim=64,
        dropout_rate=0.3,
        batch_size=32,
    ),
    "iter3_bilstm_dropout_05": IterationConfig(
        name="iter3_bilstm_dropout_05",
        model_type="bilstm",
        embedding_dim=128,
        hidden_dim=64,
        bidirectional=True,
        dropout_rate=0.5,
        epochs=30,
        early_stopping_patience=5,
    ),
}
