"""Single-turn prompt injection classifiers: LSTM, BiLSTM, GRU.

Iterations 1-4 from PRD Section 2.4.
"""

from src.utils.seed import set_global_seed
set_global_seed(42)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTurnLSTM(nn.Module):
    """Iteration 1: Simple LSTM with random embeddings.

    Args:
        vocab_size: Size of vocabulary.
        embedding_dim: Embedding dimension (128 for random, 100 for GloVe).
        hidden_dim: LSTM hidden state dimension.
        num_layers: Number of LSTM layers.
        dense_dim: Dense layer dimension before output.
        pretrained_embeddings: Optional pretrained embedding matrix (numpy array).
        freeze_embeddings: Whether to freeze pretrained embeddings.
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, num_layers=1,
                 dense_dim=32, pretrained_embeddings=None, freeze_embeddings=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeddings),
                freeze=freeze_embeddings,
                padding_idx=0,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.fc2 = nn.Linear(dense_dim, 1)

    def forward(self, x):
        """Forward pass for classification.

        Args:
            x: Token IDs, shape (batch, seq_len).

        Returns:
            Sigmoid probability, shape (batch, 1).
        """
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = F.relu(self.fc1(hidden.squeeze(0)))
        return torch.sigmoid(self.fc2(out))

    def encode(self, x):
        """Return hidden representation before classification head.

        Args:
            x: Token IDs, shape (batch, seq_len).

        Returns:
            Encoding, shape (batch, dense_dim).
        """
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return F.relu(self.fc1(hidden.squeeze(0)))


class BiLSTMClassifier(nn.Module):
    """Iteration 3: Bidirectional LSTM with dropout.

    Args:
        vocab_size: Size of vocabulary.
        embedding_dim: Embedding dimension.
        hidden_dim: LSTM hidden state dimension (per direction).
        dropout_rate: Dropout probability.
        dense_dim: Dense layer dimension before output.
        pretrained_embeddings: Optional pretrained embedding matrix.
        freeze_embeddings: Whether to freeze pretrained embeddings.
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, dropout_rate=0.3,
                 dense_dim=32, pretrained_embeddings=None, freeze_embeddings=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeddings),
                freeze=freeze_embeddings,
                padding_idx=0,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim * 2, dense_dim)
        self.fc2 = nn.Linear(dense_dim, 1)

    def forward(self, x):
        """Forward pass for classification.

        Args:
            x: Token IDs, shape (batch, seq_len).

        Returns:
            Sigmoid probability, shape (batch, 1).
        """
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=1)
        out = self.dropout(F.relu(self.fc1(hidden_cat)))
        return torch.sigmoid(self.fc2(self.dropout(out)))

    def encode(self, x):
        """Return hidden representation before classification head.

        Args:
            x: Token IDs, shape (batch, seq_len).

        Returns:
            Encoding, shape (batch, dense_dim).
        """
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=1)
        return F.relu(self.fc1(hidden_cat))


class GRUClassifier(nn.Module):
    """Iteration 4: Bidirectional GRU (comparison with BiLSTM).

    Args:
        vocab_size: Size of vocabulary.
        embedding_dim: Embedding dimension.
        hidden_dim: GRU hidden state dimension (per direction).
        dropout_rate: Dropout probability.
        dense_dim: Dense layer dimension before output.
        pretrained_embeddings: Optional pretrained embedding matrix.
        freeze_embeddings: Whether to freeze pretrained embeddings.
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, dropout_rate=0.3,
                 dense_dim=32, pretrained_embeddings=None, freeze_embeddings=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dense_dim = dense_dim

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_embeddings),
                freeze=freeze_embeddings,
                padding_idx=0,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True,
                          bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim * 2, dense_dim)
        self.fc2 = nn.Linear(dense_dim, 1)

    def forward(self, x):
        """Forward pass for classification.

        Args:
            x: Token IDs, shape (batch, seq_len).

        Returns:
            Sigmoid probability, shape (batch, 1).
        """
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=1)
        out = self.dropout(F.relu(self.fc1(hidden_cat)))
        return torch.sigmoid(self.fc2(self.dropout(out)))

    def encode(self, x):
        """Return hidden representation before classification head.

        Args:
            x: Token IDs, shape (batch, seq_len).

        Returns:
            Encoding, shape (batch, dense_dim).
        """
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=1)
        return F.relu(self.fc1(hidden_cat))
