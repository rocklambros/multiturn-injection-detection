"""PM-1a: Hierarchical DistilBERT multi-turn baseline.

Encodes each turn independently via DistilBERT [CLS] token,
then applies a small self-attention layer over the [CLS] sequence.
Fair comparison: both this and the dual-encoder LSTM see turn-level
representations, differing only in temporal aggregation.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel


class HierarchicalDistilBERT(nn.Module):
    """Hierarchical DistilBERT: turn-level encoding + cross-turn attention.

    Args:
        num_attention_heads: Heads in the cross-turn attention layer.
        cross_turn_layers: Number of transformer layers over CLS tokens.
        max_turns: Maximum conversation turns.
        dropout_rate: Dropout probability.
        freeze_bert: Whether to freeze DistilBERT weights.
    """

    def __init__(self, num_attention_heads=4, cross_turn_layers=2,
                 max_turns=10, dropout_rate=0.3, freeze_bert=True):
        super().__init__()
        self.max_turns = max_turns
        self.freeze_bert = freeze_bert
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        bert_dim = self.bert.config.hidden_size  # 768

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bert_dim,
            nhead=num_attention_heads,
            dim_feedforward=256,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.cross_turn_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cross_turn_layers,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(bert_dim, 1)

    def forward(self, input_ids, attention_mask, turn_mask):
        """Forward pass.

        Args:
            input_ids: (batch, max_turns, seq_len) -- token IDs per turn.
            attention_mask: (batch, max_turns, seq_len) -- attention mask per turn.
            turn_mask: (batch, max_turns) -- 1=real turn, 0=padding.

        Returns:
            Logits, shape (batch, 1).
        """
        batch_size, max_turns, seq_len = input_ids.shape
        cls_tokens = []

        ctx = torch.no_grad if self.freeze_bert else torch.enable_grad
        for t in range(max_turns):
            turn_ids = input_ids[:, t, :]
            turn_attn = attention_mask[:, t, :]

            with ctx():
                outputs = self.bert(input_ids=turn_ids, attention_mask=turn_attn)

            cls_tokens.append(outputs.last_hidden_state[:, 0, :])

        cls_sequence = torch.stack(cls_tokens, dim=1)  # (batch, max_turns, 768)

        cls_sequence = cls_sequence * turn_mask.unsqueeze(-1)

        # TransformerEncoder expects src_key_padding_mask: True = ignore
        padding_mask = (turn_mask == 0)
        cross_out = self.cross_turn_transformer(cls_sequence, src_key_padding_mask=padding_mask)

        turn_counts = turn_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (cross_out * turn_mask.unsqueeze(-1)).sum(dim=1) / turn_counts

        return self.classifier(self.dropout(pooled))
