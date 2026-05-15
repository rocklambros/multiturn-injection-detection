"""PM-1b: Concatenated DistilBERT baseline (naive strong baseline).

Concatenates all turns with [SEP] tokens and processes as single
sequence through DistilBERT. Known limitation: 512-token context
limit truncates most multi-turn conversations.
"""

import torch.nn as nn
from transformers import DistilBertModel


class ConcatenatedDistilBERT(nn.Module):
    """Concatenate all turns, process through DistilBERT.

    Args:
        max_length: Maximum total tokens (DistilBERT limit = 512).
        dropout_rate: Dropout probability.
        freeze_bert: Whether to freeze DistilBERT weights.
    """

    def __init__(self, max_length=512, dropout_rate=0.3, freeze_bert=False):
        super().__init__()
        self.max_length = max_length
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        bert_dim = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(bert_dim, 1)

    def forward(self, input_ids, attention_mask):
        """Forward pass.

        Args:
            input_ids: (batch, max_length) — concatenated and truncated.
            attention_mask: (batch, max_length).

        Returns:
            Logits, shape (batch, 1).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]
        return self.classifier(self.dropout(cls_output))
