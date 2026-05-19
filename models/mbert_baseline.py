import torch

import torch.nn as nn

from transformers import AutoModel


class MBERT_Baseline(nn.Module):

    def __init__(

        self,

        num_classes=2,

        dropout=0.3
    ):

        super(MBERT_Baseline, self).__init__()

        # Load pretrained mBERT
        self.bert = AutoModel.from_pretrained(
            "bert-base-multilingual-cased"
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Final classification layer
        self.fc = nn.Linear(
            768,
            num_classes
        )

    def forward(

        self,

        input_ids,

        attention_mask
    ):

        # BERT outputs
        outputs = self.bert(

            input_ids=input_ids,

            attention_mask=attention_mask
        )

        # CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Dropout
        cls_output = self.dropout(cls_output)

        # Classification
        logits = self.fc(cls_output)

        return logits