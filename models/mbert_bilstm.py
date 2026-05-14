import torch

import torch.nn as nn

from transformers import AutoModel


class MBERT_BiLSTM(nn.Module):

    def __init__(

        self,

        hidden_size=128,

        num_classes=2,

        dropout=0.3
    ):

        super(MBERT_BiLSTM, self).__init__()

        # Load pretrained mBERT
        self.bert = AutoModel.from_pretrained(
            "bert-base-multilingual-cased"
        )

        # BiLSTM layer
        self.lstm = nn.LSTM(

            input_size=768,

            hidden_size=hidden_size,

            batch_first=True,

            bidirectional=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Final classification layer
        self.fc = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    def forward(

        self,

        input_ids,

        attention_mask
    ):

        # Get mBERT outputs
        outputs = self.bert(

            input_ids=input_ids,

            attention_mask=attention_mask
        )

        # Last hidden states
        sequence_output = outputs.last_hidden_state

        # Pass through BiLSTM
        lstm_output, (hidden, cell) = self.lstm(
            sequence_output
        )

        # Concatenate forward + backward hidden states
        hidden = torch.cat(

            (hidden[-2], hidden[-1]),

            dim=1
        )

        # Apply dropout
        hidden = self.dropout(hidden)

        # Final classification
        logits = self.fc(hidden)

        return logits
    
    # Testing model
if __name__ == "__main__":

    from training.dataloader import FakeNewsDataset

    from torch.utils.data import DataLoader


    # Load dataset
    dataset = FakeNewsDataset(
        "Datasets/splits/train.csv"
    )

    # Create DataLoader
    loader = DataLoader(

        dataset,

        batch_size=4,

        shuffle=True
    )

    # Get one batch
    batch = next(iter(loader))

    # Initialize model
    model = MBERT_BiLSTM()

    # Forward pass
    outputs = model(

        input_ids=batch["input_ids"],

        attention_mask=batch["attention_mask"]
    )

    print("Model Output Shape:")
    print(outputs.shape)

    print("\nModel Outputs:")
    print(outputs)