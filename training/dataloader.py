import pandas as pd

import torch

from torch.utils.data import Dataset

from transformers import AutoTokenizer

from torch.utils.data import DataLoader


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased"
)


class FakeNewsDataset(Dataset):

    def __init__(self, csv_file, max_length=128):

        # Load CSV
        self.df = pd.read_csv(csv_file)

        # Store max length
        self.max_length = max_length

        # Convert labels into numbers
        self.label_map = {
            "fake": 0,
            "real": 1
        }

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        # Get article text
        text = str(
            self.df.iloc[idx]["text"]
        )

        # Get label
        label = self.df.iloc[idx]["label"]

        # Convert label to number
        label = self.label_map[label]

        # Tokenize text
        encoding = tokenizer(

            text,

            padding="max_length",

            truncation=True,

            max_length=self.max_length,

            return_tensors="pt"
        )

        return {

            "input_ids":
            encoding["input_ids"].squeeze(0),

            "attention_mask":
            encoding["attention_mask"].squeeze(0),

            "label":
            torch.tensor(label, dtype=torch.long)
        }


# Testing dataset loading
# Testing DataLoader
if __name__ == "__main__":

    train_dataset = FakeNewsDataset(
        "Datasets/splits/train.csv"
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=8,

        shuffle=True
    )

    # Get first batch
    batch = next(iter(train_loader))

    print("Input IDs Shape:")
    print(batch["input_ids"].shape)

    print("\nAttention Mask Shape:")
    print(batch["attention_mask"].shape)

    print("\nLabels Shape:")
    print(batch["label"].shape)