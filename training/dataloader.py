import pandas as pd

import torch

from torch.utils.data import Dataset

from transformers import AutoTokenizer

from torch.utils.data import DataLoader


class FakeNewsDataset(Dataset):

    def __init__(self, csv_file, max_length=128):

        print("Dataloader Step 1")

        # Load CSV
        self.df = pd.read_csv(csv_file)

        print("Dataloader Step 2")

        # Store max length
        self.max_length = max_length

        print("Dataloader Step 3")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )

        print("Dataloader Step 4")

        # Convert labels into numbers
        self.label_map = {
            "fake": 0,
            "real": 1
        }

        print("Dataloader Step 5")

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
        encoding = self.tokenizer(

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