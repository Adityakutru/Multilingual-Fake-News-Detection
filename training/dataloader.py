import pandas as pd

import torch

from torch.utils.data import Dataset


class FakeNewsDataset(Dataset):

    def __init__(

        self,

        csv_file,

        tokenizer,

        max_length=128
    ):

        print(f"Loading CSV: {csv_file}")

        self.df = pd.read_csv(

            csv_file,

            engine="python",

            encoding="utf-8",

            on_bad_lines="skip"
        )

        print("CSV Loaded")

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.label_map = {

            "fake": 0,

            "real": 1
        }

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        text = str(
            self.df.iloc[idx]["text"]
        )

        label = self.df.iloc[idx]["label"]

        label = self.label_map[label]

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
            torch.tensor(
                label,
                dtype=torch.long
            )
        }