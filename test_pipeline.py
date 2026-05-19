import os

os.environ["OMP_NUM_THREADS"] = "1"

os.environ["MKL_NUM_THREADS"] = "1"

import torch

print("Torch imported")

from transformers import AutoTokenizer

print("Transformers imported")

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased",
    use_fast=False
)

print("Tokenizer loaded")

import pandas as pd

print("Pandas imported")

df = pd.read_csv(

    "Datasets/splits/train.csv",

    engine="python",

    encoding="utf-8",

    on_bad_lines="skip"
)

print("CSV loaded")

print(df.shape)