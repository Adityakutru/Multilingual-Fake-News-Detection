import os

os.environ["OMP_NUM_THREADS"] = "1"

os.environ["MKL_NUM_THREADS"] = "1"


import torch

import pandas as pd

from transformers import AutoTokenizer

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from collections import defaultdict

from training.dataloader import FakeNewsDataset

from models.mbert_bilstm import MBERT_BiLSTM



# Device

device = torch.device(

    "cuda" if torch.cuda.is_available()

    else "cpu"
)

print("Using Device:")
print(device)



# Load Tokenizer

tokenizer = AutoTokenizer.from_pretrained(

    "bert-base-multilingual-cased",

    use_fast=False
)



# Load Dataset CSV

df = pd.read_csv(

    "Datasets/splits/test.csv",

    engine="python",

    encoding="utf-8",

    on_bad_lines="skip"
)



# Dataset + Loader

test_dataset = FakeNewsDataset(

    "Datasets/splits/test.csv",

    tokenizer
)

test_loader = DataLoader(

    test_dataset,

    batch_size=8,

    shuffle=False
)



# Load Model

model = MBERT_BiLSTM()

model.load_state_dict(

    torch.load(

        "outputs/checkpoints/mbert_bilstm_best.pth",

        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded")


# Store Results

language_predictions = defaultdict(list)

language_labels = defaultdict(list)



# Evaluation

with torch.no_grad():

    for i, batch in enumerate(test_loader):

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["label"].to(device)

        outputs = model(

            input_ids=input_ids,

            attention_mask=attention_mask
        )

        predictions = torch.argmax(

            outputs,

            dim=1
        )

        predictions = predictions.cpu().numpy()

        labels = labels.cpu().numpy()

        batch_languages = df.iloc[

            i * 8 : i * 8 + len(predictions)
        ]["language"].values

        for lang, pred, label in zip(

            batch_languages,

            predictions,

            labels
        ):

            language_predictions[lang].append(pred)

            language_labels[lang].append(label)



# Calculate Accuracy

print("\nLanguage-wise Accuracy:\n")

results = []

for lang in language_predictions:

    accuracy = accuracy_score(

        language_labels[lang],

        language_predictions[lang]
    )

    results.append((lang, accuracy))

    print(f"{lang}: {accuracy:.4f}")



# Save Results

results_df = pd.DataFrame(

    results,

    columns=["Language", "Accuracy"]
)

results_df.to_csv(

    "outputs/language_accuracy.csv",

    index=False
)

print("\nResults saved successfully!")