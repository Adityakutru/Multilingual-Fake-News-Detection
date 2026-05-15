import pandas as pd

import torch

from tqdm import tqdm

from transformers import AutoTokenizer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from models.mbert_bilstm import MBERT_BiLSTM


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:")
print(device)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased"
)


# Load test dataset
df = pd.read_csv(
    "Datasets/splits/test.csv"
)

print("Test Dataset Loaded")

print(df.shape)


# Load model
model = MBERT_BiLSTM()

model.load_state_dict(
    torch.load(
        "outputs/checkpoints/mbert_bilstm_gpu.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model Loaded")


# Label mapping
label_map = {
    "fake": 0,
    "real": 1
}


# Containers
all_predictions = []

all_labels = []


# Evaluation loop
with torch.no_grad():

    for _, row in tqdm(df.iterrows(), total=len(df)):

        text = str(row["text"])

        label = label_map[row["label"]]

        encoding = tokenizer(

            text,

            padding="max_length",

            truncation=True,

            max_length=128,

            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)

        attention_mask = encoding["attention_mask"].to(device)

        outputs = model(

            input_ids=input_ids,

            attention_mask=attention_mask
        )

        prediction = torch.argmax(
            outputs,
            dim=1
        ).item()

        all_predictions.append(prediction)

        all_labels.append(label)


# Metrics
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions
)

recall = recall_score(
    all_labels,
    all_predictions
)

f1 = f1_score(
    all_labels,
    all_predictions
)


# Print results
print("\nAccuracy:")
print(accuracy)

print("\nPrecision:")
print(precision)

print("\nRecall:")
print(recall)

print("\nF1 Score:")
print(f1)

print("\nConfusion Matrix:")
print(confusion_matrix(
    all_labels,
    all_predictions
))

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_predictions
))