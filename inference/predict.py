import os

os.environ["OMP_NUM_THREADS"] = "1"

os.environ["MKL_NUM_THREADS"] = "1"


import torch

from transformers import AutoTokenizer

from models.mbert_bilstm import MBERT_BiLSTM


# Device setup
device = torch.device(

    "cuda" if torch.cuda.is_available()

    else "cpu"
)

print("Using Device:")
print(device)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(

    "bert-base-multilingual-cased",

    use_fast=False
)

print("Tokenizer loaded")


# Load model
model = MBERT_BiLSTM()

model.load_state_dict(

    torch.load(

        "outputs/checkpoints/mbert_bilstm_best.pth",

        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# Label mapping
label_map = {

    0: "Fake News",

    1: "Real News"
}


# User input
text = input("\nEnter news article:\n")


# Tokenize input
encoding = tokenizer(

    text,

    padding="max_length",

    truncation=True,

    max_length=128,

    return_tensors="pt"
)


# Move tensors to device
input_ids = encoding["input_ids"].to(device)

attention_mask = encoding["attention_mask"].to(device)


# Prediction
with torch.no_grad():

    outputs = model(

        input_ids=input_ids,

        attention_mask=attention_mask
    )

    probabilities = torch.softmax(

        outputs,

        dim=1
    )

    prediction = torch.argmax(

        probabilities,

        dim=1
    ).item()

    confidence = probabilities[0][prediction].item()


# Print result
print("\nPrediction:")
print(label_map[prediction])

print(f"\nConfidence: {confidence:.4f}")