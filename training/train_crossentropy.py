import os

os.environ["OMP_NUM_THREADS"] = "1"

os.environ["MKL_NUM_THREADS"] = "1"

import torch

from transformers import AutoTokenizer

from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader

from torch.optim import AdamW

from tqdm import tqdm

from training.dataloader import FakeNewsDataset

from models.mbert_bilstm import MBERT_BiLSTM

import torch.nn as nn


# Device configuration
device = torch.device(

    "cuda" if torch.cuda.is_available()

    else "cpu"
)

print("Using Device:")
print(device)

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased",
    use_fast=False
)

print("Tokenizer loaded")

# Load training dataset
train_dataset = FakeNewsDataset(

    "Datasets/splits/train.csv",

    tokenizer
)

print("Train dataset loaded")


# Load validation dataset
validation_dataset = FakeNewsDataset(

    "Datasets/splits/validation.csv",

    tokenizer
)

print("Validation dataset loaded")


# Create DataLoaders
train_loader = DataLoader(

    train_dataset,

    batch_size=8,

    shuffle=True
)

validation_loader = DataLoader(

    validation_dataset,

    batch_size=8,

    shuffle=False
)


# Initialize model
model = MBERT_BiLSTM()

model = model.to(device)


# Loss function
criterion = nn.CrossEntropyLoss()


# Optimizer
optimizer = AdamW(

    model.parameters(),

    lr=2e-5
)


# Number of epochs
epochs = 1


# Metric containers
train_losses = []

validation_losses = []

validation_accuracies = []


# Training loop
for epoch in range(epochs):

    print(f"\nEpoch {epoch + 1}/{epochs}")

    model.train()

    total_loss = 0

    progress_bar = tqdm(train_loader)

    # Training phase
    for batch in progress_bar:

        # Move tensors to GPU
        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["label"].to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(

            input_ids=input_ids,

            attention_mask=attention_mask
        )

        # Compute loss
        loss = criterion(

            outputs,

            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({

            "Loss": loss.item()
        })

    # Average train loss
    avg_train_loss = (

        total_loss / len(train_loader)
    )

    train_losses.append(
        avg_train_loss
    )

    print(f"\nTrain Loss: {avg_train_loss:.4f}")


    # Validation phase
    model.eval()

    validation_loss = 0

    all_predictions = []

    all_labels = []

    with torch.no_grad():

        for batch in validation_loader:

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(

                input_ids=input_ids,

                attention_mask=attention_mask
            )

            # Validation loss
            loss = criterion(

                outputs,

                labels
            )

            validation_loss += loss.item()

            # Predictions
            predictions = torch.argmax(

                outputs,

                dim=1
            )

            all_predictions.extend(

                predictions.cpu().numpy()
            )

            all_labels.extend(

                labels.cpu().numpy()
            )

    # Average validation loss
    avg_validation_loss = (

        validation_loss / len(validation_loader)
    )

    validation_losses.append(
        avg_validation_loss
    )

    # Validation accuracy
    validation_accuracy = accuracy_score(

        all_labels,

        all_predictions
    )

    validation_accuracies.append(
        validation_accuracy
    )

    print(f"Validation Loss: {avg_validation_loss:.4f}")

    print(f"Validation Accuracy: {validation_accuracy:.4f}")

    # Save model after every epoch
    torch.save(

        model.state_dict(),

        "outputs/checkpoints/mbert_crossentropy.pth"
    )

    print("Model saved successfully!")


print("\nTraining Completed!")

print("\nTrain Losses:")
print(train_losses)

print("\nValidation Losses:")
print(validation_losses)

print("\nValidation Accuracies:")
print(validation_accuracies)