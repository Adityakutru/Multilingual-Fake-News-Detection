import torch

from torch.utils.data import DataLoader

from torch.optim import AdamW

from tqdm import tqdm


from training.dataloader import FakeNewsDataset

from models.mbert_bilstm import MBERT_BiLSTM

from models.focal_loss import FocalLoss


# Device configuration
device = torch.device(

    "cuda" if torch.cuda.is_available()

    else "cpu"
)

print("Using Device:")

print(device)


# Load training dataset
train_dataset = FakeNewsDataset(

    "Datasets/splits/train.csv"
)

# Create DataLoader
train_loader = DataLoader(

    train_dataset,

    batch_size=16,

    shuffle=True
)


# Initialize model
model = MBERT_BiLSTM()

model = model.to(device)


# Loss function
criterion = FocalLoss()


# Optimizer
optimizer = AdamW(

    model.parameters(),

    lr=2e-5
)


# Number of epochs
epochs = 1


# Training loop
for epoch in range(epochs):

    print(f"\nEpoch {epoch + 1}/{epochs}")

    model.train()

    total_loss = 0

    progress_bar = tqdm(train_loader)

    for batch in progress_bar:

        # Move tensors to device
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

    # Average loss
    avg_loss = total_loss / len(train_loader)

    print(f"\nAverage Loss: {avg_loss:.4f}")


# Save model
torch.save(

    model.state_dict(),

    "outputs/checkpoints/mbert_bilstm_gpu.pth"
)

print("\nModel saved successfully!")