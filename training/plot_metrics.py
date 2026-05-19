import matplotlib.pyplot as plt


# Training history
train_losses = [

    0.0596,

    0.0512,

    0.0487
]

validation_losses = [

    0.0493,

    0.0506,

    0.0480
]

validation_accuracies = [

    0.9078,

    0.8968,

    0.9107
]


# Epoch numbers
epochs = [1, 2, 3]


# -------------------------------
# Train vs Validation Loss
# -------------------------------
plt.figure(figsize=(8, 5))

plt.plot(

    epochs,

    train_losses,

    marker="o",

    label="Train Loss"
)

plt.plot(

    epochs,

    validation_losses,

    marker="o",

    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.grid(True)

plt.savefig(
    "outputs/plots/loss_curve.png"
)

plt.show()


# -------------------------------
# Validation Accuracy
# -------------------------------
plt.figure(figsize=(8, 5))

plt.plot(

    epochs,

    validation_accuracies,

    marker="o"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Validation Accuracy")

plt.grid(True)

plt.savefig(
    "outputs/plots/accuracy_curve.png"
)

plt.show()

print("Plots saved successfully!")