import pandas as pd

import matplotlib.pyplot as plt


# Load CSV
df = pd.read_csv(
    "outputs/language_accuracy.csv"
)


# Sort values
df = df.sort_values(
    by="Accuracy",
    ascending=False
)


# Plot
plt.figure(figsize=(10, 6))

plt.bar(

    df["Language"],

    df["Accuracy"]
)

plt.title(
    "Language-wise Accuracy"
)

plt.xlabel(
    "Language"
)

plt.ylabel(
    "Accuracy"
)

plt.xticks(rotation=45)

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "outputs/plots/language_accuracy.png"
)

plt.show()

print("Language accuracy plot saved successfully!")