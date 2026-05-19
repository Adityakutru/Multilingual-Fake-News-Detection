import pandas as pd

import matplotlib.pyplot as plt


# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(

    "Datasets/splits/train.csv",

    engine="python",

    encoding="utf-8",

    on_bad_lines="skip"
)


# -----------------------------
# Total Samples Per Language
# -----------------------------
language_counts = df["language"].value_counts()

print("\nSamples Per Language:\n")

print(language_counts)


# Save counts
language_counts.to_csv(

    "outputs/language_distribution.csv"
)


# -----------------------------
# Plot Total Samples
# -----------------------------
plt.figure(figsize=(10, 6))

language_counts.plot(

    kind="bar"
)

plt.title(
    "Dataset Distribution Across Languages"
)

plt.xlabel(
    "Language"
)

plt.ylabel(
    "Number of Samples"
)

plt.xticks(rotation=45)

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "outputs/plots/language_distribution.png"
)

plt.close()


# -----------------------------
# Fake vs Real Per Language
# -----------------------------
label_distribution = pd.crosstab(

    df["language"],

    df["label"]
)

print("\nFake vs Real Distribution:\n")

print(label_distribution)


# Save distribution
label_distribution.to_csv(

    "outputs/language_label_distribution.csv"
)


# -----------------------------
# Plot Fake vs Real
# -----------------------------
label_distribution.plot(

    kind="bar",

    figsize=(12, 6)
)

plt.title(
    "Fake vs Real News Distribution Per Language"
)

plt.xlabel(
    "Language"
)

plt.ylabel(
    "Number of Samples"
)

plt.xticks(rotation=45)

plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "outputs/plots/fake_real_distribution.png"
)

plt.close()


print("\nDistribution analysis completed!")