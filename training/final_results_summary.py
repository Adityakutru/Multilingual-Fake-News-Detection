import pandas as pd


# -----------------------------
# Model Comparison
# -----------------------------
model_results = pd.DataFrame({

    "Model": [

        "mBERT Baseline",

        "mBERT + BiLSTM + Focal Loss"
    ],

    "Validation Accuracy": [

        90.95,

        91.07
    ]
})


# -----------------------------
# Language Accuracy
# -----------------------------
language_accuracy = pd.read_csv(

    "outputs/language_accuracy.csv"
)


# -----------------------------
# Dataset Distribution
# -----------------------------
language_distribution = pd.read_csv(

    "outputs/language_distribution.csv"
)

language_distribution.columns = [

    "Language",

    "Samples"
]


# -----------------------------
# Tokenization Analysis
# -----------------------------
tokenization = pd.read_csv(

    "outputs/tokenization_analysis.csv"
)


# -----------------------------
# Merge Data
# -----------------------------
merged = language_accuracy.merge(

    language_distribution,

    on="Language"
)

merged = merged.merge(

    tokenization,

    on="Language"
)


# -----------------------------
# Sort by Accuracy
# -----------------------------
merged = merged.sort_values(

    by="Accuracy",

    ascending=False
)


# -----------------------------
# Save Final Results
# -----------------------------
model_results.to_csv(

    "outputs/model_comparison.csv",

    index=False
)

merged.to_csv(

    "outputs/final_language_analysis.csv",

    index=False
)


# -----------------------------
# Print Results
# -----------------------------
print("\nModel Comparison:\n")

print(model_results)

print("\nFinal Language Analysis:\n")

print(merged)


print("\nFinal research summary generated successfully!")