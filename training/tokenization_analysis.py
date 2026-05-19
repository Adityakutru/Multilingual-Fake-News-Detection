import pandas as pd

from transformers import AutoTokenizer


# -----------------------------
# Load Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(

    "bert-base-multilingual-cased",

    use_fast=False
)

print("Tokenizer loaded")


# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(

    "Datasets/splits/train.csv",

    engine="python",

    encoding="utf-8",

    on_bad_lines="skip"
)

print("Dataset loaded")


# -----------------------------
# Store Token Counts
# -----------------------------
language_token_counts = {}


# -----------------------------
# Analyze Each Language
# -----------------------------
languages = df["language"].unique()

for language in languages:

    language_df = df[

        df["language"] == language
    ]

    total_tokens = 0

    sample_count = 0

    # Limit samples for speed
    samples = language_df["text"].head(200)

    for text in samples:

        tokens = tokenizer.tokenize(

            str(text)
        )

        total_tokens += len(tokens)

        sample_count += 1

    average_tokens = (

        total_tokens / sample_count
    )

    language_token_counts[language] = average_tokens


# -----------------------------
# Print Results
# -----------------------------
print("\nAverage Token Count Per Language:\n")

for language, avg_tokens in sorted(

    language_token_counts.items(),

    key=lambda x: x[1],

    reverse=True
):

    print(f"{language}: {avg_tokens:.2f}")


# -----------------------------
# Save Results
# -----------------------------
results_df = pd.DataFrame({

    "Language": list(language_token_counts.keys()),

    "Average Tokens":
    list(language_token_counts.values())
})

results_df.to_csv(

    "outputs/tokenization_analysis.csv",

    index=False
)

print("\nTokenization analysis completed!")