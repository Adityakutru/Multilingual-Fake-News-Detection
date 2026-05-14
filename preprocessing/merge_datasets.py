import pandas as pd

# Load cleaned datasets
zenodo_df = pd.read_csv(
    "Datasets/zenodo_dataset_cleaned.csv"
)

bharat_df = pd.read_csv(
    "Datasets/bharat_dataset_cleaned.csv"
)

ifnd_df = pd.read_csv(
    "Datasets/ifnd_dataset_cleaned.csv"
)

# Show individual shapes
# print("Zenodo Shape:")
# print(zenodo_df.shape)

# print("\nBharatFakeNewsKosh Shape:")
# print(bharat_df.shape)

# print("\nIFND Shape:")
# print(ifnd_df.shape)

# Merge all datasets
final_df = pd.concat(
    [zenodo_df, bharat_df, ifnd_df],
    ignore_index=True
)

# print("\nMerged Dataset Shape:")
# print(final_df.shape)

# Check duplicates across datasets
duplicate_count = final_df.duplicated(
    subset=["text"]
).sum()

# print("\nCross-Dataset Duplicates:")
# print(duplicate_count)

# Remove duplicates across datasets
final_df = final_df.drop_duplicates(
    subset=["text"]
)

# print("\nFinal Dataset Shape After Deduplication:")
# print(final_df.shape)

# # Label distribution
# print("\nFinal Label Distribution:")
# print(final_df["label"].value_counts())

# # Language distribution
# print("\nFinal Language Distribution:")
# print(final_df["language"].value_counts())

# Save final merged dataset
final_df.to_csv(
    "Datasets/final_master_dataset.csv",
    index=False,
    encoding="utf-8"
)

print("\nFinal master dataset saved successfully!")