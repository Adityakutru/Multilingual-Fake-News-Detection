import pandas as pd

# Load IFND dataset
df = pd.read_csv(
    "Datasets/IFND/IFND.csv",
    encoding="latin1"
)

# # Show first rows
# print(df.head())

# print("\n")

# # Show columns
# print(df.columns)

# print("\nDataset Shape:")
# print(df.shape)

# print("\nLabel Distribution:")
# print(df["Label"].value_counts())

# print("\nMissing Values:")
# print(df.isnull().sum())

# Create standardized text column
df["text"] = df["Statement"]

# Standardize labels
df["label"] = df["Label"].replace({
    "TRUE": "real",
    "Fake": "fake"
})

# Add language column
df["language"] = "English"

# Add source column
df["source"] = "IFND"

# Keep only required columns
final_df = df[[
    "text",
    "label",
    "language",
    "source"
]]

# print(final_df.head())

# print("\nFinal Dataset Shape:")
# print(final_df.shape)

# # Check duplicates
# duplicate_count = final_df.duplicated(
#     subset=["text"]
# ).sum()

# print("\nDuplicate Articles:")
# print(duplicate_count)

# Remove duplicates
final_df = final_df.drop_duplicates(
    subset=["text"]
)

print("\nShape After Removing Duplicates:")
print(final_df.shape)

# Save cleaned dataset
final_df.to_csv(
    "Datasets/ifnd_dataset_cleaned.csv",
    index=False,
    encoding="utf-8"
)

print("\nIFND cleaned dataset saved successfully!")