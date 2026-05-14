import pandas as pd

# Load Excel dataset
df = pd.read_excel(
    "Datasets/BharatFakeNewsKosh/bharatfakenewskosh.xlsx"
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

# Fill missing News Body values
df["News Body"] = df["News Body"].fillna("")

# Create combined text column
df["text"] = df["Statement"] + " " + df["News Body"]

# Convert labels
df["label"] = df["Label"].map({
    True: "real",
    False: "fake"
})

# Standardize language column
df["language"] = df["Language"]

# Add source column
df["source"] = "BharatFakeNewsKosh"

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

final_df.to_csv(
    "Datasets/bharat_dataset_cleaned.csv",
    index=False,
    encoding="utf-8"
)

print("\nBharatFakeNewsKosh cleaned dataset saved successfully!")