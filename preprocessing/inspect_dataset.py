import pandas as pd

# Load dataset
# df = pd.read_csv("Datasets/zenodo_dataset.csv")

# Dataset shape
# print("Dataset Shape:")
# print(df.shape)

# print("\n")

# # Column names
# print("Columns:")
# print(df.columns)

# print("\n")

# # Missing values
# print("Missing Values:")
# print(df.isnull().sum())

# print("\n")

# # Label distribution
# print("Label Distribution:")
# print(df["label"].value_counts())

# print("\n")

# # Language distribution
# print("Language Distribution:")
# print(df["language"].value_counts())

# # Duplicate articles
# duplicate_count = df.duplicated(subset=["text"]).sum()

# print("\nDuplicate Articles:")
# print(duplicate_count)

# Remove duplicate articles
# df = df.drop_duplicates(subset=["text"])

# print("\nDataset Shape After Removing Duplicates:")
# print(df.shape)

# # Clean text formatting
# df["text"] = df["text"].str.replace("\n", " ", regex=False)

# df["text"] = df["text"].str.replace("\t", " ", regex=False)

# df["text"] = df["text"].str.strip()

# df.to_csv(
#     "Datasets/zenodo_dataset_cleaned.csv",
#     index=False,
#     encoding="utf-8"
# )

# df = pd.read_csv("Datasets/zenodo_dataset_cleaned.csv")

# # Create text length column
# df["text_length"] = df["text"].apply(len)

# print(df["text_length"].describe())

# short_articles = df[df["text_length"] < 50]

# print("\nVery Short Articles:")
# print(len(short_articles))

# print(short_articles["text"].head(10))

# # Remove very short articles
# df = df[df["text_length"] >= 50]

# print("\nDataset Shape After Removing Short Articles:")
# print(df.shape)

# # Remove temporary column
# df = df.drop(columns=["text_length"])

# df.to_csv(
#     "Datasets/zenodo_dataset_cleaned.csv",
#     index=False,
#     encoding="utf-8"
# )

# print("\nFinal cleaned dataset saved successfully!")

df = pd.read_csv("Datasets/zenodo_dataset_cleaned.csv")
print(df.shape)