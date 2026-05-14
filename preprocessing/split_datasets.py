import pandas as pd

from sklearn.model_selection import train_test_split

# Load final dataset
df = pd.read_csv(
    "Datasets/final_master_dataset.csv"
)

print("Original Dataset Shape:")
print(df.shape)

# First split:
# 80% train
# 20% temp (validation + test)

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Second split:
# 10% validation
# 10% test

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"]
)

# Show shapes
print("\nTrain Shape:")
print(train_df.shape)

print("\nValidation Shape:")
print(val_df.shape)

print("\nTest Shape:")
print(test_df.shape)

# Save splits
train_df.to_csv(
    "Datasets/train.csv",
    index=False,
    encoding="utf-8"
)

val_df.to_csv(
    "Datasets/validation.csv",
    index=False,
    encoding="utf-8"
)

test_df.to_csv(
    "Datasets/test.csv",
    index=False,
    encoding="utf-8"
)

print("\nTrain, validation, and test datasets saved successfully!")