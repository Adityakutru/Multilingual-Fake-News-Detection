import pandas as pd

print("Starting CSV test")

df = pd.read_csv(
    "Datasets/splits/train.csv"
)

print("CSV loaded successfully")

print(df.shape)