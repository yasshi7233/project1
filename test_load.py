# ==============================
# test_load.py
# ==============================

import pandas as pd

# CSV का पूरा path (raw string से बताओ ताकि \ ठीक रहें)
csv_path = r"E:\kavya\dataset\train.csv"

print("Loading dataset from:", csv_path)
train_df = pd.read_csv(csv_path)

print("\n✅ Training data loaded successfully!")
print("Training data shape:", train_df.shape)
print("\nColumns:", list(train_df.columns))
print("\nFirst 5 rows:")
print(train_df.head())
print("\nSummary statistics:")
print(train_df.describe())
