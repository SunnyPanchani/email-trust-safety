import pandas as pd

# Load processed dataset (contains labels)
df = pd.read_parquet("data/processed/train.parquet")[["label"]]

# Load metadata without label
meta = pd.read_parquet("data/features/metadata_train_v2.parquet")

# Attach label
meta["label"] = df["label"]

# Save back
meta.to_parquet("data/features/metadata_train_v2.parquet")

print("✅ metadata_train_v2.parquet now includes label column!")
