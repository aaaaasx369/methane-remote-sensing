from pathlib import Path
import pandas as pd


CR_PATH = Path("outputs/22_controlled_release_s2_dataset_table.csv")
MA_PATH = Path("outputs/18_methaneair_s2_dataset_table.csv")
OUT_PATH = Path("outputs/23_combined_classification_dataset.csv")

cr_df = pd.read_csv(CR_PATH)
ma_df = pd.read_csv(MA_PATH)

cr_df["dataset_group"] = "controlled_release_s2"
ma_df["dataset_group"] = "methaneair_s2_positive"

cr_df["label"] = pd.to_numeric(cr_df["label"], errors="coerce")
ma_df["label"] = pd.to_numeric(ma_df["label"], errors="coerce")

combined = pd.concat([cr_df, ma_df], ignore_index=True, sort=False)

combined.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("Total samples:", len(combined))

print("\nSamples by dataset_group:")
print(combined["dataset_group"].value_counts(dropna=False))

print("\nSamples by label:")
print(combined["label"].value_counts(dropna=False))

print("\nLabel counts by dataset_group:")
print(pd.crosstab(combined["dataset_group"], combined["label"]))