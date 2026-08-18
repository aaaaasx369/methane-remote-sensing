from pathlib import Path
import pandas as pd


INDEX_PATH = Path("outputs/30_controlled_release_landsat_patch_index.csv")
QUALITY_PATH = Path("outputs/31_controlled_release_landsat_patch_quality.csv")
OUT_PATH = Path("outputs/32_controlled_release_landsat_dataset_table.csv")

if not INDEX_PATH.exists():
    raise FileNotFoundError(f"Cannot find index file: {INDEX_PATH}")

if not QUALITY_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find quality file: {QUALITY_PATH}. "
        "Run check_controlled_release_landsat_patches.py first."
    )

index_df = pd.read_csv(INDEX_PATH)

if "download_status" in index_df.columns:
    index_df = index_df[index_df["download_status"].isin([
        "success",
        "success_existing"
    ])].copy()

quality_df = pd.read_csv(QUALITY_PATH)

dataset_df = index_df.merge(
    quality_df,
    on="filename",
    how="left"
)

dataset_df["dataset_group"] = "controlled_release_landsat"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
dataset_df.to_csv(OUT_PATH, index=False)

print(dataset_df)
print("Saved:", OUT_PATH)
print("Number of samples:", len(dataset_df))

if len(dataset_df) > 0:
    print("\nLabel counts:")
    print(dataset_df["label"].value_counts(dropna=False))

    print("\nLandsat sensor counts:")
    if "landsat_sensor" in dataset_df.columns:
        print(dataset_df["landsat_sensor"].value_counts(dropna=False))

    if "all_zero" in dataset_df.columns:
        print("\nall_zero counts:")
        print(dataset_df["all_zero"].value_counts(dropna=False))

    if "has_nan" in dataset_df.columns:
        print("\nhas_nan counts:")
        print(dataset_df["has_nan"].value_counts(dropna=False))
