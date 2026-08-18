from pathlib import Path
import pandas as pd
import numpy as np
import rasterio


INPUT_PATH = Path("outputs/24_experiment_split.csv")
OUT_PATH = Path("outputs/25_s2_patch_features.csv")

BAND_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12"]


def safe_stats(arr):
    arr = arr.astype(float)

    # Remove invalid / zero-only background if needed
    valid = arr[np.isfinite(arr)]

    if len(valid) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
        }

    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "p25": float(np.percentile(valid, 25)),
        "p50": float(np.percentile(valid, 50)),
        "p75": float(np.percentile(valid, 75)),
    }


def extract_features_from_tif(path):
    features = {}

    with rasterio.open(path) as src:
        arr = src.read()  # shape: bands, height, width

    # Basic band statistics
    for i, band_name in enumerate(BAND_NAMES):
        band = arr[i]
        stats = safe_stats(band)

        for stat_name, value in stats.items():
            features[f"{band_name}_{stat_name}"] = value

    # Convert to float for ratio features
    B2 = arr[0].astype(float)
    B3 = arr[1].astype(float)
    B4 = arr[2].astype(float)
    B8 = arr[3].astype(float)
    B11 = arr[4].astype(float)
    B12 = arr[5].astype(float)

    eps = 1e-6

    ratio_12_11 = B12 / (B11 + eps)
    ratio_11_8 = B11 / (B8 + eps)
    diff_12_11 = B12 - B11
    ndvi = (B8 - B4) / (B8 + B4 + eps)

    extra_maps = {
        "ratio_B12_B11": ratio_12_11,
        "ratio_B11_B8": ratio_11_8,
        "diff_B12_B11": diff_12_11,
        "ndvi": ndvi,
    }

    for name, value_map in extra_maps.items():
        stats = safe_stats(value_map)
        for stat_name, value in stats.items():
            features[f"{name}_{stat_name}"] = value

    return features


def main():
    df = pd.read_csv(INPUT_PATH)

    rows = []

    for idx, row in df.iterrows():
        path = Path(row["relative_path"])

        if not path.exists():
            print("Missing file:", path)
            continue

        try:
            features = extract_features_from_tif(path)
        except Exception as e:
            print("Error reading:", path, e)
            continue

        meta = {
            "filename": row.get("filename", ""),
            "relative_path": row.get("relative_path", ""),
            "label": row.get("label", ""),
            "split": row.get("split", ""),
            "dataset_group": row.get("dataset_group", ""),
            "event_id": row.get("event_id", ""),
            "ground_truth_type": row.get("ground_truth_type", ""),
            "sensor": row.get("sensor", ""),
            "datetime_utc": row.get("datetime_utc", ""),
            "lat": row.get("lat", ""),
            "lon": row.get("lon", ""),
        }

        meta.update(features)
        rows.append(meta)

    feature_df = pd.DataFrame(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Feature table shape:", feature_df.shape)

    print("\nLabel counts:")
    print(feature_df["label"].value_counts(dropna=False))

    print("\nSplit counts:")
    print(feature_df["split"].value_counts(dropna=False))


if __name__ == "__main__":
    main()