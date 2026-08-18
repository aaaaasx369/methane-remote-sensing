from pathlib import Path
import pandas as pd


FILES = {
    "controlled_release_s2": Path("outputs/22_controlled_release_s2_dataset_table.csv"),
    "methaneair_s2_positive": Path("outputs/18_methaneair_s2_dataset_table.csv"),
    "controlled_release_landsat": Path("outputs/32_controlled_release_landsat_dataset_table.csv"),
}

summary_rows = []
label_rows = []

for dataset_group, path in FILES.items():
    if not path.exists():
        print(f"[MISSING] {dataset_group}: {path}")
        continue

    df = pd.read_csv(path)

    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce")

    if dataset_group == "controlled_release_s2":
        sensor = "Sentinel-2"
    elif dataset_group == "methaneair_s2_positive":
        sensor = "Sentinel-2"
    elif dataset_group == "controlled_release_landsat":
        sensor = "Landsat-8/9"
    else:
        sensor = "unknown"

    n_total = len(df)
    n_label0 = int((df["label"] == 0).sum()) if "label" in df.columns else 0
    n_label1 = int((df["label"] == 1).sum()) if "label" in df.columns else 0

    summary_rows.append({
        "dataset_group": dataset_group,
        "sensor": sensor,
        "total_samples": n_total,
        "label_0_no_release": n_label0,
        "label_1_release_or_methane": n_label1,
    })

    if "label" in df.columns:
        counts = df["label"].value_counts(dropna=False)
        for label, count in counts.items():
            label_rows.append({
                "dataset_group": dataset_group,
                "sensor": sensor,
                "label": label,
                "count": int(count),
            })

summary_df = pd.DataFrame(summary_rows)
label_df = pd.DataFrame(label_rows)

summary_out = Path("outputs/33_multisensor_dataset_summary.csv")
label_out = Path("outputs/34_multisensor_label_summary.csv")

summary_df.to_csv(summary_out, index=False)
label_df.to_csv(label_out, index=False)

print("\nMulti-sensor dataset summary:")
print(summary_df.to_string(index=False))

print("\nLabel summary:")
print(label_df.to_string(index=False))

print("\nSaved:")
print(summary_out)
print(label_out)
