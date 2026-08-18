from pathlib import Path
import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/80_s2_verified_emission_predictions.csv"
)

OUTPUT = Path(
    "outputs/81_s2_recall_by_emission_bin.csv"
)

if not INPUT.exists():
    raise FileNotFoundError(
        f"找不到輸入檔：{INPUT.resolve()}"
    )

df = pd.read_csv(INPUT, low_memory=False)

required = [
    "metered_release_rate_kg_hr",
    "true_label",
    "predicted_label",
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"缺少必要欄位：{missing}\n"
        f"現有欄位：{df.columns.tolist()}"
    )

# 只使用真正有時間重疊的 controlled-release rows
if "exact_release_overlap" in df.columns:
    overlap = (
        df["exact_release_overlap"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    df = df[overlap].copy()

df["metered_release_rate_kg_hr"] = pd.to_numeric(
    df["metered_release_rate_kg_hr"],
    errors="coerce",
)

df["true_label"] = pd.to_numeric(
    df["true_label"],
    errors="coerce",
)

df["predicted_label"] = pd.to_numeric(
    df["predicted_label"],
    errors="coerce",
)

df = df.dropna(
    subset=[
        "metered_release_rate_kg_hr",
        "true_label",
        "predicted_label",
    ]
).copy()

df["true_label"] = df["true_label"].astype(int)
df["predicted_label"] = df["predicted_label"].astype(int)

bins = [
    -0.001,
    0,
    100,
    200,
    500,
    1000,
    2000,
    np.inf,
]

labels = [
    "0",
    "0-100",
    "100-200",
    "200-500",
    "500-1000",
    "1000-2000",
    ">2000",
]

df["emission_bin_kg_hr"] = pd.cut(
    df["metered_release_rate_kg_hr"],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=True,
)

records = []

for emission_bin, group in df.groupby(
    "emission_bin_kg_hr",
    observed=False,
):
    if len(group) == 0:
        continue

    true_positive = int(
        (
            (group["true_label"] == 1)
            & (group["predicted_label"] == 1)
        ).sum()
    )

    false_negative = int(
        (
            (group["true_label"] == 1)
            & (group["predicted_label"] == 0)
        ).sum()
    )

    true_negative = int(
        (
            (group["true_label"] == 0)
            & (group["predicted_label"] == 0)
        ).sum()
    )

    false_positive = int(
        (
            (group["true_label"] == 0)
            & (group["predicted_label"] == 1)
        ).sum()
    )

    positive_support = true_positive + false_negative
    negative_support = true_negative + false_positive

    recall = (
        true_positive / positive_support
        if positive_support > 0
        else np.nan
    )

    specificity = (
        true_negative / negative_support
        if negative_support > 0
        else np.nan
    )

    record = {
        "emission_bin_kg_hr": str(emission_bin),
        "total_samples": len(group),
        "positive_support": positive_support,
        "negative_support": negative_support,
        "tp": true_positive,
        "fn": false_negative,
        "tn": true_negative,
        "fp": false_positive,
        "recall": recall,
        "specificity": specificity,
        "minimum_rate_kg_hr":
            group["metered_release_rate_kg_hr"].min(),
        "median_rate_kg_hr":
            group["metered_release_rate_kg_hr"].median(),
        "maximum_rate_kg_hr":
            group["metered_release_rate_kg_hr"].max(),
    }

    if "positive_probability" in group.columns:
        record["mean_positive_probability"] = pd.to_numeric(
            group["positive_probability"],
            errors="coerce",
        ).mean()

    records.append(record)

result = pd.DataFrame(records)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT,
    index=False,
)

print("Verified rows:", len(df))
print()
print(result.to_string(index=False))
print()
print("Created:", OUTPUT.resolve())
