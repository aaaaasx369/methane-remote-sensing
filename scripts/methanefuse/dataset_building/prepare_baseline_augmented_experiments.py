from pathlib import Path

import pandas as pd

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError as exc:
    raise SystemExit(
        "找不到 scikit-learn，請先確認目前使用 carbonmapper311 環境。"
    ) from exc


ROOT = Path("/project/6002520/yunjung1/MethaneFuse")

CONFIRMED_PATH = (
    ROOT
    / "data/methaneair_full"
    / "sentinel2_v2_full_record_readiness_grouped.csv"
)

WEAK_PATH = (
    ROOT
    / "data/methaneair_full"
    / "negative_pilot50_stage2_qa_pass_grouped.csv"
)

WEAK_MANIFEST_PATH = (
    ROOT
    / "data/methaneair_full"
    / "sentinel2_temporal_manifest_negative_pilot50.csv"
)

OUTPUT_DIR = ROOT / "outputs/negative_augmentation_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def find_first(columns, candidates):
    lookup = {
        str(column).lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


# ============================================================
# 1. Confirmed strict model-ready dataset
# ============================================================

confirmed = pd.read_csv(
    CONFIRMED_PATH,
    low_memory=False,
)

required_confirmed = {
    "record_id",
    "label",
    "strict_model_ready",
}

missing = required_confirmed - set(confirmed.columns)

if missing:
    raise SystemExit(
        f"Confirmed table 缺少欄位：{sorted(missing)}"
    )

confirmed = confirmed[
    as_bool(confirmed["strict_model_ready"])
].copy()

confirmed["label"] = pd.to_numeric(
    confirmed["label"],
    errors="coerce",
)

confirmed = confirmed[
    confirmed["label"].isin([0, 1])
].copy()

confirmed["label"] = confirmed["label"].astype(int)

print("Confirmed strict model-ready:", len(confirmed))
print(confirmed["label"].value_counts().sort_index())

if len(confirmed) != 506:
    print(
        "WARNING：strict model-ready 不等於預期的 506，"
        f"目前為 {len(confirmed)}。"
    )


# ============================================================
# 2. Select a leakage-control group
# ============================================================

group_col = find_first(
    confirmed.columns,
    [
        "group_id",
        "site_id",
        "facility_id",
        "site",
        "source_site_id",
        "location_id",
    ],
)

if group_col is None:
    raise SystemExit(
        "Confirmed table 沒有 site_id／facility_id。"
        "請先不要使用 record-level 隨機切分，否則可能場址洩漏。"
    )

confirmed[group_col] = (
    confirmed[group_col]
    .astype("string")
    .fillna("missing_group")
)

group_summary = (
    confirmed
    .groupby(group_col, dropna=False)
    .agg(
        records=("record_id", "size"),
        positives=("label", lambda x: int((x == 1).sum())),
        negatives=("label", lambda x: int((x == 0).sum())),
    )
    .reset_index()
)

print("\nGrouping column:", group_col)
print("Unique groups:", confirmed[group_col].nunique())
print(
    "Groups containing negatives:",
    int((group_summary["negatives"] > 0).sum()),
)

if (group_summary["negatives"] > 0).sum() < 3:
    raise SystemExit(
        "具有 negative 的場址少於 3 個，"
        "無法安全建立 train/validation/test site split。"
    )


# ============================================================
# 3. Create a fixed confirmed-only split
#
# Approximate split:
# train 60%, validation 20%, test 20%
# ============================================================

outer = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=2026,
)

outer_splits = list(
    outer.split(
        confirmed,
        y=confirmed["label"],
        groups=confirmed[group_col],
    )
)

# Select the outer fold whose test set contains both classes
# and has a negative proportion closest to the full dataset.
overall_negative_rate = float(
    (confirmed["label"] == 0).mean()
)

best_outer = None
best_outer_score = None

for trainval_idx, test_idx in outer_splits:
    test = confirmed.iloc[test_idx]

    if test["label"].nunique() < 2:
        continue

    score = abs(
        float((test["label"] == 0).mean())
        - overall_negative_rate
    )

    if best_outer_score is None or score < best_outer_score:
        best_outer_score = score
        best_outer = (trainval_idx, test_idx)

if best_outer is None:
    raise SystemExit(
        "找不到同時含 positive 與 negative 的 test fold。"
    )

trainval_idx, test_idx = best_outer

trainval = confirmed.iloc[trainval_idx].copy()
test = confirmed.iloc[test_idx].copy()

inner = StratifiedGroupKFold(
    n_splits=4,
    shuffle=True,
    random_state=2027,
)

inner_splits = list(
    inner.split(
        trainval,
        y=trainval["label"],
        groups=trainval[group_col],
    )
)

trainval_negative_rate = float(
    (trainval["label"] == 0).mean()
)

best_inner = None
best_inner_score = None

for train_idx, val_idx in inner_splits:
    train_candidate = trainval.iloc[train_idx]
    val_candidate = trainval.iloc[val_idx]

    if train_candidate["label"].nunique() < 2:
        continue

    if val_candidate["label"].nunique() < 2:
        continue

    score = abs(
        float((val_candidate["label"] == 0).mean())
        - trainval_negative_rate
    )

    if best_inner_score is None or score < best_inner_score:
        best_inner_score = score
        best_inner = (train_idx, val_idx)

if best_inner is None:
    raise SystemExit(
        "找不到同時含 positive 與 negative 的 validation fold。"
    )

train_idx, val_idx = best_inner

train = trainval.iloc[train_idx].copy()
validation = trainval.iloc[val_idx].copy()

train["split"] = "train"
validation["split"] = "validation"
test["split"] = "test"

confirmed_split = pd.concat(
    [train, validation, test],
    ignore_index=True,
)

# Check group leakage
group_split_counts = (
    confirmed_split
    .groupby(group_col)["split"]
    .nunique()
)

leaking_groups = group_split_counts[
    group_split_counts > 1
]

if len(leaking_groups):
    raise SystemExit(
        f"偵測到 {len(leaking_groups)} 個 group 跨 split。"
    )


# ============================================================
# 4. Prepare baseline manifest
# ============================================================

baseline = confirmed_split.copy()

baseline["label_quality"] = "confirmed"
baseline["experiment_source"] = (
    "strict_model_ready_confirmed"
)
baseline["sample_weight"] = 1.0

baseline_path = (
    OUTPUT_DIR
    / "baseline_confirmed_manifest.csv"
)

baseline.to_csv(
    baseline_path,
    index=False,
)


# ============================================================
# 5. Prepare the 33 weak candidate negatives
# ============================================================

weak = pd.read_csv(
    WEAK_PATH,
    low_memory=False,
)

weak_manifest = pd.read_csv(
    WEAK_MANIFEST_PATH,
    low_memory=False,
)

if "record_id" not in weak.columns:
    raise SystemExit(
        "Weak-negative table 缺少 record_id。"
    )

if "record_id" not in weak_manifest.columns:
    raise SystemExit(
        "Weak manifest 缺少 record_id。"
    )

# Bring temporal paths and scene metadata back into the weak table.
manifest_columns = [
    "record_id",
    "t0_path",
    "t0_scl_path",
    "t0_scene_id",
    "t0_scene_time_utc",
    "t0_clear_fraction",
    "t0_qa_pass",
    "t0_status",
    "t90_path",
    "t90_scl_path",
    "t90_scene_id",
    "t90_scene_time_utc",
    "t90_clear_fraction",
    "t90_qa_pass",
    "t90_status",
    "t360_path",
    "t360_scl_path",
    "t360_scene_id",
    "t360_scene_time_utc",
    "t360_clear_fraction",
    "t360_qa_pass",
    "t360_status",
    "all_three_downloaded",
    "all_three_qa_pass",
]

manifest_columns = [
    column
    for column in manifest_columns
    if column in weak_manifest.columns
]

# Avoid duplicate columns from the earlier Stage-2 table.
merge_columns = [
    column
    for column in manifest_columns
    if column == "record_id"
    or column not in weak.columns
]

weak = weak.merge(
    weak_manifest[merge_columns],
    on="record_id",
    how="left",
    validate="one_to_one",
)

if "cloud_snow_qa_pass" in weak.columns:
    weak = weak[
        as_bool(weak["cloud_snow_qa_pass"])
    ].copy()
elif "all_three_qa_pass" in weak.columns:
    weak = weak[
        as_bool(weak["all_three_qa_pass"])
    ].copy()

print("\nQA-passed weak candidates:", len(weak))

if len(weak) != 33:
    print(
        "WARNING：QA-passed weak candidate 不等於 33，"
        f"目前為 {len(weak)}。"
    )

weak_group_col = find_first(
    weak.columns,
    [
        group_col,
        "site_id",
        "facility_id",
        "site",
        "source_site_id",
    ],
)

if weak_group_col is None:
    raise SystemExit(
        "Weak-negative table 找不到與 confirmed 相容的場址欄位。"
    )

weak[weak_group_col] = (
    weak[weak_group_col]
    .astype("string")
    .fillna("missing_group")
)

train_groups = set(
    train[group_col].astype(str)
)

validation_groups = set(
    validation[group_col].astype(str)
)

test_groups = set(
    test[group_col].astype(str)
)

weak["_group_text"] = weak[
    weak_group_col
].astype(str)

# Weak candidates only enter training sites.
weak_train = weak[
    weak["_group_text"].isin(train_groups)
].copy()

excluded_val_site = weak[
    weak["_group_text"].isin(validation_groups)
].copy()

excluded_test_site = weak[
    weak["_group_text"].isin(test_groups)
].copy()

unmatched_site = weak[
    ~weak["_group_text"].isin(
        train_groups
        | validation_groups
        | test_groups
    )
].copy()

weak_train = weak_train.drop(
    columns=["_group_text"]
)

weak_train["label"] = 0
weak_train["split"] = "train"
weak_train["label_quality"] = "weak_temporal_negative"
weak_train["experiment_source"] = (
    "methaneair_plus7day_candidate"
)

# Keep a weight column for later use.
# The training pipeline may initially ignore it.
weak_train["sample_weight"] = 0.5


# ============================================================
# 6. Prepare augmented manifest
# ============================================================

augmented = pd.concat(
    [
        baseline,
        weak_train,
    ],
    ignore_index=True,
    sort=False,
)

augmented_path = (
    OUTPUT_DIR
    / "augmented_weak_negative_manifest.csv"
)

augmented.to_csv(
    augmented_path,
    index=False,
)

weak_train_path = (
    OUTPUT_DIR
    / "weak_negative_train_only.csv"
)

weak_train.to_csv(
    weak_train_path,
    index=False,
)


# ============================================================
# 7. Summaries
# ============================================================

def summarize(df, experiment):
    result = (
        df.groupby(
            ["split", "label"],
            dropna=False,
        )
        .size()
        .rename("records")
        .reset_index()
    )

    result.insert(
        0,
        "experiment",
        experiment,
    )

    return result


summary = pd.concat(
    [
        summarize(baseline, "baseline"),
        summarize(augmented, "augmented"),
    ],
    ignore_index=True,
)

summary_path = (
    OUTPUT_DIR
    / "experiment_split_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

exclusion_summary = pd.DataFrame(
    [
        {
            "status": "weak_total_qa_passed",
            "records": len(weak),
        },
        {
            "status": "weak_added_to_train",
            "records": len(weak_train),
        },
        {
            "status": "excluded_validation_site",
            "records": len(excluded_val_site),
        },
        {
            "status": "excluded_test_site",
            "records": len(excluded_test_site),
        },
        {
            "status": "unmatched_site",
            "records": len(unmatched_site),
        },
    ]
)

exclusion_summary.to_csv(
    OUTPUT_DIR
    / "weak_negative_split_audit.csv",
    index=False,
)

print("\n=== CONFIRMED SPLIT ===")
print(
    summarize(
        baseline,
        "baseline",
    ).to_string(index=False)
)

print("\n=== AUGMENTED SPLIT ===")
print(
    summarize(
        augmented,
        "augmented",
    ).to_string(index=False)
)

print("\n=== WEAK NEGATIVE AUDIT ===")
print(exclusion_summary.to_string(index=False))

print("\nSaved:")
print(baseline_path)
print(augmented_path)
print(weak_train_path)
print(summary_path)
