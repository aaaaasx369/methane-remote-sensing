from pathlib import Path
import pandas as pd


ROOT = Path("/project/6002520/yunjung1/MethaneFuse")
DATA = ROOT / "data/methaneair_full"
OUTPUT = ROOT / "outputs/two_negative_group_cv"
OUTPUT.mkdir(parents=True, exist_ok=True)

CONFIRMED_PATH = (
    DATA / "sentinel2_v2_full_record_readiness_grouped.csv"
)

WEAK_PATH = (
    DATA / "negative_pilot50_stage2_qa_pass_grouped.csv"
)


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


# ============================================================
# 1. Confirmed strict model-ready data
# ============================================================

confirmed = pd.read_csv(CONFIRMED_PATH, low_memory=False)

required = {
    "record_id",
    "group_id",
    "label",
    "strict_model_ready",
}

missing = required - set(confirmed.columns)

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
confirmed["group_id"] = confirmed["group_id"].astype(str)

if confirmed["record_id"].duplicated().any():
    raise SystemExit(
        "Confirmed record_id 有重複值。"
    )

print("Confirmed strict model-ready:", len(confirmed))
print(confirmed["label"].value_counts().sort_index())


# ============================================================
# 2. Identify the two negative groups
# ============================================================

group_summary = (
    confirmed.groupby("group_id", dropna=False)
    .agg(
        records=("record_id", "size"),
        positives=("label", lambda x: int((x == 1).sum())),
        negatives=("label", lambda x: int((x == 0).sum())),
    )
    .reset_index()
)

negative_summary = (
    group_summary[
        group_summary["negatives"] > 0
    ]
    .sort_values("group_id")
    .reset_index(drop=True)
)

print("\nNegative groups:")
print(negative_summary.to_string(index=False))

if len(negative_summary) != 2:
    raise SystemExit(
        "\n這支程式預期正好有 2 個 negative groups，"
        f"目前找到 {len(negative_summary)} 個。"
    )

negative_groups = negative_summary["group_id"].tolist()

positive_only_summary = group_summary[
    group_summary["negatives"] == 0
].copy()


# ============================================================
# 3. Divide positive-only groups between two test folds
#
# Start each fold with one negative group, then greedily assign
# positive-only groups so total test sizes remain balanced.
# Every confirmed group appears in test exactly once.
# ============================================================

negative_group_sizes = {
    row["group_id"]: int(row["records"])
    for _, row in negative_summary.iterrows()
}

test_group_bins = [
    {negative_groups[0]},
    {negative_groups[1]},
]

test_size_totals = [
    negative_group_sizes[negative_groups[0]],
    negative_group_sizes[negative_groups[1]],
]

positive_groups_sorted = (
    positive_only_summary
    .sort_values(
        ["records", "group_id"],
        ascending=[False, True],
    )
)

for _, row in positive_groups_sorted.iterrows():
    target_fold = (
        0 if test_size_totals[0] <= test_size_totals[1]
        else 1
    )

    group_id = str(row["group_id"])
    group_size = int(row["records"])

    test_group_bins[target_fold].add(group_id)
    test_size_totals[target_fold] += group_size

print("\nPlanned test sizes:")
print("Fold 1:", test_size_totals[0])
print("Fold 2:", test_size_totals[1])


# ============================================================
# 4. Weak negative data
# ============================================================

weak = pd.read_csv(WEAK_PATH, low_memory=False)

required_weak = {
    "record_id",
    "group_id",
}

missing_weak = required_weak - set(weak.columns)

if missing_weak:
    raise SystemExit(
        f"Weak table 缺少欄位：{sorted(missing_weak)}"
    )

weak["group_id"] = weak["group_id"].astype(str)
weak["label"] = 0
weak["label_quality"] = "weak_temporal_negative"
weak["experiment_source"] = (
    "methaneair_temporal_candidate"
)
weak["sample_weight"] = 0.5

print("\nQA-passed weak negatives:", len(weak))


# ============================================================
# 5. Build the two folds
# ============================================================

summary_rows = []
weak_audit_rows = []

all_groups = set(confirmed["group_id"])

for fold_number in [1, 2]:
    test_groups = test_group_bins[fold_number - 1]
    train_groups = all_groups - test_groups

    baseline = confirmed.copy()

    baseline["split"] = baseline["group_id"].map(
        lambda group: (
            "test" if group in test_groups else "train"
        )
    )

    baseline["label_quality"] = "confirmed"
    baseline["experiment_source"] = (
        "strict_model_ready_confirmed"
    )
    baseline["sample_weight"] = 1.0

    train = baseline[
        baseline["split"] == "train"
    ]

    test = baseline[
        baseline["split"] == "test"
    ]

    for split_name, split_df in [
        ("train", train),
        ("test", test),
    ]:
        labels = set(split_df["label"].unique())

        if labels != {0, 1}:
            raise SystemExit(
                f"Fold {fold_number} 的 {split_name} "
                f"沒有同時包含兩個 classes：{labels}"
            )

    # Verify no group leakage.
    train_group_set = set(train["group_id"])
    test_group_set = set(test["group_id"])

    overlap = train_group_set & test_group_set

    if overlap:
        raise SystemExit(
            f"Fold {fold_number} 發現 group leakage："
            f"{sorted(overlap)[:10]}"
        )

    # Weak negatives can only enter training groups.
    weak_train = weak[
        weak["group_id"].isin(train_group_set)
    ].copy()

    weak_excluded_test = weak[
        weak["group_id"].isin(test_group_set)
    ].copy()

    weak_train["split"] = "train"

    augmented = pd.concat(
        [baseline, weak_train],
        ignore_index=True,
        sort=False,
    )

    baseline_path = (
        OUTPUT
        / f"fold_{fold_number}_baseline_manifest.csv"
    )

    augmented_path = (
        OUTPUT
        / f"fold_{fold_number}_augmented_manifest.csv"
    )

    weak_path = (
        OUTPUT
        / f"fold_{fold_number}_weak_train_only.csv"
    )

    baseline.to_csv(baseline_path, index=False)
    augmented.to_csv(augmented_path, index=False)
    weak_train.to_csv(weak_path, index=False)

    for experiment, table in [
        ("baseline", baseline),
        ("augmented", augmented),
    ]:
        counts = (
            table.groupby(["split", "label"])
            .size()
            .rename("records")
            .reset_index()
        )

        for _, row in counts.iterrows():
            summary_rows.append(
                {
                    "fold": fold_number,
                    "experiment": experiment,
                    "split": row["split"],
                    "label": int(row["label"]),
                    "records": int(row["records"]),
                }
            )

    weak_audit_rows.append(
        {
            "fold": fold_number,
            "weak_total": len(weak),
            "weak_added_to_train": len(weak_train),
            "weak_excluded_test_group": len(
                weak_excluded_test
            ),
            "held_out_negative_group": (
                negative_groups[fold_number - 1]
            ),
        }
    )

    print(f"\n=== FOLD {fold_number} BASELINE ===")
    print(
        baseline.groupby(["split", "label"])
        .size()
        .rename("records")
        .to_string()
    )

    print(f"\n=== FOLD {fold_number} AUGMENTED ===")
    print(
        augmented.groupby(["split", "label"])
        .size()
        .rename("records")
        .to_string()
    )

    print(
        "\nWeak negatives added to train:",
        len(weak_train),
    )

    print(
        "Weak negatives excluded because their group is test:",
        len(weak_excluded_test),
    )


# ============================================================
# 6. Save audit summaries
# ============================================================

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    OUTPUT / "two_fold_split_summary.csv",
    index=False,
)

weak_audit = pd.DataFrame(weak_audit_rows)
weak_audit.to_csv(
    OUTPUT / "weak_negative_fold_audit.csv",
    index=False,
)

group_summary.to_csv(
    OUTPUT / "confirmed_group_summary.csv",
    index=False,
)

print("\n=== WEAK NEGATIVE FOLD AUDIT ===")
print(weak_audit.to_string(index=False))

print("\nSaved under:")
print(OUTPUT)
