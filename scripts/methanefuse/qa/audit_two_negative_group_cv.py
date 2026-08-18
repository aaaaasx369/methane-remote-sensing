from pathlib import Path
import pandas as pd

ROOT = Path("/project/6002520/yunjung1/MethaneFuse")
CV_DIR = ROOT / "outputs/two_negative_group_cv"


def as_bool(series):
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


for fold in [1, 2]:
    baseline_path = (
        CV_DIR / f"fold_{fold}_baseline_manifest.csv"
    )
    augmented_path = (
        CV_DIR / f"fold_{fold}_augmented_manifest.csv"
    )

    baseline = pd.read_csv(
        baseline_path,
        low_memory=False,
    )
    augmented = pd.read_csv(
        augmented_path,
        low_memory=False,
    )

    print(f"\n{'=' * 60}")
    print(f"FOLD {fold}")
    print("=" * 60)

    baseline_test = baseline[
        baseline["split"] == "test"
    ].copy()

    augmented_test = augmented[
        augmented["split"] == "test"
    ].copy()

    baseline_test_ids = set(
        baseline_test["record_id"].astype(str)
    )
    augmented_test_ids = set(
        augmented_test["record_id"].astype(str)
    )

    print(
        "Identical baseline/augmented test records:",
        baseline_test_ids == augmented_test_ids,
    )

    train_groups = set(
        augmented.loc[
            augmented["split"] == "train",
            "group_id",
        ].astype(str)
    )

    test_groups = set(
        augmented.loc[
            augmented["split"] == "test",
            "group_id",
        ].astype(str)
    )

    overlap = train_groups & test_groups

    print("Train groups:", len(train_groups))
    print("Test groups:", len(test_groups))
    print("Group overlap:", len(overlap))

    if overlap:
        print("Leaking groups:")
        print(sorted(overlap)[:20])

    weak_mask = (
        augmented["label_quality"]
        .astype(str)
        .eq("weak_temporal_negative")
    )

    weak = augmented[weak_mask].copy()

    print("Weak negatives:", len(weak))
    print(
        "Weak negatives in train:",
        int((weak["split"] == "train").sum()),
    )
    print(
        "Weak negatives outside train:",
        int((weak["split"] != "train").sum()),
    )

    print("\nBaseline counts:")
    print(
        baseline.groupby(["split", "label"])
        .size()
        .rename("records")
        .to_string()
    )

    print("\nAugmented counts:")
    print(
        augmented.groupby(["split", "label"])
        .size()
        .rename("records")
        .to_string()
    )

    path_columns = [
        column
        for column in [
            "t0_path",
            "t90_path",
            "t360_path",
        ]
        if column in augmented.columns
    ]

    if not path_columns:
        print("\nWARNING: No temporal path columns found.")
        continue

    print("\nImage-path audit:")

    for column in path_columns:
        paths = augmented[column].astype("string")

        missing_value = paths.isna()

        nonexistent = (
            paths[~missing_value]
            .map(lambda value: not Path(value).exists())
        )

        print(
            f"{column}: "
            f"missing={int(missing_value.sum())}, "
            f"nonexistent={int(nonexistent.sum())}"
        )

    if "sample_weight" in augmented.columns:
        print("\nSample weights:")
        print(
            augmented.groupby(
                ["label_quality", "sample_weight"],
                dropna=False,
            )
            .size()
            .rename("records")
            .to_string()
        )
