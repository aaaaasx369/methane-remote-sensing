from pathlib import Path
import pandas as pd


ROOT = Path("/project/6002520/yunjung1/MethaneFuse")
SOURCE = ROOT / "outputs/two_negative_group_cv"
OUTPUT = ROOT / "data/custom/two_negative_group_cv"
OUTPUT.mkdir(parents=True, exist_ok=True)

VAL_FRACTION = 0.15
REFERENCE_COLUMNS = [
    "id",
    "label",
    "s2_0_path",
    "s2_90_path",
    "s2_360_path",
    "site_id",
    "source_scene_id",
    "source_acquisition_time_utc",
    "source_tiff_path",
    "source_path_resolution_method",
    "smoke_test_only",
    "label_provenance",
    "five_site_experiment_scope",
]


def select_validation_ids(
    train_pool: pd.DataFrame,
    seed: int,
) -> set[str]:
    """Create a deterministic class-stratified validation subset."""
    selected = []

    for label, part in train_pool.groupby("label"):
        if len(part) < 2:
            raise RuntimeError(
                f"Label {label} has only {len(part)} training record."
            )

        count = max(
            1,
            int(round(len(part) * VAL_FRACTION)),
        )

        # Always leave at least one record of each class in training.
        count = min(count, len(part) - 1)

        sampled = part.sample(
            n=count,
            random_state=seed + int(label),
        )

        selected.extend(
            sampled["record_id"].astype(str).tolist()
        )

    return set(selected)


def to_model_schema(
    df: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    out = df.copy()

    out["id"] = out["record_id"].astype(str)
    out["label"] = pd.to_numeric(
        out["label"],
        errors="raise",
    ).astype(int)

    out["site_id"] = (
        out["site_id"]
        .astype("string")
        .fillna(out["group_id"].astype("string"))
    )

    if "source_scene_id" not in out.columns:
        out["source_scene_id"] = out.get(
            "t0_scene_id",
            pd.Series(pd.NA, index=out.index),
        )

    if "source_acquisition_time_utc" not in out.columns:
        out["source_acquisition_time_utc"] = out.get(
            "t0_scene_time_utc",
            pd.Series(pd.NA, index=out.index),
        )

    out["source_tiff_path"] = out["s2_0_path"]

    out["source_path_resolution_method"] = (
        "merged_from_sentinel2_temporal_manifest"
    )

    out["smoke_test_only"] = False

    out["label_provenance"] = out.get(
        "label_quality",
        pd.Series("unknown", index=out.index),
    )

    out["five_site_experiment_scope"] = scope

    missing_columns = [
        column
        for column in REFERENCE_COLUMNS
        if column not in out.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"Missing model columns: {missing_columns}"
        )

    result = out[REFERENCE_COLUMNS].copy()

    if result["id"].duplicated().any():
        duplicates = result.loc[
            result["id"].duplicated(False),
            "id",
        ].tolist()

        raise RuntimeError(
            f"Duplicate IDs: {duplicates[:10]}"
        )

    for column in [
        "s2_0_path",
        "s2_90_path",
        "s2_360_path",
    ]:
        if result[column].isna().any():
            raise RuntimeError(
                f"{scope}: {column} contains missing values."
            )

        missing_files = [
            path
            for path in result[column].astype(str)
            if not Path(path).exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                f"{scope}: {column} missing file: "
                f"{missing_files[0]}"
            )

    return result


run_records = []
audit_records = []

for fold in [1, 2]:
    baseline_path = (
        SOURCE
        / f"fold_{fold}_baseline_manifest_with_paths.csv"
    )

    augmented_path = (
        SOURCE
        / f"fold_{fold}_augmented_manifest_with_paths.csv"
    )

    baseline = pd.read_csv(
        baseline_path,
        low_memory=False,
    )

    augmented = pd.read_csv(
        augmented_path,
        low_memory=False,
    )

    baseline["record_id"] = baseline["record_id"].astype(str)
    augmented["record_id"] = augmented["record_id"].astype(str)

    confirmed_train_pool = baseline[
        baseline["split"] == "train"
    ].copy()

    confirmed_test = baseline[
        baseline["split"] == "test"
    ].copy()

    validation_ids = select_validation_ids(
        confirmed_train_pool,
        seed=20260 + fold,
    )

    confirmed_validation = confirmed_train_pool[
        confirmed_train_pool["record_id"].isin(
            validation_ids
        )
    ].copy()

    confirmed_train = confirmed_train_pool[
        ~confirmed_train_pool["record_id"].isin(
            validation_ids
        )
    ].copy()

    weak_mask = (
        augmented.get(
            "label_quality",
            pd.Series("", index=augmented.index),
        )
        .astype(str)
        .eq("weak_temporal_negative")
        |
        augmented.get(
            "path_source",
            pd.Series("", index=augmented.index),
        )
        .astype(str)
        .eq("weak_negative_pilot50")
    )

    weak_train = augmented[
        weak_mask
        & augmented["split"].eq("train")
    ].copy()

    held_out_groups = sorted(
        confirmed_test.loc[
            confirmed_test["label"] == 0,
            "group_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    if len(held_out_groups) != 1:
        raise RuntimeError(
            f"Fold {fold} expected one held-out negative group, "
            f"found {held_out_groups}"
        )

    held_out_site = held_out_groups[0]

    experiments = {
        "baseline": confirmed_train,
        "augmented": pd.concat(
            [confirmed_train, weak_train],
            ignore_index=True,
            sort=False,
        ),
    }

    for experiment, training_data in experiments.items():
        run_slug = f"fold_{fold}_{experiment}"
        run_dir = OUTPUT / run_slug
        run_dir.mkdir(parents=True, exist_ok=True)

        train_model = to_model_schema(
            training_data,
            scope=f"{run_slug}_train",
        )

        val_model = to_model_schema(
            confirmed_validation,
            scope=f"{run_slug}_validation",
        )

        test_model = to_model_schema(
            confirmed_test,
            scope=f"{run_slug}_test",
        )

        train_path = run_dir / "train.csv"
        val_path = run_dir / "val.csv"
        test_path = run_dir / "test.csv"

        train_model.to_csv(train_path, index=False)
        val_model.to_csv(val_path, index=False)
        test_model.to_csv(test_path, index=False)

        run_records.append({
            "fold_index": len(run_records),
            "fold_slug": run_slug,
            "held_out_site": held_out_site,
            "train_csv": f"{run_slug}/train.csv",
            "val_csv": f"{run_slug}/val.csv",
            "test_csv": f"{run_slug}/test.csv",
        })

        for split_name, table in [
            ("train", train_model),
            ("validation", val_model),
            ("test", test_model),
        ]:
            counts = table["label"].value_counts()

            audit_records.append({
                "fold": fold,
                "experiment": experiment,
                "split": split_name,
                "records": len(table),
                "positive": int(counts.get(1, 0)),
                "negative": int(counts.get(0, 0)),
                "weak_negative": int(
                    table["label_provenance"]
                    .astype(str)
                    .eq("weak_temporal_negative")
                    .sum()
                ),
            })


runs = pd.DataFrame(run_records)
audit = pd.DataFrame(audit_records)

runs.to_csv(
    OUTPUT / "folds.csv",
    index=False,
)

runs[
    [
        "fold_index",
        "fold_slug",
        "held_out_site",
        "train_csv",
        "val_csv",
        "test_csv",
    ]
].to_csv(
    OUTPUT / "folds.tsv",
    sep="\t",
    index=False,
    header=False,
)

audit.to_csv(
    OUTPUT / "split_audit.csv",
    index=False,
)

print("=== RUNS ===")
print(runs.to_string(index=False))

print("\n=== SPLIT AUDIT ===")
print(audit.to_string(index=False))

print("\nSaved under:")
print(OUTPUT)
