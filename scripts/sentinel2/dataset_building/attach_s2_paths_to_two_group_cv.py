from pathlib import Path
import pandas as pd


ROOT = Path(
    "/project/6002520/yunjung1/MethaneFuse"
)

DATA = ROOT / "data/methaneair_full"

CV_DIR = (
    ROOT / "outputs/two_negative_group_cv"
)

CONFIRMED_MANIFEST = (
    DATA / "sentinel2_temporal_manifest_best_qa_v2.csv"
)

WEAK_MANIFEST = (
    DATA / "sentinel2_temporal_manifest_negative_pilot50.csv"
)


def detect_column(columns, candidates):
    lookup = {
        str(column).lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def resolve_path(value, source_file):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()

    if not text:
        return pd.NA

    path = Path(text).expanduser()

    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                ROOT / path,
                source_file.parent / path,
                DATA / path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    # Preserve the most likely absolute path for audit.
    if path.is_absolute():
        return str(path)

    return str((ROOT / path).resolve())


def build_path_map(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    id_column = detect_column(
        df.columns,
        [
            "record_id",
            "id",
        ],
    )

    aliases = {
        "s2_0_path": [
            "s2_0_path",
            "t0_path",
            "t0_image_path",
            "image_t0_path",
        ],
        "s2_90_path": [
            "s2_90_path",
            "t90_path",
            "t90_image_path",
            "image_t90_path",
        ],
        "s2_360_path": [
            "s2_360_path",
            "t360_path",
            "t360_image_path",
            "image_t360_path",
        ],
    }

    detected = {
        target: detect_column(
            df.columns,
            candidates,
        )
        for target, candidates in aliases.items()
    }

    print("\nSource:", path)
    print("ID column:", id_column)
    print("Detected paths:", detected)

    if id_column is None:
        raise SystemExit(
            f"{path.name} 找不到 record_id/id。"
        )

    missing = [
        target
        for target, source in detected.items()
        if source is None
    ]

    if missing:
        path_like = [
            column
            for column in df.columns
            if any(
                word in column.lower()
                for word in [
                    "path",
                    "file",
                    "tif",
                    "image",
                    "asset",
                ]
            )
        ]

        raise SystemExit(
            f"{path.name} 找不到 {missing}。\n"
            f"目前可能的路徑欄位：{path_like}"
        )

    output = pd.DataFrame(
        {
            "record_id": (
                df[id_column]
                .astype(str)
                .str.strip()
            )
        }
    )

    for target, source in detected.items():
        output[target] = df[source].map(
            lambda value: resolve_path(
                value,
                path,
            )
        )

    optional_mapping = {
        "t0_scene_id": [
            "t0_scene_id",
            "s2_0_scene_id",
        ],
        "t0_scene_time_utc": [
            "t0_scene_time_utc",
            "s2_0_scene_time_utc",
        ],
    }

    for target, candidates in optional_mapping.items():
        source = detect_column(
            df.columns,
            candidates,
        )

        if source is not None:
            output[target] = df[source]

    output = output.drop_duplicates(
        subset=["record_id"],
        keep="first",
    )

    return output


confirmed_paths = build_path_map(
    CONFIRMED_MANIFEST
)

weak_paths = build_path_map(
    WEAK_MANIFEST
)

combined_paths = pd.concat(
    [
        confirmed_paths.assign(
            path_source="confirmed_full_v2"
        ),
        weak_paths.assign(
            path_source="weak_negative_pilot50"
        ),
    ],
    ignore_index=True,
    sort=False,
)

duplicates = combined_paths[
    combined_paths["record_id"].duplicated(
        keep=False
    )
]

if len(duplicates):
    print(
        "\nOverlapping record IDs:",
        duplicates["record_id"].nunique(),
    )

    # Confirmed record paths take priority.
    combined_paths["_priority"] = (
        combined_paths["path_source"]
        .map(
            {
                "confirmed_full_v2": 0,
                "weak_negative_pilot50": 1,
            }
        )
        .fillna(99)
    )

    combined_paths = (
        combined_paths
        .sort_values(
            ["record_id", "_priority"]
        )
        .drop_duplicates(
            "record_id",
            keep="first",
        )
        .drop(columns=["_priority"])
    )


input_names = [
    "fold_1_baseline_manifest.csv",
    "fold_1_augmented_manifest.csv",
    "fold_2_baseline_manifest.csv",
    "fold_2_augmented_manifest.csv",
]

for input_name in input_names:
    input_path = CV_DIR / input_name

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(
        input_path,
        low_memory=False,
    )

    # Remove stale versions before merge.
    stale_columns = [
        column
        for column in [
            "s2_0_path",
            "s2_90_path",
            "s2_360_path",
            "t0_scene_id",
            "t0_scene_time_utc",
            "path_source",
        ]
        if column in df.columns
    ]

    if stale_columns:
        df = df.drop(
            columns=stale_columns
        )

    df["record_id"] = (
        df["record_id"]
        .astype(str)
        .str.strip()
    )

    merged = df.merge(
        combined_paths,
        on="record_id",
        how="left",
        validate="many_to_one",
    )

    merged["id"] = merged["record_id"]

    if "site_id" not in merged.columns:
        merged["site_id"] = merged["group_id"]
    else:
        merged["site_id"] = (
            merged["site_id"]
            .astype("string")
            .fillna(
                merged["group_id"]
                .astype("string")
            )
        )

    merged["source_scene_id"] = merged.get(
        "t0_scene_id",
        pd.Series(
            pd.NA,
            index=merged.index,
        ),
    )

    merged["source_acquisition_time_utc"] = (
        merged.get(
            "t0_scene_time_utc",
            pd.Series(
                pd.NA,
                index=merged.index,
            ),
        )
    )

    merged["source_tiff_path"] = (
        merged["s2_0_path"]
    )

    merged["source_path_resolution_method"] = (
        "merged_from_sentinel2_temporal_manifest"
    )

    merged["smoke_test_only"] = False

    merged["label_provenance"] = (
        merged.get(
            "label_quality",
            pd.Series(
                "unknown",
                index=merged.index,
            ),
        )
    )

    merged["five_site_experiment_scope"] = (
        "two_negative_group_baseline_augmented"
    )

    required = [
        "id",
        "label",
        "s2_0_path",
        "s2_90_path",
        "s2_360_path",
        "site_id",
    ]

    missing_values = {
        column: int(
            merged[column].isna().sum()
        )
        for column in required
    }

    nonexistent = {}

    for column in [
        "s2_0_path",
        "s2_90_path",
        "s2_360_path",
    ]:
        paths = merged[column].dropna()

        nonexistent[column] = int(
            paths.map(
                lambda value: not Path(
                    str(value)
                ).exists()
            ).sum()
        )

    output_path = input_path.with_name(
        input_path.stem
        + "_with_paths.csv"
    )

    merged.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print(output_path.name)
    print("Rows:", len(merged))
    print("Missing required values:", missing_values)
    print("Nonexistent files:", nonexistent)
    print("Path sources:")
    print(
        merged["path_source"]
        .value_counts(dropna=False)
        .to_string()
    )
