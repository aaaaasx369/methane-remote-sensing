from pathlib import Path

import numpy as np
import pandas as pd


ACTIVE_MANIFEST_INPUT = Path(
    "outputs/260_marss2l_development_active_manifest_v1_1.csv"
)

DOWNLOAD_INDEX_INPUT = Path(
    "outputs/257_marss2l_development_patch_index.csv"
)

FINAL_INDEX_OUTPUT = Path(
    "outputs/263_marss2l_development_clean_index_v2.csv"
)

FINAL_MANIFEST_OUTPUT = Path(
    "outputs/264_marss2l_development_clean_manifest_v2.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/265_marss2l_development_cleaning_audit_v2.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/266_marss2l_development_clean_summary_v2.csv"
)


QA_THRESHOLD = 0.8
MIN_CALIBRATION_NEGATIVES = 5
MIN_MODEL_NEGATIVES = 4
MIN_MODEL_POSITIVES = 1

DATASET_VERSION = (
    "marss2l_development_clean_v2_"
    "qa080_site_disjoint"
)


def days_to_nearest_positive(
    timestamp,
    positive_times,
):
    if pd.isna(timestamp) or not positive_times:
        return np.nan

    return min(
        abs(
            (
                timestamp
                - positive_time
            ).total_seconds()
        )
        for positive_time in positive_times
    ) / 86400.0


def add_missing_role_columns(table):
    for column in [
        "calibration_negative",
        "model_negative",
        "model_positive",
    ]:
        if column not in table.columns:
            table[column] = 0

    return table


def main():
    for path in [
        ACTIVE_MANIFEST_INPUT,
        DOWNLOAD_INDEX_INPUT,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = pd.read_csv(
        ACTIVE_MANIFEST_INPUT,
        low_memory=False,
    )

    index = pd.read_csv(
        DOWNLOAD_INDEX_INPUT,
        low_memory=False,
    )

    manifest["download_id"] = (
        manifest["download_id"]
        .astype(str)
        .str.strip()
    )

    manifest["site_key"] = (
        manifest["site_key"]
        .astype(str)
        .str.strip()
    )

    index["download_id"] = (
        index["download_id"]
        .astype(str)
        .str.strip()
    )

    index["site_key"] = (
        index["site_key"]
        .astype(str)
        .str.strip()
    )

    index["qa_clear_fraction"] = pd.to_numeric(
        index["qa_clear_fraction"],
        errors="coerce",
    )

    if "acquisition_datetime_utc" in index.columns:
        index["acquisition_datetime_utc"] = (
            pd.to_datetime(
                index["acquisition_datetime_utc"],
                errors="coerce",
                utc=True,
            )
        )

    # 每個 download_id 只保留最新紀錄。
    index = (
        index.sort_values("download_id")
        .drop_duplicates(
            subset=["download_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    active_ids = set(
        manifest["download_id"]
    )

    active = index[
        index["download_id"].isin(
            active_ids
        )
    ].copy()

    missing_index_ids = (
        active_ids
        - set(active["download_id"])
    )

    if missing_index_ids:
        raise RuntimeError(
            "Active manifest rows are missing "
            "from the download index:\n"
            + "\n".join(
                sorted(missing_index_ids)[:20]
            )
        )

    usable = active[
        active["download_status"].eq(
            "success"
        )
        & active[
            "qa_clear_fraction"
        ].ge(QA_THRESHOLD)
    ].copy()

    all_sites = sorted(
        manifest["site_key"]
        .dropna()
        .astype(str)
        .unique()
    )

    before_table = pd.crosstab(
        usable["site_key"],
        usable["development_role"],
    ).reindex(
        all_sites,
        fill_value=0,
    )

    before_table = add_missing_role_columns(
        before_table
    )

    directly_usable = (
        before_table[
            "calibration_negative"
        ].ge(MIN_CALIBRATION_NEGATIVES)
        & before_table[
            "model_negative"
        ].ge(MIN_MODEL_NEGATIVES)
        & before_table[
            "model_positive"
        ].ge(MIN_MODEL_POSITIVES)
    )

    # 可由一張 model negative 改成
    # calibration negative 救回的場址。
    salvageable = (
        before_table[
            "calibration_negative"
        ].eq(4)
        & before_table[
            "model_negative"
        ].ge(5)
        & before_table[
            "model_positive"
        ].ge(1)
    )

    salvage_sites = sorted(
        before_table.index[
            salvageable
        ].astype(str)
    )

    drop_sites = sorted(
        before_table.index[
            ~(directly_usable | salvageable)
        ].astype(str)
    )

    final = usable[
        ~usable["site_key"].isin(
            drop_sites
        )
    ].copy()

    final[
        "clean_dataset_version"
    ] = DATASET_VERSION

    final[
        "qa_acceptance_threshold"
    ] = QA_THRESHOLD

    final[
        "role_reassigned_for_completeness"
    ] = False

    final[
        "original_development_role"
    ] = final["development_role"]

    audit_rows = []

    # 為每個可救場址挑一張乾淨的
    # model negative 改為 calibration negative。
    for site_key in salvage_sites:
        site_rows = final[
            final["site_key"].eq(
                site_key
            )
        ].copy()

        positives = site_rows[
            site_rows[
                "development_role"
            ].eq("model_positive")
        ].copy()

        model_negatives = site_rows[
            site_rows[
                "development_role"
            ].eq("model_negative")
        ].copy()

        positive_sensors = set(
            positives[
                "sensor_code"
            ].dropna().astype(str)
        )

        positive_times = list(
            positives[
                "acquisition_datetime_utc"
            ].dropna()
        )

        model_negatives[
            "same_sensor_as_positive"
        ] = model_negatives[
            "sensor_code"
        ].astype(str).isin(
            positive_sensors
        )

        model_negatives[
            "days_to_nearest_positive"
        ] = model_negatives[
            "acquisition_datetime_utc"
        ].apply(
            lambda timestamp:
                days_to_nearest_positive(
                    timestamp,
                    positive_times,
                )
        )

        model_negatives = (
            model_negatives.sort_values(
                [
                    "same_sensor_as_positive",
                    "days_to_nearest_positive",
                    "qa_clear_fraction",
                    "acquisition_datetime_utc",
                    "download_id",
                ],
                ascending=[
                    False,
                    True,
                    False,
                    True,
                    True,
                ],
                na_position="last",
            )
            .reset_index()
        )

        if model_negatives.empty:
            raise RuntimeError(
                f"{site_key}: no usable model "
                "negative is available for "
                "role reassignment."
            )

        selected = model_negatives.iloc[0]

        selected_index = int(
            selected["index"]
        )

        selected_download_id = str(
            selected["download_id"]
        )

        final.loc[
            selected_index,
            "development_role",
        ] = "calibration_negative"

        if "external_role" in final.columns:
            final.loc[
                selected_index,
                "external_role",
            ] = "calibration_negative"

        final.loc[
            selected_index,
            "role_reassigned_for_completeness",
        ] = True

        audit_rows.append({
            "action":
                "reassign_role",
            "site_key":
                site_key,
            "download_id":
                selected_download_id,
            "old_role":
                "model_negative",
            "new_role":
                "calibration_negative",
            "reason":
                (
                    "site had four usable "
                    "calibration negatives and "
                    "five usable model negatives"
                ),
            "qa_clear_fraction":
                selected[
                    "qa_clear_fraction"
                ],
            "same_sensor_as_positive":
                selected[
                    "same_sensor_as_positive"
                ],
            "days_to_nearest_positive":
                selected[
                    "days_to_nearest_positive"
                ],
            "selection_used_model_output":
                False,
        })

    for site_key in drop_sites:
        counts = before_table.loc[
            site_key
        ]

        if (
            counts["model_positive"]
            < MIN_MODEL_POSITIVES
        ):
            reason = (
                "no usable model-positive scene"
            )
        elif (
            counts["calibration_negative"]
            < MIN_CALIBRATION_NEGATIVES
        ):
            reason = (
                "insufficient usable calibration "
                "negatives and no valid "
                "role reassignment"
            )
        else:
            reason = (
                "insufficient usable development "
                "images"
            )

        audit_rows.append({
            "action":
                "drop_site",
            "site_key":
                site_key,
            "download_id":
                "",
            "old_role":
                "",
            "new_role":
                "",
            "reason":
                reason,
            "usable_calibration_negative":
                int(
                    counts[
                        "calibration_negative"
                    ]
                ),
            "usable_model_negative":
                int(
                    counts[
                        "model_negative"
                    ]
                ),
            "usable_model_positive":
                int(
                    counts[
                        "model_positive"
                    ]
                ),
            "selection_used_model_output":
                False,
        })

    final_table = pd.crosstab(
        final["site_key"],
        final["development_role"],
    )

    final_table = add_missing_role_columns(
        final_table
    )

    final_table["site_usable"] = (
        final_table[
            "calibration_negative"
        ].ge(MIN_CALIBRATION_NEGATIVES)
        & final_table[
            "model_negative"
        ].ge(MIN_MODEL_NEGATIVES)
        & final_table[
            "model_positive"
        ].ge(MIN_MODEL_POSITIVES)
    )

    invalid_sites = final_table[
        ~final_table["site_usable"]
    ]

    if not invalid_sites.empty:
        raise RuntimeError(
            "Invalid sites remain after cleaning:\n"
            + invalid_sites.to_string()
        )

    # 建立與最終 index 一致的 clean manifest。
    final_ids = set(
        final["download_id"]
    )

    final_manifest = manifest[
        manifest["download_id"].isin(
            final_ids
        )
    ].copy()

    role_map = (
        final.set_index(
            "download_id"
        )["development_role"]
    )

    final_manifest[
        "original_development_role"
    ] = final_manifest[
        "development_role"
    ]

    final_manifest[
        "development_role"
    ] = final_manifest[
        "download_id"
    ].map(role_map)

    final_manifest[
        "external_role"
    ] = final_manifest[
        "development_role"
    ]

    reassigned_ids = set(
        final.loc[
            final[
                "role_reassigned_for_completeness"
            ],
            "download_id",
        ]
    )

    final_manifest[
        "role_reassigned_for_completeness"
    ] = final_manifest[
        "download_id"
    ].isin(reassigned_ids)

    final_manifest[
        "clean_dataset_version"
    ] = DATASET_VERSION

    final_manifest[
        "qa_acceptance_threshold"
    ] = QA_THRESHOLD

    final = final.sort_values(
        [
            "development_split",
            "site_key",
            "development_role",
            "acquisition_datetime_utc",
            "download_id",
        ],
        na_position="last",
    ).reset_index(drop=True)

    final_manifest = final_manifest.sort_values(
        [
            "development_split",
            "site_key",
            "development_role",
            "acquisition_datetime_utc",
            "download_id",
        ],
        na_position="last",
    ).reset_index(drop=True)

    summary = (
        final.groupby(
            [
                "development_split",
                "development_role",
                "sensor_code",
            ],
            dropna=False,
        )
        .agg(
            image_count=(
                "download_id",
                "size",
            ),
            site_count=(
                "site_key",
                "nunique",
            ),
            minimum_qa_clear_fraction=(
                "qa_clear_fraction",
                "min",
            ),
            median_qa_clear_fraction=(
                "qa_clear_fraction",
                "median",
            ),
        )
        .reset_index()
    )

    FINAL_INDEX_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        FINAL_INDEX_OUTPUT,
        index=False,
    )

    final_manifest.to_csv(
        FINAL_MANIFEST_OUTPUT,
        index=False,
    )

    pd.DataFrame(
        audit_rows
    ).to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("=" * 108)
    print("FINAL MARS-S2L DEVELOPMENT DATASET")
    print("=" * 108)

    print(
        "\nOriginal active sites:",
        len(all_sites),
    )

    print(
        "Directly usable sites:",
        int(directly_usable.sum()),
    )

    print(
        "Salvaged sites:",
        len(salvage_sites),
    )

    print(
        "Dropped sites:",
        len(drop_sites),
    )

    print(
        "\nFinal usable sites:",
        final["site_key"].nunique(),
    )

    print(
        "Final usable images:",
        len(final),
    )

    print("\nSites by split:")
    print(
        final.groupby(
            "development_split"
        )["site_key"].nunique()
    )

    print("\nImages by split and role:")
    print(
        pd.crosstab(
            final[
                "development_split"
            ],
            final[
                "development_role"
            ],
            margins=True,
        )
    )

    print("\nImages by sensor:")
    print(
        final["sensor_code"]
        .value_counts()
    )

    print("\nMinimum QA:")
    print(
        final["qa_clear_fraction"].min()
    )

    print("\nDropped sites:")
    for site_key in drop_sites:
        print(site_key)

    print("\nSalvaged sites:")
    for site_key in salvage_sites:
        print(site_key)

    print("\nSaved:")
    print(FINAL_INDEX_OUTPUT)
    print(FINAL_MANIFEST_OUTPUT)
    print(AUDIT_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
