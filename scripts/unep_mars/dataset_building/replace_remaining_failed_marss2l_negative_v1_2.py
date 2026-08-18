from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_INPUT = Path(
    "outputs/222_marss2l_external_candidate_images.csv"
)

ACTIVE_MANIFEST_INPUT = Path(
    "outputs/228_marss2l_frozen_external_manifest_v1_1.csv"
)

DOWNLOAD_INDEX_INPUT = Path(
    "outputs/226_marss2l_frozen_external_patch_index.csv"
)

NEW_REPLACEMENT_OUTPUT = Path(
    "outputs/231_marss2l_remaining_replacement_manifest.csv"
)

AMENDED_MANIFEST_OUTPUT = Path(
    "outputs/232_marss2l_frozen_external_manifest_v1_2.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/233_marss2l_v1_2_amendment_audit.csv"
)


TARGET_CALIBRATION = 5
TARGET_TEST = 3

SPLIT_VERSION = (
    "marss2l_external_v1_2_"
    "second_failed_download_replacement"
)


def nearest_positive_days(
    candidate_time,
    positive_times,
):
    if pd.isna(candidate_time) or not positive_times:
        return np.nan

    return min(
        abs(
            (
                candidate_time
                - positive_time
            ).total_seconds()
        )
        for positive_time in positive_times
    ) / 86400.0


def main():
    for path in [
        CANDIDATE_INPUT,
        ACTIVE_MANIFEST_INPUT,
        DOWNLOAD_INDEX_INPUT,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    candidates = pd.read_csv(
        CANDIDATE_INPUT,
        low_memory=False,
    )

    manifest = pd.read_csv(
        ACTIVE_MANIFEST_INPUT,
        low_memory=False,
    )

    index = pd.read_csv(
        DOWNLOAD_INDEX_INPUT,
        low_memory=False,
    )

    for dataframe in [
        candidates,
        manifest,
        index,
    ]:
        if "acquisition_datetime_utc" in dataframe.columns:
            dataframe[
                "acquisition_datetime_utc"
            ] = pd.to_datetime(
                dataframe[
                    "acquisition_datetime_utc"
                ],
                errors="coerce",
                utc=True,
            )

    active_ids = set(
        manifest["download_id"].astype(str)
    )

    active_index = index[
        index["download_id"]
        .astype(str)
        .isin(active_ids)
    ].copy()

    active_index = (
        active_index.sort_values(
            "download_id"
        )
        .drop_duplicates(
            subset=["download_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    success = active_index[
        active_index["download_status"]
        == "success"
    ].copy()

    failed_active = active_index[
        active_index["download_status"]
        != "success"
    ].copy()

    site_keys = sorted(
        manifest["site_key"]
        .dropna()
        .astype(str)
        .unique()
    )

    role_table = pd.crosstab(
        success["site_key"],
        success["external_role"],
    ).reindex(
        site_keys,
        fill_value=0,
    )

    for column in [
        "calibration_negative",
        "test_negative",
        "high_emission_positive",
    ]:
        if column not in role_table.columns:
            role_table[column] = 0

    role_table["calibration_needed"] = (
        TARGET_CALIBRATION
        - role_table[
            "calibration_negative"
        ]
    ).clip(lower=0)

    role_table["test_needed"] = (
        TARGET_TEST
        - role_table[
            "test_negative"
        ]
    ).clip(lower=0)

    incomplete = role_table[
        role_table["calibration_needed"].gt(0)
        | role_table["test_needed"].gt(0)
    ].copy()

    print("=" * 108)
    print("MARS-S2L V1.2 FINAL REPLACEMENT")
    print("=" * 108)

    print("\nIncomplete sites before replacement:")
    print(
        incomplete[
            [
                "calibration_negative",
                "test_negative",
                "high_emission_positive",
                "calibration_needed",
                "test_needed",
            ]
        ].to_string()
    )

    if incomplete.empty:
        print("\nNo replacement is required.")
        return

    # 排除所有曾經被指定或下載過的影像，
    # 包含先前下載失敗的影像。
    assigned_image_keys = set(
        manifest["image_key"]
        .dropna()
        .astype(str)
    )

    if "image_key" in index.columns:
        assigned_image_keys.update(
            index["image_key"]
            .dropna()
            .astype(str)
        )

    replacement_rows = []
    audit_rows = []

    replacement_counter = 1

    for site_key, counts in incomplete.iterrows():
        calibration_needed = int(
            counts["calibration_needed"]
        )

        test_needed = int(
            counts["test_needed"]
        )

        roles_needed = (
            ["calibration_negative"]
            * calibration_needed
            + ["test_negative"]
            * test_needed
        )

        site_manifest = manifest[
            manifest["site_key"]
            .astype(str)
            .eq(str(site_key))
        ].copy()

        positives = site_manifest[
            site_manifest["external_role"]
            == "high_emission_positive"
        ].copy()

        positive_times = list(
            positives[
                "acquisition_datetime_utc"
            ].dropna()
        )

        positive_sensors = set(
            positives[
                "satellite_normalized"
            ]
            .dropna()
            .astype(str)
        )

        pool = candidates[
            candidates["site_key"]
            .astype(str)
            .eq(str(site_key))
            & candidates[
                "benchmark_role"
            ].eq("no_plume_negative")
            & ~candidates[
                "image_key"
            ].astype(str).isin(
                assigned_image_keys
            )
        ].copy()

        pool = pool.drop_duplicates(
            subset=["image_key"],
            keep="first",
        )

        pool[
            "sensor_matches_positive"
        ] = pool[
            "satellite_normalized"
        ].astype(str).isin(
            positive_sensors
        )

        pool[
            "days_to_nearest_positive"
        ] = pool[
            "acquisition_datetime_utc"
        ].apply(
            lambda timestamp:
                nearest_positive_days(
                    timestamp,
                    positive_times,
                )
        )

        pool = pool.sort_values(
            [
                "sensor_matches_positive",
                "days_to_nearest_positive",
                "acquisition_datetime_utc",
                "image_key",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
            na_position="last",
        ).reset_index(drop=True)

        if len(pool) < len(roles_needed):
            raise RuntimeError(
                f"{site_key}: needs "
                f"{len(roles_needed)} replacement(s), "
                f"but only {len(pool)} unused "
                "negative candidates remain."
            )

        chosen = pool.head(
            len(roles_needed)
        ).copy()

        chosen[
            "external_role"
        ] = roles_needed

        chosen[
            "evaluation_label"
        ] = 0

        chosen[
            "ground_truth_type"
        ] = (
            "mars_s2l_human_reviewed_"
            "clear_no_plume"
        )

        chosen[
            "split_version"
        ] = SPLIT_VERSION

        chosen[
            "frozen_high_emission_threshold_kg_h"
        ] = 1000.0

        chosen[
            "frozen_alert_probability_threshold"
        ] = 0.559805

        chosen[
            "selection_used_model_output"
        ] = False

        chosen[
            "amendment_reason"
        ] = (
            "replacement_for_persistent_"
            "download_failure"
        )

        download_ids = []

        for _ in range(len(chosen)):
            download_ids.append(
                f"MARS_REPL2_"
                f"{replacement_counter:04d}"
            )
            replacement_counter += 1

        chosen[
            "download_id"
        ] = download_ids

        replacement_rows.extend(
            chosen.to_dict("records")
        )

        failed_site_rows = failed_active[
            failed_active["site_key"]
            .astype(str)
            .eq(str(site_key))
        ]

        audit_rows.append({
            "site_key":
                site_key,
            "calibration_needed":
                calibration_needed,
            "test_needed":
                test_needed,
            "removed_failed_download_ids":
                "|".join(
                    failed_site_rows[
                        "download_id"
                    ].astype(str)
                ),
            "new_replacement_download_ids":
                "|".join(download_ids),
            "selection_rule":
                (
                    "unused clear no-plume scene; "
                    "same-sensor preference; "
                    "nearest date to positive"
                ),
            "selection_used_model_output":
                False,
        })

    replacements = pd.DataFrame(
        replacement_rows
    )

    failed_active_ids = set(
        failed_active[
            "download_id"
        ].astype(str)
    )

    retained_manifest = manifest[
        ~manifest["download_id"]
        .astype(str)
        .isin(failed_active_ids)
    ].copy()

    amended_manifest = pd.concat(
        [
            retained_manifest,
            replacements,
        ],
        ignore_index=True,
        sort=False,
    )

    if amended_manifest[
        "download_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate download_id in v1.2 manifest."
        )

    if amended_manifest[
        ["site_key", "image_key"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate site/image pair in v1.2 manifest."
        )

    if len(amended_manifest) != 327:
        raise RuntimeError(
            "Expected 327 active rows, "
            f"found {len(amended_manifest)}."
        )

    NEW_REPLACEMENT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    replacements.to_csv(
        NEW_REPLACEMENT_OUTPUT,
        index=False,
    )

    amended_manifest.to_csv(
        AMENDED_MANIFEST_OUTPUT,
        index=False,
    )

    pd.DataFrame(
        audit_rows
    ).to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    print("\nNew replacement rows:")
    print(
        replacements[
            [
                "download_id",
                "site_key",
                "external_role",
                "satellite_normalized",
                "landsat_tile",
                "acquisition_datetime_utc",
                "days_to_nearest_positive",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.2f}",
        )
    )

    print(
        "\nNew replacement count:",
        len(replacements),
    )

    print(
        "V1.2 active manifest rows:",
        len(amended_manifest),
    )

    print("\nSaved:")
    print(NEW_REPLACEMENT_OUTPUT)
    print(AMENDED_MANIFEST_OUTPUT)
    print(AUDIT_OUTPUT)


if __name__ == "__main__":
    main()
