from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_INPUT = Path(
    "outputs/222_marss2l_external_candidate_images.csv"
)

ORIGINAL_MANIFEST_INPUT = Path(
    "outputs/224_marss2l_frozen_external_download_manifest.csv"
)

DOWNLOAD_INDEX_INPUT = Path(
    "outputs/226_marss2l_frozen_external_patch_index.csv"
)

REPLACEMENT_OUTPUT = Path(
    "outputs/227_marss2l_replacement_negative_manifest.csv"
)

AMENDED_MANIFEST_OUTPUT = Path(
    "outputs/228_marss2l_frozen_external_manifest_v1_1.csv"
)

AMENDMENT_AUDIT_OUTPUT = Path(
    "outputs/229_marss2l_manifest_amendment_audit.csv"
)


TARGET_CALIBRATION = 5
TARGET_TEST = 3

SPLIT_VERSION = (
    "marss2l_external_v1_1_"
    "failed_download_replacements"
)


def nearest_positive_days(
    candidate_time,
    positive_times,
):
    if (
        pd.isna(candidate_time)
        or not positive_times
    ):
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
        ORIGINAL_MANIFEST_INPUT,
        DOWNLOAD_INDEX_INPUT,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    candidates = pd.read_csv(
        CANDIDATE_INPUT,
        low_memory=False,
    )

    original = pd.read_csv(
        ORIGINAL_MANIFEST_INPUT,
        low_memory=False,
    )

    index = pd.read_csv(
        DOWNLOAD_INDEX_INPUT,
        low_memory=False,
    )

    for dataframe in [
        candidates,
        original,
        index,
    ]:
        if (
            "acquisition_datetime_utc"
            in dataframe.columns
        ):
            dataframe[
                "acquisition_datetime_utc"
            ] = pd.to_datetime(
                dataframe[
                    "acquisition_datetime_utc"
                ],
                errors="coerce",
                utc=True,
            )

    success = index[
        index["download_status"] == "success"
    ].copy()

    failed = index[
        index["download_status"] == "failed"
    ].copy()

    role_table = pd.crosstab(
        success["site_key"],
        success["external_role"],
    )

    for column in [
        "calibration_negative",
        "test_negative",
        "high_emission_positive",
    ]:
        if column not in role_table.columns:
            role_table[column] = 0

    incomplete_sites = role_table[
        role_table["calibration_negative"].lt(
            TARGET_CALIBRATION
        )
        | role_table["test_negative"].lt(
            TARGET_TEST
        )
    ].copy()

    print("=" * 105)
    print("MARS-S2L FAILED-DOWNLOAD REPLACEMENT")
    print("=" * 105)

    print("\nIncomplete sites before replacement:")
    print(
        incomplete_sites[
            [
                "calibration_negative",
                "test_negative",
                "high_emission_positive",
            ]
        ].to_string()
    )

    # 不允許重新選取原 manifest 中任何已指派影像，
    # 包括下載失敗的三張。
    already_assigned_image_keys = set(
        original["image_key"]
        .dropna()
        .astype(str)
    )

    replacement_rows = []
    audit_rows = []
    replacement_number = 1

    for site_key, counts in (
        incomplete_sites.iterrows()
    ):
        calibration_needed = max(
            0,
            TARGET_CALIBRATION
            - int(
                counts[
                    "calibration_negative"
                ]
            ),
        )

        test_needed = max(
            0,
            TARGET_TEST
            - int(
                counts["test_negative"]
            ),
        )

        roles_needed = (
            ["calibration_negative"]
            * calibration_needed
            + ["test_negative"]
            * test_needed
        )

        positive_rows = original[
            (original["site_key"] == site_key)
            & (
                original["external_role"]
                == "high_emission_positive"
            )
        ].copy()

        positive_times = list(
            positive_rows[
                "acquisition_datetime_utc"
            ].dropna()
        )

        positive_sensors = set(
            positive_rows[
                "satellite_normalized"
            ].dropna().astype(str)
        )

        pool = candidates[
            (candidates["site_key"] == site_key)
            & (
                candidates["benchmark_role"]
                == "no_plume_negative"
            )
            & ~candidates[
                "image_key"
            ].astype(str).isin(
                already_assigned_image_keys
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
            lambda value:
                nearest_positive_days(
                    value,
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
                f"{site_key}: requires "
                f"{len(roles_needed)} replacements, "
                f"but only {len(pool)} are available."
            )

        chosen = pool.head(
            len(roles_needed)
        ).copy()

        chosen["external_role"] = roles_needed
        chosen["evaluation_label"] = 0

        chosen["ground_truth_type"] = (
            "mars_s2l_human_reviewed_"
            "clear_no_plume"
        )

        chosen["split_version"] = SPLIT_VERSION

        chosen[
            "frozen_high_emission_threshold_kg_h"
        ] = 1000.0

        chosen[
            "frozen_alert_probability_threshold"
        ] = 0.559805

        chosen[
            "selection_used_model_output"
        ] = False

        chosen["amendment_reason"] = (
            "replacement_for_failed_download"
        )

        replacement_ids = []

        for _ in range(len(chosen)):
            replacement_ids.append(
                f"MARS_REPL_{replacement_number:04d}"
            )
            replacement_number += 1

        chosen["download_id"] = (
            replacement_ids
        )

        replacement_rows.extend(
            chosen.to_dict("records")
        )

        original_failed = failed[
            failed["site_key"] == site_key
        ]

        audit_rows.append({
            "site_key":
                site_key,
            "calibration_needed":
                calibration_needed,
            "test_needed":
                test_needed,
            "original_failed_download_ids":
                "|".join(
                    original_failed[
                        "download_id"
                    ].astype(str)
                ),
            "replacement_download_ids":
                "|".join(replacement_ids),
            "selection_rule":
                (
                    "unused clear no-plume image; "
                    "same-sensor preference; "
                    "minimum temporal distance "
                    "to positive observations"
                ),
            "selection_used_model_output":
                False,
        })

    replacements = pd.DataFrame(
        replacement_rows
    )

    if len(replacements) != 3:
        raise RuntimeError(
            "Expected exactly 3 replacements, "
            f"created {len(replacements)}."
        )

    # 從 active manifest 移除下載失敗的原始項目，
    # 再加入三張替補影像。
    failed_download_ids = set(
        failed["download_id"]
        .dropna()
        .astype(str)
    )

    active_original = original[
        ~original["download_id"]
        .astype(str)
        .isin(failed_download_ids)
    ].copy()

    amended = pd.concat(
        [
            active_original,
            replacements,
        ],
        ignore_index=True,
        sort=False,
    )

    if amended["download_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate download_id found."
        )

    if amended[
        ["site_key", "image_key"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate site/image assignment found."
        )

    replacements.to_csv(
        REPLACEMENT_OUTPUT,
        index=False,
    )

    amended.to_csv(
        AMENDED_MANIFEST_OUTPUT,
        index=False,
    )

    pd.DataFrame(
        audit_rows
    ).to_csv(
        AMENDMENT_AUDIT_OUTPUT,
        index=False,
    )

    print("\nReplacement roles:")
    print(
        replacements[
            "external_role"
        ].value_counts()
    )

    print("\nReplacements:")
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
        "\nAmended active manifest rows:",
        len(amended),
    )

    print("\nSaved:")
    print(REPLACEMENT_OUTPUT)
    print(AMENDED_MANIFEST_OUTPUT)
    print(AMENDMENT_AUDIT_OUTPUT)


if __name__ == "__main__":
    main()
