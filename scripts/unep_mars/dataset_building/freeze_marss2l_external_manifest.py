from pathlib import Path

import numpy as np
import pandas as pd


ELIGIBLE_SITE_INPUT = Path(
    "outputs/221_marss2l_external_eligible_sites.csv"
)

CANDIDATE_IMAGE_INPUT = Path(
    "outputs/222_marss2l_external_candidate_images.csv"
)

SITE_SPLIT_OUTPUT = Path(
    "outputs/223_marss2l_frozen_external_site_split.csv"
)

DOWNLOAD_MANIFEST_OUTPUT = Path(
    "outputs/224_marss2l_frozen_external_download_manifest.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/225_marss2l_frozen_external_split_summary.csv"
)


CALIBRATION_NEGATIVES_PER_SITE = 5
TEST_NEGATIVES_PER_SITE = 3
TOTAL_NEGATIVES_PER_SITE = (
    CALIBRATION_NEGATIVES_PER_SITE
    + TEST_NEGATIVES_PER_SITE
)

HIGH_EMISSION_THRESHOLD_KG_H = 1000.0
ALERT_PROBABILITY_THRESHOLD = 0.559805

SPLIT_VERSION = (
    "marss2l_external_v1_"
    "frozen_before_model_application"
)


def nearest_positive_days(
    negative_time,
    positive_times,
):
    if (
        pd.isna(negative_time)
        or len(positive_times) == 0
    ):
        return np.nan

    return float(
        min(
            abs(
                (
                    negative_time
                    - positive_time
                ).total_seconds()
            )
            for positive_time in positive_times
        )
        / 86400.0
    )


def main():
    if not ELIGIBLE_SITE_INPUT.exists():
        raise FileNotFoundError(
            ELIGIBLE_SITE_INPUT
        )

    if not CANDIDATE_IMAGE_INPUT.exists():
        raise FileNotFoundError(
            CANDIDATE_IMAGE_INPUT
        )

    eligible_sites = pd.read_csv(
        ELIGIBLE_SITE_INPUT,
        low_memory=False,
    )

    images = pd.read_csv(
        CANDIDATE_IMAGE_INPUT,
        low_memory=False,
    )

    required_columns = [
        "site_key",
        "image_key",
        "benchmark_role",
        "satellite_normalized",
        "landsat_tile",
        "acquisition_datetime_utc",
        "ch4_fluxrate",
    ]

    missing = [
        column
        for column in required_columns
        if column not in images.columns
    ]

    if missing:
        raise KeyError(
            f"Missing columns: {missing}"
        )

    eligible_keys = set(
        eligible_sites[
            "site_key"
        ].astype(str)
    )

    images["site_key"] = (
        images["site_key"]
        .astype(str)
        .str.strip()
    )

    images = images[
        images["site_key"].isin(
            eligible_keys
        )
    ].copy()

    images[
        "acquisition_datetime_utc"
    ] = pd.to_datetime(
        images[
            "acquisition_datetime_utc"
        ],
        errors="coerce",
        utc=True,
    )

    images["ch4_fluxrate"] = (
        pd.to_numeric(
            images["ch4_fluxrate"],
            errors="coerce",
        )
    )

    images["satellite_normalized"] = (
        images["satellite_normalized"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    images["image_key"] = (
        images["image_key"]
        .astype(str)
        .str.strip()
    )

    images = (
        images.sort_values(
            [
                "site_key",
                "benchmark_role",
                "ch4_fluxrate",
                "acquisition_datetime_utc",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
            na_position="last",
        )
        .drop_duplicates(
            subset=[
                "site_key",
                "image_key",
                "benchmark_role",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    frozen_rows = []
    site_rows = []

    for site_key in sorted(
        eligible_keys
    ):
        site_images = images[
            images["site_key"]
            == site_key
        ].copy()

        positives = site_images[
            site_images["benchmark_role"]
            == "high_emission_positive"
        ].copy()

        negatives = site_images[
            site_images["benchmark_role"]
            == "no_plume_negative"
        ].copy()

        positives = positives[
            positives["ch4_fluxrate"]
            >= HIGH_EMISSION_THRESHOLD_KG_H
        ].copy()

        positives = (
            positives.sort_values(
                [
                    "acquisition_datetime_utc",
                    "ch4_fluxrate",
                ],
                ascending=[
                    True,
                    False,
                ],
                na_position="last",
            )
            .drop_duplicates(
                subset=["image_key"],
                keep="first",
            )
            .reset_index(drop=True)
        )

        negatives = (
            negatives.sort_values(
                "acquisition_datetime_utc",
                na_position="last",
            )
            .drop_duplicates(
                subset=["image_key"],
                keep="first",
            )
            .reset_index(drop=True)
        )

        if positives.empty:
            raise RuntimeError(
                f"{site_key}: no high-emission "
                "positive image."
            )

        if (
            len(negatives)
            < TOTAL_NEGATIVES_PER_SITE
        ):
            raise RuntimeError(
                f"{site_key}: only "
                f"{len(negatives)} negatives."
            )

        positive_times = [
            timestamp
            for timestamp in positives[
                "acquisition_datetime_utc"
            ].dropna()
        ]

        positive_sensors = set(
            positives[
                "satellite_normalized"
            ].dropna().astype(str)
        )

        negatives[
            "sensor_matches_positive_site"
        ] = negatives[
            "satellite_normalized"
        ].isin(
            positive_sensors
        )

        negatives[
            "days_to_nearest_positive"
        ] = negatives[
            "acquisition_datetime_utc"
        ].apply(
            lambda timestamp:
                nearest_positive_days(
                    timestamp,
                    positive_times,
                )
        )

        # 先選感測器相同且日期最接近正樣本的負樣本。
        selected_negatives = (
            negatives.sort_values(
                [
                    "sensor_matches_positive_site",
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
            )
            .head(
                TOTAL_NEGATIVES_PER_SITE
            )
            .copy()
        )

        # 再按照時間排序並交錯分配，
        # 避免 calibration 全在較近日期、
        # test 全在較遠日期。
        selected_negatives = (
            selected_negatives.sort_values(
                [
                    "acquisition_datetime_utc",
                    "image_key",
                ],
                na_position="last",
            )
            .reset_index(drop=True)
        )

        role_pattern = [
            "calibration_negative",
            "test_negative",
            "calibration_negative",
            "test_negative",
            "calibration_negative",
            "test_negative",
            "calibration_negative",
            "calibration_negative",
        ]

        selected_negatives[
            "external_role"
        ] = role_pattern[
            :len(selected_negatives)
        ]

        selected_negatives[
            "evaluation_label"
        ] = 0

        selected_negatives[
            "ground_truth_type"
        ] = (
            "mars_s2l_human_reviewed_"
            "clear_no_plume"
        )

        positives[
            "external_role"
        ] = "high_emission_positive"

        positives[
            "evaluation_label"
        ] = 1

        positives[
            "ground_truth_type"
        ] = (
            "mars_s2l_human_confirmed_"
            "landsat_plume_"
            "satellite_estimated_flux"
        )

        site_frozen = pd.concat(
            [
                positives,
                selected_negatives,
            ],
            ignore_index=True,
            sort=False,
        )

        site_frozen[
            "split_version"
        ] = SPLIT_VERSION

        site_frozen[
            "frozen_high_emission_threshold_kg_h"
        ] = HIGH_EMISSION_THRESHOLD_KG_H

        site_frozen[
            "frozen_alert_probability_threshold"
        ] = ALERT_PROBABILITY_THRESHOLD

        site_frozen[
            "selection_used_model_output"
        ] = False

        frozen_rows.extend(
            site_frozen.to_dict(
                "records"
            )
        )

        calibration_count = int(
            (
                selected_negatives[
                    "external_role"
                ]
                == "calibration_negative"
            ).sum()
        )

        test_count = int(
            (
                selected_negatives[
                    "external_role"
                ]
                == "test_negative"
            ).sum()
        )

        site_rows.append({
            "site_key":
                site_key,
            "split_version":
                SPLIT_VERSION,
            "positive_count":
                len(positives),
            "calibration_negative_count":
                calibration_count,
            "test_negative_count":
                test_count,
            "total_download_count":
                (
                    len(positives)
                    + calibration_count
                    + test_count
                ),
            "positive_landsat8_count":
                int(
                    (
                        positives[
                            "satellite_normalized"
                        ]
                        == "LC08"
                    ).sum()
                ),
            "positive_landsat9_count":
                int(
                    (
                        positives[
                            "satellite_normalized"
                        ]
                        == "LC09"
                    ).sum()
                ),
            "minimum_positive_flux_kg_h":
                positives[
                    "ch4_fluxrate"
                ].min(),
            "median_positive_flux_kg_h":
                positives[
                    "ch4_fluxrate"
                ].median(),
            "maximum_positive_flux_kg_h":
                positives[
                    "ch4_fluxrate"
                ].max(),
            "selection_rule":
                (
                    "same-sensor preference, "
                    "nearest positive date, "
                    "temporally interleaved "
                    "calibration/test assignment"
                ),
        })

    frozen = pd.DataFrame(
        frozen_rows
    )

    site_split = pd.DataFrame(
        site_rows
    )

    duplicate_assignments = (
        frozen.duplicated(
            subset=[
                "site_key",
                "image_key",
            ],
            keep=False,
        )
    )

    if duplicate_assignments.any():
        duplicates = frozen.loc[
            duplicate_assignments,
            [
                "site_key",
                "image_key",
                "external_role",
            ],
        ]

        raise RuntimeError(
            "Duplicate site/image assignments:\n"
            + duplicates.to_string(
                index=False
            )
        )

    role_counts = (
        frozen.groupby(
            [
                "site_key",
                "external_role",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    for site_key in sorted(
        eligible_keys
    ):
        calibration_count = int(
            role_counts.loc[
                site_key
            ].get(
                "calibration_negative",
                0,
            )
        )

        test_count = int(
            role_counts.loc[
                site_key
            ].get(
                "test_negative",
                0,
            )
        )

        positive_count = int(
            role_counts.loc[
                site_key
            ].get(
                "high_emission_positive",
                0,
            )
        )

        if (
            calibration_count
            != CALIBRATION_NEGATIVES_PER_SITE
        ):
            raise RuntimeError(
                f"{site_key}: expected "
                f"{CALIBRATION_NEGATIVES_PER_SITE} "
                "calibration negatives, found "
                f"{calibration_count}."
            )

        if (
            test_count
            != TEST_NEGATIVES_PER_SITE
        ):
            raise RuntimeError(
                f"{site_key}: expected "
                f"{TEST_NEGATIVES_PER_SITE} "
                "test negatives, found "
                f"{test_count}."
            )

        if positive_count < 1:
            raise RuntimeError(
                f"{site_key}: no positive."
            )

    frozen[
        "download_id"
    ] = [
        f"MARS_EXT_{number:04d}"
        for number in range(
            1,
            len(frozen) + 1,
        )
    ]

    frozen = frozen.sort_values(
        [
            "site_key",
            "external_role",
            "acquisition_datetime_utc",
        ],
        na_position="last",
    ).reset_index(drop=True)

    site_split = site_split.sort_values(
        [
            "positive_count",
            "site_key",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    summary = (
        frozen.groupby(
            [
                "external_role",
                "satellite_normalized",
            ],
            dropna=False,
        )
        .agg(
            image_count=(
                "image_key",
                "size",
            ),
            unique_site_count=(
                "site_key",
                "nunique",
            ),
            first_acquisition=(
                "acquisition_datetime_utc",
                "min",
            ),
            last_acquisition=(
                "acquisition_datetime_utc",
                "max",
            ),
            minimum_flux_kg_h=(
                "ch4_fluxrate",
                "min",
            ),
            median_flux_kg_h=(
                "ch4_fluxrate",
                "median",
            ),
            maximum_flux_kg_h=(
                "ch4_fluxrate",
                "max",
            ),
        )
        .reset_index()
    )

    SITE_SPLIT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    site_split.to_csv(
        SITE_SPLIT_OUTPUT,
        index=False,
    )

    frozen.to_csv(
        DOWNLOAD_MANIFEST_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("=" * 108)
    print("FROZEN MARS-S2L EXTERNAL MANIFEST")
    print("=" * 108)

    print(
        "\nFrozen unseen sites:",
        frozen["site_key"].nunique(),
    )

    print(
        "Total images:",
        len(frozen),
    )

    print("\nRole counts:")
    print(
        frozen[
            "external_role"
        ].value_counts()
    )

    print("\nRole by sensor:")
    print(
        pd.crosstab(
            frozen[
                "external_role"
            ],
            frozen[
                "satellite_normalized"
            ],
            margins=True,
        )
    )

    print("\nPositive flux summary:")
    print(
        frozen.loc[
            frozen[
                "external_role"
            ]
            == "high_emission_positive",
            "ch4_fluxrate",
        ].describe()
    )

    print(
        "\nAll sites have exactly "
        f"{CALIBRATION_NEGATIVES_PER_SITE} "
        "calibration negatives and "
        f"{TEST_NEGATIVES_PER_SITE} "
        "test negatives."
    )

    print("\nSaved:")
    print(SITE_SPLIT_OUTPUT)
    print(DOWNLOAD_MANIFEST_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
