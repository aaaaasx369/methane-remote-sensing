from pathlib import Path

import numpy as np
import pandas as pd


CORE_DATASET = Path(
    "outputs/95_landsat_strict_core_v2_features.csv"
)

POSITIVE_GROUND_TRUTH = Path(
    "outputs/106_landsat_positive_ground_truth_final.csv"
)

TRAINING_OUTPUT = Path(
    "outputs/108_landsat_high_emission_core_manifest.csv"
)

EXCLUDED_OUTPUT = Path(
    "outputs/109_landsat_low_medium_release_exclusions.csv"
)

BACKGROUND_PAIR_OUTPUT = Path(
    "outputs/110_landsat_matched_background_pairs.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/111_landsat_high_emission_manifest_summary.csv"
)


HIGH_EMISSION_THRESHOLD_KG_H = 1000.0
N_BACKGROUNDS_PER_TARGET = 4


IDENTIFIER_COLUMNS = [
    "scene_key",
    "raster_group_id",
    "overpass_id",
    "event_id",
    "ground_truth_key",
    "landsat_product_id_normalized",
    "landsat_product_id",
    "LANDSAT_PRODUCT_ID",
]


def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
    }:
        return ""

    return text


def collect_identifiers(row):
    identifiers = set()

    for column in IDENTIFIER_COLUMNS:
        if column not in row.index:
            continue

        value = clean_text(row[column])

        if value:
            identifiers.add(
                value.lower()
            )

    return identifiers


def first_value(row, columns):
    for column in columns:
        if column not in row.index:
            continue

        value = clean_text(row[column])

        if value:
            return value

    return ""


def parse_time(row):
    value = first_value(
        row,
        [
            "acquisition_time_utc",
            "landsat_image_time_utc",
            "landsat_image_time",
            "candidate_time_utc",
            "satellite_time",
        ],
    )

    return pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )


def numeric_value(row, columns):
    for column in columns:
        if column not in row.index:
            continue

        value = pd.to_numeric(
            pd.Series([row[column]]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(value):
            return float(value)

    return np.nan


def normalize_site(row):
    value = first_value(
        row,
        [
            "site_key_normalized",
            "site_key",
            "site_name",
        ],
    ).lower()

    if "ehrenberg" in value:
        return "ehrenberg"

    if (
        "casa" in value
        or "grande" in value
    ):
        return "casa_grande"

    return value or "unknown"


def normalize_sensor(row):
    value = first_value(
        row,
        [
            "landsat_sensor",
            "sensor",
            "SPACECRAFT_ID",
            "gee_SPACECRAFT_ID",
        ],
    )

    if "8" in value:
        return "Landsat-8"

    if "9" in value:
        return "Landsat-9"

    product_id = first_value(
        row,
        [
            "landsat_product_id_normalized",
            "landsat_product_id",
            "LANDSAT_PRODUCT_ID",
        ],
    )

    if product_id.startswith("LC08"):
        return "Landsat-8"

    if product_id.startswith("LC09"):
        return "Landsat-9"

    return "Unknown"


def match_positive_ground_truth(
    core_row,
    ground_truth,
):
    core_identifiers = collect_identifiers(
        core_row
    )

    matches = []

    for index, gt_row in (
        ground_truth.iterrows()
    ):
        gt_identifiers = collect_identifiers(
            gt_row
        )

        if (
            core_identifiers
            & gt_identifiers
        ):
            matches.append(index)

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one ground-truth "
            "match for positive scene.\n"
            f"Core identifiers: "
            f"{sorted(core_identifiers)}\n"
            f"Matching rows: {matches}"
        )

    return ground_truth.loc[
        matches[0]
    ]


def same_wrs(target, background):
    target_path = numeric_value(
        target,
        [
            "WRS_PATH",
            "gee_WRS_PATH",
        ],
    )

    target_row = numeric_value(
        target,
        [
            "WRS_ROW",
            "gee_WRS_ROW",
        ],
    )

    background_path = numeric_value(
        background,
        [
            "WRS_PATH",
            "gee_WRS_PATH",
        ],
    )

    background_row = numeric_value(
        background,
        [
            "WRS_ROW",
            "gee_WRS_ROW",
        ],
    )

    if any(
        pd.isna(value)
        for value in [
            target_path,
            target_row,
            background_path,
            background_row,
        ]
    ):
        return False

    return bool(
        target_path == background_path
        and target_row == background_row
    )


def build_manifest(
    core,
    ground_truth,
):
    manifest_rows = []

    for _, row in core.iterrows():
        original_label = int(
            pd.to_numeric(
                row["label"],
                errors="raise",
            )
        )

        output_row = row.to_dict()

        output_row[
            "site_key_normalized"
        ] = normalize_site(row)

        output_row[
            "landsat_sensor"
        ] = normalize_sensor(row)

        output_row[
            "acquisition_time_utc"
        ] = parse_time(row)

        output_row[
            "original_release_label"
        ] = original_label

        if original_label == 0:
            output_row.update({
                "release_rate_kg_h": 0.0,
                "release_rate_confidence":
                    "confirmed_no_controlled_release",
                "high_emission_target":
                    0,
                "training_class":
                    "no_release",
                "include_primary_training":
                    True,
                "exclusion_reason":
                    "",
            })

        else:
            gt_row = (
                match_positive_ground_truth(
                    row,
                    ground_truth,
                )
            )

            release_rate = float(
                gt_row[
                    "release_rate_kg_h"
                ]
            )

            high_emission = (
                release_rate
                >= HIGH_EMISSION_THRESHOLD_KG_H
            )

            output_row.update({
                "ground_truth_key":
                    gt_row.get(
                        "ground_truth_key",
                        "",
                    ),
                "release_rate_kg_h":
                    release_rate,
                "release_rate_confidence":
                    gt_row.get(
                        "release_rate_confidence",
                        "",
                    ),
                "exact_overlap":
                    gt_row.get(
                        "exact_overlap",
                        True,
                    ),
                "high_emission_target":
                    (
                        1
                        if high_emission
                        else np.nan
                    ),
                "training_class":
                    (
                        "high_emission"
                        if high_emission
                        else "low_medium_release"
                    ),
                "include_primary_training":
                    bool(high_emission),
                "exclusion_reason":
                    (
                        ""
                        if high_emission
                        else
                        "positive controlled release below "
                        "the primary 1000 kg/h "
                        "high-emission threshold"
                    ),
            })

        manifest_rows.append(
            output_row
        )

    return pd.DataFrame(
        manifest_rows
    )


def build_background_pairs(
    training,
    full_manifest,
):
    negative_pool = full_manifest[
        full_manifest[
            "original_release_label"
        ] == 0
    ].copy()

    pair_rows = []

    for _, target in training.iterrows():
        target_site = target[
            "site_key_normalized"
        ]

        target_sensor = target[
            "landsat_sensor"
        ]

        target_time = pd.to_datetime(
            target[
                "acquisition_time_utc"
            ],
            errors="coerce",
            utc=True,
        )

        target_hash = clean_text(
            target.get(
                "canonical_pixel_hash"
            )
        )

        target_scene = clean_text(
            target.get("scene_key")
        )

        candidates = negative_pool[
            negative_pool[
                "site_key_normalized"
            ] == target_site
        ].copy()

        # Negative target不能拿自己當背景，
        # 否則差值會人工變成零。
        candidates = candidates[
            candidates[
                "canonical_pixel_hash"
            ].astype(str)
            != target_hash
        ].copy()

        ranked_rows = []

        for _, background in (
            candidates.iterrows()
        ):
            background_time = pd.to_datetime(
                background[
                    "acquisition_time_utc"
                ],
                errors="coerce",
                utc=True,
            )

            if (
                pd.notna(target_time)
                and pd.notna(background_time)
            ):
                time_difference_days = abs(
                    (
                        target_time
                        - background_time
                    ).total_seconds()
                ) / 86400
            else:
                time_difference_days = 99999

            same_sensor_flag = (
                background[
                    "landsat_sensor"
                ]
                == target_sensor
            )

            same_wrs_flag = same_wrs(
                target,
                background,
            )

            cloud_cover = numeric_value(
                background,
                [
                    "CLOUD_COVER",
                    "gee_CLOUD_COVER",
                ],
            )

            if pd.isna(cloud_cover):
                cloud_cover = 999.0

            ranked_rows.append({
                "background_row":
                    background,
                "same_sensor":
                    same_sensor_flag,
                "same_wrs":
                    same_wrs_flag,
                "time_difference_days":
                    time_difference_days,
                "cloud_cover":
                    cloud_cover,
            })

        ranked_rows = sorted(
            ranked_rows,
            key=lambda item: (
                not item["same_sensor"],
                not item["same_wrs"],
                item[
                    "time_difference_days"
                ],
                item["cloud_cover"],
            ),
        )

        selected = ranked_rows[
            :N_BACKGROUNDS_PER_TARGET
        ]

        if (
            len(selected)
            < N_BACKGROUNDS_PER_TARGET
        ):
            raise ValueError(
                f"Target {target_scene} at "
                f"{target_site} has only "
                f"{len(selected)} eligible "
                "background scenes."
            )

        for rank, item in enumerate(
            selected,
            start=1,
        ):
            background = item[
                "background_row"
            ]

            pair_rows.append({
                "target_scene_key":
                    target_scene,
                "target_site":
                    target_site,
                "target_sensor":
                    target_sensor,
                "target_time":
                    target_time,
                "target_label":
                    int(
                        target[
                            "high_emission_target"
                        ]
                    ),
                "target_training_class":
                    target[
                        "training_class"
                    ],
                "target_release_rate_kg_h":
                    target[
                        "release_rate_kg_h"
                    ],
                "target_patch_path":
                    target[
                        "resolved_patch_path"
                    ],
                "target_pixel_hash":
                    target[
                        "canonical_pixel_hash"
                    ],
                "background_rank":
                    rank,
                "background_scene_key":
                    background[
                        "scene_key"
                    ],
                "background_site":
                    background[
                        "site_key_normalized"
                    ],
                "background_sensor":
                    background[
                        "landsat_sensor"
                    ],
                "background_time":
                    background[
                        "acquisition_time_utc"
                    ],
                "background_patch_path":
                    background[
                        "resolved_patch_path"
                    ],
                "background_pixel_hash":
                    background[
                        "canonical_pixel_hash"
                    ],
                "same_sensor":
                    item["same_sensor"],
                "same_wrs":
                    item["same_wrs"],
                "time_difference_days":
                    item[
                        "time_difference_days"
                    ],
                "background_cloud_cover":
                    item["cloud_cover"],
            })

    return pd.DataFrame(
        pair_rows
    )


def main():
    for path in [
        CORE_DATASET,
        POSITIVE_GROUND_TRUTH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input file: {path}"
            )

    core = pd.read_csv(
        CORE_DATASET,
        low_memory=False,
    )

    ground_truth = pd.read_csv(
        POSITIVE_GROUND_TRUTH,
        low_memory=False,
    )

    print("=" * 105)
    print("LANDSAT HIGH-EMISSION TRAINING MANIFEST")
    print("=" * 105)

    print(f"\nStrict Core scenes: {len(core)}")
    print(
        f"High-emission threshold: "
        f"{HIGH_EMISSION_THRESHOLD_KG_H} kg/h"
    )

    manifest = build_manifest(
        core,
        ground_truth,
    )

    training = manifest[
        manifest[
            "include_primary_training"
        ] == True
    ].copy()

    exclusions = manifest[
        manifest[
            "include_primary_training"
        ] != True
    ].copy()

    training[
        "high_emission_target"
    ] = training[
        "high_emission_target"
    ].astype(int)

    if len(training) != 16:
        raise ValueError(
            f"Expected 16 primary training "
            f"scenes, found {len(training)}."
        )

    label_counts = (
        training[
            "high_emission_target"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if label_counts != {
        0: 10,
        1: 6,
    }:
        raise ValueError(
            "Unexpected primary label counts: "
            f"{label_counts}"
        )

    pairs = build_background_pairs(
        training,
        manifest,
    )

    pair_counts = (
        pairs.groupby(
            "target_scene_key"
        ).size()
    )

    if not (
        pair_counts
        == N_BACKGROUNDS_PER_TARGET
    ).all():
        raise ValueError(
            "Not every target has exactly "
            f"{N_BACKGROUNDS_PER_TARGET} "
            "background scenes."
        )

    if (
        pairs["target_pixel_hash"]
        == pairs["background_pixel_hash"]
    ).any():
        raise ValueError(
            "A target scene was paired with "
            "itself as background."
        )

    TRAINING_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training.to_csv(
        TRAINING_OUTPUT,
        index=False,
    )

    exclusions.to_csv(
        EXCLUDED_OUTPUT,
        index=False,
    )

    pairs.to_csv(
        BACKGROUND_PAIR_OUTPUT,
        index=False,
    )

    summary_rows = []

    for (
        site,
        target_label,
        training_class,
    ), group in training.groupby(
        [
            "site_key_normalized",
            "high_emission_target",
            "training_class",
        ],
        dropna=False,
    ):
        summary_rows.append({
            "site_key": site,
            "high_emission_target":
                target_label,
            "training_class":
                training_class,
            "scene_count":
                len(group),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("\nPrimary training scenes:")
    print(
        training[
            [
                "scene_key",
                "site_key_normalized",
                "landsat_sensor",
                "acquisition_time_utc",
                "original_release_label",
                "release_rate_kg_h",
                "training_class",
                "high_emission_target",
            ]
        ].sort_values(
            [
                "site_key_normalized",
                "high_emission_target",
                "acquisition_time_utc",
            ]
        ).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}",
        )
    )

    print("\nLabel by site:")
    print(
        pd.crosstab(
            training[
                "site_key_normalized"
            ],
            training[
                "high_emission_target"
            ],
            margins=True,
        )
    )

    print("\nExcluded low/medium releases:")
    print(
        exclusions[
            [
                "scene_key",
                "site_key_normalized",
                "release_rate_kg_h",
                "training_class",
                "exclusion_reason",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}",
        )
    )

    print("\nMatched-background summary:")
    print(
        pairs.groupby(
            [
                "target_site",
                "target_label",
            ]
        ).agg(
            target_count=(
                "target_scene_key",
                "nunique",
            ),
            pair_count=(
                "background_scene_key",
                "size",
            ),
            same_sensor_pairs=(
                "same_sensor",
                "sum",
            ),
            same_wrs_pairs=(
                "same_wrs",
                "sum",
            ),
            median_time_difference_days=(
                "time_difference_days",
                "median",
            ),
        )
    )

    print(
        "\nBackground scenes per target:"
    )

    print(
        pair_counts.value_counts()
        .sort_index()
    )

    print("\nSaved:")
    print(TRAINING_OUTPUT)
    print(EXCLUDED_OUTPUT)
    print(BACKGROUND_PAIR_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
