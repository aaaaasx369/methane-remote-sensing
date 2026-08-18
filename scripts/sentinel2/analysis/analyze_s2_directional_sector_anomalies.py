from pathlib import Path

import numpy as np
import pandas as pd

import analyze_s2_low_emission_spatial_anomalies as base


BENCHMARK_INPUT = Path(
    "outputs/327_s2_low_emission_pilot_benchmark_v1.csv"
)

SECTOR_OUTPUT = Path(
    "outputs/335_s2_directional_sector_metrics.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/336_s2_directional_sector_summary.csv"
)


INDEX_NAMES = [
    "B12_B11_nd",
    "B12_B11_ratio",
    "B12_B8A_ratio",
]

INNER_RADIUS_METERS = 50
OUTER_RADIUS_METERS = 300
SECTOR_COUNT = 8


def make_anomaly(
    target_id,
    reference_ids,
    scene_indices,
    scene_valid_masks,
    index_name,
):
    reference_stack = np.stack([
        scene_indices[
            reference_id
        ][index_name]
        for reference_id in reference_ids
    ]).astype(float)

    reference_valid = np.stack([
        scene_valid_masks[
            reference_id
        ]
        for reference_id in reference_ids
    ])

    reference_stack[
        ~reference_valid
    ] = np.nan

    reference_median = np.nanmedian(
        reference_stack,
        axis=0,
    )

    target = scene_indices[
        target_id
    ][index_name]

    valid = (
        scene_valid_masks[target_id]
        & np.isfinite(target)
        & np.isfinite(reference_median)
    )

    return (
        target - reference_median,
        valid,
    )


def direction_grids(
    profile,
    latitude,
    longitude,
):
    distance, source_row, source_col = (
        base.source_distance_grid(
            profile,
            latitude,
            longitude,
        )
    )

    rows, columns = np.indices(
        distance.shape
    )

    xs, ys = base.rasterio.transform.xy(
        profile["transform"],
        rows,
        columns,
        offset="center",
    )

    xs = np.asarray(
        xs,
        dtype=float,
    ).reshape(distance.shape)

    ys = np.asarray(
        ys,
        dtype=float,
    ).reshape(distance.shape)

    source_x, source_y = (
        base.transform(
            "EPSG:4326",
            profile["crs"],
            [longitude],
            [latitude],
        )
    )

    dx = xs - source_x[0]
    dy = ys - source_y[0]

    # 北方為 0°，順時針增加。
    angle = (
        np.degrees(
            np.arctan2(
                dx,
                dy,
            )
        )
        + 360
    ) % 360

    return (
        distance,
        angle,
        source_row,
        source_col,
    )


def sector_name(index):
    names = [
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
    ]

    return names[index]


def extract_sector_metrics(
    anomaly,
    valid,
    distance,
    angle,
):
    rows = []

    sector_width = (
        360 / SECTOR_COUNT
    )

    annulus = (
        valid
        & np.isfinite(anomaly)
        & (
            distance
            > INNER_RADIUS_METERS
        )
        & (
            distance
            <= OUTER_RADIUS_METERS
        )
    )

    for sector_index in range(
        SECTOR_COUNT
    ):
        center_angle = (
            sector_index
            * sector_width
        )

        start_angle = (
            center_angle
            - sector_width / 2
        ) % 360

        end_angle = (
            center_angle
            + sector_width / 2
        ) % 360

        if start_angle < end_angle:
            sector_mask = (
                angle >= start_angle
            ) & (
                angle < end_angle
            )
        else:
            sector_mask = (
                angle >= start_angle
            ) | (
                angle < end_angle
            )

        values = anomaly[
            annulus
            & sector_mask
        ]

        values = values[
            np.isfinite(values)
        ]

        rows.append({
            "sector_index":
                sector_index,

            "sector_name":
                sector_name(
                    sector_index
                ),

            "sector_center_degrees":
                center_angle,

            "pixel_count":
                int(values.size),

            "mean_anomaly":
                (
                    float(
                        np.mean(values)
                    )
                    if values.size
                    else np.nan
                ),

            "median_anomaly":
                (
                    float(
                        np.median(values)
                    )
                    if values.size
                    else np.nan
                ),
        })

    return rows


def main():
    benchmark = pd.read_csv(
        BENCHMARK_INPUT,
        low_memory=False,
    )

    benchmark["label"] = pd.to_numeric(
        benchmark["label"],
        errors="coerce",
    )

    benchmark["lat"] = pd.to_numeric(
        benchmark["lat"],
        errors="coerce",
    )

    benchmark["lon"] = pd.to_numeric(
        benchmark["lon"],
        errors="coerce",
    )

    sector_rows = []

    for group_id, group in (
        benchmark.groupby(
            "matched_group_id"
        )
    ):
        positive = group[
            group["label"].eq(1)
        ]

        negatives = group[
            group["label"].eq(0)
        ]

        if (
            len(positive) != 1
            or len(negatives) != 4
        ):
            continue

        positive_row = positive.iloc[0]

        positive_id = str(
            positive_row["sample_id"]
        )

        negative_ids = (
            negatives["sample_id"]
            .astype(str)
            .tolist()
        )

        reference_array, profile = (
            base.read_reference(
                positive_row["patch_path"]
            )
        )

        (
            distance,
            angle,
            _,
            _,
        ) = direction_grids(
            profile,
            float(positive_row["lat"]),
            float(positive_row["lon"]),
        )

        arrays = {
            positive_id:
                reference_array
        }

        roles = {
            positive_id:
                "positive"
        }

        for _, row in (
            negatives.iterrows()
        ):
            sample_id = str(
                row["sample_id"]
            )

            arrays[sample_id] = (
                base.read_to_reference(
                    row["patch_path"],
                    profile,
                )
            )

            roles[sample_id] = (
                "negative_control"
            )

        indices = {}
        valid_masks = {}

        for sample_id, array in (
            arrays.items()
        ):
            (
                indices[sample_id],
                valid_masks[sample_id],
            ) = base.calculate_indices(
                array
            )

        all_ids = [
            positive_id,
            *negative_ids,
        ]

        for sample_id in all_ids:
            if sample_id == positive_id:
                reference_ids = (
                    negative_ids
                )
            else:
                reference_ids = [
                    item
                    for item in negative_ids
                    if item != sample_id
                ]

            for index_name in (
                INDEX_NAMES
            ):
                anomaly, valid = (
                    make_anomaly(
                        sample_id,
                        reference_ids,
                        indices,
                        valid_masks,
                        index_name,
                    )
                )

                rows = (
                    extract_sector_metrics(
                        anomaly,
                        valid,
                        distance,
                        angle,
                    )
                )

                for row in rows:
                    sector_rows.append({
                        "matched_group_id":
                            group_id,

                        "release_rate_kg_h":
                            float(
                                positive_row[
                                    "release_rate_kg_h"
                                ]
                            ),

                        "sample_id":
                            sample_id,

                        "sample_role":
                            roles[
                                sample_id
                            ],

                        "index_name":
                            index_name,

                        **row,
                    })

    sectors = pd.DataFrame(
        sector_rows
    )

    sectors.to_csv(
        SECTOR_OUTPUT,
        index=False,
    )

    summary_rows = []

    for keys, group in (
        sectors.groupby([
            "matched_group_id",
            "release_rate_kg_h",
            "sample_id",
            "sample_role",
            "index_name",
        ])
    ):
        (
            group_id,
            rate,
            sample_id,
            role,
            index_name,
        ) = keys

        for statistic in [
            "mean_anomaly",
            "median_anomaly",
        ]:
            values = pd.to_numeric(
                group[statistic],
                errors="coerce",
            )

            valid = values.notna()

            if not valid.any():
                continue

            valid_group = group.loc[
                valid
            ].copy()

            valid_values = values.loc[
                valid
            ]

            strongest_index = (
                valid_values.abs().idxmax()
            )

            strongest = group.loc[
                strongest_index
            ]

            maximum_absolute = float(
                abs(
                    strongest[
                        statistic
                    ]
                )
            )

            mean_absolute = float(
                valid_values.abs().mean()
            )

            summary_rows.append({
                "matched_group_id":
                    group_id,

                "release_rate_kg_h":
                    rate,

                "sample_id":
                    sample_id,

                "sample_role":
                    role,

                "index_name":
                    index_name,

                "statistic":
                    statistic,

                "strongest_sector":
                    strongest[
                        "sector_name"
                    ],

                "strongest_sector_degrees":
                    strongest[
                        "sector_center_degrees"
                    ],

                "strongest_sector_value":
                    strongest[
                        statistic
                    ],

                "maximum_absolute_sector_value":
                    maximum_absolute,

                "mean_absolute_sector_value":
                    mean_absolute,

                "sector_concentration_ratio":
                    (
                        maximum_absolute
                        / mean_absolute
                        if mean_absolute > 0
                        else np.nan
                    ),
            })

    summary = pd.DataFrame(
        summary_rows
    )

    comparison_rows = []

    for keys, group in (
        summary.groupby([
            "matched_group_id",
            "release_rate_kg_h",
            "index_name",
            "statistic",
        ])
    ):
        (
            group_id,
            rate,
            index_name,
            statistic,
        ) = keys

        positive = group[
            group[
                "sample_role"
            ].eq("positive")
        ]

        controls = group[
            group[
                "sample_role"
            ].eq(
                "negative_control"
            )
        ]

        if (
            len(positive) != 1
            or len(controls) != 4
        ):
            continue

        positive_row = (
            positive.iloc[0]
        )

        control_values = (
            controls[
                "maximum_absolute_sector_value"
            ]
        )

        comparison_rows.append({
            **positive_row.to_dict(),

            "control_maximum_sector_min":
                float(
                    control_values.min()
                ),

            "control_maximum_sector_max":
                float(
                    control_values.max()
                ),

            "positive_absolute_rank_of_5":
                int(
                    (
                        control_values
                        < positive_row[
                            "maximum_absolute_sector_value"
                        ]
                    ).sum()
                    + 1
                ),

            "positive_more_directional_than_all_controls":
                bool(
                    positive_row[
                        "maximum_absolute_sector_value"
                    ]
                    > control_values.max()
                ),
        })

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("=" * 110)
    print(
        "DIRECTIONAL SECTOR ANOMALY SUMMARY"
    )
    print("=" * 110)

    print(
        "\nSector metric rows:",
        len(sectors),
    )

    print(
        "Comparison rows:",
        len(comparison),
    )

    strongest = comparison[
        comparison[
            "positive_more_directional_than_all_controls"
        ].eq(True)
    ]

    print(
        "\nPositive maximum directional anomaly "
        "larger than all controls:"
    )

    if strongest.empty:
        print("None")
    else:
        print(
            strongest[
                [
                    "release_rate_kg_h",
                    "index_name",
                    "statistic",
                    "strongest_sector",
                    "strongest_sector_degrees",
                    "strongest_sector_value",
                    "maximum_absolute_sector_value",
                    "sector_concentration_ratio",
                    "control_maximum_sector_min",
                    "control_maximum_sector_max",
                    "positive_absolute_rank_of_5",
                ]
            ].to_string(
                index=False,
            )
        )

    print("\nSaved:")
    print(SECTOR_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
