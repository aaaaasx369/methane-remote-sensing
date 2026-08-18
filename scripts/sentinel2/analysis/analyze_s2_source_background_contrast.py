from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_s2_low_emission_spatial_anomalies as base


BENCHMARK_INPUT = Path(
    "outputs/327_s2_low_emission_pilot_benchmark_v1.csv"
)

METRIC_OUTPUT = Path(
    "outputs/332_s2_source_background_contrast_metrics.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/333_s2_source_background_contrast_summary.csv"
)

PROFILE_OUTPUT = Path(
    "outputs/334_s2_radial_anomaly_profiles.csv"
)

PLOT_DIR = Path(
    "outputs/s2_low_emission_radial_profiles"
)


INDEX_NAMES = [
    "B12_B11_nd",
    "B12_B11_ratio",
    "B12_B8A_ratio",
]

RADIAL_BINS = [
    (0, 100),
    (100, 200),
    (200, 300),
    (300, 400),
    (400, 500),
]

SOURCE_MAX_METERS = 100
BACKGROUND_MIN_METERS = 300
BACKGROUND_MAX_METERS = 500


def finite_values(
    array,
    valid,
    distance,
    minimum_distance,
    maximum_distance,
):
    mask = (
        valid
        & np.isfinite(array)
        & (distance > minimum_distance)
        & (distance <= maximum_distance)
    )

    return array[mask]


def safe_statistic(
    values,
    statistic,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return np.nan

    if statistic == "mean":
        return float(
            np.mean(values)
        )

    if statistic == "median":
        return float(
            np.median(values)
        )

    raise ValueError(
        f"Unknown statistic: {statistic}"
    )


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
        for reference_id
        in reference_ids
    ]).astype(float)

    reference_valid = np.stack([
        scene_valid_masks[
            reference_id
        ]
        for reference_id
        in reference_ids
    ])

    reference_stack[
        ~reference_valid
    ] = np.nan

    reference_median = np.nanmedian(
        reference_stack,
        axis=0,
    )

    target_values = scene_indices[
        target_id
    ][index_name]

    valid = (
        scene_valid_masks[
            target_id
        ]
        & np.isfinite(
            reference_median
        )
        & np.isfinite(
            target_values
        )
    )

    anomaly = (
        target_values
        - reference_median
    )

    return anomaly, valid


def calculate_contrast(
    anomaly,
    valid,
    distance,
    statistic,
):
    source_values = finite_values(
        array=anomaly,
        valid=valid,
        distance=distance,
        minimum_distance=-1,
        maximum_distance=
            SOURCE_MAX_METERS,
    )

    background_values = finite_values(
        array=anomaly,
        valid=valid,
        distance=distance,
        minimum_distance=
            BACKGROUND_MIN_METERS,
        maximum_distance=
            BACKGROUND_MAX_METERS,
    )

    source_stat = safe_statistic(
        source_values,
        statistic,
    )

    background_stat = safe_statistic(
        background_values,
        statistic,
    )

    contrast = (
        source_stat
        - background_stat
    )

    return {
        "statistic":
            statistic,

        "source_value":
            source_stat,

        "background_value":
            background_stat,

        "source_background_contrast":
            contrast,

        "absolute_contrast":
            abs(contrast)
            if np.isfinite(contrast)
            else np.nan,

        "source_pixel_count":
            int(
                np.isfinite(
                    source_values
                ).sum()
            ),

        "background_pixel_count":
            int(
                np.isfinite(
                    background_values
                ).sum()
            ),
    }


def radial_profile_rows(
    anomaly,
    valid,
    distance,
    metadata,
):
    rows = []

    for inner_radius, outer_radius in (
        RADIAL_BINS
    ):
        values = finite_values(
            array=anomaly,
            valid=valid,
            distance=distance,
            minimum_distance=
                inner_radius,
            maximum_distance=
                outer_radius,
        )

        rows.append({
            **metadata,

            "inner_radius_m":
                inner_radius,

            "outer_radius_m":
                outer_radius,

            "radial_midpoint_m":
                (
                    inner_radius
                    + outer_radius
                ) / 2,

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

            "mean_absolute_anomaly":
                (
                    float(
                        np.mean(
                            np.abs(values)
                        )
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

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metric_rows = []
    profile_rows = []

    print("=" * 110)
    print(
        "SENTINEL-2 SOURCE–BACKGROUND "
        "CONTRAST ANALYSIS"
    )
    print("=" * 110)

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
            print(
                "Skipping invalid group:",
                group_id,
            )

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
                positive_row[
                    "patch_path"
                ]
            )
        )

        (
            distance,
            source_row,
            source_col,
        ) = base.source_distance_grid(
            profile,
            float(
                positive_row["lat"]
            ),
            float(
                positive_row["lon"]
            ),
        )

        scene_arrays = {
            positive_id:
                reference_array
        }

        scene_roles = {
            positive_id:
                "positive"
        }

        for _, row in (
            negatives.iterrows()
        ):
            sample_id = str(
                row["sample_id"]
            )

            scene_arrays[
                sample_id
            ] = base.read_to_reference(
                row["patch_path"],
                profile,
            )

            scene_roles[
                sample_id
            ] = "negative_control"

        scene_indices = {}
        scene_valid_masks = {}

        for sample_id, array in (
            scene_arrays.items()
        ):
            indices, valid = (
                base.calculate_indices(
                    array
                )
            )

            scene_indices[
                sample_id
            ] = indices

            scene_valid_masks[
                sample_id
            ] = valid

        sample_ids = [
            positive_id,
            *negative_ids,
        ]

        for sample_id in sample_ids:
            if sample_id == positive_id:
                reference_ids = (
                    negative_ids
                )
            else:
                reference_ids = [
                    negative_id
                    for negative_id
                    in negative_ids
                    if negative_id
                    != sample_id
                ]

            for index_name in (
                INDEX_NAMES
            ):
                anomaly, valid = (
                    make_anomaly(
                        target_id=
                            sample_id,
                        reference_ids=
                            reference_ids,
                        scene_indices=
                            scene_indices,
                        scene_valid_masks=
                            scene_valid_masks,
                        index_name=
                            index_name,
                    )
                )

                metadata = {
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
                        scene_roles[
                            sample_id
                        ],

                    "index_name":
                        index_name,
                }

                for statistic in [
                    "mean",
                    "median",
                ]:
                    contrast = (
                        calculate_contrast(
                            anomaly=
                                anomaly,
                            valid=
                                valid,
                            distance=
                                distance,
                            statistic=
                                statistic,
                        )
                    )

                    metric_rows.append({
                        **metadata,
                        **contrast,
                    })

                profile_rows.extend(
                    radial_profile_rows(
                        anomaly=
                            anomaly,
                        valid=
                            valid,
                        distance=
                            distance,
                        metadata=
                            metadata,
                    )
                )

    metrics = pd.DataFrame(
        metric_rows
    )

    profiles = pd.DataFrame(
        profile_rows
    )

    metrics.to_csv(
        METRIC_OUTPUT,
        index=False,
    )

    profiles.to_csv(
        PROFILE_OUTPUT,
        index=False,
    )

    summary_rows = []

    group_columns = [
        "matched_group_id",
        "release_rate_kg_h",
        "index_name",
        "statistic",
    ]

    for keys, group in (
        metrics.groupby(
            group_columns
        )
    ):
        (
            group_id,
            release_rate,
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

        positive_contrast = float(
            positive_row[
                "source_background_contrast"
            ]
        )

        control_contrasts = (
            pd.to_numeric(
                controls[
                    "source_background_contrast"
                ],
                errors="coerce",
            )
            .dropna()
        )

        if control_contrasts.empty:
            continue

        more_extreme = bool(
            abs(positive_contrast)
            > np.abs(
                control_contrasts
            ).max()
        )

        rank = int(
            (
                np.abs(
                    control_contrasts
                )
                < abs(
                    positive_contrast
                )
            ).sum()
            + 1
        )

        summary_rows.append({
            "matched_group_id":
                group_id,

            "release_rate_kg_h":
                release_rate,

            "index_name":
                index_name,

            "statistic":
                statistic,

            "positive_source_value":
                positive_row[
                    "source_value"
                ],

            "positive_background_value":
                positive_row[
                    "background_value"
                ],

            "positive_source_background_contrast":
                positive_contrast,

            "control_contrast_min":
                float(
                    control_contrasts.min()
                ),

            "control_contrast_max":
                float(
                    control_contrasts.max()
                ),

            "positive_absolute_rank_of_5":
                rank,

            "positive_more_extreme_than_all_controls":
                more_extreme,
        })

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    for (
        group_id,
        index_name,
    ), plot_data in (
        profiles.groupby([
            "matched_group_id",
            "index_name",
        ])
    ):
        plt.figure(
            figsize=(8, 5)
        )

        for sample_id, sample in (
            plot_data.groupby(
                "sample_id"
            )
        ):
            sample = sample.sort_values(
                "radial_midpoint_m"
            )

            label = (
                f"{sample_id} "
                f"({sample['sample_role'].iloc[0]})"
            )

            plt.plot(
                sample[
                    "radial_midpoint_m"
                ],
                sample[
                    "median_anomaly"
                ],
                marker="o",
                label=label,
            )

        plt.axhline(
            0,
            linewidth=1,
        )

        plt.xlabel(
            "Distance from release source (m)"
        )

        plt.ylabel(
            "Median anomaly"
        )

        plt.title(
            f"{index_name}: radial anomaly profile"
        )

        plt.legend(
            fontsize=7
        )

        plt.tight_layout()

        safe_group = (
            str(group_id)
            .replace("/", "_")
            .replace(":", "_")
        )

        output_path = (
            PLOT_DIR
            / (
                f"{safe_group}_"
                f"{index_name}_radial_profile.png"
            )
        )

        plt.savefig(
            output_path,
            dpi=180,
        )

        plt.close()

    print(
        "\nContrast metric rows:",
        len(metrics),
    )

    print(
        "Summary rows:",
        len(summary),
    )

    strongest = summary[
        summary[
            "positive_more_extreme_than_all_controls"
        ].eq(True)
    ].copy()

    print(
        "\nPositive source-background contrast "
        "more extreme than all controls:"
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
                    "positive_source_value",
                    "positive_background_value",
                    "positive_source_background_contrast",
                    "control_contrast_min",
                    "control_contrast_max",
                    "positive_absolute_rank_of_5",
                ]
            ].to_string(
                index=False,
            )
        )

    print("\nAll positive contrasts:")

    print(
        summary[
            [
                "release_rate_kg_h",
                "index_name",
                "statistic",
                "positive_source_background_contrast",
                "positive_absolute_rank_of_5",
                "positive_more_extreme_than_all_controls",
            ]
        ].to_string(
            index=False,
        )
    )

    print("\nSaved:")
    print(METRIC_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(PROFILE_OUTPUT)
    print(PLOT_DIR)


if __name__ == "__main__":
    main()
