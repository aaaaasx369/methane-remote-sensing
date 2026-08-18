from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

import analyze_s2_low_emission_spatial_anomalies as base


LOCKED_BENCHMARK_INPUT = Path(
    "outputs/341_s2_low_emission_pilot_v1_locked.csv"
)

LOCKED_PLAN_INPUT = Path(
    "outputs/342_s2_low_emission_wind_aligned_plan_v1.csv"
)

METRIC_OUTPUT = Path(
    "outputs/345_s2_locked_wind_aligned_metrics_v1.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/346_s2_locked_wind_aligned_summary_v1.csv"
)

SCENE_OUTPUT = Path(
    "outputs/347_s2_locked_wind_aligned_scene_conclusions_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/348_s2_locked_wind_aligned_report_v1.txt"
)


INDEX_NAMES = [
    "B12_B11_nd",
    "B12_B11_ratio",
    "B12_B8A_ratio",
]

CORE_INDEX_NAMES = [
    "B12_B11_nd",
    "B12_B11_ratio",
]


def circular_difference(
    angle_a,
    angle_b,
):
    return np.abs(
        (
            angle_a
            - angle_b
            + 180.0
        ) % 360.0
        - 180.0
    )


def direction_grids(
    profile,
    latitude,
    longitude,
):
    distance, _, _ = (
        base.source_distance_grid(
            profile,
            latitude,
            longitude,
        )
    )

    rows, columns = np.indices(
        distance.shape
    )

    xs, ys = rasterio.transform.xy(
        profile["transform"],
        rows,
        columns,
        offset="center",
    )

    xs = np.asarray(
        xs,
        dtype=np.float64,
    ).reshape(distance.shape)

    ys = np.asarray(
        ys,
        dtype=np.float64,
    ).reshape(distance.shape)

    source_x, source_y = transform(
        "EPSG:4326",
        profile["crs"],
        [longitude],
        [latitude],
    )

    dx = xs - float(source_x[0])
    dy = ys - float(source_y[0])

    # 0° = North，順時針增加。
    direction = (
        np.degrees(
            np.arctan2(
                dx,
                dy,
            )
        )
        + 360.0
    ) % 360.0

    return distance, direction


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
    ]).astype(np.float64)

    reference_valid_stack = np.stack([
        scene_valid_masks[
            reference_id
        ]
        for reference_id
        in reference_ids
    ])

    reference_stack[
        ~reference_valid_stack
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
            target_values
        )
        & np.isfinite(
            reference_median
        )
    )

    anomaly = (
        target_values
        - reference_median
    )

    return anomaly, valid


def extract_cone_values(
    anomaly,
    valid,
    distance,
    direction,
    center_direction,
    half_width,
    inner_radius,
    outer_radius,
):
    angular_difference = (
        circular_difference(
            direction,
            center_direction,
        )
    )

    mask = (
        valid
        & np.isfinite(anomaly)
        & (
            distance
            > inner_radius
        )
        & (
            distance
            <= outer_radius
        )
        & (
            angular_difference
            <= half_width
        )
    )

    values = anomaly[mask]

    return values[
        np.isfinite(values)
    ]


def calculate_statistic(
    values,
    statistic,
):
    values = np.asarray(
        values,
        dtype=np.float64,
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


def validate_inputs(
    benchmark,
    plan,
):
    if len(benchmark) != 10:
        raise RuntimeError(
            "Locked benchmark 應有 10 張影像，"
            f"實際為 {len(benchmark)}。"
        )

    labels = (
        pd.to_numeric(
            benchmark["label"],
            errors="raise",
        )
        .astype(int)
        .value_counts()
        .to_dict()
    )

    if labels != {
        0: 8,
        1: 2,
    }:
        raise RuntimeError(
            f"標籤數量錯誤：{labels}"
        )

    if (
        "benchmark_lock_status"
        not in benchmark.columns
    ):
        raise RuntimeError(
            "找不到 benchmark_lock_status。"
        )

    if not benchmark[
        "benchmark_lock_status"
    ].astype(str).eq("locked").all():
        raise RuntimeError(
            "Benchmark 尚未全部鎖定。"
        )

    if len(plan) != 2:
        raise RuntimeError(
            "Locked wind plan 應有 2 筆。"
        )


def main():
    benchmark = pd.read_csv(
        LOCKED_BENCHMARK_INPUT,
        low_memory=False,
    )

    plan = pd.read_csv(
        LOCKED_PLAN_INPUT,
        low_memory=False,
    )

    benchmark["label"] = pd.to_numeric(
        benchmark["label"],
        errors="raise",
    ).astype(int)

    benchmark["lat"] = pd.to_numeric(
        benchmark["lat"],
        errors="raise",
    )

    benchmark["lon"] = pd.to_numeric(
        benchmark["lon"],
        errors="raise",
    )

    numeric_plan_columns = [
        "release_rate_kg_h",
        "fixed_downwind_direction_degrees",
        "fixed_upwind_direction_degrees",
        "inner_radius_m",
        "outer_radius_m",
        "cone_half_width_degrees",
    ]

    for column in numeric_plan_columns:
        plan[column] = pd.to_numeric(
            plan[column],
            errors="raise",
        )

    validate_inputs(
        benchmark,
        plan,
    )

    metric_rows = []

    print("=" * 110)
    print(
        "LOCKED SENTINEL-2 WIND-ALIGNED ANALYSIS"
    )
    print("=" * 110)

    for group_id, group in (
        benchmark.groupby(
            "matched_group_id",
            sort=False,
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
            raise RuntimeError(
                f"Invalid matched group: {group_id}"
            )

        group_plan = plan[
            plan[
                "matched_group_id"
            ].astype(str).eq(
                str(group_id)
            )
        ]

        if len(group_plan) != 1:
            raise RuntimeError(
                "找不到唯一的 locked plan："
                f"{group_id}"
            )

        group_plan = (
            group_plan.iloc[0]
        )

        positive_row = (
            positive.iloc[0]
        )

        positive_id = str(
            positive_row["sample_id"]
        )

        negative_ids = (
            negatives["sample_id"]
            .astype(str)
            .tolist()
        )

        downwind_direction = float(
            group_plan[
                "fixed_downwind_direction_degrees"
            ]
        )

        upwind_direction = float(
            group_plan[
                "fixed_upwind_direction_degrees"
            ]
        )

        inner_radius = float(
            group_plan[
                "inner_radius_m"
            ]
        )

        outer_radius = float(
            group_plan[
                "outer_radius_m"
            ]
        )

        half_width = float(
            group_plan[
                "cone_half_width_degrees"
            ]
        )

        print(
            "\nMatched group:",
            group_id,
        )

        print(
            "Release rate:",
            group_plan[
                "release_rate_kg_h"
            ],
            "kg/h",
        )

        print(
            "Fixed downwind:",
            downwind_direction,
            "degrees",
        )

        print(
            "Fixed upwind:",
            upwind_direction,
            "degrees",
        )

        reference_array, profile = (
            base.read_reference(
                positive_row[
                    "patch_path"
                ]
            )
        )

        distance, direction = (
            direction_grids(
                profile,
                float(
                    positive_row["lat"]
                ),
                float(
                    positive_row["lon"]
                ),
            )
        )

        scene_arrays = {
            positive_id:
                reference_array
        }

        scene_roles = {
            positive_id:
                "positive"
        }

        for _, negative_row in (
            negatives.iterrows()
        ):
            sample_id = str(
                negative_row[
                    "sample_id"
                ]
            )

            scene_arrays[
                sample_id
            ] = base.read_to_reference(
                negative_row[
                    "patch_path"
                ],
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
            (
                scene_indices[
                    sample_id
                ],
                scene_valid_masks[
                    sample_id
                ],
            ) = base.calculate_indices(
                array
            )

        all_sample_ids = [
            positive_id,
            *negative_ids,
        ]

        for sample_id in (
            all_sample_ids
        ):
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

                downwind_values = (
                    extract_cone_values(
                        anomaly=
                            anomaly,
                        valid=
                            valid,
                        distance=
                            distance,
                        direction=
                            direction,
                        center_direction=
                            downwind_direction,
                        half_width=
                            half_width,
                        inner_radius=
                            inner_radius,
                        outer_radius=
                            outer_radius,
                    )
                )

                upwind_values = (
                    extract_cone_values(
                        anomaly=
                            anomaly,
                        valid=
                            valid,
                        distance=
                            distance,
                        direction=
                            direction,
                        center_direction=
                            upwind_direction,
                        half_width=
                            half_width,
                        inner_radius=
                            inner_radius,
                        outer_radius=
                            outer_radius,
                    )
                )

                for statistic in [
                    "mean",
                    "median",
                ]:
                    downwind_value = (
                        calculate_statistic(
                            downwind_values,
                            statistic,
                        )
                    )

                    upwind_value = (
                        calculate_statistic(
                            upwind_values,
                            statistic,
                        )
                    )

                    contrast = (
                        downwind_value
                        - upwind_value
                    )

                    metric_rows.append({
                        "matched_group_id":
                            group_id,

                        "release_rate_kg_h":
                            float(
                                group_plan[
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

                        "statistic":
                            statistic,

                        "fixed_downwind_direction_degrees":
                            downwind_direction,

                        "fixed_upwind_direction_degrees":
                            upwind_direction,

                        "inner_radius_m":
                            inner_radius,

                        "outer_radius_m":
                            outer_radius,

                        "cone_half_width_degrees":
                            half_width,

                        "downwind_pixel_count":
                            int(
                                len(
                                    downwind_values
                                )
                            ),

                        "upwind_pixel_count":
                            int(
                                len(
                                    upwind_values
                                )
                            ),

                        "downwind_value":
                            downwind_value,

                        "upwind_value":
                            upwind_value,

                        "downwind_minus_upwind":
                            contrast,

                        "absolute_downwind_upwind_contrast":
                            (
                                abs(contrast)
                                if np.isfinite(
                                    contrast
                                )
                                else np.nan
                            ),
                    })

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics.to_csv(
        METRIC_OUTPUT,
        index=False,
    )

    summary_rows = []

    group_columns = [
        "matched_group_id",
        "release_rate_kg_h",
        "index_name",
        "statistic",
    ]

    for keys, group in metrics.groupby(
        group_columns,
        sort=False,
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
            raise RuntimeError(
                "Comparison group 不完整："
                f"{keys}"
            )

        positive_row = (
            positive.iloc[0]
        )

        positive_contrast = float(
            positive_row[
                "downwind_minus_upwind"
            ]
        )

        control_contrasts = (
            pd.to_numeric(
                controls[
                    "downwind_minus_upwind"
                ],
                errors="coerce",
            )
            .dropna()
        )

        positive_absolute = abs(
            positive_contrast
        )

        control_absolute = (
            control_contrasts.abs()
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

            "fixed_downwind_direction_degrees":
                positive_row[
                    "fixed_downwind_direction_degrees"
                ],

            "positive_downwind_value":
                positive_row[
                    "downwind_value"
                ],

            "positive_upwind_value":
                positive_row[
                    "upwind_value"
                ],

            "positive_downwind_minus_upwind":
                positive_contrast,

            "positive_absolute_contrast":
                positive_absolute,

            "control_contrast_min":
                float(
                    control_contrasts.min()
                ),

            "control_contrast_max":
                float(
                    control_contrasts.max()
                ),

            "control_absolute_max":
                float(
                    control_absolute.max()
                ),

            "positive_signed_rank_of_5":
                int(
                    (
                        control_contrasts
                        < positive_contrast
                    ).sum()
                    + 1
                ),

            "positive_absolute_rank_of_5":
                int(
                    (
                        control_absolute
                        < positive_absolute
                    ).sum()
                    + 1
                ),

            "positive_greater_than_all_controls":
                bool(
                    positive_contrast
                    > control_contrasts.max()
                ),

            "positive_more_extreme_than_all_controls":
                bool(
                    positive_absolute
                    > control_absolute.max()
                ),
        })

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    scene_rows = []

    for (
        group_id,
        release_rate,
    ), group in summary.groupby(
        [
            "matched_group_id",
            "release_rate_kg_h",
        ],
        sort=False,
    ):
        core = group[
            group[
                "index_name"
            ].isin(
                CORE_INDEX_NAMES
            )
        ].copy()

        core_tests = len(core)

        extreme_tests = int(
            core[
                "positive_more_extreme_than_all_controls"
            ].sum()
        )

        directions = np.sign(
            pd.to_numeric(
                core[
                    "positive_downwind_minus_upwind"
                ],
                errors="coerce",
            )
        )

        nonzero_directions = (
            directions[
                directions.ne(0)
            ]
        )

        direction_consistent = bool(
            not nonzero_directions.empty
            and (
                nonzero_directions.nunique()
                == 1
            )
        )

        if (
            extreme_tests == core_tests
            and direction_consistent
        ):
            final_status = (
                "wind_aligned_anomaly_supported_"
                "in_this_scene"
            )

        elif extreme_tests >= 2:
            final_status = (
                "partial_or_inconsistent_"
                "wind_aligned_evidence"
            )

        else:
            final_status = (
                "no_robust_wind_aligned_detection"
            )

        scene_rows.append({
            "matched_group_id":
                group_id,

            "release_rate_kg_h":
                release_rate,

            "core_test_count":
                core_tests,

            "core_tests_more_extreme_than_all_controls":
                extreme_tests,

            "core_contrast_direction_consistent":
                direction_consistent,

            "final_wind_aligned_status":
                final_status,

            "interpretation_limit":
                (
                    "Pilot evidence only; one positive "
                    "and four matched controls per group. "
                    "Rank 5 of 5 is not a formal "
                    "statistical significance test."
                ),

            "analysis_stop_rule_applied":
                True,
        })

    scene_conclusions = (
        pd.DataFrame(
            scene_rows
        )
    )

    scene_conclusions.to_csv(
        SCENE_OUTPUT,
        index=False,
    )

    report_lines = [
        "=" * 110,
        "LOCKED SENTINEL-2 WIND-ALIGNED ANALYSIS REPORT",
        "=" * 110,
        "",
        (
            f"Metric rows: "
            f"{len(metrics)}"
        ),
        (
            f"Summary rows: "
            f"{len(summary)}"
        ),
        "",
        "LOCKED PARAMETERS",
    ]

    for _, row in plan.iterrows():
        report_lines.extend([
            "",
            (
                f"Release rate: "
                f"{row['release_rate_kg_h']:.6f} kg/h"
            ),
            (
                f"Downwind direction: "
                f"{row['fixed_downwind_direction_degrees']:.1f}°"
            ),
            (
                f"Upwind direction: "
                f"{row['fixed_upwind_direction_degrees']:.1f}°"
            ),
            (
                f"Radial range: "
                f"{row['inner_radius_m']:.0f}–"
                f"{row['outer_radius_m']:.0f} m"
            ),
            (
                f"Cone half width: "
                f"{row['cone_half_width_degrees']:.1f}°"
            ),
        ])

    report_lines.extend([
        "",
        "POSITIVE COMPARISONS",
        summary[
            [
                "release_rate_kg_h",
                "index_name",
                "statistic",
                "positive_downwind_value",
                "positive_upwind_value",
                "positive_downwind_minus_upwind",
                "control_absolute_max",
                "positive_absolute_rank_of_5",
                "positive_more_extreme_than_all_controls",
            ]
        ].to_string(index=False),
        "",
        "SCENE CONCLUSIONS",
        scene_conclusions.to_string(
            index=False
        ),
        "",
        "STOP RULE",
        (
            "No additional feature, direction, radius, "
            "or cone-width search should be performed "
            "for these two locked low-emission scenes."
        ),
    ])

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nMetric rows:", len(metrics))
    print("Summary rows:", len(summary))

    print(
        "\nWind-aligned positive comparisons:"
    )

    print(
        summary[
            [
                "release_rate_kg_h",
                "index_name",
                "statistic",
                "positive_downwind_value",
                "positive_upwind_value",
                "positive_downwind_minus_upwind",
                "control_absolute_max",
                "positive_absolute_rank_of_5",
                "positive_more_extreme_than_all_controls",
            ]
        ].to_string(
            index=False,
        )
    )

    print("\nFinal scene conclusions:")

    print(
        scene_conclusions[
            [
                "release_rate_kg_h",
                "core_test_count",
                "core_tests_more_extreme_than_all_controls",
                "core_contrast_direction_consistent",
                "final_wind_aligned_status",
            ]
        ].to_string(
            index=False,
        )
    )

    print("\nSaved:")
    print(METRIC_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(SCENE_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
