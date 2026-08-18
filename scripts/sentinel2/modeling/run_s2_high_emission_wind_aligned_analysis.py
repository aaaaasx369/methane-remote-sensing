from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

import analyze_s2_low_emission_spatial_anomalies as base


BENCHMARK_INPUT = Path(
    "outputs/372_s2_high_emission_benchmark_v1.csv"
)

WIND_PLAN_INPUT = Path(
    "outputs/383_s2_high_emission_wind_plan_complete_v2.csv"
)

METRIC_OUTPUT = Path(
    "outputs/385_s2_high_emission_wind_aligned_metrics_v1.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/386_s2_high_emission_wind_aligned_summary_v1.csv"
)

SCENE_OUTPUT = Path(
    "outputs/387_s2_high_emission_wind_aligned_scene_conclusions_v1.csv"
)

TIER_OUTPUT = Path(
    "outputs/388_s2_high_emission_wind_aligned_tier_summary_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/389_s2_high_emission_wind_aligned_report_v1.txt"
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

STATISTICS = [
    "mean",
    "median",
]

INNER_RADIUS_M = 50.0
OUTER_RADIUS_M = 300.0
CONE_HALF_WIDTH_DEGREES = 22.5

EXPECTED_GROUPS = 7
EXPECTED_POSITIVES = 7
EXPECTED_NEGATIVES = 28
EXPECTED_TOTAL = 35


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
            > INNER_RADIUS_M
        )
        & (
            distance
            <= OUTER_RADIUS_M
        )
        & (
            angular_difference
            <= CONE_HALF_WIDTH_DEGREES
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
    benchmark["label"] = pd.to_numeric(
        benchmark["label"],
        errors="raise",
    ).astype(int)

    counts = (
        benchmark["label"]
        .value_counts()
        .to_dict()
    )

    if len(benchmark) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Benchmark 應有 {EXPECTED_TOTAL} 筆，"
            f"實際為 {len(benchmark)}。"
        )

    if counts != {
        0: EXPECTED_NEGATIVES,
        1: EXPECTED_POSITIVES,
    }:
        raise RuntimeError(
            f"標籤數量不正確：{counts}"
        )

    if (
        benchmark[
            "matched_group_id"
        ].nunique()
        != EXPECTED_GROUPS
    ):
        raise RuntimeError(
            "Matched group 數量不是 7。"
        )

    group_structure = (
        benchmark.groupby(
            "matched_group_id"
        )["label"]
        .agg(
            total="size",
            positives="sum",
        )
    )

    invalid = group_structure[
        (group_structure["total"] != 5)
        | (
            group_structure["positives"]
            != 1
        )
    ]

    if not invalid.empty:
        raise RuntimeError(
            "Matched group 結構錯誤：\n"
            + invalid.to_string()
        )

    if len(plan) != EXPECTED_GROUPS:
        raise RuntimeError(
            "Wind plan 應有 7 筆，"
            f"實際為 {len(plan)}。"
        )

    required_plan_columns = [
        "positive_id",
        "scene_id",
        "fixed_downwind_direction_degrees",
        "fixed_upwind_direction_degrees",
        "wind_source_type",
        "wind_evidence_tier",
        "wind_plan_status",
    ]

    missing = [
        column
        for column in required_plan_columns
        if column not in plan.columns
    ]

    if missing:
        raise KeyError(
            "Wind plan 缺少欄位："
            + ", ".join(missing)
        )

    if plan[
        [
            "fixed_downwind_direction_degrees",
            "fixed_upwind_direction_degrees",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Wind plan 還有缺失方向。"
        )

    return group_structure


def prepare_plan(plan):
    plan = plan.copy()

    plan[
        "fixed_downwind_direction_degrees"
    ] = pd.to_numeric(
        plan[
            "fixed_downwind_direction_degrees"
        ],
        errors="raise",
    )

    plan[
        "fixed_upwind_direction_degrees"
    ] = pd.to_numeric(
        plan[
            "fixed_upwind_direction_degrees"
        ],
        errors="raise",
    )

    plan[
        "release_rate_kg_h"
    ] = pd.to_numeric(
        plan[
            "release_rate_kg_h"
        ],
        errors="raise",
    )

    # Wind plan 的 positive scene_id
    # 就是 benchmark 的 matched_group_id。
    plan[
        "matched_group_id"
    ] = plan[
        "scene_id"
    ].astype(str)

    return plan


def build_metric_table(
    benchmark,
    plan,
):
    metric_rows = []

    for group_number, (
        group_id,
        group,
    ) in enumerate(
        benchmark.groupby(
            "matched_group_id",
            sort=False,
        ),
        start=1,
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

        plan_group = plan[
            plan[
                "matched_group_id"
            ].astype(str).eq(
                str(group_id)
            )
        ]

        if len(plan_group) != 1:
            raise RuntimeError(
                "找不到唯一 wind plan："
                f"{group_id}"
            )

        plan_row = plan_group.iloc[0]
        positive_row = positive.iloc[0]

        positive_id = str(
            positive_row["sample_id"]
        )

        negative_ids = (
            negatives["sample_id"]
            .astype(str)
            .tolist()
        )

        downwind_direction = float(
            plan_row[
                "fixed_downwind_direction_degrees"
            ]
        )

        upwind_direction = float(
            plan_row[
                "fixed_upwind_direction_degrees"
            ]
        )

        print(
            f"\n[{group_number}/{EXPECTED_GROUPS}] "
            f"{plan_row['positive_id']} | "
            f"{plan_row['release_rate_kg_h']:.3f} kg/h",
            flush=True,
        )

        print(
            "  Wind tier:",
            plan_row[
                "wind_evidence_tier"
            ],
        )

        print(
            "  Downwind:",
            f"{downwind_direction:.2f}°",
        )

        print(
            "  Upwind:",
            f"{upwind_direction:.2f}°",
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

        for sample_id in all_sample_ids:
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

            for index_name in INDEX_NAMES:
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
                    )
                )

                for statistic in STATISTICS:
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
                        "positive_id":
                            plan_row[
                                "positive_id"
                            ],

                        "matched_group_id":
                            group_id,

                        "positive_scene_id":
                            plan_row[
                                "scene_id"
                            ],

                        "release_rate_kg_h":
                            plan_row[
                                "release_rate_kg_h"
                            ],

                        "site":
                            plan_row[
                                "site"
                            ],

                        "wind_evidence_tier":
                            plan_row[
                                "wind_evidence_tier"
                            ],

                        "wind_source_type":
                            plan_row[
                                "wind_source_type"
                            ],

                        "wind_plan_status":
                            plan_row[
                                "wind_plan_status"
                            ],

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
                            INNER_RADIUS_M,

                        "outer_radius_m":
                            OUTER_RADIUS_M,

                        "cone_half_width_degrees":
                            CONE_HALF_WIDTH_DEGREES,

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

    return pd.DataFrame(
        metric_rows
    )


def build_positive_summary(metrics):
    summary_rows = []

    group_columns = [
        "positive_id",
        "matched_group_id",
        "release_rate_kg_h",
        "site",
        "wind_evidence_tier",
        "wind_source_type",
        "index_name",
        "statistic",
    ]

    for keys, group in metrics.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        (
            positive_id,
            group_id,
            release_rate,
            site,
            wind_tier,
            wind_source,
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
                "Summary group 不完整："
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

        control_absolute_max = float(
            control_absolute.max()
        )

        if control_absolute_max > 0:
            relative_to_control_max = (
                positive_absolute
                / control_absolute_max
            )
        else:
            relative_to_control_max = np.nan

        summary_rows.append({
            "positive_id":
                positive_id,

            "matched_group_id":
                group_id,

            "release_rate_kg_h":
                release_rate,

            "site":
                site,

            "wind_evidence_tier":
                wind_tier,

            "wind_source_type":
                wind_source,

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
                control_absolute_max,

            "positive_absolute_to_control_max_ratio":
                relative_to_control_max,

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

    return pd.DataFrame(
        summary_rows
    )


def build_scene_conclusions(summary):
    scene_rows = []

    group_columns = [
        "positive_id",
        "matched_group_id",
        "release_rate_kg_h",
        "site",
        "wind_evidence_tier",
        "wind_source_type",
    ]

    for keys, group in summary.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        (
            positive_id,
            matched_group_id,
            release_rate,
            site,
            wind_tier,
            wind_source,
        ) = keys

        core = group[
            group[
                "index_name"
            ].isin(
                CORE_INDEX_NAMES
            )
        ].copy()

        core_test_count = len(core)

        extreme_count = int(
            core[
                "positive_more_extreme_than_all_controls"
            ].sum()
        )

        positive_signs = np.sign(
            pd.to_numeric(
                core[
                    "positive_downwind_minus_upwind"
                ],
                errors="coerce",
            )
        )

        nonzero_signs = (
            positive_signs[
                positive_signs.ne(0)
            ]
        )

        sign_consistent = bool(
            not nonzero_signs.empty
            and nonzero_signs.nunique()
            == 1
        )

        all_positive_direction = bool(
            len(nonzero_signs)
            == core_test_count
            and (
                nonzero_signs
                > 0
            ).all()
        )

        median_rank = float(
            core[
                "positive_absolute_rank_of_5"
            ].median()
        )

        median_control_ratio = float(
            core[
                "positive_absolute_to_control_max_ratio"
            ].median()
        )

        if (
            extreme_count == core_test_count
            and sign_consistent
        ):
            status = (
                "strong_consistent_wind_aligned_anomaly"
            )

        elif extreme_count >= 2:
            status = (
                "partial_wind_aligned_anomaly"
            )

        else:
            status = (
                "no_robust_wind_aligned_anomaly"
            )

        if wind_tier == "primary":
            interpretation = (
                "Primary wind evidence from "
                "experiment or ground-truth records."
            )
        else:
            interpretation = (
                "Secondary sensitivity analysis "
                "using hourly external reanalysis wind."
            )

        scene_rows.append({
            "positive_id":
                positive_id,

            "matched_group_id":
                matched_group_id,

            "release_rate_kg_h":
                release_rate,

            "site":
                site,

            "wind_evidence_tier":
                wind_tier,

            "wind_source_type":
                wind_source,

            "core_test_count":
                core_test_count,

            "core_tests_more_extreme_than_all_controls":
                extreme_count,

            "core_contrast_sign_consistent":
                sign_consistent,

            "all_core_contrasts_downwind_positive":
                all_positive_direction,

            "median_core_absolute_rank_of_5":
                median_rank,

            "median_core_absolute_to_control_max_ratio":
                median_control_ratio,

            "final_wind_aligned_status":
                status,

            "interpretation":
                interpretation,

            "statistical_limit":
                (
                    "Rank 5 of 5 is descriptive only "
                    "and is not a formal significance test."
                ),
        })

    return pd.DataFrame(
        scene_rows
    )


def build_tier_summary(scene_conclusions):
    rows = []

    for tier, group in (
        scene_conclusions.groupby(
            "wind_evidence_tier",
            sort=False,
        )
    ):
        rows.append({
            "wind_evidence_tier":
                tier,

            "scene_count":
                len(group),

            "strong_consistent_count":
                int(
                    group[
                        "final_wind_aligned_status"
                    ].eq(
                        "strong_consistent_wind_aligned_anomaly"
                    ).sum()
                ),

            "partial_count":
                int(
                    group[
                        "final_wind_aligned_status"
                    ].eq(
                        "partial_wind_aligned_anomaly"
                    ).sum()
                ),

            "no_robust_count":
                int(
                    group[
                        "final_wind_aligned_status"
                    ].eq(
                        "no_robust_wind_aligned_anomaly"
                    ).sum()
                ),

            "median_release_rate_kg_h":
                float(
                    group[
                        "release_rate_kg_h"
                    ].median()
                ),

            "median_core_extreme_test_count":
                float(
                    group[
                        "core_tests_more_extreme_than_all_controls"
                    ].median()
                ),

            "median_core_control_ratio":
                float(
                    group[
                        "median_core_absolute_to_control_max_ratio"
                    ].median()
                ),
        })

    total = scene_conclusions

    rows.append({
        "wind_evidence_tier":
            "all_descriptive_only",

        "scene_count":
            len(total),

        "strong_consistent_count":
            int(
                total[
                    "final_wind_aligned_status"
                ].eq(
                    "strong_consistent_wind_aligned_anomaly"
                ).sum()
            ),

        "partial_count":
            int(
                total[
                    "final_wind_aligned_status"
                ].eq(
                    "partial_wind_aligned_anomaly"
                ).sum()
            ),

        "no_robust_count":
            int(
                total[
                    "final_wind_aligned_status"
                ].eq(
                    "no_robust_wind_aligned_anomaly"
                ).sum()
            ),

        "median_release_rate_kg_h":
            float(
                total[
                    "release_rate_kg_h"
                ].median()
            ),

        "median_core_extreme_test_count":
            float(
                total[
                    "core_tests_more_extreme_than_all_controls"
                ].median()
            ),

        "median_core_control_ratio":
            float(
                total[
                    "median_core_absolute_to_control_max_ratio"
                ].median()
            ),
    })

    return pd.DataFrame(rows)


def main():
    benchmark = pd.read_csv(
        BENCHMARK_INPUT,
        low_memory=False,
    )

    plan = pd.read_csv(
        WIND_PLAN_INPUT,
        low_memory=False,
    )

    benchmark["lat"] = pd.to_numeric(
        benchmark["lat"],
        errors="raise",
    )

    benchmark["lon"] = pd.to_numeric(
        benchmark["lon"],
        errors="raise",
    )

    plan = prepare_plan(plan)

    group_structure = validate_inputs(
        benchmark,
        plan,
    )

    print("=" * 115)
    print(
        "SENTINEL-2 HIGH-EMISSION "
        "LOCKED WIND-ALIGNED ANALYSIS"
    )
    print("=" * 115)

    print(
        "\nBenchmark rows:",
        len(benchmark),
    )

    print(
        "Matched groups:",
        len(group_structure),
    )

    print(
        "\nLocked parameters:"
    )

    print(
        f"  Radius: "
        f"{INNER_RADIUS_M:.0f}–"
        f"{OUTER_RADIUS_M:.0f} m"
    )

    print(
        f"  Cone half width: "
        f"{CONE_HALF_WIDTH_DEGREES:.1f}°"
    )

    print(
        "  Indices:",
        ", ".join(INDEX_NAMES),
    )

    metrics = build_metric_table(
        benchmark,
        plan,
    )

    metrics.to_csv(
        METRIC_OUTPUT,
        index=False,
    )

    summary = build_positive_summary(
        metrics
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    scene_conclusions = (
        build_scene_conclusions(
            summary
        )
    )

    scene_conclusions.to_csv(
        SCENE_OUTPUT,
        index=False,
    )

    tier_summary = build_tier_summary(
        scene_conclusions
    )

    tier_summary.to_csv(
        TIER_OUTPUT,
        index=False,
    )

    expected_metric_rows = (
        EXPECTED_GROUPS
        * 5
        * len(INDEX_NAMES)
        * len(STATISTICS)
    )

    expected_summary_rows = (
        EXPECTED_GROUPS
        * len(INDEX_NAMES)
        * len(STATISTICS)
    )

    if len(metrics) != expected_metric_rows:
        raise RuntimeError(
            "Metric rows 數量錯誤："
            f"{len(metrics)}，"
            f"預期 {expected_metric_rows}。"
        )

    if len(summary) != expected_summary_rows:
        raise RuntimeError(
            "Summary rows 數量錯誤："
            f"{len(summary)}，"
            f"預期 {expected_summary_rows}。"
        )

    report_lines = [
        "=" * 115,
        (
            "SENTINEL-2 HIGH-EMISSION "
            "WIND-ALIGNED ANALYSIS REPORT V1"
        ),
        "=" * 115,
        "",
        f"Benchmark rows: {len(benchmark)}",
        f"Matched groups: {EXPECTED_GROUPS}",
        f"Metric rows: {len(metrics)}",
        f"Positive summary rows: {len(summary)}",
        "",
        "Locked parameters:",
        (
            f"  Radial range: "
            f"{INNER_RADIUS_M:.0f}–"
            f"{OUTER_RADIUS_M:.0f} m"
        ),
        (
            f"  Cone half width: "
            f"{CONE_HALF_WIDTH_DEGREES:.1f} degrees"
        ),
        (
            "  Indices: "
            + ", ".join(INDEX_NAMES)
        ),
        (
            "  Statistics: "
            + ", ".join(STATISTICS)
        ),
        "",
        "Scene conclusions:",
        scene_conclusions[
            [
                "positive_id",
                "release_rate_kg_h",
                "site",
                "wind_evidence_tier",
                "core_tests_more_extreme_than_all_controls",
                "core_contrast_sign_consistent",
                "all_core_contrasts_downwind_positive",
                "median_core_absolute_to_control_max_ratio",
                "final_wind_aligned_status",
            ]
        ].to_string(index=False),
        "",
        "Evidence-tier summary:",
        tier_summary.to_string(
            index=False
        ),
        "",
        "Interpretation limits:",
        (
            "Primary and secondary wind evidence "
            "must be reported separately."
        ),
        (
            "The analysis identifies wind-aligned "
            "spectral anomalies, not confirmed methane."
        ),
        (
            "Rank 5 of 5 is descriptive and not a "
            "formal statistical significance result."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 115)
    print("WIND-ALIGNED ANALYSIS SUMMARY")
    print("=" * 115)

    print(
        "\nMetric rows:",
        len(metrics),
    )

    print(
        "Positive summary rows:",
        len(summary),
    )

    print("\nScene conclusions:")

    print(
        scene_conclusions[
            [
                "positive_id",
                "release_rate_kg_h",
                "site",
                "wind_evidence_tier",
                "core_tests_more_extreme_than_all_controls",
                "core_contrast_sign_consistent",
                "all_core_contrasts_downwind_positive",
                "median_core_absolute_to_control_max_ratio",
                "final_wind_aligned_status",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nEvidence-tier summary:")

    print(
        tier_summary.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(METRIC_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(SCENE_OUTPUT)
    print(TIER_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
