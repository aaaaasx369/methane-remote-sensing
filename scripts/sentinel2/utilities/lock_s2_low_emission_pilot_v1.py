from pathlib import Path
import hashlib

import pandas as pd


BENCHMARK_INPUT = Path(
    "outputs/327_s2_low_emission_pilot_benchmark_v1.csv"
)

ANALYSIS_INPUTS = [
    Path(
        "outputs/318_s2_low_emission_primary_scenes_v1.csv"
    ),
    Path(
        "outputs/324_s2_low_emission_matched_negative_manifest_v2.csv"
    ),
    Path(
        "outputs/327_s2_low_emission_pilot_benchmark_v1.csv"
    ),
    Path(
        "outputs/329_s2_low_emission_matched_anomaly_features_v1.csv"
    ),
    Path(
        "outputs/331_s2_low_emission_spatial_anomaly_summary.csv"
    ),
    Path(
        "outputs/333_s2_source_background_contrast_summary.csv"
    ),
    Path(
        "outputs/336_s2_directional_sector_summary.csv"
    ),
    Path(
        "outputs/339_s2_true_wind_direction_evidence.csv"
    ),
]

LOCKED_BENCHMARK_OUTPUT = Path(
    "outputs/341_s2_low_emission_pilot_v1_locked.csv"
)

WIND_PLAN_OUTPUT = Path(
    "outputs/342_s2_low_emission_wind_aligned_plan_v1.csv"
)

HASH_OUTPUT = Path(
    "outputs/343_s2_low_emission_pilot_v1_input_hashes.csv"
)

REPORT_OUTPUT = Path(
    "outputs/344_s2_low_emission_pilot_v1_final_report.txt"
)


INDEX_NAMES = [
    "B12_B11_nd",
    "B12_B11_ratio",
    "B12_B8A_ratio",
]

INNER_RADIUS_M = 50
OUTER_RADIUS_M = 300
CONE_HALF_WIDTH_DEGREES = 22.5


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def choose_fixed_plan(release_rate):
    release_rate = float(
        release_rate
    )

    if release_rate < 200:
        return {
            "fixed_downwind_direction_degrees":
                254.0,

            "fixed_upwind_direction_degrees":
                74.0,

            "previous_observed_anomaly_direction_degrees":
                315.0,

            "pre_wind_result":
                (
                    "B12-relative anomaly detected; "
                    "strongest post-hoc sector was NW"
                ),

            "wind_validation_result":
                (
                    "NW anomaly inconsistent with "
                    "short-term wind-derived plume direction"
                ),

            "locked_scene_conclusion":
                (
                    "spectral_anomaly_detected_but_"
                    "wind_inconsistent_not_attributed_to_methane"
                ),
        }

    return {
        "fixed_downwind_direction_degrees":
            230.0,

        "fixed_upwind_direction_degrees":
            50.0,

        "previous_observed_anomaly_direction_degrees":
            pd.NA,

        "pre_wind_result":
            (
                "No robust reproducible "
                "B12-relative detection"
            ),

        "wind_validation_result":
            (
                "No convincing anomaly aligned "
                "with the expected plume direction"
            ),

        "locked_scene_conclusion":
            (
                "no_robust_detection_inconclusive"
            ),
    }


def validate_benchmark(benchmark):
    required = [
        "sample_id",
        "scene_id",
        "matched_group_id",
        "site",
        "acquisition_time_utc",
        "release_rate_kg_h",
        "label",
        "patch_path",
    ]

    missing = [
        column
        for column in required
        if column not in benchmark.columns
    ]

    if missing:
        raise KeyError(
            "Benchmark 缺少欄位："
            + ", ".join(missing)
        )

    benchmark["label"] = pd.to_numeric(
        benchmark["label"],
        errors="raise",
    ).astype(int)

    counts = (
        benchmark["label"]
        .value_counts()
        .to_dict()
    )

    if counts != {
        0: 8,
        1: 2,
    }:
        raise RuntimeError(
            "預期 label 0=8、label 1=2，"
            f"實際為 {counts}"
        )

    if len(benchmark) != 10:
        raise RuntimeError(
            "預期總樣本為 10，"
            f"實際為 {len(benchmark)}"
        )

    group_summary = (
        benchmark.groupby(
            "matched_group_id"
        )["label"]
        .agg(
            total="size",
            positives="sum",
        )
    )

    invalid = group_summary[
        (group_summary["total"] != 5)
        | (
            group_summary[
                "positives"
            ] != 1
        )
    ]

    if not invalid.empty:
        raise RuntimeError(
            "Matched group 結構不正確：\n"
            + invalid.to_string()
        )

    return group_summary


def main():
    if not BENCHMARK_INPUT.exists():
        raise FileNotFoundError(
            BENCHMARK_INPUT
        )

    benchmark = pd.read_csv(
        BENCHMARK_INPUT,
        low_memory=False,
    )

    group_summary = validate_benchmark(
        benchmark
    )

    positive = benchmark[
        benchmark["label"].eq(1)
    ].copy()

    positive[
        "release_rate_kg_h"
    ] = pd.to_numeric(
        positive[
            "release_rate_kg_h"
        ],
        errors="raise",
    )

    positive = positive.sort_values(
        "release_rate_kg_h"
    ).reset_index(drop=True)

    if len(positive) != 2:
        raise RuntimeError(
            "預期有 2 張 positive。"
        )

    plan_rows = []

    for _, row in positive.iterrows():
        plan = choose_fixed_plan(
            row["release_rate_kg_h"]
        )

        plan_rows.append({
            "matched_group_id":
                row[
                    "matched_group_id"
                ],

            "positive_sample_id":
                row["sample_id"],

            "positive_scene_id":
                row["scene_id"],

            "site":
                row["site"],

            "acquisition_time_utc":
                row[
                    "acquisition_time_utc"
                ],

            "release_rate_kg_h":
                row[
                    "release_rate_kg_h"
                ],

            **plan,

            "inner_radius_m":
                INNER_RADIUS_M,

            "outer_radius_m":
                OUTER_RADIUS_M,

            "cone_half_width_degrees":
                CONE_HALF_WIDTH_DEGREES,

            "fixed_indices":
                "|".join(
                    INDEX_NAMES
                ),

            "fixed_statistics":
                "mean|median",

            "comparison":
                (
                    "downwind_cone_minus_"
                    "upwind_cone"
                ),

            "control_design":
                (
                    "positive compared with four "
                    "matched negatives; each negative "
                    "compared with the other three"
                ),

            "analysis_stop_rule":
                (
                    "After this preregistered "
                    "wind-aligned analysis, do not "
                    "search additional features, "
                    "directions, cone widths, or radii "
                    "for these two scenes."
                ),

            "plan_version":
                "s2_low_emission_wind_plan_v1",
        })

    plan_table = pd.DataFrame(
        plan_rows
    )

    locked = benchmark.merge(
        plan_table[
            [
                "matched_group_id",
                "fixed_downwind_direction_degrees",
                "fixed_upwind_direction_degrees",
                "inner_radius_m",
                "outer_radius_m",
                "cone_half_width_degrees",
                "fixed_indices",
                "locked_scene_conclusion",
                "analysis_stop_rule",
                "plan_version",
            ]
        ],
        on="matched_group_id",
        how="left",
        validate="many_to_one",
    )

    locked[
        "benchmark_lock_status"
    ] = "locked"

    locked[
        "benchmark_version"
    ] = (
        "s2_low_emission_pilot_v1_locked"
    )

    locked.to_csv(
        LOCKED_BENCHMARK_OUTPUT,
        index=False,
    )

    plan_table.to_csv(
        WIND_PLAN_OUTPUT,
        index=False,
    )

    hash_rows = []

    for path in ANALYSIS_INPUTS:
        hash_rows.append({
            "file_path":
                str(path),

            "file_exists":
                path.exists(),

            "file_size_bytes":
                (
                    path.stat().st_size
                    if path.exists()
                    else pd.NA
                ),

            "sha256":
                (
                    sha256_file(path)
                    if path.exists()
                    else pd.NA
                ),
        })

    hashes = pd.DataFrame(
        hash_rows
    )

    hashes.to_csv(
        HASH_OUTPUT,
        index=False,
    )

    report_lines = [
        "=" * 105,
        "SENTINEL-2 LOW-EMISSION PILOT V1 — LOCKED REPORT",
        "=" * 105,
        "",
        "Dataset structure:",
        f"  Total samples: {len(benchmark)}",
        (
            "  Positive samples: "
            f"{int((benchmark['label'] == 1).sum())}"
        ),
        (
            "  Matched negative samples: "
            f"{int((benchmark['label'] == 0).sum())}"
        ),
        (
            "  Matched groups: "
            f"{benchmark['matched_group_id'].nunique()}"
        ),
        "",
        "Each matched group:",
        group_summary.to_string(),
        "",
        "Locked scientific conclusions:",
    ]

    for _, row in plan_table.iterrows():
        report_lines.extend([
            "",
            (
                f"Release rate: "
                f"{row['release_rate_kg_h']:.6f} kg/h"
            ),
            (
                f"Scene: "
                f"{row['positive_scene_id']}"
            ),
            (
                f"Fixed downwind direction: "
                f"{row['fixed_downwind_direction_degrees']:.1f}°"
            ),
            (
                f"Fixed upwind direction: "
                f"{row['fixed_upwind_direction_degrees']:.1f}°"
            ),
            (
                f"Previous result: "
                f"{row['pre_wind_result']}"
            ),
            (
                f"Wind validation: "
                f"{row['wind_validation_result']}"
            ),
            (
                f"Final locked conclusion: "
                f"{row['locked_scene_conclusion']}"
            ),
        ])

    report_lines.extend([
        "",
        "Final preregistered analysis:",
        (
            f"  Radial range: "
            f"{INNER_RADIUS_M}–{OUTER_RADIUS_M} m"
        ),
        (
            f"  Cone half width: "
            f"{CONE_HALF_WIDTH_DEGREES}°"
        ),
        (
            "  Indices: "
            + ", ".join(
                INDEX_NAMES
            )
        ),
        "  Statistics: mean and median",
        (
            "  Comparison: "
            "downwind cone minus upwind cone"
        ),
        "",
        "Stop rule:",
        (
            "  After the wind-aligned analysis, "
            "no additional feature, direction, "
            "radius, or cone-width search will be "
            "performed for these two scenes."
        ),
        "",
        "Locked outputs:",
        f"  {LOCKED_BENCHMARK_OUTPUT}",
        f"  {WIND_PLAN_OUTPUT}",
        f"  {HASH_OUTPUT}",
    ])

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 105)
    print(
        "SENTINEL-2 LOW-EMISSION PILOT V1 LOCKED"
    )
    print("=" * 105)

    print(
        "\nTotal samples:",
        len(locked),
    )

    print("\nLabels:")
    print(
        locked[
            "label"
        ].value_counts().sort_index()
    )

    print("\nMatched groups:")
    print(group_summary)

    print("\nLocked wind plan:")
    print(
        plan_table[
            [
                "release_rate_kg_h",
                "fixed_downwind_direction_degrees",
                "fixed_upwind_direction_degrees",
                "inner_radius_m",
                "outer_radius_m",
                "cone_half_width_degrees",
                "locked_scene_conclusion",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nExisting analysis files hashed:",
        int(
            hashes[
                "file_exists"
            ].sum()
        ),
        "/",
        len(hashes),
    )

    print("\nSaved:")
    print(LOCKED_BENCHMARK_OUTPUT)
    print(WIND_PLAN_OUTPUT)
    print(HASH_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
