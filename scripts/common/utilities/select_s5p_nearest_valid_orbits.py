from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_INPUT = Path(
    "outputs/500_s5p_ch4_event_orbit_candidates_v1.csv"
)

EVENT_INPUT = Path(
    "outputs/501_s5p_ch4_event_availability_v1.csv"
)

SELECTED_OUTPUT = Path(
    "outputs/503_s5p_nearest_valid_orbit_manifest_v1.csv"
)

TIER_OUTPUT = Path(
    "outputs/504_s5p_temporal_tier_summary_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/505_s5p_nearest_valid_orbit_report_v1.txt"
)


def parse_boolean(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0", "yes"])
    )


def temporal_tier(hours):
    if pd.isna(hours):
        return "missing"

    if hours <= 3:
        return "tier_a_0_to_3h"

    if hours <= 6:
        return "tier_b_3_to_6h"

    if hours <= 12:
        return "tier_c_6_to_12h"

    if hours <= 24:
        return "tier_d_12_to_24h"

    return "outside_24h"


def main():
    candidates = pd.read_csv(
        CANDIDATE_INPUT,
        low_memory=False,
    )

    events = pd.read_csv(
        EVENT_INPUT,
        low_memory=False,
    )

    required = [
        "event_id",
        "s5p_system_index",
        "s5p_orbit_time_utc",
        "time_difference_hours",
        "valid_local_ch4_coverage",
        "local_gridded_cell_count",
        "local_ch4_mean_ppb",
        "local_ch4_median_ppb",
        "local_uncertainty_mean_ppb",
    ]

    missing = [
        column
        for column in required
        if column not in candidates.columns
    ]

    if missing:
        raise KeyError(
            "Missing candidate columns: "
            + ", ".join(missing)
        )

    candidates[
        "_valid_local"
    ] = parse_boolean(
        candidates[
            "valid_local_ch4_coverage"
        ]
    )

    candidates[
        "time_difference_hours"
    ] = pd.to_numeric(
        candidates[
            "time_difference_hours"
        ],
        errors="coerce",
    )

    candidates[
        "local_gridded_cell_count"
    ] = pd.to_numeric(
        candidates[
            "local_gridded_cell_count"
        ],
        errors="coerce",
    )

    candidates[
        "local_uncertainty_mean_ppb"
    ] = pd.to_numeric(
        candidates[
            "local_uncertainty_mean_ppb"
        ],
        errors="coerce",
    )

    valid = candidates[
        candidates["_valid_local"]
        & candidates[
            "time_difference_hours"
        ].notna()
    ].copy()

    valid = valid.drop_duplicates(
        subset=[
            "event_id",
            "s5p_system_index",
        ]
    )

    valid[
        "_uncertainty_sort"
    ] = valid[
        "local_uncertainty_mean_ppb"
    ].fillna(np.inf)

    valid[
        "_cell_count_sort"
    ] = -valid[
        "local_gridded_cell_count"
    ].fillna(0)

    valid = valid.sort_values(
        [
            "event_id",
            "time_difference_hours",
            "_uncertainty_sort",
            "_cell_count_sort",
            "s5p_system_index",
        ]
    )

    candidate_counts = (
        valid.groupby("event_id")
        .size()
        .rename(
            "valid_candidate_orbit_count"
        )
    )

    selected = (
        valid.groupby(
            "event_id",
            as_index=False,
        )
        .first()
    )

    selected = selected.merge(
        candidate_counts,
        left_on="event_id",
        right_index=True,
        how="left",
    )

    selected[
        "s5p_temporal_tier"
    ] = selected[
        "time_difference_hours"
    ].map(temporal_tier)

    selected[
        "exact_temporal_overlap_confirmed"
    ] = False

    selected[
        "recommended_analysis_role"
    ] = np.select(
        [
            selected[
                "time_difference_hours"
            ].le(6),

            selected[
                "time_difference_hours"
            ].le(12),

            selected[
                "time_difference_hours"
            ].le(24),
        ],
        [
            "primary_near_time_regional_context",
            "secondary_near_time_regional_context",
            "sensitivity_only_regional_context",
        ],
        default="exclude_outside_24h",
    )

    event_id_column = (
        "s5p_event_id"
        if "s5p_event_id" in events.columns
        else "event_id"
    )

    event_columns = [
        column
        for column in [
            event_id_column,
            "s5p_event_time_utc",
            "s5p_true_release",
            "exact_interval_match_count",
            "exact_release_start_utc",
            "exact_release_end_utc",
            "exact_release_rate_median_kg_h",
            "s5p_ch4_status",
        ]
        if column in events.columns
    ]

    event_metadata = events[
        event_columns
    ].copy()

    event_metadata = event_metadata.rename(
        columns={
            event_id_column: "event_id"
        }
    )

    duplicate_columns = [
        column
        for column in event_metadata.columns
        if column != "event_id"
        and column in selected.columns
    ]

    event_metadata = event_metadata.drop(
        columns=duplicate_columns
    )

    selected = selected.merge(
        event_metadata,
        on="event_id",
        how="left",
    )

    selected = selected.drop(
        columns=[
            "_valid_local",
            "_uncertainty_sort",
            "_cell_count_sort",
        ],
        errors="ignore",
    )

    selected = selected.sort_values(
        [
            "time_difference_hours",
            "event_id",
        ]
    ).reset_index(drop=True)

    selected.to_csv(
        SELECTED_OUTPUT,
        index=False,
    )

    tier_summary = (
        selected.groupby(
            [
                "s5p_temporal_tier",
                "s5p_true_release",
            ],
            dropna=False,
        )
        .size()
        .rename("event_count")
        .reset_index()
    )

    tier_summary.to_csv(
        TIER_OUTPUT,
        index=False,
    )

    overall_tiers = (
        selected[
            "s5p_temporal_tier"
        ]
        .value_counts()
        .reindex(
            [
                "tier_a_0_to_3h",
                "tier_b_3_to_6h",
                "tier_c_6_to_12h",
                "tier_d_12_to_24h",
                "outside_24h",
            ],
            fill_value=0,
        )
    )

    role_summary = (
        selected[
            "recommended_analysis_role"
        ]
        .value_counts()
    )

    time_stats = selected[
        "time_difference_hours"
    ].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
        ]
    )

    label_summary = (
        selected[
            "s5p_true_release"
        ]
        .value_counts(
            dropna=False
        )
    )

    report_lines = [
        "=" * 110,
        "S5P NEAREST VALID ORBIT SELECTION V1",
        "=" * 110,
        "",
        f"Input candidate rows: {len(candidates)}",
        f"Valid candidate rows: {len(valid)}",
        (
            "Events with selected nearest valid orbit: "
            f"{len(selected)}"
        ),
        "",
        "Selected event labels:",
        label_summary.to_string(),
        "",
        "Nearest-orbit temporal tiers:",
        overall_tiers.to_string(),
        "",
        "Recommended analysis roles:",
        role_summary.to_string(),
        "",
        "Nearest time-difference statistics, hours:",
        time_stats.to_string(),
        "",
        "Important interpretation:",
        (
            "These observations are near-time regional "
            "XCH4 context only. No exact controlled-release "
            "temporal overlap has been confirmed."
        ),
        (
            "Earth Engine system:time_start is a product/orbit "
            "timestamp and is not the native pixel acquisition "
            "time at the source."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 110)
    print("S5P NEAREST VALID ORBIT SELECTION")
    print("=" * 110)

    print("\nInput candidate rows:", len(candidates))
    print("Valid candidate rows:", len(valid))

    print(
        "Events with selected nearest valid orbit:",
        len(selected),
    )

    print("\nSelected event labels:")
    print(label_summary)

    print("\nNearest-orbit temporal tiers:")
    print(overall_tiers)

    print("\nRecommended analysis roles:")
    print(role_summary)

    print("\nNearest time-difference statistics, hours:")
    print(time_stats)

    print("\nSaved:")
    print(SELECTED_OUTPUT)
    print(TIER_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
