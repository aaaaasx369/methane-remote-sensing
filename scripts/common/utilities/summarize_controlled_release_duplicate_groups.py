from pathlib import Path
from math import radians, sin, cos, asin, sqrt

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/469_controlled_release_possible_duplicate_groups_v1.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/471_controlled_release_duplicate_group_summary_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/472_controlled_release_duplicate_group_review_report_v1.txt"
)


def haversine_m(lat1, lon1, lat2, lon2):
    radius_m = 6371000.0

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    value = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    return 2 * radius_m * asin(sqrt(value))


def summarize_group(group_id, group):
    starts = group["release_start_utc"]
    ends = group["release_end_utc"]

    earliest_start = starts.min()
    latest_start = starts.max()

    earliest_end = ends.min()
    latest_end = ends.max()

    # 所有紀錄共同存在的時間區間。
    consensus_start = latest_start
    consensus_end = earliest_end

    common_overlap_minutes = (
        consensus_end - consensus_start
    ).total_seconds() / 60.0

    union_duration_minutes = (
        latest_end - earliest_start
    ).total_seconds() / 60.0

    start_spread_minutes = (
        latest_start - earliest_start
    ).total_seconds() / 60.0

    end_spread_minutes = (
        latest_end - earliest_end
    ).total_seconds() / 60.0

    median_latitude = group["latitude"].median()
    median_longitude = group["longitude"].median()

    distances = group.apply(
        lambda row: haversine_m(
            row["latitude"],
            row["longitude"],
            median_latitude,
            median_longitude,
        ),
        axis=1,
    )

    emission = group[
        "emission_kg_hr"
    ].dropna()

    if emission.empty:
        emission_min = np.nan
        emission_median = np.nan
        emission_max = np.nan
        emission_ratio = np.nan
    else:
        emission_min = emission.min()
        emission_median = emission.median()
        emission_max = emission.max()

        if emission_min > 0:
            emission_ratio = (
                emission_max / emission_min
            )
        else:
            emission_ratio = np.nan

    reasons = []

    if common_overlap_minutes <= 0:
        reasons.append(
            "no_common_temporal_intersection"
        )

    if start_spread_minutes > 10:
        reasons.append(
            "large_start_time_spread"
        )

    if end_spread_minutes > 10:
        reasons.append(
            "large_end_time_spread"
        )

    if distances.max() > 100:
        reasons.append(
            "large_location_spread"
        )

    if (
        pd.notna(emission_ratio)
        and emission_ratio > 2
    ):
        reasons.append(
            "large_emission_disagreement"
        )

    if len(group) >= 8:
        reasons.append(
            "large_group_possible_chain_merge"
        )

    # 自動安全群組只根據非常保守的條件。
    safe_temporal_location_merge = (
        common_overlap_minutes > 0
        and start_spread_minutes <= 10
        and end_spread_minutes <= 10
        and distances.max() <= 100
        and len(group) < 8
    )

    if safe_temporal_location_merge:
        review_status = (
            "safe_temporal_location_consensus"
        )
    else:
        review_status = "manual_review_required"

    return {
        "possible_release_group_id":
            group_id,

        "group_row_count":
            len(group),

        "unique_source_dataset_count":
            group["source_dataset"].nunique(),

        "source_datasets":
            " || ".join(
                sorted(
                    group["source_dataset"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            ),

        "earliest_release_start_utc":
            earliest_start,

        "latest_release_start_utc":
            latest_start,

        "earliest_release_end_utc":
            earliest_end,

        "latest_release_end_utc":
            latest_end,

        "consensus_start_utc":
            consensus_start,

        "consensus_end_utc":
            consensus_end,

        "common_overlap_minutes":
            common_overlap_minutes,

        "union_duration_minutes":
            union_duration_minutes,

        "start_spread_minutes":
            start_spread_minutes,

        "end_spread_minutes":
            end_spread_minutes,

        "median_latitude":
            median_latitude,

        "median_longitude":
            median_longitude,

        "maximum_distance_from_median_m":
            distances.max(),

        "emission_min_kg_hr":
            emission_min,

        "emission_median_kg_hr":
            emission_median,

        "emission_max_kg_hr":
            emission_max,

        "emission_max_min_ratio":
            emission_ratio,

        "safe_temporal_location_merge":
            safe_temporal_location_merge,

        "review_status":
            review_status,

        "review_reasons":
            (
                " | ".join(reasons)
                if reasons
                else ""
            ),
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    frame = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    frame["release_start_utc"] = pd.to_datetime(
        frame["release_start_utc"],
        errors="coerce",
        utc=True,
    )

    frame["release_end_utc"] = pd.to_datetime(
        frame["release_end_utc"],
        errors="coerce",
        utc=True,
    )

    for column in [
        "latitude",
        "longitude",
        "emission_kg_hr",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    summaries = []

    for group_id, group in frame.groupby(
        "possible_release_group_id",
        sort=True,
    ):
        summaries.append(
            summarize_group(
                group_id,
                group,
            )
        )

    summary = pd.DataFrame(summaries)

    summary = summary.sort_values(
        [
            "review_status",
            "group_row_count",
            "possible_release_group_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    status_summary = (
        summary["review_status"]
        .value_counts(dropna=False)
    )

    reason_summary = (
        summary["review_reasons"]
        .str.split(r"\s+\|\s+")
        .explode()
    )

    reason_summary = (
        reason_summary[
            reason_summary.notna()
            & reason_summary.ne("")
        ]
        .value_counts()
    )

    common_overlap_summary = pd.cut(
        summary["common_overlap_minutes"],
        bins=[
            -np.inf,
            0,
            5,
            15,
            60,
            np.inf,
        ],
        labels=[
            "no_common_overlap",
            "0_to_5_min",
            "5_to_15_min",
            "15_to_60_min",
            "above_60_min",
        ],
        right=False,
    ).value_counts().sort_index()

    manual = summary[
        summary["review_status"].eq(
            "manual_review_required"
        )
    ].copy()

    report_lines = [
        "=" * 115,
        "CONTROLLED-RELEASE DUPLICATE-GROUP REVIEW V1",
        "=" * 115,
        "",
        f"Possible duplicate groups: {len(summary)}",
        "",
        "Review status:",
        status_summary.to_string(),
        "",
        "Common temporal overlap:",
        common_overlap_summary.to_string(),
        "",
        "Review reasons:",
        (
            reason_summary.to_string()
            if not reason_summary.empty
            else "None"
        ),
        "",
        "Manual-review groups:",
        manual[
            [
                "possible_release_group_id",
                "group_row_count",
                "unique_source_dataset_count",
                "common_overlap_minutes",
                "start_spread_minutes",
                "end_spread_minutes",
                "maximum_distance_from_median_m",
                "emission_max_min_ratio",
                "review_reasons",
            ]
        ].to_string(index=False),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 115)
    print(
        "CONTROLLED-RELEASE DUPLICATE-GROUP SUMMARY"
    )
    print("=" * 115)

    print(
        "\nPossible duplicate groups:",
        len(summary),
    )

    print("\nReview status:")
    print(status_summary)

    print("\nCommon temporal overlap:")
    print(common_overlap_summary)

    print("\nReview reasons:")
    if reason_summary.empty:
        print("None")
    else:
        print(reason_summary)

    print("\nLargest groups:")
    print(
        summary[
            [
                "possible_release_group_id",
                "group_row_count",
                "unique_source_dataset_count",
                "common_overlap_minutes",
                "start_spread_minutes",
                "end_spread_minutes",
                "emission_max_min_ratio",
                "review_status",
                "review_reasons",
            ]
        ]
        .sort_values(
            "group_row_count",
            ascending=False,
        )
        .head(15)
        .to_string(
            index=False,
            max_colwidth=65,
        )
    )

    print("\nSaved:")
    print(SUMMARY_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
