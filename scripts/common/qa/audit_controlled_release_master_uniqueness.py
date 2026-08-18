from pathlib import Path
from math import radians, sin, cos, asin, sqrt

import pandas as pd


VALID_INPUT = Path(
    "outputs/465_controlled_release_master_ground_truth_v1.csv"
)

EXCLUDED_INPUT = Path(
    "outputs/466_controlled_release_master_excluded_rows_v1.csv"
)

PAIR_OUTPUT = Path(
    "outputs/468_controlled_release_possible_duplicate_pairs_v1.csv"
)

GROUP_OUTPUT = Path(
    "outputs/469_controlled_release_possible_duplicate_groups_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/470_controlled_release_uniqueness_audit_report_v1.txt"
)


# This is only an audit threshold.
# We are not automatically deleting or merging any records.
MAX_DISTANCE_M = 100.0
MAX_START_DIFFERENCE_MIN = 5.0
MAX_END_DIFFERENCE_MIN = 5.0


def to_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def haversine_m(lat1, lon1, lat2, lon2):
    earth_radius_m = 6371000.0

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    value = (
        sin(delta_lat / 2.0) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(delta_lon / 2.0) ** 2
    )

    return (
        2.0
        * earth_radius_m
        * asin(sqrt(value))
    )


class UnionFind:
    def __init__(self, values):
        self.parent = {
            value: value
            for value in values
        }

    def find(self, value):
        if self.parent[value] != value:
            self.parent[value] = self.find(
                self.parent[value]
            )

        return self.parent[value]

    def union(self, first, second):
        root_first = self.find(first)
        root_second = self.find(second)

        if root_first != root_second:
            self.parent[root_second] = root_first


def classify_exclusion_reason(row):
    reasons = []

    if not bool(row["time_valid"]):
        reasons.append("invalid_or_missing_time")

    if not bool(row["location_valid"]):
        reasons.append("invalid_or_missing_location")

    if not bool(row["emission_valid"]):
        reasons.append(
            "missing_zero_or_negative_emission"
        )

    if not reasons:
        reasons.append("other")

    return " + ".join(reasons)


def intervals_overlap(
    first_start,
    first_end,
    second_start,
    second_end,
):
    return (
        first_start <= second_end
        and second_start <= first_end
    )


def main():
    valid = pd.read_csv(
        VALID_INPUT,
        low_memory=False,
    )

    excluded = pd.read_csv(
        EXCLUDED_INPUT,
        low_memory=False,
    )

    for frame in [valid, excluded]:
        for column in [
            "time_valid",
            "location_valid",
            "emission_valid",
            "ground_truth_valid",
        ]:
            if column in frame.columns:
                frame[column] = to_bool(
                    frame[column]
                )

    valid["release_start_utc"] = pd.to_datetime(
        valid["release_start_utc"],
        errors="coerce",
        utc=True,
    )

    valid["release_end_utc"] = pd.to_datetime(
        valid["release_end_utc"],
        errors="coerce",
        utc=True,
    )

    valid["latitude"] = pd.to_numeric(
        valid["latitude"],
        errors="coerce",
    )

    valid["longitude"] = pd.to_numeric(
        valid["longitude"],
        errors="coerce",
    )

    valid["emission_kg_hr"] = pd.to_numeric(
        valid["emission_kg_hr"],
        errors="coerce",
    )

    excluded["exclusion_reason"] = excluded.apply(
        classify_exclusion_reason,
        axis=1,
    )

    exclusion_summary = (
        excluded["exclusion_reason"]
        .value_counts(dropna=False)
    )

    valid["release_date_utc"] = (
        valid["release_start_utc"]
        .dt.date
        .astype(str)
    )

    indices = list(valid.index)
    union_find = UnionFind(indices)

    pair_records = []

    # Only compare records occurring on the same UTC date.
    for release_date, group in valid.groupby(
        "release_date_utc",
        sort=True,
    ):
        group_indices = list(group.index)

        for first_position in range(
            len(group_indices)
        ):
            for second_position in range(
                first_position + 1,
                len(group_indices),
            ):
                first_index = group_indices[
                    first_position
                ]

                second_index = group_indices[
                    second_position
                ]

                first = valid.loc[first_index]
                second = valid.loc[second_index]

                distance_m = haversine_m(
                    first["latitude"],
                    first["longitude"],
                    second["latitude"],
                    second["longitude"],
                )

                if distance_m > MAX_DISTANCE_M:
                    continue

                start_difference_min = abs(
                    (
                        first["release_start_utc"]
                        - second["release_start_utc"]
                    ).total_seconds()
                ) / 60.0

                end_difference_min = abs(
                    (
                        first["release_end_utc"]
                        - second["release_end_utc"]
                    ).total_seconds()
                ) / 60.0

                overlap = intervals_overlap(
                    first["release_start_utc"],
                    first["release_end_utc"],
                    second["release_start_utc"],
                    second["release_end_utc"],
                )

                near_same_interval = (
                    start_difference_min
                    <= MAX_START_DIFFERENCE_MIN
                    and end_difference_min
                    <= MAX_END_DIFFERENCE_MIN
                )

                possible_same_release = (
                    overlap or near_same_interval
                )

                if not possible_same_release:
                    continue

                union_find.union(
                    first_index,
                    second_index,
                )

                rate_1 = first["emission_kg_hr"]
                rate_2 = second["emission_kg_hr"]

                if rate_1 > 0 and rate_2 > 0:
                    rate_ratio = max(
                        rate_1,
                        rate_2,
                    ) / min(
                        rate_1,
                        rate_2,
                    )
                else:
                    rate_ratio = None

                pair_records.append({
                    "release_date_utc":
                        release_date,

                    "interval_id_1":
                        first["interval_id"],

                    "interval_id_2":
                        second["interval_id"],

                    "source_dataset_1":
                        first["source_dataset"],

                    "source_dataset_2":
                        second["source_dataset"],

                    "release_start_utc_1":
                        first["release_start_utc"],

                    "release_start_utc_2":
                        second["release_start_utc"],

                    "release_end_utc_1":
                        first["release_end_utc"],

                    "release_end_utc_2":
                        second["release_end_utc"],

                    "latitude_1":
                        first["latitude"],

                    "longitude_1":
                        first["longitude"],

                    "latitude_2":
                        second["latitude"],

                    "longitude_2":
                        second["longitude"],

                    "distance_m":
                        distance_m,

                    "start_difference_minutes":
                        start_difference_min,

                    "end_difference_minutes":
                        end_difference_min,

                    "intervals_overlap":
                        overlap,

                    "emission_kg_hr_1":
                        rate_1,

                    "emission_kg_hr_2":
                        rate_2,

                    "emission_rate_ratio":
                        rate_ratio,

                    "possible_same_release":
                        True,
                })

    pairs = pd.DataFrame(pair_records)

    if pairs.empty:
        pairs = pd.DataFrame(columns=[
            "release_date_utc",
            "interval_id_1",
            "interval_id_2",
            "source_dataset_1",
            "source_dataset_2",
            "release_start_utc_1",
            "release_start_utc_2",
            "release_end_utc_1",
            "release_end_utc_2",
            "latitude_1",
            "longitude_1",
            "latitude_2",
            "longitude_2",
            "distance_m",
            "start_difference_minutes",
            "end_difference_minutes",
            "intervals_overlap",
            "emission_kg_hr_1",
            "emission_kg_hr_2",
            "emission_rate_ratio",
            "possible_same_release",
        ])

    pairs.to_csv(
        PAIR_OUTPUT,
        index=False,
    )

    valid[
        "possible_release_group_root"
    ] = valid.index.map(
        union_find.find
    )

    grouped_counts = (
        valid.groupby(
            "possible_release_group_root"
        )
        .size()
    )

    duplicate_roots = grouped_counts[
        grouped_counts > 1
    ].index

    possible_duplicates = valid[
        valid[
            "possible_release_group_root"
        ].isin(duplicate_roots)
    ].copy()

    root_to_group_id = {
        root: (
            f"POSSIBLE_RELEASE_GROUP_"
            f"{number:04d}"
        )
        for number, root in enumerate(
            sorted(duplicate_roots),
            start=1,
        )
    }

    possible_duplicates[
        "possible_release_group_id"
    ] = possible_duplicates[
        "possible_release_group_root"
    ].map(root_to_group_id)

    possible_duplicates[
        "possible_release_group_size"
    ] = possible_duplicates[
        "possible_release_group_root"
    ].map(grouped_counts)

    possible_duplicates[
        "group_source_count"
    ] = possible_duplicates.groupby(
        "possible_release_group_id"
    )["source_dataset"].transform("nunique")

    possible_duplicates = (
        possible_duplicates.sort_values(
            [
                "possible_release_group_id",
                "release_start_utc",
                "source_dataset",
            ]
        )
    )

    possible_duplicates.to_csv(
        GROUP_OUTPUT,
        index=False,
    )

    possible_group_count = (
        possible_duplicates[
            "possible_release_group_id"
        ].nunique()
        if not possible_duplicates.empty
        else 0
    )

    possible_duplicate_row_count = len(
        possible_duplicates
    )

    singleton_row_count = (
        len(valid)
        - possible_duplicate_row_count
    )

    estimated_units_before_review = (
        singleton_row_count
        + possible_group_count
    )

    source_summary = (
        valid["source_dataset"]
        .value_counts(dropna=False)
    )

    group_size_summary = (
        possible_duplicates[
            "possible_release_group_size"
        ].value_counts().sort_index()
        if not possible_duplicates.empty
        else pd.Series(dtype="int64")
    )

    report_lines = [
        "=" * 115,
        "CONTROLLED-RELEASE MASTER UNIQUENESS AUDIT V1",
        "=" * 115,
        "",
        f"Valid interval rows: {len(valid)}",
        f"Excluded rows: {len(excluded)}",
        "",
        "Excluded-row reasons:",
        exclusion_summary.to_string(),
        "",
        (
            "Important: possible duplicate groups are "
            "review candidates only. They have not been merged."
        ),
        "",
        (
            "Possible same-release pair rows: "
            f"{len(pairs)}"
        ),
        (
            "Possible same-release groups: "
            f"{possible_group_count}"
        ),
        (
            "Rows inside possible duplicate groups: "
            f"{possible_duplicate_row_count}"
        ),
        (
            "Rows not in possible duplicate groups: "
            f"{singleton_row_count}"
        ),
        (
            "Estimated independent release units before "
            "manual review: "
            f"{estimated_units_before_review}"
        ),
        "",
        "Possible group-size distribution:",
        (
            group_size_summary.to_string()
            if not group_size_summary.empty
            else "No possible duplicate groups."
        ),
        "",
        "Valid rows by source dataset:",
        source_summary.to_string(),
        "",
        "Audit thresholds:",
        f"Maximum distance: {MAX_DISTANCE_M} m",
        (
            "Maximum start-time difference: "
            f"{MAX_START_DIFFERENCE_MIN} min"
        ),
        (
            "Maximum end-time difference: "
            f"{MAX_END_DIFFERENCE_MIN} min"
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 115)
    print(
        "CONTROLLED-RELEASE MASTER UNIQUENESS AUDIT"
    )
    print("=" * 115)

    print("\nValid interval rows:", len(valid))
    print("Excluded rows:", len(excluded))

    print("\nExcluded-row reasons:")
    print(exclusion_summary)

    print(
        "\nPossible same-release pair rows:",
        len(pairs),
    )

    print(
        "Possible same-release groups:",
        possible_group_count,
    )

    print(
        "Rows inside possible duplicate groups:",
        possible_duplicate_row_count,
    )

    print(
        "Rows not in possible duplicate groups:",
        singleton_row_count,
    )

    print(
        "Estimated independent release units "
        "before manual review:",
        estimated_units_before_review,
    )

    print("\nPossible group-size distribution:")
    if group_size_summary.empty:
        print("No possible duplicate groups.")
    else:
        print(group_size_summary)

    print("\nSaved:")
    print(PAIR_OUTPUT)
    print(GROUP_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
