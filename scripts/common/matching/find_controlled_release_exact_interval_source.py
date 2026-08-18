from pathlib import Path
import re

import pandas as pd


SEARCH_DIRECTORIES = [
    Path("outputs"),
    Path("data"),
    Path("metadata"),
    Path("ground_truth"),
]

OUTPUT_CSV = Path(
    "outputs/462_controlled_release_exact_interval_source_candidates_v1.csv"
)

OUTPUT_REPORT = Path(
    "outputs/463_controlled_release_exact_interval_source_report_v1.txt"
)


def normalize_name(name):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(name).strip().lower(),
    ).strip("_")


def find_matching_columns(columns, patterns):
    matches = []

    for original in columns:
        normalized = normalize_name(original)

        if any(
            re.search(pattern, normalized)
            for pattern in patterns
        ):
            matches.append(original)

    return matches


def approximate_row_count(path):
    try:
        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as handle:
            count = sum(1 for _ in handle)

        return max(0, count - 1)

    except Exception:
        return None


def inspect_csv(path):
    try:
        sample = pd.read_csv(
            path,
            nrows=5,
            low_memory=False,
        )
    except Exception as error:
        return {
            "path": str(path),
            "read_success": False,
            "read_error": str(error),
            "score": -1,
        }

    columns = list(sample.columns)
    normalized_columns = {
        column: normalize_name(column)
        for column in columns
    }

    event_columns = find_matching_columns(
        columns,
        [
            r"^event_id$",
            r"release_event_id",
            r"controlled_release_id",
            r"experiment_id",
            r"plume_id",
        ],
    )

    latitude_columns = find_matching_columns(
        columns,
        [
            r"^lat$",
            r"latitude",
            r"source_lat",
            r"release_lat",
        ],
    )

    longitude_columns = find_matching_columns(
        columns,
        [
            r"^lon$",
            r"^lng$",
            r"longitude",
            r"source_lon",
            r"release_lon",
        ],
    )

    exact_start_columns = find_matching_columns(
        columns,
        [
            r"release_start",
            r"release_begin",
            r"start_release",
            r"experiment_start",
            r"controlled_release_start",
            r"interval_start",
            r"time_coverage_start",
            r"datetime_start",
            r"start_datetime",
            r"start_time_utc",
        ],
    )

    exact_end_columns = find_matching_columns(
        columns,
        [
            r"release_end",
            r"release_stop",
            r"end_release",
            r"experiment_end",
            r"controlled_release_end",
            r"interval_end",
            r"time_coverage_end",
            r"datetime_end",
            r"end_datetime",
            r"end_time_utc",
        ],
    )

    search_start_columns = find_matching_columns(
        columns,
        [
            r"search_start",
            r"window_start",
            r"query_start",
        ],
    )

    search_end_columns = find_matching_columns(
        columns,
        [
            r"search_end",
            r"window_end",
            r"query_end",
        ],
    )

    generic_datetime_columns = find_matching_columns(
        columns,
        [
            r"^datetime_utc$",
            r"acquisition_datetime",
            r"release_datetime",
            r"event_datetime",
            r"timestamp",
            r"utc_time",
        ],
    )

    emission_columns = find_matching_columns(
        columns,
        [
            r"emission",
            r"release_rate",
            r"flow_rate",
            r"mass_flow",
            r"kg_h",
            r"kg_hr",
            r"kg_per_h",
            r"tph",
        ],
    )

    label_columns = find_matching_columns(
        columns,
        [
            r"^label$",
            r"true_release",
            r"release_label",
            r"ground_truth",
            r"plume_present",
        ],
    )

    source_columns = find_matching_columns(
        columns,
        [
            r"source_dataset",
            r"source_file",
            r"study",
            r"experiment",
            r"campaign",
        ],
    )

    exact_time_pair = bool(
        exact_start_columns
        and exact_end_columns
    )

    search_window_pair = bool(
        search_start_columns
        and search_end_columns
    )

    location_pair = bool(
        latitude_columns
        and longitude_columns
    )

    score = 0

    if exact_time_pair:
        score += 45

    if location_pair:
        score += 25

    if emission_columns:
        score += 12

    if event_columns:
        score += 7

    if label_columns:
        score += 5

    if source_columns:
        score += 3

    if generic_datetime_columns:
        score += 4

    # Search windows are useful, but they are not proof of
    # an exact controlled-release interval.
    if search_window_pair:
        score += 2

    path_text = str(path).lower()

    if "release" in path_text:
        score += 4

    if "interval" in path_text:
        score += 4

    if "ground" in path_text:
        score += 3

    if "availability" in path_text:
        score += 1

    row_count = approximate_row_count(path)

    return {
        "path":
            str(path),

        "read_success":
            True,

        "read_error":
            "",

        "approximate_rows":
            row_count,

        "column_count":
            len(columns),

        "score":
            score,

        "has_exact_time_pair":
            exact_time_pair,

        "has_search_window_pair":
            search_window_pair,

        "has_location_pair":
            location_pair,

        "event_columns":
            " | ".join(event_columns),

        "exact_start_columns":
            " | ".join(exact_start_columns),

        "exact_end_columns":
            " | ".join(exact_end_columns),

        "search_start_columns":
            " | ".join(search_start_columns),

        "search_end_columns":
            " | ".join(search_end_columns),

        "generic_datetime_columns":
            " | ".join(generic_datetime_columns),

        "latitude_columns":
            " | ".join(latitude_columns),

        "longitude_columns":
            " | ".join(longitude_columns),

        "emission_columns":
            " | ".join(emission_columns),

        "label_columns":
            " | ".join(label_columns),

        "source_columns":
            " | ".join(source_columns),

        "all_columns":
            " | ".join(columns),
    }


def main():
    paths = set()

    for directory in SEARCH_DIRECTORIES:
        if not directory.exists():
            continue

        for path in directory.rglob("*.csv"):
            paths.add(path)

    # Also inspect CSV files stored directly in the project root.
    for path in Path(".").glob("*.csv"):
        paths.add(path)

    print("=" * 110)
    print("CONTROLLED-RELEASE EXACT-INTERVAL SOURCE AUDIT")
    print("=" * 110)

    print("\nCSV files found:", len(paths))

    records = []

    for number, path in enumerate(
        sorted(paths),
        start=1,
    ):
        result = inspect_csv(path)
        records.append(result)

        if number % 25 == 0:
            print(
                f"Inspected {number}/{len(paths)} files..."
            )

    audit = pd.DataFrame(records)

    if audit.empty:
        raise RuntimeError(
            "No CSV files were found."
        )

    audit = audit.sort_values(
        [
            "score",
            "has_exact_time_pair",
            "approximate_rows",
            "path",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)

    audit["candidate_rank"] = (
        audit.index + 1
    )

    output_columns = [
        "candidate_rank",
        "path",
        "approximate_rows",
        "score",
        "has_exact_time_pair",
        "has_search_window_pair",
        "has_location_pair",
        "event_columns",
        "exact_start_columns",
        "exact_end_columns",
        "search_start_columns",
        "search_end_columns",
        "generic_datetime_columns",
        "latitude_columns",
        "longitude_columns",
        "emission_columns",
        "label_columns",
        "source_columns",
        "column_count",
        "all_columns",
        "read_success",
        "read_error",
    ]

    audit[output_columns].to_csv(
        OUTPUT_CSV,
        index=False,
    )

    successful = audit[
        audit["read_success"].eq(True)
    ].copy()

    exact_candidates = successful[
        successful[
            "has_exact_time_pair"
        ].eq(True)
        & successful[
            "has_location_pair"
        ].eq(True)
    ].copy()

    report_lines = [
        "=" * 110,
        "CONTROLLED-RELEASE EXACT-INTERVAL SOURCE AUDIT V1",
        "=" * 110,
        "",
        f"CSV files inspected: {len(audit)}",
        (
            "Files with exact start/end and location: "
            f"{len(exact_candidates)}"
        ),
        "",
        "Top 15 candidates:",
        successful[
            [
                "candidate_rank",
                "path",
                "approximate_rows",
                "score",
                "has_exact_time_pair",
                "has_search_window_pair",
                "event_columns",
                "exact_start_columns",
                "exact_end_columns",
                "latitude_columns",
                "longitude_columns",
                "emission_columns",
            ]
        ].head(15).to_string(index=False),
        "",
        (
            "Important: search_start/search_end are only "
            "satellite-query windows unless separately proven "
            "to be the true release interval."
        ),
    ]

    OUTPUT_REPORT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print("TOP CANDIDATES")
    print("=" * 110)

    display_columns = [
        "candidate_rank",
        "path",
        "approximate_rows",
        "score",
        "has_exact_time_pair",
        "has_search_window_pair",
        "event_columns",
        "exact_start_columns",
        "exact_end_columns",
        "latitude_columns",
        "longitude_columns",
        "emission_columns",
    ]

    print(
        successful[
            display_columns
        ].head(15).to_string(
            index=False,
            max_colwidth=55,
        )
    )

    print("\nFiles with exact time pair and location:")
    print(len(exact_candidates))

    print("\nSaved:")
    print(OUTPUT_CSV)
    print(OUTPUT_REPORT)


if __name__ == "__main__":
    main()
