from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/309_all_exact_release_intervals_for_s2.csv"
)

MASTER_OUTPUT = Path(
    "outputs/465_controlled_release_master_ground_truth_v1.csv"
)

EXCLUDED_OUTPUT = Path(
    "outputs/466_controlled_release_master_excluded_rows_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/467_controlled_release_master_ground_truth_report_v1.txt"
)


def make_interval_id(row):
    raw = "|".join([
        str(row["source_dataset"]),
        str(row["release_start_utc"]),
        str(row["release_end_utc"]),
        f"{row['latitude']:.7f}",
        f"{row['longitude']:.7f}",
        f"{row['emission_kg_hr']:.6f}",
    ])

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:12]

    return f"CR_INTERVAL_{digest}"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    frame = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    print("=" * 110)
    print("CONTROLLED-RELEASE MASTER GROUND TRUTH")
    print("=" * 110)

    print("\nInput rows:", len(frame))

    required = [
        "release_start_utc",
        "release_end_utc",
        "lat",
        "lon",
        "release_rate_kg_h",
        "source_file",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    normalized = pd.DataFrame({
        "release_start_utc":
            pd.to_datetime(
                frame["release_start_utc"],
                errors="coerce",
                utc=True,
            ),

        "release_end_utc":
            pd.to_datetime(
                frame["release_end_utc"],
                errors="coerce",
                utc=True,
            ),

        "latitude":
            pd.to_numeric(
                frame["lat"],
                errors="coerce",
            ),

        "longitude":
            pd.to_numeric(
                frame["lon"],
                errors="coerce",
            ),

        "emission_kg_hr":
            pd.to_numeric(
                frame["release_rate_kg_h"],
                errors="coerce",
            ),

        "source_dataset":
            frame["source_file"]
            .fillna("unknown")
            .astype(str),

        "release_rate_source":
            (
                frame["release_rate_source"]
                if "release_rate_source" in frame.columns
                else ""
            ),

        "original_emission_bin":
            (
                frame["emission_bin"]
                if "emission_bin" in frame.columns
                else ""
            ),
    })

    # Preserve useful original identifiers when they exist.
    for column in [
        "event_id",
        "site",
        "site_name",
        "location",
        "campaign",
        "experiment",
        "release_id",
        "source_id",
    ]:
        if column in frame.columns:
            normalized[
                f"original_{column}"
            ] = frame[column]

    normalized[
        "release_duration_minutes"
    ] = (
        normalized["release_end_utc"]
        - normalized["release_start_utc"]
    ).dt.total_seconds() / 60.0

    normalized[
        "time_valid"
    ] = (
        normalized["release_start_utc"].notna()
        & normalized["release_end_utc"].notna()
        & normalized[
            "release_duration_minutes"
        ].gt(0)
    )

    normalized[
        "location_valid"
    ] = (
        normalized["latitude"].between(
            -90,
            90,
            inclusive="both",
        )
        & normalized["longitude"].between(
            -180,
            180,
            inclusive="both",
        )
    )

    normalized[
        "emission_valid"
    ] = normalized[
        "emission_kg_hr"
    ].gt(0)

    normalized[
        "ground_truth_valid"
    ] = (
        normalized["time_valid"]
        & normalized["location_valid"]
        & normalized["emission_valid"]
    )

    normalized[
        "emission_class"
    ] = pd.cut(
        normalized["emission_kg_hr"],
        bins=[
            0,
            200,
            500,
            1000,
            5000,
            np.inf,
        ],
        labels=[
            "0_to_200",
            "200_to_500",
            "500_to_1000",
            "1000_to_5000",
            "above_5000",
        ],
        right=False,
    )

    normalized[
        "ground_truth_type"
    ] = "controlled_release_exact_interval"

    normalized[
        "positive_definition"
    ] = (
        "satellite_acquisition_inside_release_interval"
    )

    normalized[
        "negative_definition"
    ] = (
        "same_location_acquisition_outside_all_known_release_intervals"
    )

    normalized[
        "ground_truth_label"
    ] = 1

    valid = normalized[
        normalized["ground_truth_valid"]
    ].copy()

    excluded = normalized[
        ~normalized["ground_truth_valid"]
    ].copy()

    valid[
        "interval_id"
    ] = valid.apply(
        make_interval_id,
        axis=1,
    )

    duplicate_columns = [
        "release_start_utc",
        "release_end_utc",
        "latitude",
        "longitude",
        "emission_kg_hr",
        "source_dataset",
    ]

    valid[
        "exact_duplicate"
    ] = valid.duplicated(
        subset=duplicate_columns,
        keep=False,
    )

    valid[
        "exact_duplicate_group_size"
    ] = valid.groupby(
        duplicate_columns,
        dropna=False,
    )["source_dataset"].transform("size")

    valid = valid.sort_values(
        [
            "release_start_utc",
            "latitude",
            "longitude",
            "emission_kg_hr",
            "source_dataset",
        ]
    ).reset_index(drop=True)

    valid[
        "master_row_number"
    ] = valid.index + 1

    preferred_columns = [
        "master_row_number",
        "interval_id",
        "release_start_utc",
        "release_end_utc",
        "release_duration_minutes",
        "latitude",
        "longitude",
        "emission_kg_hr",
        "emission_class",
        "source_dataset",
        "release_rate_source",
        "ground_truth_type",
        "ground_truth_label",
        "positive_definition",
        "negative_definition",
        "exact_duplicate",
        "exact_duplicate_group_size",
        "time_valid",
        "location_valid",
        "emission_valid",
        "ground_truth_valid",
    ]

    remaining_columns = [
        column
        for column in valid.columns
        if column not in preferred_columns
    ]

    valid[
        preferred_columns + remaining_columns
    ].to_csv(
        MASTER_OUTPUT,
        index=False,
    )

    excluded.to_csv(
        EXCLUDED_OUTPUT,
        index=False,
    )

    source_summary = (
        valid["source_dataset"]
        .value_counts(dropna=False)
    )

    emission_summary = (
        valid["emission_class"]
        .value_counts(dropna=False)
        .reindex(
            [
                "0_to_200",
                "200_to_500",
                "500_to_1000",
                "1000_to_5000",
                "above_5000",
            ],
            fill_value=0,
        )
    )

    duplicate_rows = int(
        valid["exact_duplicate"].sum()
    )

    duplicate_groups = int(
        valid.loc[
            valid["exact_duplicate"],
            duplicate_columns,
        ].drop_duplicates().shape[0]
    )

    report_lines = [
        "=" * 110,
        "CONTROLLED-RELEASE MASTER GROUND TRUTH V1",
        "=" * 110,
        "",
        f"Input rows: {len(frame)}",
        f"Valid exact release intervals: {len(valid)}",
        f"Excluded invalid rows: {len(excluded)}",
        f"Exact duplicate rows: {duplicate_rows}",
        f"Exact duplicate groups: {duplicate_groups}",
        "",
        "Release-time range:",
        (
            f"{valid['release_start_utc'].min()} "
            f"to {valid['release_end_utc'].max()}"
        ),
        "",
        "Emission-rate statistics in kg/h:",
        valid[
            "emission_kg_hr"
        ].describe().to_string(),
        "",
        "Emission classes:",
        emission_summary.to_string(),
        "",
        "Source datasets:",
        source_summary.to_string(),
        "",
        "Duration statistics in minutes:",
        valid[
            "release_duration_minutes"
        ].describe().to_string(),
        "",
        "Positive rule:",
        (
            "A satellite acquisition is positive only when "
            "its acquisition time falls inside the exact "
            "controlled-release interval and its valid footprint "
            "contains the release location."
        ),
        "",
        "Negative rule:",
        (
            "A negative must be acquired at the same location "
            "outside every known release interval and pass "
            "sensor-specific QA."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nValid exact release intervals:", len(valid))
    print("Excluded invalid rows:", len(excluded))
    print("Exact duplicate rows:", duplicate_rows)
    print("Exact duplicate groups:", duplicate_groups)

    print("\nRelease-time range:")
    print(
        valid["release_start_utc"].min(),
        "to",
        valid["release_end_utc"].max(),
    )

    print("\nEmission classes:")
    print(emission_summary)

    print("\nSource datasets:")
    print(source_summary)

    print("\nSaved:")
    print(MASTER_OUTPUT)
    print(EXCLUDED_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
