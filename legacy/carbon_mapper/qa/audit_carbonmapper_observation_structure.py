from pathlib import Path
import re

import numpy as np
import pandas as pd


INPUT = Path(
    "raw_data/2023_Controlled_Release_2021/"
    "Dataframes for Stanford analysis/"
    "matchedDF_CarbonMapper_23822.csv"
)

COLUMN_OUTPUT = Path(
    "outputs/487_carbonmapper_all_columns_v1.csv"
)

TIMESTAMP_OUTPUT = Path(
    "outputs/488_carbonmapper_timestamp_groups_v1.csv"
)

DOCUMENT_OUTPUT = Path(
    "outputs/489_carbonmapper_companion_files_v1.txt"
)

REPORT_OUTPUT = Path(
    "outputs/490_carbonmapper_observation_structure_report_v1.txt"
)


def normalize(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def search_companion_files(root):
    relevant_name_patterns = [
        "carbonmapper",
        "carbon_mapper",
        "matchedDF",
        "readme",
        "codebook",
        "dictionary",
        "method",
        "metadata",
    ]

    candidates = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        lower_name = path.name.lower()

        name_match = any(
            pattern.lower() in lower_name
            for pattern in relevant_name_patterns
        )

        if name_match:
            candidates.append({
                "path": str(path),
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "match_reason": "filename",
            })

    return pd.DataFrame(candidates)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    frame = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    print("=" * 115)
    print("CARBON MAPPER OBSERVATION-STRUCTURE AUDIT")
    print("=" * 115)

    print("\nRows:", len(frame))
    print("Columns:", len(frame.columns))

    # --------------------------------------------------
    # 1. Complete column inventory
    # --------------------------------------------------
    column_records = []

    for position, column in enumerate(
        frame.columns,
        start=1,
    ):
        series = frame[column]

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        column_records.append({
            "column_number": position,
            "column": column,
            "normalized_column": normalize(column),
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "unique_non_null_count":
                int(series.nunique(dropna=True)),
            "numeric_count":
                int(numeric.notna().sum()),
            "example_values":
                " || ".join(
                    series.dropna()
                    .astype(str)
                    .drop_duplicates()
                    .head(5)
                    .tolist()
                ),
        })

    columns = pd.DataFrame(
        column_records
    )

    columns.to_csv(
        COLUMN_OUTPUT,
        index=False,
    )

    print("\nAll columns:")
    for record in column_records:
        print(
            f"{record['column_number']:03d}: "
            f"{record['column']}"
        )

    # --------------------------------------------------
    # 2. Timestamp-level grouping
    # --------------------------------------------------
    time_column = "datetime_UTC"

    if time_column not in frame.columns:
        raise KeyError(
            f"Missing {time_column}"
        )

    frame["_datetime_utc"] = pd.to_datetime(
        frame[time_column],
        errors="coerce",
        utc=True,
    )

    frame["Detection"] = pd.to_numeric(
        frame["Detection"],
        errors="coerce",
    )

    frame["FacilityEmissionRate"] = pd.to_numeric(
        frame["FacilityEmissionRate"],
        errors="coerce",
    )

    group_records = []

    for timestamp, group in frame.groupby(
        "_datetime_utc",
        dropna=False,
        sort=True,
    ):
        detection_values = sorted(
            group["Detection"]
            .dropna()
            .unique()
            .tolist()
        )

        wind_types = (
            group["WindType"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
            if "WindType" in group.columns
            else []
        )

        reported_rates = group[
            "FacilityEmissionRate"
        ].dropna()

        group_records.append({
            "datetime_utc":
                timestamp,

            "row_count":
                len(group),

            "detection_values":
                " | ".join(
                    str(value)
                    for value in detection_values
                ),

            "detection_1_count":
                int(
                    group["Detection"]
                    .eq(1).sum()
                ),

            "detection_0_count":
                int(
                    group["Detection"]
                    .eq(0).sum()
                ),

            "detection_minus1_count":
                int(
                    group["Detection"]
                    .eq(-1).sum()
                ),

            "wind_types":
                " | ".join(
                    sorted(wind_types)
                ),

            "wind_type_count":
                len(wind_types),

            "reported_rate_count":
                len(reported_rates),

            "reported_rate_min_kg_hr":
                (
                    reported_rates.min()
                    if not reported_rates.empty
                    else np.nan
                ),

            "reported_rate_median_kg_hr":
                (
                    reported_rates.median()
                    if not reported_rates.empty
                    else np.nan
                ),

            "reported_rate_max_kg_hr":
                (
                    reported_rates.max()
                    if not reported_rates.empty
                    else np.nan
                ),

            "plume_established_values":
                (
                    " | ".join(
                        sorted(
                            group["PlumeEstablished"]
                            .dropna()
                            .astype(str)
                            .unique()
                        )
                    )
                    if "PlumeEstablished"
                    in group.columns
                    else ""
                ),

            "plume_steady_values":
                (
                    " | ".join(
                        sorted(
                            group["PlumeSteady"]
                            .dropna()
                            .astype(str)
                            .unique()
                        )
                    )
                    if "PlumeSteady"
                    in group.columns
                    else ""
                ),
        })

    timestamp_groups = pd.DataFrame(
        group_records
    )

    timestamp_groups.to_csv(
        TIMESTAMP_OUTPUT,
        index=False,
    )

    # --------------------------------------------------
    # 3. Detection-level summaries
    # --------------------------------------------------
    detection_summary = (
        frame.groupby(
            "Detection",
            dropna=False,
        )
        .agg(
            rows=("Detection", "size"),

            unique_timestamps=(
                "_datetime_utc",
                "nunique",
            ),

            reported_rate_count=(
                "FacilityEmissionRate",
                "count",
            ),

            reported_rate_min_kg_hr=(
                "FacilityEmissionRate",
                "min",
            ),

            reported_rate_median_kg_hr=(
                "FacilityEmissionRate",
                "median",
            ),

            reported_rate_max_kg_hr=(
                "FacilityEmissionRate",
                "max",
            ),
        )
        .reset_index()
    )

    row_count_summary = (
        timestamp_groups["row_count"]
        .value_counts()
        .sort_index()
    )

    mixed_detection_groups = (
        timestamp_groups[
            timestamp_groups[
                "detection_values"
            ].str.contains(
                r"\|",
                regex=True,
                na=False,
            )
        ]
    )

    # --------------------------------------------------
    # 4. Companion files
    # --------------------------------------------------
    search_root = Path(
        "raw_data/2023_Controlled_Release_2021"
    )

    companion = search_companion_files(
        search_root
    )

    if companion.empty:
        DOCUMENT_OUTPUT.write_text(
            "No companion files found.",
            encoding="utf-8",
        )
    else:
        companion = companion.sort_values(
            [
                "match_reason",
                "path",
            ]
        )

        DOCUMENT_OUTPUT.write_text(
            companion.to_string(index=False),
            encoding="utf-8",
        )

    # --------------------------------------------------
    # 5. Report
    # --------------------------------------------------
    report_lines = [
        "=" * 115,
        "CARBON MAPPER OBSERVATION-STRUCTURE AUDIT V1",
        "=" * 115,
        "",
        f"Input rows: {len(frame)}",
        (
            "Unique datetime_UTC values: "
            f"{frame['_datetime_utc'].nunique()}"
        ),
        (
            "Timestamp groups with multiple rows: "
            f"{int(timestamp_groups['row_count'].gt(1).sum())}"
        ),
        (
            "Timestamp groups with mixed Detection values: "
            f"{len(mixed_detection_groups)}"
        ),
        "",
        "Rows per timestamp:",
        row_count_summary.to_string(),
        "",
        "Detection summary:",
        detection_summary.to_string(index=False),
        "",
        "Important:",
        (
            "Rows sharing the same datetime_UTC are not "
            "automatically independent observations."
        ),
        (
            "Detection=-1 must not be assigned a meaning "
            "until the project codebook or processing code "
            "has been identified."
        ),
        "",
        "Companion file candidates:",
        (
            companion.head(50).to_string(index=False)
            if not companion.empty
            else "None"
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nUnique datetime_UTC values:")
    print(
        frame["_datetime_utc"].nunique()
    )

    print(
        "Timestamp groups with multiple rows:",
        int(
            timestamp_groups[
                "row_count"
            ].gt(1).sum()
        ),
    )

    print(
        "Timestamp groups with mixed Detection values:",
        len(mixed_detection_groups),
    )

    print("\nRows per timestamp:")
    print(row_count_summary)

    print("\nDetection summary:")
    print(
        detection_summary.to_string(
            index=False
        )
    )

    print("\nCompanion file candidates:")
    if companion.empty:
        print("None")
    else:
        print(
            companion.head(30).to_string(
                index=False,
                max_colwidth=90,
            )
        )

    print("\nSaved:")
    print(COLUMN_OUTPUT)
    print(TIMESTAMP_OUTPUT)
    print(DOCUMENT_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
