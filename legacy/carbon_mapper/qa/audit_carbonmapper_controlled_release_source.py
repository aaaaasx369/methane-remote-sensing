from pathlib import Path
import re

import pandas as pd


EXPECTED_PATH = Path(
    "raw_data/2023_Controlled_Release_2021/"
    "Dataframes for Stanford analysis/"
    "matchedDF_CarbonMapper_23822.csv"
)

AUDIT_CSV = Path(
    "outputs/484_carbonmapper_controlled_release_column_audit_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/485_carbonmapper_controlled_release_raw_audit_v1.txt"
)

PREVIEW_OUTPUT = Path(
    "outputs/486_carbonmapper_controlled_release_raw_preview_v1.csv"
)


def normalize_name(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def find_columns(columns, patterns):
    results = []

    for column in columns:
        normalized = normalize_name(column)

        if any(
            re.search(pattern, normalized)
            for pattern in patterns
        ):
            results.append(column)

    return results


def locate_input():
    if EXPECTED_PATH.exists():
        return EXPECTED_PATH

    candidates = list(
        Path("raw_data").rglob(
            "matchedDF_CarbonMapper*.csv"
        )
    )

    candidates = [
        path
        for path in candidates
        if "superceded" not in str(path).lower()
    ]

    if not candidates:
        raise FileNotFoundError(
            "No non-superseded Carbon Mapper "
            "controlled-release CSV found."
        )

    candidates.sort(
        key=lambda path: (
            "23822" not in path.name,
            len(str(path)),
            str(path),
        )
    )

    return candidates[0]


def summarize_values(frame, columns, max_values=20):
    lines = []

    for column in columns:
        values = (
            frame[column]
            .dropna()
            .astype(str)
            .value_counts()
            .head(max_values)
        )

        lines.extend([
            "",
            f"{column}:",
            (
                values.to_string()
                if not values.empty
                else "No non-null values"
            ),
        ])

    return lines


def main():
    input_path = locate_input()

    frame = pd.read_csv(
        input_path,
        low_memory=False,
    )

    columns = list(frame.columns)

    groups = {
        "identifier": find_columns(
            columns,
            [
                r"event",
                r"release_id",
                r"plume_id",
                r"source_id",
                r"overpass",
                r"flight",
            ],
        ),

        "datetime": find_columns(
            columns,
            [
                r"date",
                r"time",
                r"utc",
                r"timestamp",
                r"overpass",
            ],
        ),

        "latitude": find_columns(
            columns,
            [
                r"^lat$",
                r"latitude",
                r"source_lat",
                r"plume_lat",
            ],
        ),

        "longitude": find_columns(
            columns,
            [
                r"^lon$",
                r"^lng$",
                r"longitude",
                r"source_lon",
                r"plume_lon",
            ],
        ),

        "emission": find_columns(
            columns,
            [
                r"emission",
                r"release_rate",
                r"flow_rate",
                r"kg_h",
                r"kg_hr",
                r"kgph",
                r"tph",
            ],
        ),

        "detection_label": find_columns(
            columns,
            [
                r"detect",
                r"plume",
                r"label",
                r"positive",
                r"true_release",
                r"reported",
            ],
        ),

        "instrument_platform": find_columns(
            columns,
            [
                r"instrument",
                r"sensor",
                r"platform",
                r"aircraft",
                r"satellite",
                r"provider",
                r"team",
            ],
        ),

        "wind": find_columns(
            columns,
            [
                r"wind",
            ],
        ),

        "uncertainty": find_columns(
            columns,
            [
                r"uncert",
                r"sigma",
                r"error",
                r"precision",
            ],
        ),
    }

    audit_records = []

    for column in columns:
        series = frame[column]

        audit_records.append({
            "column":
                column,

            "normalized_column":
                normalize_name(column),

            "dtype":
                str(series.dtype),

            "non_null_count":
                int(series.notna().sum()),

            "null_count":
                int(series.isna().sum()),

            "unique_non_null_count":
                int(series.nunique(dropna=True)),

            "example_values":
                " || ".join(
                    series.dropna()
                    .astype(str)
                    .drop_duplicates()
                    .head(5)
                    .tolist()
                ),
        })

    pd.DataFrame(
        audit_records
    ).to_csv(
        AUDIT_CSV,
        index=False,
    )

    preview_columns = []

    for group_name in [
        "identifier",
        "datetime",
        "latitude",
        "longitude",
        "emission",
        "detection_label",
        "instrument_platform",
        "wind",
        "uncertainty",
    ]:
        for column in groups[group_name]:
            if column not in preview_columns:
                preview_columns.append(column)

    if preview_columns:
        preview = frame[
            preview_columns
        ].head(30)
    else:
        preview = frame.head(30)

    preview.to_csv(
        PREVIEW_OUTPUT,
        index=False,
    )

    datetime_statistics = []

    for column in groups["datetime"]:
        parsed = pd.to_datetime(
            frame[column],
            errors="coerce",
            utc=True,
        )

        parse_count = int(
            parsed.notna().sum()
        )

        if parse_count:
            datetime_statistics.append({
                "column":
                    column,

                "parsed_count":
                    parse_count,

                "minimum":
                    parsed.min(),

                "maximum":
                    parsed.max(),
            })

    numeric_statistics = []

    for group_name in [
        "emission",
        "latitude",
        "longitude",
        "wind",
        "uncertainty",
    ]:
        for column in groups[group_name]:
            numeric = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            if numeric.notna().any():
                numeric_statistics.append({
                    "group":
                        group_name,

                    "column":
                        column,

                    "numeric_count":
                        int(numeric.notna().sum()),

                    "minimum":
                        float(numeric.min()),

                    "median":
                        float(numeric.median()),

                    "maximum":
                        float(numeric.max()),
                })

    report_lines = [
        "=" * 115,
        "CARBON MAPPER CONTROLLED-RELEASE RAW DATA AUDIT V1",
        "=" * 115,
        "",
        f"Input path: {input_path}",
        f"Rows: {len(frame)}",
        f"Columns: {len(columns)}",
        "",
        "Candidate column groups:",
    ]

    for group_name, group_columns in groups.items():
        report_lines.append(
            f"{group_name}: "
            + (
                " | ".join(group_columns)
                if group_columns
                else "NONE"
            )
        )

    report_lines.extend([
        "",
        "Datetime statistics:",
        (
            pd.DataFrame(
                datetime_statistics
            ).to_string(index=False)
            if datetime_statistics
            else "No parseable datetime columns."
        ),
        "",
        "Numeric statistics:",
        (
            pd.DataFrame(
                numeric_statistics
            ).to_string(index=False)
            if numeric_statistics
            else "No relevant numeric columns."
        ),
        "",
        "Instrument/platform/provider values:",
    ])

    report_lines.extend(
        summarize_values(
            frame,
            groups[
                "instrument_platform"
            ],
        )
    )

    report_lines.extend([
        "",
        "Detection/label values:",
    ])

    report_lines.extend(
        summarize_values(
            frame,
            groups["detection_label"],
        )
    )

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 115)
    print(
        "CARBON MAPPER CONTROLLED-RELEASE RAW DATA AUDIT"
    )
    print("=" * 115)

    print("\nInput path:")
    print(input_path)

    print("\nRows:", len(frame))
    print("Columns:", len(columns))

    print("\nCandidate column groups:")
    for group_name, group_columns in groups.items():
        print(
            f"{group_name}:",
            (
                " | ".join(group_columns)
                if group_columns
                else "NONE"
            ),
        )

    print("\nDatetime statistics:")
    if datetime_statistics:
        print(
            pd.DataFrame(
                datetime_statistics
            ).to_string(index=False)
        )
    else:
        print(
            "No parseable datetime columns."
        )

    print("\nNumeric statistics:")
    if numeric_statistics:
        print(
            pd.DataFrame(
                numeric_statistics
            ).to_string(index=False)
        )
    else:
        print(
            "No relevant numeric columns."
        )

    print("\nInstrument/platform/provider values:")
    instrument_lines = summarize_values(
        frame,
        groups["instrument_platform"],
    )

    if instrument_lines:
        print("\n".join(instrument_lines))
    else:
        print("No explicit instrument/platform columns.")

    print("\nDetection/label values:")
    detection_lines = summarize_values(
        frame,
        groups["detection_label"],
    )

    if detection_lines:
        print("\n".join(detection_lines))
    else:
        print("No explicit detection/label columns.")

    print("\nFirst 10 selected rows:")
    print(
        preview.head(10).to_string(
            index=False,
            max_colwidth=60,
        )
    )

    print("\nSaved:")
    print(AUDIT_CSV)
    print(REPORT_OUTPUT)
    print(PREVIEW_OUTPUT)


if __name__ == "__main__":
    main()
