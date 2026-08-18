from pathlib import Path
import re

import pandas as pd


INPUT = Path(
    "raw_data/2023_Controlled_Release_2021/"
    "Dataframes for Stanford analysis/"
    "matchedDF_CarbonMapper_23822.csv"
)

CODE_PATH = Path(
    "raw_data/2023_Controlled_Release_2021/"
    "AnalysisCode/matchMethods.py"
)

TEST_DATA_DIR = Path(
    "raw_data/2023_Controlled_Release_2021/"
    "CarbonMapperTestData"
)

COLUMN_OUTPUT = Path(
    "outputs/491_carbonmapper_identifier_column_audit_v1.csv"
)

TEST_FILE_OUTPUT = Path(
    "outputs/492_carbonmapper_test_data_file_audit_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/493_carbonmapper_detection_definition_report_v1.txt"
)


def normalize(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def code_hits(path, patterns, context=5):
    if not path.exists():
        return [
            f"Code file not found: {path}"
        ]

    lines = path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    matched_ranges = []

    for number, line in enumerate(lines):
        if any(
            re.search(
                pattern,
                line,
                flags=re.IGNORECASE,
            )
            for pattern in patterns
        ):
            start = max(0, number - context)
            end = min(
                len(lines),
                number + context + 1,
            )

            matched_ranges.append(
                (start, end)
            )

    merged = []

    for start, end in sorted(matched_ranges):
        if (
            merged
            and start <= merged[-1][1]
        ):
            merged[-1] = (
                merged[-1][0],
                max(end, merged[-1][1]),
            )
        else:
            merged.append((start, end))

    output = []

    for start, end in merged:
        output.append("-" * 100)

        for number in range(start, end):
            output.append(
                f"{number + 1:05d}: "
                f"{lines[number]}"
            )

    if not output:
        output.append(
            "No matching definitions found."
        )

    return output


def inspect_test_files(directory):
    records = []

    if not directory.exists():
        return pd.DataFrame()

    for path in sorted(
        directory.rglob("*.csv")
    ):
        try:
            frame = pd.read_csv(
                path,
                low_memory=False,
            )

            detection_column = next(
                (
                    column
                    for column in frame.columns
                    if normalize(column)
                    == "detection"
                ),
                None,
            )

            if detection_column is None:
                detection_values = ""
            else:
                detection_values = (
                    frame[detection_column]
                    .value_counts(dropna=False)
                    .to_dict()
                )

            relevant_columns = [
                column
                for column in frame.columns
                if re.search(
                    r"detect|class|facility|"
                    r"emission|release|flow|"
                    r"timestamp|survey|plume|"
                    r"source|id|idx",
                    normalize(column),
                )
            ]

            records.append({
                "path":
                    str(path),

                "rows":
                    len(frame),

                "columns":
                    len(frame.columns),

                "detection_column":
                    detection_column or "",

                "detection_values":
                    str(detection_values),

                "relevant_columns":
                    " | ".join(
                        relevant_columns
                    ),
            })

        except Exception as error:
            records.append({
                "path":
                    str(path),

                "rows":
                    None,

                "columns":
                    None,

                "detection_column":
                    "",

                "detection_values":
                    "",

                "relevant_columns":
                    "",

                "error":
                    str(error),
            })

    return pd.DataFrame(records)


def main():
    frame = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    print("=" * 115)
    print(
        "CARBON MAPPER DETECTION-DEFINITION AUDIT"
    )
    print("=" * 115)

    relevant_columns = [
        column
        for column in frame.columns
        if re.search(
            r"(^id$|_id$|idx|index|"
            r"cr_|tc_|class|detect|"
            r"facility|reported|emission|"
            r"release|flow|survey|timestamp|"
            r"plume|windtype|team|satellite)",
            normalize(column),
        )
    ]

    column_records = []

    for column in relevant_columns:
        series = frame[column]

        column_records.append({
            "column":
                column,

            "dtype":
                str(series.dtype),

            "non_null_count":
                int(series.notna().sum()),

            "unique_non_null_count":
                int(
                    series.nunique(
                        dropna=True
                    )
                ),

            "example_values":
                " || ".join(
                    series.dropna()
                    .astype(str)
                    .drop_duplicates()
                    .head(10)
                    .tolist()
                ),
        })

    column_audit = pd.DataFrame(
        column_records
    )

    column_audit.to_csv(
        COLUMN_OUTPUT,
        index=False,
    )

    print("\nRELEVANT COLUMNS")
    print("-" * 115)

    for column in relevant_columns:
        print(column)

    time_candidates = [
        column
        for column in [
            "datetime_UTC",
            "Operator_Timestamp",
            "Stanford_timestamp",
            "DateOfSurvey",
            "SurveyTime",
            "StartTime",
            "EndTime",
            "cr_idx",
        ]
        if column in frame.columns
    ]

    key_records = []

    for column in time_candidates:
        key_records.append({
            "candidate_key":
                column,

            "non_null_rows":
                int(
                    frame[column]
                    .notna()
                    .sum()
                ),

            "unique_values":
                int(
                    frame[column]
                    .nunique(
                        dropna=True
                    )
                ),
        })

    combination_candidates = [
        [
            column
            for column in [
                "datetime_UTC",
                "cr_idx",
            ]
            if column in frame.columns
        ],
        [
            column
            for column in [
                "Operator_Timestamp",
                "cr_idx",
            ]
            if column in frame.columns
        ],
        [
            column
            for column in [
                "DateOfSurvey",
                "SurveyTime",
            ]
            if column in frame.columns
        ],
    ]

    for columns in combination_candidates:
        if len(columns) < 2:
            continue

        key_records.append({
            "candidate_key":
                " + ".join(columns),

            "non_null_rows":
                int(
                    frame[columns]
                    .notna()
                    .all(axis=1)
                    .sum()
                ),

            "unique_values":
                int(
                    frame[columns]
                    .dropna()
                    .drop_duplicates()
                    .shape[0]
                ),
        })

    key_summary = pd.DataFrame(
        key_records
    )

    print("\nCANDIDATE KEY COUNTS")
    print("-" * 115)
    print(
        key_summary.to_string(
            index=False
        )
    )

    crosstab_sections = []

    for column in [
        "tc_Classification",
        "PlumeEstablished",
        "PlumeSteady",
        "WindType",
    ]:
        if (
            column in frame.columns
            and "Detection" in frame.columns
        ):
            table = pd.crosstab(
                frame[column],
                frame["Detection"],
                dropna=False,
            )

            crosstab_sections.extend([
                "",
                f"Detection by {column}:",
                table.to_string(),
            ])

    print("\nDETECTION CROSSTABS")
    print("-" * 115)

    if crosstab_sections:
        print(
            "\n".join(
                crosstab_sections
            )
        )
    else:
        print(
            "No classification columns found."
        )

    patterns = [
        r"Detection",
        r"tc_Classification",
        r"FacilityEmissionRate",
        r"cr_kgh",
        r"Detection\s*==",
        r"Detection\s*=",
        r"TP|TN|FP|FN",
        r"false positive",
        r"false negative",
    ]

    hits = code_hits(
        CODE_PATH,
        patterns,
        context=7,
    )

    print("\nDETECTION CODE HITS")
    print("-" * 115)

    print("\n".join(hits))

    test_files = inspect_test_files(
        TEST_DATA_DIR
    )

    test_files.to_csv(
        TEST_FILE_OUTPUT,
        index=False,
    )

    print("\nTEST DATA FILES")
    print("-" * 115)

    if test_files.empty:
        print("No CSV test files found.")
    else:
        print(
            test_files.to_string(
                index=False,
                max_colwidth=90,
            )
        )

    report_lines = [
        "=" * 115,
        "CARBON MAPPER DETECTION-DEFINITION AUDIT V1",
        "=" * 115,
        "",
        "RELEVANT COLUMNS",
        *relevant_columns,
        "",
        "CANDIDATE KEY COUNTS",
        key_summary.to_string(index=False),
        "",
        "DETECTION CROSSTABS",
        *crosstab_sections,
        "",
        "DETECTION CODE HITS",
        *hits,
        "",
        "TEST DATA FILES",
        (
            test_files.to_string(index=False)
            if not test_files.empty
            else "No CSV test files found."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(COLUMN_OUTPUT)
    print(TEST_FILE_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
