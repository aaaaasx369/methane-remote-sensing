from pathlib import Path
import re

import pandas as pd


OUTPUT_DIR = Path("outputs")

SUMMARY_OUTPUT = OUTPUT_DIR / "440_s2_low_emission_source_audit_v1.csv"
REPORT_OUTPUT = OUTPUT_DIR / "441_s2_low_emission_source_audit_report_v1.txt"

KNOWN_FILES = [
    OUTPUT_DIR / "18_methaneair_s2_dataset_table.csv",
    OUTPUT_DIR / "20_controlled_release_s2_patch_index.csv",
    OUTPUT_DIR / "22_controlled_release_s2_dataset_table.csv",
    OUTPUT_DIR / "25_s2_patch_features.csv",
    OUTPUT_DIR / "309_all_exact_release_intervals_for_s2.csv",
    OUTPUT_DIR / "390_multisensor_master_manifest_v1.csv",
]

EMISSION_PATTERNS = [
    "emission",
    "release_rate",
    "release.*rate",
    "flow_rate",
    "flow.*rate",
    "kg_h",
    "kgph",
    "kg_hr",
    "tph",
    "ton.*hour",
    "methane_rate",
    "ch4_rate",
    "rate_mean",
    "rate_median",
    "rate_max",
]

TIME_PATTERNS = [
    "datetime",
    "time_utc",
    "acquisition",
    "release_start",
    "release_end",
    "overpass",
]

ID_PATTERNS = [
    "event_id",
    "release_id",
    "sample_id",
    "scene_id",
    "source_id",
    "site",
    "filename",
    "patch",
    "relative_path",
]

SENSOR_PATTERNS = [
    "sensor",
    "platform",
    "satellite",
    "dataset_group",
]

LABEL_PATTERNS = [
    "label",
    "target",
    "class",
]


def matches(column, patterns):
    name = str(column).lower()

    return any(
        re.search(pattern, name)
        for pattern in patterns
    )


def find_columns(columns, patterns):
    return [
        column
        for column in columns
        if matches(column, patterns)
    ]


def summarize_numeric(series):
    empty_summary = {
        "non_null_numeric": 0,
        "minimum": None,
        "p25": None,
        "median": None,
        "p75": None,
        "maximum": None,
        "unique_numeric": 0,
    }

    # True/False flags are not emission-rate measurements.
    if pd.api.types.is_bool_dtype(series.dtype):
        return empty_summary

    numeric = pd.to_numeric(
        series.astype("object"),
        errors="coerce",
    ).astype("float64")

    valid = numeric.dropna()

    if valid.empty:
        return empty_summary

    return {
        "non_null_numeric": int(valid.notna().sum()),
        "minimum": float(valid.min()),
        "p25": float(valid.quantile(0.25)),
        "median": float(valid.median()),
        "p75": float(valid.quantile(0.75)),
        "maximum": float(valid.max()),
        "unique_numeric": int(valid.nunique()),
    }


def get_candidate_files():
    candidates = []

    for path in KNOWN_FILES:
        if path.exists():
            candidates.append(path)

    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        lower_name = path.name.lower()

        if (
            "sentinel" in lower_name
            or "_s2_" in lower_name
            or lower_name.startswith("s2_")
            or "release_interval" in lower_name
        ):
            candidates.append(path)

    unique = []
    seen = set()

    for path in candidates:
        resolved = str(path)

        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)

    return unique


def main():
    candidate_files = get_candidate_files()

    audit_rows = []
    report_lines = [
        "=" * 110,
        "SENTINEL-2 LOW-EMISSION INPUT AUDIT V1",
        "=" * 110,
        "",
        f"Candidate CSV files found: {len(candidate_files)}",
        "",
    ]

    useful_files = []

    for path in candidate_files:
        try:
            frame = pd.read_csv(
                path,
                low_memory=False,
            )
        except Exception as error:
            report_lines.extend([
                "-" * 110,
                str(path),
                f"READ ERROR: {error}",
                "",
            ])
            continue

        columns = list(frame.columns)

        emission_columns = find_columns(
            columns,
            EMISSION_PATTERNS,
        )

        time_columns = find_columns(
            columns,
            TIME_PATTERNS,
        )

        id_columns = find_columns(
            columns,
            ID_PATTERNS,
        )

        sensor_columns = find_columns(
            columns,
            SENSOR_PATTERNS,
        )

        label_columns = find_columns(
            columns,
            LABEL_PATTERNS,
        )

        likely_s2 = any(
            "s2" in str(value).lower()
            or "sentinel-2" in str(value).lower()
            or "sentinel 2" in str(value).lower()
            for column in sensor_columns
            for value in frame[column]
                .dropna()
                .astype(str)
                .head(500)
        )

        has_emission_data = False

        emission_summaries = []

        for column in emission_columns:
            summary = summarize_numeric(
                frame[column]
            )

            if summary["non_null_numeric"] > 0:
                has_emission_data = True

            emission_summaries.append({
                "column": column,
                **summary,
            })

            audit_rows.append({
                "file":
                    str(path),

                "rows":
                    len(frame),

                "columns":
                    len(columns),

                "likely_s2":
                    likely_s2,

                "emission_column":
                    column,

                **summary,
            })

        if emission_columns or likely_s2:
            useful_files.append(path)

        report_lines.extend([
            "-" * 110,
            str(path),
            f"Rows: {len(frame)}",
            f"Columns: {len(columns)}",
            f"Likely Sentinel-2 content: {likely_s2}",
            f"Emission columns: {emission_columns}",
            f"Time columns: {time_columns[:12]}",
            f"ID/path columns: {id_columns[:15]}",
            f"Sensor/group columns: {sensor_columns[:10]}",
            f"Label columns: {label_columns[:10]}",
        ])

        if label_columns:
            label_column = label_columns[0]

            report_lines.append(
                "Label counts: "
                + str(
                    frame[label_column]
                    .value_counts(
                        dropna=False
                    )
                    .head(10)
                    .to_dict()
                )
            )

        if emission_summaries:
            report_lines.append(
                "Emission summaries:"
            )

            for item in emission_summaries:
                report_lines.append(
                    "  "
                    + f"{item['column']}: "
                    + f"n={item['non_null_numeric']}, "
                    + f"min={item['minimum']}, "
                    + f"p25={item['p25']}, "
                    + f"median={item['median']}, "
                    + f"p75={item['p75']}, "
                    + f"max={item['maximum']}"
                )
        else:
            report_lines.append(
                "Emission summaries: none"
            )

        report_lines.append("")

    audit = pd.DataFrame(audit_rows)

    if audit.empty:
        audit = pd.DataFrame(columns=[
            "file",
            "rows",
            "columns",
            "likely_s2",
            "emission_column",
            "non_null_numeric",
            "minimum",
            "p25",
            "median",
            "p75",
            "maximum",
            "unique_numeric",
        ])

    audit.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    report_lines.extend([
        "=" * 110,
        "MOST USEFUL FILES",
        "=" * 110,
    ])

    if audit.empty:
        report_lines.append(
            "No numeric emission-rate columns found."
        )
    else:
        ranked = (
            audit[
                audit["non_null_numeric"].gt(0)
            ]
            .sort_values(
                [
                    "non_null_numeric",
                    "unique_numeric",
                ],
                ascending=False,
            )
        )

        if ranked.empty:
            report_lines.append(
                "No numeric emission-rate columns found."
            )
        else:
            for _, row in ranked.head(15).iterrows():
                report_lines.append(
                    f"{row['file']} | "
                    f"{row['emission_column']} | "
                    f"n={int(row['non_null_numeric'])} | "
                    f"min={row['minimum']} | "
                    f"median={row['median']} | "
                    f"max={row['maximum']}"
                )

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 100)
    print("S2 LOW-EMISSION INPUT AUDIT")
    print("=" * 100)

    print(
        "\nCSV files inspected:",
        len(candidate_files),
    )

    valid = audit[
        audit["non_null_numeric"].gt(0)
    ].copy()

    print(
        "Numeric emission-rate fields found:",
        len(valid),
    )

    if not valid.empty:
        print("\nTop candidate fields:")

        display = (
            valid.sort_values(
                [
                    "non_null_numeric",
                    "unique_numeric",
                ],
                ascending=False,
            )
            .head(12)
            [
                [
                    "file",
                    "emission_column",
                    "non_null_numeric",
                    "minimum",
                    "median",
                    "maximum",
                ]
            ]
        )

        print(
            display.to_string(
                index=False
            )
        )

    print("\nSaved:")
    print(SUMMARY_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
