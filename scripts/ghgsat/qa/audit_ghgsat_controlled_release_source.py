from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(
    "raw_data/2023_Controlled_Release_2021/"
    "Dataframes for Stanford analysis"
)

CANDIDATE_OUTPUT = Path(
    "outputs/513_ghgsat_matched_file_candidates_v1.csv"
)

OBSERVATION_OUTPUT = Path(
    "outputs/514_ghgsat_timestamp_structure_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/515_ghgsat_controlled_release_audit_report_v1.txt"
)

PREVIEW_OUTPUT = Path(
    "outputs/516_ghgsat_controlled_release_preview_v1.csv"
)


def numeric_suffix(path):
    match = re.search(
        r"_(\d+)\.csv$",
        path.name,
    )

    if match:
        return int(match.group(1))

    return -1


def unique_text(series):
    values = (
        series.dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values.ne("")
    ]

    return " | ".join(
        sorted(
            values.unique().tolist()
        )
    )


def numeric_summary(frame, columns):
    records = []

    for column in columns:
        if column not in frame.columns:
            continue

        numeric = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if not numeric.notna().any():
            continue

        records.append({
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

    return pd.DataFrame(records)


def main():
    candidates = [
        path
        for path in ROOT.glob(
            "matchedDF_GHGSat_*.csv"
        )
        if "superceded" not in str(path).lower()
    ]

    if not candidates:
        raise FileNotFoundError(
            "No matchedDF_GHGSat_*.csv files found."
        )

    candidates = sorted(
        candidates,
        key=lambda path: (
            numeric_suffix(path),
            path.stat().st_mtime,
        ),
        reverse=True,
    )

    candidate_records = []

    for path in candidates:
        try:
            candidate = pd.read_csv(
                path,
                low_memory=False,
            )

            candidate_records.append({
                "path":
                    str(path),

                "numeric_suffix":
                    numeric_suffix(path),

                "rows":
                    len(candidate),

                "columns":
                    len(candidate.columns),

                "modified_time":
                    pd.Timestamp(
                        path.stat().st_mtime,
                        unit="s",
                    ),
            })

        except Exception as error:
            candidate_records.append({
                "path":
                    str(path),

                "numeric_suffix":
                    numeric_suffix(path),

                "rows":
                    np.nan,

                "columns":
                    np.nan,

                "modified_time":
                    pd.NaT,

                "error":
                    str(error),
            })

    candidate_audit = pd.DataFrame(
        candidate_records
    )

    candidate_audit.to_csv(
        CANDIDATE_OUTPUT,
        index=False,
    )

    selected_path = candidates[0]

    frame = pd.read_csv(
        selected_path,
        low_memory=False,
    )

    print("=" * 115)
    print(
        "GHGSAT CONTROLLED-RELEASE SOURCE AUDIT"
    )
    print("=" * 115)

    print("\nCandidate files:")
    print(
        candidate_audit.to_string(
            index=False,
            max_colwidth=100,
        )
    )

    print("\nSelected input:")
    print(selected_path)

    print("\nRows:", len(frame))
    print("Columns:", len(frame.columns))

    required_candidates = [
        "datetime_UTC",
        "Operator_Timestamp",
        "Stanford_timestamp",
        "cr_idx",
        "tc_Classification",
        "Detection",
        "QC filter",
        "FacilityEmissionRate",
        "cr_kgh_CH4_mean60",
        "Satellite",
        "Team",
        "WindType",
        "PlumeEstablished",
        "PlumeSteady",
    ]

    available_columns = [
        column
        for column in required_candidates
        if column in frame.columns
    ]

    print("\nAvailable important columns:")
    print(
        " | ".join(available_columns)
    )

    time_column = next(
        (
            column
            for column in [
                "datetime_UTC",
                "Stanford_timestamp",
                "Operator_Timestamp",
            ]
            if column in frame.columns
        ),
        None,
    )

    if time_column is None:
        raise KeyError(
            "No usable observation-time column found."
        )

    frame["_observation_time"] = (
        pd.to_datetime(
            frame[time_column],
            errors="coerce",
            utc=True,
        )
    )

    key_records = []

    for column in [
        "datetime_UTC",
        "Operator_Timestamp",
        "Stanford_timestamp",
        "cr_idx",
    ]:
        if column not in frame.columns:
            continue

        key_records.append({
            "candidate_key":
                column,

            "non_null_rows":
                int(frame[column].notna().sum()),

            "unique_values":
                int(
                    frame[column].nunique(
                        dropna=True
                    )
                ),
        })

    key_summary = pd.DataFrame(
        key_records
    )

    observation_records = []

    for timestamp, group in frame.groupby(
        "_observation_time",
        dropna=False,
        sort=True,
    ):
        classes = (
            group["tc_Classification"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda values: values.ne("")]
            .unique()
            .tolist()
            if "tc_Classification"
            in group.columns
            else []
        )

        detections = (
            pd.to_numeric(
                group["Detection"],
                errors="coerce",
            )
            .dropna()
            .unique()
            .tolist()
            if "Detection"
            in group.columns
            else []
        )

        cr_indices = (
            pd.to_numeric(
                group["cr_idx"],
                errors="coerce",
            )
            .dropna()
            .unique()
            .tolist()
            if "cr_idx" in group.columns
            else []
        )

        observation_records.append({
            "observation_time_utc":
                timestamp,

            "source_row_count":
                len(group),

            "tc_classification_values":
                " | ".join(
                    sorted(
                        str(value)
                        for value in classes
                    )
                ),

            "tc_classification_count":
                len(classes),

            "detection_values":
                " | ".join(
                    sorted(
                        str(value)
                        for value in detections
                    )
                ),

            "detection_value_count":
                len(detections),

            "cr_idx_values":
                " | ".join(
                    sorted(
                        str(value)
                        for value in cr_indices
                    )
                ),

            "cr_idx_count":
                len(cr_indices),

            "wind_types":
                (
                    unique_text(
                        group["WindType"]
                    )
                    if "WindType"
                    in group.columns
                    else ""
                ),

            "qc_filter_values":
                (
                    unique_text(
                        group["QC filter"]
                    )
                    if "QC filter"
                    in group.columns
                    else ""
                ),

            "satellite_values":
                (
                    unique_text(
                        group["Satellite"]
                    )
                    if "Satellite"
                    in group.columns
                    else ""
                ),

            "team_values":
                (
                    unique_text(
                        group["Team"]
                    )
                    if "Team"
                    in group.columns
                    else ""
                ),
        })

    observations = pd.DataFrame(
        observation_records
    )

    observations.to_csv(
        OBSERVATION_OUTPUT,
        index=False,
    )

    row_count_summary = (
        observations[
            "source_row_count"
        ]
        .value_counts()
        .sort_index()
    )

    mixed_classification_count = int(
        observations[
            "tc_classification_count"
        ].gt(1).sum()
    )

    mixed_detection_count = int(
        observations[
            "detection_value_count"
        ].gt(1).sum()
    )

    row_classification = (
        frame[
            "tc_Classification"
        ]
        .value_counts(
            dropna=False
        )
        if "tc_Classification"
        in frame.columns
        else pd.Series(dtype=int)
    )

    observation_classification = (
        observations.loc[
            observations[
                "tc_classification_count"
            ].eq(1),
            "tc_classification_values",
        ]
        .value_counts(
            dropna=False
        )
    )

    if (
        "tc_Classification" in frame.columns
        and "Detection" in frame.columns
    ):
        detection_crosstab = pd.crosstab(
            frame["tc_Classification"],
            frame["Detection"],
            dropna=False,
        )
    else:
        detection_crosstab = pd.DataFrame()

    if (
        "tc_Classification" in frame.columns
        and "QC filter" in frame.columns
    ):
        qc_crosstab = pd.crosstab(
            frame["tc_Classification"],
            frame["QC filter"],
            dropna=False,
        )
    else:
        qc_crosstab = pd.DataFrame()

    rate_summary = numeric_summary(
        frame,
        [
            "FacilityEmissionRate",
            "FacilityEmissionRateLower",
            "FacilityEmissionRateUpper",
            "cr_kgh_CH4_mean30",
            "cr_kgh_CH4_mean60",
            "cr_kgh_CH4_mean90",
            "cr_kgh_CH4_mean300",
            "cr_kgh_CH4_mean600",
            "cr_kgh_CH4_mean900",
        ],
    )

    platform_summary_lines = []

    for column in [
        "Satellite",
        "Team",
        "OperatorSet",
    ]:
        if column not in frame.columns:
            continue

        values = frame[
            column
        ].value_counts(
            dropna=False
        )

        platform_summary_lines.extend([
            "",
            f"{column}:",
            values.to_string(),
        ])

    preview_columns = [
        column
        for column in [
            time_column,
            "Operator_Timestamp",
            "Stanford_timestamp",
            "cr_idx",
            "tc_Classification",
            "Detection",
            "QC filter",
            "FacilityEmissionRate",
            "cr_kgh_CH4_mean60",
            "Satellite",
            "Team",
            "WindType",
            "PlumeEstablished",
            "PlumeSteady",
        ]
        if column in frame.columns
    ]

    frame[
        preview_columns
    ].head(40).to_csv(
        PREVIEW_OUTPUT,
        index=False,
    )

    report_lines = [
        "=" * 115,
        "GHGSAT CONTROLLED-RELEASE SOURCE AUDIT V1",
        "=" * 115,
        "",
        f"Selected input: {selected_path}",
        f"Input rows: {len(frame)}",
        f"Input columns: {len(frame.columns)}",
        f"Selected observation-time column: {time_column}",
        "",
        "Candidate key counts:",
        key_summary.to_string(index=False),
        "",
        (
            "Independent timestamp observations: "
            f"{len(observations)}"
        ),
        (
            "Timestamp groups with multiple rows: "
            f"{int(observations['source_row_count'].gt(1).sum())}"
        ),
        (
            "Timestamp groups with mixed classifications: "
            f"{mixed_classification_count}"
        ),
        (
            "Timestamp groups with mixed Detection values: "
            f"{mixed_detection_count}"
        ),
        "",
        "Rows per timestamp:",
        row_count_summary.to_string(),
        "",
        "Row-level classifications:",
        row_classification.to_string(),
        "",
        "Observation-level classifications:",
        observation_classification.to_string(),
        "",
        "Classification by Detection:",
        (
            detection_crosstab.to_string()
            if not detection_crosstab.empty
            else "Unavailable"
        ),
        "",
        "Classification by QC filter:",
        (
            qc_crosstab.to_string()
            if not qc_crosstab.empty
            else "Unavailable"
        ),
        "",
        "Emission-rate columns:",
        (
            rate_summary.to_string(index=False)
            if not rate_summary.empty
            else "Unavailable"
        ),
        "",
        "Platform/team values:",
        (
            "\n".join(
                platform_summary_lines
            )
            if platform_summary_lines
            else "Unavailable"
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nCandidate key counts:")
    print(
        key_summary.to_string(
            index=False
        )
    )

    print(
        "\nIndependent timestamp observations:",
        len(observations),
    )

    print(
        "Timestamp groups with multiple rows:",
        int(
            observations[
                "source_row_count"
            ].gt(1).sum()
        ),
    )

    print(
        "Timestamp groups with mixed classifications:",
        mixed_classification_count,
    )

    print(
        "Timestamp groups with mixed Detection values:",
        mixed_detection_count,
    )

    print("\nRows per timestamp:")
    print(row_count_summary)

    print("\nRow-level classifications:")
    print(row_classification)

    print("\nObservation-level classifications:")
    print(observation_classification)

    print("\nClassification by Detection:")
    print(
        detection_crosstab
        if not detection_crosstab.empty
        else "Unavailable"
    )

    print("\nClassification by QC filter:")
    print(
        qc_crosstab
        if not qc_crosstab.empty
        else "Unavailable"
    )

    print("\nEmission-rate columns:")
    print(
        rate_summary.to_string(
            index=False
        )
        if not rate_summary.empty
        else "Unavailable"
    )

    print("\nPlatform/team values:")
    print(
        "\n".join(
            platform_summary_lines
        )
        if platform_summary_lines
        else "Unavailable"
    )

    print("\nSaved:")
    print(CANDIDATE_OUTPUT)
    print(OBSERVATION_OUTPUT)
    print(REPORT_OUTPUT)
    print(PREVIEW_OUTPUT)


if __name__ == "__main__":
    main()
