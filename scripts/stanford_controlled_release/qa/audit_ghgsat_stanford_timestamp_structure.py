from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "raw_data/2023_Controlled_Release_2021/"
    "Dataframes for Stanford analysis/"
    "matchedDF_GHGSat_23822.csv"
)

GROUP_OUTPUT = Path(
    "outputs/517_ghgsat_stanford_timestamp_groups_v1.csv"
)

MISSING_DATETIME_OUTPUT = Path(
    "outputs/518_ghgsat_missing_datetime_utc_rows_v1.csv"
)

REVIEW_OUTPUT = Path(
    "outputs/519_ghgsat_stanford_timestamp_review_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/520_ghgsat_stanford_timestamp_audit_report_v1.txt"
)


def unique_values(series):
    return sorted(
        series.dropna()
        .astype(str)
        .str.strip()
        .loc[lambda values: values.ne("")]
        .unique()
        .tolist()
    )


def unique_text(series):
    return " | ".join(
        unique_values(series)
    )


def main():
    frame = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required = [
        "Stanford_timestamp",
        "datetime_UTC",
        "Operator_Timestamp",
        "cr_idx",
        "tc_Classification",
        "Detection",
        "QC filter",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            "Missing columns: "
            + ", ".join(missing)
        )

    frame["_stanford_time"] = pd.to_datetime(
        frame["Stanford_timestamp"],
        errors="coerce",
        utc=True,
    )

    frame["_datetime_utc"] = pd.to_datetime(
        frame["datetime_UTC"],
        errors="coerce",
        utc=True,
    )

    frame["_operator_time"] = pd.to_datetime(
        frame["Operator_Timestamp"],
        errors="coerce",
        utc=True,
    )

    frame["_detection"] = pd.to_numeric(
        frame["Detection"],
        errors="coerce",
    )

    frame["_cr_idx"] = pd.to_numeric(
        frame["cr_idx"],
        errors="coerce",
    )

    frame["_qc_filter"] = pd.to_numeric(
        frame["QC filter"],
        errors="coerce",
    )

    invalid_stanford = frame[
        frame["_stanford_time"].isna()
    ]

    if not invalid_stanford.empty:
        raise RuntimeError(
            "Stanford_timestamp contains "
            f"{len(invalid_stanford)} invalid rows."
        )

    missing_datetime = frame[
        frame["_datetime_utc"].isna()
    ].copy()

    missing_datetime.to_csv(
        MISSING_DATETIME_OUTPUT,
        index=False,
    )

    records = []

    for timestamp, group in frame.groupby(
        "_stanford_time",
        sort=True,
    ):
        classes = unique_values(
            group["tc_Classification"]
        )

        detections = sorted(
            group["_detection"]
            .dropna()
            .unique()
            .tolist()
        )

        cr_indices = sorted(
            group["_cr_idx"]
            .dropna()
            .unique()
            .tolist()
        )

        qc_values = sorted(
            group["_qc_filter"]
            .dropna()
            .unique()
            .tolist()
        )

        datetime_values = (
            group["_datetime_utc"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        operator_values = (
            group["_operator_time"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        review_reasons = []

        if len(classes) != 1:
            review_reasons.append(
                "mixed_tc_classification"
            )

        if len(detections) != 1:
            review_reasons.append(
                "mixed_detection"
            )

        if len(cr_indices) != 1:
            review_reasons.append(
                "multiple_cr_idx"
            )

        if len(group) not in {3, 6}:
            review_reasons.append(
                "unusual_source_row_count"
            )

        if len(datetime_values) > 1:
            review_reasons.append(
                "multiple_datetime_utc_values"
            )

        if len(operator_values) > 1:
            operator_spread_seconds = (
                operator_values[-1]
                - operator_values[0]
            ).total_seconds()

            if operator_spread_seconds > 120:
                review_reasons.append(
                    "operator_timestamp_spread_over_2min"
                )
        else:
            operator_spread_seconds = 0.0

        records.append({
            "stanford_timestamp":
                timestamp,

            "source_row_count":
                len(group),

            "source_row_indices":
                " | ".join(
                    str(index)
                    for index in group.index
                ),

            "tc_classification_values":
                " | ".join(classes),

            "tc_classification_count":
                len(classes),

            "detection_values":
                " | ".join(
                    str(value)
                    for value in detections
                ),

            "detection_value_count":
                len(detections),

            "cr_idx_values":
                " | ".join(
                    str(value)
                    for value in cr_indices
                ),

            "cr_idx_count":
                len(cr_indices),

            "qc_filter_values":
                " | ".join(
                    str(value)
                    for value in qc_values
                ),

            "datetime_utc_non_null_count":
                int(
                    group[
                        "_datetime_utc"
                    ].notna().sum()
                ),

            "datetime_utc_unique_count":
                len(datetime_values),

            "datetime_utc_values":
                " | ".join(
                    value.isoformat()
                    for value in datetime_values
                ),

            "operator_timestamp_non_null_count":
                int(
                    group[
                        "_operator_time"
                    ].notna().sum()
                ),

            "operator_timestamp_unique_count":
                len(operator_values),

            "operator_timestamp_spread_seconds":
                operator_spread_seconds,

            "wind_types":
                (
                    unique_text(
                        group["WindType"]
                    )
                    if "WindType" in group.columns
                    else ""
                ),

            "plume_established_values":
                (
                    unique_text(
                        group["PlumeEstablished"]
                    )
                    if "PlumeEstablished"
                    in group.columns
                    else ""
                ),

            "plume_steady_values":
                (
                    unique_text(
                        group["PlumeSteady"]
                    )
                    if "PlumeSteady" in group.columns
                    else ""
                ),

            "manual_review_required":
                len(review_reasons) > 0,

            "review_reasons":
                " | ".join(review_reasons),
        })

    groups = pd.DataFrame(records)

    groups.to_csv(
        GROUP_OUTPUT,
        index=False,
    )

    review = groups[
        groups["manual_review_required"]
    ].copy()

    review.to_csv(
        REVIEW_OUTPUT,
        index=False,
    )

    row_count_summary = (
        groups["source_row_count"]
        .value_counts()
        .sort_index()
    )

    observation_class_summary = (
        groups.loc[
            groups[
                "tc_classification_count"
            ].eq(1),
            "tc_classification_values",
        ]
        .value_counts()
    )

    missing_datetime_class_summary = (
        missing_datetime[
            "tc_Classification"
        ]
        .value_counts(
            dropna=False
        )
    )

    review_reason_summary = (
        review["review_reasons"]
        .value_counts()
        if not review.empty
        else pd.Series(dtype=int)
    )

    print("=" * 115)
    print(
        "GHGSAT STANFORD-TIMESTAMP STRUCTURE AUDIT"
    )
    print("=" * 115)

    print("\nInput rows:", len(frame))

    print(
        "Rows missing datetime_UTC:",
        len(missing_datetime),
    )

    print(
        "Unique Stanford timestamps:",
        len(groups),
    )

    print(
        "Groups with multiple rows:",
        int(
            groups[
                "source_row_count"
            ].gt(1).sum()
        ),
    )

    print(
        "Groups with mixed classifications:",
        int(
            groups[
                "tc_classification_count"
            ].gt(1).sum()
        ),
    )

    print(
        "Groups with mixed Detection values:",
        int(
            groups[
                "detection_value_count"
            ].gt(1).sum()
        ),
    )

    print(
        "Groups requiring review:",
        len(review),
    )

    print("\nRows per Stanford timestamp:")
    print(row_count_summary)

    print(
        "\nObservation-level classifications:"
    )
    print(observation_class_summary)

    print(
        "\nClassifications among rows "
        "missing datetime_UTC:"
    )
    print(missing_datetime_class_summary)

    print("\nReview reasons:")
    if review_reason_summary.empty:
        print("None")
    else:
        print(review_reason_summary)

    if not review.empty:
        display_columns = [
            "stanford_timestamp",
            "source_row_count",
            "tc_classification_values",
            "detection_values",
            "cr_idx_values",
            "qc_filter_values",
            "datetime_utc_non_null_count",
            "datetime_utc_values",
            "operator_timestamp_unique_count",
            "operator_timestamp_spread_seconds",
            "review_reasons",
            "source_row_indices",
        ]

        print("\nReview groups:")
        print(
            review[
                display_columns
            ].to_string(
                index=False,
                max_colwidth=100,
            )
        )

    report_lines = [
        "=" * 115,
        "GHGSAT STANFORD-TIMESTAMP STRUCTURE AUDIT V1",
        "=" * 115,
        "",
        f"Input rows: {len(frame)}",
        (
            "Rows missing datetime_UTC: "
            f"{len(missing_datetime)}"
        ),
        (
            "Unique Stanford timestamps: "
            f"{len(groups)}"
        ),
        (
            "Groups with mixed classifications: "
            f"{int(groups['tc_classification_count'].gt(1).sum())}"
        ),
        (
            "Groups with mixed Detection values: "
            f"{int(groups['detection_value_count'].gt(1).sum())}"
        ),
        (
            "Groups requiring review: "
            f"{len(review)}"
        ),
        "",
        "Rows per Stanford timestamp:",
        row_count_summary.to_string(),
        "",
        "Observation-level classifications:",
        observation_class_summary.to_string(),
        "",
        "Rows missing datetime_UTC by classification:",
        missing_datetime_class_summary.to_string(),
        "",
        "Review reasons:",
        (
            review_reason_summary.to_string()
            if not review_reason_summary.empty
            else "None"
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(GROUP_OUTPUT)
    print(MISSING_DATETIME_OUTPUT)
    print(REVIEW_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
