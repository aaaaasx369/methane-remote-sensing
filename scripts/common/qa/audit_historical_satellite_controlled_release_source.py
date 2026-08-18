from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(
    "raw_data/2023_Controlled_Release_2021/"
    "Dataframes for Stanford analysis"
)

FILE_OUTPUT = Path(
    "outputs/530_historical_satellite_file_candidates_v1.csv"
)

PLATFORM_OUTPUT = Path(
    "outputs/531_historical_satellite_platform_summary_v1.csv"
)

GROUP_OUTPUT = Path(
    "outputs/532_historical_satellite_observation_groups_v1.csv"
)

REVIEW_OUTPUT = Path(
    "outputs/533_historical_satellite_review_groups_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/534_historical_satellite_audit_report_v1.txt"
)


def numeric_suffix(path):
    match = re.search(
        r"_(\d+)\.csv$",
        path.name,
    )

    return (
        int(match.group(1))
        if match
        else -1
    )


def normalize(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def unique_values(series):
    values = (
        series.dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values.ne("")
        & values.str.lower().ne("nan")
    ]

    return sorted(
        values.unique().tolist()
    )


def make_platform_key(row, columns):
    parts = []

    for column in columns:
        value = row.get(column)

        if pd.isna(value):
            continue

        text = str(value).strip()

        if (
            not text
            or text.lower() == "nan"
        ):
            continue

        parts.append(
            f"{column}={text}"
        )

    return (
        " | ".join(parts)
        if parts
        else "UNKNOWN_PLATFORM"
    )


def count_class(group, name):
    return int(
        group[
            "tc_Classification"
        ]
        .astype(str)
        .str.strip()
        .eq(name)
        .sum()
    )


def main():
    candidates = [
        path
        for path in ROOT.glob(
            "matchedDF_Satellites_*.csv"
        )
        if "superceded" not in str(path).lower()
    ]

    if not candidates:
        raise FileNotFoundError(
            "No matchedDF_Satellites_*.csv found."
        )

    candidates = sorted(
        candidates,
        key=lambda path: (
            numeric_suffix(path),
            path.stat().st_mtime,
        ),
        reverse=True,
    )

    file_records = []

    for path in candidates:
        try:
            candidate = pd.read_csv(
                path,
                low_memory=False,
            )

            file_records.append({
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
            file_records.append({
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

    file_audit = pd.DataFrame(
        file_records
    )

    file_audit.to_csv(
        FILE_OUTPUT,
        index=False,
    )

    selected_path = candidates[0]

    frame = pd.read_csv(
        selected_path,
        low_memory=False,
    )

    # --------------------------------------------------
    # Candidate platform columns
    # --------------------------------------------------
    platform_columns = [
        column
        for column in frame.columns
        if re.search(
            r"satellite|platform|sensor|"
            r"instrument|technology|team|"
            r"operator|provider|performer",
            normalize(column),
        )
    ]

    platform_columns = [
        column
        for column in platform_columns
        if frame[column].notna().any()
    ]

    frame["_platform_key"] = frame.apply(
        lambda row: make_platform_key(
            row,
            platform_columns,
        ),
        axis=1,
    )

    # --------------------------------------------------
    # Observation time
    # --------------------------------------------------
    time_columns = [
        column
        for column in [
            "Stanford_timestamp",
            "datetime_UTC",
            "Operator_Timestamp",
        ]
        if column in frame.columns
    ]

    if not time_columns:
        raise KeyError(
            "No observation-time column found."
        )

    for column in time_columns:
        frame[
            f"_{column}_parsed"
        ] = pd.to_datetime(
            frame[column],
            errors="coerce",
            utc=True,
        )

    frame["_observation_time"] = pd.NaT

    for column in time_columns:
        parsed_column = (
            f"_{column}_parsed"
        )

        frame[
            "_observation_time"
        ] = frame[
            "_observation_time"
        ].fillna(
            frame[parsed_column]
        )

    invalid_time_count = int(
        frame[
            "_observation_time"
        ].isna().sum()
    )

    # --------------------------------------------------
    # Platform column values
    # --------------------------------------------------
    platform_value_lines = []

    for column in platform_columns:
        values = (
            frame[column]
            .value_counts(
                dropna=False
            )
        )

        platform_value_lines.extend([
            "",
            f"{column}:",
            values.to_string(),
        ])

    # --------------------------------------------------
    # Observation groups
    # --------------------------------------------------
    valid = frame[
        frame[
            "_observation_time"
        ].notna()
    ].copy()

    group_records = []

    for (
        platform_key,
        observation_time,
    ), group in valid.groupby(
        [
            "_platform_key",
            "_observation_time",
        ],
        sort=True,
    ):
        classifications = unique_values(
            group["tc_Classification"]
        ) if "tc_Classification" in group.columns else []

        detections = (
            sorted(
                pd.to_numeric(
                    group["Detection"],
                    errors="coerce",
                )
                .dropna()
                .unique()
                .tolist()
            )
            if "Detection" in group.columns
            else []
        )

        cr_indices = (
            sorted(
                pd.to_numeric(
                    group["cr_idx"],
                    errors="coerce",
                )
                .dropna()
                .unique()
                .tolist()
            )
            if "cr_idx" in group.columns
            else []
        )

        review_reasons = []

        if len(classifications) != 1:
            review_reasons.append(
                "mixed_tc_classification"
            )

        if len(detections) != 1:
            review_reasons.append(
                "mixed_detection"
            )

        if len(cr_indices) > 1:
            review_reasons.append(
                "multiple_cr_idx"
            )

        group_records.append({
            "platform_key":
                platform_key,

            "observation_time_utc":
                observation_time,

            "source_row_count":
                len(group),

            "source_row_indices":
                " | ".join(
                    str(index)
                    for index in group.index
                ),

            "tc_classification_values":
                " | ".join(
                    classifications
                ),

            "tc_classification_count":
                len(classifications),

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

            "manual_review_required":
                len(review_reasons) > 0,

            "review_reasons":
                " | ".join(
                    review_reasons
                ),
        })

    groups = pd.DataFrame(
        group_records
    )

    groups.to_csv(
        GROUP_OUTPUT,
        index=False,
    )

    review = groups[
        groups[
            "manual_review_required"
        ]
    ].copy()

    review.to_csv(
        REVIEW_OUTPUT,
        index=False,
    )

    # --------------------------------------------------
    # Platform-level summary
    # --------------------------------------------------
    platform_records = []

    for platform_key, group in groups.groupby(
        "platform_key",
        sort=True,
    ):
        class_series = group.loc[
            group[
                "tc_classification_count"
            ].eq(1),
            "tc_classification_values",
        ]

        class_counts = (
            class_series.value_counts()
        )

        tp = int(
            class_counts.get("TP", 0)
        )

        fn = int(
            class_counts.get("FN", 0)
        )

        tn = int(
            class_counts.get("TN", 0)
        )

        fp = int(
            class_counts.get("FP", 0)
        )

        positive_total = tp + fn
        negative_total = tn + fp

        platform_records.append({
            "platform_key":
                platform_key,

            "independent_observations":
                len(group),

            "tp":
                tp,

            "fn":
                fn,

            "tn":
                tn,

            "fp":
                fp,

            "ne":
                int(
                    class_counts.get(
                        "NE",
                        0,
                    )
                ),

            "ns":
                int(
                    class_counts.get(
                        "NS",
                        0,
                    )
                ),

            "error_or_other":
                int(
                    len(group)
                    - tp
                    - fn
                    - tn
                    - fp
                    - class_counts.get(
                        "NE",
                        0,
                    )
                    - class_counts.get(
                        "NS",
                        0,
                    )
                ),

            "primary_evaluable_observations":
                (
                    positive_total
                    + negative_total
                ),

            "recall":
                (
                    tp / positive_total
                    if positive_total
                    else np.nan
                ),

            "specificity":
                (
                    tn / negative_total
                    if negative_total
                    else np.nan
                ),

            "mixed_classification_groups":
                int(
                    group[
                        "tc_classification_count"
                    ].gt(1).sum()
                ),

            "manual_review_groups":
                int(
                    group[
                        "manual_review_required"
                    ].sum()
                ),

            "minimum_observation_time":
                group[
                    "observation_time_utc"
                ].min(),

            "maximum_observation_time":
                group[
                    "observation_time_utc"
                ].max(),
        })

    platform_summary = pd.DataFrame(
        platform_records
    )

    platform_summary = (
        platform_summary.sort_values(
            [
                "independent_observations",
                "platform_key",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    platform_summary.to_csv(
        PLATFORM_OUTPUT,
        index=False,
    )

    row_count_summary = (
        groups[
            "source_row_count"
        ]
        .value_counts()
        .sort_index()
    )

    report_lines = [
        "=" * 120,
        "HISTORICAL SATELLITE CONTROLLED-RELEASE AUDIT V1",
        "=" * 120,
        "",
        f"Selected input: {selected_path}",
        f"Input rows: {len(frame)}",
        f"Input columns: {len(frame.columns)}",
        (
            "Rows without any usable observation time: "
            f"{invalid_time_count}"
        ),
        "",
        "Candidate platform columns:",
        (
            " | ".join(platform_columns)
            if platform_columns
            else "NONE"
        ),
        "",
        "Platform-column values:",
        *platform_value_lines,
        "",
        (
            "Independent platform-time observations: "
            f"{len(groups)}"
        ),
        (
            "Observation groups requiring review: "
            f"{len(review)}"
        ),
        "",
        "Rows per observation group:",
        row_count_summary.to_string(),
        "",
        "Platform summary:",
        platform_summary.to_string(
            index=False
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 120)
    print(
        "HISTORICAL SATELLITE CONTROLLED-RELEASE AUDIT"
    )
    print("=" * 120)

    print("\nSelected input:")
    print(selected_path)

    print("\nRows:", len(frame))
    print("Columns:", len(frame.columns))

    print(
        "Rows without usable observation time:",
        invalid_time_count,
    )

    print("\nCandidate platform columns:")
    print(
        " | ".join(platform_columns)
        if platform_columns
        else "NONE"
    )

    print("\nPlatform-column values:")
    print(
        "\n".join(platform_value_lines)
        if platform_value_lines
        else "No non-empty platform columns."
    )

    print(
        "\nIndependent platform-time observations:",
        len(groups),
    )

    print(
        "Observation groups requiring review:",
        len(review),
    )

    print("\nRows per observation group:")
    print(row_count_summary)

    print("\nPlatform summary:")
    print(
        platform_summary.to_string(
            index=False,
            max_colwidth=100,
        )
    )

    if not review.empty:
        print("\nReview groups:")
        print(
            review.head(30).to_string(
                index=False,
                max_colwidth=100,
            )
        )

    print("\nSaved:")
    print(FILE_OUTPUT)
    print(PLATFORM_OUTPUT)
    print(GROUP_OUTPUT)
    print(REVIEW_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
