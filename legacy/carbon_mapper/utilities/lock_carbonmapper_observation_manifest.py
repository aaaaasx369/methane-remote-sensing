from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/494_carbonmapper_observation_manifest_v1.csv"
)

LOCKED_OUTPUT = Path(
    "outputs/498_carbonmapper_observation_manifest_locked_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/499_carbonmapper_observation_manifest_locked_report_v1.txt"
)

REVIEWED_OBSERVATION_ID = "CM_OBS_1c1eba59156b"


def to_boolean(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    frame = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required = [
        "carbonmapper_observation_id",
        "source_row_count",
        "cr_idx",
        "tc_classification",
        "numeric_detection",
        "ground_truth_rate_ratio",
        "qc_filter_values",
        "plume_established_values",
        "plume_steady_values",
        "manual_review_required",
        "analysis_role",
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

    frame[
        "manual_review_required"
    ] = to_boolean(
        frame["manual_review_required"]
    )

    reviewed = frame[
        frame[
            "carbonmapper_observation_id"
        ].eq(REVIEWED_OBSERVATION_ID)
    ].copy()

    if len(reviewed) != 1:
        raise RuntimeError(
            "Expected exactly one reviewed observation, "
            f"found {len(reviewed)}."
        )

    row = reviewed.iloc[0]

    validation_checks = {
        "source_row_count_is_6":
            int(row["source_row_count"]) == 6,

        "classification_is_TP":
            str(row["tc_classification"]) == "TP",

        "numeric_detection_is_1":
            np.isclose(
                float(row["numeric_detection"]),
                1.0,
            ),

        "ground_truth_rate_consistent":
            float(
                row["ground_truth_rate_ratio"]
            ) <= 1.01,

        "qc_filter_pass":
            str(
                row["qc_filter_values"]
            ).strip() in {
                "1",
                "1.0",
            },

        "plume_established":
            str(
                row["plume_established_values"]
            ).strip().lower() == "true",

        "plume_steady":
            str(
                row["plume_steady_values"]
            ).strip().lower() == "true",
    }

    failed_checks = [
        name
        for name, passed
        in validation_checks.items()
        if not passed
    ]

    if failed_checks:
        raise RuntimeError(
            "Reviewed observation failed validation: "
            + ", ".join(failed_checks)
        )

    target = frame[
        "carbonmapper_observation_id"
    ].eq(REVIEWED_OBSERVATION_ID)

    frame.loc[
        target,
        "manual_review_required",
    ] = False

    frame.loc[
        target,
        "review_status",
    ] = (
        "accepted_as_single_observation"
    )

    frame.loc[
        target,
        "review_resolution",
    ] = (
        "six_rows_are_consistent_multi_wind_"
        "and_emission_estimation_versions"
    )

    frame.loc[
        target,
        "review_reasons",
    ] = ""

    frame[
        "locked_for_analysis"
    ] = (
        ~frame["manual_review_required"]
    )

    frame[
        "independent_observation_definition"
    ] = (
        "one_unique_datetime_UTC"
    )

    frame[
        "aggregation_definition"
    ] = (
        "multiple_wind_or_estimation_rows_"
        "aggregated_to_one_observation"
    )

    frame = frame.sort_values(
        [
            "observation_time_utc",
            "carbonmapper_observation_id",
        ]
    ).reset_index(drop=True)

    frame.to_csv(
        LOCKED_OUTPUT,
        index=False,
    )

    class_summary = (
        frame["tc_classification"]
        .value_counts(dropna=False)
    )

    role_summary = (
        frame["analysis_role"]
        .value_counts(dropna=False)
    )

    primary = frame[
        frame["analysis_role"].eq(
            "primary_evaluable"
        )
        & frame["locked_for_analysis"]
    ].copy()

    confusion = (
        primary["tc_classification"]
        .value_counts()
        .reindex(
            ["TP", "FN", "TN", "FP"],
            fill_value=0,
        )
    )

    tp = int(confusion["TP"])
    fn = int(confusion["FN"])
    tn = int(confusion["TN"])
    fp = int(confusion["FP"])

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else np.nan
    )

    balanced_accuracy = (
        (recall + specificity) / 2
        if (
            pd.notna(recall)
            and pd.notna(specificity)
        )
        else np.nan
    )

    unresolved_count = int(
        frame[
            "manual_review_required"
        ].sum()
    )

    report_lines = [
        "=" * 110,
        "CARBON MAPPER LOCKED OBSERVATION MANIFEST V1",
        "=" * 110,
        "",
        f"Independent observations: {len(frame)}",
        (
            "Locked observations: "
            f"{int(frame['locked_for_analysis'].sum())}"
        ),
        (
            "Unresolved structural reviews: "
            f"{unresolved_count}"
        ),
        "",
        "Observation classifications:",
        class_summary.to_string(),
        "",
        "Analysis roles:",
        role_summary.to_string(),
        "",
        "Primary confusion counts:",
        confusion.to_string(),
        "",
        f"Recall: {recall}",
        f"Specificity: {specificity}",
        f"Balanced accuracy: {balanced_accuracy}",
        "",
        "Reviewed observation:",
        REVIEWED_OBSERVATION_ID,
        (
            "Decision: retain as one independent TP observation. "
            "The six source rows are consistent estimates using "
            "multiple wind or quantification configurations."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 110)
    print(
        "CARBON MAPPER LOCKED OBSERVATION MANIFEST"
    )
    print("=" * 110)

    print(
        "\nIndependent observations:",
        len(frame),
    )

    print(
        "Locked observations:",
        int(
            frame[
                "locked_for_analysis"
            ].sum()
        ),
    )

    print(
        "Unresolved structural reviews:",
        unresolved_count,
    )

    print("\nPrimary confusion counts:")
    print(confusion)

    print("\nRecall:", recall)
    print("Specificity:", specificity)
    print(
        "Balanced accuracy:",
        balanced_accuracy,
    )

    print("\nSaved:")
    print(LOCKED_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
