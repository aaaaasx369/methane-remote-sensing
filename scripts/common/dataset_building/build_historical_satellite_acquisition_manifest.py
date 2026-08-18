from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


INPUT = Path(
    "raw_data/2023_Controlled_Release_2021/"
    "Dataframes for Stanford analysis/"
    "matchedDF_Satellites_220601.csv"
)

TEAM_OUTPUT = Path(
    "outputs/535_historical_satellite_team_observations_v1.csv"
)

ACQUISITION_OUTPUT = Path(
    "outputs/536_historical_satellite_acquisitions_v1.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/537_historical_satellite_sensor_summary_v1.csv"
)

REVIEW_OUTPUT = Path(
    "outputs/538_historical_satellite_acquisition_review_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/539_historical_satellite_acquisition_report_v1.txt"
)


GROUND_TRUTH_RATE_COLUMN = "cr_kgh_CH4_mean60"
REPORTED_RATE_COLUMN = "FacilityEmissionRate"


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


def unique_text(series):
    return " | ".join(
        unique_values(series)
    )


def make_id(prefix, *parts):
    raw = "|".join(
        str(part)
        for part in parts
    )

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:12]

    return f"{prefix}_{digest}"


def numeric_values(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()


def safe_rate_ratio(minimum, maximum):
    if (
        pd.isna(minimum)
        or pd.isna(maximum)
        or minimum <= 0
    ):
        return np.nan

    return maximum / minimum


def emission_bin(rate):
    if pd.isna(rate):
        return "unknown"

    if rate <= 0:
        return "zero_release"

    if rate < 200:
        return "0_to_200"

    if rate < 500:
        return "200_to_500"

    if rate < 1000:
        return "500_to_1000"

    return "above_1000"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    frame = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required = [
        "Operator_Timestamp",
        "Satellite",
        "Team",
        "OperatorSet",
        "tc_Classification",
        "Detection",
        GROUND_TRUTH_RATE_COLUMN,
        REPORTED_RATE_COLUMN,
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

    frame["_acquisition_time"] = pd.to_datetime(
        frame["Operator_Timestamp"],
        errors="coerce",
        utc=True,
    )

    frame["_satellite"] = (
        frame["Satellite"]
        .astype(str)
        .str.strip()
    )

    frame["_team"] = (
        frame["Team"]
        .astype(str)
        .str.strip()
    )

    frame["_classification"] = (
        frame["tc_Classification"]
        .astype(str)
        .str.strip()
    )

    frame["_detection"] = pd.to_numeric(
        frame["Detection"],
        errors="coerce",
    )

    frame["_ground_truth_rate"] = pd.to_numeric(
        frame[GROUND_TRUTH_RATE_COLUMN],
        errors="coerce",
    )

    frame["_reported_rate"] = pd.to_numeric(
        frame[REPORTED_RATE_COLUMN],
        errors="coerce",
    )

    invalid = frame[
        frame["_acquisition_time"].isna()
        | frame["_satellite"].eq("")
        | frame["_satellite"].str.lower().eq("nan")
    ]

    if not invalid.empty:
        raise RuntimeError(
            "Rows missing acquisition time or satellite: "
            f"{len(invalid)}"
        )

    # ==================================================
    # Level 1: one satellite + time + team
    # ==================================================
    team_records = []

    team_group_columns = [
        "_satellite",
        "_acquisition_time",
        "_team",
        "OperatorSet",
    ]

    for keys, group in frame.groupby(
        team_group_columns,
        sort=True,
        dropna=False,
    ):
        satellite, acquisition_time, team, operator_set = keys

        classifications = unique_values(
            group["_classification"]
        )

        detections = sorted(
            group["_detection"]
            .dropna()
            .unique()
            .tolist()
        )

        performer_ids = (
            unique_values(
                group["PerformerExperimentID"]
            )
            if "PerformerExperimentID"
            in group.columns
            else []
        )

        ground_truth_rates = numeric_values(
            group["_ground_truth_rate"]
        )

        reported_rates = numeric_values(
            group["_reported_rate"]
        )

        gt_min = (
            ground_truth_rates.min()
            if not ground_truth_rates.empty
            else np.nan
        )

        gt_median = (
            ground_truth_rates.median()
            if not ground_truth_rates.empty
            else np.nan
        )

        gt_max = (
            ground_truth_rates.max()
            if not ground_truth_rates.empty
            else np.nan
        )

        reported_min = (
            reported_rates.min()
            if not reported_rates.empty
            else np.nan
        )

        reported_median = (
            reported_rates.median()
            if not reported_rates.empty
            else np.nan
        )

        reported_max = (
            reported_rates.max()
            if not reported_rates.empty
            else np.nan
        )

        review_reasons = []

        if len(classifications) != 1:
            review_reasons.append(
                "mixed_classification_within_team_report"
            )

        if len(detections) != 1:
            review_reasons.append(
                "mixed_detection_within_team_report"
            )

        if len(group) != 2:
            review_reasons.append(
                "unexpected_team_source_row_count"
            )

        gt_ratio = safe_rate_ratio(
            gt_min,
            gt_max,
        )

        if (
            pd.notna(gt_ratio)
            and gt_ratio > 1.10
        ):
            review_reasons.append(
                "ground_truth_rate_variation_over_10pct"
            )

        classification = (
            classifications[0]
            if len(classifications) == 1
            else "MIXED"
        )

        detection = (
            detections[0]
            if len(detections) == 1
            else np.nan
        )

        team_records.append({
            "team_observation_id":
                make_id(
                    "SAT_TEAM",
                    satellite,
                    acquisition_time,
                    team,
                    operator_set,
                ),

            "satellite":
                satellite,

            "acquisition_time_utc":
                acquisition_time,

            "team":
                team,

            "operator_set":
                operator_set,

            "performer_experiment_ids":
                " | ".join(performer_ids),

            "source_row_count":
                len(group),

            "source_row_indices":
                " | ".join(
                    str(index)
                    for index in group.index
                ),

            "tc_classification":
                classification,

            "classification_values":
                " | ".join(classifications),

            "numeric_detection":
                detection,

            "detection_values":
                " | ".join(
                    str(value)
                    for value in detections
                ),

            "ground_truth_rate_min_kg_hr":
                gt_min,

            "ground_truth_rate_median_kg_hr":
                gt_median,

            "ground_truth_rate_max_kg_hr":
                gt_max,

            "reported_rate_count":
                len(reported_rates),

            "reported_rate_min_kg_hr":
                reported_min,

            "reported_rate_median_kg_hr":
                reported_median,

            "reported_rate_max_kg_hr":
                reported_max,

            "manual_review_required":
                len(review_reasons) > 0,

            "review_reasons":
                " | ".join(review_reasons),
        })

    team_manifest = pd.DataFrame(
        team_records
    )

    team_manifest.to_csv(
        TEAM_OUTPUT,
        index=False,
    )

    # ==================================================
    # Level 2: one satellite + acquisition time
    # ==================================================
    acquisition_records = []

    for (
        satellite,
        acquisition_time,
    ), group in team_manifest.groupby(
        [
            "satellite",
            "acquisition_time_utc",
        ],
        sort=True,
    ):
        classifications = unique_values(
            group["tc_classification"]
        )

        teams = unique_values(
            group["team"]
        )

        gt_rates = numeric_values(
            group[
                "ground_truth_rate_median_kg_hr"
            ]
        )

        reported_rates = numeric_values(
            group[
                "reported_rate_median_kg_hr"
            ]
        )

        if len(classifications) == 1:
            acquisition_classification = (
                classifications[0]
            )
        else:
            acquisition_classification = (
                "MIXED"
            )

        gt_min = (
            gt_rates.min()
            if not gt_rates.empty
            else np.nan
        )

        gt_median = (
            gt_rates.median()
            if not gt_rates.empty
            else np.nan
        )

        gt_max = (
            gt_rates.max()
            if not gt_rates.empty
            else np.nan
        )

        reported_min = (
            reported_rates.min()
            if not reported_rates.empty
            else np.nan
        )

        reported_median = (
            reported_rates.median()
            if not reported_rates.empty
            else np.nan
        )

        reported_max = (
            reported_rates.max()
            if not reported_rates.empty
            else np.nan
        )

        review_reasons = []

        if len(classifications) != 1:
            review_reasons.append(
                "team_classification_disagreement"
            )

        if group[
            "manual_review_required"
        ].fillna(False).any():
            review_reasons.append(
                "team_report_requires_review"
            )

        gt_ratio = safe_rate_ratio(
            gt_min,
            gt_max,
        )

        if (
            pd.notna(gt_ratio)
            and gt_ratio > 1.10
        ):
            review_reasons.append(
                "ground_truth_rate_disagreement_over_10pct"
            )

        acquisition_records.append({
            "satellite_acquisition_id":
                make_id(
                    "SAT_ACQ",
                    satellite,
                    acquisition_time,
                ),

            "satellite":
                satellite,

            "acquisition_time_utc":
                acquisition_time,

            "team_observation_count":
                len(group),

            "team_count":
                len(teams),

            "teams":
                " | ".join(teams),

            "team_observation_ids":
                " | ".join(
                    group[
                        "team_observation_id"
                    ].astype(str)
                ),

            "acquisition_classification":
                acquisition_classification,

            "team_classification_values":
                " | ".join(classifications),

            "all_teams_tp":
                bool(
                    group[
                        "tc_classification"
                    ].eq("TP").all()
                ),

            "ground_truth_rate_min_kg_hr":
                gt_min,

            "ground_truth_rate_median_kg_hr":
                gt_median,

            "ground_truth_rate_max_kg_hr":
                gt_max,

            "emission_bin":
                emission_bin(
                    gt_median
                ),

            "reported_rate_team_count":
                len(reported_rates),

            "reported_rate_min_kg_hr":
                reported_min,

            "reported_rate_median_kg_hr":
                reported_median,

            "reported_rate_max_kg_hr":
                reported_max,

            "manual_review_required":
                len(review_reasons) > 0,

            "review_reasons":
                " | ".join(review_reasons),

            "benchmark_role":
                (
                    "confirmed_positive_detection_evidence"
                    if acquisition_classification == "TP"
                    else "requires_classification_review"
                ),
        })

    acquisitions = pd.DataFrame(
        acquisition_records
    )

    acquisitions = acquisitions.sort_values(
        [
            "satellite",
            "acquisition_time_utc",
        ]
    ).reset_index(drop=True)

    acquisitions.to_csv(
        ACQUISITION_OUTPUT,
        index=False,
    )

    review = acquisitions[
        acquisitions[
            "manual_review_required"
        ]
    ].copy()

    review.to_csv(
        REVIEW_OUTPUT,
        index=False,
    )

    # ==================================================
    # Satellite-level descriptive summary
    # ==================================================
    summary_records = []

    for satellite, group in acquisitions.groupby(
        "satellite",
        sort=True,
    ):
        class_counts = (
            group[
                "acquisition_classification"
            ]
            .value_counts()
        )

        summary_records.append({
            "satellite":
                satellite,

            "independent_acquisitions":
                len(group),

            "team_level_reports":
                int(
                    group[
                        "team_observation_count"
                    ].sum()
                ),

            "unique_teams":
                len(
                    set(
                        " | ".join(
                            group["teams"]
                            .fillna("")
                        ).split(" | ")
                    )
                    - {""}
                ),

            "tp_acquisitions":
                int(
                    class_counts.get(
                        "TP",
                        0,
                    )
                ),

            "fn_acquisitions":
                int(
                    class_counts.get(
                        "FN",
                        0,
                    )
                ),

            "tn_acquisitions":
                int(
                    class_counts.get(
                        "TN",
                        0,
                    )
                ),

            "fp_acquisitions":
                int(
                    class_counts.get(
                        "FP",
                        0,
                    )
                ),

            "other_or_mixed_acquisitions":
                int(
                    len(group)
                    - class_counts.get(
                        "TP",
                        0,
                    )
                    - class_counts.get(
                        "FN",
                        0,
                    )
                    - class_counts.get(
                        "TN",
                        0,
                    )
                    - class_counts.get(
                        "FP",
                        0,
                    )
                ),

            "all_acquisitions_tp":
                bool(
                    group[
                        "acquisition_classification"
                    ].eq("TP").all()
                ),

            "minimum_ground_truth_rate_kg_hr":
                group[
                    "ground_truth_rate_median_kg_hr"
                ].min(),

            "median_ground_truth_rate_kg_hr":
                group[
                    "ground_truth_rate_median_kg_hr"
                ].median(),

            "maximum_ground_truth_rate_kg_hr":
                group[
                    "ground_truth_rate_median_kg_hr"
                ].max(),

            "minimum_acquisition_time":
                group[
                    "acquisition_time_utc"
                ].min(),

            "maximum_acquisition_time":
                group[
                    "acquisition_time_utc"
                ].max(),

            "performance_interpretation":
                (
                    "positive_only_no_recall_denominator"
                    if class_counts.get("FN", 0) == 0
                    and class_counts.get("TN", 0) == 0
                    and class_counts.get("FP", 0) == 0
                    else "contains_performance_denominator"
                ),
        })

    summary = pd.DataFrame(
        summary_records
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    team_row_counts = (
        team_manifest[
            "source_row_count"
        ]
        .value_counts()
        .sort_index()
    )

    report_lines = [
        "=" * 120,
        "HISTORICAL SATELLITE ACQUISITION MANIFEST V1",
        "=" * 120,
        "",
        f"Input source rows: {len(frame)}",
        (
            "Team-level observations: "
            f"{len(team_manifest)}"
        ),
        (
            "Independent satellite acquisitions: "
            f"{len(acquisitions)}"
        ),
        (
            "Acquisitions requiring review: "
            f"{len(review)}"
        ),
        "",
        "Rows per team-level observation:",
        team_row_counts.to_string(),
        "",
        "Satellite-level summary:",
        summary.to_string(index=False),
        "",
        "Interpretation:",
        (
            "Team-level reports are not independent satellite "
            "acquisitions. Reports from multiple teams analysing "
            "the same satellite and acquisition time are merged."
        ),
        (
            "A dataset containing only TP acquisitions cannot "
            "provide recall, specificity, or false-positive rate "
            "because FN, TN, and FP denominators are absent."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 120)
    print(
        "HISTORICAL SATELLITE ACQUISITION MANIFEST"
    )
    print("=" * 120)

    print(
        "\nInput source rows:",
        len(frame),
    )

    print(
        "Team-level observations:",
        len(team_manifest),
    )

    print(
        "Independent satellite acquisitions:",
        len(acquisitions),
    )

    print(
        "Acquisitions requiring review:",
        len(review),
    )

    print(
        "\nRows per team-level observation:"
    )
    print(team_row_counts)

    print("\nSatellite-level summary:")
    print(
        summary.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(TEAM_OUTPUT)
    print(ACQUISITION_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(REVIEW_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
