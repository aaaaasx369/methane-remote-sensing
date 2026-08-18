from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


INPUT = Path(
    "raw_data/2023_Controlled_Release_2021/"
    "Dataframes for Stanford analysis/"
    "matchedDF_GHGSat_23822.csv"
)

MANIFEST_OUTPUT = Path(
    "outputs/521_ghgsat_observation_manifest_v1.csv"
)

PRIMARY_OUTPUT = Path(
    "outputs/522_ghgsat_primary_detection_benchmark_v1.csv"
)

BIN_OUTPUT = Path(
    "outputs/523_ghgsat_detection_performance_by_emission_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/524_ghgsat_observation_performance_report_v1.txt"
)


GROUND_TRUTH_RATE_COLUMN = (
    "cr_kgh_CH4_mean60"
)

PRIMARY_CLASSES = {
    "TP",
    "FN",
    "TN",
    "FP",
}

ANALYSIS_ROLE_MAP = {
    "TP": "primary_evaluable",
    "FN": "primary_evaluable",
    "TN": "primary_evaluable",
    "FP": "primary_evaluable",

    "NE":
        "exclude_plume_not_established",

    "ER_MIS":
        "exclude_quality_control_error",

    "ER_FAQ":
        "exclude_quality_control_error",
}


def unique_values(series):
    values = (
        series.dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values.ne("")
    ]

    return sorted(
        values.unique().tolist()
    )


def unique_text(series):
    return " | ".join(
        unique_values(series)
    )


def timestamp_median(series):
    values = (
        pd.to_datetime(
            series,
            errors="coerce",
            utc=True,
        )
        .dropna()
        .sort_values()
    )

    if values.empty:
        return pd.NaT

    middle = len(values) // 2

    if len(values) % 2 == 1:
        return values.iloc[middle]

    first = values.iloc[middle - 1]
    second = values.iloc[middle]

    return first + (
        second - first
    ) / 2


def make_observation_id(timestamp, cr_idx):
    raw = (
        f"{pd.Timestamp(timestamp).isoformat()}"
        f"|{cr_idx}"
    )

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:12]

    return f"GHGSAT_OBS_{digest}"


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


def safe_divide(numerator, denominator):
    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator == 0
    ):
        return np.nan

    return numerator / denominator


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

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
        GROUND_TRUTH_RATE_COLUMN,
        "FacilityEmissionRate",
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

    frame["_cr_idx"] = pd.to_numeric(
        frame["cr_idx"],
        errors="coerce",
    )

    frame["_detection"] = pd.to_numeric(
        frame["Detection"],
        errors="coerce",
    )

    frame["_qc_filter"] = pd.to_numeric(
        frame["QC filter"],
        errors="coerce",
    )

    frame["_ground_truth_rate"] = pd.to_numeric(
        frame[GROUND_TRUTH_RATE_COLUMN],
        errors="coerce",
    )

    frame["_reported_rate"] = pd.to_numeric(
        frame["FacilityEmissionRate"],
        errors="coerce",
    )

    frame["_reported_lower"] = pd.to_numeric(
        frame.get(
            "FacilityEmissionRateLower"
        ),
        errors="coerce",
    )

    frame["_reported_upper"] = pd.to_numeric(
        frame.get(
            "FacilityEmissionRateUpper"
        ),
        errors="coerce",
    )

    if frame["_stanford_time"].isna().any():
        raise RuntimeError(
            "Invalid Stanford_timestamp rows: "
            f"{int(frame['_stanford_time'].isna().sum())}"
        )

    records = []

    for stanford_time, group in frame.groupby(
        "_stanford_time",
        sort=True,
    ):
        classifications = unique_values(
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

        if len(classifications) == 1:
            classification = classifications[0]
        else:
            classification = ""

        if len(detections) == 1:
            detection = detections[0]
        else:
            detection = np.nan

        if len(cr_indices) == 1:
            cr_idx = cr_indices[0]
        else:
            cr_idx = np.nan

        standardized_time = timestamp_median(
            group["_datetime_utc"]
        )

        operator_time = timestamp_median(
            group["_operator_time"]
        )

        if pd.notna(standardized_time):
            best_acquisition_time = (
                standardized_time
            )

            acquisition_time_source = (
                "datetime_UTC"
            )

        elif pd.notna(operator_time):
            best_acquisition_time = (
                operator_time
            )

            acquisition_time_source = (
                "Operator_Timestamp"
            )

        else:
            best_acquisition_time = (
                stanford_time
            )

            acquisition_time_source = (
                "Stanford_timestamp_fallback"
            )

        ground_truth_rates = group[
            "_ground_truth_rate"
        ].dropna()

        reported_rates = group[
            "_reported_rate"
        ].dropna()

        reported_lowers = group[
            "_reported_lower"
        ].dropna()

        reported_uppers = group[
            "_reported_upper"
        ].dropna()

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

        quantification_error = (
            reported_median - gt_median
            if (
                pd.notna(reported_median)
                and pd.notna(gt_median)
            )
            else np.nan
        )

        quantification_error_percent = (
            100
            * safe_divide(
                quantification_error,
                gt_median,
            )
        )

        review_reasons = []

        if len(classifications) != 1:
            review_reasons.append(
                "mixed_tc_classification"
            )

        if len(detections) != 1:
            review_reasons.append(
                "mixed_numeric_detection"
            )

        if len(cr_indices) != 1:
            review_reasons.append(
                "multiple_cr_idx"
            )

        if len(group) not in {3, 6}:
            review_reasons.append(
                "unexpected_source_row_count"
            )

        if (
            pd.notna(gt_min)
            and pd.notna(gt_max)
            and gt_min > 0
            and gt_max / gt_min > 1.10
        ):
            review_reasons.append(
                "ground_truth_rate_variation_over_10pct"
            )

        analysis_role = ANALYSIS_ROLE_MAP.get(
            classification,
            "manual_review_required",
        )

        primary_evaluable = (
            classification
            in PRIMARY_CLASSES
        )

        if classification in {
            "TP",
            "FN",
        }:
            release_present = True

        elif classification in {
            "TN",
            "FP",
        }:
            release_present = False

        else:
            release_present = np.nan

        sensor_reported_detection = (
            classification in {
                "TP",
                "FP",
            }
        )

        records.append({
            "ghgsat_observation_id":
                make_observation_id(
                    stanford_time,
                    cr_idx,
                ),

            "stanford_timestamp":
                stanford_time,

            "best_available_acquisition_time_utc":
                best_acquisition_time,

            "acquisition_time_source":
                acquisition_time_source,

            "standardized_datetime_utc":
                standardized_time,

            "operator_timestamp_utc":
                operator_time,

            "has_standardized_datetime_utc":
                pd.notna(standardized_time),

            "has_operator_timestamp":
                pd.notna(operator_time),

            "source_row_count":
                len(group),

            "source_row_indices":
                " | ".join(
                    str(index)
                    for index in group.index
                ),

            "cr_idx":
                cr_idx,

            "cr_idx_values":
                " | ".join(
                    str(value)
                    for value in cr_indices
                ),

            "tc_classification":
                classification,

            "tc_classification_values":
                " | ".join(
                    classifications
                ),

            "numeric_detection":
                detection,

            "numeric_detection_values":
                " | ".join(
                    str(value)
                    for value in detections
                ),

            "qc_filter_values":
                " | ".join(
                    str(value)
                    for value in qc_values
                ),

            "analysis_role":
                analysis_role,

            "primary_evaluable":
                primary_evaluable,

            "release_present_from_class":
                release_present,

            "sensor_reported_detection":
                sensor_reported_detection,

            "ground_truth_rate_column":
                GROUND_TRUTH_RATE_COLUMN,

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

            "reported_rate_count":
                len(reported_rates),

            "reported_rate_min_kg_hr":
                reported_min,

            "reported_rate_median_kg_hr":
                reported_median,

            "reported_rate_max_kg_hr":
                reported_max,

            "reported_lower_min_kg_hr":
                (
                    reported_lowers.min()
                    if not reported_lowers.empty
                    else np.nan
                ),

            "reported_upper_max_kg_hr":
                (
                    reported_uppers.max()
                    if not reported_uppers.empty
                    else np.nan
                ),

            "quantification_error_median_kg_hr":
                quantification_error,

            "quantification_error_percent":
                quantification_error_percent,

            "wind_types":
                (
                    unique_text(
                        group["WindType"]
                    )
                    if "WindType"
                    in group.columns
                    else ""
                ),

            "plume_established_values":
                (
                    unique_text(
                        group[
                            "PlumeEstablished"
                        ]
                    )
                    if "PlumeEstablished"
                    in group.columns
                    else ""
                ),

            "plume_steady_values":
                (
                    unique_text(
                        group[
                            "PlumeSteady"
                        ]
                    )
                    if "PlumeSteady"
                    in group.columns
                    else ""
                ),

            "manual_review_required":
                len(review_reasons) > 0,

            "review_reasons":
                " | ".join(
                    review_reasons
                ),
        })

    manifest = pd.DataFrame(records)

    manifest = manifest.sort_values(
        [
            "stanford_timestamp",
            "ghgsat_observation_id",
        ]
    ).reset_index(drop=True)

    acquisition_group_size = (
        manifest.groupby(
            "best_available_acquisition_time_utc",
            dropna=False,
        )[
            "ghgsat_observation_id"
        ]
        .transform("size")
    )

    manifest[
        "acquisition_time_group_size"
    ] = acquisition_group_size

    manifest[
        "shared_acquisition_time"
    ] = acquisition_group_size.gt(1)

    manifest[
        "locked_for_analysis"
    ] = ~manifest[
        "manual_review_required"
    ]

    manifest.to_csv(
        MANIFEST_OUTPUT,
        index=False,
    )

    primary = manifest[
        manifest[
            "primary_evaluable"
        ]
        & manifest[
            "locked_for_analysis"
        ]
    ].copy()

    primary.to_csv(
        PRIMARY_OUTPUT,
        index=False,
    )

    confusion = (
        primary[
            "tc_classification"
        ]
        .value_counts()
        .reindex(
            [
                "TP",
                "FN",
                "TN",
                "FP",
            ],
            fill_value=0,
        )
    )

    tp = int(confusion["TP"])
    fn = int(confusion["FN"])
    tn = int(confusion["TN"])
    fp = int(confusion["FP"])

    recall = safe_divide(
        tp,
        tp + fn,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    false_positive_rate = safe_divide(
        fp,
        tn + fp,
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    accuracy = safe_divide(
        tp + tn,
        tp + fn + tn + fp,
    )

    balanced_accuracy = (
        (recall + specificity) / 2
        if (
            pd.notna(recall)
            and pd.notna(specificity)
        )
        else np.nan
    )

    positive_primary = primary[
        primary[
            "tc_classification"
        ].isin(
            ["TP", "FN"]
        )
    ].copy()

    bin_order = [
        "0_to_200",
        "200_to_500",
        "500_to_1000",
        "above_1000",
        "unknown",
    ]

    bin_records = []

    for category in bin_order:
        group = positive_primary[
            positive_primary[
                "emission_bin"
            ].eq(category)
        ]

        bin_tp = int(
            group[
                "tc_classification"
            ].eq("TP").sum()
        )

        bin_fn = int(
            group[
                "tc_classification"
            ].eq("FN").sum()
        )

        denominator = bin_tp + bin_fn

        bin_records.append({
            "emission_bin":
                category,

            "positive_observations":
                denominator,

            "true_positive_observations":
                bin_tp,

            "false_negative_observations":
                bin_fn,

            "detection_recall":
                safe_divide(
                    bin_tp,
                    denominator,
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
        })

    bin_performance = pd.DataFrame(
        bin_records
    )

    bin_performance.to_csv(
        BIN_OUTPUT,
        index=False,
    )

    tp_rates = primary.loc[
        primary[
            "tc_classification"
        ].eq("TP"),
        "ground_truth_rate_median_kg_hr",
    ].dropna()

    fn_rates = primary.loc[
        primary[
            "tc_classification"
        ].eq("FN"),
        "ground_truth_rate_median_kg_hr",
    ].dropna()

    tp_quantification = primary[
        primary[
            "tc_classification"
        ].eq("TP")
        & primary[
            "quantification_error_median_kg_hr"
        ].notna()
    ].copy()

    class_summary = (
        manifest[
            "tc_classification"
        ]
        .value_counts(
            dropna=False
        )
    )

    role_summary = (
        manifest[
            "analysis_role"
        ]
        .value_counts(
            dropna=False
        )
    )

    acquisition_source_summary = (
        manifest[
            "acquisition_time_source"
        ]
        .value_counts(
            dropna=False
        )
    )

    review_count = int(
        manifest[
            "manual_review_required"
        ].sum()
    )

    shared_acquisition_count = int(
        manifest[
            "shared_acquisition_time"
        ].sum()
    )

    report_lines = [
        "=" * 115,
        "GHGSAT OBSERVATION-LEVEL PERFORMANCE V1",
        "=" * 115,
        "",
        f"Input source rows: {len(frame)}",
        (
            "Independent Stanford-timestamp observations: "
            f"{len(manifest)}"
        ),
        (
            "Observations requiring structural review: "
            f"{review_count}"
        ),
        "",
        "Observation-level classifications:",
        class_summary.to_string(),
        "",
        "Analysis roles:",
        role_summary.to_string(),
        "",
        "Acquisition-time source:",
        acquisition_source_summary.to_string(),
        "",
        (
            "Observations sharing the same best acquisition time: "
            f"{shared_acquisition_count}"
        ),
        "",
        "Primary confusion counts:",
        confusion.to_string(),
        "",
        f"Recall: {recall}",
        f"Specificity: {specificity}",
        f"False-positive rate: {false_positive_rate}",
        f"Precision: {precision}",
        f"Accuracy: {accuracy}",
        f"Balanced accuracy: {balanced_accuracy}",
        "",
        (
            "Lowest successful TP rate kg/h: "
            f"{tp_rates.min() if not tp_rates.empty else np.nan}"
        ),
        (
            "Highest FN rate kg/h: "
            f"{fn_rates.max() if not fn_rates.empty else np.nan}"
        ),
        "",
        "Performance by emission bin:",
        bin_performance.to_string(index=False),
        "",
        "TP quantification summary:",
        (
            tp_quantification[
                [
                    "quantification_error_median_kg_hr",
                    "quantification_error_percent",
                ]
            ].describe().to_string()
            if not tp_quantification.empty
            else "No valid quantification pairs."
        ),
        "",
        "Important interpretation:",
        (
            "The primary performance calculation includes "
            "TP, FN, TN, and FP only."
        ),
        (
            "NE, ER_MIS, and ER_FAQ are retained in the manifest "
            "but excluded from primary detection performance."
        ),
        (
            "Observations lacking datetime_UTC may still be used "
            "for GHGSat internal detection performance, but their "
            "cross-sensor temporal matching evidence is weaker."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 115)
    print(
        "GHGSAT OBSERVATION-LEVEL PERFORMANCE"
    )
    print("=" * 115)

    print("\nIndependent observations:", len(manifest))

    print(
        "Observations requiring structural review:",
        review_count,
    )

    print("\nObservation-level classifications:")
    print(class_summary)

    print("\nAnalysis roles:")
    print(role_summary)

    print("\nAcquisition-time source:")
    print(acquisition_source_summary)

    print(
        "\nObservations sharing the same "
        "best acquisition time:",
        shared_acquisition_count,
    )

    print("\nPrimary confusion counts:")
    print(confusion)

    print("\nRecall:", recall)
    print("Specificity:", specificity)
    print(
        "False-positive rate:",
        false_positive_rate,
    )
    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print(
        "Balanced accuracy:",
        balanced_accuracy,
    )

    print(
        "\nLowest successful TP rate kg/h:",
        (
            tp_rates.min()
            if not tp_rates.empty
            else np.nan
        ),
    )

    print(
        "Highest FN rate kg/h:",
        (
            fn_rates.max()
            if not fn_rates.empty
            else np.nan
        ),
    )

    print("\nPerformance by emission bin:")
    print(
        bin_performance.to_string(
            index=False
        )
    )

    print("\nTP quantification summary:")
    if tp_quantification.empty:
        print(
            "No valid quantification pairs."
        )
    else:
        print(
            tp_quantification[
                [
                    "quantification_error_median_kg_hr",
                    "quantification_error_percent",
                ]
            ].describe()
        )

    print("\nSaved:")
    print(MANIFEST_OUTPUT)
    print(PRIMARY_OUTPUT)
    print(BIN_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
