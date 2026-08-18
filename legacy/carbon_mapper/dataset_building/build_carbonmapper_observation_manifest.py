from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


INPUT = Path(
    "raw_data/2023_Controlled_Release_2021/"
    "Dataframes for Stanford analysis/"
    "matchedDF_CarbonMapper_23822.csv"
)

MANIFEST_OUTPUT = Path(
    "outputs/494_carbonmapper_observation_manifest_v1.csv"
)

REVIEW_OUTPUT = Path(
    "outputs/495_carbonmapper_observation_review_v1.csv"
)

BIN_OUTPUT = Path(
    "outputs/496_carbonmapper_detection_performance_by_emission_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/497_carbonmapper_observation_level_report_v1.txt"
)


GROUND_TRUTH_RATE_PREFERENCE = [
    "cr_kgh_CH4_mean60",
    "cr_kgh_CH4_mean300",
    "cr_kgh_CH4_mean90",
    "cr_kgh_CH4_mean30",
    "cr_kgh_CH4_mean600",
    "cr_kgh_CH4_mean900",
    "cr_kgh",
]

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
    "NS": "exclude_plume_not_steady",
    "NE": "exclude_plume_not_established",
    "ER_FAL": "exclude_quality_control_error",
}


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


def choose_ground_truth_rate_column(frame):
    for column in GROUND_TRUTH_RATE_PREFERENCE:
        if column not in frame.columns:
            continue

        numeric = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if numeric.notna().any():
            return column

    raise KeyError(
        "No controlled-release kg/h column found. "
        "Tried: "
        + ", ".join(
            GROUND_TRUTH_RATE_PREFERENCE
        )
    )


def make_observation_id(timestamp, cr_idx):
    raw = f"{timestamp}|{cr_idx}"

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:12]

    return f"CM_OBS_{digest}"


def classify_emission(rate):
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


def safe_ratio(maximum, minimum):
    if (
        pd.isna(maximum)
        or pd.isna(minimum)
        or minimum <= 0
    ):
        return np.nan

    return maximum / minimum


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    frame = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required = [
        "datetime_UTC",
        "cr_idx",
        "tc_Classification",
        "Detection",
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

    ground_truth_rate_column = (
        choose_ground_truth_rate_column(
            frame
        )
    )

    print("=" * 115)
    print(
        "CARBON MAPPER OBSERVATION-LEVEL MANIFEST"
    )
    print("=" * 115)

    print("\nInput rows:", len(frame))
    print(
        "Selected ground-truth rate column:",
        ground_truth_rate_column,
    )

    frame["_datetime_utc"] = pd.to_datetime(
        frame["datetime_UTC"],
        errors="coerce",
        utc=True,
    )

    frame["_operator_timestamp"] = pd.to_datetime(
        frame.get("Operator_Timestamp"),
        errors="coerce",
        utc=True,
    )

    frame["_stanford_timestamp"] = pd.to_datetime(
        frame.get("Stanford_timestamp"),
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

    frame["_ground_truth_rate"] = pd.to_numeric(
        frame[ground_truth_rate_column],
        errors="coerce",
    )

    frame["_reported_rate"] = pd.to_numeric(
        frame["FacilityEmissionRate"],
        errors="coerce",
    )

    if "FacilityEmissionRateLower" in frame.columns:
        frame["_reported_lower"] = pd.to_numeric(
            frame["FacilityEmissionRateLower"],
            errors="coerce",
        )
    else:
        frame["_reported_lower"] = np.nan

    if "FacilityEmissionRateUpper" in frame.columns:
        frame["_reported_upper"] = pd.to_numeric(
            frame["FacilityEmissionRateUpper"],
            errors="coerce",
        )
    else:
        frame["_reported_upper"] = np.nan

    frame = frame.dropna(
        subset=["_datetime_utc"]
    ).copy()

    records = []

    for timestamp, group in frame.groupby(
        "_datetime_utc",
        sort=True,
    ):
        class_values = unique_values(
            group["tc_Classification"]
        )

        detection_values = sorted(
            group["_detection"]
            .dropna()
            .unique()
            .tolist()
        )

        cr_idx_values = sorted(
            group["_cr_idx"]
            .dropna()
            .unique()
            .tolist()
        )

        if len(class_values) == 1:
            classification = class_values[0]
        else:
            classification = ""

        if len(detection_values) == 1:
            numeric_detection = (
                detection_values[0]
            )
        else:
            numeric_detection = np.nan

        if len(cr_idx_values) == 1:
            selected_cr_idx = (
                cr_idx_values[0]
            )
        else:
            selected_cr_idx = np.nan

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

        ground_truth_min = (
            ground_truth_rates.min()
            if not ground_truth_rates.empty
            else np.nan
        )

        ground_truth_median = (
            ground_truth_rates.median()
            if not ground_truth_rates.empty
            else np.nan
        )

        ground_truth_max = (
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

        mixed_classification = (
            len(class_values) != 1
        )

        mixed_detection = (
            len(detection_values) != 1
        )

        multiple_cr_idx = (
            len(cr_idx_values) != 1
        )

        unusual_row_count = (
            len(group) not in {1, 2, 3}
        )

        review_reasons = []

        if mixed_classification:
            review_reasons.append(
                "mixed_tc_classification"
            )

        if mixed_detection:
            review_reasons.append(
                "mixed_numeric_detection"
            )

        if multiple_cr_idx:
            review_reasons.append(
                "multiple_cr_idx"
            )

        if unusual_row_count:
            review_reasons.append(
                "unusual_row_count"
            )

        rate_ratio = safe_ratio(
            ground_truth_max,
            ground_truth_min,
        )

        if (
            pd.notna(rate_ratio)
            and rate_ratio > 1.10
        ):
            review_reasons.append(
                "ground_truth_rate_variation_over_10pct"
            )

        analysis_role = ANALYSIS_ROLE_MAP.get(
            classification,
            "manual_review_required",
        )

        primary_evaluable = (
            classification in PRIMARY_CLASSES
        )

        sensor_reported_detection = (
            classification in {"TP", "FP"}
        )

        if classification in {"TP", "FN", "NS"}:
            release_present_from_class = True
        elif classification in {"TN", "FP"}:
            release_present_from_class = False
        else:
            release_present_from_class = np.nan

        observation_id = make_observation_id(
            timestamp.isoformat(),
            selected_cr_idx,
        )

        records.append({
            "carbonmapper_observation_id":
                observation_id,

            "observation_time_utc":
                timestamp,

            "operator_timestamp_min":
                group[
                    "_operator_timestamp"
                ].min(),

            "operator_timestamp_max":
                group[
                    "_operator_timestamp"
                ].max(),

            "stanford_timestamp_min":
                group[
                    "_stanford_timestamp"
                ].min(),

            "stanford_timestamp_max":
                group[
                    "_stanford_timestamp"
                ].max(),

            "source_row_count":
                len(group),

            "source_row_indices":
                " | ".join(
                    str(index)
                    for index in group.index
                ),

            "cr_idx":
                selected_cr_idx,

            "cr_idx_values":
                " | ".join(
                    str(value)
                    for value in cr_idx_values
                ),

            "tc_classification":
                classification,

            "tc_classification_values":
                " | ".join(class_values),

            "numeric_detection":
                numeric_detection,

            "numeric_detection_values":
                " | ".join(
                    str(value)
                    for value in detection_values
                ),

            "analysis_role":
                analysis_role,

            "primary_evaluable":
                primary_evaluable,

            "sensor_reported_detection":
                sensor_reported_detection,

            "release_present_from_class":
                release_present_from_class,

            "ground_truth_rate_column":
                ground_truth_rate_column,

            "ground_truth_rate_min_kg_hr":
                ground_truth_min,

            "ground_truth_rate_median_kg_hr":
                ground_truth_median,

            "ground_truth_rate_max_kg_hr":
                ground_truth_max,

            "ground_truth_rate_ratio":
                rate_ratio,

            "emission_bin":
                classify_emission(
                    ground_truth_median
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

            "wind_types":
                (
                    unique_text(
                        group["WindType"]
                    )
                    if "WindType"
                    in group.columns
                    else ""
                ),

            "wind_type_count":
                (
                    group["WindType"]
                    .dropna()
                    .nunique()
                    if "WindType"
                    in group.columns
                    else 0
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
                        group["PlumeSteady"]
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
            "observation_time_utc",
            "carbonmapper_observation_id",
        ]
    ).reset_index(drop=True)

    manifest.to_csv(
        MANIFEST_OUTPUT,
        index=False,
    )

    review = manifest[
        manifest[
            "manual_review_required"
        ]
    ].copy()

    review.to_csv(
        REVIEW_OUTPUT,
        index=False,
    )

    primary = manifest[
        manifest[
            "primary_evaluable"
        ]
    ].copy()

    confusion_counts = (
        primary[
            "tc_classification"
        ]
        .value_counts()
        .reindex(
            ["TP", "FN", "TN", "FP"],
            fill_value=0,
        )
    )

    tp = int(confusion_counts["TP"])
    fn = int(confusion_counts["FN"])
    tn = int(confusion_counts["TN"])
    fp = int(confusion_counts["FP"])

    positive_denominator = tp + fn
    negative_denominator = tn + fp

    recall = (
        tp / positive_denominator
        if positive_denominator
        else np.nan
    )

    specificity = (
        tn / negative_denominator
        if negative_denominator
        else np.nan
    )

    false_positive_rate = (
        fp / negative_denominator
        if negative_denominator
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

    positive_primary = primary[
        primary[
            "tc_classification"
        ].isin(["TP", "FN"])
    ].copy()

    bin_order = [
        "0_to_200",
        "200_to_500",
        "500_to_1000",
        "above_1000",
        "unknown",
    ]

    bin_records = []

    for emission_bin in bin_order:
        group = positive_primary[
            positive_primary[
                "emission_bin"
            ].eq(emission_bin)
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
                emission_bin,

            "positive_observations":
                denominator,

            "true_positive_observations":
                bin_tp,

            "false_negative_observations":
                bin_fn,

            "detection_recall":
                (
                    bin_tp / denominator
                    if denominator
                    else np.nan
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

    class_summary = (
        manifest[
            "tc_classification"
        ]
        .replace("", "missing_or_mixed")
        .value_counts(
            dropna=False
        )
    )

    role_summary = (
        manifest["analysis_role"]
        .value_counts(
            dropna=False
        )
    )

    tp_rates = manifest.loc[
        manifest[
            "tc_classification"
        ].eq("TP"),
        "ground_truth_rate_median_kg_hr",
    ].dropna()

    fn_rates = manifest.loc[
        manifest[
            "tc_classification"
        ].eq("FN"),
        "ground_truth_rate_median_kg_hr",
    ].dropna()

    report_lines = [
        "=" * 115,
        "CARBON MAPPER OBSERVATION-LEVEL PERFORMANCE V1",
        "=" * 115,
        "",
        f"Input source rows: {len(frame)}",
        (
            "Independent observation timestamps: "
            f"{len(manifest)}"
        ),
        (
            "Selected ground-truth rate column: "
            f"{ground_truth_rate_column}"
        ),
        (
            "Observations requiring structural review: "
            f"{len(review)}"
        ),
        "",
        "Observation-level tc_Classification:",
        class_summary.to_string(),
        "",
        "Analysis roles:",
        role_summary.to_string(),
        "",
        "Primary confusion counts:",
        confusion_counts.to_string(),
        "",
        f"Detection recall TP/(TP+FN): {recall}",
        f"Specificity TN/(TN+FP): {specificity}",
        f"False-positive rate FP/(TN+FP): {false_positive_rate}",
        f"Balanced accuracy: {balanced_accuracy}",
        "",
        (
            "Lowest successful TP ground-truth rate kg/h: "
            f"{tp_rates.min() if not tp_rates.empty else np.nan}"
        ),
        (
            "Highest FN ground-truth rate kg/h: "
            f"{fn_rates.max() if not fn_rates.empty else np.nan}"
        ),
        "",
        "Performance by emission bin:",
        bin_performance.to_string(index=False),
        "",
        "Important interpretation:",
        (
            "Numeric Detection is not used as the binary "
            "sensor-detection target because Detection=0 "
            "contains FN, TN, FP, and NS. The authoritative "
            "performance label is tc_Classification."
        ),
        (
            "NE, NS, and ER_FAL are excluded from the primary "
            "classification performance calculation."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nIndependent observations:", len(manifest))
    print(
        "Observations requiring structural review:",
        len(review),
    )

    print("\nObservation-level classifications:")
    print(class_summary)

    print("\nAnalysis roles:")
    print(role_summary)

    print("\nPrimary confusion counts:")
    print(confusion_counts)

    print(
        "\nDetection recall TP/(TP+FN):",
        recall,
    )

    print(
        "Specificity TN/(TN+FP):",
        specificity,
    )

    print(
        "False-positive rate FP/(TN+FP):",
        false_positive_rate,
    )

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

    print("\nSaved:")
    print(MANIFEST_OUTPUT)
    print(REVIEW_OUTPUT)
    print(BIN_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
