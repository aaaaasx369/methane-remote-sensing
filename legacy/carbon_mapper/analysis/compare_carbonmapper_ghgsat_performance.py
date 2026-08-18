from pathlib import Path
import math

import numpy as np
import pandas as pd


CARBONMAPPER_INPUT = Path(
    "outputs/498_carbonmapper_observation_manifest_locked_v1.csv"
)

GHGSAT_INPUT = Path(
    "outputs/521_ghgsat_observation_manifest_v1.csv"
)

OBSERVATION_OUTPUT = Path(
    "outputs/525_methane_specific_sensor_observations_v1.csv"
)

PERFORMANCE_OUTPUT = Path(
    "outputs/526_methane_specific_sensor_performance_summary_v1.csv"
)

BIN_OUTPUT = Path(
    "outputs/527_methane_specific_sensor_emission_bin_summary_v1.csv"
)

QUANTIFICATION_OUTPUT = Path(
    "outputs/528_methane_specific_sensor_quantification_summary_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/529_methane_specific_sensor_comparison_report_v1.txt"
)


BIN_ORDER = [
    "0_to_200",
    "200_to_500",
    "500_to_1000",
    "above_1000",
    "unknown",
]


def parse_boolean(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0", "yes"])
    )


def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan

    return numerator / denominator


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return np.nan, np.nan

    proportion = successes / total

    denominator = (
        1
        + z ** 2 / total
    )

    centre = (
        proportion
        + z ** 2
        / (2 * total)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            proportion
            * (1 - proportion)
            / total
            + z ** 2
            / (4 * total ** 2)
        )
        / denominator
    )

    return (
        max(0.0, centre - half_width),
        min(1.0, centre + half_width),
    )


def bootstrap_median_interval(
    values,
    repetitions=10000,
    seed=20260723,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    medians = np.empty(
        repetitions,
        dtype=float,
    )

    for index in range(repetitions):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        medians[index] = np.median(
            sample
        )

    return (
        float(
            np.percentile(
                medians,
                2.5,
            )
        ),
        float(
            np.percentile(
                medians,
                97.5,
            )
        ),
    )


def normalize_sensor(
    frame,
    sensor,
    observation_id_column,
):
    required = [
        observation_id_column,
        "tc_classification",
        "analysis_role",
        "ground_truth_rate_median_kg_hr",
        "reported_rate_median_kg_hr",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            f"{sensor} missing columns: "
            + ", ".join(missing)
        )

    output = pd.DataFrame({
        "sensor":
            sensor,

        "observation_id":
            frame[
                observation_id_column
            ].astype(str),

        "tc_classification":
            frame[
                "tc_classification"
            ].astype(str),

        "analysis_role":
            frame[
                "analysis_role"
            ].astype(str),

        "ground_truth_rate_kg_hr":
            pd.to_numeric(
                frame[
                    "ground_truth_rate_median_kg_hr"
                ],
                errors="coerce",
            ),

        "reported_rate_kg_hr":
            pd.to_numeric(
                frame[
                    "reported_rate_median_kg_hr"
                ],
                errors="coerce",
            ),

        "emission_bin":
            frame.get(
                "emission_bin",
                "unknown",
            ),
    })

    if "locked_for_analysis" in frame.columns:
        output[
            "locked_for_analysis"
        ] = parse_boolean(
            frame[
                "locked_for_analysis"
            ]
        )
    else:
        output[
            "locked_for_analysis"
        ] = True

    if (
        "best_available_acquisition_time_utc"
        in frame.columns
    ):
        output[
            "observation_time_utc"
        ] = frame[
            "best_available_acquisition_time_utc"
        ]

        output[
            "acquisition_time_source"
        ] = frame.get(
            "acquisition_time_source",
            "",
        )

    elif "observation_time_utc" in frame.columns:
        output[
            "observation_time_utc"
        ] = frame[
            "observation_time_utc"
        ]

        output[
            "acquisition_time_source"
        ] = "datetime_UTC"

    else:
        output[
            "observation_time_utc"
        ] = pd.NaT

        output[
            "acquisition_time_source"
        ] = ""

    output[
        "primary_evaluable"
    ] = (
        output[
            "analysis_role"
        ].eq("primary_evaluable")
        & output[
            "locked_for_analysis"
        ]
    )

    output[
        "release_present"
    ] = np.select(
        [
            output[
                "tc_classification"
            ].isin(["TP", "FN"]),

            output[
                "tc_classification"
            ].isin(["TN", "FP"]),
        ],
        [
            True,
            False,
        ],
        default=np.nan,
    )

    output[
        "sensor_detected"
    ] = output[
        "tc_classification"
    ].isin(["TP", "FP"])

    output[
        "quantification_error_kg_hr"
    ] = (
        output[
            "reported_rate_kg_hr"
        ]
        - output[
            "ground_truth_rate_kg_hr"
        ]
    )

    output[
        "quantification_error_percent"
    ] = (
        100
        * output[
            "quantification_error_kg_hr"
        ]
        / output[
            "ground_truth_rate_kg_hr"
        ]
    )

    return output


def calculate_performance(sensor_frame):
    primary = sensor_frame[
        sensor_frame[
            "primary_evaluable"
        ]
    ].copy()

    counts = (
        primary[
            "tc_classification"
        ]
        .value_counts()
        .reindex(
            ["TP", "FN", "TN", "FP"],
            fill_value=0,
        )
    )

    tp = int(counts["TP"])
    fn = int(counts["FN"])
    tn = int(counts["TN"])
    fp = int(counts["FP"])

    positive_total = tp + fn
    negative_total = tn + fp

    recall = safe_divide(
        tp,
        positive_total,
    )

    specificity = safe_divide(
        tn,
        negative_total,
    )

    recall_low, recall_high = (
        wilson_interval(
            tp,
            positive_total,
        )
    )

    specificity_low, specificity_high = (
        wilson_interval(
            tn,
            negative_total,
        )
    )

    balanced_accuracy = (
        (recall + specificity) / 2
        if (
            pd.notna(recall)
            and pd.notna(specificity)
        )
        else np.nan
    )

    positive = primary[
        primary[
            "tc_classification"
        ].isin(["TP", "FN"])
    ]

    tp_rates = positive.loc[
        positive[
            "tc_classification"
        ].eq("TP"),
        "ground_truth_rate_kg_hr",
    ].dropna()

    fn_rates = positive.loc[
        positive[
            "tc_classification"
        ].eq("FN"),
        "ground_truth_rate_kg_hr",
    ].dropna()

    return {
        "sensor":
            primary["sensor"].iloc[0],

        "primary_observations":
            len(primary),

        "tp":
            tp,

        "fn":
            fn,

        "tn":
            tn,

        "fp":
            fp,

        "positive_observations":
            positive_total,

        "negative_observations":
            negative_total,

        "recall":
            recall,

        "recall_ci95_low":
            recall_low,

        "recall_ci95_high":
            recall_high,

        "specificity":
            specificity,

        "specificity_ci95_low":
            specificity_low,

        "specificity_ci95_high":
            specificity_high,

        "false_positive_rate":
            safe_divide(
                fp,
                negative_total,
            ),

        "precision":
            safe_divide(
                tp,
                tp + fp,
            ),

        "accuracy":
            safe_divide(
                tp + tn,
                len(primary),
            ),

        "balanced_accuracy":
            balanced_accuracy,

        "lowest_successful_tp_rate_kg_hr":
            (
                tp_rates.min()
                if not tp_rates.empty
                else np.nan
            ),

        "highest_fn_rate_kg_hr":
            (
                fn_rates.max()
                if not fn_rates.empty
                else np.nan
            ),

        "all_evaluable_positive_above_200_detected":
            bool(
                positive.loc[
                    positive[
                        "ground_truth_rate_kg_hr"
                    ].ge(200),
                    "tc_classification",
                ].eq("TP").all()
            ),
    }


def calculate_bins(combined):
    records = []

    for sensor, sensor_frame in combined.groupby(
        "sensor"
    ):
        positive = sensor_frame[
            sensor_frame[
                "primary_evaluable"
            ]
            & sensor_frame[
                "tc_classification"
            ].isin(["TP", "FN"])
        ].copy()

        for emission_bin in BIN_ORDER:
            group = positive[
                positive[
                    "emission_bin"
                ].eq(emission_bin)
            ]

            tp = int(
                group[
                    "tc_classification"
                ].eq("TP").sum()
            )

            fn = int(
                group[
                    "tc_classification"
                ].eq("FN").sum()
            )

            total = tp + fn

            low, high = wilson_interval(
                tp,
                total,
            )

            records.append({
                "sensor":
                    sensor,

                "emission_bin":
                    emission_bin,

                "positive_observations":
                    total,

                "tp":
                    tp,

                "fn":
                    fn,

                "recall":
                    safe_divide(
                        tp,
                        total,
                    ),

                "recall_ci95_low":
                    low,

                "recall_ci95_high":
                    high,

                "minimum_rate_kg_hr":
                    group[
                        "ground_truth_rate_kg_hr"
                    ].min(),

                "median_rate_kg_hr":
                    group[
                        "ground_truth_rate_kg_hr"
                    ].median(),

                "maximum_rate_kg_hr":
                    group[
                        "ground_truth_rate_kg_hr"
                    ].max(),
            })

    return pd.DataFrame(records)


def calculate_quantification(combined):
    records = []

    for sensor, sensor_frame in combined.groupby(
        "sensor"
    ):
        quantifiable = sensor_frame[
            sensor_frame[
                "primary_evaluable"
            ]
            & sensor_frame[
                "tc_classification"
            ].eq("TP")
            & sensor_frame[
                "ground_truth_rate_kg_hr"
            ].gt(0)
            & sensor_frame[
                "reported_rate_kg_hr"
            ].notna()
        ].copy()

        percent_error = quantifiable[
            "quantification_error_percent"
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        absolute_percent_error = (
            percent_error.abs()
        )

        median_low, median_high = (
            bootstrap_median_interval(
                percent_error
            )
        )

        records.append({
            "sensor":
                sensor,

            "quantifiable_tp_observations":
                len(quantifiable),

            "mean_error_kg_hr":
                quantifiable[
                    "quantification_error_kg_hr"
                ].mean(),

            "median_error_kg_hr":
                quantifiable[
                    "quantification_error_kg_hr"
                ].median(),

            "mean_error_percent":
                percent_error.mean(),

            "median_error_percent":
                percent_error.median(),

            "median_error_percent_ci95_low":
                median_low,

            "median_error_percent_ci95_high":
                median_high,

            "median_absolute_percentage_error":
                absolute_percent_error.median(),

            "mean_absolute_percentage_error":
                absolute_percent_error.mean(),

            "fraction_underestimating":
                percent_error.lt(0).mean(),

            "minimum_error_percent":
                percent_error.min(),

            "maximum_error_percent":
                percent_error.max(),
        })

    return pd.DataFrame(records)


def main():
    if not CARBONMAPPER_INPUT.exists():
        raise FileNotFoundError(
            CARBONMAPPER_INPUT
        )

    if not GHGSAT_INPUT.exists():
        raise FileNotFoundError(
            GHGSAT_INPUT
        )

    carbonmapper_raw = pd.read_csv(
        CARBONMAPPER_INPUT,
        low_memory=False,
    )

    ghgsat_raw = pd.read_csv(
        GHGSAT_INPUT,
        low_memory=False,
    )

    carbonmapper = normalize_sensor(
        carbonmapper_raw,
        sensor="Carbon Mapper",
        observation_id_column=(
            "carbonmapper_observation_id"
        ),
    )

    ghgsat = normalize_sensor(
        ghgsat_raw,
        sensor="GHGSat",
        observation_id_column=(
            "ghgsat_observation_id"
        ),
    )

    combined = pd.concat(
        [
            carbonmapper,
            ghgsat,
        ],
        ignore_index=True,
    )

    combined.to_csv(
        OBSERVATION_OUTPUT,
        index=False,
    )

    performance = pd.DataFrame([
        calculate_performance(
            carbonmapper
        ),
        calculate_performance(
            ghgsat
        ),
    ])

    performance.to_csv(
        PERFORMANCE_OUTPUT,
        index=False,
    )

    bin_summary = calculate_bins(
        combined
    )

    bin_summary.to_csv(
        BIN_OUTPUT,
        index=False,
    )

    quantification = (
        calculate_quantification(
            combined
        )
    )

    quantification.to_csv(
        QUANTIFICATION_OUTPUT,
        index=False,
    )

    report_lines = [
        "=" * 115,
        "METHANE-SPECIFIC SENSOR COMPARISON V1",
        "=" * 115,
        "",
        "Detection performance:",
        performance.to_string(index=False),
        "",
        "Performance by emission bin:",
        bin_summary.to_string(index=False),
        "",
        "Quantification performance for TP observations:",
        quantification.to_string(index=False),
        "",
        "Interpretation:",
        (
            "This is a descriptive comparison. Carbon Mapper "
            "and GHGSat were not necessarily evaluated under "
            "identical acquisition, wind, surface, and quality-"
            "control conditions."
        ),
        (
            "The absence of false positives is based on only "
            "30 Carbon Mapper and 17 GHGSat evaluable negative "
            "observations."
        ),
        (
            "Perfect recall above 200 kg/h in these samples "
            "must not be interpreted as a universal detection "
            "threshold."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 115)
    print(
        "METHANE-SPECIFIC SENSOR COMPARISON"
    )
    print("=" * 115)

    print("\nDetection performance:")
    print(
        performance[
            [
                "sensor",
                "primary_observations",
                "tp",
                "fn",
                "tn",
                "fp",
                "recall",
                "recall_ci95_low",
                "recall_ci95_high",
                "specificity",
                "balanced_accuracy",
                "lowest_successful_tp_rate_kg_hr",
                "highest_fn_rate_kg_hr",
            ]
        ].to_string(index=False)
    )

    print("\nPerformance by emission bin:")
    print(
        bin_summary.to_string(
            index=False
        )
    )

    print("\nQuantification performance:")
    print(
        quantification.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(OBSERVATION_OUTPUT)
    print(PERFORMANCE_OUTPUT)
    print(BIN_OUTPUT)
    print(QUANTIFICATION_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
