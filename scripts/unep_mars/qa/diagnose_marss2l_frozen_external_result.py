from pathlib import Path

import numpy as np
import pandas as pd


PREDICTION_INPUT = Path(
    "outputs/238_marss2l_frozen_external_predictions.csv"
)

MODEL_INPUT = Path(
    "outputs/243_marss2l_frozen_model_contract.csv"
)

SCORE_MAPPING_OUTPUT = Path(
    "outputs/245_marss2l_percentile_score_mapping.csv"
)

ERROR_CASE_OUTPUT = Path(
    "outputs/246_marss2l_external_error_cases.csv"
)

SITE_OUTPUT = Path(
    "outputs/247_marss2l_external_site_diagnostics.csv"
)

COUNTRY_OUTPUT = Path(
    "outputs/248_marss2l_external_country_diagnostics.csv"
)

QA_OUTPUT = Path(
    "outputs/249_marss2l_external_qa_diagnostics.csv"
)


FEATURE = (
    "cal_temporal_z_source_p95_percentile"
)


def safe_rate(numerator, denominator):
    if denominator == 0:
        return np.nan

    return numerator / denominator


def main():
    if not PREDICTION_INPUT.exists():
        raise FileNotFoundError(
            PREDICTION_INPUT
        )

    df = pd.read_csv(
        PREDICTION_INPUT,
        low_memory=False,
    )

    required = [
        "site_key",
        "external_role",
        "actual_label",
        "predicted_label",
        "prediction_score",
        FEATURE,
        "sensor_code",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing columns: {missing}"
        )

    df["actual_label"] = pd.to_numeric(
        df["actual_label"],
        errors="raise",
    ).astype(int)

    df["predicted_label"] = pd.to_numeric(
        df["predicted_label"],
        errors="raise",
    ).astype(int)

    df["prediction_score"] = pd.to_numeric(
        df["prediction_score"],
        errors="coerce",
    )

    df[FEATURE] = pd.to_numeric(
        df[FEATURE],
        errors="coerce",
    )

    df["error_type"] = np.select(
        [
            df["actual_label"].eq(1)
            & df["predicted_label"].eq(1),

            df["actual_label"].eq(1)
            & df["predicted_label"].eq(0),

            df["actual_label"].eq(0)
            & df["predicted_label"].eq(1),

            df["actual_label"].eq(0)
            & df["predicted_label"].eq(0),
        ],
        [
            "true_positive",
            "false_negative",
            "false_positive",
            "true_negative",
        ],
        default="unknown",
    )

    # ========================================================
    # Percentile → probability → prediction mapping
    # ========================================================

    score_mapping = (
        df.groupby(
            FEATURE,
            dropna=False,
        )
        .agg(
            sample_count=(
                "actual_label",
                "size",
            ),
            negative_count=(
                "actual_label",
                lambda values:
                    int((values == 0).sum()),
            ),
            positive_count=(
                "actual_label",
                lambda values:
                    int((values == 1).sum()),
            ),
            observed_positive_rate=(
                "actual_label",
                "mean",
            ),
            median_prediction_score=(
                "prediction_score",
                "median",
            ),
            minimum_prediction_score=(
                "prediction_score",
                "min",
            ),
            maximum_prediction_score=(
                "prediction_score",
                "max",
            ),
            predicted_positive_rate=(
                "predicted_label",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(FEATURE)
    )

    score_mapping.to_csv(
        SCORE_MAPPING_OUTPUT,
        index=False,
    )

    # ========================================================
    # Error cases
    # ========================================================

    error_columns = [
        "download_id",
        "site_key",
        "country",
        "location_name",
        "sensor_code",
        "external_role",
        "actual_label",
        "predicted_label",
        "error_type",
        FEATURE,
        "prediction_score",
        "temporal_z_source_p95",
        "release_rate_kg_h",
        "qa_clear_fraction",
        "same_sensor_background_count",
        "acquisition_datetime_utc",
        "landsat_tile",
        "patch_path",
        "plume",
    ]

    error_columns = [
        column
        for column in error_columns
        if column in df.columns
    ]

    errors = df[
        df["actual_label"]
        .ne(df["predicted_label"])
    ][error_columns].copy()

    errors = errors.sort_values(
        [
            "error_type",
            "prediction_score",
        ],
        ascending=[
            True,
            False,
        ],
        na_position="last",
    )

    errors.to_csv(
        ERROR_CASE_OUTPUT,
        index=False,
    )

    # ========================================================
    # Site diagnostics
    # ========================================================

    site_rows = []

    for site_key, group in df.groupby(
        "site_key"
    ):
        negative = group[
            group["actual_label"].eq(0)
        ]

        positive = group[
            group["actual_label"].eq(1)
        ]

        tp = int(
            (
                positive[
                    "predicted_label"
                ] == 1
            ).sum()
        )

        fn = int(
            (
                positive[
                    "predicted_label"
                ] == 0
            ).sum()
        )

        fp = int(
            (
                negative[
                    "predicted_label"
                ] == 1
            ).sum()
        )

        tn = int(
            (
                negative[
                    "predicted_label"
                ] == 0
            ).sum()
        )

        row = {
            "site_key":
                site_key,
            "country":
                group["country"].iloc[0]
                if "country" in group.columns
                else "",
            "location_name":
                group["location_name"].iloc[0]
                if "location_name"
                in group.columns
                else "",
            "negative_count":
                len(negative),
            "positive_count":
                len(positive),
            "true_negative":
                tn,
            "false_positive":
                fp,
            "false_negative":
                fn,
            "true_positive":
                tp,
            "positive_recall":
                safe_rate(
                    tp,
                    tp + fn,
                ),
            "negative_recall":
                safe_rate(
                    tn,
                    tn + fp,
                ),
            "false_positive_rate":
                safe_rate(
                    fp,
                    tn + fp,
                ),
            "median_positive_score":
                positive[
                    "prediction_score"
                ].median(),
            "median_negative_score":
                negative[
                    "prediction_score"
                ].median(),
            "median_positive_feature":
                positive[
                    FEATURE
                ].median(),
            "median_negative_feature":
                negative[
                    FEATURE
                ].median(),
            "minimum_qa_clear_fraction":
                group[
                    "qa_clear_fraction"
                ].min()
                if "qa_clear_fraction"
                in group.columns
                else np.nan,
        }

        site_rows.append(row)

    site_diagnostics = pd.DataFrame(
        site_rows
    )

    site_diagnostics[
        "site_balanced_accuracy"
    ] = (
        site_diagnostics[
            "positive_recall"
        ]
        + site_diagnostics[
            "negative_recall"
        ]
    ) / 2

    site_diagnostics = (
        site_diagnostics.sort_values(
            [
                "site_balanced_accuracy",
                "positive_count",
            ],
            ascending=[
                True,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    site_diagnostics.to_csv(
        SITE_OUTPUT,
        index=False,
    )

    # ========================================================
    # Country diagnostics
    # ========================================================

    country_rows = []

    if "country" in df.columns:
        for country, group in df.groupby(
            "country",
            dropna=False,
        ):
            negative = group[
                group["actual_label"].eq(0)
            ]

            positive = group[
                group["actual_label"].eq(1)
            ]

            tp = int(
                (
                    positive[
                        "predicted_label"
                    ] == 1
                ).sum()
            )

            fn = int(
                (
                    positive[
                        "predicted_label"
                    ] == 0
                ).sum()
            )

            fp = int(
                (
                    negative[
                        "predicted_label"
                    ] == 1
                ).sum()
            )

            tn = int(
                (
                    negative[
                        "predicted_label"
                    ] == 0
                ).sum()
            )

            country_rows.append({
                "country":
                    country,
                "site_count":
                    group[
                        "site_key"
                    ].nunique(),
                "negative_count":
                    len(negative),
                "positive_count":
                    len(positive),
                "true_negative":
                    tn,
                "false_positive":
                    fp,
                "false_negative":
                    fn,
                "true_positive":
                    tp,
                "positive_recall":
                    safe_rate(
                        tp,
                        tp + fn,
                    ),
                "negative_recall":
                    safe_rate(
                        tn,
                        tn + fp,
                    ),
                "false_positive_rate":
                    safe_rate(
                        fp,
                        tn + fp,
                    ),
            })

    country_diagnostics = pd.DataFrame(
        country_rows
    )

    if not country_diagnostics.empty:
        country_diagnostics = (
            country_diagnostics.sort_values(
                [
                    "positive_count",
                    "site_count",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
        )

    country_diagnostics.to_csv(
        COUNTRY_OUTPUT,
        index=False,
    )

    # ========================================================
    # QA versus correctness
    # ========================================================

    if "qa_clear_fraction" in df.columns:
        qa_diagnostics = (
            df.groupby(
                "error_type"
            )
            .agg(
                sample_count=(
                    "actual_label",
                    "size",
                ),
                qa_clear_mean=(
                    "qa_clear_fraction",
                    "mean",
                ),
                qa_clear_median=(
                    "qa_clear_fraction",
                    "median",
                ),
                qa_clear_minimum=(
                    "qa_clear_fraction",
                    "min",
                ),
                temporal_valid_median=(
                    "temporal_valid_fraction",
                    "median",
                ),
                source_valid_median=(
                    "source_valid_fraction",
                    "median",
                ),
                median_prediction_score=(
                    "prediction_score",
                    "median",
                ),
            )
            .reset_index()
        )
    else:
        qa_diagnostics = pd.DataFrame()

    qa_diagnostics.to_csv(
        QA_OUTPUT,
        index=False,
    )

    print("=" * 110)
    print("MARS-S2L EXTERNAL ERROR DIAGNOSTICS")
    print("=" * 110)

    print("\nPercentile → score mapping:")
    print(
        score_mapping.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nError types:")
    print(
        df["error_type"]
        .value_counts()
    )

    print("\nWorst ten sites:")
    print(
        site_diagnostics[
            [
                "site_key",
                "country",
                "positive_count",
                "true_positive",
                "false_negative",
                "true_negative",
                "false_positive",
                "positive_recall",
                "negative_recall",
                "site_balanced_accuracy",
            ]
        ].head(10).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nQA by error type:")
    print(
        qa_diagnostics.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    if MODEL_INPUT.exists():
        model_contract = pd.read_csv(
            MODEL_INPUT,
            low_memory=False,
        )

        print("\nFrozen model contract:")
        print(
            model_contract.to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.6f}",
            )
        )

    print("\nSaved:")
    print(SCORE_MAPPING_OUTPUT)
    print(ERROR_CASE_OUTPUT)
    print(SITE_OUTPUT)
    print(COUNTRY_OUTPUT)
    print(QA_OUTPUT)


if __name__ == "__main__":
    main()
