from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_CSV = Path(
    "outputs/112_landsat_temporal_anomaly_features.csv"
)

CALIBRATED_OUTPUT = Path(
    "outputs/118_landsat_site_calibrated_anomaly_features.csv"
)

CALIBRATION_AUDIT_OUTPUT = Path(
    "outputs/119_landsat_site_calibration_audit.csv"
)

FOLD_OUTPUT = Path(
    "outputs/120_landsat_site_calibrated_loso_folds.csv"
)

PREDICTION_OUTPUT = Path(
    "outputs/121_landsat_site_calibrated_loso_predictions.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/122_landsat_site_calibrated_loso_summary.csv"
)


# 先只校正少數具有明確物理意義的 source-region 特徵。
BASE_FEATURES = [
    "temporal_z_source_p95",
    "temporal_z_source_max",
    "delta_ratio_source_p95",
]


FEATURE_SETS = {
    # 上一輪最佳特徵，保留作比較。
    "raw_source_p95_1": [
        "temporal_z_source_p95",
    ],

    # 最重要的新特徵：
    # 相對於同場址無排放影像的 robust Z-score。
    "calibrated_source_p95_z_1": [
        "cal_temporal_z_source_p95_z",
    ],

    # 使用同場址負樣本中的經驗百分位。
    "calibrated_source_p95_percentile_1": [
        "cal_temporal_z_source_p95_percentile",
    ],

    # 少量場址校正後的 source features。
    "calibrated_source_3": [
        "cal_temporal_z_source_p95_z",
        "cal_temporal_z_source_max_z",
        "cal_delta_ratio_source_p95_z",
    ],
}


def robust_reference_statistics(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 3:
        raise ValueError(
            "At least three historical negative "
            "reference values are required."
        )

    median = float(
        np.median(values)
    )

    mad = float(
        np.median(
            np.abs(values - median)
        )
    )

    sigma = 1.4826 * mad
    scale_method = "mad"

    # MAD 為零時改用 IQR。
    if (
        not np.isfinite(sigma)
        or sigma < 1e-8
    ):
        q25, q75 = np.percentile(
            values,
            [25, 75],
        )

        sigma = float(
            (q75 - q25) / 1.349
        )

        scale_method = "iqr"

    # IQR 仍為零時改用標準差。
    if (
        not np.isfinite(sigma)
        or sigma < 1e-8
    ):
        sigma = float(
            np.std(values, ddof=1)
        )

        scale_method = "standard_deviation"

    # 所有負樣本完全相同時，避免除以零。
    if (
        not np.isfinite(sigma)
        or sigma < 1e-8
    ):
        sigma = 1.0
        scale_method = "unit_fallback"

    return {
        "median": median,
        "sigma": sigma,
        "scale_method": scale_method,
        "minimum": float(
            np.min(values)
        ),
        "maximum": float(
            np.max(values)
        ),
        "count": int(
            len(values)
        ),
    }


def empirical_percentile(
    value,
    reference_values,
):
    reference_values = np.asarray(
        reference_values,
        dtype=float,
    )

    reference_values = reference_values[
        np.isfinite(reference_values)
    ]

    less = np.sum(
        reference_values < value
    )

    equal = np.sum(
        reference_values == value
    )

    # Mid-rank empirical percentile。
    return float(
        (
            less
            + 0.5 * equal
        )
        / len(reference_values)
    )


def add_site_calibration(dataframe):
    df = dataframe.copy()

    audit_rows = []

    for feature in BASE_FEATURES:
        df[
            f"cal_{feature}_negative_median"
        ] = np.nan

        df[
            f"cal_{feature}_negative_sigma"
        ] = np.nan

        df[
            f"cal_{feature}_negative_count"
        ] = np.nan

        df[
            f"cal_{feature}_delta"
        ] = np.nan

        df[
            f"cal_{feature}_z"
        ] = np.nan

        df[
            f"cal_{feature}_percentile"
        ] = np.nan

        df[
            f"cal_{feature}_scale_method"
        ] = ""

    for row_index, row in df.iterrows():
        site = row[
            "site_key_normalized"
        ]

        scene_key = str(
            row["scene_key"]
        )

        label = int(
            row["high_emission_target"]
        )

        site_negative_pool = df[
            (
                df["site_key_normalized"]
                == site
            )
            & (
                df["high_emission_target"]
                == 0
            )
        ].copy()

        # 負樣本自身不能用來校正自己。
        # 這是 leave-one-negative-out calibration。
        if label == 0:
            site_negative_pool = (
                site_negative_pool[
                    site_negative_pool[
                        "scene_key"
                    ].astype(str)
                    != scene_key
                ].copy()
            )

        if len(site_negative_pool) < 3:
            raise ValueError(
                f"{scene_key}: only "
                f"{len(site_negative_pool)} "
                "eligible site-negative references."
            )

        for feature in BASE_FEATURES:
            value = pd.to_numeric(
                pd.Series([
                    row[feature]
                ]),
                errors="coerce",
            ).iloc[0]

            reference_values = (
                pd.to_numeric(
                    site_negative_pool[
                        feature
                    ],
                    errors="coerce",
                )
                .dropna()
                .to_numpy()
            )

            statistics = (
                robust_reference_statistics(
                    reference_values
                )
            )

            calibrated_delta = (
                value
                - statistics["median"]
            )

            calibrated_z = (
                calibrated_delta
                / statistics["sigma"]
            )

            percentile = (
                empirical_percentile(
                    value,
                    reference_values,
                )
            )

            df.at[
                row_index,
                f"cal_{feature}_negative_median",
            ] = statistics["median"]

            df.at[
                row_index,
                f"cal_{feature}_negative_sigma",
            ] = statistics["sigma"]

            df.at[
                row_index,
                f"cal_{feature}_negative_count",
            ] = statistics["count"]

            df.at[
                row_index,
                f"cal_{feature}_delta",
            ] = calibrated_delta

            df.at[
                row_index,
                f"cal_{feature}_z",
            ] = calibrated_z

            df.at[
                row_index,
                f"cal_{feature}_percentile",
            ] = percentile

            df.at[
                row_index,
                f"cal_{feature}_scale_method",
            ] = statistics[
                "scale_method"
            ]

            audit_rows.append({
                "scene_key": scene_key,
                "site_key": site,
                "high_emission_target": label,
                "feature_name": feature,
                "raw_value": value,
                "negative_reference_count":
                    statistics["count"],
                "negative_reference_median":
                    statistics["median"],
                "negative_reference_sigma":
                    statistics["sigma"],
                "negative_reference_min":
                    statistics["minimum"],
                "negative_reference_max":
                    statistics["maximum"],
                "scale_method":
                    statistics["scale_method"],
                "calibrated_delta":
                    calibrated_delta,
                "calibrated_z":
                    calibrated_z,
                "empirical_percentile":
                    percentile,
                "calibration_mode":
                    (
                        "historical_site_negatives_"
                        "leave_one_negative_out"
                    ),
            })

    return (
        df,
        pd.DataFrame(audit_rows),
    )


def calculate_metrics(
    y_true,
    y_pred,
    score,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    recall_negative = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(
            y_true,
            score,
        )
    else:
        auc = np.nan

    return {
        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),
        "precision_positive":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
        "recall_positive":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
        "recall_negative":
            recall_negative,
        "f1_positive":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
        "roc_auc":
            auc,
        "true_negative":
            int(tn),
        "false_positive":
            int(fp),
        "false_negative":
            int(fn),
        "true_positive":
            int(tp),
    }


def build_logistic_model():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                C=0.5,
                max_iter=10000,
                random_state=42,
            ),
        ),
    ])


def build_dummy_model():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "classifier",
            DummyClassifier(
                strategy="prior",
                random_state=42,
            ),
        ),
    ])


def positive_probability(
    model,
    x_test,
):
    probabilities = model.predict_proba(
        x_test
    )

    classes = list(
        model.named_steps[
            "classifier"
        ].classes_
    )

    return probabilities[
        :,
        classes.index(1)
    ]


def evaluate_logistic(
    dataframe,
    model_name,
    feature_set_name,
    feature_columns,
    model_builder,
):
    fold_rows = []
    prediction_rows = []

    sites = sorted(
        dataframe[
            "site_key_normalized"
        ].unique()
    )

    for test_site in sites:
        train = dataframe[
            dataframe[
                "site_key_normalized"
            ] != test_site
        ].copy()

        test = dataframe[
            dataframe[
                "site_key_normalized"
            ] == test_site
        ].copy()

        y_train = train[
            "high_emission_target"
        ].astype(int)

        y_test = test[
            "high_emission_target"
        ].astype(int)

        model = model_builder()

        model.fit(
            train[feature_columns],
            y_train,
        )

        predicted = model.predict(
            test[feature_columns]
        ).astype(int)

        probability = positive_probability(
            model,
            test[feature_columns],
        )

        metrics = calculate_metrics(
            y_test,
            predicted,
            probability,
        )

        fold_rows.append({
            "model_name": model_name,
            "feature_set":
                feature_set_name,
            "train_site":
                train[
                    "site_key_normalized"
                ].iloc[0],
            "test_site": test_site,
            "train_rows": len(train),
            "test_rows": len(test),
            **metrics,
        })

        for position, (
            dataframe_index,
            row,
        ) in enumerate(
            test.iterrows()
        ):
            prediction_rows.append({
                "scene_key":
                    row["scene_key"],
                "site_key":
                    row[
                        "site_key_normalized"
                    ],
                "release_rate_kg_h":
                    row[
                        "release_rate_kg_h"
                    ],
                "model_name":
                    model_name,
                "feature_set":
                    feature_set_name,
                "train_site":
                    train[
                        "site_key_normalized"
                    ].iloc[0],
                "test_site":
                    test_site,
                "actual_label":
                    int(
                        y_test.loc[
                            dataframe_index
                        ]
                    ),
                "predicted_label":
                    int(
                        predicted[position]
                    ),
                "prediction_score":
                    float(
                        probability[position]
                    ),
                "correct":
                    bool(
                        predicted[position]
                        == y_test.loc[
                            dataframe_index
                        ]
                    ),
            })

    return fold_rows, prediction_rows


def fit_median_threshold(
    values,
    labels,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    negative_median = float(
        np.median(
            values[labels == 0]
        )
    )

    positive_median = float(
        np.median(
            values[labels == 1]
        )
    )

    threshold = (
        negative_median
        + positive_median
    ) / 2

    direction = (
        "greater"
        if positive_median
        >= negative_median
        else "less"
    )

    return (
        threshold,
        direction,
        negative_median,
        positive_median,
    )


def evaluate_threshold(dataframe):
    feature = (
        "cal_temporal_z_source_p95_z"
    )

    fold_rows = []
    prediction_rows = []

    for test_site in sorted(
        dataframe[
            "site_key_normalized"
        ].unique()
    ):
        train = dataframe[
            dataframe[
                "site_key_normalized"
            ] != test_site
        ].copy()

        test = dataframe[
            dataframe[
                "site_key_normalized"
            ] == test_site
        ].copy()

        train_values = train[
            feature
        ].to_numpy(dtype=float)

        train_labels = train[
            "high_emission_target"
        ].astype(int).to_numpy()

        (
            threshold,
            direction,
            negative_median,
            positive_median,
        ) = fit_median_threshold(
            train_values,
            train_labels,
        )

        test_values = test[
            feature
        ].to_numpy(dtype=float)

        if direction == "greater":
            predicted = (
                test_values >= threshold
            ).astype(int)

            score = test_values

        else:
            predicted = (
                test_values <= threshold
            ).astype(int)

            score = -test_values

        y_test = test[
            "high_emission_target"
        ].astype(int)

        metrics = calculate_metrics(
            y_test,
            predicted,
            score,
        )

        fold_rows.append({
            "model_name":
                "training_median_threshold",
            "feature_set":
                "calibrated_source_p95_z_1",
            "train_site":
                train[
                    "site_key_normalized"
                ].iloc[0],
            "test_site":
                test_site,
            "train_rows":
                len(train),
            "test_rows":
                len(test),
            "threshold":
                threshold,
            "threshold_direction":
                direction,
            "train_negative_median":
                negative_median,
            "train_positive_median":
                positive_median,
            **metrics,
        })

        for position, (
            dataframe_index,
            row,
        ) in enumerate(
            test.iterrows()
        ):
            prediction_rows.append({
                "scene_key":
                    row["scene_key"],
                "site_key":
                    row[
                        "site_key_normalized"
                    ],
                "release_rate_kg_h":
                    row[
                        "release_rate_kg_h"
                    ],
                "model_name":
                    "training_median_threshold",
                "feature_set":
                    "calibrated_source_p95_z_1",
                "train_site":
                    train[
                        "site_key_normalized"
                    ].iloc[0],
                "test_site":
                    test_site,
                "actual_label":
                    int(
                        y_test.loc[
                            dataframe_index
                        ]
                    ),
                "predicted_label":
                    int(
                        predicted[position]
                    ),
                "prediction_score":
                    float(score[position]),
                "threshold":
                    threshold,
                "threshold_direction":
                    direction,
                "correct":
                    bool(
                        predicted[position]
                        == y_test.loc[
                            dataframe_index
                        ]
                    ),
            })

    return fold_rows, prediction_rows


def build_summary(
    fold_df,
    prediction_df,
):
    rows = []

    for (
        model_name,
        feature_set,
    ), predictions in (
        prediction_df.groupby([
            "model_name",
            "feature_set",
        ])
    ):
        folds = fold_df[
            (
                fold_df["model_name"]
                == model_name
            )
            & (
                fold_df["feature_set"]
                == feature_set
            )
        ]

        pooled_metrics = (
            calculate_metrics(
                predictions[
                    "actual_label"
                ].astype(int),
                predictions[
                    "predicted_label"
                ].astype(int),
                predictions[
                    "prediction_score"
                ].astype(float),
            )
        )

        rows.append({
            "model_name":
                model_name,
            "feature_set":
                feature_set,
            "pooled_rows":
                len(predictions),
            "mean_fold_balanced_accuracy":
                folds[
                    "balanced_accuracy"
                ].mean(),
            "mean_fold_roc_auc":
                folds[
                    "roc_auc"
                ].mean(),
            **{
                f"pooled_{key}": value
                for key, value
                in pooled_metrics.items()
            },
        })

    return pd.DataFrame(rows)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            INPUT_CSV
        )

    data = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    data[
        "high_emission_target"
    ] = pd.to_numeric(
        data[
            "high_emission_target"
        ],
        errors="raise",
    ).astype(int)

    calibrated, audit = (
        add_site_calibration(data)
    )

    calibrated.to_csv(
        CALIBRATED_OUTPUT,
        index=False,
    )

    audit.to_csv(
        CALIBRATION_AUDIT_OUTPUT,
        index=False,
    )

    print("=" * 105)
    print("SITE-CALIBRATED TEMPORAL ANOMALY")
    print("=" * 105)

    print("\nRaw source p95 by site and label:")

    print(
        calibrated.groupby([
            "site_key_normalized",
            "high_emission_target",
        ])[
            "temporal_z_source_p95"
        ].median()
    )

    print(
        "\nCalibrated source p95 Z "
        "by site and label:"
    )

    print(
        calibrated.groupby([
            "site_key_normalized",
            "high_emission_target",
        ])[
            "cal_temporal_z_source_p95_z"
        ].median()
    )

    print(
        "\nCalibrated source p95 percentile "
        "by site and label:"
    )

    print(
        calibrated.groupby([
            "site_key_normalized",
            "high_emission_target",
        ])[
            "cal_temporal_z_source_p95_percentile"
        ].median()
    )

    all_folds = []
    all_predictions = []

    # Dummy baseline。
    folds, predictions = (
        evaluate_logistic(
            calibrated,
            model_name="dummy_prior",
            feature_set_name="dummy",
            feature_columns=[
                "temporal_z_source_p95"
            ],
            model_builder=
                build_dummy_model,
        )
    )

    all_folds.extend(folds)
    all_predictions.extend(predictions)

    # Learned threshold。
    folds, predictions = (
        evaluate_threshold(
            calibrated
        )
    )

    all_folds.extend(folds)
    all_predictions.extend(predictions)

    # Logistic models。
    for (
        feature_set_name,
        feature_columns,
    ) in FEATURE_SETS.items():
        missing = [
            column
            for column in feature_columns
            if column
            not in calibrated.columns
        ]

        if missing:
            raise KeyError(
                f"{feature_set_name}: "
                f"missing {missing}"
            )

        folds, predictions = (
            evaluate_logistic(
                calibrated,
                model_name=
                    "logistic_regression",
                feature_set_name=
                    feature_set_name,
                feature_columns=
                    feature_columns,
                model_builder=
                    build_logistic_model,
            )
        )

        all_folds.extend(folds)
        all_predictions.extend(
            predictions
        )

    fold_df = pd.DataFrame(
        all_folds
    )

    prediction_df = pd.DataFrame(
        all_predictions
    )

    summary_df = build_summary(
        fold_df,
        prediction_df,
    )

    fold_df.to_csv(
        FOLD_OUTPUT,
        index=False,
    )

    prediction_df.to_csv(
        PREDICTION_OUTPUT,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 105)
    print("SITE-CALIBRATED LOSO SUMMARY")
    print("=" * 105)

    summary_columns = [
        "model_name",
        "feature_set",
        "pooled_rows",
        "mean_fold_balanced_accuracy",
        "mean_fold_roc_auc",
        "pooled_accuracy",
        "pooled_balanced_accuracy",
        "pooled_recall_positive",
        "pooled_recall_negative",
        "pooled_precision_positive",
        "pooled_f1_positive",
        "pooled_roc_auc",
        "pooled_true_negative",
        "pooled_false_positive",
        "pooled_false_negative",
        "pooled_true_positive",
    ]

    print(
        summary_df[
            summary_columns
        ].sort_values(
            [
                "mean_fold_balanced_accuracy",
                "mean_fold_roc_auc",
            ],
            ascending=False,
        ).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}",
        )
    )

    print("\n" + "=" * 105)
    print("SITE-CALIBRATED PER-SITE PERFORMANCE")
    print("=" * 105)

    fold_columns = [
        "model_name",
        "feature_set",
        "train_site",
        "test_site",
        "balanced_accuracy",
        "recall_positive",
        "recall_negative",
        "roc_auc",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]

    print(
        fold_df[
            fold_columns
        ].sort_values(
            [
                "model_name",
                "feature_set",
                "test_site",
            ]
        ).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}",
        )
    )

    print("\nSaved:")
    print(CALIBRATED_OUTPUT)
    print(CALIBRATION_AUDIT_OUTPUT)
    print(FOLD_OUTPUT)
    print(PREDICTION_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
