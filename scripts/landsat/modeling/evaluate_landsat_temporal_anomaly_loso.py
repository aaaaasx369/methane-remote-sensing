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

FOLD_OUTPUT = Path(
    "outputs/114_landsat_temporal_loso_fold_metrics.csv"
)

PREDICTION_OUTPUT = Path(
    "outputs/115_landsat_temporal_loso_predictions.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/116_landsat_temporal_loso_summary.csv"
)

ENGINEERED_OUTPUT = Path(
    "outputs/117_landsat_temporal_anomaly_engineered_features.csv"
)


FEATURE_SETS = {
    # 只測剛剛看起來最有希望的單一特徵。
    "source_p95_1": [
        "temporal_z_source_p95",
    ],

    # 排放源附近的 temporal anomaly。
    "source_anomaly_4": [
        "temporal_z_source_p95",
        "temporal_z_source_max",
        "positive_z3_fraction_source",
        "log_positive_z3_source_connected_pixels",
    ],

    # 排放源相對於同張影像外圍背景的差異。
    "source_outer_contrast_4": [
        "source_minus_outer_p95",
        "source_minus_outer_mean",
        "source_minus_outer_z3_fraction",
        "log_positive_z3_source_connected_pixels",
    ],

    # 結合 temporal difference 與 source/background contrast。
    "combined_6": [
        "temporal_z_source_p95",
        "delta_ratio_source_p95",
        "positive_z3_fraction_source",
        "source_minus_outer_p95",
        "source_minus_outer_z3_fraction",
        "log_positive_z3_source_connected_pixels",
    ],
}


def calculate_metrics(
    y_true,
    y_pred,
    probabilities,
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
            probabilities,
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


def add_engineered_features(dataframe):
    df = dataframe.copy()

    # 排放源附近的異常，減去影像外圍的異常。
    df["source_minus_outer_p95"] = (
        df["temporal_z_source_p95"]
        - df["temporal_z_outer_p95"]
    )

    df["source_minus_outer_mean"] = (
        df["temporal_z_source_mean"]
        - df["temporal_z_outer_mean"]
    )

    df[
        "source_minus_outer_z3_fraction"
    ] = (
        df[
            "positive_z3_fraction_source"
        ]
        - df[
            "positive_z3_fraction_outer"
        ]
    )

    df[
        "absolute_source_minus_outer_p95"
    ] = (
        df[
            "abs_temporal_z_source_p95"
        ]
        - df[
            "abs_temporal_z_outer_p95"
        ]
    )

    # Pixel 數量跨度可能很大，使用 log1p 縮小尺度。
    connected_pixels = pd.to_numeric(
        df[
            "positive_z3_source_connected_pixels"
        ],
        errors="coerce",
    ).clip(lower=0)

    df[
        "log_positive_z3_source_connected_pixels"
    ] = np.log1p(
        connected_pixels
    )

    return df


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

    if 1 not in classes:
        return np.zeros(
            len(x_test)
        )

    return probabilities[
        :,
        classes.index(1)
    ]


def fit_training_median_threshold(
    train_values,
    train_labels,
):
    """
    使用訓練場址中正負類別中位數的中點作門檻。

    門檻完全由訓練場址決定，不使用測試場址答案。
    """
    negative_median = float(
        np.median(
            train_values[
                train_labels == 0
            ]
        )
    )

    positive_median = float(
        np.median(
            train_values[
                train_labels == 1
            ]
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


def apply_threshold(
    values,
    threshold,
    direction,
):
    if direction == "greater":
        prediction = (
            values >= threshold
        ).astype(int)

        # 建立可排序的分數供 AUC 使用。
        score = values

    else:
        prediction = (
            values <= threshold
        ).astype(int)

        score = -values

    return prediction, score


def scene_metadata(row):
    columns = [
        "scene_key",
        "site_key_normalized",
        "landsat_sensor",
        "acquisition_time_utc",
        "release_rate_kg_h",
        "training_class",
        "high_emission_target",
        "temporal_preview_path",
    ]

    return {
        column: row.get(
            column,
            np.nan,
        )
        for column in columns
    }


def evaluate_logistic(
    dataframe,
    feature_set_name,
    feature_columns,
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

        x_train = train[
            feature_columns
        ]

        y_train = train[
            "high_emission_target"
        ].astype(int)

        x_test = test[
            feature_columns
        ]

        y_test = test[
            "high_emission_target"
        ].astype(int)

        model = build_logistic_model()

        model.fit(
            x_train,
            y_train,
        )

        predicted = model.predict(
            x_test
        ).astype(int)

        probabilities = (
            positive_probability(
                model,
                x_test,
            )
        )

        metrics = calculate_metrics(
            y_test,
            predicted,
            probabilities,
        )

        fold_rows.append({
            "model_name":
                "logistic_regression",
            "feature_set":
                feature_set_name,
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
            **metrics,
        })

        for position, (
            index,
            row,
        ) in enumerate(
            test.iterrows()
        ):
            output = scene_metadata(row)

            output.update({
                "model_name":
                    "logistic_regression",
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
                        y_test.loc[index]
                    ),
                "predicted_label":
                    int(
                        predicted[
                            position
                        ]
                    ),
                "predicted_probability":
                    float(
                        probabilities[
                            position
                        ]
                    ),
                "correct":
                    bool(
                        predicted[
                            position
                        ]
                        == y_test.loc[index]
                    ),
            })

            prediction_rows.append(
                output
            )

    return fold_rows, prediction_rows


def evaluate_threshold_model(
    dataframe,
):
    fold_rows = []
    prediction_rows = []

    feature_name = (
        "temporal_z_source_p95"
    )

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

        train_values = pd.to_numeric(
            train[feature_name],
            errors="raise",
        ).to_numpy()

        train_labels = train[
            "high_emission_target"
        ].astype(int).to_numpy()

        test_values = pd.to_numeric(
            test[feature_name],
            errors="raise",
        ).to_numpy()

        y_test = test[
            "high_emission_target"
        ].astype(int)

        (
            threshold,
            direction,
            train_negative_median,
            train_positive_median,
        ) = fit_training_median_threshold(
            train_values,
            train_labels,
        )

        predicted, score = apply_threshold(
            test_values,
            threshold,
            direction,
        )

        metrics = calculate_metrics(
            y_test,
            predicted,
            score,
        )

        fold_rows.append({
            "model_name":
                "training_median_threshold",
            "feature_set":
                "source_p95_1",
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
                train_negative_median,
            "train_positive_median":
                train_positive_median,
            **metrics,
        })

        for position, (
            index,
            row,
        ) in enumerate(
            test.iterrows()
        ):
            output = scene_metadata(row)

            output.update({
                "model_name":
                    "training_median_threshold",
                "feature_set":
                    "source_p95_1",
                "train_site":
                    train[
                        "site_key_normalized"
                    ].iloc[0],
                "test_site":
                    test_site,
                "actual_label":
                    int(
                        y_test.loc[index]
                    ),
                "predicted_label":
                    int(
                        predicted[
                            position
                        ]
                    ),
                "predicted_probability":
                    float(
                        score[position]
                    ),
                "threshold":
                    threshold,
                "threshold_direction":
                    direction,
                "correct":
                    bool(
                        predicted[position]
                        == y_test.loc[index]
                    ),
            })

            prediction_rows.append(
                output
            )

    return fold_rows, prediction_rows


def evaluate_dummy(dataframe):
    fold_rows = []
    prediction_rows = []

    sites = sorted(
        dataframe[
            "site_key_normalized"
        ].unique()
    )

    dummy_feature = [
        "temporal_z_source_p95"
    ]

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

        model = build_dummy_model()

        model.fit(
            train[dummy_feature],
            train[
                "high_emission_target"
            ].astype(int),
        )

        predicted = model.predict(
            test[dummy_feature]
        ).astype(int)

        probabilities = (
            positive_probability(
                model,
                test[dummy_feature],
            )
        )

        y_test = test[
            "high_emission_target"
        ].astype(int)

        metrics = calculate_metrics(
            y_test,
            predicted,
            probabilities,
        )

        fold_rows.append({
            "model_name":
                "dummy_prior",
            "feature_set":
                "dummy",
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
            **metrics,
        })

        for position, (
            index,
            row,
        ) in enumerate(
            test.iterrows()
        ):
            output = scene_metadata(row)

            output.update({
                "model_name":
                    "dummy_prior",
                "feature_set":
                    "dummy",
                "train_site":
                    train[
                        "site_key_normalized"
                    ].iloc[0],
                "test_site":
                    test_site,
                "actual_label":
                    int(
                        y_test.loc[index]
                    ),
                "predicted_label":
                    int(
                        predicted[
                            position
                        ]
                    ),
                "predicted_probability":
                    float(
                        probabilities[
                            position
                        ]
                    ),
                "correct":
                    bool(
                        predicted[position]
                        == y_test.loc[index]
                    ),
            })

            prediction_rows.append(
                output
            )

    return fold_rows, prediction_rows


def pooled_summary(predictions):
    rows = []

    for (
        model_name,
        feature_set,
    ), group in predictions.groupby(
        [
            "model_name",
            "feature_set",
        ]
    ):
        metrics = calculate_metrics(
            group[
                "actual_label"
            ].astype(int),
            group[
                "predicted_label"
            ].astype(int),
            group[
                "predicted_probability"
            ].astype(float),
        )

        rows.append({
            "model_name":
                model_name,
            "feature_set":
                feature_set,
            "pooled_rows":
                len(group),
            **metrics,
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

    data = add_engineered_features(
        data
    )

    if len(data) != 16:
        raise ValueError(
            f"Expected 16 scenes, "
            f"found {len(data)}."
        )

    data[
        "high_emission_target"
    ] = pd.to_numeric(
        data[
            "high_emission_target"
        ],
        errors="raise",
    ).astype(int)

    print("=" * 105)
    print("TEMPORAL ANOMALY LEAVE-ONE-SITE-OUT")
    print("=" * 105)

    print("\nLabel by site:")
    print(
        pd.crosstab(
            data[
                "site_key_normalized"
            ],
            data[
                "high_emission_target"
            ],
            margins=True,
        )
    )

    required_features = sorted(
        {
            feature
            for columns in (
                FEATURE_SETS.values()
            )
            for feature in columns
        }
    )

    missing = [
        column
        for column in required_features
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            f"Missing features: {missing}"
        )

    all_folds = []
    all_predictions = []

    folds, predictions = (
        evaluate_dummy(data)
    )

    all_folds.extend(folds)
    all_predictions.extend(predictions)

    folds, predictions = (
        evaluate_threshold_model(
            data
        )
    )

    all_folds.extend(folds)
    all_predictions.extend(predictions)

    for (
        feature_set_name,
        feature_columns,
    ) in FEATURE_SETS.items():
        folds, predictions = (
            evaluate_logistic(
                data,
                feature_set_name,
                feature_columns,
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

    summary_df = pooled_summary(
        prediction_df
    )

    ENGINEERED_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        ENGINEERED_OUTPUT,
        index=False,
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

    display_columns = [
        "model_name",
        "feature_set",
        "pooled_rows",
        "accuracy",
        "balanced_accuracy",
        "recall_positive",
        "recall_negative",
        "precision_positive",
        "f1_positive",
        "roc_auc",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]

    print("\n" + "=" * 105)
    print("POOLED TEMPORAL OUT-OF-SITE PERFORMANCE")
    print("=" * 105)

    print(
        summary_df[
            display_columns
        ].sort_values(
            [
                "balanced_accuracy",
                "roc_auc",
            ],
            ascending=False,
        ).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}",
        )
    )

    fold_columns = [
        "model_name",
        "feature_set",
        "train_site",
        "test_site",
        "train_rows",
        "test_rows",
        "balanced_accuracy",
        "recall_positive",
        "recall_negative",
        "roc_auc",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]

    print("\n" + "=" * 105)
    print("PER-SITE TEMPORAL PERFORMANCE")
    print("=" * 105)

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
    print(ENGINEERED_OUTPUT)
    print(FOLD_OUTPUT)
    print(PREDICTION_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
