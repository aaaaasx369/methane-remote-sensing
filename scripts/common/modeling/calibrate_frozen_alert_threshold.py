from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


TRAIN_PREDICTIONS = Path(
    "outputs/121_landsat_site_calibrated_loso_predictions.csv"
)

EXTERNAL_PREDICTIONS = Path(
    "outputs/149_evanston_external_predictions.csv"
)

THRESHOLD_AUDIT_OUTPUT = Path(
    "outputs/153_training_only_threshold_audit.csv"
)

EXTERNAL_PREDICTIONS_OUTPUT = Path(
    "outputs/154_evanston_threshold_calibrated_predictions.csv"
)

EXTERNAL_METRICS_OUTPUT = Path(
    "outputs/155_evanston_threshold_calibrated_metrics.csv"
)


def find_column(dataframe, candidates):
    for column in candidates:
        if column in dataframe.columns:
            return column

    raise KeyError(
        f"Could not find any of these columns: {candidates}\n"
        f"Available columns:\n{dataframe.columns.tolist()}"
    )


def parse_boolean(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )


def calculate_metrics(y_true, probability, threshold):
    predicted = (
        np.asarray(probability, dtype=float)
        >= threshold
    ).astype(int)

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1],
    ).ravel()

    recall_negative = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    roc_auc = (
        roc_auc_score(
            y_true,
            probability,
        )
        if len(np.unique(y_true)) == 2
        else np.nan
    )

    return {
        "threshold":
            float(threshold),
        "rows":
            len(y_true),
        "accuracy":
            accuracy_score(
                y_true,
                predicted,
            ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                predicted,
            ),
        "recall_positive":
            recall_score(
                y_true,
                predicted,
                zero_division=0,
            ),
        "recall_negative":
            recall_negative,
        "precision_positive":
            precision_score(
                y_true,
                predicted,
                zero_division=0,
            ),
        "roc_auc":
            roc_auc,
        "true_negative":
            int(tn),
        "false_positive":
            int(fp),
        "false_negative":
            int(fn),
        "true_positive":
            int(tp),
    }


def prepare_training_predictions():
    if not TRAIN_PREDICTIONS.exists():
        raise FileNotFoundError(
            TRAIN_PREDICTIONS
        )

    dataframe = pd.read_csv(
        TRAIN_PREDICTIONS,
        low_memory=False,
    )

    if "model_name" in dataframe.columns:
        dataframe = dataframe[
            dataframe["model_name"]
            .astype(str)
            .str.lower()
            .eq("logistic_regression")
        ].copy()

    if "feature_set" in dataframe.columns:
        dataframe = dataframe[
            dataframe["feature_set"]
            .astype(str)
            .str.lower()
            .str.contains(
                "percentile",
                na=False,
            )
        ].copy()

    label_column = find_column(
        dataframe,
        [
            "actual_label",
            "high_emission_target",
            "external_target",
            "label",
            "y_true",
            "target",
        ],
    )

    probability_column = find_column(
        dataframe,
        [
            "predicted_probability",
            "probability",
            "positive_probability",
            "y_probability",
            "prediction_score",
        ],
    )

    fold_column = find_column(
        dataframe,
        [
            "test_site",
            "site_key_normalized",
            "site_key",
        ],
    )

    dataframe["_label"] = pd.to_numeric(
        dataframe[label_column],
        errors="coerce",
    )

    dataframe["_probability"] = pd.to_numeric(
        dataframe[probability_column],
        errors="coerce",
    )

    dataframe["_fold"] = (
        dataframe[fold_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe = dataframe.dropna(
        subset=[
            "_label",
            "_probability",
        ]
    ).copy()

    dataframe["_label"] = (
        dataframe["_label"]
        .astype(int)
    )

    return dataframe


def generate_thresholds(probabilities):
    unique_values = np.sort(
        np.unique(
            np.asarray(
                probabilities,
                dtype=float,
            )
        )
    )

    thresholds = [0.0, 1.0]

    thresholds.extend(
        unique_values.tolist()
    )

    if len(unique_values) >= 2:
        midpoints = (
            unique_values[:-1]
            + unique_values[1:]
        ) / 2

        thresholds.extend(
            midpoints.tolist()
        )

    return np.sort(
        np.unique(thresholds)
    )


def select_training_threshold(training):
    thresholds = generate_thresholds(
        training["_probability"]
    )

    audit_rows = []

    for threshold in thresholds:
        fold_metrics = []

        for fold_name, fold in training.groupby(
            "_fold"
        ):
            metrics = calculate_metrics(
                y_true=fold["_label"],
                probability=fold[
                    "_probability"
                ],
                threshold=threshold,
            )

            metrics["fold"] = fold_name

            fold_metrics.append(
                metrics
            )

        fold_frame = pd.DataFrame(
            fold_metrics
        )

        pooled = calculate_metrics(
            y_true=training["_label"],
            probability=training[
                "_probability"
            ],
            threshold=threshold,
        )

        audit_rows.append({
            "threshold":
                threshold,
            "mean_fold_balanced_accuracy":
                fold_frame[
                    "balanced_accuracy"
                ].mean(),
            "minimum_fold_balanced_accuracy":
                fold_frame[
                    "balanced_accuracy"
                ].min(),
            "mean_fold_recall_positive":
                fold_frame[
                    "recall_positive"
                ].mean(),
            "mean_fold_recall_negative":
                fold_frame[
                    "recall_negative"
                ].mean(),
            "pooled_balanced_accuracy":
                pooled[
                    "balanced_accuracy"
                ],
            "pooled_recall_positive":
                pooled[
                    "recall_positive"
                ],
            "pooled_recall_negative":
                pooled[
                    "recall_negative"
                ],
            "pooled_false_positive":
                pooled[
                    "false_positive"
                ],
            "pooled_false_negative":
                pooled[
                    "false_negative"
                ],
        })

    audit = pd.DataFrame(
        audit_rows
    )

    # 門檻選擇規則在看到 Evanston 之前應固定：
    # 1. 最大化兩個 LOSO folds 的平均 balanced accuracy
    # 2. 優先改善最差場址
    # 3. 優先保留 high-emission recall
    # 4. 再降低 false positives
    selected = (
        audit.sort_values(
            [
                "mean_fold_balanced_accuracy",
                "minimum_fold_balanced_accuracy",
                "mean_fold_recall_positive",
                "mean_fold_recall_negative",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )

    return (
        float(selected["threshold"]),
        audit,
    )


def evaluate_external(threshold):
    if not EXTERNAL_PREDICTIONS.exists():
        raise FileNotFoundError(
            EXTERNAL_PREDICTIONS
        )

    dataframe = pd.read_csv(
        EXTERNAL_PREDICTIONS,
        low_memory=False,
    )

    dataframe = dataframe[
        dataframe["model_name"]
        .astype(str)
        .str.lower()
        .eq("logistic_regression")
        & dataframe["feature_set"]
        .astype(str)
        .str.lower()
        .eq("primary_percentile")
    ].copy()

    included = parse_boolean(
        dataframe[
            "primary_evaluation_included"
        ]
    )

    dataframe = dataframe[
        included == True
    ].copy()

    dataframe[
        "external_target"
    ] = pd.to_numeric(
        dataframe[
            "external_target"
        ],
        errors="coerce",
    )

    dataframe[
        "predicted_probability"
    ] = pd.to_numeric(
        dataframe[
            "predicted_probability"
        ],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            "external_target",
            "predicted_probability",
        ]
    ).copy()

    dataframe[
        "threshold_selected_from_training_only"
    ] = threshold

    dataframe[
        "threshold_calibrated_label"
    ] = (
        dataframe[
            "predicted_probability"
        ]
        >= threshold
    ).astype(int)

    dataframe[
        "threshold_calibrated_correct"
    ] = (
        dataframe[
            "threshold_calibrated_label"
        ]
        == dataframe[
            "external_target"
        ].astype(int)
    )

    metrics = calculate_metrics(
        y_true=dataframe[
            "external_target"
        ].astype(int),
        probability=dataframe[
            "predicted_probability"
        ],
        threshold=threshold,
    )

    return dataframe, pd.DataFrame(
        [
            {
                "model_name":
                    "logistic_regression",
                "feature_set":
                    "primary_percentile",
                "threshold_source":
                    "Casa Grande + Ehrenberg LOSO only",
                **metrics,
            }
        ]
    )


def main():
    training = (
        prepare_training_predictions()
    )

    print("=" * 105)
    print("TRAINING-ONLY THRESHOLD CALIBRATION")
    print("=" * 105)

    print("\nTraining OOF rows:", len(training))

    print("\nTraining labels:")
    print(
        training["_label"]
        .value_counts()
        .sort_index()
    )

    print("\nRows by held-out site:")
    print(
        training["_fold"]
        .value_counts()
    )

    threshold, threshold_audit = (
        select_training_threshold(
            training
        )
    )

    print(
        "\nSelected threshold "
        "(without using Evanston):",
        f"{threshold:.6f}",
    )

    external_predictions, external_metrics = (
        evaluate_external(
            threshold
        )
    )

    THRESHOLD_AUDIT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_audit.to_csv(
        THRESHOLD_AUDIT_OUTPUT,
        index=False,
    )

    external_predictions.to_csv(
        EXTERNAL_PREDICTIONS_OUTPUT,
        index=False,
    )

    external_metrics.to_csv(
        EXTERNAL_METRICS_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 105)
    print("EVANSTON RESULTS WITH TRAINING-ONLY THRESHOLD")
    print("=" * 105)

    print(
        external_metrics.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}",
        )
    )

    display_columns = [
        "external_target_class",
        "landsat_product_id",
        "predicted_probability",
        "threshold_calibrated_label",
        "external_target",
        "threshold_calibrated_correct",
    ]

    print("\nPredictions:")
    print(
        external_predictions[
            display_columns
        ].sort_values(
            "predicted_probability",
            ascending=False,
        ).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}",
        )
    )

    print("\nSaved:")
    print(THRESHOLD_AUDIT_OUTPUT)
    print(EXTERNAL_PREDICTIONS_OUTPUT)
    print(EXTERNAL_METRICS_OUTPUT)


if __name__ == "__main__":
    main()
