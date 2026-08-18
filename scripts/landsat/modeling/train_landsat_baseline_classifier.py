from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_CSV = Path(
    "outputs/35_landsat_patch_features.csv"
)

PREDICTION_OUTPUT = Path(
    "outputs/36_landsat_baseline_predictions.csv"
)

METRICS_OUTPUT = Path(
    "outputs/37_landsat_baseline_metrics.csv"
)

MODEL_DIRECTORY = Path(
    "outputs/models"
)


FEATURE_PREFIXES = (
    "blue_",
    "green_",
    "red_",
    "nir_",
    "swir1_",
    "swir2_",
    "ndvi_",
    "ndmi_",
    "nbr_",
    "ndsi_",
    "swir_nd_",
    "swir_ratio_",
    "valid_pixel_fraction",
)


def select_feature_columns(df):
    feature_columns = []

    for column in df.columns:
        if column.startswith(FEATURE_PREFIXES):
            if pd.api.types.is_numeric_dtype(df[column]):
                feature_columns.append(column)

    return feature_columns


def calculate_metrics(
    model_name,
    y_true,
    y_pred,
    y_probability,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "model": model_name,
        "n_samples": len(y_true),
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "precision_label_0": precision_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0,
        ),
        "recall_label_0": recall_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0,
        ),
        "f1_label_0": f1_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0,
        ),
        "precision_label_1": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "recall_label_1": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "f1_label_1": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return metrics


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Feature CSV not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    if "label" not in df.columns:
        raise ValueError(
            "The feature CSV does not contain a label column"
        )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    feature_columns = select_feature_columns(df)

    if not feature_columns:
        raise ValueError(
            "No feature columns were detected"
        )

    X = df[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    y = df["label"]

    print(f"Samples: {len(df)}")
    print(f"Features: {len(feature_columns)}")

    print("\nLabel counts:")
    print(y.value_counts().sort_index())

    minimum_class_count = int(
        y.value_counts().min()
    )

    n_splits = min(
        5,
        minimum_class_count,
    )

    if n_splits < 2:
        raise ValueError(
            "Not enough samples for cross-validation"
        )

    print(f"\nCross-validation folds: {n_splits}")

    cross_validation = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    logistic_model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            ),
        ),
    ])

    random_forest_model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    models = {
        "logistic_regression": logistic_model,
        "random_forest": random_forest_model,
    }

    prediction_tables = []
    metric_rows = []

    identifier_columns = [
        column
        for column in [
            "event_id",
            "patch_id",
            "filename",
            "file_name",
            "resolved_patch_path",
            "landsat_sensor",
            "acquisition_date",
        ]
        if column in df.columns
    ]

    for model_name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        probabilities = cross_val_predict(
            model,
            X,
            y,
            cv=cross_validation,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        print(
            classification_report(
                y,
                predictions,
                digits=3,
                zero_division=0,
            )
        )

        matrix = confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        )

        print("Confusion matrix:")
        print(matrix)

        metrics = calculate_metrics(
            model_name,
            y,
            predictions,
            probabilities,
        )

        metric_rows.append(metrics)

        prediction_df = df[
            identifier_columns
        ].copy()

        prediction_df["model"] = model_name
        prediction_df["true_label"] = y.values
        prediction_df["predicted_label"] = predictions
        prediction_df["probability_label_1"] = probabilities
        prediction_df["correct"] = (
            prediction_df["true_label"]
            == prediction_df["predicted_label"]
        )

        prediction_tables.append(prediction_df)

        # Cross-validation 完成後，
        # 再使用全部 37 筆資料訓練最終模型並儲存。
        model.fit(X, y)

        MODEL_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            MODEL_DIRECTORY
            / f"landsat_{model_name}.joblib"
        )

        joblib.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
            },
            model_path,
        )

        print(f"Saved fitted model: {model_path}")

    all_predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    metrics_df = pd.DataFrame(metric_rows)

    all_predictions.to_csv(
        PREDICTION_OUTPUT,
        index=False,
    )

    metrics_df.to_csv(
        METRICS_OUTPUT,
        index=False,
    )

    print("\nModel comparison:")
    print(
        metrics_df[
            [
                "model",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "roc_auc",
                "false_positive",
                "false_negative",
            ]
        ].to_string(index=False)
    )

    wrong_predictions = all_predictions[
        ~all_predictions["correct"]
    ]

    print(
        f"\nWrong prediction rows: "
        f"{len(wrong_predictions)}"
    )

    print("\nSaved:")
    print(PREDICTION_OUTPUT)
    print(METRICS_OUTPUT)


if __name__ == "__main__":
    main()
