from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
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
    LeaveOneOut,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Files
# ============================================================

INPUT_CSV = Path(
    "outputs/43_landsat_unique_clean_features.csv"
)

PREDICTIONS_CSV = Path(
    "outputs/45_landsat_loocv_predictions.csv"
)

METRICS_CSV = Path(
    "outputs/46_landsat_loocv_metrics.csv"
)

COEFFICIENTS_CSV = Path(
    "outputs/47_landsat_logistic_coefficients.csv"
)


# ============================================================
# Fixed feature sets
#
# Important:
# These are chosen before looking at classification performance.
# We are NOT automatically selecting the best features from all
# 86 columns because only 13 independent rasters are available.
# ============================================================

FEATURE_SETS = {
    # Minimal methane/SWIR-oriented representation
    "logistic_swir_2": [
        "log_swir1_over_swir2_mean",
        "log_swir1_over_swir2_standardized_contrast",
    ],

    # Adds scene brightness and vegetation context
    "logistic_context_5": [
        "swir1_mean",
        "swir2_mean",
        "log_swir1_over_swir2_mean",
        "log_swir1_over_swir2_standardized_contrast",
        "ndvi_mean",
    ],
}


# ============================================================
# Metric helpers
# ============================================================

def calculate_metrics(
    model_name,
    feature_set_name,
    y_true,
    y_pred,
    probability_label_1,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "model": model_name,
        "feature_set": feature_set_name,
        "n_samples": len(y_true),
        "n_label_0": int((y_true == 0).sum()),
        "n_label_1": int((y_true == 1).sum()),

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
            probability_label_1,
        ),

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def bootstrap_intervals(
    y_true,
    y_pred,
    probabilities,
    iterations=5000,
    random_seed=42,
):
    """
    Approximate stratified bootstrap intervals.

    Label-0 and Label-1 samples are resampled separately so every
    bootstrap replicate contains both classes.
    """
    rng = np.random.default_rng(
        random_seed
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probabilities = np.asarray(probabilities)

    label_0_indices = np.where(
        y_true == 0
    )[0]

    label_1_indices = np.where(
        y_true == 1
    )[0]

    bootstrap_values = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "roc_auc": [],
    }

    for _ in range(iterations):
        sampled_label_0 = rng.choice(
            label_0_indices,
            size=len(label_0_indices),
            replace=True,
        )

        sampled_label_1 = rng.choice(
            label_1_indices,
            size=len(label_1_indices),
            replace=True,
        )

        sampled_indices = np.concatenate([
            sampled_label_0,
            sampled_label_1,
        ])

        sampled_true = y_true[
            sampled_indices
        ]

        sampled_pred = y_pred[
            sampled_indices
        ]

        sampled_probabilities = probabilities[
            sampled_indices
        ]

        bootstrap_values["accuracy"].append(
            accuracy_score(
                sampled_true,
                sampled_pred,
            )
        )

        bootstrap_values[
            "balanced_accuracy"
        ].append(
            balanced_accuracy_score(
                sampled_true,
                sampled_pred,
            )
        )

        bootstrap_values["macro_f1"].append(
            f1_score(
                sampled_true,
                sampled_pred,
                average="macro",
                zero_division=0,
            )
        )

        bootstrap_values["roc_auc"].append(
            roc_auc_score(
                sampled_true,
                sampled_probabilities,
            )
        )

    intervals = {}

    for metric_name, values in (
        bootstrap_values.items()
    ):
        lower, upper = np.percentile(
            values,
            [2.5, 97.5],
        )

        intervals[
            f"{metric_name}_ci_lower"
        ] = float(lower)

        intervals[
            f"{metric_name}_ci_upper"
        ] = float(upper)

    return intervals


# ============================================================
# Model construction
# ============================================================

def build_logistic_model():
    """
    Strongly regularized logistic regression.

    C=0.1 means stronger regularization than the default C=1.
    This is helpful because the dataset is extremely small.
    """
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
                C=0.1,
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
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
                strategy="most_frequent"
            ),
        ),
    ])


# ============================================================
# Main program
# ============================================================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    print("=" * 80)
    print("CLEAN LANDSAT EXPLORATORY BASELINE")
    print("=" * 80)

    print(f"\nInput file: {INPUT_CSV}")
    print(f"Samples: {len(df)}")

    if "label" not in df.columns:
        raise ValueError(
            "label column is missing."
        )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    if not df["label"].isin([0, 1]).all():
        raise ValueError(
            "Labels must contain only 0 and 1."
        )

    df["label"] = df["label"].astype(int)

    if (
        "pixel_hash" in df.columns
        and df["pixel_hash"].duplicated().any()
    ):
        raise ValueError(
            "Duplicate pixel hashes remain."
        )

    if (
        "raster_group_id" in df.columns
        and df["raster_group_id"].duplicated().any()
    ):
        raise ValueError(
            "Duplicate raster groups remain."
        )

    y = df["label"].to_numpy()

    print("\nLabel counts:")
    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    if "landsat_sensor" in df.columns:
        print("\nSensor counts:")
        print(
            df["landsat_sensor"]
            .value_counts()
        )

        print("\nLabel by sensor:")
        print(
            pd.crosstab(
                df["landsat_sensor"],
                df["label"],
                margins=True,
            )
        )

    # Leave one independent raster out at a time.
    cross_validation = LeaveOneOut()

    identifier_columns = [
        column
        for column in [
            "raster_group_id",
            "pixel_hash",
            "event_id",
            "filename",
            "site_name",
            "landsat_sensor",
            "landsat_image_time",
            "datetime_utc",
            "label_decision_source",
            "label_decision_confidence",
        ]
        if column in df.columns
    ]

    prediction_tables = []
    metrics_rows = []
    coefficient_rows = []

    # --------------------------------------------------------
    # Majority-class dummy baseline
    # --------------------------------------------------------

    dummy_feature = np.zeros(
        (len(df), 1),
        dtype=float,
    )

    dummy_model = build_dummy_model()

    dummy_predictions = cross_val_predict(
        dummy_model,
        dummy_feature,
        y,
        cv=cross_validation,
        method="predict",
    )

    dummy_probabilities_full = (
        cross_val_predict(
            dummy_model,
            dummy_feature,
            y,
            cv=cross_validation,
            method="predict_proba",
        )
    )

    # Both classes are present in every LOOCV training fold.
    dummy_probabilities = (
        dummy_probabilities_full[:, 1]
    )

    dummy_metrics = calculate_metrics(
        model_name="dummy_most_frequent",
        feature_set_name="none",
        y_true=y,
        y_pred=dummy_predictions,
        probability_label_1=dummy_probabilities,
    )

    dummy_metrics.update(
        bootstrap_intervals(
            y,
            dummy_predictions,
            dummy_probabilities,
        )
    )

    metrics_rows.append(
        dummy_metrics
    )

    dummy_prediction_df = df[
        identifier_columns
    ].copy()

    dummy_prediction_df["model"] = (
        "dummy_most_frequent"
    )

    dummy_prediction_df["feature_set"] = (
        "none"
    )

    dummy_prediction_df["true_label"] = y
    dummy_prediction_df[
        "predicted_label"
    ] = dummy_predictions

    dummy_prediction_df[
        "probability_label_1"
    ] = dummy_probabilities

    dummy_prediction_df["correct"] = (
        dummy_predictions == y
    )

    prediction_tables.append(
        dummy_prediction_df
    )

    print("\n" + "=" * 80)
    print("MODEL: dummy_most_frequent")
    print("=" * 80)

    print(
        classification_report(
            y,
            dummy_predictions,
            digits=3,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y,
            dummy_predictions,
            labels=[0, 1],
        )
    )

    # --------------------------------------------------------
    # Logistic models
    # --------------------------------------------------------

    for model_name, feature_columns in (
        FEATURE_SETS.items()
    ):
        missing_columns = [
            column
            for column in feature_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{model_name} is missing columns: "
                + ", ".join(missing_columns)
            )

        X = (
            df[feature_columns]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        model = build_logistic_model()

        probabilities = cross_val_predict(
            model,
            X,
            y,
            cv=cross_validation,
            method="predict_proba",
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        metrics = calculate_metrics(
            model_name=model_name,
            feature_set_name=" | ".join(
                feature_columns
            ),
            y_true=y,
            y_pred=predictions,
            probability_label_1=probabilities,
        )

        metrics.update(
            bootstrap_intervals(
                y,
                predictions,
                probabilities,
            )
        )

        metrics_rows.append(metrics)

        prediction_df = df[
            identifier_columns
        ].copy()

        prediction_df["model"] = model_name

        prediction_df["feature_set"] = (
            " | ".join(feature_columns)
        )

        prediction_df["true_label"] = y
        prediction_df[
            "predicted_label"
        ] = predictions

        prediction_df[
            "probability_label_1"
        ] = probabilities

        prediction_df["correct"] = (
            predictions == y
        )

        prediction_tables.append(
            prediction_df
        )

        print("\n" + "=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)

        print("Features:")

        for feature in feature_columns:
            print(f"  {feature}")

        print()

        print(
            classification_report(
                y,
                predictions,
                digits=3,
                zero_division=0,
            )
        )

        print("Confusion matrix:")
        print(
            confusion_matrix(
                y,
                predictions,
                labels=[0, 1],
            )
        )

        print(
            "ROC AUC:",
            f"{metrics['roc_auc']:.3f}",
        )

        # Fit once using all 13 samples only to inspect
        # standardized model coefficients.
        model.fit(
            X,
            y,
        )

        classifier = model.named_steps[
            "classifier"
        ]

        coefficients = classifier.coef_[0]

        for feature, coefficient in zip(
            feature_columns,
            coefficients,
        ):
            coefficient_rows.append({
                "model": model_name,
                "feature": feature,
                "standardized_coefficient":
                    float(coefficient),
                "odds_ratio_per_1sd":
                    float(np.exp(coefficient)),
            })

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    predictions_df = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    coefficients_df = pd.DataFrame(
        coefficient_rows
    )

    predictions_df.to_csv(
        PREDICTIONS_CSV,
        index=False,
    )

    metrics_df.to_csv(
        METRICS_CSV,
        index=False,
    )

    coefficients_df.to_csv(
        COEFFICIENTS_CSV,
        index=False,
    )

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    comparison_columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "recall_label_0",
        "recall_label_1",
        "false_positive",
        "false_negative",
        "balanced_accuracy_ci_lower",
        "balanced_accuracy_ci_upper",
    ]

    print(
        metrics_df[
            comparison_columns
        ].to_string(
            index=False
        )
    )

    print("\nImportant interpretation:")
    print(
        "These are exploratory results based on only "
        "13 independent Landsat rasters."
    )
    print(
        "They must not be interpreted as a definitive "
        "generalization estimate."
    )

    print("\nSaved:")
    print(PREDICTIONS_CSV)
    print(METRICS_CSV)
    print(COEFFICIENTS_CSV)


if __name__ == "__main__":
    main()
