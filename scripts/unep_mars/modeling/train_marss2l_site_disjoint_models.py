from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Input and outputs
# ============================================================

INPUT = Path(
    "outputs/268_marss2l_development_model_ready.csv"
)

COMPARISON_OUTPUT = Path(
    "outputs/270_marss2l_validation_model_comparison.csv"
)

PREDICTION_OUTPUT = Path(
    "outputs/271_marss2l_validation_predictions.csv"
)

SITE_METRIC_OUTPUT = Path(
    "outputs/272_marss2l_validation_site_metrics.csv"
)

THRESHOLD_OUTPUT = Path(
    "outputs/273_marss2l_validation_threshold_search.csv"
)

SELECTED_CONTRACT_OUTPUT = Path(
    "outputs/274_marss2l_selected_model_contract.csv"
)

SELECTED_MODEL_OUTPUT = Path(
    "outputs/275_marss2l_selected_train_only_model.joblib"
)

FEATURE_IMPORTANCE_OUTPUT = Path(
    "outputs/276_marss2l_selected_feature_importance.csv"
)


RANDOM_STATE = 42

EXPECTED_TRAIN_SITES = 51
EXPECTED_VALIDATION_SITES = 11
EXPECTED_MODEL_ROWS = 506

THRESHOLD_GRID = np.round(
    np.arange(
        0.05,
        0.951,
        0.01,
    ),
    2,
)


# ============================================================
# Feature sets
# ============================================================

BASELINE_FEATURES = [
    "cal_temporal_z_source_p95_percentile",
]


RAW_TEMPORAL_FEATURES = [
    "target_valid_fraction",
    "temporal_valid_fraction",
    "source_valid_fraction",
    "temporal_z_source_mean",
    "temporal_z_source_median",
    "temporal_z_source_p90",
    "temporal_z_source_p95",
    "temporal_z_source_max",
    "temporal_z_source_positive_fraction",
    "temporal_z_source_gt2_fraction",
    "temporal_z_source_gt3_fraction",
    "temporal_z_center_p95",
    "temporal_delta_outer_center",
    "temporal_delta_outer_scale",
]


CALIBRATED_BASE_FEATURES = [
    "temporal_z_source_mean",
    "temporal_z_source_median",
    "temporal_z_source_p90",
    "temporal_z_source_p95",
    "temporal_z_source_max",
    "temporal_z_source_positive_fraction",
    "temporal_z_source_gt2_fraction",
    "temporal_z_source_gt3_fraction",
    "temporal_z_center_p95",
]


CALIBRATED_FEATURES = []

for feature in CALIBRATED_BASE_FEATURES:
    CALIBRATED_FEATURES.extend([
        f"cal_{feature}_z",
        f"cal_{feature}_percentile",
    ])


AUXILIARY_FEATURES = [
    "qa_clear_fraction",
    "same_sensor_background_count",
    "sensor_is_lc09",
]


MULTI_FEATURES = (
    RAW_TEMPORAL_FEATURES
    + CALIBRATED_FEATURES
    + AUXILIARY_FEATURES
)


# ============================================================
# Model builders
# ============================================================

def build_single_feature_logistic():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
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
                penalty="l2",
                C=0.5,
                max_iter=10000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_multi_feature_logistic():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
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
                penalty="l2",
                C=0.2,
                max_iter=20000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_random_forest():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=4,
                max_features="sqrt",
                bootstrap=True,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_hist_gradient_boosting():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
            ),
        ),
        (
            "classifier",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=15,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


MODEL_SPECS = {
    "single_percentile_logistic": {
        "builder":
            build_single_feature_logistic,
        "features":
            BASELINE_FEATURES,
    },
    "multi_feature_logistic": {
        "builder":
            build_multi_feature_logistic,
        "features":
            MULTI_FEATURES,
    },
    "random_forest": {
        "builder":
            build_random_forest,
        "features":
            MULTI_FEATURES,
    },
    "hist_gradient_boosting": {
        "builder":
            build_hist_gradient_boosting,
        "features":
            MULTI_FEATURES,
    },
}


# ============================================================
# Utility functions
# ============================================================

def calculate_training_weights(frame):
    """
    讓每個場址的總權重接近相同，
    同時平衡正負樣本。
    """

    site_counts = (
        frame["site_key"]
        .value_counts()
    )

    label_counts = (
        frame["target_label"]
        .value_counts()
    )

    total_rows = len(frame)

    weights = []

    for _, row in frame.iterrows():
        site_weight = (
            1.0
            / site_counts[
                row["site_key"]
            ]
        )

        class_weight = (
            total_rows
            / (
                2.0
                * label_counts[
                    row["target_label"]
                ]
            )
        )

        weights.append(
            site_weight
            * class_weight
        )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    weights = (
        weights
        / np.mean(weights)
    )

    return weights


def confusion_values(
    y_true,
    y_pred,
):
    (
        true_negative,
        false_positive,
        false_negative,
        true_positive,
    ) = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return (
        int(true_negative),
        int(false_positive),
        int(false_negative),
        int(true_positive),
    )


def safe_ratio(
    numerator,
    denominator,
):
    if denominator == 0:
        return np.nan

    return numerator / denominator


def calculate_site_metrics(
    frame,
    predicted_column,
):
    rows = []

    for site_key, group in frame.groupby(
        "site_key"
    ):
        y_true = (
            group["target_label"]
            .astype(int)
            .to_numpy()
        )

        y_pred = (
            group[predicted_column]
            .astype(int)
            .to_numpy()
        )

        (
            true_negative,
            false_positive,
            false_negative,
            true_positive,
        ) = confusion_values(
            y_true,
            y_pred,
        )

        positive_recall = safe_ratio(
            true_positive,
            true_positive
            + false_negative,
        )

        negative_recall = safe_ratio(
            true_negative,
            true_negative
            + false_positive,
        )

        site_balanced_accuracy = (
            (
                positive_recall
                + negative_recall
            )
            / 2.0
            if (
                np.isfinite(
                    positive_recall
                )
                and np.isfinite(
                    negative_recall
                )
            )
            else np.nan
        )

        rows.append({
            "site_key":
                site_key,
            "sample_count":
                len(group),
            "negative_count":
                int((y_true == 0).sum()),
            "positive_count":
                int((y_true == 1).sum()),
            "true_negative":
                true_negative,
            "false_positive":
                false_positive,
            "false_negative":
                false_negative,
            "true_positive":
                true_positive,
            "positive_recall":
                positive_recall,
            "negative_recall":
                negative_recall,
            "site_balanced_accuracy":
                site_balanced_accuracy,
        })

    return pd.DataFrame(rows)


def evaluate_threshold(
    validation,
    threshold,
):
    frame = validation.copy()

    frame["temporary_prediction"] = (
        frame["prediction_score"]
        >= threshold
    ).astype(int)

    y_true = (
        frame["target_label"]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        frame["temporary_prediction"]
        .astype(int)
        .to_numpy()
    )

    (
        true_negative,
        false_positive,
        false_negative,
        true_positive,
    ) = confusion_values(
        y_true,
        y_pred,
    )

    positive_recall = safe_ratio(
        true_positive,
        true_positive
        + false_negative,
    )

    negative_recall = safe_ratio(
        true_negative,
        true_negative
        + false_positive,
    )

    false_positive_rate = safe_ratio(
        false_positive,
        true_negative
        + false_positive,
    )

    site_metrics = calculate_site_metrics(
        frame,
        predicted_column=
            "temporary_prediction",
    )

    return {
        "threshold":
            threshold,
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
        "positive_recall":
            positive_recall,
        "negative_recall":
            negative_recall,
        "false_positive_rate":
            false_positive_rate,
        "precision_positive":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
        "true_negative":
            true_negative,
        "false_positive":
            false_positive,
        "false_negative":
            false_negative,
        "true_positive":
            true_positive,
        "macro_site_balanced_accuracy":
            site_metrics[
                "site_balanced_accuracy"
            ].mean(),
        "macro_site_positive_recall":
            site_metrics[
                "positive_recall"
            ].mean(),
        "macro_site_negative_recall":
            site_metrics[
                "negative_recall"
            ].mean(),
    }


def select_threshold(
    validation,
):
    rows = []

    for threshold in THRESHOLD_GRID:
        rows.append(
            evaluate_threshold(
                validation,
                threshold,
            )
        )

    search = pd.DataFrame(rows)

    search[
        "distance_from_half"
    ] = np.abs(
        search["threshold"]
        - 0.5
    )

    search = search.sort_values(
        [
            "macro_site_balanced_accuracy",
            "balanced_accuracy",
            "false_positive_rate",
            "distance_from_half",
        ],
        ascending=[
            False,
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    return (
        float(
            search.iloc[0][
                "threshold"
            ]
        ),
        search,
    )


def fit_model(
    model,
    X_train,
    y_train,
    sample_weights,
):
    model.fit(
        X_train,
        y_train,
        classifier__sample_weight=
            sample_weights,
    )

    return model


def extract_feature_importance(
    model_name,
    model,
    feature_names,
):
    classifier = model.named_steps[
        "classifier"
    ]

    if hasattr(
        classifier,
        "coef_",
    ):
        values = (
            classifier.coef_[0]
        )

        importance_type = (
            "standardized_logistic_coefficient"
        )

    elif hasattr(
        classifier,
        "feature_importances_",
    ):
        values = (
            classifier.feature_importances_
        )

        importance_type = (
            "random_forest_importance"
        )

    else:
        return pd.DataFrame({
            "model_name":
                [model_name],
            "feature":
                ["not_available"],
            "importance":
                [np.nan],
            "absolute_importance":
                [np.nan],
            "importance_type":
                ["not_available"],
        })

    result = pd.DataFrame({
        "model_name":
            model_name,
        "feature":
            feature_names,
        "importance":
            values,
        "absolute_importance":
            np.abs(values),
        "importance_type":
            importance_type,
    })

    return result.sort_values(
        "absolute_importance",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# Main
# ============================================================

def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required = [
        "download_id",
        "site_key",
        "development_split",
        "target_label",
        "sensor_code",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}"
        )

    if len(df) != EXPECTED_MODEL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_MODEL_ROWS} rows, "
            f"found {len(df)}."
        )

    df["site_key"] = (
        df["site_key"]
        .astype(str)
        .str.strip()
    )

    df["target_label"] = (
        pd.to_numeric(
            df["target_label"],
            errors="raise",
        )
        .astype(int)
    )

    df["sensor_is_lc09"] = (
        df["sensor_code"]
        .astype(str)
        .str.upper()
        .eq("LC09")
        .astype(int)
    )

    train = df[
        df["development_split"]
        .eq("train")
    ].copy()

    validation = df[
        df["development_split"]
        .eq("validation")
    ].copy()

    train_sites = set(
        train["site_key"]
    )

    validation_sites = set(
        validation["site_key"]
    )

    if train_sites & validation_sites:
        raise RuntimeError(
            "Train/validation site overlap."
        )

    if (
        len(train_sites)
        != EXPECTED_TRAIN_SITES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_SITES} "
            f"training sites, found "
            f"{len(train_sites)}."
        )

    if (
        len(validation_sites)
        != EXPECTED_VALIDATION_SITES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_VALIDATION_SITES} "
            f"validation sites, found "
            f"{len(validation_sites)}."
        )

    missing_multi_features = [
        feature
        for feature in MULTI_FEATURES
        if feature not in df.columns
    ]

    if missing_multi_features:
        raise KeyError(
            "Missing model features:\n"
            + "\n".join(
                missing_multi_features
            )
        )

    print("=" * 112)
    print("MARS-S2L SITE-DISJOINT MODEL COMPARISON")
    print("=" * 112)

    print("\nTraining rows:", len(train))
    print("Validation rows:", len(validation))

    print(
        "Training sites:",
        len(train_sites),
    )

    print(
        "Validation sites:",
        len(validation_sites),
    )

    print("\nTraining labels:")
    print(
        train["target_label"]
        .value_counts()
        .sort_index()
    )

    print("\nValidation labels:")
    print(
        validation["target_label"]
        .value_counts()
        .sort_index()
    )

    training_weights = (
        calculate_training_weights(
            train
        )
    )

    comparison_rows = []
    prediction_frames = []
    threshold_frames = []
    site_metric_frames = []
    fitted_models = {}

    for (
        model_name,
        specification,
    ) in MODEL_SPECS.items():
        print(
            f"\nTraining {model_name}...",
            flush=True,
        )

        features = specification[
            "features"
        ]

        model = specification[
            "builder"
        ]()

        X_train = train[features]
        y_train = train[
            "target_label"
        ]

        X_validation = validation[
            features
        ]

        model = fit_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            sample_weights=
                training_weights,
        )

        scores = model.predict_proba(
            X_validation
        )[:, 1]

        validation_predictions = (
            validation.copy()
        )

        validation_predictions[
            "model_name"
        ] = model_name

        validation_predictions[
            "prediction_score"
        ] = scores

        selected_threshold, (
            threshold_search
        ) = select_threshold(
            validation_predictions
        )

        validation_predictions[
            "selected_threshold"
        ] = selected_threshold

        validation_predictions[
            "predicted_label"
        ] = (
            validation_predictions[
                "prediction_score"
            ]
            >= selected_threshold
        ).astype(int)

        validation_predictions[
            "correct"
        ] = (
            validation_predictions[
                "predicted_label"
            ]
            == validation_predictions[
                "target_label"
            ]
        )

        threshold_search[
            "model_name"
        ] = model_name

        threshold_frames.append(
            threshold_search
        )

        site_metrics = (
            calculate_site_metrics(
                validation_predictions,
                predicted_column=
                    "predicted_label",
            )
        )

        site_metrics[
            "model_name"
        ] = model_name

        site_metrics[
            "selected_threshold"
        ] = selected_threshold

        site_metric_frames.append(
            site_metrics
        )

        y_true = (
            validation_predictions[
                "target_label"
            ].astype(int).to_numpy()
        )

        y_pred = (
            validation_predictions[
                "predicted_label"
            ].astype(int).to_numpy()
        )

        (
            true_negative,
            false_positive,
            false_negative,
            true_positive,
        ) = confusion_values(
            y_true,
            y_pred,
        )

        positive_recall = safe_ratio(
            true_positive,
            true_positive
            + false_negative,
        )

        negative_recall = safe_ratio(
            true_negative,
            true_negative
            + false_positive,
        )

        comparison_rows.append({
            "model_name":
                model_name,
            "feature_count":
                len(features),
            "selected_threshold":
                selected_threshold,
            "validation_count":
                len(validation_predictions),
            "negative_count":
                int((y_true == 0).sum()),
            "positive_count":
                int((y_true == 1).sum()),
            "true_negative":
                true_negative,
            "false_positive":
                false_positive,
            "false_negative":
                false_negative,
            "true_positive":
                true_positive,
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
            "positive_recall":
                positive_recall,
            "negative_recall":
                negative_recall,
            "false_positive_rate":
                safe_ratio(
                    false_positive,
                    true_negative
                    + false_positive,
                ),
            "precision_positive":
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
            "roc_auc":
                roc_auc_score(
                    y_true,
                    scores,
                ),
            "average_precision":
                average_precision_score(
                    y_true,
                    scores,
                ),
            "macro_site_balanced_accuracy":
                site_metrics[
                    "site_balanced_accuracy"
                ].mean(),
            "macro_site_positive_recall":
                site_metrics[
                    "positive_recall"
                ].mean(),
            "macro_site_negative_recall":
                site_metrics[
                    "negative_recall"
                ].mean(),
        })

        prediction_frames.append(
            validation_predictions
        )

        fitted_models[
            model_name
        ] = {
            "model":
                model,
            "features":
                features,
            "threshold":
                selected_threshold,
        }

        print(
            "  threshold:",
            selected_threshold,
        )

        print(
            "  balanced accuracy:",
            comparison_rows[-1][
                "balanced_accuracy"
            ],
        )

        print(
            "  macro-site balanced accuracy:",
            comparison_rows[-1][
                "macro_site_balanced_accuracy"
            ],
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison[
        "distance_from_threshold_half"
    ] = np.abs(
        comparison[
            "selected_threshold"
        ]
        - 0.5
    )

    comparison = comparison.sort_values(
        [
            "macro_site_balanced_accuracy",
            "balanced_accuracy",
            "roc_auc",
            "false_positive_rate",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    comparison[
        "validation_rank"
    ] = np.arange(
        1,
        len(comparison) + 1,
    )

    selected_model_name = str(
        comparison.iloc[0][
            "model_name"
        ]
    )

    selected_info = fitted_models[
        selected_model_name
    ]

    selected_model = selected_info[
        "model"
    ]

    selected_features = selected_info[
        "features"
    ]

    selected_threshold = (
        selected_info["threshold"]
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
        sort=False,
    )

    threshold_results = pd.concat(
        threshold_frames,
        ignore_index=True,
        sort=False,
    )

    site_metrics_all = pd.concat(
        site_metric_frames,
        ignore_index=True,
        sort=False,
    )

    comparison.to_csv(
        COMPARISON_OUTPUT,
        index=False,
    )

    predictions.to_csv(
        PREDICTION_OUTPUT,
        index=False,
    )

    site_metrics_all.to_csv(
        SITE_METRIC_OUTPUT,
        index=False,
    )

    threshold_results.to_csv(
        THRESHOLD_OUTPUT,
        index=False,
    )

    joblib.dump(
        {
            "model_name":
                selected_model_name,
            "model":
                selected_model,
            "feature_columns":
                selected_features,
            "selected_threshold":
                selected_threshold,
            "training_sites":
                sorted(train_sites),
            "validation_sites":
                sorted(validation_sites),
            "dataset":
                "MARS-S2L",
            "selection_used_external_test":
                False,
        },
        SELECTED_MODEL_OUTPUT,
    )

    selected_row = (
        comparison[
            comparison["model_name"]
            .eq(selected_model_name)
        ]
        .iloc[0]
        .to_dict()
    )

    contract = pd.DataFrame([{
        **selected_row,
        "training_site_count":
            len(train_sites),
        "validation_site_count":
            len(validation_sites),
        "training_row_count":
            len(train),
        "validation_row_count":
            len(validation),
        "feature_columns":
            "|".join(
                selected_features
            ),
        "selection_metric":
            (
                "macro_site_balanced_accuracy"
            ),
        "threshold_selection_data":
            "validation_sites_only",
        "model_selection_data":
            "validation_sites_only",
        "external_test_used":
            False,
        "random_state":
            RANDOM_STATE,
    }])

    contract.to_csv(
        SELECTED_CONTRACT_OUTPUT,
        index=False,
    )

    feature_importance = (
        extract_feature_importance(
            model_name=
                selected_model_name,
            model=
                selected_model,
            feature_names=
                selected_features,
        )
    )

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 112)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 112)

    display_columns = [
        "validation_rank",
        "model_name",
        "feature_count",
        "selected_threshold",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "balanced_accuracy",
        "positive_recall",
        "negative_recall",
        "false_positive_rate",
        "roc_auc",
        "average_precision",
        "macro_site_balanced_accuracy",
    ]

    print(
        comparison[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nSelected model:")
    print(selected_model_name)

    print(
        "Selected threshold:",
        selected_threshold,
    )

    print(
        "Selected feature count:",
        len(selected_features),
    )

    print("\nTop feature importance:")
    print(
        feature_importance.head(20)
        .to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nSaved:")
    print(COMPARISON_OUTPUT)
    print(PREDICTION_OUTPUT)
    print(SITE_METRIC_OUTPUT)
    print(THRESHOLD_OUTPUT)
    print(SELECTED_CONTRACT_OUTPUT)
    print(SELECTED_MODEL_OUTPUT)
    print(FEATURE_IMPORTANCE_OUTPUT)


if __name__ == "__main__":
    main()
