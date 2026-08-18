from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
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


TRAIN_INPUT = Path(
    "outputs/118_landsat_site_calibrated_anomaly_features.csv"
)

EXTERNAL_INPUT = Path(
    "outputs/146_evanston_external_temporal_features.csv"
)

THRESHOLD_INPUT = Path(
    "outputs/107_landsat_high_emission_threshold_summary.csv"
)

PREDICTION_OUTPUT = Path(
    "outputs/149_evanston_external_predictions.csv"
)

METRICS_OUTPUT = Path(
    "outputs/150_evanston_external_metrics.csv"
)

MODEL_OUTPUT = Path(
    "outputs/151_evanston_external_model_parameters.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/152_evanston_external_evaluation_audit.csv"
)


FEATURE_SETS = {
    # 主要模型：先前 LOSO 表現最佳的固定特徵。
    "primary_percentile": [
        "cal_temporal_z_source_p95_percentile",
    ],

    # 敏感度分析：這兩個特徵先前已經定義過，
    # 不是看到 Evanston 結果後新創造的特徵。
    "sensitivity_calibrated_z": [
        "cal_temporal_z_source_p95_z",
    ],

    "sensitivity_raw_p95": [
        "temporal_z_source_p95",
    ],
}


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str:
    for column in candidates:
        if column in dataframe.columns:
            return column

    raise KeyError(
        f"None of these columns were found: "
        f"{candidates}"
    )


def detect_high_emission_threshold() -> float:
    environment_value = os.environ.get(
        "HIGH_EMISSION_THRESHOLD_KG_H"
    )

    if environment_value:
        return float(environment_value)

    if not THRESHOLD_INPUT.exists():
        raise FileNotFoundError(
            f"Missing threshold file: "
            f"{THRESHOLD_INPUT}\n"
            "Set HIGH_EMISSION_THRESHOLD_KG_H "
            "to the previously frozen threshold."
        )

    dataframe = pd.read_csv(
        THRESHOLD_INPUT,
        low_memory=False,
    )

    # 情況一：threshold 為獨立數值欄位。
    candidate_columns = [
        column
        for column in dataframe.columns
        if (
            "threshold"
            in column.lower()
            and any(
                token in column.lower()
                for token in [
                    "kg",
                    "flow",
                    "emission",
                    "rate",
                ]
            )
        )
    ]

    for column in candidate_columns:
        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).dropna().unique()

        values = values[
            values > 0
        ]

        if len(values) == 1:
            return float(values[0])

    # 情況二：item/value 或 metric/value 格式。
    text_columns = [
        column
        for column in dataframe.columns
        if dataframe[column].dtype == "object"
    ]

    numeric_columns = [
        column
        for column in dataframe.columns
        if pd.to_numeric(
            dataframe[column],
            errors="coerce",
        ).notna().any()
    ]

    for text_column in text_columns:
        matching_rows = (
            dataframe[text_column]
            .astype(str)
            .str.lower()
            .str.contains(
                "threshold",
                na=False,
            )
        )

        for numeric_column in numeric_columns:
            values = pd.to_numeric(
                dataframe.loc[
                    matching_rows,
                    numeric_column,
                ],
                errors="coerce",
            ).dropna()

            values = values[
                values > 0
            ]

            if len(values) == 1:
                return float(
                    values.iloc[0]
                )

    raise RuntimeError(
        "Could not uniquely detect the frozen "
        "high-emission threshold.\n"
        f"Columns: {dataframe.columns.tolist()}\n"
        f"First rows:\n"
        f"{dataframe.head(10).to_string(index=False)}\n\n"
        "Run again with:\n"
        "HIGH_EMISSION_THRESHOLD_KG_H=<value> "
        "python evaluate_evanston_external_frozen_model.py"
    )


def build_logistic_model():
    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                C=1.0,
                max_iter=10000,
                random_state=42,
            ),
        ),
    ])


def build_dummy_model():
    return DummyClassifier(
        strategy="prior",
        random_state=42,
    )


def positive_probability(
    model,
    features,
):
    probabilities = model.predict_proba(
        features
    )

    classes = list(
        model.classes_
        if hasattr(model, "classes_")
        else model.named_steps[
            "classifier"
        ].classes_
    )

    positive_index = classes.index(1)

    return probabilities[
        :,
        positive_index
    ]


def calculate_metrics(
    y_true,
    y_predicted,
    probability,
):
    matrix = confusion_matrix(
        y_true,
        y_predicted,
        labels=[0, 1],
    )

    true_negative, false_positive, (
        false_negative
    ), true_positive = matrix.ravel()

    recall_negative = (
        true_negative
        / (
            true_negative
            + false_positive
        )
        if (
            true_negative
            + false_positive
        ) > 0
        else np.nan
    )

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(
            y_true,
            probability,
        )
    else:
        roc_auc = np.nan

    return {
        "test_rows": len(y_true),
        "accuracy": accuracy_score(
            y_true,
            y_predicted,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_predicted,
            ),
        "recall_positive":
            recall_score(
                y_true,
                y_predicted,
                zero_division=0,
            ),
        "recall_negative":
            recall_negative,
        "precision_positive":
            precision_score(
                y_true,
                y_predicted,
                zero_division=0,
            ),
        "f1_positive":
            f1_score(
                y_true,
                y_predicted,
                zero_division=0,
            ),
        "roc_auc":
            roc_auc,
        "true_negative":
            int(true_negative),
        "false_positive":
            int(false_positive),
        "false_negative":
            int(false_negative),
        "true_positive":
            int(true_positive),
    }


def prepare_training():
    if not TRAIN_INPUT.exists():
        raise FileNotFoundError(
            TRAIN_INPUT
        )

    training = pd.read_csv(
        TRAIN_INPUT,
        low_memory=False,
    )

    site_column = find_column(
        training,
        [
            "site_key_normalized",
            "site_key",
        ],
    )

    label_column = find_column(
        training,
        [
            "high_emission_target",
            "label",
        ],
    )

    training[
        "training_site"
    ] = (
        training[site_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    training[
        "training_target"
    ] = pd.to_numeric(
        training[label_column],
        errors="coerce",
    )

    training = training[
        training["training_site"]
        .isin([
            "casa_grande",
            "ehrenberg",
        ])
        & training[
            "training_target"
        ].isin([0, 1])
    ].copy()

    if "feature_status" in training.columns:
        training = training[
            training["feature_status"]
            == "success"
        ].copy()

    training[
        "training_target"
    ] = training[
        "training_target"
    ].astype(int)

    return training


def prepare_external(
    high_emission_threshold,
):
    if not EXTERNAL_INPUT.exists():
        raise FileNotFoundError(
            EXTERNAL_INPUT
        )

    external = pd.read_csv(
        EXTERNAL_INPUT,
        low_memory=False,
    )

    if "feature_status" in external.columns:
        external = external[
            external["feature_status"]
            == "success"
        ].copy()

    external[
        "flow_at_scene_kg_h"
    ] = pd.to_numeric(
        external.get(
            "flow_at_scene_kg_h",
            np.nan,
        ),
        errors="coerce",
    )

    external[
        "external_target"
    ] = np.nan

    external[
        "primary_evaluation_included"
    ] = False

    # 真正保留的測試負樣本。
    test_negative_mask = (
        external["external_role"]
        == "test_negative"
    )

    external.loc[
        test_negative_mask,
        "external_target",
    ] = 0

    external.loc[
        test_negative_mask,
        "primary_evaluation_included",
    ] = True

    # Controlled release 中只有超過凍結門檻的，
    # 才屬於 high-emission target。
    positive_mask = (
        external["external_role"]
        == "positive"
    )

    high_emission_mask = (
        positive_mask
        & (
            external[
                "flow_at_scene_kg_h"
            ]
            >= high_emission_threshold
        )
    )

    lower_emission_mask = (
        positive_mask
        & ~high_emission_mask
    )

    external.loc[
        high_emission_mask,
        "external_target",
    ] = 1

    external.loc[
        high_emission_mask,
        "primary_evaluation_included",
    ] = True

    external[
        "external_target_class"
    ] = "excluded_calibration"

    external.loc[
        test_negative_mask,
        "external_target_class",
    ] = "no_release"

    external.loc[
        high_emission_mask,
        "external_target_class",
    ] = "high_emission"

    external.loc[
        lower_emission_mask,
        "external_target_class",
    ] = "lower_emission_release_excluded"

    return external


def main():
    threshold = (
        detect_high_emission_threshold()
    )

    print("=" * 108)
    print("EVANSTON FROZEN EXTERNAL VALIDATION")
    print("=" * 108)

    print(
        "\nFrozen high-emission threshold:",
        f"{threshold:.3f} kg/h",
    )

    training = prepare_training()

    external = prepare_external(
        threshold
    )

    print("\nTraining label by site:")
    print(
        pd.crosstab(
            training["training_site"],
            training["training_target"],
            margins=True,
        )
    )

    print("\nExternal target classes:")
    print(
        external[
            "external_target_class"
        ].value_counts()
    )

    evaluation = external[
        external[
            "primary_evaluation_included"
        ]
    ].copy()

    evaluation[
        "external_target"
    ] = evaluation[
        "external_target"
    ].astype(int)

    print("\nPrimary external evaluation:")
    print(
        pd.crosstab(
            evaluation[
                "external_role"
            ],
            evaluation[
                "external_target"
            ],
            margins=True,
        )
    )

    prediction_rows = []
    metric_rows = []
    parameter_rows = []

    models = [
        (
            "dummy_prior",
            "dummy",
            None,
        ),
    ]

    for (
        feature_set_name,
        feature_columns,
    ) in FEATURE_SETS.items():
        models.append((
            "logistic_regression",
            feature_set_name,
            feature_columns,
        ))

    for (
        model_name,
        feature_set_name,
        feature_columns,
    ) in models:
        if model_name == "dummy_prior":
            # DummyClassifier still expects a feature array.
            dummy_feature = (
                "cal_temporal_z_source_p95_percentile"
            )

            x_train = training[
                [dummy_feature]
            ]

            x_external = external[
                [dummy_feature]
            ]

            model = build_dummy_model()

        else:
            missing_training = [
                column
                for column
                in feature_columns
                if column
                not in training.columns
            ]

            missing_external = [
                column
                for column
                in feature_columns
                if column
                not in external.columns
            ]

            if (
                missing_training
                or missing_external
            ):
                print(
                    f"[SKIP] {feature_set_name} | "
                    f"training missing="
                    f"{missing_training} | "
                    f"external missing="
                    f"{missing_external}"
                )

                continue

            x_train = training[
                feature_columns
            ]

            x_external = external[
                feature_columns
            ]

            model = (
                build_logistic_model()
            )

        valid_train = (
            x_train.notna().all(axis=1)
            & training[
                "training_target"
            ].notna()
        )

        model.fit(
            x_train.loc[
                valid_train
            ],
            training.loc[
                valid_train,
                "training_target",
            ],
        )

        predicted_label = model.predict(
            x_external
        ).astype(int)

        probability = (
            positive_probability(
                model,
                x_external,
            )
        )

        model_predictions = (
            external.copy()
        )

        model_predictions[
            "model_name"
        ] = model_name

        model_predictions[
            "feature_set"
        ] = feature_set_name

        model_predictions[
            "predicted_probability"
        ] = probability

        model_predictions[
            "predicted_label"
        ] = predicted_label

        model_predictions[
            "correct_when_included"
        ] = np.where(
            model_predictions[
                "primary_evaluation_included"
            ],
            (
                model_predictions[
                    "predicted_label"
                ]
                == model_predictions[
                    "external_target"
                ]
            ),
            np.nan,
        )

        prediction_rows.append(
            model_predictions
        )

        included_predictions = (
            model_predictions[
                model_predictions[
                    "primary_evaluation_included"
                ]
            ].copy()
        )

        metrics = calculate_metrics(
            y_true=included_predictions[
                "external_target"
            ].astype(int),
            y_predicted=included_predictions[
                "predicted_label"
            ].astype(int),
            probability=included_predictions[
                "predicted_probability"
            ].astype(float),
        )

        metric_rows.append({
            "model_name":
                model_name,
            "feature_set":
                feature_set_name,
            "training_rows":
                int(valid_train.sum()),
            "high_emission_threshold_kg_h":
                threshold,
            **metrics,
        })

        if model_name == (
            "logistic_regression"
        ):
            classifier = (
                model.named_steps[
                    "classifier"
                ]
            )

            scaler = (
                model.named_steps[
                    "scaler"
                ]
            )

            for position, column in enumerate(
                feature_columns
            ):
                parameter_rows.append({
                    "model_name":
                        model_name,
                    "feature_set":
                        feature_set_name,
                    "feature_name":
                        column,
                    "scaled_coefficient":
                        float(
                            classifier.coef_[
                                0,
                                position,
                            ]
                        ),
                    "training_mean":
                        float(
                            scaler.mean_[
                                position
                            ]
                        ),
                    "training_scale":
                        float(
                            scaler.scale_[
                                position
                            ]
                        ),
                    "intercept":
                        float(
                            classifier.intercept_[
                                0
                            ]
                        ),
                })

    predictions = pd.concat(
        prediction_rows,
        ignore_index=True,
        sort=False,
    )

    metrics = pd.DataFrame(
        metric_rows
    )

    parameters = pd.DataFrame(
        parameter_rows
    )

    audit_columns = [
        "external_role",
        "external_target_class",
        "primary_evaluation_included",
        "landsat_product_id",
        "acquisition_time_utc",
        "flow_at_scene_kg_h",
        "temporal_z_source_p95",
        "cal_temporal_z_source_p95_z",
        "cal_temporal_z_source_p95_percentile",
        "source_valid_fraction",
    ]

    audit = external[
        [
            column
            for column
            in audit_columns
            if column in external.columns
        ]
    ].copy()

    PREDICTION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTION_OUTPUT,
        index=False,
    )

    metrics.to_csv(
        METRICS_OUTPUT,
        index=False,
    )

    parameters.to_csv(
        MODEL_OUTPUT,
        index=False,
    )

    audit.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 108)
    print("EXTERNAL VALIDATION METRICS")
    print("=" * 108)

    display_columns = [
        "model_name",
        "feature_set",
        "test_rows",
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

    print(
        metrics[
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

    print("\n" + "=" * 108)
    print("PRIMARY MODEL INDIVIDUAL PREDICTIONS")
    print("=" * 108)

    primary = predictions[
        (
            predictions["model_name"]
            == "logistic_regression"
        )
        & (
            predictions["feature_set"]
            == "primary_percentile"
        )
        & predictions[
            "primary_evaluation_included"
        ]
    ].copy()

    prediction_display = [
        "external_target_class",
        "landsat_product_id",
        "acquisition_time_utc",
        "flow_at_scene_kg_h",
        "cal_temporal_z_source_p95_percentile",
        "predicted_probability",
        "predicted_label",
        "external_target",
        "correct_when_included",
    ]

    print(
        primary[
            prediction_display
        ].sort_values(
            [
                "external_target",
                "predicted_probability",
            ],
            ascending=[
                False,
                False,
            ],
        ).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}",
        )
    )

    print("\nSaved:")
    print(PREDICTION_OUTPUT)
    print(METRICS_OUTPUT)
    print(MODEL_OUTPUT)
    print(AUDIT_OUTPUT)


if __name__ == "__main__":
    main()
