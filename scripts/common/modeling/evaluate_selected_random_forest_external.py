from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from train_marss2l_site_disjoint_models import (
    MULTI_FEATURES,
    build_random_forest,
    calculate_training_weights,
)


DEVELOPMENT_INPUT = Path(
    "outputs/268_marss2l_development_model_ready.csv"
)

EXTERNAL_INPUT = Path(
    "outputs/278_marss2l_external_model_ready.csv"
)

PREDICTION_OUTPUT = Path(
    "outputs/280_marss2l_random_forest_external_predictions.csv"
)

OVERALL_OUTPUT = Path(
    "outputs/281_marss2l_random_forest_external_metrics.csv"
)

SITE_OUTPUT = Path(
    "outputs/282_marss2l_random_forest_external_site_metrics.csv"
)

SENSOR_OUTPUT = Path(
    "outputs/283_marss2l_random_forest_external_sensor_metrics.csv"
)

MODEL_OUTPUT = Path(
    "outputs/284_marss2l_final_random_forest.joblib"
)

COMPARISON_OUTPUT = Path(
    "outputs/285_marss2l_baseline_vs_random_forest.csv"
)


FROZEN_THRESHOLD = 0.34
EXPECTED_DEVELOPMENT_ROWS = 506
EXPECTED_EXTERNAL_ROWS = 162
EXPECTED_DEVELOPMENT_SITES = 62
EXPECTED_EXTERNAL_SITES = 33


def safe_ratio(a, b):
    if b == 0:
        return np.nan

    return a / b


def metric_row(
    frame,
    group_type,
    group_value,
):
    y_true = (
        frame["target_label"]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        frame["predicted_label"]
        .astype(int)
        .to_numpy()
    )

    scores = (
        frame["prediction_score"]
        .astype(float)
        .to_numpy()
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    positive_recall = safe_ratio(
        tp,
        tp + fn,
    )

    negative_recall = safe_ratio(
        tn,
        tn + fp,
    )

    return {
        "group_type":
            group_type,
        "group_value":
            group_value,
        "evaluation_count":
            len(frame),
        "negative_count":
            int((y_true == 0).sum()),
        "positive_count":
            int((y_true == 1).sum()),
        "true_negative":
            int(tn),
        "false_positive":
            int(fp),
        "false_negative":
            int(fn),
        "true_positive":
            int(tp),
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
        "positive_recall":
            positive_recall,
        "negative_recall":
            negative_recall,
        "false_positive_rate":
            safe_ratio(
                fp,
                tn + fp,
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
        "threshold":
            FROZEN_THRESHOLD,
    }


def main():
    development = pd.read_csv(
        DEVELOPMENT_INPUT,
        low_memory=False,
    )

    external = pd.read_csv(
        EXTERNAL_INPUT,
        low_memory=False,
    )

    if len(development) != (
        EXPECTED_DEVELOPMENT_ROWS
    ):
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_DEVELOPMENT_ROWS} "
            f"development rows, found "
            f"{len(development)}."
        )

    if len(external) != (
        EXPECTED_EXTERNAL_ROWS
    ):
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_EXTERNAL_ROWS} "
            f"external rows, found "
            f"{len(external)}."
        )

    if (
        development["site_key"].nunique()
        != EXPECTED_DEVELOPMENT_SITES
    ):
        raise RuntimeError(
            "Unexpected development site count."
        )

    if (
        external["site_key"].nunique()
        != EXPECTED_EXTERNAL_SITES
    ):
        raise RuntimeError(
            "Unexpected external site count."
        )

    overlap = (
        set(development["site_key"])
        & set(external["site_key"])
    )

    if overlap:
        raise RuntimeError(
            "Development/external site overlap."
        )

    development[
        "sensor_is_lc09"
    ] = (
        development[
            "sensor_code"
        ]
        .astype(str)
        .str.upper()
        .eq("LC09")
        .astype(int)
    )

    external[
        "sensor_is_lc09"
    ] = (
        external[
            "sensor_code"
        ]
        .astype(str)
        .str.upper()
        .eq("LC09")
        .astype(int)
    )

    missing_features = [
        feature
        for feature in MULTI_FEATURES
        if (
            feature
            not in development.columns
            or feature
            not in external.columns
        )
    ]

    if missing_features:
        raise KeyError(
            "Missing features:\n"
            + "\n".join(missing_features)
        )

    weights = (
        calculate_training_weights(
            development
        )
    )

    model = build_random_forest()

    model.fit(
        development[MULTI_FEATURES],
        development["target_label"],
        classifier__sample_weight=
            weights,
    )

    external[
        "prediction_score"
    ] = model.predict_proba(
        external[MULTI_FEATURES]
    )[:, 1]

    external[
        "predicted_label"
    ] = (
        external[
            "prediction_score"
        ]
        >= FROZEN_THRESHOLD
    ).astype(int)

    external[
        "correct"
    ] = (
        external["predicted_label"]
        == external["target_label"]
    )

    external[
        "model_name"
    ] = "random_forest"

    external[
        "threshold"
    ] = FROZEN_THRESHOLD

    external.to_csv(
        PREDICTION_OUTPUT,
        index=False,
    )

    site_rows = []

    for site_key, group in (
        external.groupby("site_key")
    ):
        site_rows.append(
            metric_row(
                group,
                "site",
                site_key,
            )
        )

    site_metrics = pd.DataFrame(
        site_rows
    )

    site_metrics.to_csv(
        SITE_OUTPUT,
        index=False,
    )

    overall = metric_row(
        external,
        "overall",
        "all_external_sites",
    )

    overall[
        "macro_site_balanced_accuracy"
    ] = site_metrics[
        "balanced_accuracy"
    ].mean()

    overall[
        "macro_site_positive_recall"
    ] = site_metrics[
        "positive_recall"
    ].mean()

    overall[
        "macro_site_negative_recall"
    ] = site_metrics[
        "negative_recall"
    ].mean()

    overall_frame = pd.DataFrame([
        overall
    ])

    overall_frame.to_csv(
        OVERALL_OUTPUT,
        index=False,
    )

    sensor_rows = []

    for sensor, group in (
        external.groupby("sensor_code")
    ):
        sensor_rows.append(
            metric_row(
                group,
                "sensor",
                sensor,
            )
        )

    sensor_metrics = pd.DataFrame(
        sensor_rows
    )

    sensor_metrics.to_csv(
        SENSOR_OUTPUT,
        index=False,
    )

    joblib.dump(
        {
            "model_name":
                "random_forest",
            "model":
                model,
            "feature_columns":
                MULTI_FEATURES,
            "threshold":
                FROZEN_THRESHOLD,
            "development_site_count":
                EXPECTED_DEVELOPMENT_SITES,
            "external_site_count":
                EXPECTED_EXTERNAL_SITES,
            "external_test_used_for_selection":
                False,
        },
        MODEL_OUTPUT,
    )

    baseline_path = Path(
        "outputs/"
        "239_marss2l_external_overall_metrics.csv"
    )

    comparison_rows = []

    if baseline_path.exists():
        baseline = pd.read_csv(
            baseline_path,
            low_memory=False,
        ).iloc[0]

        comparison_rows.append({
            "model_name":
                "controlled_release_single_feature",
            "balanced_accuracy":
                baseline.get(
                    "balanced_accuracy"
                ),
            "positive_recall":
                baseline.get(
                    "positive_recall"
                ),
            "negative_recall":
                baseline.get(
                    "negative_recall"
                ),
            "false_positive_rate":
                baseline.get(
                    "false_positive_rate"
                ),
            "roc_auc":
                baseline.get("roc_auc"),
            "macro_site_balanced_accuracy":
                baseline.get(
                    "macro_site_balanced_accuracy"
                ),
        })

    comparison_rows.append({
        "model_name":
            "marss2l_random_forest",
        "balanced_accuracy":
            overall[
                "balanced_accuracy"
            ],
        "positive_recall":
            overall[
                "positive_recall"
            ],
        "negative_recall":
            overall[
                "negative_recall"
            ],
        "false_positive_rate":
            overall[
                "false_positive_rate"
            ],
        "roc_auc":
            overall["roc_auc"],
        "macro_site_balanced_accuracy":
            overall[
                "macro_site_balanced_accuracy"
            ],
    })

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        COMPARISON_OUTPUT,
        index=False,
    )

    print("=" * 110)
    print("FINAL RANDOM FOREST EXTERNAL EVALUATION")
    print("=" * 110)

    print(
        overall_frame.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nResults by sensor:")
    print(
        sensor_metrics.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nBaseline comparison:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nSaved:")
    print(PREDICTION_OUTPUT)
    print(OVERALL_OUTPUT)
    print(SITE_OUTPUT)
    print(SENSOR_OUTPUT)
    print(MODEL_OUTPUT)
    print(COMPARISON_OUTPUT)


if __name__ == "__main__":
    main()
