from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT = Path(
    "outputs/429_landsat_temporal_anomaly_features_v1.csv"
)

PREDICTION_OUTPUT = Path(
    "outputs/435_landsat_temporal_context_only_predictions_v1.csv"
)

METRIC_OUTPUT = Path(
    "outputs/436_landsat_temporal_context_only_metrics_v1.csv"
)


FEATURE_COLUMNS = [
    "temporal_context_abs_z_mean",
    "temporal_context_abs_z_max",
    "temporal_context_signed_z_mean",
]


def main():
    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            "Missing features: "
            + ", ".join(missing)
        )

    y = pd.to_numeric(
        df["label"],
        errors="raise",
    ).astype(int).to_numpy()

    groups = (
        df["matched_positive_id"]
        .astype(str)
        .to_numpy()
    )

    model = Pipeline([
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
                class_weight="balanced",
                C=0.2,
                solver="liblinear",
                max_iter=5000,
                random_state=42,
            ),
        ),
    ])

    logo = LeaveOneGroupOut()
    rows = []

    for fold, (train_index, test_index) in enumerate(
        logo.split(df, y, groups),
        start=1,
    ):
        train_x = df.iloc[
            train_index
        ][FEATURE_COLUMNS]

        test_x = df.iloc[
            test_index
        ][FEATURE_COLUMNS]

        train_y = y[train_index]
        test_y = y[test_index]

        model.fit(
            train_x,
            train_y,
        )

        probability = model.predict_proba(
            test_x
        )[:, 1]

        prediction = (
            probability >= 0.5
        ).astype(int)

        for position, row_index in enumerate(
            test_index
        ):
            rows.append({
                "fold":
                    fold,

                "held_out_group":
                    groups[row_index],

                "sample_id":
                    df.iloc[row_index][
                        "sample_id"
                    ],

                "site_alias":
                    df.iloc[row_index][
                        "site_alias"
                    ],

                "true_label":
                    int(test_y[position]),

                "predicted_label":
                    int(prediction[position]),

                "probability_positive":
                    float(probability[position]),
            })

    predictions = pd.DataFrame(rows)

    true = predictions[
        "true_label"
    ]

    predicted = predictions[
        "predicted_label"
    ]

    probability = predictions[
        "probability_positive"
    ]

    metrics = pd.DataFrame([{
        "model":
            "logistic_temporal_context_only",

        "accuracy":
            accuracy_score(
                true,
                predicted,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                true,
                predicted,
            ),

        "roc_auc":
            roc_auc_score(
                true,
                probability,
            ),

        "precision_0":
            precision_score(
                true,
                predicted,
                pos_label=0,
                zero_division=0,
            ),

        "recall_0":
            recall_score(
                true,
                predicted,
                pos_label=0,
                zero_division=0,
            ),

        "precision_1":
            precision_score(
                true,
                predicted,
                pos_label=1,
                zero_division=0,
            ),

        "recall_1":
            recall_score(
                true,
                predicted,
                pos_label=1,
                zero_division=0,
            ),

        "f1_1":
            f1_score(
                true,
                predicted,
                pos_label=1,
                zero_division=0,
            ),
    }])

    predictions.to_csv(
        PREDICTION_OUTPUT,
        index=False,
    )

    metrics.to_csv(
        METRIC_OUTPUT,
        index=False,
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    print("\nConfusion counts:")

    print(
        pd.crosstab(
            predictions["true_label"],
            predictions["predicted_label"],
            margins=True,
        )
    )

    print("\nSaved:")
    print(PREDICTION_OUTPUT)
    print(METRIC_OUTPUT)


if __name__ == "__main__":
    main()
