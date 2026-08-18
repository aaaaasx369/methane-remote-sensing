from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_INPUT = Path(
    "outputs/424_landsat_35_scene_features_v1.csv"
)

TEMPORAL_FEATURE_OUTPUT = Path(
    "outputs/429_landsat_temporal_anomaly_features_v1.csv"
)

PREDICTION_OUTPUT = Path(
    "outputs/430_landsat_temporal_anomaly_predictions_v1.csv"
)

METRIC_OUTPUT = Path(
    "outputs/431_landsat_temporal_anomaly_metrics_v1.csv"
)

FOLD_OUTPUT = Path(
    "outputs/432_landsat_temporal_anomaly_fold_metrics_v1.csv"
)

RANK_OUTPUT = Path(
    "outputs/433_landsat_temporal_anomaly_positive_ranks_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/434_landsat_temporal_anomaly_report_v1.txt"
)


SWIR_RAW_FEATURES = [
    "full_swir_diff_p95",
    "full_swir_nd_p95",
    "core_swir_diff_mean",
    "core_swir_nd_mean",
    "inner_swir_diff_p95",
    "inner_swir_nd_p95",
    "core_minus_ring_swir_diff_mean",
    "core_minus_ring_swir_nd_mean",
]

CONTEXT_RAW_FEATURES = [
    "core_ndvi_mean",
    "core_nbr_mean",
    "inner_ndvi_p95",
    "inner_nbr_p95",
]

ALL_RAW_FEATURES = (
    SWIR_RAW_FEATURES
    + CONTEXT_RAW_FEATURES
)

SWIR_MODEL_FEATURES = [
    "temporal_swir_abs_z_mean",
    "temporal_swir_abs_z_max",
    "temporal_swir_signed_z_mean",
    "temporal_swir_diff_abs_z_mean",
    "temporal_swir_nd_abs_z_mean",
]

CONTEXT_MODEL_FEATURES = (
    SWIR_MODEL_FEATURES
    + [
        "temporal_context_abs_z_mean",
        "temporal_context_abs_z_max",
        "temporal_context_signed_z_mean",
    ]
)

EPSILON = 1e-9


def robust_scale(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    median = np.median(values)

    mad = np.median(
        np.abs(values - median)
    )

    scale = 1.4826 * mad

    if (
        not np.isfinite(scale)
        or scale <= EPSILON
    ):
        q25, q75 = np.percentile(
            values,
            [25, 75],
        )

        scale = (
            q75 - q25
        ) / 1.349

    if (
        not np.isfinite(scale)
        or scale <= EPSILON
    ):
        scale = np.std(values)

    if (
        not np.isfinite(scale)
        or scale <= EPSILON
    ):
        scale = 1.0

    return float(scale)


def add_leave_one_out_temporal_features(group):
    group = group.copy()

    for feature in ALL_RAW_FEATURES:
        values = pd.to_numeric(
            group[feature],
            errors="coerce",
        ).to_numpy(dtype=float)

        z_values = []
        delta_values = []

        for index, value in enumerate(values):
            peer_values = np.delete(
                values,
                index,
            )

            peer_values = peer_values[
                np.isfinite(peer_values)
            ]

            if (
                not np.isfinite(value)
                or len(peer_values) == 0
            ):
                z_values.append(np.nan)
                delta_values.append(np.nan)
                continue

            peer_median = float(
                np.median(peer_values)
            )

            scale = robust_scale(
                peer_values
            )

            delta = (
                value - peer_median
            )

            z_score = (
                delta / scale
            )

            delta_values.append(
                float(delta)
            )

            z_values.append(
                float(z_score)
            )

        group[
            f"{feature}_loo_delta"
        ] = delta_values

        group[
            f"{feature}_loo_z"
        ] = z_values

        group[
            f"{feature}_loo_abs_z"
        ] = np.abs(z_values)

    swir_z_columns = [
        f"{feature}_loo_z"
        for feature in SWIR_RAW_FEATURES
    ]

    swir_abs_columns = [
        f"{feature}_loo_abs_z"
        for feature in SWIR_RAW_FEATURES
    ]

    swir_diff_abs_columns = [
        f"{feature}_loo_abs_z"
        for feature in SWIR_RAW_FEATURES
        if "swir_diff" in feature
    ]

    swir_nd_abs_columns = [
        f"{feature}_loo_abs_z"
        for feature in SWIR_RAW_FEATURES
        if "swir_nd" in feature
    ]

    context_z_columns = [
        f"{feature}_loo_z"
        for feature in CONTEXT_RAW_FEATURES
    ]

    context_abs_columns = [
        f"{feature}_loo_abs_z"
        for feature in CONTEXT_RAW_FEATURES
    ]

    group[
        "temporal_swir_abs_z_mean"
    ] = group[
        swir_abs_columns
    ].mean(axis=1)

    group[
        "temporal_swir_abs_z_max"
    ] = group[
        swir_abs_columns
    ].max(axis=1)

    group[
        "temporal_swir_signed_z_mean"
    ] = group[
        swir_z_columns
    ].mean(axis=1)

    group[
        "temporal_swir_diff_abs_z_mean"
    ] = group[
        swir_diff_abs_columns
    ].mean(axis=1)

    group[
        "temporal_swir_nd_abs_z_mean"
    ] = group[
        swir_nd_abs_columns
    ].mean(axis=1)

    group[
        "temporal_context_abs_z_mean"
    ] = group[
        context_abs_columns
    ].mean(axis=1)

    group[
        "temporal_context_abs_z_max"
    ] = group[
        context_abs_columns
    ].max(axis=1)

    group[
        "temporal_context_signed_z_mean"
    ] = group[
        context_z_columns
    ].mean(axis=1)

    return group


def make_logistic_pipeline():
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
                class_weight="balanced",
                C=0.2,
                solver="liblinear",
                max_iter=5000,
                random_state=42,
            ),
        ),
    ])


def calculate_metrics(
    true_labels,
    predictions,
    probabilities,
):
    return {
        "accuracy":
            accuracy_score(
                true_labels,
                predictions,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                true_labels,
                predictions,
            ),

        "roc_auc":
            roc_auc_score(
                true_labels,
                probabilities,
            ),

        "precision_0":
            precision_score(
                true_labels,
                predictions,
                pos_label=0,
                zero_division=0,
            ),

        "recall_0":
            recall_score(
                true_labels,
                predictions,
                pos_label=0,
                zero_division=0,
            ),

        "f1_0":
            f1_score(
                true_labels,
                predictions,
                pos_label=0,
                zero_division=0,
            ),

        "precision_1":
            precision_score(
                true_labels,
                predictions,
                pos_label=1,
                zero_division=0,
            ),

        "recall_1":
            recall_score(
                true_labels,
                predictions,
                pos_label=1,
                zero_division=0,
            ),

        "f1_1":
            f1_score(
                true_labels,
                predictions,
                pos_label=1,
                zero_division=0,
            ),
    }


def build_rank_table(features):
    rows = []

    for group_id, group in features.groupby(
        "matched_positive_id",
        sort=False,
    ):
        group = group.copy()

        positive = group[
            group["label"].eq(1)
        ]

        negatives = group[
            group["label"].eq(0)
        ]

        if (
            len(positive) != 1
            or len(negatives) != 4
        ):
            raise RuntimeError(
                f"{group_id} 不是 1 positive + 4 negatives。"
            )

        positive = positive.iloc[0]

        swir_rank = (
            group[
                "temporal_swir_abs_z_mean"
            ]
            .rank(
                ascending=False,
                method="min",
            )
            .loc[
                positive.name
            ]
        )

        context_score_column = (
            "temporal_context_abs_z_mean"
        )

        context_rank = (
            group[
                context_score_column
            ]
            .rank(
                ascending=False,
                method="min",
            )
            .loc[
                positive.name
            ]
        )

        rows.append({
            "matched_positive_id":
                group_id,

            "site_alias":
                positive[
                    "site_alias"
                ],

            "positive_sample_id":
                positive[
                    "sample_id"
                ],

            "positive_swir_anomaly_score":
                positive[
                    "temporal_swir_abs_z_mean"
                ],

            "maximum_negative_swir_score":
                negatives[
                    "temporal_swir_abs_z_mean"
                ].max(),

            "positive_swir_rank":
                int(swir_rank),

            "positive_swir_top1":
                bool(swir_rank == 1),

            "positive_swir_top2":
                bool(swir_rank <= 2),

            "positive_context_anomaly_score":
                positive[
                    context_score_column
                ],

            "maximum_negative_context_score":
                negatives[
                    context_score_column
                ].max(),

            "positive_context_rank":
                int(context_rank),

            "positive_context_top1":
                bool(context_rank == 1),

            "positive_context_top2":
                bool(context_rank <= 2),
        })

    return pd.DataFrame(rows)


def main():
    features = pd.read_csv(
        FEATURE_INPUT,
        low_memory=False,
    )

    required_columns = [
        "sample_id",
        "label",
        "matched_positive_id",
        "site_alias",
        "landsat_sensor",
        "sample_role",
        *ALL_RAW_FEATURES,
    ]

    missing = [
        column
        for column in required_columns
        if column not in features.columns
    ]

    if missing:
        raise KeyError(
            "Feature table 缺少欄位："
            + ", ".join(missing)
        )

    features["label"] = pd.to_numeric(
        features["label"],
        errors="raise",
    ).astype(int)

    group_counts = (
        features.groupby(
            "matched_positive_id"
        )["label"]
        .agg(
            scene_count="size",
            positive_count="sum",
        )
    )

    invalid_groups = group_counts[
        ~(
            group_counts[
                "scene_count"
            ].eq(5)
            & group_counts[
                "positive_count"
            ].eq(1)
        )
    ]

    if not invalid_groups.empty:
        raise RuntimeError(
            "部分 matched groups 不是 "
            "1 positive + 4 negatives：\n"
            + invalid_groups.to_string()
        )

    temporal_groups = []

    for _, group in features.groupby(
        "matched_positive_id",
        sort=False,
    ):
        temporal_groups.append(
            add_leave_one_out_temporal_features(
                group
            )
        )

    temporal = pd.concat(
        temporal_groups,
        ignore_index=True,
        sort=False,
    )

    temporal.to_csv(
        TEMPORAL_FEATURE_OUTPUT,
        index=False,
    )

    rank_table = build_rank_table(
        temporal
    )

    rank_table.to_csv(
        RANK_OUTPUT,
        index=False,
    )

    model_definitions = {
        "dummy_prior": {
            "features":
                SWIR_MODEL_FEATURES,

            "model":
                DummyClassifier(
                    strategy="prior",
                ),
        },

        "logistic_temporal_swir": {
            "features":
                SWIR_MODEL_FEATURES,

            "model":
                make_logistic_pipeline(),
        },

        "logistic_temporal_context": {
            "features":
                CONTEXT_MODEL_FEATURES,

            "model":
                make_logistic_pipeline(),
        },
    }

    labels = temporal[
        "label"
    ].to_numpy(dtype=int)

    groups = temporal[
        "matched_positive_id"
    ].astype(str).to_numpy()

    logo = LeaveOneGroupOut()

    prediction_rows = []
    fold_rows = []

    for model_name, definition in (
        model_definitions.items()
    ):
        feature_columns = definition[
            "features"
        ]

        for fold_number, (
            train_indices,
            test_indices,
        ) in enumerate(
            logo.split(
                temporal,
                labels,
                groups,
            ),
            start=1,
        ):
            held_out_group = str(
                groups[test_indices][0]
            )

            model = clone(
                definition["model"]
            )

            train_x = temporal.iloc[
                train_indices
            ][feature_columns]

            test_x = temporal.iloc[
                test_indices
            ][feature_columns]

            train_y = labels[
                train_indices
            ]

            test_y = labels[
                test_indices
            ]

            model.fit(
                train_x,
                train_y,
            )

            probabilities = (
                model.predict_proba(
                    test_x
                )[:, 1]
            )

            predictions = (
                probabilities >= 0.5
            ).astype(int)

            fold_metrics = (
                calculate_metrics(
                    test_y,
                    predictions,
                    probabilities,
                )
            )

            fold_rows.append({
                "model":
                    model_name,

                "fold":
                    fold_number,

                "held_out_group":
                    held_out_group,

                "test_count":
                    len(test_indices),

                **fold_metrics,
            })

            for local_index, row_index in enumerate(
                test_indices
            ):
                source = temporal.iloc[
                    row_index
                ]

                prediction_rows.append({
                    "model":
                        model_name,

                    "fold":
                        fold_number,

                    "held_out_group":
                        held_out_group,

                    "sample_id":
                        source[
                            "sample_id"
                        ],

                    "matched_positive_id":
                        source[
                            "matched_positive_id"
                        ],

                    "site_alias":
                        source[
                            "site_alias"
                        ],

                    "sample_role":
                        source[
                            "sample_role"
                        ],

                    "true_label":
                        int(
                            test_y[
                                local_index
                            ]
                        ),

                    "predicted_label":
                        int(
                            predictions[
                                local_index
                            ]
                        ),

                    "probability_positive":
                        float(
                            probabilities[
                                local_index
                            ]
                        ),

                    "correct":
                        bool(
                            predictions[
                                local_index
                            ]
                            ==
                            test_y[
                                local_index
                            ]
                        ),
                })

    predictions = pd.DataFrame(
        prediction_rows
    )

    folds = pd.DataFrame(
        fold_rows
    )

    predictions.to_csv(
        PREDICTION_OUTPUT,
        index=False,
    )

    folds.to_csv(
        FOLD_OUTPUT,
        index=False,
    )

    metric_rows = []

    for model_name, group in (
        predictions.groupby("model")
    ):
        metrics = calculate_metrics(
            group["true_label"],
            group["predicted_label"],
            group[
                "probability_positive"
            ],
        )

        metric_rows.append({
            "model":
                model_name,

            "evaluation":
                "leave_one_matched_group_out",

            "scene_count":
                len(group),

            "positive_count":
                int(
                    group[
                        "true_label"
                    ].eq(1).sum()
                ),

            "negative_count":
                int(
                    group[
                        "true_label"
                    ].eq(0).sum()
                ),

            "threshold":
                0.5,

            **metrics,
        })

    metrics = pd.DataFrame(
        metric_rows
    ).sort_values(
        [
            "balanced_accuracy",
            "roc_auc",
        ],
        ascending=False,
    )

    metrics.to_csv(
        METRIC_OUTPUT,
        index=False,
    )

    swir_top1 = int(
        rank_table[
            "positive_swir_top1"
        ].sum()
    )

    swir_top2 = int(
        rank_table[
            "positive_swir_top2"
        ].sum()
    )

    context_top1 = int(
        rank_table[
            "positive_context_top1"
        ].sum()
    )

    context_top2 = int(
        rank_table[
            "positive_context_top2"
        ].sum()
    )

    report_lines = [
        "=" * 110,
        "LANDSAT TEMPORAL-ANOMALY GROUPED REPORT V1",
        "=" * 110,
        "",
        f"Scenes: {len(temporal)}",
        (
            "Matched groups: "
            f"{temporal['matched_positive_id'].nunique()}"
        ),
        "",
        (
            "Temporal features use leave-one-scene-out "
            "normalization against the other four scenes "
            "in the same matched group."
        ),
        (
            "No labels are used when constructing the "
            "temporal anomaly features."
        ),
        (
            "The classifier is evaluated with "
            "leave-one-matched-group-out cross-validation."
        ),
        "",
        "Overall metrics:",
        metrics.to_string(index=False),
        "",
        (
            "Positive SWIR anomaly rank top-1: "
            f"{swir_top1}/7"
        ),
        (
            "Positive SWIR anomaly rank top-2: "
            f"{swir_top2}/7"
        ),
        (
            "Positive context anomaly rank top-1: "
            f"{context_top1}/7"
        ),
        (
            "Positive context anomaly rank top-2: "
            f"{context_top2}/7"
        ),
        "",
        "Per-group positive ranks:",
        rank_table.to_string(index=False),
        "",
        "Interpretation limitation:",
        (
            "This is a temporal-context benchmark. "
            "It assumes multiple matched observations of "
            "the same source are available and is not an "
            "independent single-scene detector."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 110)
    print(
        "LANDSAT TEMPORAL-ANOMALY RESULTS"
    )
    print("=" * 110)

    display_columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "precision_0",
        "recall_0",
        "precision_1",
        "recall_1",
        "f1_1",
    ]

    print(
        "\n"
        + metrics[
            display_columns
        ].to_string(index=False)
    )

    print("\nPositive anomaly ranking:")
    print(
        "SWIR top-1:",
        swir_top1,
        "/ 7",
    )
    print(
        "SWIR top-2:",
        swir_top2,
        "/ 7",
    )
    print(
        "Context top-1:",
        context_top1,
        "/ 7",
    )
    print(
        "Context top-2:",
        context_top2,
        "/ 7",
    )

    print("\nSaved:")
    print(TEMPORAL_FEATURE_OUTPUT)
    print(PREDICTION_OUTPUT)
    print(METRIC_OUTPUT)
    print(FOLD_OUTPUT)
    print(RANK_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
