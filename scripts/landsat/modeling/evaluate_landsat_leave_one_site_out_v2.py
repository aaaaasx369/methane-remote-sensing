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


DATASETS = {
    "strict_core_v2": Path(
        "outputs/95_landsat_strict_core_v2_features.csv"
    ),
    "extended_v2": Path(
        "outputs/96_landsat_extended_v2_features.csv"
    ),
}

FOLD_METRICS_OUTPUT = Path(
    "outputs/99_landsat_leave_one_site_out_fold_metrics.csv"
)

PREDICTIONS_OUTPUT = Path(
    "outputs/100_landsat_leave_one_site_out_predictions.csv"
)

POOLED_SUMMARY_OUTPUT = Path(
    "outputs/101_landsat_leave_one_site_out_summary.csv"
)

FEATURE_CHECK_OUTPUT = Path(
    "outputs/102_landsat_leave_one_site_out_feature_check.csv"
)


FEATURE_SETS = {
    "swir_means_2": [
        "swir1_mean",
        "swir2_mean",
    ],
    "swir_ratio_3": [
        "swir1_mean",
        "swir2_mean",
        "log_swir1_over_swir2_mean",
    ],
    "context_5": [
        "swir1_mean",
        "swir2_mean",
        "log_swir1_over_swir2_mean",
        "log_swir1_over_swir2_standardized_contrast",
        "ndvi_mean",
    ],
}


def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
):
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(
            y_true,
            y_probability,
        )
    else:
        roc_auc = np.nan

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

    return {
        "accuracy": accuracy,
        "balanced_accuracy":
            balanced_accuracy,
        "precision_positive":
            precision,
        "recall_positive":
            recall,
        "recall_negative":
            recall_negative,
        "f1_positive":
            f1,
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
                C=1.0,
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

    classifier = model.named_steps[
        "classifier"
    ]

    classes = list(
        classifier.classes_
    )

    if 1 not in classes:
        return np.zeros(
            len(x_test),
            dtype=float,
        )

    positive_index = classes.index(1)

    return probabilities[
        :,
        positive_index
    ]


def get_scene_metadata(row):
    preferred_columns = [
        "scene_key",
        "overpass_id",
        "raster_group_id",
        "event_id",
        "label",
        "site_key_normalized",
        "landsat_sensor",
        "landsat_product_id_normalized",
        "acquisition_time_utc",
        "resolved_patch_path",
        "canonical_pixel_hash",
        "source_dataset",
    ]

    return {
        column: row.get(
            column,
            np.nan,
        )
        for column in preferred_columns
    }


def evaluate_model(
    dataset_name,
    dataframe,
    model_name,
    feature_set_name,
    feature_columns,
    model_builder,
):
    fold_metric_rows = []
    prediction_rows = []

    sites = sorted(
        dataframe[
            "site_key_normalized"
        ].dropna().unique()
    )

    if len(sites) != 2:
        raise ValueError(
            f"{dataset_name}: expected exactly "
            f"two sites, found {sites}."
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

        train_sites = sorted(
            train[
                "site_key_normalized"
            ].unique()
        )

        if train["label"].nunique() != 2:
            raise ValueError(
                f"{dataset_name}, test site "
                f"{test_site}: training set "
                "does not contain both labels."
            )

        x_train = train[
            feature_columns
        ]

        y_train = train[
            "label"
        ].astype(int)

        x_test = test[
            feature_columns
        ]

        y_test = test[
            "label"
        ].astype(int)

        model = model_builder()

        model.fit(
            x_train,
            y_train,
        )

        predicted_label = model.predict(
            x_test
        ).astype(int)

        predicted_probability = (
            positive_probability(
                model,
                x_test,
            )
        )

        fold_metrics = calculate_metrics(
            y_test,
            predicted_label,
            predicted_probability,
        )

        fold_metric_rows.append({
            "dataset_name":
                dataset_name,
            "model_name":
                model_name,
            "feature_set":
                feature_set_name,
            "train_sites":
                "|".join(train_sites),
            "test_site":
                test_site,
            "train_rows":
                len(train),
            "test_rows":
                len(test),
            "train_label_0":
                int(
                    (y_train == 0).sum()
                ),
            "train_label_1":
                int(
                    (y_train == 1).sum()
                ),
            "test_label_0":
                int(
                    (y_test == 0).sum()
                ),
            "test_label_1":
                int(
                    (y_test == 1).sum()
                ),
            **fold_metrics,
        })

        for position, (
            dataframe_index,
            row,
        ) in enumerate(
            test.iterrows()
        ):
            prediction_row = (
                get_scene_metadata(row)
            )

            prediction_row.update({
                "dataset_name":
                    dataset_name,
                "model_name":
                    model_name,
                "feature_set":
                    feature_set_name,
                "train_sites":
                    "|".join(train_sites),
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
                        predicted_label[
                            position
                        ]
                    ),
                "predicted_probability":
                    float(
                        predicted_probability[
                            position
                        ]
                    ),
                "correct":
                    bool(
                        predicted_label[
                            position
                        ]
                        == y_test.loc[
                            dataframe_index
                        ]
                    ),
            })

            prediction_rows.append(
                prediction_row
            )

    return (
        fold_metric_rows,
        prediction_rows,
    )


def build_pooled_summary(
    predictions,
):
    summary_rows = []

    group_columns = [
        "dataset_name",
        "model_name",
        "feature_set",
    ]

    for group_values, group in (
        predictions.groupby(
            group_columns,
            dropna=False,
        )
    ):
        (
            dataset_name,
            model_name,
            feature_set,
        ) = group_values

        y_true = group[
            "actual_label"
        ].astype(int)

        y_pred = group[
            "predicted_label"
        ].astype(int)

        probability = group[
            "predicted_probability"
        ].astype(float)

        metrics = calculate_metrics(
            y_true,
            y_pred,
            probability,
        )

        summary_rows.append({
            "dataset_name":
                dataset_name,
            "model_name":
                model_name,
            "feature_set":
                feature_set,
            "pooled_test_rows":
                len(group),
            **metrics,
        })

    return pd.DataFrame(
        summary_rows
    )


def main():
    print("=" * 105)
    print("LANDSAT LEAVE-ONE-SITE-OUT EVALUATION")
    print("=" * 105)

    all_fold_metrics = []
    all_predictions = []
    feature_check_rows = []

    for dataset_name, path in (
        DATASETS.items()
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {path}"
            )

        dataframe = pd.read_csv(
            path,
            low_memory=False,
        )

        required_metadata = [
            "label",
            "site_key_normalized",
        ]

        missing_metadata = [
            column
            for column in required_metadata
            if column
            not in dataframe.columns
        ]

        if missing_metadata:
            raise KeyError(
                f"{dataset_name}: missing "
                f"metadata columns "
                f"{missing_metadata}."
            )

        dataframe["label"] = (
            pd.to_numeric(
                dataframe["label"],
                errors="raise",
            ).astype(int)
        )

        print("\n" + "-" * 105)
        print(dataset_name.upper())
        print("-" * 105)

        print(f"\nRows: {len(dataframe)}")

        print("\nLabel by site:")
        print(
            pd.crosstab(
                dataframe[
                    "site_key_normalized"
                ],
                dataframe["label"],
                margins=True,
            )
        )

        valid_feature_sets = {}

        for (
            feature_set_name,
            feature_columns,
        ) in FEATURE_SETS.items():
            missing = [
                column
                for column in feature_columns
                if column
                not in dataframe.columns
            ]

            feature_check_rows.append({
                "dataset_name":
                    dataset_name,
                "feature_set":
                    feature_set_name,
                "requested_feature_count":
                    len(feature_columns),
                "available_feature_count":
                    len(feature_columns)
                    - len(missing),
                "missing_features":
                    " | ".join(missing),
                "feature_set_available":
                    len(missing) == 0,
            })

            if len(missing) == 0:
                valid_feature_sets[
                    feature_set_name
                ] = feature_columns

                print(
                    f"[AVAILABLE] "
                    f"{feature_set_name}: "
                    f"{feature_columns}"
                )

            else:
                print(
                    f"[SKIP] "
                    f"{feature_set_name}; "
                    f"missing: {missing}"
                )

        if len(valid_feature_sets) == 0:
            raise RuntimeError(
                f"{dataset_name}: no requested "
                "feature set is available."
            )

        # Dummy baseline only needs one numeric
        # feature because it ignores the feature values.
        dummy_feature_name = next(
            iter(
                valid_feature_sets.values()
            )
        )[0]

        dummy_folds, dummy_predictions = (
            evaluate_model(
                dataset_name=
                    dataset_name,
                dataframe=
                    dataframe,
                model_name=
                    "dummy_prior",
                feature_set_name=
                    "dummy",
                feature_columns=[
                    dummy_feature_name
                ],
                model_builder=
                    build_dummy_model,
            )
        )

        all_fold_metrics.extend(
            dummy_folds
        )

        all_predictions.extend(
            dummy_predictions
        )

        for (
            feature_set_name,
            feature_columns,
        ) in valid_feature_sets.items():
            folds, predictions = (
                evaluate_model(
                    dataset_name=
                        dataset_name,
                    dataframe=
                        dataframe,
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

            all_fold_metrics.extend(
                folds
            )

            all_predictions.extend(
                predictions
            )

    fold_metrics_df = pd.DataFrame(
        all_fold_metrics
    )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    pooled_summary_df = (
        build_pooled_summary(
            predictions_df
        )
    )

    feature_check_df = pd.DataFrame(
        feature_check_rows
    )

    FOLD_METRICS_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_metrics_df.to_csv(
        FOLD_METRICS_OUTPUT,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_OUTPUT,
        index=False,
    )

    pooled_summary_df.to_csv(
        POOLED_SUMMARY_OUTPUT,
        index=False,
    )

    feature_check_df.to_csv(
        FEATURE_CHECK_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 105)
    print("POOLED OUT-OF-SITE PERFORMANCE")
    print("=" * 105)

    display_columns = [
        "dataset_name",
        "model_name",
        "feature_set",
        "pooled_test_rows",
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
        pooled_summary_df[
            display_columns
        ].sort_values(
            [
                "dataset_name",
                "balanced_accuracy",
                "roc_auc",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        ).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}"
        )
    )

    print("\n" + "=" * 105)
    print("PER-SITE FOLD PERFORMANCE")
    print("=" * 105)

    fold_display_columns = [
        "dataset_name",
        "model_name",
        "feature_set",
        "train_sites",
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

    print(
        fold_metrics_df[
            fold_display_columns
        ].sort_values(
            [
                "dataset_name",
                "model_name",
                "feature_set",
                "test_site",
            ]
        ).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}"
        )
    )

    print("\nSaved:")
    print(FOLD_METRICS_OUTPUT)
    print(PREDICTIONS_OUTPUT)
    print(POOLED_SUMMARY_OUTPUT)
    print(FEATURE_CHECK_OUTPUT)


if __name__ == "__main__":
    main()
