from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from sklearn.compose import ColumnTransformer
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


MANIFEST_INPUT = Path(
    "outputs/421_landsat_35_scene_benchmark_manifest_v1.csv"
)

FEATURE_OUTPUT = Path(
    "outputs/424_landsat_35_scene_features_v1.csv"
)

PREDICTION_OUTPUT = Path(
    "outputs/425_landsat_35_scene_grouped_predictions_v1.csv"
)

METRIC_OUTPUT = Path(
    "outputs/426_landsat_35_scene_grouped_metrics_v1.csv"
)

FOLD_OUTPUT = Path(
    "outputs/427_landsat_35_scene_grouped_fold_metrics_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/428_landsat_35_scene_grouped_report_v1.txt"
)


BAND_NAMES = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
]

# Landsat Collection 2 Level-2 surface-reflectance scale.
SR_SCALE = 0.0000275
SR_OFFSET = -0.2

EPSILON = 1e-6


SWIR_FEATURES = [
    "full_swir_diff_mean",
    "full_swir_diff_p95",
    "full_swir_nd_mean",
    "full_swir_nd_p95",
    "core_swir_diff_mean",
    "core_swir_nd_mean",
    "inner_swir_diff_p95",
    "inner_swir_nd_p95",
    "core_minus_ring_swir_diff_mean",
    "core_minus_ring_swir_nd_mean",
]


CONTEXT_FEATURES = SWIR_FEATURES + [
    "full_nir_p95",
    "full_swir1_p95",
    "full_swir2_p95",
    "core_ndvi_mean",
    "core_nbr_mean",
    "inner_ndvi_p95",
    "inner_nbr_p95",
    "inner_minus_full_swir_diff_p95",
    "inner_minus_full_swir_nd_p95",
]


def safe_divide(numerator, denominator):
    result = np.full(
        numerator.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (np.abs(denominator) > EPSILON)
    )

    result[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    return result


def calculate_statistics(values, prefix):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_p99": np.nan,
        }

    return {
        f"{prefix}_count":
            int(values.size),

        f"{prefix}_mean":
            float(np.mean(values)),

        f"{prefix}_std":
            float(np.std(values)),

        f"{prefix}_p10":
            float(np.percentile(values, 10)),

        f"{prefix}_p50":
            float(np.percentile(values, 50)),

        f"{prefix}_p90":
            float(np.percentile(values, 90)),

        f"{prefix}_p95":
            float(np.percentile(values, 95)),

        f"{prefix}_p99":
            float(np.percentile(values, 99)),
    }


def square_region_mask(height, width, half_size):
    center_y = (height - 1) / 2
    center_x = (width - 1) / 2

    y_indices, x_indices = np.indices(
        (height, width)
    )

    return (
        (np.abs(y_indices - center_y) <= half_size)
        & (np.abs(x_indices - center_x) <= half_size)
    )


def load_reflectance(path):
    with rasterio.open(path) as source:
        raw = source.read().astype(
            np.float64
        )

        metadata = {
            "band_count": source.count,
            "height": source.height,
            "width": source.width,
            "crs": str(source.crs),
        }

    if raw.shape[0] != 6:
        raise RuntimeError(
            f"{path} has {raw.shape[0]} bands; expected 6."
        )

    valid = (
        np.all(np.isfinite(raw), axis=0)
        & np.all(raw != 0, axis=0)
    )

    valid_values = raw[:, valid]

    if valid_values.size == 0:
        raise RuntimeError(
            f"{path} has no valid pixels."
        )

    median_absolute_value = float(
        np.nanmedian(
            np.abs(valid_values)
        )
    )

    # GEE exports of SR_B bands usually retain integer DN values.
    # If values already look like reflectance, do not scale again.
    if median_absolute_value > 2:
        reflectance = (
            raw * SR_SCALE
            + SR_OFFSET
        )

        scale_status = (
            "landsat_c2_scale_and_offset_applied"
        )

    else:
        reflectance = raw.copy()
        scale_status = (
            "already_reflectance_no_scaling"
        )

    reflectance[:, ~valid] = np.nan

    metadata.update({
        "valid_pixel_count":
            int(valid.sum()),

        "valid_pixel_fraction":
            float(valid.mean()),

        "median_absolute_raw_value":
            median_absolute_value,

        "reflectance_scaling":
            scale_status,
    })

    return reflectance, valid, metadata


def extract_scene_features(row):
    patch_path = Path(
        str(row["patch_path"])
    )

    reflectance, valid_mask, metadata = (
        load_reflectance(
            patch_path
        )
    )

    _, height, width = reflectance.shape

    full_mask = valid_mask

    # Approximately:
    # core  = 5 × 5 pixels  ≈ 150 × 150 m
    # inner = 11 × 11 pixels ≈ 330 × 330 m
    core_geometry = square_region_mask(
        height,
        width,
        half_size=2,
    )

    inner_geometry = square_region_mask(
        height,
        width,
        half_size=5,
    )

    core_mask = (
        full_mask
        & core_geometry
    )

    inner_mask = (
        full_mask
        & inner_geometry
    )

    ring_mask = (
        full_mask
        & inner_geometry
        & ~core_geometry
    )

    band_arrays = {
        band_name: reflectance[index]
        for index, band_name
        in enumerate(BAND_NAMES)
    }

    blue = band_arrays["blue"]
    green = band_arrays["green"]
    red = band_arrays["red"]
    nir = band_arrays["nir"]
    swir1 = band_arrays["swir1"]
    swir2 = band_arrays["swir2"]

    index_arrays = {
        "swir_diff":
            swir2 - swir1,

        "swir_nd":
            safe_divide(
                swir2 - swir1,
                np.abs(swir2)
                + np.abs(swir1),
            ),

        "ndvi":
            safe_divide(
                nir - red,
                np.abs(nir)
                + np.abs(red),
            ),

        "nbr":
            safe_divide(
                nir - swir2,
                np.abs(nir)
                + np.abs(swir2),
            ),

        "swir1_nir_nd":
            safe_divide(
                swir1 - nir,
                np.abs(swir1)
                + np.abs(nir),
            ),
    }

    regions = {
        "full": full_mask,
        "core": core_mask,
        "inner": inner_mask,
        "ring": ring_mask,
    }

    features = {
        "sample_id": row["sample_id"],
        "label": int(row["label"]),
        "matched_positive_id":
            row["matched_positive_id"],
        "site_alias": row["site_alias"],
        "landsat_sensor":
            row["landsat_sensor"],
        "sample_role": row["sample_role"],
        "patch_path": str(patch_path),
        **metadata,
    }

    all_arrays = {
        **band_arrays,
        **index_arrays,
    }

    for region_name, region_mask in regions.items():
        features[
            f"{region_name}_valid_pixel_count"
        ] = int(region_mask.sum())

        for variable_name, variable_array in (
            all_arrays.items()
        ):
            features.update(
                calculate_statistics(
                    variable_array[region_mask],
                    prefix=(
                        f"{region_name}_"
                        f"{variable_name}"
                    ),
                )
            )

    for variable_name, variable_array in (
        all_arrays.items()
    ):
        core_values = variable_array[
            core_mask
        ]

        ring_values = variable_array[
            ring_mask
        ]

        inner_values = variable_array[
            inner_mask
        ]

        full_values = variable_array[
            full_mask
        ]

        features[
            f"core_minus_ring_"
            f"{variable_name}_mean"
        ] = (
            float(np.nanmean(core_values))
            - float(np.nanmean(ring_values))
        )

        features[
            f"inner_minus_full_"
            f"{variable_name}_p95"
        ] = (
            float(
                np.nanpercentile(
                    inner_values,
                    95,
                )
            )
            - float(
                np.nanpercentile(
                    full_values,
                    95,
                )
            )
        )

    return features


def create_logistic_pipeline(feature_columns):
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
                class_weight="balanced",
                C=1.0,
                solver="liblinear",
                max_iter=5000,
                random_state=42,
            ),
        ),
    ])


def calculate_metrics(
    true_labels,
    predicted_labels,
    probabilities,
):
    return {
        "accuracy":
            accuracy_score(
                true_labels,
                predicted_labels,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                true_labels,
                predicted_labels,
            ),

        "roc_auc":
            roc_auc_score(
                true_labels,
                probabilities,
            ),

        "precision_0":
            precision_score(
                true_labels,
                predicted_labels,
                pos_label=0,
                zero_division=0,
            ),

        "recall_0":
            recall_score(
                true_labels,
                predicted_labels,
                pos_label=0,
                zero_division=0,
            ),

        "f1_0":
            f1_score(
                true_labels,
                predicted_labels,
                pos_label=0,
                zero_division=0,
            ),

        "precision_1":
            precision_score(
                true_labels,
                predicted_labels,
                pos_label=1,
                zero_division=0,
            ),

        "recall_1":
            recall_score(
                true_labels,
                predicted_labels,
                pos_label=1,
                zero_division=0,
            ),

        "f1_1":
            f1_score(
                true_labels,
                predicted_labels,
                pos_label=1,
                zero_division=0,
            ),
    }


def main():
    manifest = pd.read_csv(
        MANIFEST_INPUT,
        low_memory=False,
    )

    required_columns = [
        "sample_id",
        "label",
        "matched_positive_id",
        "site_alias",
        "landsat_sensor",
        "sample_role",
        "patch_path",
        "benchmark_ready",
    ]

    missing = [
        column
        for column in required_columns
        if column not in manifest.columns
    ]

    if missing:
        raise KeyError(
            "Benchmark manifest missing columns: "
            + ", ".join(missing)
        )

    ready = (
        manifest[
            "benchmark_ready"
        ]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    manifest = manifest[
        ready
    ].copy()

    if len(manifest) != 35:
        raise RuntimeError(
            "Expected 35 benchmark-ready scenes, "
            f"found {len(manifest)}."
        )

    print("=" * 105)
    print(
        "LANDSAT 35-SCENE FEATURE EXTRACTION"
    )
    print("=" * 105)

    feature_rows = []

    for number, (_, row) in enumerate(
        manifest.iterrows(),
        start=1,
    ):
        print(
            f"[{number:02d}/{len(manifest):02d}] "
            f"{row['sample_id']}",
            flush=True,
        )

        feature_rows.append(
            extract_scene_features(row)
        )

    features = pd.DataFrame(
        feature_rows
    )

    features.to_csv(
        FEATURE_OUTPUT,
        index=False,
    )

    print(
        "\nFeature table:",
        features.shape,
    )

    for column in (
        SWIR_FEATURES
        + CONTEXT_FEATURES
    ):
        if column not in features.columns:
            raise KeyError(
                f"Required model feature missing: {column}"
            )

    model_definitions = {
        "dummy_prior": {
            "features": SWIR_FEATURES,
            "model": DummyClassifier(
                strategy="prior"
            ),
        },

        "logistic_swir": {
            "features": SWIR_FEATURES,
            "model": create_logistic_pipeline(
                SWIR_FEATURES
            ),
        },

        "logistic_context": {
            "features": CONTEXT_FEATURES,
            "model": create_logistic_pipeline(
                CONTEXT_FEATURES
            ),
        },
    }

    labels = features[
        "label"
    ].astype(int).to_numpy()

    groups = features[
        "matched_positive_id"
    ].astype(str).to_numpy()

    logo = LeaveOneGroupOut()

    prediction_rows = []
    fold_metric_rows = []

    unique_groups = sorted(
        features[
            "matched_positive_id"
        ].astype(str).unique()
    )

    print("\n" + "=" * 105)
    print(
        "LEAVE-ONE-MATCHED-GROUP-OUT EVALUATION"
    )
    print("=" * 105)

    for model_name, definition in (
        model_definitions.items()
    ):
        feature_columns = definition[
            "features"
        ]

        print(
            f"\nModel: {model_name}"
        )

        for fold_number, (
            train_indices,
            test_indices,
        ) in enumerate(
            logo.split(
                features,
                labels,
                groups,
            ),
            start=1,
        ):
            held_out_group = str(
                groups[test_indices][0]
            )

            model = definition["model"]

            train_x = features.iloc[
                train_indices
            ][feature_columns]

            test_x = features.iloc[
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

            probability = model.predict_proba(
                test_x
            )[:, 1]

            prediction = (
                probability >= 0.5
            ).astype(int)

            fold_metrics = calculate_metrics(
                test_y,
                prediction,
                probability,
            )

            fold_metric_rows.append({
                "model":
                    model_name,

                "fold":
                    fold_number,

                "held_out_group":
                    held_out_group,

                "test_count":
                    len(test_indices),

                "test_positive_count":
                    int(
                        np.sum(test_y == 1)
                    ),

                "test_negative_count":
                    int(
                        np.sum(test_y == 0)
                    ),

                **fold_metrics,
            })

            for local_position, row_index in (
                enumerate(test_indices)
            ):
                source_row = features.iloc[
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
                        source_row[
                            "sample_id"
                        ],

                    "matched_positive_id":
                        source_row[
                            "matched_positive_id"
                        ],

                    "site_alias":
                        source_row[
                            "site_alias"
                        ],

                    "landsat_sensor":
                        source_row[
                            "landsat_sensor"
                        ],

                    "sample_role":
                        source_row[
                            "sample_role"
                        ],

                    "true_label":
                        int(
                            test_y[
                                local_position
                            ]
                        ),

                    "predicted_label":
                        int(
                            prediction[
                                local_position
                            ]
                        ),

                    "probability_positive":
                        float(
                            probability[
                                local_position
                            ]
                        ),

                    "correct":
                        bool(
                            prediction[
                                local_position
                            ]
                            ==
                            test_y[
                                local_position
                            ]
                        ),
                })

            print(
                f"  Fold {fold_number}: "
                f"{held_out_group} | "
                f"BA={fold_metrics['balanced_accuracy']:.3f} | "
                f"AUC={fold_metrics['roc_auc']:.3f}"
            )

    predictions = pd.DataFrame(
        prediction_rows
    )

    fold_metrics = pd.DataFrame(
        fold_metric_rows
    )

    predictions.to_csv(
        PREDICTION_OUTPUT,
        index=False,
    )

    fold_metrics.to_csv(
        FOLD_OUTPUT,
        index=False,
    )

    overall_metric_rows = []

    for model_name, group in (
        predictions.groupby("model")
    ):
        overall_metrics = calculate_metrics(
            group["true_label"],
            group["predicted_label"],
            group[
                "probability_positive"
            ],
        )

        overall_metric_rows.append({
            "model": model_name,
            "evaluation":
                "pooled_leave_one_matched_group_out",

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

            **overall_metrics,
        })

    metrics = pd.DataFrame(
        overall_metric_rows
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

    wrong_predictions = predictions[
        ~predictions["correct"]
    ].copy()

    wrong_by_model = (
        wrong_predictions[
            "model"
        ].value_counts()
    )

    report_lines = [
        "=" * 105,
        (
            "LANDSAT 35-SCENE GROUPED "
            "BASELINE REPORT V1"
        ),
        "=" * 105,
        "",
        f"Benchmark scenes: {len(features)}",
        (
            "Matched groups: "
            f"{features['matched_positive_id'].nunique()}"
        ),
        (
            "Positive scenes: "
            f"{int(features['label'].eq(1).sum())}"
        ),
        (
            "Negative scenes: "
            f"{int(features['label'].eq(0).sum())}"
        ),
        "",
        (
            "Evaluation: pooled leave-one-matched-"
            "group-out cross-validation"
        ),
        (
            "Each held-out fold contains one positive "
            "and its four matched negatives."
        ),
        (
            "Classification threshold is fixed at 0.5; "
            "no test-fold threshold tuning."
        ),
        "",
        "Overall metrics:",
        metrics.to_string(index=False),
        "",
        "Wrong predictions by model:",
        wrong_by_model.to_string(),
        "",
        "Interpretation warning:",
        (
            "This is a small seven-positive controlled-"
            "release benchmark. Results estimate baseline "
            "separability but do not establish a deployable "
            "methane detector."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 105)
    print("POOLED GROUPED RESULTS")
    print("=" * 105)

    print(
        "\n",
        metrics.to_string(index=False),
    )

    print("\nWrong predictions by model:")
    print(wrong_by_model)

    print("\nSaved:")
    print(FEATURE_OUTPUT)
    print(PREDICTION_OUTPUT)
    print(METRIC_OUTPUT)
    print(FOLD_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
