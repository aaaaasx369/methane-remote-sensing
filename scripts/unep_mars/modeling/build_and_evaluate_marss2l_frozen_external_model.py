from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from build_evanston_external_anomaly_features import (
    calculate_scene_feature,
    empirical_percentile,
)

from calibrate_landsat_site_baseline_and_evaluate import (
    build_logistic_model,
)


# ============================================================
# Frozen inputs
# ============================================================

EXTERNAL_INDEX_INPUT = Path(
    "outputs/234_marss2l_external_patch_index_v1_2.csv"
)

DEVELOPMENT_FEATURE_INPUT = Path(
    "outputs/118_landsat_site_calibrated_anomaly_features.csv"
)

DEVELOPMENT_OOF_INPUT = Path(
    "outputs/121_landsat_site_calibrated_loso_predictions.csv"
)


# ============================================================
# Outputs
# ============================================================

FEATURE_OUTPUT = Path(
    "outputs/237_marss2l_external_temporal_features.csv"
)

PREDICTION_OUTPUT = Path(
    "outputs/238_marss2l_frozen_external_predictions.csv"
)

OVERALL_METRIC_OUTPUT = Path(
    "outputs/239_marss2l_external_overall_metrics.csv"
)

SITE_METRIC_OUTPUT = Path(
    "outputs/240_marss2l_external_site_metrics.csv"
)

SENSOR_METRIC_OUTPUT = Path(
    "outputs/241_marss2l_external_sensor_metrics.csv"
)

FLUX_METRIC_OUTPUT = Path(
    "outputs/242_marss2l_external_positive_flux_metrics.csv"
)

MODEL_CONTRACT_OUTPUT = Path(
    "outputs/243_marss2l_frozen_model_contract.csv"
)

FEATURE_AUDIT_OUTPUT = Path(
    "outputs/244_marss2l_external_feature_audit.csv"
)


# ============================================================
# Frozen analysis contract
# ============================================================

PRIMARY_MODEL_NAME = "logistic_regression"

PRIMARY_FEATURE_SET = (
    "calibrated_source_p95_percentile_1"
)

PRIMARY_FEATURE = (
    "cal_temporal_z_source_p95_percentile"
)

FROZEN_ALERT_THRESHOLD = 0.559805

HIGH_EMISSION_THRESHOLD_KG_H = 1000.0

EXPECTED_SITE_COUNT = 33
EXPECTED_CALIBRATION_COUNT = 165
EXPECTED_TEST_NEGATIVE_COUNT = 99
EXPECTED_POSITIVE_COUNT = 63
EXPECTED_TOTAL_COUNT = 327
EXPECTED_EVALUATION_COUNT = 162

FEATURE_CONTRACT_VERSION = (
    "marss2l_external_temporal_v1_"
    "evanston_function_exact"
)


def robust_calibration_scale(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        raise RuntimeError(
            "No finite calibration values."
        )

    center = float(
        np.median(values)
    )

    mad = float(
        np.median(
            np.abs(
                values - center
            )
        )
    )

    scale = 1.4826 * mad
    scale_method = "mad"

    if (
        not np.isfinite(scale)
        or scale < 1e-8
    ):
        scale = float(
            np.std(values)
        )

        scale_method = "standard_deviation"

    if (
        not np.isfinite(scale)
        or scale < 1e-8
    ):
        raise RuntimeError(
            "Calibration negative spread "
            "is too small."
        )

    return (
        center,
        scale,
        scale_method,
    )


def validate_external_index(index):
    required = [
        "download_id",
        "site_key",
        "external_role",
        "evaluation_label",
        "patch_path",
        "download_status",
        "pixel_sha256",
        "sensor_code",
    ]

    missing = [
        column
        for column in required
        if column not in index.columns
    ]

    if missing:
        raise KeyError(
            f"Missing index columns: {missing}"
        )

    index = index[
        index["download_status"].eq(
            "success"
        )
    ].copy()

    index["site_key"] = (
        index["site_key"]
        .astype(str)
        .str.strip()
    )

    index["download_id"] = (
        index["download_id"]
        .astype(str)
        .str.strip()
    )

    index["evaluation_label"] = (
        pd.to_numeric(
            index["evaluation_label"],
            errors="coerce",
        )
    )

    if len(index) != EXPECTED_TOTAL_COUNT:
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_TOTAL_COUNT} successful rows, "
            f"found {len(index)}."
        )

    if (
        index["site_key"].nunique()
        != EXPECTED_SITE_COUNT
    ):
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_SITE_COUNT} sites, found "
            f"{index['site_key'].nunique()}."
        )

    role_counts = (
        index["external_role"]
        .value_counts()
    )

    expected_roles = {
        "calibration_negative":
            EXPECTED_CALIBRATION_COUNT,
        "test_negative":
            EXPECTED_TEST_NEGATIVE_COUNT,
        "high_emission_positive":
            EXPECTED_POSITIVE_COUNT,
    }

    for role, expected_count in (
        expected_roles.items()
    ):
        actual_count = int(
            role_counts.get(
                role,
                0,
            )
        )

        if actual_count != expected_count:
            raise RuntimeError(
                f"{role}: expected "
                f"{expected_count}, found "
                f"{actual_count}."
            )

    if (
        index["pixel_sha256"].nunique()
        != EXPECTED_TOTAL_COUNT
    ):
        raise RuntimeError(
            "Raster hashes are not unique."
        )

    missing_paths = []

    for path_text in index["patch_path"]:
        path = Path(str(path_text))

        if not path.exists():
            missing_paths.append(
                str(path)
            )

    if missing_paths:
        raise FileNotFoundError(
            "Missing patch files:\n"
            + "\n".join(
                missing_paths[:20]
            )
        )

    return index


def calculate_site_features(site_rows):
    site_rows = site_rows.copy()

    site_key = str(
        site_rows["site_key"].iloc[0]
    )

    calibration = site_rows[
        site_rows["external_role"].eq(
            "calibration_negative"
        )
    ].copy()

    calibration = calibration.sort_values(
        [
            "acquisition_datetime_utc",
            "download_id",
        ],
        na_position="last",
    ).reset_index(drop=True)

    evaluation = site_rows[
        site_rows["external_role"].isin([
            "test_negative",
            "high_emission_positive",
        ])
    ].copy()

    if len(calibration) != 5:
        raise RuntimeError(
            f"{site_key}: expected 5 calibration "
            f"negatives, found {len(calibration)}."
        )

    if len(evaluation) < 4:
        raise RuntimeError(
            f"{site_key}: too few evaluation rows."
        )

    calibration_paths = {
        str(row["download_id"]):
            Path(str(row["patch_path"]))
        for _, row in calibration.iterrows()
    }

    feature_rows = []

    for _, row in site_rows.iterrows():
        output_row = row.to_dict()

        download_id = str(
            row["download_id"]
        )

        target_path = Path(
            str(row["patch_path"])
        )

        role = str(
            row["external_role"]
        )

        if role == "calibration_negative":
            background_rows = calibration[
                calibration["download_id"]
                .astype(str)
                .ne(download_id)
            ].copy()
        else:
            background_rows = calibration.copy()

        background_paths = [
            Path(str(path))
            for path
            in background_rows[
                "patch_path"
            ]
        ]

        if target_path in background_paths:
            raise RuntimeError(
                f"{download_id}: target raster "
                "was included as its own background."
            )

        output_row[
            "feature_contract_version"
        ] = FEATURE_CONTRACT_VERSION

        output_row[
            "target_patch_path"
        ] = str(target_path)

        output_row[
            "background_download_ids"
        ] = "|".join(
            background_rows[
                "download_id"
            ].astype(str)
        )

        output_row[
            "background_scene_count_expected"
        ] = len(background_paths)

        output_row[
            "same_sensor_background_count"
        ] = int(
            background_rows[
                "sensor_code"
            ].astype(str).eq(
                str(
                    row["sensor_code"]
                )
            ).sum()
        )

        output_row[
            "feature_status"
        ] = ""

        output_row[
            "feature_error"
        ] = ""

        try:
            result = calculate_scene_feature(
                target_path=target_path,
                background_paths=
                    background_paths,
            )

            output_row.update(
                result
            )

            output_row[
                "feature_status"
            ] = "success"

        except Exception as error:
            output_row[
                "feature_status"
            ] = "failed"

            output_row[
                "feature_error"
            ] = str(error)

        feature_rows.append(
            output_row
        )

    features = pd.DataFrame(
        feature_rows
    )

    calibration_values = features.loc[
        features["external_role"].eq(
            "calibration_negative"
        )
        & features["feature_status"].eq(
            "success"
        ),
        "temporal_z_source_p95",
    ].dropna().to_numpy(
        dtype=float
    )

    if len(calibration_values) != 5:
        features[
            "site_calibration_status"
        ] = "failed"

        features[
            "site_calibration_error"
        ] = (
            f"Expected 5 usable calibration "
            f"features, found "
            f"{len(calibration_values)}."
        )

        return features

    try:
        (
            calibration_center,
            calibration_scale,
            calibration_scale_method,
        ) = robust_calibration_scale(
            calibration_values
        )

        features[
            "cal_temporal_z_source_p95_z"
        ] = (
            features[
                "temporal_z_source_p95"
            ]
            - calibration_center
        ) / calibration_scale

        features[
            PRIMARY_FEATURE
        ] = features[
            "temporal_z_source_p95"
        ].apply(
            lambda value:
                empirical_percentile(
                    calibration_values,
                    value,
                )
        )

        features[
            "calibration_reference_count"
        ] = len(
            calibration_values
        )

        features[
            "calibration_reference_median"
        ] = calibration_center

        features[
            "calibration_reference_scale"
        ] = calibration_scale

        features[
            "calibration_scale_method"
        ] = calibration_scale_method

        features[
            "site_calibration_status"
        ] = "success"

        features[
            "site_calibration_error"
        ] = ""

    except Exception as error:
        features[
            "site_calibration_status"
        ] = "failed"

        features[
            "site_calibration_error"
        ] = str(error)

    return features


def extract_all_external_features(index):
    existing = pd.DataFrame()

    if FEATURE_OUTPUT.exists():
        try:
            existing = pd.read_csv(
                FEATURE_OUTPUT,
                low_memory=False,
            )

            if (
                "feature_contract_version"
                not in existing.columns
                or not existing[
                    "feature_contract_version"
                ].eq(
                    FEATURE_CONTRACT_VERSION
                ).all()
            ):
                existing = pd.DataFrame()

        except Exception:
            existing = pd.DataFrame()

    completed_sites = set()

    if not existing.empty:
        for site_key, group in (
            existing.groupby("site_key")
        ):
            expected_ids = set(
                index.loc[
                    index["site_key"].eq(
                        str(site_key)
                    ),
                    "download_id",
                ].astype(str)
            )

            existing_ids = set(
                group["download_id"]
                .astype(str)
            )

            all_success = (
                group["feature_status"]
                .eq("success")
                .all()
                and group[
                    "site_calibration_status"
                ].eq("success").all()
            )

            if (
                expected_ids == existing_ids
                and all_success
            ):
                completed_sites.add(
                    str(site_key)
                )

    all_features = existing.copy()

    site_keys = sorted(
        index["site_key"]
        .astype(str)
        .unique()
    )

    for number, site_key in enumerate(
        site_keys,
        start=1,
    ):
        if site_key in completed_sites:
            print(
                f"[{number}/{len(site_keys)}] "
                f"{site_key}: checkpoint success"
            )

            continue

        print(
            f"[{number}/{len(site_keys)}] "
            f"{site_key}: calculating features",
            flush=True,
        )

        site_rows = index[
            index["site_key"].eq(
                site_key
            )
        ].copy()

        site_features = (
            calculate_site_features(
                site_rows
            )
        )

        if not all_features.empty:
            all_features = all_features[
                ~all_features["site_key"]
                .astype(str)
                .eq(site_key)
            ].copy()

        all_features = pd.concat(
            [
                all_features,
                site_features,
            ],
            ignore_index=True,
            sort=False,
        )

        all_features = (
            all_features.sort_values(
                [
                    "site_key",
                    "external_role",
                    "acquisition_datetime_utc",
                    "download_id",
                ],
                na_position="last",
            )
            .reset_index(drop=True)
        )

        all_features.to_csv(
            FEATURE_OUTPUT,
            index=False,
        )

        status_counts = (
            site_features[
                "feature_status"
            ].value_counts()
            .to_dict()
        )

        calibration_status = (
            site_features[
                "site_calibration_status"
            ].iloc[0]
        )

        print(
            "  feature status:",
            status_counts,
            "| calibration:",
            calibration_status,
            flush=True,
        )

    return all_features


def build_final_frozen_model():
    development_features = pd.read_csv(
        DEVELOPMENT_FEATURE_INPUT,
        low_memory=False,
    )

    oof = pd.read_csv(
        DEVELOPMENT_OOF_INPUT,
        low_memory=False,
    )

    primary_oof = oof[
        oof["model_name"].eq(
            PRIMARY_MODEL_NAME
        )
        & oof["feature_set"].eq(
            PRIMARY_FEATURE_SET
        )
    ].copy()

    primary_oof = primary_oof[
        [
            "scene_key",
            "actual_label",
        ]
    ].drop_duplicates()

    if len(primary_oof) != 16:
        raise RuntimeError(
            "Expected 16 primary OOF scenes, "
            f"found {len(primary_oof)}."
        )

    if PRIMARY_FEATURE not in (
        development_features.columns
    ):
        raise KeyError(
            f"{PRIMARY_FEATURE} is missing from "
            "development feature table."
        )

    selected = development_features[
        development_features[
            "scene_key"
        ].isin(
            primary_oof["scene_key"]
        )
    ].copy()

    duplicated = selected[
        "scene_key"
    ].duplicated(
        keep=False
    )

    if duplicated.any():
        duplicate_check = (
            selected.loc[
                duplicated,
                [
                    "scene_key",
                    PRIMARY_FEATURE,
                ],
            ]
            .groupby("scene_key")[
                PRIMARY_FEATURE
            ]
            .nunique(
                dropna=False
            )
        )

        if (
            duplicate_check > 1
        ).any():
            raise RuntimeError(
                "Conflicting duplicate development "
                "feature rows."
            )

        selected = (
            selected.sort_values(
                "scene_key"
            )
            .drop_duplicates(
                subset=["scene_key"],
                keep="first",
            )
        )

    training = primary_oof.merge(
        selected[
            [
                "scene_key",
                PRIMARY_FEATURE,
            ]
        ],
        on="scene_key",
        how="left",
        validate="one_to_one",
    )

    if len(training) != 16:
        raise RuntimeError(
            "Development training merge did not "
            "produce 16 rows."
        )

    X_train = training[
        [PRIMARY_FEATURE]
    ]

    y_train = training[
        "actual_label"
    ].astype(int)

    model = build_logistic_model()

    model.fit(
        X_train,
        y_train,
    )

    return (
        model,
        training,
    )


def calculate_metric_row(
    frame,
    group_type,
    group_value,
):
    y_true = frame[
        "actual_label"
    ].astype(int).to_numpy()

    y_pred = frame[
        "predicted_label"
    ].astype(int).to_numpy()

    scores = frame[
        "prediction_score"
    ].astype(float).to_numpy()

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

    positive_total = (
        true_positive
        + false_negative
    )

    negative_total = (
        true_negative
        + false_positive
    )

    positive_recall = (
        true_positive / positive_total
        if positive_total > 0
        else np.nan
    )

    negative_recall = (
        true_negative / negative_total
        if negative_total > 0
        else np.nan
    )

    false_positive_rate = (
        false_positive / negative_total
        if negative_total > 0
        else np.nan
    )

    if len(np.unique(y_true)) == 2:
        roc_auc = float(
            roc_auc_score(
                y_true,
                scores,
            )
        )

        average_precision = float(
            average_precision_score(
                y_true,
                scores,
            )
        )

        balanced_accuracy = float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        )
    else:
        roc_auc = np.nan
        average_precision = np.nan
        balanced_accuracy = np.nan

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
            int(true_negative),
        "false_positive":
            int(false_positive),
        "false_negative":
            int(false_negative),
        "true_positive":
            int(true_positive),
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
        "balanced_accuracy":
            balanced_accuracy,
        "precision_positive":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
        "positive_recall":
            positive_recall,
        "negative_recall":
            negative_recall,
        "false_positive_rate":
            false_positive_rate,
        "f1_positive":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
        "roc_auc":
            roc_auc,
        "average_precision":
            average_precision,
        "alert_threshold":
            FROZEN_ALERT_THRESHOLD,
    }


def build_feature_audit(features):
    rows = []

    for site_key, group in (
        features.groupby("site_key")
    ):
        rows.append({
            "site_key":
                site_key,
            "total_feature_rows":
                len(group),
            "successful_feature_rows":
                int(
                    group[
                        "feature_status"
                    ].eq("success").sum()
                ),
            "failed_feature_rows":
                int(
                    group[
                        "feature_status"
                    ].eq("failed").sum()
                ),
            "calibration_negative_rows":
                int(
                    group[
                        "external_role"
                    ].eq(
                        "calibration_negative"
                    ).sum()
                ),
            "test_negative_rows":
                int(
                    group[
                        "external_role"
                    ].eq(
                        "test_negative"
                    ).sum()
                ),
            "positive_rows":
                int(
                    group[
                        "external_role"
                    ].eq(
                        "high_emission_positive"
                    ).sum()
                ),
            "calibration_reference_count":
                group[
                    "calibration_reference_count"
                ].dropna().iloc[0]
                if group[
                    "calibration_reference_count"
                ].notna().any()
                else np.nan,
            "calibration_reference_median":
                group[
                    "calibration_reference_median"
                ].dropna().iloc[0]
                if group[
                    "calibration_reference_median"
                ].notna().any()
                else np.nan,
            "calibration_reference_scale":
                group[
                    "calibration_reference_scale"
                ].dropna().iloc[0]
                if group[
                    "calibration_reference_scale"
                ].notna().any()
                else np.nan,
            "site_calibration_status":
                group[
                    "site_calibration_status"
                ].iloc[0],
            "minimum_qa_clear_fraction":
                group[
                    "qa_clear_fraction"
                ].min()
                if "qa_clear_fraction"
                in group.columns
                else np.nan,
            "median_qa_clear_fraction":
                group[
                    "qa_clear_fraction"
                ].median()
                if "qa_clear_fraction"
                in group.columns
                else np.nan,
        })

    return pd.DataFrame(rows)


def main():
    for path in [
        EXTERNAL_INDEX_INPUT,
        DEVELOPMENT_FEATURE_INPUT,
        DEVELOPMENT_OOF_INPUT,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    FEATURE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    index = pd.read_csv(
        EXTERNAL_INDEX_INPUT,
        low_memory=False,
    )

    index[
        "acquisition_datetime_utc"
    ] = pd.to_datetime(
        index[
            "acquisition_datetime_utc"
        ],
        errors="coerce",
        utc=True,
    )

    index = validate_external_index(
        index
    )

    print("=" * 112)
    print("MARS-S2L FROZEN EXTERNAL EVALUATION")
    print("=" * 112)

    print(
        "\nExternal rows:",
        len(index),
    )

    print(
        "External sites:",
        index["site_key"].nunique(),
    )

    print("\nExtracting temporal features...")

    features = extract_all_external_features(
        index
    )

    feature_audit = build_feature_audit(
        features
    )

    feature_audit.to_csv(
        FEATURE_AUDIT_OUTPUT,
        index=False,
    )

    print("\nFeature status:")
    print(
        features[
            "feature_status"
        ].value_counts(
            dropna=False
        )
    )

    print("\nSite calibration status:")
    print(
        features[
            "site_calibration_status"
        ].value_counts(
            dropna=False
        )
    )

    failed_features = features[
        ~features[
            "feature_status"
        ].eq("success")
        | ~features[
            "site_calibration_status"
        ].eq("success")
    ]

    if not failed_features.empty:
        print("\nFailed feature rows:")
        print(
            failed_features[
                [
                    "download_id",
                    "site_key",
                    "external_role",
                    "feature_error",
                    "site_calibration_error",
                ]
            ].to_string(
                index=False,
                max_colwidth=160,
            )
        )

        raise RuntimeError(
            "Feature extraction is incomplete. "
            "See outputs/237 and outputs/244."
        )

    evaluation = features[
        features["external_role"].isin([
            "test_negative",
            "high_emission_positive",
        ])
    ].copy()

    if len(evaluation) != (
        EXPECTED_EVALUATION_COUNT
    ):
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_EVALUATION_COUNT} evaluation "
            f"rows, found {len(evaluation)}."
        )

    if (
        evaluation[
            PRIMARY_FEATURE
        ].isna().any()
    ):
        raise RuntimeError(
            "Primary external feature contains NaN."
        )

    model, training = (
        build_final_frozen_model()
    )

    X_external = evaluation[
        [PRIMARY_FEATURE]
    ]

    evaluation[
        "prediction_score"
    ] = model.predict_proba(
        X_external
    )[:, 1]

    evaluation[
        "predicted_label"
    ] = (
        evaluation[
            "prediction_score"
        ]
        >= FROZEN_ALERT_THRESHOLD
    ).astype(int)

    evaluation[
        "actual_label"
    ] = evaluation[
        "evaluation_label"
    ].astype(int)

    evaluation[
        "correct"
    ] = (
        evaluation[
            "predicted_label"
        ]
        == evaluation[
            "actual_label"
        ]
    )

    evaluation[
        "model_name"
    ] = PRIMARY_MODEL_NAME

    evaluation[
        "feature_set"
    ] = PRIMARY_FEATURE_SET

    evaluation[
        "frozen_alert_threshold"
    ] = FROZEN_ALERT_THRESHOLD

    evaluation[
        "frozen_high_emission_threshold_kg_h"
    ] = HIGH_EMISSION_THRESHOLD_KG_H

    evaluation[
        "release_rate_kg_h"
    ] = pd.to_numeric(
        evaluation.get(
            "ch4_fluxrate",
            np.nan,
        ),
        errors="coerce",
    )

    evaluation = evaluation.sort_values(
        [
            "site_key",
            "actual_label",
            "acquisition_datetime_utc",
        ],
        na_position="last",
    ).reset_index(drop=True)

    evaluation.to_csv(
        PREDICTION_OUTPUT,
        index=False,
    )

    overall_row = calculate_metric_row(
        evaluation,
        group_type="overall",
        group_value="all_external_sites",
    )

    site_rows = []

    for site_key, group in (
        evaluation.groupby("site_key")
    ):
        site_rows.append(
            calculate_metric_row(
                group,
                group_type="site",
                group_value=site_key,
            )
        )

    site_metrics = pd.DataFrame(
        site_rows
    )

    site_metrics.to_csv(
        SITE_METRIC_OUTPUT,
        index=False,
    )

    overall_row[
        "macro_site_positive_recall"
    ] = site_metrics[
        "positive_recall"
    ].mean()

    overall_row[
        "macro_site_negative_recall"
    ] = site_metrics[
        "negative_recall"
    ].mean()

    overall_row[
        "macro_site_balanced_accuracy"
    ] = site_metrics[
        "balanced_accuracy"
    ].mean()

    overall_metrics = pd.DataFrame([
        overall_row
    ])

    overall_metrics.to_csv(
        OVERALL_METRIC_OUTPUT,
        index=False,
    )

    sensor_rows = []

    for sensor_code, group in (
        evaluation.groupby("sensor_code")
    ):
        sensor_rows.append(
            calculate_metric_row(
                group,
                group_type="sensor",
                group_value=sensor_code,
            )
        )

    sensor_metrics = pd.DataFrame(
        sensor_rows
    )

    sensor_metrics.to_csv(
        SENSOR_METRIC_OUTPUT,
        index=False,
    )

    positives = evaluation[
        evaluation["actual_label"].eq(1)
    ].copy()

    positives[
        "flux_bin"
    ] = pd.cut(
        positives[
            "release_rate_kg_h"
        ],
        bins=[
            1000.0,
            2000.0,
            5000.0,
            np.inf,
        ],
        right=False,
        labels=[
            "1000_to_2000",
            "2000_to_5000",
            "ge_5000",
        ],
    )

    flux_metrics = (
        positives.groupby(
            "flux_bin",
            observed=False,
        )
        .agg(
            positive_count=(
                "actual_label",
                "size",
            ),
            detected_count=(
                "predicted_label",
                "sum",
            ),
            detection_rate=(
                "predicted_label",
                "mean",
            ),
            median_prediction_score=(
                "prediction_score",
                "median",
            ),
            minimum_flux_kg_h=(
                "release_rate_kg_h",
                "min",
            ),
            median_flux_kg_h=(
                "release_rate_kg_h",
                "median",
            ),
            maximum_flux_kg_h=(
                "release_rate_kg_h",
                "max",
            ),
        )
        .reset_index()
    )

    flux_metrics.to_csv(
        FLUX_METRIC_OUTPUT,
        index=False,
    )

    imputer = model.named_steps[
        "imputer"
    ]

    scaler = model.named_steps[
        "scaler"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    model_contract = pd.DataFrame([{
        "model_name":
            PRIMARY_MODEL_NAME,
        "feature_set":
            PRIMARY_FEATURE_SET,
        "primary_feature":
            PRIMARY_FEATURE,
        "development_training_rows":
            len(training),
        "development_negative_count":
            int(
                (
                    training[
                        "actual_label"
                    ] == 0
                ).sum()
            ),
        "development_positive_count":
            int(
                (
                    training[
                        "actual_label"
                    ] == 1
                ).sum()
            ),
        "imputer_strategy":
            "median",
        "imputer_statistic":
            float(
                imputer.statistics_[0]
            ),
        "scaler_mean":
            float(
                scaler.mean_[0]
            ),
        "scaler_scale":
            float(
                scaler.scale_[0]
            ),
        "logistic_solver":
            classifier.solver,
        "logistic_class_weight":
            str(
                classifier.class_weight
            ),
        "logistic_C":
            classifier.C,
        "logistic_max_iter":
            classifier.max_iter,
        "logistic_random_state":
            classifier.random_state,
        "logistic_coefficient":
            float(
                classifier.coef_[0, 0]
            ),
        "logistic_intercept":
            float(
                classifier.intercept_[0]
            ),
        "frozen_alert_threshold":
            FROZEN_ALERT_THRESHOLD,
        "high_emission_threshold_kg_h":
            HIGH_EMISSION_THRESHOLD_KG_H,
        "external_site_count":
            evaluation[
                "site_key"
            ].nunique(),
        "external_evaluation_count":
            len(evaluation),
        "selection_used_external_predictions":
            False,
    }])

    model_contract.to_csv(
        MODEL_CONTRACT_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 112)
    print("FINAL EXTERNAL RESULTS")
    print("=" * 112)

    display_columns = [
        "evaluation_count",
        "negative_count",
        "positive_count",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "accuracy",
        "balanced_accuracy",
        "precision_positive",
        "positive_recall",
        "negative_recall",
        "false_positive_rate",
        "f1_positive",
        "roc_auc",
        "average_precision",
        "macro_site_balanced_accuracy",
    ]

    print(
        overall_metrics[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nResults by sensor:")
    print(
        sensor_metrics[
            [
                "group_value",
                "evaluation_count",
                "negative_count",
                "positive_count",
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
                "balanced_accuracy",
                "positive_recall",
                "negative_recall",
                "roc_auc",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nPositive detection by flux bin:")
    print(
        flux_metrics.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nSaved:")
    print(FEATURE_OUTPUT)
    print(PREDICTION_OUTPUT)
    print(OVERALL_METRIC_OUTPUT)
    print(SITE_METRIC_OUTPUT)
    print(SENSOR_METRIC_OUTPUT)
    print(FLUX_METRIC_OUTPUT)
    print(MODEL_CONTRACT_OUTPUT)
    print(FEATURE_AUDIT_OUTPUT)


if __name__ == "__main__":
    main()
