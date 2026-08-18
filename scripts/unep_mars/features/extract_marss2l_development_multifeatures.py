from pathlib import Path

import numpy as np
import pandas as pd

from build_evanston_external_anomaly_features import (
    calculate_scene_feature,
    empirical_percentile,
)


# ============================================================
# Inputs and outputs
# ============================================================

INPUT = Path(
    "outputs/263_marss2l_development_clean_index_v2.csv"
)

FEATURE_OUTPUT = Path(
    "outputs/267_marss2l_development_multifeatures.csv"
)

MODEL_READY_OUTPUT = Path(
    "outputs/268_marss2l_development_model_ready.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/269_marss2l_development_feature_audit.csv"
)


# ============================================================
# Frozen dataset contract
# ============================================================

EXPECTED_ROWS = 816
EXPECTED_SITES = 62

EXPECTED_CALIBRATION = 310
EXPECTED_MODEL_NEGATIVE = 305
EXPECTED_MODEL_POSITIVE = 201

EXPECTED_MODEL_ROWS = (
    EXPECTED_MODEL_NEGATIVE
    + EXPECTED_MODEL_POSITIVE
)

FEATURE_VERSION = (
    "marss2l_development_multifeature_v1"
)


# These features are returned by the previously validated
# calculate_scene_feature() function.
RAW_FEATURES = [
    "background_scene_count",
    "target_valid_fraction",
    "temporal_valid_fraction",
    "source_valid_pixels",
    "source_total_pixels",
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


# Build site-calibrated versions of the most physically useful
# methane-anomaly features.
CALIBRATED_FEATURES = [
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


def robust_reference_statistics(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {
            "count": 0,
            "center": np.nan,
            "scale": np.nan,
            "scale_method": "no_values",
        }

    center = float(
        np.median(values)
    )

    mad = float(
        np.median(
            np.abs(values - center)
        )
    )

    scale = float(
        1.4826 * mad
    )

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
        scale = np.nan
        scale_method = "degenerate"

    return {
        "count": len(values),
        "center": center,
        "scale": scale,
        "scale_method": scale_method,
    }


def validate_input(frame):
    required = [
        "download_id",
        "site_key",
        "development_split",
        "development_role",
        "patch_path",
        "download_status",
        "qa_clear_fraction",
        "sensor_code",
        "pixel_sha256",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}"
        )

    frame = frame[
        frame["download_status"].eq(
            "success"
        )
    ].copy()

    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} successful rows, "
            f"found {len(frame)}."
        )

    if (
        frame["site_key"].nunique()
        != EXPECTED_SITES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_SITES} sites, "
            f"found {frame['site_key'].nunique()}."
        )

    role_counts = (
        frame["development_role"]
        .value_counts()
    )

    expected_roles = {
        "calibration_negative":
            EXPECTED_CALIBRATION,
        "model_negative":
            EXPECTED_MODEL_NEGATIVE,
        "model_positive":
            EXPECTED_MODEL_POSITIVE,
    }

    for role, expected in (
        expected_roles.items()
    ):
        actual = int(
            role_counts.get(role, 0)
        )

        if actual != expected:
            raise RuntimeError(
                f"{role}: expected {expected}, "
                f"found {actual}."
            )

    if (
        frame["pixel_sha256"].nunique()
        != EXPECTED_ROWS
    ):
        raise RuntimeError(
            "Raster hashes are not unique."
        )

    frame["qa_clear_fraction"] = (
        pd.to_numeric(
            frame["qa_clear_fraction"],
            errors="coerce",
        )
    )

    if (
        frame["qa_clear_fraction"]
        .lt(0.8)
        .any()
    ):
        raise RuntimeError(
            "Input still contains QA < 0.8."
        )

    missing_paths = []

    for path_text in frame["patch_path"]:
        path = Path(str(path_text))

        if not path.exists():
            missing_paths.append(
                str(path)
            )

    if missing_paths:
        raise FileNotFoundError(
            "Missing raster files:\n"
            + "\n".join(
                missing_paths[:20]
            )
        )

    return frame


def calculate_site_features(site_rows):
    site_rows = site_rows.copy()

    site_key = str(
        site_rows["site_key"].iloc[0]
    )

    calibration = site_rows[
        site_rows[
            "development_role"
        ].eq("calibration_negative")
    ].copy()

    calibration = (
        calibration.sort_values(
            [
                "acquisition_datetime_utc",
                "download_id",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    if len(calibration) != 5:
        raise RuntimeError(
            f"{site_key}: expected five "
            f"calibration negatives, found "
            f"{len(calibration)}."
        )

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
            row["development_role"]
        )

        # Calibration negatives use leave-one-out:
        # target scene is compared with the other four.
        if role == "calibration_negative":
            backgrounds = calibration[
                calibration["download_id"]
                .astype(str)
                .ne(download_id)
            ].copy()
        else:
            # Model positives and negatives use all
            # five calibration scenes as background.
            backgrounds = calibration.copy()

        background_paths = [
            Path(str(path))
            for path in backgrounds[
                "patch_path"
            ]
        ]

        if target_path in background_paths:
            raise RuntimeError(
                f"{download_id}: target included "
                "in its own background."
            )

        output_row[
            "feature_version"
        ] = FEATURE_VERSION

        output_row[
            "background_download_ids"
        ] = "|".join(
            backgrounds[
                "download_id"
            ].astype(str)
        )

        output_row[
            "same_sensor_background_count"
        ] = int(
            backgrounds[
                "sensor_code"
            ]
            .astype(str)
            .eq(
                str(row["sensor_code"])
            )
            .sum()
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

            output_row.update(result)

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

    successful_calibration = features[
        features[
            "development_role"
        ].eq("calibration_negative")
        & features[
            "feature_status"
        ].eq("success")
    ].copy()

    if len(successful_calibration) != 5:
        features[
            "site_calibration_status"
        ] = "failed"

        features[
            "site_calibration_error"
        ] = (
            "Not all five calibration "
            "features succeeded."
        )

        return features

    calibration_audit = {}

    for base_feature in (
        CALIBRATED_FEATURES
    ):
        calibration_values = (
            successful_calibration[
                base_feature
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        reference = (
            robust_reference_statistics(
                calibration_values
            )
        )

        prefix = (
            f"cal_{base_feature}"
        )

        features[
            f"{prefix}_reference_count"
        ] = reference["count"]

        features[
            f"{prefix}_reference_median"
        ] = reference["center"]

        features[
            f"{prefix}_reference_scale"
        ] = reference["scale"]

        features[
            f"{prefix}_scale_method"
        ] = reference["scale_method"]

        if np.isfinite(
            reference["scale"]
        ):
            features[
                f"{prefix}_z"
            ] = (
                features[base_feature]
                - reference["center"]
            ) / reference["scale"]
        else:
            features[
                f"{prefix}_z"
            ] = np.nan

        features[
            f"{prefix}_percentile"
        ] = features[
            base_feature
        ].apply(
            lambda value:
                empirical_percentile(
                    calibration_values,
                    value,
                )
        )

        calibration_audit[
            base_feature
        ] = reference

    features[
        "site_calibration_status"
    ] = "success"

    features[
        "site_calibration_error"
    ] = ""

    return features


def load_checkpoint(input_frame):
    if not FEATURE_OUTPUT.exists():
        return pd.DataFrame()

    try:
        existing = pd.read_csv(
            FEATURE_OUTPUT,
            low_memory=False,
        )
    except Exception:
        return pd.DataFrame()

    if (
        "feature_version"
        not in existing.columns
    ):
        return pd.DataFrame()

    if not existing[
        "feature_version"
    ].eq(FEATURE_VERSION).all():
        return pd.DataFrame()

    return existing


def completed_site_keys(
    existing,
    input_frame,
):
    completed = set()

    if existing.empty:
        return completed

    for site_key, expected_group in (
        input_frame.groupby("site_key")
    ):
        existing_group = existing[
            existing["site_key"]
            .astype(str)
            .eq(str(site_key))
        ]

        expected_ids = set(
            expected_group[
                "download_id"
            ].astype(str)
        )

        existing_ids = set(
            existing_group[
                "download_id"
            ].astype(str)
        )

        all_success = (
            not existing_group.empty
            and existing_group[
                "feature_status"
            ].eq("success").all()
            and existing_group[
                "site_calibration_status"
            ].eq("success").all()
        )

        if (
            expected_ids == existing_ids
            and all_success
        ):
            completed.add(
                str(site_key)
            )

    return completed


def build_audit(features):
    rows = []

    for site_key, group in (
        features.groupby("site_key")
    ):
        rows.append({
            "site_key":
                site_key,
            "development_split":
                group[
                    "development_split"
                ].iloc[0],
            "total_rows":
                len(group),
            "calibration_negative_count":
                int(
                    group[
                        "development_role"
                    ].eq(
                        "calibration_negative"
                    ).sum()
                ),
            "model_negative_count":
                int(
                    group[
                        "development_role"
                    ].eq(
                        "model_negative"
                    ).sum()
                ),
            "model_positive_count":
                int(
                    group[
                        "development_role"
                    ].eq(
                        "model_positive"
                    ).sum()
                ),
            "successful_feature_count":
                int(
                    group[
                        "feature_status"
                    ].eq("success").sum()
                ),
            "failed_feature_count":
                int(
                    group[
                        "feature_status"
                    ].eq("failed").sum()
                ),
            "site_calibration_status":
                group[
                    "site_calibration_status"
                ].iloc[0],
            "minimum_qa_clear_fraction":
                group[
                    "qa_clear_fraction"
                ].min(),
            "median_qa_clear_fraction":
                group[
                    "qa_clear_fraction"
                ].median(),
            "minimum_source_valid_fraction":
                group[
                    "source_valid_fraction"
                ].min(),
            "median_source_valid_fraction":
                group[
                    "source_valid_fraction"
                ].median(),
        })

    return pd.DataFrame(rows)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    frame = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    frame[
        "acquisition_datetime_utc"
    ] = pd.to_datetime(
        frame[
            "acquisition_datetime_utc"
        ],
        errors="coerce",
        utc=True,
    )

    frame["site_key"] = (
        frame["site_key"]
        .astype(str)
        .str.strip()
    )

    frame["download_id"] = (
        frame["download_id"]
        .astype(str)
        .str.strip()
    )

    frame = validate_input(frame)

    print("=" * 112)
    print("MARS-S2L DEVELOPMENT FEATURE EXTRACTION")
    print("=" * 112)

    print("\nInput rows:", len(frame))
    print(
        "Input sites:",
        frame["site_key"].nunique(),
    )

    existing = load_checkpoint(
        frame
    )

    completed = completed_site_keys(
        existing,
        frame,
    )

    all_features = existing.copy()

    site_keys = sorted(
        frame["site_key"].unique()
    )

    for number, site_key in enumerate(
        site_keys,
        start=1,
    ):
        if site_key in completed:
            print(
                f"[{number}/{len(site_keys)}] "
                f"{site_key}: checkpoint success"
            )

            continue

        print(
            f"[{number}/{len(site_keys)}] "
            f"{site_key}: extracting features",
            flush=True,
        )

        site_rows = frame[
            frame["site_key"].eq(
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
                ~all_features[
                    "site_key"
                ].astype(str).eq(
                    site_key
                )
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
                    "development_split",
                    "site_key",
                    "development_role",
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

        print(
            "  status:",
            site_features[
                "feature_status"
            ].value_counts().to_dict(),
            "| calibration:",
            site_features[
                "site_calibration_status"
            ].iloc[0],
            flush=True,
        )

    if len(all_features) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} feature rows, "
            f"found {len(all_features)}."
        )

    failed = all_features[
        ~all_features[
            "feature_status"
        ].eq("success")
        | ~all_features[
            "site_calibration_status"
        ].eq("success")
    ]

    audit = build_audit(
        all_features
    )

    audit.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    print("\nFeature status:")
    print(
        all_features[
            "feature_status"
        ].value_counts(
            dropna=False
        )
    )

    print("\nSite calibration status:")
    print(
        all_features[
            "site_calibration_status"
        ].value_counts(
            dropna=False
        )
    )

    if not failed.empty:
        print("\nFailed rows:")
        print(
            failed[
                [
                    "download_id",
                    "site_key",
                    "development_role",
                    "feature_error",
                    "site_calibration_error",
                ]
            ].to_string(
                index=False,
                max_colwidth=150,
            )
        )

        raise RuntimeError(
            "Feature extraction is incomplete."
        )

    model_ready = all_features[
        all_features[
            "development_role"
        ].isin([
            "model_negative",
            "model_positive",
        ])
    ].copy()

    model_ready[
        "target_label"
    ] = (
        model_ready[
            "development_role"
        ]
        .eq("model_positive")
        .astype(int)
    )

    if len(model_ready) != EXPECTED_MODEL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_MODEL_ROWS} "
            f"model rows, found "
            f"{len(model_ready)}."
        )

    train_sites = set(
        model_ready.loc[
            model_ready[
                "development_split"
            ].eq("train"),
            "site_key",
        ]
    )

    validation_sites = set(
        model_ready.loc[
            model_ready[
                "development_split"
            ].eq("validation"),
            "site_key",
        ]
    )

    overlap = train_sites & validation_sites

    if overlap:
        raise RuntimeError(
            "Train/validation site overlap."
        )

    model_ready.to_csv(
        MODEL_READY_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 112)
    print("MODEL-READY DEVELOPMENT DATA")
    print("=" * 112)

    print(
        "\nModel-ready rows:",
        len(model_ready),
    )

    print("\nRows by split and label:")
    print(
        pd.crosstab(
            model_ready[
                "development_split"
            ],
            model_ready[
                "target_label"
            ],
            margins=True,
        )
    )

    print("\nSites by split:")
    print(
        model_ready.groupby(
            "development_split"
        )["site_key"].nunique()
    )

    print(
        "\nTrain/validation site overlap:",
        len(overlap),
    )

    print("\nRaw feature columns:")
    for feature in RAW_FEATURES:
        if feature in all_features.columns:
            print(feature)

    print("\nCalibrated feature columns:")
    calibrated_columns = [
        column
        for column in all_features.columns
        if column.startswith("cal_")
        and (
            column.endswith("_z")
            or column.endswith(
                "_percentile"
            )
        )
    ]

    for column in calibrated_columns:
        print(column)

    print("\nSaved:")
    print(FEATURE_OUTPUT)
    print(MODEL_READY_OUTPUT)
    print(AUDIT_OUTPUT)


if __name__ == "__main__":
    main()
