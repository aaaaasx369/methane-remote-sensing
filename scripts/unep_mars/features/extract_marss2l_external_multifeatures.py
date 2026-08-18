from pathlib import Path

import numpy as np
import pandas as pd

from build_evanston_external_anomaly_features import (
    calculate_scene_feature,
    empirical_percentile,
)


INPUT = Path(
    "outputs/234_marss2l_external_patch_index_v1_2.csv"
)

FEATURE_OUTPUT = Path(
    "outputs/277_marss2l_external_multifeatures.csv"
)

MODEL_READY_OUTPUT = Path(
    "outputs/278_marss2l_external_model_ready.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/279_marss2l_external_multifeature_audit.csv"
)


EXPECTED_ROWS = 327
EXPECTED_SITES = 33
EXPECTED_CALIBRATION = 165
EXPECTED_TEST_NEGATIVES = 99
EXPECTED_POSITIVES = 63
EXPECTED_EVALUATION_ROWS = 162

FEATURE_VERSION = (
    "marss2l_external_multifeature_v1"
)


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
        "external_role",
        "evaluation_label",
        "patch_path",
        "download_status",
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
            f"Missing columns: {missing}"
        )

    frame = frame[
        frame["download_status"].eq(
            "success"
        )
    ].copy()

    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
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
        frame["external_role"]
        .value_counts()
    )

    expected = {
        "calibration_negative":
            EXPECTED_CALIBRATION,
        "test_negative":
            EXPECTED_TEST_NEGATIVES,
        "high_emission_positive":
            EXPECTED_POSITIVES,
    }

    for role, expected_count in expected.items():
        actual = int(
            role_counts.get(role, 0)
        )

        if actual != expected_count:
            raise RuntimeError(
                f"{role}: expected "
                f"{expected_count}, found {actual}."
            )

    if (
        frame["pixel_sha256"].nunique()
        != EXPECTED_ROWS
    ):
        raise RuntimeError(
            "Raster hashes are not unique."
        )

    missing_paths = [
        str(path)
        for path in frame["patch_path"]
        if not Path(str(path)).exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing raster files:\n"
            + "\n".join(missing_paths[:20])
        )

    return frame


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

    rows = []

    for _, row in site_rows.iterrows():
        output = row.to_dict()

        target_id = str(
            row["download_id"]
        )

        target_path = Path(
            str(row["patch_path"])
        )

        if (
            row["external_role"]
            == "calibration_negative"
        ):
            backgrounds = calibration[
                calibration["download_id"]
                .astype(str)
                .ne(target_id)
            ].copy()
        else:
            backgrounds = calibration.copy()

        background_paths = [
            Path(str(path))
            for path in backgrounds[
                "patch_path"
            ]
        ]

        output[
            "feature_version"
        ] = FEATURE_VERSION

        output[
            "background_download_ids"
        ] = "|".join(
            backgrounds[
                "download_id"
            ].astype(str)
        )

        output[
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

        output["feature_status"] = ""
        output["feature_error"] = ""

        try:
            result = calculate_scene_feature(
                target_path=target_path,
                background_paths=
                    background_paths,
            )

            output.update(result)
            output["feature_status"] = "success"

        except Exception as error:
            output["feature_status"] = "failed"
            output["feature_error"] = str(error)

        rows.append(output)

    features = pd.DataFrame(rows)

    calibration_features = features[
        features["external_role"].eq(
            "calibration_negative"
        )
        & features["feature_status"].eq(
            "success"
        )
    ].copy()

    if len(calibration_features) != 5:
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

    for base_feature in (
        CALIBRATED_BASE_FEATURES
    ):
        calibration_values = (
            calibration_features[
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

        prefix = f"cal_{base_feature}"

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

    features[
        "site_calibration_status"
    ] = "success"

    features[
        "site_calibration_error"
    ] = ""

    return features


def load_checkpoint():
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


def completed_sites(
    existing,
    expected,
):
    completed = set()

    if existing.empty:
        return completed

    for site_key, expected_group in (
        expected.groupby("site_key")
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

        success = (
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
            and success
        ):
            completed.add(str(site_key))

    return completed


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

    frame = validate_input(frame)

    existing = load_checkpoint()

    complete = completed_sites(
        existing,
        frame,
    )

    all_features = existing.copy()

    site_keys = sorted(
        frame["site_key"].unique()
    )

    print("=" * 110)
    print("MARS-S2L EXTERNAL MULTIFEATURE EXTRACTION")
    print("=" * 110)

    for number, site_key in enumerate(
        site_keys,
        start=1,
    ):
        if site_key in complete:
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
                ~all_features["site_key"]
                .astype(str)
                .eq(site_key)
            ]

        all_features = pd.concat(
            [
                all_features,
                site_features,
            ],
            ignore_index=True,
            sort=False,
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
        )

    failed = all_features[
        ~all_features[
            "feature_status"
        ].eq("success")
        | ~all_features[
            "site_calibration_status"
        ].eq("success")
    ]

    if not failed.empty:
        raise RuntimeError(
            "External feature extraction "
            "contains failed rows."
        )

    evaluation = all_features[
        all_features[
            "external_role"
        ].isin([
            "test_negative",
            "high_emission_positive",
        ])
    ].copy()

    evaluation[
        "target_label"
    ] = pd.to_numeric(
        evaluation[
            "evaluation_label"
        ],
        errors="raise",
    ).astype(int)

    evaluation[
        "sensor_is_lc09"
    ] = (
        evaluation[
            "sensor_code"
        ]
        .astype(str)
        .str.upper()
        .eq("LC09")
        .astype(int)
    )

    if len(evaluation) != (
        EXPECTED_EVALUATION_ROWS
    ):
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_EVALUATION_ROWS} "
            f"evaluation rows, found "
            f"{len(evaluation)}."
        )

    evaluation.to_csv(
        MODEL_READY_OUTPUT,
        index=False,
    )

    audit = (
        all_features.groupby(
            "site_key"
        )
        .agg(
            total_rows=(
                "download_id",
                "size",
            ),
            successful_rows=(
                "feature_status",
                lambda values:
                    int(
                        (
                            values == "success"
                        ).sum()
                    ),
            ),
            calibration_count=(
                "external_role",
                lambda values:
                    int(
                        (
                            values
                            == "calibration_negative"
                        ).sum()
                    ),
            ),
            test_negative_count=(
                "external_role",
                lambda values:
                    int(
                        (
                            values
                            == "test_negative"
                        ).sum()
                    ),
            ),
            positive_count=(
                "external_role",
                lambda values:
                    int(
                        (
                            values
                            == "high_emission_positive"
                        ).sum()
                    ),
            ),
        )
        .reset_index()
    )

    audit.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    print("\nFeature status:")
    print(
        all_features[
            "feature_status"
        ].value_counts()
    )

    print("\nModel-ready rows:")
    print(len(evaluation))

    print("\nLabels:")
    print(
        evaluation[
            "target_label"
        ].value_counts()
        .sort_index()
    )

    print("\nSaved:")
    print(FEATURE_OUTPUT)
    print(MODEL_READY_OUTPUT)
    print(AUDIT_OUTPUT)


if __name__ == "__main__":
    main()
