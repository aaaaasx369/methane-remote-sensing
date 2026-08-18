from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import (
    Resampling,
    reproject,
)


POSITIVE_INPUT = Path(
    "outputs/141_evanston_confirmed_positive_patch_index.csv"
)

NEGATIVE_INPUT = Path(
    "outputs/145_evanston_negative_patch_index.csv"
)

FEATURE_OUTPUT = Path(
    "outputs/146_evanston_external_temporal_features.csv"
)

CALIBRATION_OUTPUT = Path(
    "outputs/147_evanston_external_calibration_audit.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/148_evanston_external_feature_summary.csv"
)


SCALE_FACTOR = 0.0000275
OFFSET = -0.2

SOURCE_RADIUS_NORMALIZED = 0.15
OUTER_BACKGROUND_RADIUS = 0.55

MIN_SOURCE_VALID_FRACTION = 0.80
MIN_CALIBRATION_NEGATIVES = 5


def read_ratio(path: Path):
    with rasterio.open(path) as dataset:
        raw = dataset.read().astype(
            np.float64
        )

        profile = {
            "height": dataset.height,
            "width": dataset.width,
            "transform": dataset.transform,
            "crs": dataset.crs,
        }

    if raw.shape[0] != 6:
        raise ValueError(
            f"{path}: expected 6 bands, "
            f"found {raw.shape[0]}"
        )

    valid = np.all(
        raw > 0,
        axis=0,
    )

    reflectance = (
        raw * SCALE_FACTOR
        + OFFSET
    )

    swir1 = reflectance[4]
    swir2 = reflectance[5]

    ratio_valid = (
        valid
        & np.isfinite(swir1)
        & np.isfinite(swir2)
        & (swir1 > 0)
        & (swir2 > 0)
    )

    ratio = np.full(
        swir1.shape,
        np.nan,
        dtype=np.float64,
    )

    ratio[ratio_valid] = np.log(
        swir1[ratio_valid]
        / swir2[ratio_valid]
    )

    return ratio, profile


def same_grid(
    source_profile,
    target_profile,
):
    return (
        source_profile["height"]
        == target_profile["height"]
        and source_profile["width"]
        == target_profile["width"]
        and source_profile["crs"]
        == target_profile["crs"]
        and source_profile["transform"]
        == target_profile["transform"]
    )


def align_to_grid(
    source_array,
    source_profile,
    target_profile,
):
    if same_grid(
        source_profile,
        target_profile,
    ):
        return source_array.copy()

    destination = np.full(
        (
            target_profile["height"],
            target_profile["width"],
        ),
        np.nan,
        dtype=np.float64,
    )

    reproject(
        source=source_array,
        destination=destination,
        src_transform=
            source_profile["transform"],
        src_crs=
            source_profile["crs"],
        dst_transform=
            target_profile["transform"],
        dst_crs=
            target_profile["crs"],
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    return destination


def radial_masks(
    height,
    width,
):
    row_grid, column_grid = np.indices(
        (height, width)
    )

    center_row = (
        height - 1
    ) / 2

    center_column = (
        width - 1
    ) / 2

    normalized_y = (
        row_grid - center_row
    ) / (height / 2)

    normalized_x = (
        column_grid - center_column
    ) / (width / 2)

    radius = np.sqrt(
        normalized_x ** 2
        + normalized_y ** 2
    )

    source_mask = (
        radius
        <= SOURCE_RADIUS_NORMALIZED
    )

    outer_mask = (
        radius
        >= OUTER_BACKGROUND_RADIUS
    )

    center_mask = (
        radius <= 0.50
    )

    return (
        source_mask,
        outer_mask,
        center_mask,
    )


def robust_center_scale(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 10:
        return np.nan, np.nan

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

    if (
        not np.isfinite(scale)
        or scale < 1e-8
    ):
        scale = float(
            np.std(values)
        )

    if (
        not np.isfinite(scale)
        or scale < 1e-8
    ):
        return center, np.nan

    return center, scale


def empirical_percentile(
    calibration_values,
    value,
):
    calibration_values = np.asarray(
        calibration_values,
        dtype=np.float64,
    )

    calibration_values = (
        calibration_values[
            np.isfinite(
                calibration_values
            )
        ]
    )

    if (
        len(calibration_values) == 0
        or not np.isfinite(value)
    ):
        return np.nan

    lower_count = np.sum(
        calibration_values < value
    )

    equal_count = np.sum(
        np.isclose(
            calibration_values,
            value,
            rtol=1e-10,
            atol=1e-12,
        )
    )

    return float(
        (
            lower_count
            + 0.5 * equal_count
        )
        / len(calibration_values)
    )


def calculate_scene_feature(
    target_path,
    background_paths,
):
    target_ratio, target_profile = (
        read_ratio(target_path)
    )

    aligned_backgrounds = []

    for background_path in (
        background_paths
    ):
        ratio, profile = read_ratio(
            background_path
        )

        aligned = align_to_grid(
            source_array=ratio,
            source_profile=profile,
            target_profile=target_profile,
        )

        aligned_backgrounds.append(
            aligned
        )

    if len(aligned_backgrounds) == 0:
        raise RuntimeError(
            "No background rasters."
        )

    stack = np.stack(
        aligned_backgrounds,
        axis=0,
    )

    with np.errstate(
        all="ignore"
    ):
        background_median = (
            np.nanmedian(
                stack,
                axis=0,
            )
        )

    delta = (
        target_ratio
        - background_median
    )

    (
        source_mask,
        outer_mask,
        center_mask,
    ) = radial_masks(
        target_profile["height"],
        target_profile["width"],
    )

    valid_delta = np.isfinite(
        delta
    )

    outer_values = delta[
        outer_mask
        & valid_delta
    ]

    delta_center, delta_scale = (
        robust_center_scale(
            outer_values
        )
    )

    if not np.isfinite(
        delta_scale
    ):
        raise RuntimeError(
            "Unable to calculate robust "
            "temporal normalization."
        )

    temporal_z = (
        delta - delta_center
    ) / delta_scale

    source_total_pixels = int(
        source_mask.sum()
    )

    source_valid_pixels = int(
        np.sum(
            source_mask
            & np.isfinite(temporal_z)
        )
    )

    source_valid_fraction = (
        source_valid_pixels
        / source_total_pixels
        if source_total_pixels > 0
        else np.nan
    )

    source_values = temporal_z[
        source_mask
        & np.isfinite(temporal_z)
    ]

    center_values = temporal_z[
        center_mask
        & np.isfinite(temporal_z)
    ]

    if len(source_values) == 0:
        raise RuntimeError(
            "No valid source-region pixels."
        )

    return {
        "background_scene_count":
            len(background_paths),
        "target_valid_fraction":
            float(
                np.mean(
                    np.isfinite(
                        target_ratio
                    )
                )
            ),
        "temporal_valid_fraction":
            float(
                np.mean(
                    np.isfinite(
                        temporal_z
                    )
                )
            ),
        "source_valid_pixels":
            source_valid_pixels,
        "source_total_pixels":
            source_total_pixels,
        "source_valid_fraction":
            source_valid_fraction,
        "temporal_z_source_mean":
            float(
                np.mean(source_values)
            ),
        "temporal_z_source_median":
            float(
                np.median(source_values)
            ),
        "temporal_z_source_p90":
            float(
                np.percentile(
                    source_values,
                    90,
                )
            ),
        "temporal_z_source_p95":
            float(
                np.percentile(
                    source_values,
                    95,
                )
            ),
        "temporal_z_source_max":
            float(
                np.max(source_values)
            ),
        "temporal_z_source_positive_fraction":
            float(
                np.mean(
                    source_values > 0
                )
            ),
        "temporal_z_source_gt2_fraction":
            float(
                np.mean(
                    source_values >= 2
                )
            ),
        "temporal_z_source_gt3_fraction":
            float(
                np.mean(
                    source_values >= 3
                )
            ),
        "temporal_z_center_p95":
            (
                float(
                    np.percentile(
                        center_values,
                        95,
                    )
                )
                if len(center_values)
                else np.nan
            ),
        "temporal_delta_outer_center":
            delta_center,
        "temporal_delta_outer_scale":
            delta_scale,
    }


def prepare_manifest():
    if not POSITIVE_INPUT.exists():
        raise FileNotFoundError(
            POSITIVE_INPUT
        )

    if not NEGATIVE_INPUT.exists():
        raise FileNotFoundError(
            NEGATIVE_INPUT
        )

    positives = pd.read_csv(
        POSITIVE_INPUT,
        low_memory=False,
    )

    negatives = pd.read_csv(
        NEGATIVE_INPUT,
        low_memory=False,
    )

    positives = positives[
        positives["download_status"]
        == "success"
    ].copy()

    negatives = negatives[
        negatives["download_status"]
        == "success"
    ].copy()

    positives[
        "external_role"
    ] = "positive"

    positives[
        "evaluation_label"
    ] = 1

    negatives[
        "external_role"
    ] = negatives[
        "negative_role"
    ]

    negatives[
        "evaluation_label"
    ] = 0

    combined = pd.concat(
        [
            positives,
            negatives,
        ],
        ignore_index=True,
        sort=False,
    )

    if "patch_path" not in combined.columns:
        raise KeyError(
            "Missing patch_path column."
        )

    combined["patch_path"] = (
        combined["patch_path"]
        .astype(str)
    )

    missing_paths = [
        path
        for path
        in combined["patch_path"]
        if not Path(path).exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing downloaded rasters:\n"
            + "\n".join(
                missing_paths[:10]
            )
        )

    calibration_count = int(
        (
            combined["external_role"]
            == "calibration_negative"
        ).sum()
    )

    if (
        calibration_count
        < MIN_CALIBRATION_NEGATIVES
    ):
        raise RuntimeError(
            "Need at least "
            f"{MIN_CALIBRATION_NEGATIVES} "
            "successful calibration negatives; "
            f"found {calibration_count}."
        )

    if "pixel_sha256" in combined.columns:
        hashes = combined[
            "pixel_sha256"
        ].dropna()

        duplicate_hashes = hashes[
            hashes.duplicated(
                keep=False
            )
        ]

        if not duplicate_hashes.empty:
            raise RuntimeError(
                "Duplicate raster hashes found:\n"
                + duplicate_hashes
                .to_string(index=False)
            )

    return combined


def main():
    manifest = prepare_manifest()

    calibration_rows = manifest[
        manifest["external_role"]
        == "calibration_negative"
    ].copy()

    calibration_paths = [
        Path(path)
        for path
        in calibration_rows[
            "patch_path"
        ]
    ]

    print("=" * 105)
    print("EVANSTON EXTERNAL TEMPORAL FEATURES")
    print("=" * 105)

    print("\nInput counts:")
    print(
        manifest[
            "external_role"
        ].value_counts()
    )

    feature_rows = []

    for index, row in (
        manifest.iterrows()
    ):
        target_path = Path(
            row["patch_path"]
        )

        role = row[
            "external_role"
        ]

        if (
            role
            == "calibration_negative"
        ):
            backgrounds = [
                path
                for path
                in calibration_paths
                if path.resolve()
                != target_path.resolve()
            ]
        else:
            backgrounds = (
                calibration_paths
            )

        print(
            f"[{index + 1}/"
            f"{len(manifest)}] "
            f"{role} | "
            f"{target_path.name}"
        )

        output_row = row.to_dict()

        try:
            features = (
                calculate_scene_feature(
                    target_path=
                        target_path,
                    background_paths=
                        backgrounds,
                )
            )

            output_row.update(
                features
            )

            output_row[
                "feature_status"
            ] = (
                "success"
                if features[
                    "source_valid_fraction"
                ]
                >= MIN_SOURCE_VALID_FRACTION
                else "low_source_validity"
            )

            output_row[
                "feature_error"
            ] = ""

            print(
                "  source_p95="
                f"{features['temporal_z_source_p95']:.3f}"
                " | valid="
                f"{features['source_valid_fraction']:.3f}"
            )

        except Exception as error:
            output_row[
                "feature_status"
            ] = "failed"

            output_row[
                "feature_error"
            ] = str(error)

            print(
                f"  [ERROR] {error}"
            )

        feature_rows.append(
            output_row
        )

    features = pd.DataFrame(
        feature_rows
    )

    usable = features[
        features["feature_status"]
        == "success"
    ].copy()

    calibration_values = usable.loc[
        usable["external_role"]
        == "calibration_negative",
        "temporal_z_source_p95",
    ].dropna().to_numpy(
        dtype=float
    )

    if (
        len(calibration_values)
        < MIN_CALIBRATION_NEGATIVES
    ):
        raise RuntimeError(
            "Too few usable calibration "
            "negative features."
        )

    calibration_center = float(
        np.median(
            calibration_values
        )
    )

    calibration_mad = float(
        np.median(
            np.abs(
                calibration_values
                - calibration_center
            )
        )
    )

    calibration_scale = (
        1.4826
        * calibration_mad
    )

    if (
        not np.isfinite(
            calibration_scale
        )
        or calibration_scale < 1e-8
    ):
        calibration_scale = float(
            np.std(
                calibration_values
            )
        )

    if (
        not np.isfinite(
            calibration_scale
        )
        or calibration_scale < 1e-8
    ):
        raise RuntimeError(
            "Calibration negative spread "
            "is too small."
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
        "cal_temporal_z_source_p95_percentile"
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

    features.to_csv(
        FEATURE_OUTPUT,
        index=False,
    )

    calibration_audit = pd.DataFrame({
        "calibration_scene_count": [
            len(calibration_values)
        ],
        "temporal_z_source_p95_median": [
            calibration_center
        ],
        "temporal_z_source_p95_mad": [
            calibration_mad
        ],
        "temporal_z_source_p95_robust_scale": [
            calibration_scale
        ],
        "minimum_source_valid_fraction": [
            MIN_SOURCE_VALID_FRACTION
        ],
        "source_radius_normalized": [
            SOURCE_RADIUS_NORMALIZED
        ],
        "outer_background_radius": [
            OUTER_BACKGROUND_RADIUS
        ],
    })

    calibration_audit.to_csv(
        CALIBRATION_OUTPUT,
        index=False,
    )

    summary_columns = [
        "temporal_z_source_p95",
        "cal_temporal_z_source_p95_z",
        "cal_temporal_z_source_p95_percentile",
        "source_valid_fraction",
    ]

    summary = (
        features.groupby(
            "external_role"
        )[summary_columns]
        .agg([
            "count",
            "median",
            "mean",
            "min",
            "max",
        ])
    )

    summary.to_csv(
        SUMMARY_OUTPUT
    )

    print("\n" + "=" * 105)
    print("FEATURE STATUS")
    print("=" * 105)

    print(
        features[
            "feature_status"
        ].value_counts(
            dropna=False
        )
    )

    print("\n" + "=" * 105)
    print("FEATURE SUMMARY BY ROLE")
    print("=" * 105)

    print(
        features.groupby(
            "external_role"
        )[[
            "temporal_z_source_p95",
            "cal_temporal_z_source_p95_z",
            "cal_temporal_z_source_p95_percentile",
            "source_valid_fraction",
        ]]
        .median()
        .to_string(
            float_format=lambda value:
                f"{value:.3f}"
        )
    )

    print("\nIndividual evaluation scenes:")

    display_columns = [
        "external_role",
        "landsat_product_id",
        "acquisition_time_utc",
        "flow_at_scene_kg_h",
        "temporal_z_source_p95",
        "cal_temporal_z_source_p95_z",
        "cal_temporal_z_source_p95_percentile",
        "source_valid_fraction",
        "feature_status",
    ]

    evaluation = features[
        features["external_role"]
        .isin([
            "positive",
            "test_negative",
        ])
    ]

    print(
        evaluation[
            [
                column
                for column
                in display_columns
                if column
                in evaluation.columns
            ]
        ].sort_values(
            [
                "external_role",
                "acquisition_time_utc",
            ]
        ).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}",
        )
    )

    print("\nSaved:")
    print(FEATURE_OUTPUT)
    print(CALIBRATION_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
