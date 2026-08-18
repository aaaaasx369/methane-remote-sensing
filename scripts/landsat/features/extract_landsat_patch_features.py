from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# File paths
# ============================================================

INPUT_CSV = Path(
    "outputs/32_controlled_release_landsat_dataset_table.csv"
)

FEATURE_OUTPUT_CSV = Path(
    "outputs/35_landsat_patch_features.csv"
)

AUDIT_OUTPUT_CSV = Path(
    "outputs/35_landsat_feature_extraction_audit.csv"
)


# ============================================================
# Landsat Collection 2 Level-2 surface-reflectance settings
# ============================================================

SCALE_FACTOR = 0.0000275
OFFSET = -0.2

EXPECTED_BAND_COUNT = 6

# TIFF band order confirmed from the download script:
# SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7
BAND_NAMES = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
]


# Possible CSV columns containing TIFF paths
PATH_COLUMN_CANDIDATES = [
    "patch_path",
    "tif_path",
    "file_path",
    "filepath",
    "local_path",
    "image_path",
    "output_path",
    "patch_file",
    "filename",
    "file_name",
    "output_filename",
]


# ============================================================
# Find TIFF files
# ============================================================

def build_tif_index():
    """
    Search the project for TIFF files.

    This lets the program find an image even when the CSV contains
    only the filename instead of the complete path.
    """
    search_directories = [
        Path("sample_patches/controlled_release_landsat"),
        Path("sample_patches"),
        Path("outputs"),
        Path("patches"),
        Path("data"),
        Path("downloads"),
    ]

    tif_index = {}

    for directory in search_directories:
        if not directory.exists():
            continue

        for pattern in ("*.tif", "*.tiff"):
            for path in directory.rglob(pattern):
                tif_index.setdefault(
                    path.name,
                    [],
                ).append(path)

    return tif_index


def resolve_patch_path(row, tif_index):
    """
    Find the actual TIFF path corresponding to one CSV row.
    """
    possible_values = []

    # First inspect common path-column names
    for column in PATH_COLUMN_CANDIDATES:
        if column in row.index and pd.notna(row[column]):
            possible_values.append(
                str(row[column]).strip()
            )

    # Also inspect every string value ending in .tif or .tiff
    for value in row.values:
        if isinstance(value, str):
            value = value.strip()

            if value.lower().endswith((".tif", ".tiff")):
                possible_values.append(value)

    # Remove duplicate candidate values
    possible_values = list(
        dict.fromkeys(possible_values)
    )

    for value in possible_values:
        path = Path(value).expanduser()

        # Full path or valid relative path
        if path.exists():
            return path.resolve()

        project_relative = Path.cwd() / path

        if project_relative.exists():
            return project_relative.resolve()

        # Search by filename
        matches = tif_index.get(path.name, [])

        if len(matches) == 1:
            return matches[0].resolve()

        if len(matches) > 1:
            print(
                f"[WARNING] Multiple TIFF files named "
                f"{path.name}; using {matches[0]}"
            )
            return matches[0].resolve()

    return None


# ============================================================
# Statistical functions
# ============================================================

def summarize_values(values, prefix):
    """
    Calculate robust summary statistics for one feature array.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p90": np.nan,
        }

    return {
        f"{prefix}_mean": float(
            np.mean(values)
        ),
        f"{prefix}_std": float(
            np.std(values)
        ),
        f"{prefix}_p10": float(
            np.percentile(values, 10)
        ),
        f"{prefix}_median": float(
            np.median(values)
        ),
        f"{prefix}_p90": float(
            np.percentile(values, 90)
        ),
    }


def normalized_difference(a, b):
    """
    Calculate:

        (a - b) / (a + b)
    """
    result = np.full(
        a.shape,
        np.nan,
        dtype=np.float64,
    )

    denominator = a + b

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & (np.abs(denominator) > 1e-8)
    )

    result[valid] = (
        a[valid] - b[valid]
    ) / denominator[valid]

    return result


def safe_ratio(numerator, denominator):
    """
    Calculate numerator / denominator while avoiding division by zero.
    """
    result = np.full(
        numerator.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-8)
    )

    result[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    return result


def safe_log_ratio(numerator, denominator):
    """
    Calculate log(numerator / denominator).

    Only positive reflectance values can be used.
    """
    result = np.full(
        numerator.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (numerator > 0)
        & (denominator > 0)
    )

    result[valid] = np.log(
        numerator[valid]
        / denominator[valid]
    )

    return result


def calculate_center_outer_feature(
    array,
    valid_mask,
    feature_name,
):
    """
    Compare the center of the patch with the outer background.

    Because the controlled-release location is near the patch center,
    this preserves some spatial information instead of using only a
    whole-image average.
    """
    height, width = array.shape

    row_start = height // 3
    row_end = height - height // 3

    col_start = width // 3
    col_end = width - width // 3

    center_mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    center_mask[
        row_start:row_end,
        col_start:col_end,
    ] = True

    center_valid = (
        valid_mask
        & center_mask
        & np.isfinite(array)
    )

    outer_valid = (
        valid_mask
        & (~center_mask)
        & np.isfinite(array)
    )

    center_values = array[center_valid]
    outer_values = array[outer_valid]

    if len(center_values) == 0:
        center_mean = np.nan
    else:
        center_mean = float(
            np.mean(center_values)
        )

    if len(outer_values) == 0:
        outer_mean = np.nan
        outer_std = np.nan
    else:
        outer_mean = float(
            np.mean(outer_values)
        )
        outer_std = float(
            np.std(outer_values)
        )

    if (
        np.isfinite(center_mean)
        and np.isfinite(outer_mean)
    ):
        center_minus_outer = (
            center_mean - outer_mean
        )
    else:
        center_minus_outer = np.nan

    if (
        np.isfinite(center_minus_outer)
        and np.isfinite(outer_std)
        and outer_std > 1e-10
    ):
        standardized_contrast = (
            center_minus_outer
            / outer_std
        )
    else:
        standardized_contrast = np.nan

    return {
        f"{feature_name}_center_mean": center_mean,
        f"{feature_name}_outer_mean": outer_mean,
        f"{feature_name}_center_minus_outer":
            center_minus_outer,
        f"{feature_name}_standardized_contrast":
            standardized_contrast,
    }


# ============================================================
# Extract features from one Landsat TIFF
# ============================================================

def extract_features(patch_path):
    with rasterio.open(patch_path) as src:
        raw = src.read().astype(
            np.float64
        )

        raster_masks = src.read_masks()

        band_count = src.count
        height = src.height
        width = src.width
        nodata = src.nodata
        crs = str(src.crs)

    if band_count != EXPECTED_BAND_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_BAND_COUNT} bands, "
            f"but found {band_count}"
        )

    # Valid only when all six bands are unmasked
    valid_mask = np.all(
        raster_masks > 0,
        axis=0,
    )

    # Explicitly remove NoData values
    if nodata is not None:
        valid_mask &= np.all(
            raw != nodata,
            axis=0,
        )

    # Landsat files use zero as NoData
    valid_mask &= np.all(
        raw != 0,
        axis=0,
    )

    # Remove non-finite values
    valid_mask &= np.all(
        np.isfinite(raw),
        axis=0,
    )

    valid_pixel_count = int(
        valid_mask.sum()
    )

    total_pixel_count = int(
        height * width
    )

    if valid_pixel_count == 0:
        raise ValueError(
            "No valid six-band pixels found"
        )

    # Store raw-value diagnostics before scaling
    raw_valid_values = raw[:, valid_mask]

    raw_minimum = float(
        np.min(raw_valid_values)
    )

    raw_maximum = float(
        np.max(raw_valid_values)
    )

    # Apply Landsat Collection 2 Level-2 scaling
    reflectance = (
        raw * SCALE_FACTOR
        + OFFSET
    )

    # Make invalid pixels NaN
    reflectance[
        :,
        ~valid_mask
    ] = np.nan

    reflectance_valid_values = (
        reflectance[:, valid_mask]
    )

    reflectance_minimum = float(
        np.min(reflectance_valid_values)
    )

    reflectance_maximum = float(
        np.max(reflectance_valid_values)
    )

    fraction_below_zero = float(
        np.mean(
            reflectance_valid_values < 0
        )
    )

    fraction_above_one = float(
        np.mean(
            reflectance_valid_values > 1
        )
    )

    features = {
        "raster_height": height,
        "raster_width": width,
        "band_count_verified": band_count,
        "crs_verified": crs,
        "nodata_verified": nodata,
        "valid_pixel_count": valid_pixel_count,
        "total_pixel_count": total_pixel_count,
        "valid_pixel_fraction": (
            valid_pixel_count
            / total_pixel_count
        ),
        "raw_dn_min": raw_minimum,
        "raw_dn_max": raw_maximum,
        "reflectance_min": reflectance_minimum,
        "reflectance_max": reflectance_maximum,
        "reflectance_fraction_below_0":
            fraction_below_zero,
        "reflectance_fraction_above_1":
            fraction_above_one,
    }

    band_arrays = {}

    # Original six-band reflectance statistics
    for band_index, band_name in enumerate(
        BAND_NAMES
    ):
        band_array = reflectance[
            band_index
        ]

        band_arrays[band_name] = band_array

        features.update(
            summarize_values(
                band_array[valid_mask],
                band_name,
            )
        )

    blue = band_arrays["blue"]
    green = band_arrays["green"]
    red = band_arrays["red"]
    nir = band_arrays["nir"]
    swir1 = band_arrays["swir1"]
    swir2 = band_arrays["swir2"]

    # Common spectral indices
    index_arrays = {
        # Vegetation
        "ndvi": normalized_difference(
            nir,
            red,
        ),

        # Moisture
        "ndmi": normalized_difference(
            nir,
            swir1,
        ),

        # Burn / SWIR response
        "nbr": normalized_difference(
            nir,
            swir2,
        ),

        # Snow / bright surface response
        "ndsi": normalized_difference(
            green,
            swir1,
        ),

        # Difference between two SWIR bands
        "swir_normalized_difference":
            normalized_difference(
                swir1,
                swir2,
            ),

        # Both ratio directions are retained so the model
        # does not depend on one arbitrary ratio convention.
        "swir2_over_swir1": safe_ratio(
            swir2,
            swir1,
        ),

        "swir1_over_swir2": safe_ratio(
            swir1,
            swir2,
        ),

        # Log-ratio useful for relative SWIR differences
        "log_swir1_over_swir2":
            safe_log_ratio(
                swir1,
                swir2,
            ),
    }

    for index_name, index_array in (
        index_arrays.items()
    ):
        values = index_array[
            valid_mask
            & np.isfinite(index_array)
        ]

        features.update(
            summarize_values(
                values,
                index_name,
            )
        )

    # Preserve limited spatial information:
    # compare release-site center with outer background.
    spatial_feature_arrays = {
        "swir1": swir1,
        "swir2": swir2,
        "swir2_over_swir1":
            index_arrays[
                "swir2_over_swir1"
            ],
        "log_swir1_over_swir2":
            index_arrays[
                "log_swir1_over_swir2"
            ],
    }

    for (
        feature_name,
        feature_array,
    ) in spatial_feature_arrays.items():
        features.update(
            calculate_center_outer_feature(
                feature_array,
                valid_mask,
                feature_name,
            )
        )

    return features


# ============================================================
# Main program
# ============================================================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}"
        )

    dataset_df = pd.read_csv(
        INPUT_CSV
    )

    print("=" * 80)
    print("LANDSAT FEATURE EXTRACTION")
    print("=" * 80)

    print(f"\nInput CSV: {INPUT_CSV}")
    print(f"Input rows: {len(dataset_df)}")

    print("\nConfirmed TIFF band order:")
    for band_number, band_name in enumerate(
        BAND_NAMES,
        start=1,
    ):
        landsat_band = band_number + 1

        print(
            f"TIFF band {band_number}: "
            f"SR_B{landsat_band} "
            f"({band_name})"
        )

    print(
        "\nReflectance conversion:"
    )
    print(
        "reflectance = DN × "
        f"{SCALE_FACTOR} + ({OFFSET})"
    )

    tif_index = build_tif_index()

    print(
        f"\nIndexed TIFF filenames: "
        f"{len(tif_index)}"
    )

    output_rows = []
    audit_rows = []

    for row_index, row in (
        dataset_df.iterrows()
    ):
        patch_path = resolve_patch_path(
            row,
            tif_index,
        )

        if patch_path is None:
            print(
                f"[NOT FOUND] row={row_index}"
            )

            audit_rows.append({
                "row_index": row_index,
                "status": "not_found",
                "error": "TIFF path could not be resolved",
            })

            continue

        try:
            features = extract_features(
                patch_path
            )

            output_row = row.to_dict()

            output_row[
                "resolved_patch_path"
            ] = str(patch_path)

            output_row.update(features)

            output_rows.append(
                output_row
            )

            audit_rows.append({
                "row_index": row_index,
                "status": "success",
                "filename": patch_path.name,
                "raw_dn_min":
                    features["raw_dn_min"],
                "raw_dn_max":
                    features["raw_dn_max"],
                "reflectance_min":
                    features["reflectance_min"],
                "reflectance_max":
                    features["reflectance_max"],
                "valid_pixel_fraction":
                    features[
                        "valid_pixel_fraction"
                    ],
            })

            print(
                f"[OK] "
                f"{len(output_rows):02d}/"
                f"{len(dataset_df)} | "
                f"{patch_path.name} | "
                f"DN={features['raw_dn_min']:.0f}"
                f"–{features['raw_dn_max']:.0f} | "
                f"reflectance="
                f"{features['reflectance_min']:.4f}"
                f"–{features['reflectance_max']:.4f} | "
                f"valid="
                f"{features['valid_pixel_fraction']:.3f}"
            )

        except Exception as error:
            print(
                f"[ERROR] row={row_index} | "
                f"{patch_path.name} | "
                f"{error}"
            )

            audit_rows.append({
                "row_index": row_index,
                "status": "error",
                "filename": patch_path.name,
                "error": str(error),
            })

    feature_df = pd.DataFrame(
        output_rows
    )

    audit_df = pd.DataFrame(
        audit_rows
    )

    FEATURE_OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        FEATURE_OUTPUT_CSV,
        index=False,
    )

    audit_df.to_csv(
        AUDIT_OUTPUT_CSV,
        index=False,
    )

    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 80)

    print(
        f"\nSuccessful patches: "
        f"{len(feature_df)}"
    )

    failed_count = (
        len(dataset_df)
        - len(feature_df)
    )

    print(
        f"Failed patches: "
        f"{failed_count}"
    )

    if len(feature_df) > 0:
        if "label" in feature_df.columns:
            print("\nLabel counts:")
            print(
                feature_df["label"]
                .value_counts()
                .sort_index()
            )

        if (
            "landsat_sensor"
            in feature_df.columns
        ):
            print("\nSensor counts:")
            print(
                feature_df[
                    "landsat_sensor"
                ].value_counts()
            )

        print(
            "\nReflectance range over "
            "all patches:"
        )

        print(
            "Minimum:",
            feature_df[
                "reflectance_min"
            ].min(),
        )

        print(
            "Maximum:",
            feature_df[
                "reflectance_max"
            ].max(),
        )

        print(
            "\nValid-pixel fraction:"
        )

        print(
            feature_df[
                "valid_pixel_fraction"
            ].describe()
        )

        print(
            f"\nTotal output columns: "
            f"{len(feature_df.columns)}"
        )

    print("\nSaved:")
    print(FEATURE_OUTPUT_CSV)
    print(AUDIT_OUTPUT_CSV)


if __name__ == "__main__":
    main()
