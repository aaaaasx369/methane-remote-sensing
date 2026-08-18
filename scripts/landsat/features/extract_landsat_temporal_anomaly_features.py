from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import label as connected_components


MANIFEST_CSV = Path(
    "outputs/108_landsat_high_emission_core_manifest.csv"
)

BACKGROUND_PAIRS_CSV = Path(
    "outputs/110_landsat_matched_background_pairs.csv"
)

FEATURE_OUTPUT = Path(
    "outputs/112_landsat_temporal_anomaly_features.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/113_landsat_temporal_anomaly_feature_audit.csv"
)

PREVIEW_DIR = Path(
    "outputs/landsat_temporal_anomaly_previews"
)


SCALE_FACTOR = 0.0000275
OFFSET = -0.2

NODATA_VALUE = -9999.0

EXPECTED_BACKGROUNDS = 4
MIN_BACKGROUND_OBSERVATIONS = 2

POSITIVE_Z_THRESHOLD = 3.0
MODERATE_Z_THRESHOLD = 2.0


def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return ""

    return text


def safe_filename(value):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )


def resolve_path(value):
    text = clean_text(value)

    if not text:
        raise ValueError(
            "Empty TIFF path."
        )

    path = Path(text).expanduser()

    if path.exists():
        return path.resolve()

    project_path = Path.cwd() / path

    if project_path.exists():
        return project_path.resolve()

    filename = path.name

    matches = list(
        Path("sample_patches").rglob(
            filename
        )
    )

    if len(matches) == 1:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Could not resolve TIFF: {value}"
    )


def reflectance_from_raw(raw):
    return (
        raw.astype(np.float64)
        * SCALE_FACTOR
        + OFFSET
    )


def read_target_image(path):
    with rasterio.open(path) as dataset:
        raw = dataset.read()

        metadata = {
            "height": dataset.height,
            "width": dataset.width,
            "transform": dataset.transform,
            "crs": dataset.crs,
            "count": dataset.count,
            "dtype": dataset.dtypes[0],
        }

    if raw.shape[0] != 6:
        raise ValueError(
            f"Expected 6 bands, found "
            f"{raw.shape[0]} in {path}"
        )

    valid = np.all(
        raw > 0,
        axis=0,
    )

    reflectance = reflectance_from_raw(
        raw
    )

    reflectance[
        :,
        ~valid
    ] = np.nan

    return (
        reflectance,
        valid,
        metadata,
    )


def align_background_to_target(
    background_path,
    target_metadata,
):
    """
    Reproject Red, NIR, SWIR1 and SWIR2 from a
    background scene onto the exact target grid.
    """
    source_band_indices = [
        2,  # Red
        3,  # NIR
        4,  # SWIR1
        5,  # SWIR2
    ]

    target_height = int(
        target_metadata["height"]
    )

    target_width = int(
        target_metadata["width"]
    )

    destination = np.full(
        (
            len(source_band_indices),
            target_height,
            target_width,
        ),
        NODATA_VALUE,
        dtype=np.float64,
    )

    with rasterio.open(
        background_path
    ) as source:
        raw = source.read()

        if raw.shape[0] != 6:
            raise ValueError(
                f"Expected 6 bands in "
                f"{background_path}, found "
                f"{raw.shape[0]}"
            )

        source_valid = np.all(
            raw > 0,
            axis=0,
        )

        source_reflectance = (
            reflectance_from_raw(raw)
        )

        for output_index, band_index in enumerate(
            source_band_indices
        ):
            source_band = source_reflectance[
                band_index
            ].copy()

            source_band[
                ~source_valid
            ] = NODATA_VALUE

            reproject(
                source=source_band,
                destination=destination[
                    output_index
                ],
                src_transform=
                    source.transform,
                src_crs=
                    source.crs,
                src_nodata=
                    NODATA_VALUE,
                dst_transform=
                    target_metadata[
                        "transform"
                    ],
                dst_crs=
                    target_metadata[
                        "crs"
                    ],
                dst_nodata=
                    NODATA_VALUE,
                resampling=
                    Resampling.bilinear,
            )

    aligned_valid = np.all(
        destination
        != NODATA_VALUE,
        axis=0,
    )

    destination[
        destination == NODATA_VALUE
    ] = np.nan

    return destination, aligned_valid


def safe_log_ratio(
    numerator,
    denominator,
):
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


def safe_ndvi(
    nir,
    red,
):
    result = np.full(
        nir.shape,
        np.nan,
        dtype=np.float64,
    )

    denominator = nir + red

    valid = (
        np.isfinite(nir)
        & np.isfinite(red)
        & np.isfinite(denominator)
        & (
            np.abs(denominator)
            > 1e-10
        )
    )

    result[valid] = (
        nir[valid] - red[valid]
    ) / denominator[valid]

    return result


def robust_zscore(
    values,
    reference_mask,
):
    reference_values = values[
        reference_mask
        & np.isfinite(values)
    ]

    if len(reference_values) < 20:
        reference_values = values[
            np.isfinite(values)
        ]

    if len(reference_values) < 20:
        raise ValueError(
            "Not enough valid pixels for "
            "robust normalization."
        )

    reference_median = float(
        np.median(reference_values)
    )

    mad = float(
        np.median(
            np.abs(
                reference_values
                - reference_median
            )
        )
    )

    robust_sigma = 1.4826 * mad

    if (
        not np.isfinite(robust_sigma)
        or robust_sigma < 1e-8
    ):
        robust_sigma = float(
            np.std(reference_values)
        )

    if (
        not np.isfinite(robust_sigma)
        or robust_sigma < 1e-8
    ):
        raise ValueError(
            "Temporal difference has "
            "insufficient background variation."
        )

    zscore = (
        values - reference_median
    ) / robust_sigma

    return (
        zscore,
        reference_median,
        robust_sigma,
    )


def summarize_values(
    values,
    prefix,
):
    finite = values[
        np.isfinite(values)
    ]

    if len(finite) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_p99": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }

    return {
        f"{prefix}_mean":
            float(np.mean(finite)),
        f"{prefix}_median":
            float(np.median(finite)),
        f"{prefix}_std":
            float(np.std(finite)),
        f"{prefix}_p90":
            float(
                np.percentile(
                    finite,
                    90,
                )
            ),
        f"{prefix}_p95":
            float(
                np.percentile(
                    finite,
                    95,
                )
            ),
        f"{prefix}_p99":
            float(
                np.percentile(
                    finite,
                    99,
                )
            ),
        f"{prefix}_min":
            float(np.min(finite)),
        f"{prefix}_max":
            float(np.max(finite)),
    }


def build_spatial_masks(shape):
    height, width = shape

    rows, columns = np.indices(
        shape
    )

    center_row = (
        height - 1
    ) / 2

    center_column = (
        width - 1
    ) / 2

    normalized_row = (
        rows - center_row
    ) / (height / 2)

    normalized_column = (
        columns - center_column
    ) / (width / 2)

    radius = np.sqrt(
        normalized_row ** 2
        + normalized_column ** 2
    )

    return {
        "source": radius <= 0.15,
        "center": radius <= 0.50,
        "outer": radius >= 0.65,
        "full": np.ones(
            shape,
            dtype=bool,
        ),
        "center_row": center_row,
        "center_column":
            center_column,
    }


def connected_component_features(
    binary_mask,
    anomaly_values,
    source_mask,
    prefix,
):
    component_map, component_count = (
        connected_components(
            binary_mask.astype(
                np.uint8
            )
        )
    )

    result = {
        f"{prefix}_component_count":
            int(component_count),
        f"{prefix}_largest_component_pixels":
            0,
        f"{prefix}_largest_component_max":
            np.nan,
        f"{prefix}_largest_component_centroid_distance_px":
            np.nan,
        f"{prefix}_source_connected_component_count":
            0,
        f"{prefix}_source_connected_pixels":
            0,
        f"{prefix}_source_connected_max":
            np.nan,
    }

    if component_count == 0:
        return result

    height, width = binary_mask.shape

    center_row = (
        height - 1
    ) / 2

    center_column = (
        width - 1
    ) / 2

    component_sizes = []

    source_component_ids = []

    for component_id in range(
        1,
        component_count + 1,
    ):
        component_mask = (
            component_map
            == component_id
        )

        size = int(
            component_mask.sum()
        )

        component_sizes.append(
            (
                component_id,
                size,
            )
        )

        if np.any(
            component_mask
            & source_mask
        ):
            source_component_ids.append(
                component_id
            )

    largest_id, largest_size = max(
        component_sizes,
        key=lambda item: item[1],
    )

    largest_mask = (
        component_map == largest_id
    )

    largest_rows, largest_columns = (
        np.where(largest_mask)
    )

    centroid_row = float(
        np.mean(largest_rows)
    )

    centroid_column = float(
        np.mean(largest_columns)
    )

    centroid_distance = float(
        np.sqrt(
            (
                centroid_row
                - center_row
            ) ** 2
            + (
                centroid_column
                - center_column
            ) ** 2
        )
    )

    largest_values = anomaly_values[
        largest_mask
        & np.isfinite(
            anomaly_values
        )
    ]

    result.update({
        f"{prefix}_largest_component_pixels":
            largest_size,
        f"{prefix}_largest_component_max":
            (
                float(
                    np.max(
                        largest_values
                    )
                )
                if len(
                    largest_values
                ) > 0
                else np.nan
            ),
        f"{prefix}_largest_component_centroid_distance_px":
            centroid_distance,
    })

    if source_component_ids:
        source_connected = np.isin(
            component_map,
            source_component_ids,
        )

        source_values = anomaly_values[
            source_connected
            & np.isfinite(
                anomaly_values
            )
        ]

        result.update({
            f"{prefix}_source_connected_component_count":
                len(
                    source_component_ids
                ),
            f"{prefix}_source_connected_pixels":
                int(
                    source_connected.sum()
                ),
            f"{prefix}_source_connected_max":
                (
                    float(
                        np.max(
                            source_values
                        )
                    )
                    if len(
                        source_values
                    ) > 0
                    else np.nan
                ),
        })

    return result


def stretch_rgb(rgb):
    stretched = np.zeros_like(
        rgb,
        dtype=np.float64,
    )

    for band_index in range(3):
        band = rgb[
            :,
            :,
            band_index
        ]

        valid = np.isfinite(band)

        if not valid.any():
            continue

        low, high = np.percentile(
            band[valid],
            [2, 98],
        )

        if high <= low:
            continue

        stretched[
            :,
            :,
            band_index
        ] = np.clip(
            (
                band - low
            ) / (
                high - low
            ),
            0,
            1,
        )

    return stretched


def save_preview(
    target_reflectance,
    target_ratio,
    delta_ratio,
    zscore,
    valid_mask,
    scene_key,
    site,
    target_label,
    output_path,
):
    red = target_reflectance[2]
    green = target_reflectance[1]
    blue = target_reflectance[0]

    rgb = stretch_rgb(
        np.dstack([
            red,
            green,
            blue,
        ])
    )

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5),
    )

    axes[0].imshow(rgb)

    axes[0].set_title(
        f"RGB\nLabel={target_label}"
    )

    ratio_image = axes[1].imshow(
        target_ratio,
        cmap="viridis",
    )

    axes[1].set_title(
        "Target log(SWIR1/SWIR2)"
    )

    figure.colorbar(
        ratio_image,
        ax=axes[1],
        fraction=0.046,
    )

    finite_delta = delta_ratio[
        np.isfinite(delta_ratio)
    ]

    if len(finite_delta) > 0:
        delta_limit = float(
            np.percentile(
                np.abs(finite_delta),
                98,
            )
        )
    else:
        delta_limit = 0.1

    if (
        not np.isfinite(delta_limit)
        or delta_limit <= 0
    ):
        delta_limit = 0.1

    delta_image = axes[2].imshow(
        delta_ratio,
        cmap="RdBu_r",
        vmin=-delta_limit,
        vmax=delta_limit,
    )

    axes[2].set_title(
        "Target − background median"
    )

    figure.colorbar(
        delta_image,
        ax=axes[2],
        fraction=0.046,
    )

    z_image = axes[3].imshow(
        zscore,
        cmap="RdBu_r",
        vmin=-5,
        vmax=5,
    )

    positive_mask = (
        zscore
        >= POSITIVE_Z_THRESHOLD
    ) & valid_mask

    if positive_mask.any():
        axes[3].contour(
            positive_mask,
            levels=[0.5],
            linewidths=1,
        )

    center_row = (
        zscore.shape[0] - 1
    ) / 2

    center_column = (
        zscore.shape[1] - 1
    ) / 2

    axes[3].scatter(
        [center_column],
        [center_row],
        marker="+",
        s=120,
    )

    axes[3].set_title(
        "Robust temporal anomaly Z"
    )

    figure.colorbar(
        z_image,
        ax=axes[3],
        fraction=0.046,
    )

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"{scene_key} | {site}"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


def process_target(
    target_row,
    target_pairs,
):
    scene_key = clean_text(
        target_row["scene_key"]
    )

    target_path = resolve_path(
        target_row[
            "resolved_patch_path"
        ]
    )

    if len(target_pairs) != (
        EXPECTED_BACKGROUNDS
    ):
        raise ValueError(
            f"{scene_key}: expected "
            f"{EXPECTED_BACKGROUNDS} "
            "backgrounds, found "
            f"{len(target_pairs)}."
        )

    if target_pairs[
        "background_scene_key"
    ].duplicated().any():
        raise ValueError(
            f"{scene_key}: duplicate "
            "background scenes."
        )

    (
        target_reflectance,
        target_valid,
        target_metadata,
    ) = read_target_image(
        target_path
    )

    target_red = target_reflectance[2]
    target_nir = target_reflectance[3]
    target_swir1 = target_reflectance[4]
    target_swir2 = target_reflectance[5]

    target_ratio = safe_log_ratio(
        target_swir1,
        target_swir2,
    )

    target_ndvi = safe_ndvi(
        target_nir,
        target_red,
    )

    background_band_stack = []
    background_ratio_stack = []
    background_ndvi_stack = []

    background_paths = []

    for _, pair_row in (
        target_pairs.sort_values(
            "background_rank"
        ).iterrows()
    ):
        background_path = resolve_path(
            pair_row[
                "background_patch_path"
            ]
        )

        background_paths.append(
            str(background_path)
        )

        (
            aligned,
            aligned_valid,
        ) = align_background_to_target(
            background_path,
            target_metadata,
        )

        aligned[
            :,
            ~aligned_valid
        ] = np.nan

        red = aligned[0]
        nir = aligned[1]
        swir1 = aligned[2]
        swir2 = aligned[3]

        ratio = safe_log_ratio(
            swir1,
            swir2,
        )

        ndvi = safe_ndvi(
            nir,
            red,
        )

        background_band_stack.append(
            aligned
        )

        background_ratio_stack.append(
            ratio
        )

        background_ndvi_stack.append(
            ndvi
        )

    band_stack = np.stack(
        background_band_stack,
        axis=0,
    )

    ratio_stack = np.stack(
        background_ratio_stack,
        axis=0,
    )

    ndvi_stack = np.stack(
        background_ndvi_stack,
        axis=0,
    )

    background_band_median = (
        np.nanmedian(
            band_stack,
            axis=0,
        )
    )

    background_ratio_median = (
        np.nanmedian(
            ratio_stack,
            axis=0,
        )
    )

    background_ndvi_median = (
        np.nanmedian(
            ndvi_stack,
            axis=0,
        )
    )

    ratio_background_count = np.sum(
        np.isfinite(ratio_stack),
        axis=0,
    )

    temporal_valid = (
        target_valid
        & np.isfinite(target_ratio)
        & np.isfinite(
            background_ratio_median
        )
        & (
            ratio_background_count
            >= MIN_BACKGROUND_OBSERVATIONS
        )
    )

    delta_ratio = (
        target_ratio
        - background_ratio_median
    )

    delta_ratio[
        ~temporal_valid
    ] = np.nan

    delta_red = (
        target_red
        - background_band_median[0]
    )

    delta_nir = (
        target_nir
        - background_band_median[1]
    )

    delta_swir1 = (
        target_swir1
        - background_band_median[2]
    )

    delta_swir2 = (
        target_swir2
        - background_band_median[3]
    )

    delta_ndvi = (
        target_ndvi
        - background_ndvi_median
    )

    spatial_masks = build_spatial_masks(
        delta_ratio.shape
    )

    outer_reference_mask = (
        spatial_masks["outer"]
        & temporal_valid
    )

    (
        temporal_z,
        outer_delta_median,
        outer_delta_sigma,
    ) = robust_zscore(
        delta_ratio,
        outer_reference_mask,
    )

    temporal_z[
        ~temporal_valid
    ] = np.nan

    features = {
        "target_valid_pixel_fraction":
            float(
                target_valid.mean()
            ),
        "temporal_valid_pixel_fraction":
            float(
                temporal_valid.mean()
            ),
        "background_count":
            EXPECTED_BACKGROUNDS,
        "background_observation_count_mean":
            float(
                np.mean(
                    ratio_background_count[
                        temporal_valid
                    ]
                )
            ),
        "outer_delta_ratio_median":
            outer_delta_median,
        "outer_delta_ratio_robust_sigma":
            outer_delta_sigma,
    }

    region_masks = {
        "full":
            spatial_masks["full"]
            & temporal_valid,
        "center":
            spatial_masks["center"]
            & temporal_valid,
        "source":
            spatial_masks["source"]
            & temporal_valid,
        "outer":
            spatial_masks["outer"]
            & temporal_valid,
    }

    arrays_to_summarize = {
        "delta_ratio":
            delta_ratio,
        "temporal_z":
            temporal_z,
        "abs_temporal_z":
            np.abs(temporal_z),
        "delta_swir1":
            delta_swir1,
        "delta_swir2":
            delta_swir2,
        "delta_ndvi":
            delta_ndvi,
    }

    for array_name, array in (
        arrays_to_summarize.items()
    ):
        for region_name, mask in (
            region_masks.items()
        ):
            features.update(
                summarize_values(
                    array[mask],
                    (
                        f"{array_name}_"
                        f"{region_name}"
                    ),
                )
            )

    for region_name, mask in (
        region_masks.items()
    ):
        valid_z_values = temporal_z[
            mask
            & np.isfinite(temporal_z)
        ]

        if len(valid_z_values) == 0:
            features[
                f"positive_z2_fraction_"
                f"{region_name}"
            ] = np.nan

            features[
                f"positive_z3_fraction_"
                f"{region_name}"
            ] = np.nan

            features[
                f"absolute_z3_fraction_"
                f"{region_name}"
            ] = np.nan

        else:
            features[
                f"positive_z2_fraction_"
                f"{region_name}"
            ] = float(
                np.mean(
                    valid_z_values
                    >= MODERATE_Z_THRESHOLD
                )
            )

            features[
                f"positive_z3_fraction_"
                f"{region_name}"
            ] = float(
                np.mean(
                    valid_z_values
                    >= POSITIVE_Z_THRESHOLD
                )
            )

            features[
                f"absolute_z3_fraction_"
                f"{region_name}"
            ] = float(
                np.mean(
                    np.abs(valid_z_values)
                    >= POSITIVE_Z_THRESHOLD
                )
            )

    positive_binary = (
        temporal_z
        >= POSITIVE_Z_THRESHOLD
    ) & temporal_valid

    absolute_binary = (
        np.abs(temporal_z)
        >= POSITIVE_Z_THRESHOLD
    ) & temporal_valid

    features.update(
        connected_component_features(
            positive_binary,
            temporal_z,
            spatial_masks["source"],
            "positive_z3",
        )
    )

    features.update(
        connected_component_features(
            absolute_binary,
            np.abs(temporal_z),
            spatial_masks["source"],
            "absolute_z3",
        )
    )

    output_filename = (
        f"label_"
        f"{int(target_row['high_emission_target'])}_"
        f"{safe_filename(scene_key)}.png"
    )

    output_path = (
        PREVIEW_DIR
        / output_filename
    )

    save_preview(
        target_reflectance=
            target_reflectance,
        target_ratio=
            target_ratio,
        delta_ratio=
            delta_ratio,
        zscore=
            temporal_z,
        valid_mask=
            temporal_valid,
        scene_key=
            scene_key,
        site=
            target_row[
                "site_key_normalized"
            ],
        target_label=
            int(
                target_row[
                    "high_emission_target"
                ]
            ),
        output_path=
            output_path,
    )

    output_row = target_row.to_dict()

    output_row.update({
        "target_patch_path":
            str(target_path),
        "background_patch_paths":
            " | ".join(
                background_paths
            ),
        "temporal_preview_path":
            str(output_path),
    })

    output_row.update(features)

    return output_row, features


def main():
    for path in [
        MANIFEST_CSV,
        BACKGROUND_PAIRS_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input: {path}"
            )

    manifest = pd.read_csv(
        MANIFEST_CSV,
        low_memory=False,
    )

    pairs = pd.read_csv(
        BACKGROUND_PAIRS_CSV,
        low_memory=False,
    )

    if len(manifest) != 16:
        raise ValueError(
            f"Expected 16 target scenes, "
            f"found {len(manifest)}."
        )

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FEATURE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_rows = []
    audit_rows = []

    print("=" * 105)
    print("LANDSAT MATCHED TEMPORAL ANOMALY EXTRACTION")
    print("=" * 105)

    print(f"\nTarget scenes: {len(manifest)}")
    print(f"Background pairs: {len(pairs)}")

    for _, target_row in (
        manifest.iterrows()
    ):
        scene_key = clean_text(
            target_row["scene_key"]
        )

        target_pairs = pairs[
            pairs["target_scene_key"]
            .astype(str)
            .eq(scene_key)
        ].copy()

        print("\n" + "-" * 105)

        print(
            f"{scene_key} | "
            f"site="
            f"{target_row['site_key_normalized']} | "
            f"label="
            f"{int(target_row['high_emission_target'])}"
        )

        try:
            output_row, features = (
                process_target(
                    target_row,
                    target_pairs,
                )
            )

            missing_features = int(
                pd.Series(features)
                .isna()
                .sum()
            )

            output_rows.append(
                output_row
            )

            audit_rows.append({
                "scene_key":
                    scene_key,
                "site_key":
                    target_row[
                        "site_key_normalized"
                    ],
                "high_emission_target":
                    int(
                        target_row[
                            "high_emission_target"
                        ]
                    ),
                "background_count":
                    len(target_pairs),
                "feature_count":
                    len(features),
                "missing_feature_values":
                    missing_features,
                "temporal_valid_pixel_fraction":
                    features[
                        "temporal_valid_pixel_fraction"
                    ],
                "temporal_z_source_p95":
                    features[
                        "temporal_z_source_p95"
                    ],
                "temporal_z_source_max":
                    features[
                        "temporal_z_source_max"
                    ],
                "positive_z3_fraction_source":
                    features[
                        "positive_z3_fraction_source"
                    ],
                "positive_z3_source_connected_pixels":
                    features[
                        "positive_z3_source_connected_pixels"
                    ],
                "status":
                    "success",
                "error":
                    "",
            })

            print(
                f"[OK] features="
                f"{len(features)} | "
                f"valid="
                f"{features['temporal_valid_pixel_fraction']:.3f} | "
                f"source_p95="
                f"{features['temporal_z_source_p95']:.3f} | "
                f"source_max="
                f"{features['temporal_z_source_max']:.3f} | "
                f"source_component_pixels="
                f"{features['positive_z3_source_connected_pixels']}"
            )

        except Exception as error:
            audit_rows.append({
                "scene_key":
                    scene_key,
                "site_key":
                    target_row.get(
                        "site_key_normalized"
                    ),
                "high_emission_target":
                    target_row.get(
                        "high_emission_target"
                    ),
                "background_count":
                    len(target_pairs),
                "feature_count":
                    np.nan,
                "missing_feature_values":
                    np.nan,
                "status":
                    "error",
                "error":
                    str(error),
            })

            print(
                f"[ERROR] {scene_key}: "
                f"{error}"
            )

    features_df = pd.DataFrame(
        output_rows
    )

    audit_df = pd.DataFrame(
        audit_rows
    )

    features_df.to_csv(
        FEATURE_OUTPUT,
        index=False,
    )

    audit_df.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 105)
    print("TEMPORAL ANOMALY EXTRACTION SUMMARY")
    print("=" * 105)

    print(
        f"\nSuccessful scenes: "
        f"{len(features_df)}"
    )

    print(
        "Failed scenes:",
        int(
            (
                audit_df["status"]
                != "success"
            ).sum()
        ),
    )

    if len(features_df) > 0:
        print("\nLabel counts:")

        print(
            features_df[
                "high_emission_target"
            ].value_counts()
            .sort_index()
        )

        selected_summary_columns = [
            "temporal_z_source_p95",
            "temporal_z_source_max",
            "abs_temporal_z_source_p95",
            "positive_z3_fraction_source",
            "positive_z3_source_connected_pixels",
            "absolute_z3_largest_component_pixels",
        ]

        print(
            "\nMedian temporal anomaly "
            "metrics by label:"
        )

        print(
            features_df.groupby(
                "high_emission_target"
            )[
                selected_summary_columns
            ].median()
        )

        print(
            "\nMedian temporal anomaly "
            "metrics by site and label:"
        )

        print(
            features_df.groupby(
                [
                    "site_key_normalized",
                    "high_emission_target",
                ]
            )[
                selected_summary_columns
            ].median()
        )

        print(
            f"\nTotal output columns: "
            f"{len(features_df.columns)}"
        )

    print("\nSaved:")
    print(FEATURE_OUTPUT)
    print(AUDIT_OUTPUT)
    print(PREVIEW_DIR)


if __name__ == "__main__":
    main()
