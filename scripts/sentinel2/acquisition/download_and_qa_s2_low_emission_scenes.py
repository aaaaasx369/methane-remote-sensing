from pathlib import Path
import io
import os
import re
import time
import zipfile

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coordinates
import requests


MANIFEST_INPUT = Path(
    "outputs/317_s2_low_emission_scene_manifest_v1.csv"
)

PATCH_DIR = Path(
    "sample_patches/s2_low_emission_v1"
)

PREVIEW_DIR = Path(
    "outputs/s2_low_emission_previews_v1"
)

INDEX_OUTPUT = Path(
    "outputs/319_s2_low_emission_patch_index_v1.csv"
)

QA_OUTPUT = Path(
    "outputs/320_s2_low_emission_local_qa_v1.csv"
)


# 3 km × 3 km patch:
# 釋放點四周各 1500 m。
PATCH_HALF_SIZE_METERS = 1500

# Local QA 使用釋放點周圍約 1 km × 1 km。
LOCAL_HALF_SIZE_METERS = 500

EXPORT_SCALE_METERS = 20

BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B8A",
    "B11",
    "B12",
    "SCL",
]

REFLECTANCE_BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B8A",
    "B11",
    "B12",
]

MAX_RETRIES = 4


def initialize_earth_engine():
    project = os.environ.get("EE_PROJECT")

    if not project:
        raise RuntimeError(
            "EE_PROJECT 尚未設定。"
        )

    ee.Initialize(project=project)

    print(
        "Earth Engine project:",
        project,
    )


def parse_bool(value):
    return (
        str(value)
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )


def safe_name(text):
    return re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(text),
    ).strip("_")


def build_image(scene_id):
    image = ee.Image(scene_id)

    reflectance = (
        image
        .select(REFLECTANCE_BANDS)
        .toFloat()
    )

    scl = (
        image
        .select(["SCL"])
        .toFloat()
    )

    return reflectance.addBands(scl)


def download_bytes(url):
    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            response = requests.get(
                url,
                timeout=180,
            )

            response.raise_for_status()

            return response.content

        except Exception as error:
            last_error = error

            if attempt == MAX_RETRIES:
                break

            wait_seconds = 2 ** (
                attempt - 1
            )

            print(
                f"  Download retry "
                f"{attempt}/{MAX_RETRIES} "
                f"after {wait_seconds}s: "
                f"{error}",
                flush=True,
            )

            time.sleep(wait_seconds)

    raise last_error


def save_downloaded_geotiff(
    content,
    output_path,
):
    buffer = io.BytesIO(content)

    if zipfile.is_zipfile(buffer):
        buffer.seek(0)

        with zipfile.ZipFile(buffer) as archive:
            tif_names = [
                name
                for name in archive.namelist()
                if name.lower().endswith(
                    (".tif", ".tiff")
                )
            ]

            if not tif_names:
                raise RuntimeError(
                    "Earth Engine ZIP 裡沒有 GeoTIFF。"
                )

            if len(tif_names) > 1:
                print(
                    "  Warning: multiple TIFFs "
                    "returned; using first:",
                    tif_names,
                )

            with archive.open(
                tif_names[0]
            ) as source:
                output_path.write_bytes(
                    source.read()
                )

    else:
        output_path.write_bytes(content)


def download_scene(
    scene_id,
    latitude,
    longitude,
    output_path,
):
    point = ee.Geometry.Point([
        longitude,
        latitude,
    ])

    region = (
        point
        .buffer(PATCH_HALF_SIZE_METERS)
        .bounds()
    )

    image = build_image(scene_id)

    url = image.getDownloadURL({
        "name": output_path.stem,
        "region": region.getInfo(),
        "scale": EXPORT_SCALE_METERS,
        "format": "GEO_TIFF",
        "filePerBand": False,
    })

    content = download_bytes(url)

    save_downloaded_geotiff(
        content,
        output_path,
    )


def percentile_stretch(
    band,
    valid_mask,
):
    output = np.zeros(
        band.shape,
        dtype=np.float32,
    )

    values = band[
        valid_mask
        & np.isfinite(band)
    ]

    if values.size == 0:
        return output

    lower, upper = np.percentile(
        values,
        [2, 98],
    )

    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or upper <= lower
    ):
        return output

    output = (
        band - lower
    ) / (
        upper - lower
    )

    return np.clip(
        output,
        0,
        1,
    )


def save_previews(
    array,
    valid_mask,
    stem,
):
    band_lookup = {
        band: index
        for index, band
        in enumerate(BANDS)
    }

    rgb = np.dstack([
        percentile_stretch(
            array[
                band_lookup["B4"]
            ],
            valid_mask,
        ),
        percentile_stretch(
            array[
                band_lookup["B3"]
            ],
            valid_mask,
        ),
        percentile_stretch(
            array[
                band_lookup["B2"]
            ],
            valid_mask,
        ),
    ])

    swir = np.dstack([
        percentile_stretch(
            array[
                band_lookup["B12"]
            ],
            valid_mask,
        ),
        percentile_stretch(
            array[
                band_lookup["B11"]
            ],
            valid_mask,
        ),
        percentile_stretch(
            array[
                band_lookup["B8A"]
            ],
            valid_mask,
        ),
    ])

    rgb_path = PREVIEW_DIR / (
        f"{stem}_rgb.png"
    )

    swir_path = PREVIEW_DIR / (
        f"{stem}_swir.png"
    )

    plt.imsave(
        rgb_path,
        rgb,
    )

    plt.imsave(
        swir_path,
        swir,
    )

    return rgb_path, swir_path


def calculate_fraction(mask, denominator_mask):
    denominator = int(
        denominator_mask.sum()
    )

    if denominator == 0:
        return np.nan

    return float(
        (
            mask
            & denominator_mask
        ).sum()
        / denominator
    )


def run_local_qa(
    tif_path,
    latitude,
    longitude,
    preview_stem,
):
    with rasterio.open(
        tif_path
    ) as dataset:
        array = dataset.read().astype(
            np.float32
        )

        masks = dataset.read_masks()

        if array.shape[0] != len(BANDS):
            raise RuntimeError(
                f"Expected {len(BANDS)} bands, "
                f"found {array.shape[0]}."
            )

        finite_mask = np.all(
            np.isfinite(array),
            axis=0,
        )

        raster_mask = np.all(
            masks > 0,
            axis=0,
        )

        reflectance = array[
            :len(REFLECTANCE_BANDS)
        ]

        all_zero = np.all(
            reflectance == 0,
            axis=0,
        )

        valid_pixel = (
            finite_mask
            & raster_mask
            & ~all_zero
        )

        scl = array[
            BANDS.index("SCL")
        ]

        scl_valid = (
            np.isfinite(scl)
            & scl.ne(0)
            if isinstance(
                scl,
                pd.Series,
            )
            else (
                np.isfinite(scl)
                & (scl != 0)
            )
        )

        # 將 release 點從經緯度轉到影像 CRS。
        transformed_x, transformed_y = (
            transform_coordinates(
                "EPSG:4326",
                dataset.crs,
                [longitude],
                [latitude],
            )
        )

        center_x = transformed_x[0]
        center_y = transformed_y[0]

        center_row, center_col = (
            dataset.index(
                center_x,
                center_y,
            )
        )

        center_inside = (
            0 <= center_row
            < dataset.height
            and
            0 <= center_col
            < dataset.width
        )

        pixel_size = float(
            np.mean([
                abs(dataset.res[0]),
                abs(dataset.res[1]),
            ])
        )

        local_radius_pixels = max(
            1,
            int(
                np.ceil(
                    LOCAL_HALF_SIZE_METERS
                    / pixel_size
                )
            ),
        )

        row_start = max(
            0,
            center_row
            - local_radius_pixels,
        )

        row_end = min(
            dataset.height,
            center_row
            + local_radius_pixels
            + 1,
        )

        col_start = max(
            0,
            center_col
            - local_radius_pixels,
        )

        col_end = min(
            dataset.width,
            center_col
            + local_radius_pixels
            + 1,
        )

        local_slice = np.s_[
            row_start:row_end,
            col_start:col_end,
        ]

        local_valid = valid_pixel[
            local_slice
        ]

        local_scl = scl[
            local_slice
        ]

        local_scl_valid = scl_valid[
            local_slice
        ]

        # Sentinel-2 SCL:
        # 3 cloud shadow
        # 8 medium-probability cloud
        # 9 high-probability cloud
        # 10 thin cirrus
        # 11 snow/ice
        local_cloud = np.isin(
            np.rint(
                local_scl
            ).astype(int),
            [8, 9, 10],
        )

        local_shadow = np.isin(
            np.rint(
                local_scl
            ).astype(int),
            [3],
        )

        local_snow = np.isin(
            np.rint(
                local_scl
            ).astype(int),
            [11],
        )

        local_bad_atmosphere = (
            local_cloud
            | local_shadow
            | local_snow
        )

        scene_valid_fraction = float(
            valid_pixel.mean()
        )

        local_valid_fraction = float(
            local_valid.mean()
        )

        local_cloud_fraction = (
            calculate_fraction(
                local_cloud,
                local_scl_valid,
            )
        )

        local_shadow_fraction = (
            calculate_fraction(
                local_shadow,
                local_scl_valid,
            )
        )

        local_snow_fraction = (
            calculate_fraction(
                local_snow,
                local_scl_valid,
            )
        )

        local_bad_fraction = (
            calculate_fraction(
                local_bad_atmosphere,
                local_scl_valid,
            )
        )

        local_all_zero_fraction = float(
            all_zero[
                local_slice
            ].mean()
        )

        band_means = {}

        for index, band in enumerate(
            REFLECTANCE_BANDS
        ):
            local_band = array[
                index
            ][local_slice]

            values = local_band[
                local_valid
                & np.isfinite(
                    local_band
                )
            ]

            band_means[
                f"local_{band}_mean"
            ] = (
                float(values.mean())
                if values.size
                else np.nan
            )

        qa_pass_preliminary = bool(
            center_inside
            and local_valid_fraction >= 0.95
            and (
                np.isnan(
                    local_bad_fraction
                )
                or local_bad_fraction <= 0.10
            )
            and local_all_zero_fraction
            <= 0.01
        )

        rgb_path, swir_path = (
            save_previews(
                array,
                valid_pixel,
                preview_stem,
            )
        )

        return {
            "width":
                dataset.width,

            "height":
                dataset.height,

            "band_count":
                dataset.count,

            "crs":
                str(dataset.crs),

            "pixel_size_m":
                pixel_size,

            "center_row":
                center_row,

            "center_col":
                center_col,

            "source_center_inside_raster":
                center_inside,

            "scene_valid_fraction":
                scene_valid_fraction,

            "local_valid_fraction":
                local_valid_fraction,

            "local_cloud_fraction":
                local_cloud_fraction,

            "local_shadow_fraction":
                local_shadow_fraction,

            "local_snow_fraction":
                local_snow_fraction,

            "local_bad_atmosphere_fraction":
                local_bad_fraction,

            "local_all_zero_fraction":
                local_all_zero_fraction,

            "qa_pass_preliminary":
                qa_pass_preliminary,

            "rgb_preview":
                str(rgb_path),

            "swir_preview":
                str(swir_path),

            **band_means,
        }


def main():
    if not MANIFEST_INPUT.exists():
        raise FileNotFoundError(
            MANIFEST_INPUT
        )

    initialize_earth_engine()

    PATCH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        MANIFEST_INPUT,
        low_memory=False,
    )

    required_columns = [
        "site",
        "scene_id",
        "acquisition_time_utc",
        "lat",
        "lon",
        "final_release_rate_kg_h",
        "primary_include",
        "review_status",
    ]

    missing = [
        column
        for column in required_columns
        if column not in manifest.columns
    ]

    if missing:
        raise KeyError(
            "Manifest 缺少欄位："
            + ", ".join(missing)
        )

    manifest["lat"] = pd.to_numeric(
        manifest["lat"],
        errors="coerce",
    )

    manifest["lon"] = pd.to_numeric(
        manifest["lon"],
        errors="coerce",
    )

    manifest[
        "final_release_rate_kg_h"
    ] = pd.to_numeric(
        manifest[
            "final_release_rate_kg_h"
        ],
        errors="coerce",
    )

    manifest[
        "acquisition_time_utc"
    ] = pd.to_datetime(
        manifest[
            "acquisition_time_utc"
        ],
        errors="coerce",
        utc=True,
    )

    manifest = manifest.dropna(
        subset=[
            "lat",
            "lon",
            "scene_id",
            "acquisition_time_utc",
        ]
    ).copy()

    index_rows = []
    qa_rows = []

    print("=" * 110)
    print(
        "DOWNLOAD AND QA SENTINEL-2 "
        "LOW-EMISSION SCENES"
    )
    print("=" * 110)

    print(
        "\nScenes:",
        len(manifest),
    )

    for number, row in (
        manifest.reset_index(
            drop=True
        ).iterrows()
    ):
        acquisition = row[
            "acquisition_time_utc"
        ]

        date_text = acquisition.strftime(
            "%Y%m%dT%H%M%S"
        )

        primary = parse_bool(
            row["primary_include"]
        )

        role = (
            "primary"
            if primary
            else "exploratory"
        )

        stem = safe_name(
            f"{date_text}_"
            f"{row['site']}_"
            f"{role}"
        )

        tif_path = PATCH_DIR / (
            f"{stem}.tif"
        )

        print(
            f"\n[{number + 1}/{len(manifest)}] "
            f"{row['site']} | "
            f"{row['final_release_rate_kg_h']:.3f} "
            f"kg/h | {role}",
            flush=True,
        )

        print(
            "  Scene:",
            row["scene_id"],
        )

        if tif_path.exists():
            print(
                "  Existing patch:",
                tif_path,
            )

            download_status = (
                "existing"
            )

        else:
            try:
                download_scene(
                    scene_id=
                        row["scene_id"],
                    latitude=
                        float(row["lat"]),
                    longitude=
                        float(row["lon"]),
                    output_path=
                        tif_path,
                )

                print(
                    "  Downloaded:",
                    tif_path,
                )

                download_status = (
                    "downloaded"
                )

            except Exception as error:
                print(
                    "  Download failed:",
                    error,
                )

                index_rows.append({
                    **row.to_dict(),

                    "patch_path":
                        str(tif_path),

                    "download_status":
                        "failed",

                    "download_error":
                        str(error),
                })

                continue

        index_record = {
            **row.to_dict(),

            "patch_path":
                str(tif_path),

            "download_status":
                download_status,

            "download_error":
                "",
        }

        index_rows.append(
            index_record
        )

        try:
            qa = run_local_qa(
                tif_path=tif_path,
                latitude=float(
                    row["lat"]
                ),
                longitude=float(
                    row["lon"]
                ),
                preview_stem=stem,
            )

            qa_record = {
                **index_record,
                **qa,
            }

            qa_rows.append(
                qa_record
            )

            print(
                "  Local valid:",
                f"{qa['local_valid_fraction']:.3f}",
            )

            print(
                "  Local cloud/shadow/snow:",
                f"{qa['local_bad_atmosphere_fraction']:.3f}",
            )

            print(
                "  Preliminary QA pass:",
                qa[
                    "qa_pass_preliminary"
                ],
            )

        except Exception as error:
            print(
                "  QA failed:",
                error,
            )

            qa_rows.append({
                **index_record,

                "qa_pass_preliminary":
                    False,

                "qa_error":
                    str(error),
            })

    index = pd.DataFrame(
        index_rows
    )

    qa_table = pd.DataFrame(
        qa_rows
    )

    index.to_csv(
        INDEX_OUTPUT,
        index=False,
    )

    qa_table.to_csv(
        QA_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 110)
    print("SENTINEL-2 LOW-EMISSION QA SUMMARY")
    print("=" * 110)

    print(
        "\nManifest scenes:",
        len(manifest),
    )

    print(
        "Downloaded/existing patches:",
        int(
            index[
                "download_status"
            ].isin([
                "downloaded",
                "existing",
            ]).sum()
        )
        if not index.empty
        else 0,
    )

    if not qa_table.empty:
        print(
            "Preliminary QA passes:",
            int(
                qa_table[
                    "qa_pass_preliminary"
                ]
                .astype(str)
                .str.lower()
                .isin([
                    "true",
                    "1",
                    "yes",
                ])
                .sum()
            ),
        )

        display_columns = [
            "site",
            "acquisition_time_utc",
            "final_release_rate_kg_h",
            "primary_include",
            "review_status",
            "source_center_inside_raster",
            "local_valid_fraction",
            "local_cloud_fraction",
            "local_shadow_fraction",
            "local_bad_atmosphere_fraction",
            "local_all_zero_fraction",
            "qa_pass_preliminary",
            "patch_path",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in qa_table.columns
        ]

        print("\nQA table:")
        print(
            qa_table[
                display_columns
            ].to_string(
                index=False,
            )
        )

    print("\nSaved:")
    print(INDEX_OUTPUT)
    print(QA_OUTPUT)
    print(PREVIEW_DIR)


if __name__ == "__main__":
    main()
