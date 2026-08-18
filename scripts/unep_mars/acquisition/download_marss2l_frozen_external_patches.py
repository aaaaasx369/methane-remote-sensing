from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import rasterio
import requests


PROJECT = os.environ.get(
    "EE_PROJECT",
    "methane-release-gee",
)

INPUT = Path(
    "outputs/224_marss2l_frozen_external_download_manifest.csv"
)

OUTPUT_DIR = Path(
    "sample_patches/"
    "marss2l_frozen_external_landsat"
)

QA_OUTPUT_DIR = Path(
    "sample_patches/"
    "marss2l_frozen_external_landsat_qa"
)

INDEX_OUTPUT = Path(
    "outputs/226_marss2l_frozen_external_patch_index.csv"
)

BANDS = [
    "SR_B2",
    "SR_B3",
    "SR_B4",
    "SR_B5",
    "SR_B6",
    "SR_B7",
]

QA_BAND = "QA_PIXEL"

COLLECTIONS = {
    "LC08": "LANDSAT/LC08/C02/T1_L2",
    "LC09": "LANDSAT/LC09/C02/T1_L2",
}

PATCH_RADIUS_METERS = 1000
SCALE_METERS = 30
MAX_RETRIES = 3


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def parse_landsat_tile(tile):
    """
    Example:
    LC09_L1TP_158035_20240421_20240421_02_T1
    """

    text = str(tile).strip()

    pattern = re.compile(
        r"^(LC0[89])_L1[A-Z0-9]{2}_"
        r"(\d{3})(\d{3})_"
        r"(\d{8})_"
    )

    match = pattern.search(text)

    if match is None:
        raise ValueError(
            f"Unable to parse Landsat tile: {text}"
        )

    sensor = match.group(1)
    wrs_path = int(match.group(2))
    wrs_row = int(match.group(3))

    acquisition_date = pd.to_datetime(
        match.group(4),
        format="%Y%m%d",
        errors="raise",
        utc=True,
    )

    return {
        "source_l1_tile": text,
        "sensor_code": sensor,
        "wrs_path": wrs_path,
        "wrs_row": wrs_row,
        "tile_acquisition_date": acquisition_date,
    }


def parse_ee_datetime(milliseconds):
    if milliseconds is None:
        return pd.NaT

    return pd.to_datetime(
        milliseconds,
        unit="ms",
        errors="coerce",
        utc=True,
    )


def find_matching_level2_image(
    sensor_code,
    wrs_path,
    wrs_row,
    target_datetime,
    longitude,
    latitude,
):
    collection_id = COLLECTIONS[
        sensor_code
    ]

    target_datetime = pd.Timestamp(
        target_datetime
    )

    if target_datetime.tzinfo is None:
        target_datetime = (
            target_datetime.tz_localize(
                "UTC"
            )
        )
    else:
        target_datetime = (
            target_datetime.tz_convert(
                "UTC"
            )
        )

    search_start = (
        target_datetime
        - pd.Timedelta(days=1)
    )

    search_end = (
        target_datetime
        + pd.Timedelta(days=2)
    )

    point = ee.Geometry.Point([
        float(longitude),
        float(latitude),
    ])

    collection = (
        ee.ImageCollection(
            collection_id
        )
        .filterBounds(point)
        .filterDate(
            search_start.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            search_end.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
        )
        .filter(
            ee.Filter.eq(
                "WRS_PATH",
                int(wrs_path),
            )
        )
        .filter(
            ee.Filter.eq(
                "WRS_ROW",
                int(wrs_row),
            )
        )
    )

    information = collection.getInfo()

    candidates = []

    for feature in information.get(
        "features",
        [],
    ):
        properties = feature.get(
            "properties",
            {},
        )

        acquisition_time = (
            parse_ee_datetime(
                properties.get(
                    "system:time_start"
                )
            )
        )

        if pd.isna(acquisition_time):
            continue

        difference_seconds = abs(
            (
                acquisition_time
                - target_datetime
            ).total_seconds()
        )

        candidates.append({
            "image_id":
                feature.get("id"),
            "matched_l2_product_id":
                properties.get(
                    "LANDSAT_PRODUCT_ID"
                ),
            "matched_acquisition_time_utc":
                acquisition_time,
            "match_time_difference_seconds":
                difference_seconds,
            "matched_cloud_cover":
                properties.get(
                    "CLOUD_COVER"
                ),
            "matched_cloud_cover_land":
                properties.get(
                    "CLOUD_COVER_LAND"
                ),
            "matched_collection_category":
                properties.get(
                    "COLLECTION_CATEGORY"
                ),
            "matched_spacecraft_id":
                properties.get(
                    "SPACECRAFT_ID"
                ),
        })

    if not candidates:
        raise RuntimeError(
            "No matching Level-2 image found for "
            f"{sensor_code}, path={wrs_path}, "
            f"row={wrs_row}, "
            f"date={target_datetime.date()}."
        )

    candidate_frame = pd.DataFrame(
        candidates
    )

    candidate_frame = candidate_frame.sort_values(
        [
            "match_time_difference_seconds",
            "matched_cloud_cover",
        ],
        ascending=[
            True,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)

    selected = candidate_frame.iloc[0]

    # Landsat 同一 scene 的 Level-1 與 Level-2
    # acquisition time 應非常接近。
    if (
        selected[
            "match_time_difference_seconds"
        ]
        > 12 * 3600
    ):
        raise RuntimeError(
            "Closest Level-2 scene is more than "
            "12 hours from the MARS-S2L timestamp."
        )

    image = ee.Image(
        selected["image_id"]
    )

    return (
        image,
        collection_id,
        selected.to_dict(),
        len(candidate_frame),
    )


def download_image(
    image,
    bands,
    longitude,
    latitude,
    output_path,
):
    region = (
        ee.Geometry.Point([
            float(longitude),
            float(latitude),
        ])
        .buffer(
            PATCH_RADIUS_METERS
        )
        .bounds()
    )

    url = (
        image.select(bands)
        .getDownloadURL({
            "region":
                region.getInfo()[
                    "coordinates"
                ],
            "scale":
                SCALE_METERS,
            "format":
                "GEO_TIFF",
            "filePerBand":
                False,
        })
    )

    temporary_path = (
        output_path.with_suffix(
            ".part.tif"
        )
    )

    with requests.get(
        url,
        stream=True,
        timeout=(30, 300),
    ) as response:
        response.raise_for_status()

        with temporary_path.open(
            "wb"
        ) as file:
            for chunk in (
                response.iter_content(
                    chunk_size=
                        1024 * 1024
                )
            ):
                if chunk:
                    file.write(chunk)

    if (
        not temporary_path.exists()
        or temporary_path.stat().st_size == 0
    ):
        temporary_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Downloaded file is empty."
        )

    temporary_path.replace(
        output_path
    )


def validate_reflectance_raster(path):
    with rasterio.open(path) as dataset:
        array = dataset.read()

        result = {
            "band_count":
                dataset.count,
            "height":
                dataset.height,
            "width":
                dataset.width,
            "dtype":
                str(array.dtype),
            "crs":
                str(dataset.crs),
            "all_zero":
                bool(
                    np.all(array == 0)
                ),
            "has_nan":
                bool(
                    np.isnan(
                        array.astype(float)
                    ).any()
                ),
            "zero_pixel_fraction":
                float(
                    np.mean(array == 0)
                ),
            "minimum_dn":
                float(np.min(array)),
            "maximum_dn":
                float(np.max(array)),
        }

    if result["band_count"] != 6:
        raise RuntimeError(
            "Expected 6 reflectance bands, "
            f"found {result['band_count']}."
        )

    if result["all_zero"]:
        raise RuntimeError(
            "Reflectance raster is entirely zero."
        )

    return result


def validate_qa_raster(path):
    with rasterio.open(path) as dataset:
        qa = dataset.read(1).astype(
            np.uint16
        )

        qa_height = dataset.height
        qa_width = dataset.width
        qa_crs = str(dataset.crs)

    fill = (
        qa & (1 << 0)
    ) != 0

    dilated_cloud = (
        qa & (1 << 1)
    ) != 0

    cirrus = (
        qa & (1 << 2)
    ) != 0

    cloud = (
        qa & (1 << 3)
    ) != 0

    cloud_shadow = (
        qa & (1 << 4)
    ) != 0

    snow = (
        qa & (1 << 5)
    ) != 0

    clear = ~(
        fill
        | dilated_cloud
        | cirrus
        | cloud
        | cloud_shadow
        | snow
    )

    return {
        "qa_height":
            qa_height,
        "qa_width":
            qa_width,
        "qa_crs":
            qa_crs,
        "qa_clear_fraction":
            float(np.mean(clear)),
        "qa_fill_fraction":
            float(np.mean(fill)),
        "qa_cloud_fraction":
            float(
                np.mean(
                    cloud
                    | dilated_cloud
                    | cirrus
                )
            ),
        "qa_shadow_fraction":
            float(
                np.mean(cloud_shadow)
            ),
        "qa_snow_fraction":
            float(np.mean(snow)),
    }


def load_existing_records():
    if not INDEX_OUTPUT.exists():
        return pd.DataFrame()

    return pd.read_csv(
        INDEX_OUTPUT,
        low_memory=False,
    )


def save_checkpoint(
    previous,
    new_records,
):
    frames = []

    if not previous.empty:
        frames.append(previous)

    if new_records:
        frames.append(
            pd.DataFrame(new_records)
        )

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    combined = (
        combined.drop_duplicates(
            subset=["download_id"],
            keep="last",
        )
        .sort_values(
            "download_id"
        )
        .reset_index(drop=True)
    )

    combined.to_csv(
        INDEX_OUTPUT,
        index=False,
    )

    return combined


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    ee.Initialize(
        project=PROJECT
    )

    print(
        f"[OK] Earth Engine initialized: "
        f"{PROJECT}"
    )

    manifest = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required = [
        "download_id",
        "site_key",
        "external_role",
        "evaluation_label",
        "landsat_tile",
        "acquisition_datetime_utc",
        "lon",
        "lat",
    ]

    missing = [
        column
        for column in required
        if column not in manifest.columns
    ]

    if missing:
        raise KeyError(
            f"Missing manifest columns: {missing}"
        )

    manifest[
        "acquisition_datetime_utc"
    ] = pd.to_datetime(
        manifest[
            "acquisition_datetime_utc"
        ],
        errors="coerce",
        utc=True,
    )

    manifest["lon"] = pd.to_numeric(
        manifest["lon"],
        errors="coerce",
    )

    manifest["lat"] = pd.to_numeric(
        manifest["lat"],
        errors="coerce",
    )

    manifest = manifest.dropna(
        subset=[
            "acquisition_datetime_utc",
            "lon",
            "lat",
        ]
    ).copy()

    existing = load_existing_records()

    if not existing.empty:
        completed_ids = set(
            existing.loc[
                existing[
                    "download_status"
                ] == "success",
                "download_id",
            ].astype(str)
        )
    else:
        completed_ids = set()

    remaining = manifest[
        ~manifest["download_id"]
        .astype(str)
        .isin(completed_ids)
    ].copy()

    maximum_downloads_text = (
        os.environ.get(
            "MAX_DOWNLOADS",
            "",
        ).strip()
    )

    if maximum_downloads_text:
        remaining = remaining.head(
            int(maximum_downloads_text)
        ).copy()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    QA_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 110)
    print("MARS-S2L FROZEN EXTERNAL PATCH DOWNLOAD")
    print("=" * 110)

    print(
        "\nManifest rows:",
        len(manifest),
    )

    print(
        "Already downloaded:",
        len(completed_ids),
    )

    print(
        "Remaining this run:",
        len(remaining),
    )

    new_records = []

    for progress, (_, row) in enumerate(
        remaining.iterrows(),
        start=1,
    ):
        download_id = str(
            row["download_id"]
        )

        print(
            f"\n[{progress}/{len(remaining)}] "
            f"{download_id} | "
            f"{row['site_key']} | "
            f"{row['external_role']}",
            flush=True,
        )

        record = row.to_dict()

        reflectance_path = (
            OUTPUT_DIR
            / (
                f"{download_id}_"
                f"{row['external_role']}_"
                f"label_{int(row['evaluation_label'])}"
                ".tif"
            )
        )

        qa_path = (
            QA_OUTPUT_DIR
            / (
                f"{download_id}_QA_PIXEL.tif"
            )
        )

        record.update({
            "patch_path":
                str(reflectance_path),
            "qa_patch_path":
                str(qa_path),
            "download_status":
                "",
            "download_error":
                "",
        })

        last_error = ""

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):
            try:
                parsed = parse_landsat_tile(
                    row["landsat_tile"]
                )

                image, collection_id, (
                    matched
                ), candidate_count = (
                    find_matching_level2_image(
                        sensor_code=
                            parsed[
                                "sensor_code"
                            ],
                        wrs_path=
                            parsed[
                                "wrs_path"
                            ],
                        wrs_row=
                            parsed[
                                "wrs_row"
                            ],
                        target_datetime=
                            row[
                                "acquisition_datetime_utc"
                            ],
                        longitude=
                            row["lon"],
                        latitude=
                            row["lat"],
                    )
                )

                record.update(parsed)

                record.update(matched)

                record[
                    "collection_id"
                ] = collection_id

                record[
                    "matching_l2_candidate_count"
                ] = candidate_count

                if not reflectance_path.exists():
                    download_image(
                        image=image,
                        bands=BANDS,
                        longitude=row["lon"],
                        latitude=row["lat"],
                        output_path=
                            reflectance_path,
                    )

                if not qa_path.exists():
                    download_image(
                        image=image,
                        bands=[QA_BAND],
                        longitude=row["lon"],
                        latitude=row["lat"],
                        output_path=qa_path,
                    )

                validation = (
                    validate_reflectance_raster(
                        reflectance_path
                    )
                )

                qa_validation = (
                    validate_qa_raster(
                        qa_path
                    )
                )

                record.update(validation)
                record.update(qa_validation)

                record[
                    "pixel_sha256"
                ] = sha256_file(
                    reflectance_path
                )

                record[
                    "qa_sha256"
                ] = sha256_file(
                    qa_path
                )

                record[
                    "download_status"
                ] = "success"

                record[
                    "download_error"
                ] = ""

                print(
                    "  [OK] "
                    f"{matched['matched_l2_product_id']} | "
                    f"Δt="
                    f"{matched['match_time_difference_seconds']:.1f}s | "
                    f"clear="
                    f"{qa_validation['qa_clear_fraction']:.3f}",
                    flush=True,
                )

                break

            except Exception as error:
                last_error = str(error)

                print(
                    f"  attempt {attempt}/"
                    f"{MAX_RETRIES} failed: "
                    f"{error}",
                    flush=True,
                )

                time.sleep(
                    2 ** attempt
                )

        if (
            record["download_status"]
            != "success"
        ):
            record[
                "download_status"
            ] = "failed"

            record[
                "download_error"
            ] = last_error

        new_records.append(record)

        save_checkpoint(
            existing,
            new_records,
        )

    result = save_checkpoint(
        existing,
        new_records,
    )

    print("\n" + "=" * 110)
    print("DOWNLOAD SUMMARY")
    print("=" * 110)

    print(
        result[
            "download_status"
        ].value_counts(
            dropna=False
        )
    )

    successful = result[
        result[
            "download_status"
        ] == "success"
    ].copy()

    print(
        "\nSuccessful patches:",
        len(successful),
    )

    if not successful.empty:
        print("\nSuccessful by role:")
        print(
            successful[
                "external_role"
            ].value_counts()
        )

        print("\nSuccessful by sensor:")
        print(
            successful[
                "sensor_code"
            ].value_counts()
        )

        print(
            "\nUnique raster hashes:",
            successful[
                "pixel_sha256"
            ].nunique(),
        )

        print("\nLocal QA clear fraction:")
        print(
            successful[
                "qa_clear_fraction"
            ].describe()
        )

        print("\nAll-zero status:")
        print(
            successful[
                "all_zero"
            ].value_counts(
                dropna=False
            )
        )

    print("\nSaved:")
    print(INDEX_OUTPUT)
    print(OUTPUT_DIR)
    print(QA_OUTPUT_DIR)


if __name__ == "__main__":
    main()
