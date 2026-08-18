from __future__ import annotations

import hashlib
import os
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
    "outputs/140_evanston_confirmed_positive_download_manifest.csv"
)

OUTPUT_DIR = Path(
    "sample_patches/"
    "evanston_confirmed_positive_landsat"
)

INDEX_OUTPUT = Path(
    "outputs/141_evanston_confirmed_positive_patch_index.csv"
)

BANDS = [
    "SR_B2",
    "SR_B3",
    "SR_B4",
    "SR_B5",
    "SR_B6",
    "SR_B7",
]

PATCH_RADIUS_METERS = 1000
SCALE_METERS = 30


def sha256_file(path: Path) -> str:
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


def find_exact_image(
    collection_id: str,
    product_id: str,
) -> ee.Image:
    collection = (
        ee.ImageCollection(
            collection_id
        )
        .filter(
            ee.Filter.eq(
                "LANDSAT_PRODUCT_ID",
                product_id,
            )
        )
    )

    count = int(
        collection.size().getInfo()
    )

    if count != 1:
        raise RuntimeError(
            f"Expected exactly one image for "
            f"{product_id}, found {count}."
        )

    return ee.Image(
        collection.first()
    )


def download_geotiff(
    image: ee.Image,
    longitude: float,
    latitude: float,
    output_path: Path,
) -> None:
    point = ee.Geometry.Point([
        longitude,
        latitude,
    ])

    region = (
        point.buffer(
            PATCH_RADIUS_METERS
        )
        .bounds()
    )

    selected = image.select(
        BANDS
    )

    region_coordinates = (
        region.getInfo()["coordinates"]
    )

    url = selected.getDownloadURL({
        "region": region_coordinates,
        "scale": SCALE_METERS,
        "format": "GEO_TIFF",
        "filePerBand": False,
    })

    temporary_path = output_path.with_suffix(
        ".part.tif"
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
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file.write(chunk)

    if temporary_path.stat().st_size == 0:
        temporary_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Downloaded file is empty."
        )

    temporary_path.replace(
        output_path
    )


def validate_raster(path: Path) -> dict:
    with rasterio.open(path) as dataset:
        array = dataset.read()

        metadata = {
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
                        array.astype(
                            np.float64
                        )
                    ).any()
                ),
            "zero_pixel_fraction":
                float(
                    np.mean(array == 0)
                ),
            "minimum_dn":
                float(
                    np.min(array)
                ),
            "maximum_dn":
                float(
                    np.max(array)
                ),
        }

    if metadata["band_count"] != 6:
        raise RuntimeError(
            f"Expected 6 bands, found "
            f"{metadata['band_count']}."
        )

    if metadata["all_zero"]:
        raise RuntimeError(
            "Raster is entirely zero."
        )

    return metadata


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            INPUT
        )

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
        "overpass_id",
        "landsat_product_id",
        "collection_id",
        "latitude",
        "longitude",
        "acquisition_time_utc",
        "flow_at_scene_kg_h",
    ]

    missing = [
        column
        for column in required
        if column not in manifest.columns
    ]

    if missing:
        raise KeyError(
            f"Missing columns: {missing}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    print("=" * 105)
    print("DOWNLOADING EVANSTON POSITIVE PATCHES")
    print("=" * 105)

    for number, row in manifest.iterrows():
        overpass_id = str(
            row["overpass_id"]
        )

        product_id = str(
            row["landsat_product_id"]
        )

        collection_id = str(
            row["collection_id"]
        )

        latitude = float(
            row["latitude"]
        )

        longitude = float(
            row["longitude"]
        )

        filename = (
            f"{overpass_id}_"
            f"{product_id}_label_1.tif"
        )

        output_path = (
            OUTPUT_DIR / filename
        )

        print(
            f"\n[{number + 1}/{len(manifest)}] "
            f"{overpass_id}"
        )

        print(product_id)

        record = row.to_dict()

        record.update({
            "label": 1,
            "site_key": "evanston",
            "patch_path":
                str(output_path),
            "download_status":
                "",
            "download_error":
                "",
        })

        try:
            image = find_exact_image(
                collection_id,
                product_id,
            )

            if output_path.exists():
                print(
                    "[SKIP] File already exists; "
                    "validating it."
                )

            else:
                download_geotiff(
                    image=image,
                    longitude=longitude,
                    latitude=latitude,
                    output_path=output_path,
                )

            validation = validate_raster(
                output_path
            )

            record.update(
                validation
            )

            record[
                "pixel_sha256"
            ] = sha256_file(
                output_path
            )

            record[
                "download_status"
            ] = "success"

            print(
                f"[OK] "
                f"{validation['band_count']} bands | "
                f"{validation['height']}×"
                f"{validation['width']} | "
                f"dtype={validation['dtype']} | "
                f"zero_fraction="
                f"{validation['zero_pixel_fraction']:.6f}"
            )

        except Exception as error:
            record[
                "download_status"
            ] = "failed"

            record[
                "download_error"
            ] = str(error)

            print(
                f"[ERROR] {error}"
            )

        records.append(record)

        pd.DataFrame(
            records
        ).to_csv(
            INDEX_OUTPUT,
            index=False,
        )

    result = pd.DataFrame(
        records
    )

    print("\n" + "=" * 105)
    print("DOWNLOAD SUMMARY")
    print("=" * 105)

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
    ]

    print(
        "\nSuccessful patches:",
        len(successful),
    )

    if not successful.empty:
        print("\nSensor counts:")
        print(
            successful[
                "expected_sensor"
            ].value_counts()
        )

        print("\nFlow at acquisition:")
        print(
            successful[
                "flow_at_scene_kg_h"
            ].describe()
        )

        print(
            "\nUnique raster hashes:",
            successful[
                "pixel_sha256"
            ].nunique(),
        )

    print("\nSaved:")
    print(INDEX_OUTPUT)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
