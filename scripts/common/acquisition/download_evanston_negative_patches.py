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
    "outputs/143_evanston_negative_split_manifest.csv"
)

OUTPUT_DIR = Path(
    "sample_patches/"
    "evanston_negative_landsat"
)

INDEX_OUTPUT = Path(
    "outputs/145_evanston_negative_patch_index.csv"
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


def find_exact_image(
    collection_id,
    product_id,
):
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
            f"{product_id}: expected one "
            f"image, found {count}."
        )

    return ee.Image(
        collection.first()
    )


def download_patch(
    image,
    longitude,
    latitude,
    output_path,
):
    region = (
        ee.Geometry.Point([
            longitude,
            latitude,
        ])
        .buffer(
            PATCH_RADIUS_METERS
        )
        .bounds()
    )

    url = (
        image.select(BANDS)
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

    temporary = output_path.with_suffix(
        ".part.tif"
    )

    with requests.get(
        url,
        stream=True,
        timeout=(30, 300),
    ) as response:
        response.raise_for_status()

        with temporary.open(
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

    if temporary.stat().st_size == 0:
        temporary.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Downloaded file is empty."
        )

    temporary.replace(
        output_path
    )


def validate_raster(path):
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
                float(
                    np.min(array)
                ),
            "maximum_dn":
                float(
                    np.max(array)
                ),
        }

    if result["band_count"] != 6:
        raise RuntimeError(
            "Expected 6 bands, found "
            f"{result['band_count']}."
        )

    if result["all_zero"]:
        raise RuntimeError(
            "Raster is entirely zero."
        )

    return result


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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    print("=" * 105)
    print("DOWNLOADING EVANSTON NEGATIVE PATCHES")
    print("=" * 105)

    for index, row in (
        manifest.iterrows()
    ):
        negative_id = str(
            row["negative_id"]
        )

        product_id = str(
            row["landsat_product_id"]
        )

        collection_id = str(
            row["collection_id"]
        )

        filename = (
            f"{negative_id}_"
            f"{product_id}_label_0.tif"
        )

        output_path = (
            OUTPUT_DIR / filename
        )

        print(
            f"\n[{index + 1}/"
            f"{len(manifest)}] "
            f"{negative_id}"
        )

        print(
            f"{row['negative_role']} | "
            f"{product_id}"
        )

        record = row.to_dict()

        record.update({
            "patch_path":
                str(output_path),
            "download_status":
                "",
            "download_error":
                "",
        })

        try:
            image = find_exact_image(
                collection_id=
                    collection_id,
                product_id=
                    product_id,
            )

            if output_path.exists():
                print(
                    "[SKIP] Existing file; "
                    "validating."
                )

            else:
                download_patch(
                    image=image,
                    longitude=float(
                        row["longitude"]
                    ),
                    latitude=float(
                        row["latitude"]
                    ),
                    output_path=
                        output_path,
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
    print("NEGATIVE DOWNLOAD SUMMARY")
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
        print("\nSuccessful by role:")
        print(
            successful[
                "negative_role"
            ].value_counts()
        )

        print("\nSuccessful by sensor:")
        print(
            successful[
                "expected_sensor"
            ].value_counts()
        )

        print(
            "\nUnique raster hashes:",
            successful[
                "pixel_sha256"
            ].nunique(),
        )

        print(
            "\nAll zero:",
            successful[
                "all_zero"
            ].value_counts()
        )

        print(
            "\nHas NaN:",
            successful[
                "has_nan"
            ].value_counts()
        )

    print("\nSaved:")
    print(INDEX_OUTPUT)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
