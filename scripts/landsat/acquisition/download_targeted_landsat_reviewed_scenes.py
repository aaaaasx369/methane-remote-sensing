from pathlib import Path
from io import BytesIO
import zipfile

import ee
import numpy as np
import pandas as pd
import rasterio
import requests


PROJECT = "methane-release-gee"

COLLECTION_ID = "LANDSAT/LC09/C02/T1_L2"

PATCH_RADIUS = 1000
SCALE = 30

LANDSAT_BANDS = [
    "SR_B2",
    "SR_B3",
    "SR_B4",
    "SR_B5",
    "SR_B6",
    "SR_B7",
]

COMMON_BANDS = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "SWIR1",
    "SWIR2",
]

SITE_LAT = 32.821821
SITE_LON = -111.785773

OUTPUT_DIR = Path(
    "sample_patches/"
    "controlled_release_landsat_targeted_reviewed"
)

OUTPUT_INDEX = Path(
    "outputs/81_landsat_targeted_reviewed_patch_index.csv"
)


TARGET_SCENES = [
    {
        "overpass_id": "OP_012",
        "label": 0,
        "label_status":
            "provisional_negative_zero_flow_tolerance",
        "label_confidence": "medium_high",
        "landsat_product_id":
            "LC09_L2SP_036037_20221018_20230325_02_T1",
    },
    {
        "overpass_id": "OP_013",
        "label": 1,
        "label_status":
            "confirmed_positive_overlap",
        "label_confidence": "high",
        "landsat_product_id":
            "LC09_L2SP_037037_20221025_20230324_02_T1",
    },
]


def initialize_earth_engine():
    ee.Initialize(project=PROJECT)
    print(
        f"Earth Engine initialized with project: "
        f"{PROJECT}"
    )


def download_geotiff(url, output_path):
    response = requests.get(
        url,
        timeout=300,
    )

    response.raise_for_status()

    content = response.content

    # Earth Engine 有時會回傳 ZIP，
    # 有時直接回傳 GeoTIFF。
    memory_file = BytesIO(content)

    if zipfile.is_zipfile(memory_file):
        memory_file.seek(0)

        with zipfile.ZipFile(memory_file) as archive:
            tif_names = [
                name
                for name in archive.namelist()
                if name.lower().endswith(
                    (".tif", ".tiff")
                )
            ]

            if len(tif_names) == 0:
                raise RuntimeError(
                    "Earth Engine ZIP contains no GeoTIFF."
                )

            # filePerBand=False 時正常應只有一個多波段檔。
            with archive.open(tif_names[0]) as source:
                output_path.write_bytes(
                    source.read()
                )

    else:
        output_path.write_bytes(content)


def validate_patch(path):
    with rasterio.open(path) as dataset:
        array = dataset.read()

        finite_mask = np.isfinite(array)

        metadata = {
            "band_count": dataset.count,
            "height": dataset.height,
            "width": dataset.width,
            "dtype": str(dataset.dtypes[0]),
            "crs": str(dataset.crs),
            "nodata": dataset.nodata,
            "all_zero": bool(
                np.all(array == 0)
            ),
            "has_nan": bool(
                np.isnan(
                    array.astype("float64")
                ).any()
            ),
            "finite_fraction": float(
                finite_mask.mean()
            ),
            "pixel_min": float(
                np.nanmin(array)
            ),
            "pixel_max": float(
                np.nanmax(array)
            ),
            "file_size_bytes":
                path.stat().st_size,
        }

        if dataset.count != 6:
            raise ValueError(
                f"Expected 6 bands, got "
                f"{dataset.count}: {path}"
            )

        if metadata["all_zero"]:
            raise ValueError(
                f"Patch is completely zero: {path}"
            )

        if dataset.width < 60 or dataset.height < 60:
            raise ValueError(
                "Patch dimensions are unexpectedly small: "
                f"{dataset.width} × {dataset.height}"
            )

        return metadata


def get_exact_image(product_id):
    collection = (
        ee.ImageCollection(COLLECTION_ID)
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


def main():
    initialize_earth_engine()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_INDEX.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    point = ee.Geometry.Point([
        SITE_LON,
        SITE_LAT,
    ])

    region = point.buffer(
        PATCH_RADIUS
    ).bounds()

    records = []

    for target in TARGET_SCENES:
        overpass_id = target["overpass_id"]
        label = int(target["label"])
        product_id = target[
            "landsat_product_id"
        ]

        filename = (
            f"CR_Landsat_{overpass_id}_"
            f"label_{label}.tif"
        )

        output_path = OUTPUT_DIR / filename

        print("\n" + "=" * 90)
        print(
            f"{overpass_id}: {product_id}"
        )
        print("=" * 90)

        record = {
            **target,
            "filename": filename,
            "file_path": str(output_path),
            "lat": SITE_LAT,
            "lon": SITE_LON,
            "collection_id": COLLECTION_ID,
            "bands": ",".join(COMMON_BANDS),
            "original_landsat_bands":
                ",".join(LANDSAT_BANDS),
            "patch_radius_m": PATCH_RADIUS,
            "scale_m": SCALE,
            "reflectance_scaling_applied":
                False,
            "download_status": "pending",
            "error": "",
        }

        try:
            image = get_exact_image(
                product_id
            )

            properties = image.toDictionary([
                "system:index",
                "system:time_start",
                "LANDSAT_PRODUCT_ID",
                "LANDSAT_SCENE_ID",
                "SPACECRAFT_ID",
                "WRS_PATH",
                "WRS_ROW",
                "CLOUD_COVER",
                "CLOUD_COVER_LAND",
                "PROCESSING_LEVEL",
            ]).getInfo()

            record.update(properties)

            time_ms = properties.get(
                "system:time_start"
            )

            if time_ms is not None:
                record["landsat_image_time"] = (
                    pd.to_datetime(
                        time_ms,
                        unit="ms",
                        utc=True,
                    )
                )

            selected_image = (
                image
                .select(
                    LANDSAT_BANDS,
                    COMMON_BANDS,
                )
                .clip(region)
            )

            if output_path.exists():
                print(
                    f"[EXISTS] Validating "
                    f"{output_path}"
                )

                record["download_status"] = (
                    "success_existing"
                )

            else:
                print(
                    f"[DOWNLOAD] {output_path}"
                )

                url = selected_image.getDownloadURL({
                    "name": output_path.stem,
                    "scale": SCALE,
                    "region": region,
                    "format": "GEO_TIFF",
                    "filePerBand": False,
                })

                download_geotiff(
                    url,
                    output_path,
                )

                record["download_status"] = (
                    "success"
                )

            validation = validate_patch(
                output_path
            )

            record.update(validation)

            print(
                f"[OK] bands={validation['band_count']}, "
                f"shape={validation['height']}×"
                f"{validation['width']}, "
                f"dtype={validation['dtype']}, "
                f"min={validation['pixel_min']}, "
                f"max={validation['pixel_max']}"
            )

        except Exception as error:
            record["download_status"] = "error"
            record["error"] = str(error)

            print(
                f"[ERROR] {overpass_id}: "
                f"{error}"
            )

        records.append(record)

        pd.DataFrame(
            records
        ).to_csv(
            OUTPUT_INDEX,
            index=False,
        )

    result = pd.DataFrame(records)

    print("\n" + "=" * 90)
    print("TARGETED LANDSAT DOWNLOAD SUMMARY")
    print("=" * 90)

    print("\nDownload status:")
    print(
        result["download_status"]
        .value_counts(dropna=False)
    )

    print("\nDownloaded scene summary:")

    display_columns = [
        column
        for column in [
            "overpass_id",
            "label",
            "label_status",
            "landsat_product_id",
            "landsat_image_time",
            "WRS_PATH",
            "WRS_ROW",
            "CLOUD_COVER",
            "download_status",
            "band_count",
            "height",
            "width",
            "dtype",
            "all_zero",
            "has_nan",
            "pixel_min",
            "pixel_max",
            "file_path",
            "error",
        ]
        if column in result.columns
    ]

    print(
        result[
            display_columns
        ].to_string(index=False)
    )

    print(f"\nSaved index: {OUTPUT_INDEX}")


if __name__ == "__main__":
    main()
