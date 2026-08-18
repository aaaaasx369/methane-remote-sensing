from pathlib import Path
from io import BytesIO
import hashlib
import zipfile

import ee
import numpy as np
import pandas as pd
import rasterio
import requests


PROJECT = "methane-release-gee"

INPUT_CSV = Path(
    "outputs/88_ehrenberg_priority_download_batch.csv"
)

OUTPUT_DIR = Path(
    "sample_patches/ehrenberg_priority_landsat"
)

OUTPUT_INDEX = Path(
    "outputs/90_ehrenberg_priority_landsat_patch_index.csv"
)

SITE_NAME = "Ehrenberg_AZ_release_stack"
SITE_LAT = 33.630645
SITE_LON = -114.489150

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


def initialize_earth_engine():
    try:
        ee.Initialize(project=PROJECT)

    except Exception:
        print("Earth Engine authentication required.")
        ee.Authenticate()
        ee.Initialize(project=PROJECT)

    print(
        f"Earth Engine initialized: {PROJECT}"
    )


def collection_from_product_id(product_id):
    product_id = str(product_id)

    if product_id.startswith("LC08"):
        return "LANDSAT/LC08/C02/T1_L2"

    if product_id.startswith("LC09"):
        return "LANDSAT/LC09/C02/T1_L2"

    raise ValueError(
        f"Unsupported Landsat product ID: {product_id}"
    )


def get_exact_image(product_id):
    collection_id = collection_from_product_id(
        product_id
    )

    collection = (
        ee.ImageCollection(collection_id)
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

    return (
        ee.Image(collection.first()),
        collection_id,
    )


def download_geotiff(url, output_path):
    response = requests.get(
        url,
        timeout=300,
    )

    response.raise_for_status()

    content = response.content
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
                    "Downloaded ZIP contains no GeoTIFF."
                )

            if len(tif_names) > 1:
                print(
                    "Warning: ZIP contains multiple TIFFs; "
                    "using the first one."
                )

            with archive.open(tif_names[0]) as source:
                output_path.write_bytes(
                    source.read()
                )

    else:
        output_path.write_bytes(content)


def calculate_file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def calculate_pixel_hash(array):
    contiguous = np.ascontiguousarray(array)

    return hashlib.sha256(
        contiguous.tobytes()
    ).hexdigest()


def validate_patch(path):
    with rasterio.open(path) as dataset:
        array = dataset.read()

        array_float = array.astype(
            np.float64
        )

        validation = {
            "band_count": int(dataset.count),
            "height": int(dataset.height),
            "width": int(dataset.width),
            "dtype": str(dataset.dtypes[0]),
            "crs": str(dataset.crs),
            "nodata": dataset.nodata,
            "all_zero": bool(
                np.all(array == 0)
            ),
            "has_nan": bool(
                np.isnan(array_float).any()
            ),
            "zero_pixel_fraction": float(
                np.mean(array == 0)
            ),
            "raw_dn_min": float(
                np.nanmin(array_float)
            ),
            "raw_dn_max": float(
                np.nanmax(array_float)
            ),
            "file_size_bytes": int(
                path.stat().st_size
            ),
            "file_sha256":
                calculate_file_sha256(path),
            "pixel_hash":
                calculate_pixel_hash(array),
        }

        if dataset.count != 6:
            raise ValueError(
                f"Expected 6 bands, got "
                f"{dataset.count}."
            )

        if validation["all_zero"]:
            raise ValueError(
                "Downloaded patch is completely zero."
            )

        if validation["has_nan"]:
            raise ValueError(
                "Downloaded patch contains NaN values."
            )

        if (
            dataset.height < 60
            or dataset.width < 60
        ):
            raise ValueError(
                "Patch dimensions are unexpectedly small: "
                f"{dataset.height} × {dataset.width}"
            )

        return validation


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing input table: {INPUT_CSV}"
        )

    batch = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    required_columns = [
        "overpass_id",
        "LANDSAT_PRODUCT_ID",
        "final_label",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in batch.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    if len(batch) != 6:
        raise ValueError(
            f"Expected 6 priority scenes, "
            f"found {len(batch)}."
        )

    if batch[
        "LANDSAT_PRODUCT_ID"
    ].duplicated().any():
        raise ValueError(
            "Duplicate Landsat product IDs exist "
            "in the priority table."
        )

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

    for _, row in batch.iterrows():
        overpass_id = str(
            row["overpass_id"]
        ).strip()

        product_id = str(
            row["LANDSAT_PRODUCT_ID"]
        ).strip()

        label = int(
            row["final_label"]
        )

        filename = (
            f"EH_Landsat_{overpass_id}_"
            f"label_{label}.tif"
        )

        output_path = (
            OUTPUT_DIR / filename
        )

        print("\n" + "=" * 95)
        print(
            f"{overpass_id} | label={label}"
        )
        print(product_id)
        print("=" * 95)

        record = row.to_dict()

        record.update({
            "site_name": SITE_NAME,
            "lat": SITE_LAT,
            "lon": SITE_LON,
            "label": label,
            "filename": filename,
            "file_path": str(output_path),
            "patch_radius_m": PATCH_RADIUS,
            "scale_m": SCALE,
            "bands": ",".join(COMMON_BANDS),
            "original_landsat_bands":
                ",".join(LANDSAT_BANDS),
            "reflectance_scaling_applied":
                False,
            "download_status": "pending",
            "error": "",
        })

        try:
            image, collection_id = (
                get_exact_image(product_id)
            )

            record["collection_id"] = (
                collection_id
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

            for key, value in properties.items():
                record[
                    f"gee_{key}"
                ] = value

            time_ms = properties.get(
                "system:time_start"
            )

            if time_ms is not None:
                record[
                    "landsat_image_time_utc"
                ] = pd.to_datetime(
                    time_ms,
                    unit="ms",
                    utc=True,
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
                    f"[EXISTS] {output_path}"
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
                f"[OK] bands="
                f"{validation['band_count']} | "
                f"shape="
                f"{validation['height']}×"
                f"{validation['width']} | "
                f"DN="
                f"{validation['raw_dn_min']:.0f}–"
                f"{validation['raw_dn_max']:.0f} | "
                f"zero="
                f"{validation['zero_pixel_fraction']:.4f}"
            )

        except Exception as error:
            record["download_status"] = (
                "error"
            )

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

    print("\n" + "=" * 95)
    print("EHRENBERG PRIORITY DOWNLOAD SUMMARY")
    print("=" * 95)

    print("\nDownload status:")
    print(
        result["download_status"]
        .value_counts(dropna=False)
    )

    print("\nLabel counts:")
    print(
        result["label"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nSensor counts:")

    sensor_column = (
        "landsat_sensor"
        if "landsat_sensor" in result.columns
        else "gee_SPACECRAFT_ID"
    )

    print(
        result[sensor_column]
        .value_counts(dropna=False)
    )

    summary_columns = [
        column
        for column in [
            "overpass_id",
            "label",
            "landsat_sensor",
            "LANDSAT_PRODUCT_ID",
            "landsat_image_time_utc",
            "CLOUD_COVER",
            "download_status",
            "band_count",
            "height",
            "width",
            "dtype",
            "all_zero",
            "has_nan",
            "zero_pixel_fraction",
            "raw_dn_min",
            "raw_dn_max",
            "pixel_hash",
            "file_path",
            "error",
        ]
        if column in result.columns
    ]

    print("\nDownloaded scenes:")
    print(
        result[
            summary_columns
        ].to_string(index=False)
    )

    successful = result[
        result["download_status"].isin([
            "success",
            "success_existing",
        ])
    ]

    if len(successful) > 0:
        duplicate_hashes = successful[
            successful["pixel_hash"]
            .duplicated(keep=False)
        ]

        print("\nDuplicate pixel hashes "
              "within this batch:")

        if len(duplicate_hashes) == 0:
            print("None")

        else:
            print(
                duplicate_hashes[
                    [
                        "overpass_id",
                        "LANDSAT_PRODUCT_ID",
                        "pixel_hash",
                    ]
                ].to_string(index=False)
            )

    print(f"\nSaved index: {OUTPUT_INDEX}")


if __name__ == "__main__":
    main()
