from pathlib import Path
import argparse
import io
import time
import zipfile

import ee
import pandas as pd
import requests


PROJECT = "methane-release-gee"

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

INPUT_PATH = Path(
    "outputs/416_landsat_selected_negative_download_manifest_v1.csv"
)

OUT_INDEX = Path(
    "outputs/417_landsat_selected_negative_download_index_v1.csv"
)


def initialize_earth_engine():
    try:
        ee.Initialize(project=PROJECT)
        print("Earth Engine initialized.")
    except Exception:
        print("Earth Engine authentication required.")
        ee.Authenticate()
        ee.Initialize(project=PROJECT)
        print("Earth Engine initialized after authentication.")


def download_file(url, output_path, timeout=300):
    response = requests.get(
        url,
        stream=True,
        timeout=timeout,
    )

    response.raise_for_status()
    content = response.content

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if content[:2] == b"PK":
        with zipfile.ZipFile(
            io.BytesIO(content)
        ) as archive:
            tif_names = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".tif")
            ]

            if not tif_names:
                raise RuntimeError(
                    "Downloaded ZIP contains no GeoTIFF."
                )

            with (
                archive.open(tif_names[0]) as source,
                open(output_path, "wb") as destination,
            ):
                destination.write(source.read())

    else:
        output_path.write_bytes(content)

    if (
        not output_path.exists()
        or output_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "Downloaded file is missing or empty."
        )


def find_exact_image(
    collection_id,
    system_index,
    product_id,
):
    collection = ee.ImageCollection(
        collection_id
    )

    matches = collection.filter(
        ee.Filter.eq(
            "system:index",
            system_index,
        )
    )

    count = matches.size().getInfo()
    match_method = "system:index"

    if count == 0:
        matches = collection.filter(
            ee.Filter.eq(
                "LANDSAT_PRODUCT_ID",
                product_id,
            )
        )

        count = matches.size().getInfo()
        match_method = "LANDSAT_PRODUCT_ID"

    if count == 0:
        raise RuntimeError(
            "Exact Landsat image not found: "
            f"{collection_id} | "
            f"{system_index} | "
            f"{product_id}"
        )

    if count > 1:
        raise RuntimeError(
            "More than one exact Landsat image found: "
            f"{count}"
        )

    image = ee.Image(
        matches.first()
    )

    return image, match_method


def save_index(records):
    OUT_INDEX.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(records).to_csv(
        OUT_INDEX,
        index=False,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    initialize_earth_engine()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            INPUT_PATH
        )

    dataframe = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    required_columns = [
        "sample_id",
        "label",
        "patch_path",
        "site_lat_standard",
        "site_lon_standard",
        "acquisition_time_utc",
        "gee_collection_id_standard",
        "gee_system_index_standard",
        "landsat_product_id_standard",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "Manifest missing columns: "
            + ", ".join(missing_columns)
        )

    if len(dataframe) != 28:
        raise RuntimeError(
            "Expected 28 selected negatives, "
            f"found {len(dataframe)}."
        )

    dataframe[
        "site_lat_standard"
    ] = pd.to_numeric(
        dataframe[
            "site_lat_standard"
        ],
        errors="coerce",
    )

    dataframe[
        "site_lon_standard"
    ] = pd.to_numeric(
        dataframe[
            "site_lon_standard"
        ],
        errors="coerce",
    )

    dataframe[
        "_expected_time"
    ] = pd.to_datetime(
        dataframe[
            "acquisition_time_utc"
        ],
        errors="coerce",
        utc=True,
    )

    invalid = dataframe[
        dataframe[
            [
                "site_lat_standard",
                "site_lon_standard",
                "_expected_time",
            ]
        ].isna().any(axis=1)
    ]

    if not invalid.empty:
        raise RuntimeError(
            "Manifest contains invalid coordinates or times:\n"
            + invalid[
                [
                    "sample_id",
                    "site_lat_standard",
                    "site_lon_standard",
                    "acquisition_time_utc",
                ]
            ].to_string(index=False)
        )

    start_index = max(
        args.start,
        0,
    )

    if args.limit is None:
        end_index = len(dataframe)
    else:
        end_index = min(
            start_index + args.limit,
            len(dataframe),
        )

    batch = dataframe.iloc[
        start_index:end_index
    ].copy()

    print("=" * 105)
    print(
        "EXACT LANDSAT MATCHED-NEGATIVE DOWNLOAD"
    )
    print("=" * 105)

    print("\nInput:", INPUT_PATH)
    print("Total manifest rows:", len(dataframe))
    print(
        "Downloading rows:",
        start_index,
        "to",
        end_index - 1,
    )
    print("Batch size:", len(batch))

    records = []

    for local_number, (_, row) in enumerate(
        batch.iterrows(),
        start=1,
    ):
        sample_id = str(
            row["sample_id"]
        )

        collection_id = str(
            row[
                "gee_collection_id_standard"
            ]
        )

        system_index = str(
            row[
                "gee_system_index_standard"
            ]
        )

        expected_product_id = str(
            row[
                "landsat_product_id_standard"
            ]
        )

        latitude = float(
            row[
                "site_lat_standard"
            ]
        )

        longitude = float(
            row[
                "site_lon_standard"
            ]
        )

        expected_time = row[
            "_expected_time"
        ]

        output_path = Path(
            str(row["patch_path"])
        )

        metadata = {
            key: value
            for key, value in row.to_dict().items()
            if key != "_expected_time"
        }

        metadata.update({
            "download_sequence":
                local_number,

            "patch_radius_m":
                PATCH_RADIUS,

            "scale_m":
                SCALE,

            "bands":
                ",".join(
                    COMMON_BANDS
                ),

            "original_landsat_bands":
                ",".join(
                    LANDSAT_BANDS
                ),

            "download_status":
                "pending",
        })

        print(
            f"\n[{local_number:02d}/{len(batch):02d}] "
            f"{sample_id}"
        )

        try:
            raw_image, match_method = (
                find_exact_image(
                    collection_id=
                        collection_id,

                    system_index=
                        system_index,

                    product_id=
                        expected_product_id,
                )
            )

            actual_system_index = str(
                raw_image.get(
                    "system:index"
                ).getInfo()
            )

            actual_product_id = str(
                raw_image.get(
                    "LANDSAT_PRODUCT_ID"
                ).getInfo()
            )

            actual_time_ms = (
                raw_image.get(
                    "system:time_start"
                ).getInfo()
            )

            actual_time = pd.to_datetime(
                actual_time_ms,
                unit="ms",
                utc=True,
            )

            time_difference_seconds = abs(
                (
                    actual_time
                    - expected_time
                ).total_seconds()
            )

            metadata.update({
                "exact_match_method":
                    match_method,

                "actual_system_index":
                    actual_system_index,

                "actual_landsat_product_id":
                    actual_product_id,

                "actual_acquisition_time_utc":
                    actual_time.isoformat(),

                "time_difference_seconds":
                    time_difference_seconds,

                "actual_cloud_cover":
                    raw_image.get(
                        "CLOUD_COVER"
                    ).getInfo(),

                "actual_spacecraft_id":
                    raw_image.get(
                        "SPACECRAFT_ID"
                    ).getInfo(),

                "actual_sensor_id":
                    raw_image.get(
                        "SENSOR_ID"
                    ).getInfo(),
            })

            if (
                actual_product_id
                != expected_product_id
            ):
                raise RuntimeError(
                    "Product ID mismatch: "
                    f"expected={expected_product_id}, "
                    f"actual={actual_product_id}"
                )

            if (
                time_difference_seconds
                > 180
            ):
                raise RuntimeError(
                    "Acquisition time mismatch: "
                    f"{time_difference_seconds:.1f} seconds"
                )

            if output_path.exists():
                metadata[
                    "download_status"
                ] = "success_existing"

                metadata[
                    "file_size_bytes"
                ] = output_path.stat().st_size

                print(
                    "[SKIP EXISTING]",
                    output_path,
                )

            else:
                point = ee.Geometry.Point(
                    [
                        longitude,
                        latitude,
                    ]
                )

                region = (
                    point.buffer(
                        PATCH_RADIUS
                    )
                    .bounds()
                )

                export_image = (
                    raw_image
                    .select(
                        LANDSAT_BANDS,
                        COMMON_BANDS,
                    )
                    .clip(region)
                )

                url = (
                    export_image
                    .getDownloadURL({
                        "name":
                            sample_id,

                        "scale":
                            SCALE,

                        "region":
                            region,

                        "format":
                            "GEO_TIFF",

                        "filePerBand":
                            False,
                    })
                )

                print(
                    "[DOWNLOAD]",
                    output_path,
                )

                download_file(
                    url,
                    output_path,
                )

                metadata[
                    "download_status"
                ] = "success"

                metadata[
                    "file_size_bytes"
                ] = output_path.stat().st_size

                print(
                    "[DONE]",
                    output_path,
                )

        except Exception as error:
            metadata[
                "download_status"
            ] = "error"

            metadata[
                "error"
            ] = str(error)

            print(
                "[ERROR]",
                error,
            )

        records.append(metadata)
        save_index(records)

        time.sleep(1)

    save_index(records)

    result = pd.DataFrame(records)

    print("\n" + "=" * 105)
    print("DOWNLOAD SUMMARY")
    print("=" * 105)

    print("\nIndex rows:", len(result))

    if not result.empty:
        print("\nDownload status:")
        print(
            result[
                "download_status"
            ].value_counts(
                dropna=False
            )
        )

        successful = result[
            "download_status"
        ].isin([
            "success",
            "success_existing",
        ])

        print(
            "\nSuccessful:",
            int(successful.sum()),
            "/",
            len(result),
        )

        if (
            "time_difference_seconds"
            in result.columns
        ):
            print(
                "Maximum time difference:",
                pd.to_numeric(
                    result[
                        "time_difference_seconds"
                    ],
                    errors="coerce",
                ).max(),
                "seconds",
            )

    print("\nSaved index:")
    print(OUT_INDEX)


if __name__ == "__main__":
    main()
