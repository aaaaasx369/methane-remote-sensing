from pathlib import Path
import argparse
import io
import re
import time
import zipfile

import ee
import pandas as pd
import requests


PROJECT = "methane-release-gee"

INPUT_PATH = Path(
    "outputs/451_methaneair_s2_below500_nonoverlap_download_shortlist_v1.csv"
)

OUT_DIR = Path(
    "sample_patches/methaneair_s2_below500_v1"
)

OUT_INDEX = Path(
    "outputs/455_methaneair_s2_below500_patch_index_v1.csv"
)

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
]

PATCH_RADIUS_M = 1000
SCALE_M = 20

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)


def initialize_earth_engine():
    try:
        ee.Initialize(project=PROJECT)
        print("Earth Engine initialized.")
    except Exception:
        print("Earth Engine authentication required.")
        ee.Authenticate()
        ee.Initialize(project=PROJECT)
        print("Earth Engine initialized after authentication.")


def clean_text(value):
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def find_column(frame, candidates, name):
    for column in candidates:
        if column in frame.columns:
            return column

    raise KeyError(
        f"Cannot find {name} column. Tried: "
        + ", ".join(candidates)
    )


def to_boolean(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


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
                    "Downloaded ZIP contains no TIFF."
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


def save_records(records):
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
        raise FileNotFoundError(INPUT_PATH)

    frame = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    event_column = find_column(
        frame,
        ["event_id"],
        "event ID",
    )

    scene_column = find_column(
        frame,
        ["scene_id"],
        "scene ID",
    )

    latitude_column = find_column(
        frame,
        ["latitude", "lat"],
        "latitude",
    )

    longitude_column = find_column(
        frame,
        ["longitude", "lon"],
        "longitude",
    )

    emission_column = find_column(
        frame,
        ["emission_kg_hr", "release_rate_kg_h"],
        "emission rate",
    )

    frame[latitude_column] = pd.to_numeric(
        frame[latitude_column],
        errors="coerce",
    )

    frame[longitude_column] = pd.to_numeric(
        frame[longitude_column],
        errors="coerce",
    )

    frame[emission_column] = pd.to_numeric(
        frame[emission_column],
        errors="coerce",
    )

    if "ready_for_direct_download" in frame.columns:
        frame = frame[
            to_boolean(
                frame["ready_for_direct_download"]
            )
        ].copy()

    frame = frame.dropna(
        subset=[
            event_column,
            scene_column,
            latitude_column,
            longitude_column,
            emission_column,
        ]
    )

    frame = frame[
        frame[emission_column].gt(0)
        & frame[emission_column].lt(500)
    ].copy()

    frame = frame.drop_duplicates(
        subset=[event_column],
        keep="first",
    )

    frame = frame.sort_values(
        [
            emission_column,
            event_column,
        ]
    ).reset_index(drop=True)

    if len(frame) != 21:
        raise RuntimeError(
            "Expected 21 download-ready events, "
            f"found {len(frame)}."
        )

    start_index = max(0, args.start)

    if args.limit is None:
        end_index = len(frame)
    else:
        end_index = min(
            start_index + args.limit,
            len(frame),
        )

    batch = frame.iloc[
        start_index:end_index
    ].copy()

    print("=" * 105)
    print("METHANEAIR–S2 BELOW-500 KG/H EXACT DOWNLOAD")
    print("=" * 105)

    print("\nInput rows:", len(frame))
    print(
        "Downloading rows:",
        start_index,
        "to",
        end_index - 1,
    )
    print("Batch size:", len(batch))

    records = []

    for local_number, (row_index, row) in enumerate(
        batch.iterrows(),
        start=1,
    ):
        event_id = str(
            row[event_column]
        )

        scene_id = str(
            row[scene_column]
        )

        latitude = float(
            row[latitude_column]
        )

        longitude = float(
            row[longitude_column]
        )

        emission_kg_hr = float(
            row[emission_column]
        )

        if not scene_id.startswith(
            "COPERNICUS/"
        ):
            scene_id = (
                f"{S2_COLLECTION}/"
                f"{scene_id}"
            )

        filename = (
            f"MA_S2_LOW500_{row_index + 1:03d}_"
            f"{clean_text(event_id)}.tif"
        )

        output_path = (
            OUT_DIR / filename
        )

        metadata = {
            key: value
            for key, value in row.to_dict().items()
        }

        metadata.update({
            "download_sequence":
                row_index + 1,

            "filename":
                filename,

            "relative_path":
                str(output_path),

            "patch_path":
                str(output_path),

            "event_id":
                event_id,

            "scene_id":
                scene_id,

            "label":
                1,

            "label_type":
                "methaneair_positive",

            "source_dataset":
                "MethaneAIR",

            "sensor":
                "Sentinel-2",

            "emission_kg_hr":
                emission_kg_hr,

            "bands":
                ",".join(BANDS),

            "patch_radius_m":
                PATCH_RADIUS_M,

            "scale_m":
                SCALE_M,

            "download_status":
                "pending",
        })

        print(
            f"\n[{local_number:02d}/{len(batch):02d}] "
            f"{event_id} | "
            f"{emission_kg_hr:.1f} kg/h"
        )

        try:
            image = ee.Image(
                scene_id
            )

            actual_system_index = image.get(
                "system:index"
            ).getInfo()

            actual_time_ms = image.get(
                "system:time_start"
            ).getInfo()

            actual_time = pd.to_datetime(
                actual_time_ms,
                unit="ms",
                utc=True,
            )

            metadata[
                "actual_system_index"
            ] = actual_system_index

            metadata[
                "actual_acquisition_time_utc"
            ] = actual_time.isoformat()

            metadata[
                "actual_product_id"
            ] = image.get(
                "PRODUCT_ID"
            ).getInfo()

            metadata[
                "actual_mgrs_tile"
            ] = image.get(
                "MGRS_TILE"
            ).getInfo()

            metadata[
                "actual_spacecraft_name"
            ] = image.get(
                "SPACECRAFT_NAME"
            ).getInfo()

            metadata[
                "actual_scene_cloud_percentage"
            ] = image.get(
                "CLOUDY_PIXEL_PERCENTAGE"
            ).getInfo()

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
                        PATCH_RADIUS_M
                    )
                    .bounds()
                )

                export_image = (
                    image
                    .select(BANDS)
                    .clip(region)
                )

                url = export_image.getDownloadURL({
                    "name":
                        clean_text(event_id),

                    "scale":
                        SCALE_M,

                    "region":
                        region,

                    "format":
                        "GEO_TIFF",

                    "filePerBand":
                        False,
                })

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
        save_records(records)

        time.sleep(1)

    save_records(records)

    result = pd.DataFrame(records)

    successful = result[
        "download_status"
    ].isin([
        "success",
        "success_existing",
    ])

    print("\n" + "=" * 105)
    print("DOWNLOAD SUMMARY")
    print("=" * 105)

    print("\nIndex rows:", len(result))

    print("\nDownload status:")
    print(
        result[
            "download_status"
        ].value_counts(
            dropna=False
        )
    )

    print(
        "\nSuccessful:",
        int(successful.sum()),
        "/",
        len(result),
    )

    print("\nSaved index:")
    print(OUT_INDEX)


if __name__ == "__main__":
    main()
