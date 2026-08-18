from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests


API_URL = (
    "https://api.carbonmapper.org/"
    "api/v1/catalog/plumes/annotated"
)

RAW_OUTPUT = Path(
    "outputs/200_carbonmapper_plume_catalog_raw.csv"
)

FILTERED_OUTPUT = Path(
    "outputs/201_carbonmapper_high_quality_ch4_candidates.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/202_carbonmapper_candidate_summary.csv"
)

PAGE_SIZE = 100
REQUEST_TIMEOUT_SECONDS = 60

# 與原本研究定義一致。
HIGH_EMISSION_THRESHOLD_KG_H = 1000.0


def first_value(record, names):
    for name in names:
        value = record.get(name)

        if value is not None:
            return value

    return None


def extract_coordinates(record):
    latitude = first_value(
        record,
        [
            "plume_latitude",
            "latitude",
            "lat",
        ],
    )

    longitude = first_value(
        record,
        [
            "plume_longitude",
            "longitude",
            "lon",
        ],
    )

    if (
        latitude is not None
        and longitude is not None
    ):
        return latitude, longitude

    geometry = record.get(
        "geometry_json"
    )

    if isinstance(geometry, str):
        try:
            geometry = json.loads(
                geometry
            )
        except json.JSONDecodeError:
            geometry = None

    if isinstance(geometry, dict):
        coordinates = geometry.get(
            "coordinates"
        )

        if (
            isinstance(coordinates, list)
            and len(coordinates) >= 2
        ):
            # GeoJSON 順序是 longitude, latitude。
            longitude = coordinates[0]
            latitude = coordinates[1]

    return latitude, longitude


def flatten_record(record):
    latitude, longitude = (
        extract_coordinates(record)
    )

    return {
        "plume_id":
            first_value(
                record,
                [
                    "plume_id",
                    "plume_name",
                    "id",
                ],
            ),
        "scene_id":
            record.get("scene_id"),
        "scene_timestamp":
            first_value(
                record,
                [
                    "scene_timestamp",
                    "datetime",
                    "timestamp",
                ],
            ),
        "gas":
            record.get("gas"),
        "instrument":
            record.get("instrument"),
        "platform":
            record.get("platform"),
        "plume_latitude":
            latitude,
        "plume_longitude":
            longitude,
        "emission_auto":
            first_value(
                record,
                [
                    "emission_auto",
                    "emission_rate",
                    "emission_kg_hr",
                ],
            ),
        "emission_uncertainty_auto":
            first_value(
                record,
                [
                    "emission_uncertainty_auto",
                    "emission_uncertainty",
                ],
            ),
        "plume_quality":
            first_value(
                record,
                [
                    "plume_quality",
                    "quality",
                    "overall_quality",
                ],
            ),
        "ipcc_sector":
            record.get("ipcc_sector"),
        "plume_tif":
            record.get("plume_tif"),
        "con_tif":
            record.get("con_tif"),
        "plume_png":
            record.get("plume_png"),
        "plume_rgb_png":
            record.get(
                "plume_rgb_png"
            ),
        "rgb_png":
            record.get("rgb_png"),
        "plume_bounds":
            json.dumps(
                record.get(
                    "plume_bounds"
                )
            ),
        "published_at":
            record.get(
                "published_at"
            ),
    }


def download_catalog():
    session = requests.Session()

    session.headers.update({
        "User-Agent":
            "methane-release-research/1.0",
        "Accept":
            "application/json",
    })

    records = []
    offset = 0
    total_count = None

    while True:
        print(
            f"Requesting offset={offset}"
        )

        response = session.get(
            API_URL,
            params={
                "limit": PAGE_SIZE,
                "offset": offset,
                "sort": "desc",
            },
            timeout=
                REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code in {
            401,
            403,
        }:
            raise RuntimeError(
                "Carbon Mapper API requires "
                "registration or authorization. "
                "Open the Carbon Mapper portal, "
                "register for API access, then "
                "repeat this request."
            )

        response.raise_for_status()

        payload = response.json()

        items = payload.get(
            "items",
            []
        )

        if total_count is None:
            total_count = payload.get(
                "total_count"
            )

            print(
                "Reported total count:",
                total_count,
            )

        if not items:
            break

        records.extend(
            flatten_record(item)
            for item in items
        )

        offset += len(items)

        print(
            "Downloaded records:",
            len(records),
        )

        if (
            total_count is not None
            and offset >= total_count
        ):
            break

        if len(items) < PAGE_SIZE:
            break

    return pd.DataFrame(records)


def main():
    RAW_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalog = download_catalog()

    if catalog.empty:
        raise RuntimeError(
            "Carbon Mapper returned no plume records."
        )

    catalog[
        "scene_datetime_utc"
    ] = pd.to_datetime(
        catalog["scene_timestamp"],
        errors="coerce",
        utc=True,
    )

    catalog[
        "emission_auto"
    ] = pd.to_numeric(
        catalog["emission_auto"],
        errors="coerce",
    )

    catalog[
        "plume_latitude"
    ] = pd.to_numeric(
        catalog["plume_latitude"],
        errors="coerce",
    )

    catalog[
        "plume_longitude"
    ] = pd.to_numeric(
        catalog["plume_longitude"],
        errors="coerce",
    )

    catalog.to_csv(
        RAW_OUTPUT,
        index=False,
    )

    gas_is_ch4 = (
        catalog["gas"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin([
            "CH4",
            "METHANE",
        ])
    )

    quality_text = (
        catalog["plume_quality"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # 只收最可靠的 plume。
    quality_is_good = (
        quality_text.eq("good")
    )

    high_emission = (
        catalog["emission_auto"]
        >= HIGH_EMISSION_THRESHOLD_KG_H
    )

    has_coordinates = (
        catalog["plume_latitude"]
        .notna()
        & catalog["plume_longitude"]
        .notna()
    )

    has_time = catalog[
        "scene_datetime_utc"
    ].notna()

    has_plume_product = (
        catalog["plume_tif"].notna()
        | catalog["con_tif"].notna()
        | catalog["plume_png"].notna()
    )

    candidates = catalog[
        gas_is_ch4
        & quality_is_good
        & high_emission
        & has_coordinates
        & has_time
        & has_plume_product
    ].copy()

    candidates[
        "candidate_label"
    ] = 1

    candidates[
        "ground_truth_type"
    ] = (
        "carbon_mapper_"
        "quality_controlled_plume"
    )

    candidates[
        "landsat_label_status"
    ] = (
        "not_yet_time_matched"
    )

    candidates = (
        candidates.sort_values(
            [
                "emission_auto",
                "scene_datetime_utc",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .drop_duplicates(
            subset=["plume_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    candidates.to_csv(
        FILTERED_OUTPUT,
        index=False,
    )

    summary = (
        candidates.groupby(
            [
                "instrument",
                "platform",
                "ipcc_sector",
            ],
            dropna=False,
        )
        .agg(
            plume_count=(
                "plume_id",
                "size",
            ),
            median_emission_kg_h=(
                "emission_auto",
                "median",
            ),
            maximum_emission_kg_h=(
                "emission_auto",
                "max",
            ),
            first_observation=(
                "scene_datetime_utc",
                "min",
            ),
            last_observation=(
                "scene_datetime_utc",
                "max",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 100)
    print("CARBON MAPPER CATALOG SUMMARY")
    print("=" * 100)

    print(
        "\nTotal plume metadata rows:",
        len(catalog),
    )

    print(
        "High-quality CH4 candidates "
        ">= 1000 kg/h:",
        len(candidates),
    )

    print("\nCandidates by instrument:")
    print(
        candidates[
            "instrument"
        ].value_counts(
            dropna=False
        )
    )

    print("\nCandidates by sector:")
    print(
        candidates[
            "ipcc_sector"
        ].value_counts(
            dropna=False
        ).head(15)
    )

    print("\nEmission-rate summary:")
    print(
        candidates[
            "emission_auto"
        ].describe()
    )

    print("\nSaved:")
    print(RAW_OUTPUT)
    print(FILTERED_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
