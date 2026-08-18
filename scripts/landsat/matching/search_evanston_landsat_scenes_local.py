from pathlib import Path

import ee
import pandas as pd


PROJECT = "methane-release-gee"

INPUT = Path(
    "outputs/131_evanston_unique_landsat_overpasses.csv"
)

OUTPUT = Path(
    "outputs/133_evanston_landsat_scene_candidates.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/134_evanston_landsat_scene_search_audit.csv"
)


COLLECTIONS = {
    "Landsat-8": "LANDSAT/LC08/C02/T1_L2",
    "Landsat-9": "LANDSAT/LC09/C02/T1_L2",
}


def format_utc_from_milliseconds(value):
    if value is None:
        return ""

    timestamp = pd.to_datetime(
        value,
        unit="ms",
        errors="coerce",
        utc=True,
    )

    if pd.isna(timestamp):
        return ""

    return timestamp.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def minutes_from_release_window(
    acquisition_time,
    release_start,
    release_end,
):
    if (
        pd.isna(acquisition_time)
        or pd.isna(release_start)
        or pd.isna(release_end)
    ):
        return None

    if release_start <= acquisition_time <= release_end:
        return 0.0

    if acquisition_time < release_start:
        difference = release_start - acquisition_time
    else:
        difference = acquisition_time - release_end

    return difference.total_seconds() / 60


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT}"
        )

    ee.Initialize(
        project=PROJECT
    )

    print(
        f"[OK] Earth Engine initialized: {PROJECT}"
    )

    overpasses = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    print(
        f"Overpasses to search: {len(overpasses)}"
    )

    result_rows = []
    audit_rows = []

    for _, row in overpasses.iterrows():
        overpass_id = str(
            row["overpass_id"]
        )

        sensor = str(
            row["landsat_sensor"]
        )

        collection_id = str(
            row.get(
                "collection_id",
                "",
            )
        ).strip()

        if not collection_id:
            collection_id = COLLECTIONS.get(
                sensor,
                "",
            )

        if not collection_id:
            print(
                f"[SKIP] {overpass_id}: "
                f"unknown sensor {sensor}"
            )

            audit_rows.append({
                "overpass_id": overpass_id,
                "status": "unknown_sensor",
                "scene_count": 0,
                "error": sensor,
            })

            continue

        latitude = float(
            row["latitude"]
        )

        longitude = float(
            row["longitude"]
        )

        release_date = pd.to_datetime(
            row["acquisition_date"],
            errors="raise",
        )

        search_start = (
            release_date
            - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")

        search_end = (
            release_date
            + pd.Timedelta(days=2)
        ).strftime("%Y-%m-%d")

        release_start = pd.to_datetime(
            row.get(
                "release_time_min_utc"
            ),
            errors="coerce",
            utc=True,
        )

        release_end = pd.to_datetime(
            row.get(
                "release_time_max_utc"
            ),
            errors="coerce",
            utc=True,
        )

        try:
            point = ee.Geometry.Point([
                longitude,
                latitude,
            ])

            collection = (
                ee.ImageCollection(
                    collection_id
                )
                .filterBounds(point)
                .filterDate(
                    search_start,
                    search_end,
                )
                .sort(
                    "system:time_start"
                )
            )

            collection_info = (
                collection.getInfo()
            )

            features = collection_info.get(
                "features",
                [],
            )

            print(
                f"[OK] {overpass_id} | "
                f"{release_date.date()} | "
                f"{sensor} | scenes={len(features)}"
            )

            audit_rows.append({
                "overpass_id":
                    overpass_id,
                "release_date":
                    release_date.strftime(
                        "%Y-%m-%d"
                    ),
                "landsat_sensor":
                    sensor,
                "collection_id":
                    collection_id,
                "search_start":
                    search_start,
                "search_end":
                    search_end,
                "scene_count":
                    len(features),
                "status":
                    (
                        "success"
                        if features
                        else "no_scene"
                    ),
                "error":
                    "",
            })

            for feature in features:
                properties = feature.get(
                    "properties",
                    {},
                )

                acquisition_time = (
                    pd.to_datetime(
                        properties.get(
                            "system:time_start"
                        ),
                        unit="ms",
                        errors="coerce",
                        utc=True,
                    )
                )

                exact_overlap = bool(
                    pd.notna(
                        acquisition_time
                    )
                    and pd.notna(
                        release_start
                    )
                    and pd.notna(
                        release_end
                    )
                    and release_start
                    <= acquisition_time
                    <= release_end
                )

                result_rows.append({
                    "overpass_id":
                        overpass_id,
                    "site_key":
                        row.get(
                            "site_key",
                            "evanston",
                        ),
                    "release_date":
                        release_date.strftime(
                            "%Y-%m-%d"
                        ),
                    "expected_sensor":
                        sensor,
                    "collection_id":
                        collection_id,
                    "latitude":
                        latitude,
                    "longitude":
                        longitude,
                    "release_rows":
                        row.get(
                            "release_rows"
                        ),
                    "flow_min_kg_h":
                        row.get(
                            "flow_min_kg_h"
                        ),
                    "flow_median_kg_h":
                        row.get(
                            "flow_median_kg_h"
                        ),
                    "flow_max_kg_h":
                        row.get(
                            "flow_max_kg_h"
                        ),
                    "release_time_min_utc":
                        row.get(
                            "release_time_min_utc"
                        ),
                    "release_time_max_utc":
                        row.get(
                            "release_time_max_utc"
                        ),
                    "representative_release_id":
                        row.get(
                            "representative_release_id"
                        ),
                    "all_release_ids":
                        row.get(
                            "all_release_ids"
                        ),
                    "system_index":
                        properties.get(
                            "system:index",
                            feature.get("id"),
                        ),
                    "landsat_product_id":
                        properties.get(
                            "LANDSAT_PRODUCT_ID"
                        ),
                    "spacecraft_id":
                        properties.get(
                            "SPACECRAFT_ID"
                        ),
                    "acquisition_time_utc":
                        format_utc_from_milliseconds(
                            properties.get(
                                "system:time_start"
                            )
                        ),
                    "cloud_cover":
                        properties.get(
                            "CLOUD_COVER"
                        ),
                    "wrs_path":
                        properties.get(
                            "WRS_PATH"
                        ),
                    "wrs_row":
                        properties.get(
                            "WRS_ROW"
                        ),
                    "collection_category":
                        properties.get(
                            "COLLECTION_CATEGORY"
                        ),
                    "exact_release_overlap":
                        exact_overlap,
                    "minutes_from_release_window":
                        minutes_from_release_window(
                            acquisition_time,
                            release_start,
                            release_end,
                        ),
                })

        except Exception as error:
            print(
                f"[ERROR] {overpass_id}: {error}"
            )

            audit_rows.append({
                "overpass_id":
                    overpass_id,
                "release_date":
                    release_date.strftime(
                        "%Y-%m-%d"
                    ),
                "landsat_sensor":
                    sensor,
                "collection_id":
                    collection_id,
                "search_start":
                    search_start,
                "search_end":
                    search_end,
                "scene_count":
                    0,
                "status":
                    "error",
                "error":
                    str(error),
            })

    results = pd.DataFrame(
        result_rows
    )

    audit = pd.DataFrame(
        audit_rows
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not results.empty:
        results["cloud_cover"] = (
            pd.to_numeric(
                results["cloud_cover"],
                errors="coerce",
            )
        )

        results = (
            results.sort_values([
                "release_date",
                "expected_sensor",
                "cloud_cover",
            ])
            .drop_duplicates(
                subset=[
                    "overpass_id",
                    "landsat_product_id",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    results.to_csv(
        OUTPUT,
        index=False,
    )

    audit.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 100)
    print("SEARCH SUMMARY")
    print("=" * 100)

    print(
        "Overpasses searched:",
        len(overpasses),
    )

    print(
        "Scene candidates:",
        len(results),
    )

    print("\nSearch status:")
    print(
        audit["status"].value_counts(
            dropna=False
        )
    )

    if not results.empty:
        display_columns = [
            "overpass_id",
            "release_date",
            "expected_sensor",
            "landsat_product_id",
            "acquisition_time_utc",
            "cloud_cover",
            "wrs_path",
            "wrs_row",
            "flow_max_kg_h",
            "exact_release_overlap",
            "minutes_from_release_window",
        ]

        print("\nCandidates:")
        print(
            results[
                display_columns
            ].to_string(
                index=False
            )
        )

    print("\nSaved:")
    print(OUTPUT)
    print(AUDIT_OUTPUT)


if __name__ == "__main__":
    main()
