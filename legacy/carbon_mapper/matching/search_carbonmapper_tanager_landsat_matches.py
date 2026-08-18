from __future__ import annotations

import os
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd


PROJECT = os.environ.get(
    "EE_PROJECT",
    "methane-release-gee",
)

INPUT = Path(
    "outputs/209_carbonmapper_tanager_scene_candidates.csv"
)

SEARCH_SUMMARY_OUTPUT = Path(
    "outputs/212_carbonmapper_tanager_landsat_search_summary.csv"
)

ALL_MATCHES_OUTPUT = Path(
    "outputs/213_carbonmapper_tanager_landsat_all_matches.csv"
)

MATCH_CANDIDATES_OUTPUT = Path(
    "outputs/214_carbonmapper_tanager_landsat_temporal_candidates.csv"
)

LOG_INTERVAL = 25
REQUEST_SLEEP_SECONDS = 0.15
MAX_RETRIES = 3

SEARCH_WINDOW_HOURS = 24

STRONG_MATCH_MINUTES = 30.0
NEAR_MATCH_MINUTES = 90.0
WEAK_MATCH_MINUTES = 360.0

MAX_IMAGES_PER_SCENE = 20


def parse_ee_time(value):
    if value is None:
        return pd.NaT

    return pd.to_datetime(
        value,
        unit="ms",
        errors="coerce",
        utc=True,
    )


def classify_temporal_match(
    minutes_difference,
):
    if not np.isfinite(
        minutes_difference
    ):
        return "no_match"

    if (
        minutes_difference
        <= STRONG_MATCH_MINUTES
    ):
        return "strong_le_30min"

    if (
        minutes_difference
        <= NEAR_MATCH_MINUTES
    ):
        return "near_30_to_90min"

    if (
        minutes_difference
        <= WEAK_MATCH_MINUTES
    ):
        return "weak_90_to_360min"

    return "outside_360min"


def get_collection():
    landsat_8 = ee.ImageCollection(
        "LANDSAT/LC08/C02/T1_L2"
    )

    landsat_9 = ee.ImageCollection(
        "LANDSAT/LC09/C02/T1_L2"
    )

    return landsat_8.merge(
        landsat_9
    )


def search_one_scene(
    collection,
    scene_row,
):
    scene_time = pd.Timestamp(
        scene_row["scene_datetime_utc"]
    )

    latitude = float(
        scene_row[
            "representative_latitude"
        ]
    )

    longitude = float(
        scene_row[
            "representative_longitude"
        ]
    )

    start_time = (
        scene_time
        - pd.Timedelta(
            hours=SEARCH_WINDOW_HOURS
        )
    )

    end_time = (
        scene_time
        + pd.Timedelta(
            hours=SEARCH_WINDOW_HOURS
        )
    )

    point = ee.Geometry.Point([
        longitude,
        latitude,
    ])

    filtered = (
        collection
        .filterBounds(point)
        .filterDate(
            start_time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            end_time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
        )
        .sort("system:time_start")
        .limit(MAX_IMAGES_PER_SCENE)
    )

    information = filtered.getInfo()

    candidate_rows = []

    for feature in information.get(
        "features",
        [],
    ):
        properties = feature.get(
            "properties",
            {},
        )

        acquisition_time = parse_ee_time(
            properties.get(
                "system:time_start"
            )
        )

        if pd.isna(acquisition_time):
            continue

        difference_minutes = abs(
            (
                acquisition_time
                - scene_time
            ).total_seconds()
        ) / 60.0

        spacecraft_id = str(
            properties.get(
                "SPACECRAFT_ID",
                "",
            )
        )

        if "8" in spacecraft_id:
            sensor = "Landsat-8"
            collection_id = (
                "LANDSAT/LC08/C02/T1_L2"
            )

        elif "9" in spacecraft_id:
            sensor = "Landsat-9"
            collection_id = (
                "LANDSAT/LC09/C02/T1_L2"
            )

        else:
            sensor = spacecraft_id
            collection_id = ""

        candidate_rows.append({
            "scene_key":
                scene_row["scene_key"],
            "scene_id":
                scene_row.get(
                    "scene_id",
                    "",
                ),
            "tanager_scene_datetime_utc":
                scene_time,
            "representative_plume_id":
                scene_row[
                    "representative_plume_id"
                ],
            "representative_latitude":
                latitude,
            "representative_longitude":
                longitude,
            "representative_emission_kg_h":
                scene_row[
                    "representative_emission_kg_h"
                ],
            "representative_uncertainty_kg_h":
                scene_row.get(
                    "representative_uncertainty_kg_h",
                    np.nan,
                ),
            "representative_relative_uncertainty":
                scene_row.get(
                    "representative_relative_uncertainty",
                    np.nan,
                ),
            "plume_count":
                scene_row["plume_count"],
            "landsat_product_id":
                properties.get(
                    "LANDSAT_PRODUCT_ID"
                ),
            "landsat_sensor":
                sensor,
            "collection_id":
                collection_id,
            "landsat_acquisition_time_utc":
                acquisition_time,
            "absolute_time_difference_minutes":
                difference_minutes,
            "temporal_match_class":
                classify_temporal_match(
                    difference_minutes
                ),
            "same_utc_date":
                (
                    acquisition_time.date()
                    == scene_time.date()
                ),
            "cloud_cover":
                properties.get(
                    "CLOUD_COVER"
                ),
            "cloud_cover_land":
                properties.get(
                    "CLOUD_COVER_LAND"
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
            "system_index":
                properties.get(
                    "system:index",
                    feature.get("id"),
                ),
        })

    return candidate_rows


def load_existing(path):
    if path.exists():
        return pd.read_csv(
            path,
            low_memory=False,
        )

    return pd.DataFrame()


def save_checkpoint(
    summary_rows,
    candidate_rows,
):
    summary = pd.DataFrame(
        summary_rows
    )

    if not summary.empty:
        # 舊 checkpoint 從 CSV 讀入後是字串，
        # 新搜尋結果則可能是 pandas Timestamp。
        # 排序前統一轉為 UTC datetime。
        if "scene_datetime_utc" in summary.columns:
            summary["scene_datetime_utc"] = pd.to_datetime(
                summary["scene_datetime_utc"],
                errors="coerce",
                utc=True,
            )

        summary = (
            summary.drop_duplicates(
                subset=["scene_key"],
                keep="last",
            )
            .sort_values(
                "scene_datetime_utc",
                na_position="last",
            )
            .reset_index(drop=True)
        )

    summary.to_csv(
        SEARCH_SUMMARY_OUTPUT,
        index=False,
    )

    matches = pd.DataFrame(
        candidate_rows
    )

    if not matches.empty:
        # 同樣統一既有 checkpoint 與新結果的時間型別。
        for time_column in [
            "tanager_scene_datetime_utc",
            "landsat_acquisition_time_utc",
        ]:
            if time_column in matches.columns:
                matches[time_column] = pd.to_datetime(
                    matches[time_column],
                    errors="coerce",
                    utc=True,
                )

        matches["absolute_time_difference_minutes"] = pd.to_numeric(
            matches["absolute_time_difference_minutes"],
            errors="coerce",
        )

        matches = (
            matches.drop_duplicates(
                subset=[
                    "scene_key",
                    "landsat_product_id",
                ],
                keep="last",
            )
            .sort_values(
                [
                    "scene_key",
                    "absolute_time_difference_minutes",
                ],
                na_position="last",
            )
            .reset_index(drop=True)
        )

    matches.to_csv(
        ALL_MATCHES_OUTPUT,
        index=False,
    )

    if not matches.empty:
        temporal_candidates = matches[
            matches[
                "absolute_time_difference_minutes"
            ]
            <= NEAR_MATCH_MINUTES
        ].copy()

    else:
        temporal_candidates = (
            pd.DataFrame()
        )

    temporal_candidates.to_csv(
        MATCH_CANDIDATES_OUTPUT,
        index=False,
    )


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

    scenes = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    scenes[
        "scene_datetime_utc"
    ] = pd.to_datetime(
        scenes[
            "scene_datetime_utc"
        ],
        errors="coerce",
        utc=True,
    )

    scenes = scenes.dropna(
        subset=[
            "scene_key",
            "scene_datetime_utc",
            "representative_latitude",
            "representative_longitude",
        ]
    ).copy()

    scenes = scenes.sort_values(
        [
            "representative_emission_kg_h",
            "scene_datetime_utc",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    maximum_scenes_text = os.environ.get(
        "MAX_SCENES",
        "",
    ).strip()

    if maximum_scenes_text:
        maximum_scenes = int(
            maximum_scenes_text
        )

        scenes = scenes.head(
            maximum_scenes
        ).copy()

    existing_summary = load_existing(
        SEARCH_SUMMARY_OUTPUT
    )

    existing_matches = load_existing(
        ALL_MATCHES_OUTPUT
    )

    if not existing_summary.empty:
        completed_mask = (
            existing_summary[
                "search_status"
            ].isin([
                "success",
                "no_landsat_scene",
            ])
        )

        completed_scene_keys = set(
            existing_summary.loc[
                completed_mask,
                "scene_key",
            ].astype(str)
        )

    else:
        completed_scene_keys = set()

    summary_rows = (
        existing_summary.to_dict(
            "records"
        )
        if not existing_summary.empty
        else []
    )

    candidate_rows = (
        existing_matches.to_dict(
            "records"
        )
        if not existing_matches.empty
        else []
    )

    remaining = scenes[
        ~scenes["scene_key"]
        .astype(str)
        .isin(completed_scene_keys)
    ].copy()

    print("=" * 110)
    print("CARBON MAPPER–LANDSAT TEMPORAL SEARCH")
    print("=" * 110)

    print(
        "\nInput Tanager scenes:",
        len(scenes),
    )

    print(
        "Already completed:",
        len(
            scenes[
                scenes["scene_key"]
                .astype(str)
                .isin(completed_scene_keys)
            ]
        ),
    )

    print(
        "Remaining scenes:",
        len(remaining),
    )

    collection = get_collection()

    for progress_number, (
        _,
        scene,
    ) in enumerate(
        remaining.iterrows(),
        start=1,
    ):
        scene_key = str(
            scene["scene_key"]
        )

        print(
            f"[{progress_number}/"
            f"{len(remaining)}] "
            f"{scene_key}",
            flush=True,
        )

        result_rows = None
        last_error = ""

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):
            try:
                result_rows = (
                    search_one_scene(
                        collection,
                        scene,
                    )
                )

                break

            except Exception as error:
                last_error = str(error)

                print(
                    f"  attempt "
                    f"{attempt}/"
                    f"{MAX_RETRIES} failed: "
                    f"{error}",
                    flush=True,
                )

                time.sleep(
                    2 ** attempt
                )

        base_summary = {
            "scene_key":
                scene_key,
            "scene_id":
                scene.get(
                    "scene_id",
                    "",
                ),
            "scene_datetime_utc":
                scene[
                    "scene_datetime_utc"
                ],
            "representative_plume_id":
                scene[
                    "representative_plume_id"
                ],
            "representative_latitude":
                scene[
                    "representative_latitude"
                ],
            "representative_longitude":
                scene[
                    "representative_longitude"
                ],
            "representative_emission_kg_h":
                scene[
                    "representative_emission_kg_h"
                ],
            "representative_uncertainty_kg_h":
                scene.get(
                    "representative_uncertainty_kg_h",
                    np.nan,
                ),
            "representative_relative_uncertainty":
                scene.get(
                    "representative_relative_uncertainty",
                    np.nan,
                ),
            "plume_count":
                scene["plume_count"],
        }

        if result_rows is None:
            summary_rows.append({
                **base_summary,
                "search_status":
                    "error",
                "search_error":
                    last_error,
                "landsat_scene_count":
                    np.nan,
                "nearest_product_id":
                    "",
                "nearest_time_difference_minutes":
                    np.nan,
                "nearest_temporal_match_class":
                    "error",
            })

        elif len(result_rows) == 0:
            summary_rows.append({
                **base_summary,
                "search_status":
                    "no_landsat_scene",
                "search_error":
                    "",
                "landsat_scene_count":
                    0,
                "nearest_product_id":
                    "",
                "nearest_time_difference_minutes":
                    np.nan,
                "nearest_temporal_match_class":
                    "no_match",
            })

            print(
                "  no Landsat scene "
                "within ±24 hours",
                flush=True,
            )

        else:
            result_frame = pd.DataFrame(
                result_rows
            )

            result_frame = (
                result_frame.sort_values(
                    [
                        "absolute_time_difference_minutes",
                        "cloud_cover",
                    ],
                    ascending=[
                        True,
                        True,
                    ],
                    na_position="last",
                )
                .reset_index(drop=True)
            )

            nearest = (
                result_frame.iloc[0]
            )

            candidate_rows.extend(
                result_frame.to_dict(
                    "records"
                )
            )

            summary_rows.append({
                **base_summary,
                "search_status":
                    "success",
                "search_error":
                    "",
                "landsat_scene_count":
                    len(result_frame),
                "nearest_product_id":
                    nearest[
                        "landsat_product_id"
                    ],
                "nearest_landsat_sensor":
                    nearest[
                        "landsat_sensor"
                    ],
                "nearest_acquisition_time_utc":
                    nearest[
                        "landsat_acquisition_time_utc"
                    ],
                "nearest_time_difference_minutes":
                    nearest[
                        "absolute_time_difference_minutes"
                    ],
                "nearest_temporal_match_class":
                    nearest[
                        "temporal_match_class"
                    ],
                "nearest_cloud_cover":
                    nearest[
                        "cloud_cover"
                    ],
                "nearest_wrs_path":
                    nearest[
                        "wrs_path"
                    ],
                "nearest_wrs_row":
                    nearest[
                        "wrs_row"
                    ],
            })

            print(
                "  nearest: "
                f"{nearest['landsat_product_id']} | "
                f"Δt="
                f"{nearest['absolute_time_difference_minutes']:.1f} min | "
                f"{nearest['temporal_match_class']}",
                flush=True,
            )

        if (
            progress_number
            % LOG_INTERVAL == 0
        ):
            save_checkpoint(
                summary_rows,
                candidate_rows,
            )

            print(
                "  [checkpoint saved]",
                flush=True,
            )

        time.sleep(
            REQUEST_SLEEP_SECONDS
        )

    save_checkpoint(
        summary_rows,
        candidate_rows,
    )

    summary = pd.read_csv(
        SEARCH_SUMMARY_OUTPUT,
        low_memory=False,
    )

    matches = pd.read_csv(
        ALL_MATCHES_OUTPUT,
        low_memory=False,
    )

    print("\n" + "=" * 110)
    print("TEMPORAL SEARCH SUMMARY")
    print("=" * 110)

    print("\nSearch status:")
    print(
        summary[
            "search_status"
        ].value_counts(
            dropna=False
        )
    )

    print("\nNearest match classes:")
    print(
        summary[
            "nearest_temporal_match_class"
        ].value_counts(
            dropna=False
        )
    )

    if not matches.empty:
        print("\nAll Landsat match classes:")
        print(
            matches[
                "temporal_match_class"
            ].value_counts(
                dropna=False
            )
        )

        strong = matches[
            matches[
                "absolute_time_difference_minutes"
            ]
            <= STRONG_MATCH_MINUTES
        ]

        near = matches[
            matches[
                "absolute_time_difference_minutes"
            ]
            <= NEAR_MATCH_MINUTES
        ]

        print(
            "\nStrong matches <=30 min:",
            len(strong),
        )

        print(
            "Unique Tanager scenes with "
            "strong match:",
            strong[
                "scene_key"
            ].nunique(),
        )

        print(
            "\nNear matches <=90 min:",
            len(near),
        )

        print(
            "Unique Tanager scenes with "
            "near match:",
            near[
                "scene_key"
            ].nunique(),
        )

        if not near.empty:
            display_columns = [
                "scene_key",
                "representative_plume_id",
                "representative_emission_kg_h",
                "tanager_scene_datetime_utc",
                "landsat_product_id",
                "landsat_sensor",
                "landsat_acquisition_time_utc",
                "absolute_time_difference_minutes",
                "temporal_match_class",
                "cloud_cover",
            ]

            print("\nBest temporal candidates:")
            print(
                near[
                    display_columns
                ]
                .sort_values(
                    [
                        "absolute_time_difference_minutes",
                        "representative_emission_kg_h",
                    ],
                    ascending=[
                        True,
                        False,
                    ],
                )
                .head(30)
                .to_string(
                    index=False,
                    float_format=lambda value:
                        f"{value:.2f}",
                )
            )

    print("\nSaved:")
    print(SEARCH_SUMMARY_OUTPUT)
    print(ALL_MATCHES_OUTPUT)
    print(MATCH_CANDIDATES_OUTPUT)


if __name__ == "__main__":
    main()
