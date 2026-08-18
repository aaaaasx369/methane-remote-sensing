from pathlib import Path
import os
import time

import ee
import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/310_low_emission_release_intervals_for_s2.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/312_s2_low_emission_exact_overlap_summary.csv"
)

MATCH_OUTPUT = Path(
    "outputs/313_s2_low_emission_exact_overlap_scenes.csv"
)


COLLECTIONS = [
    (
        "S2_SR_HARMONIZED",
        "COPERNICUS/S2_SR_HARMONIZED",
    ),
    (
        "S2_TOA_HARMONIZED",
        "COPERNICUS/S2_HARMONIZED",
    ),
]


MAX_RETRIES = 4
REQUEST_SLEEP_SECONDS = 0.15


def initialize_earth_engine():
    project = os.environ.get(
        "EE_PROJECT"
    )

    if not project:
        raise RuntimeError(
            "EE_PROJECT 尚未設定。請先執行：\n"
            'export EE_PROJECT="methane-release-gee"'
        )

    try:
        ee.Initialize(
            project=project
        )
    except Exception as error:
        raise RuntimeError(
            "Earth Engine 初始化失敗。\n"
            "請確認已完成 earthengine authenticate，"
            "且 EE_PROJECT 正確。\n"
            f"原始錯誤：{error}"
        )

    print(
        "Earth Engine project:",
        project,
    )


def run_with_retries(function):
    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            return function()

        except Exception as error:
            last_error = error

            if attempt == MAX_RETRIES:
                break

            wait_seconds = (
                2 ** (attempt - 1)
            )

            print(
                f"  Earth Engine request failed; "
                f"retry {attempt}/{MAX_RETRIES} "
                f"after {wait_seconds}s: {error}",
                flush=True,
            )

            time.sleep(
                wait_seconds
            )

    raise last_error


def fetch_collection_scenes(
    collection_name,
    collection_id,
    point,
    start_time,
    end_time,
):
    # Earth Engine filterDate 的結束時間不包含在內。
    # 加一秒，讓剛好落在 release_end 的影像也可被找到。
    end_exclusive = (
        end_time
        + pd.Timedelta(seconds=1)
    )

    start_iso = (
        start_time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    )

    end_iso = (
        end_exclusive.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    )

    collection = (
        ee.ImageCollection(
            collection_id
        )
        .filterBounds(point)
        .filterDate(
            start_iso,
            end_iso,
        )
        .sort("system:time_start")
    )

    # select([]) 避免下載 band metadata，
    # 只取得影像識別資訊與 properties。
    info = run_with_retries(
        lambda:
            collection
            .select([])
            .getInfo()
    )

    scene_records = []

    for feature in info.get(
        "features",
        [],
    ):
        properties = feature.get(
            "properties",
            {},
        )

        scene_id = feature.get(
            "id"
        )

        system_index = properties.get(
            "system:index"
        )

        if (
            not scene_id
            and system_index
        ):
            scene_id = (
                f"{collection_id}/"
                f"{system_index}"
            )

        timestamp_ms = properties.get(
            "system:time_start"
        )

        if timestamp_ms is None:
            acquisition_time = pd.NaT
        else:
            acquisition_time = (
                pd.to_datetime(
                    timestamp_ms,
                    unit="ms",
                    utc=True,
                    errors="coerce",
                )
            )

        if scene_id:
            inside_footprint = (
                run_with_retries(
                    lambda:
                        ee.Image(
                            scene_id
                        )
                        .geometry()
                        .intersects(
                            point,
                            ee.ErrorMargin(1),
                        )
                        .getInfo()
                )
            )
        else:
            inside_footprint = False

        if pd.notna(
            acquisition_time
        ):
            strict_overlap = bool(
                start_time
                <= acquisition_time
                <= end_time
            )

            seconds_from_start = (
                acquisition_time
                - start_time
            ).total_seconds()

            seconds_before_end = (
                end_time
                - acquisition_time
            ).total_seconds()

            interval_midpoint = (
                start_time
                + (
                    end_time
                    - start_time
                ) / 2
            )

            seconds_from_midpoint = (
                acquisition_time
                - interval_midpoint
            ).total_seconds()

        else:
            strict_overlap = False
            seconds_from_start = np.nan
            seconds_before_end = np.nan
            seconds_from_midpoint = np.nan

        scene_records.append({
            "collection_name":
                collection_name,
            "collection_id":
                collection_id,
            "scene_id":
                scene_id,
            "system_index":
                system_index,
            "product_id":
                properties.get(
                    "PRODUCT_ID"
                ),
            "spacecraft_name":
                properties.get(
                    "SPACECRAFT_NAME"
                ),
            "mgrs_tile":
                properties.get(
                    "MGRS_TILE"
                ),
            "processing_baseline":
                properties.get(
                    "PROCESSING_BASELINE"
                ),
            "acquisition_time_utc":
                acquisition_time,
            "cloudy_pixel_percentage":
                pd.to_numeric(
                    properties.get(
                        "CLOUDY_PIXEL_PERCENTAGE"
                    ),
                    errors="coerce",
                ),
            "source_inside_footprint":
                bool(
                    inside_footprint
                ),
            "strict_overlap":
                strict_overlap,
            "seconds_from_release_start":
                seconds_from_start,
            "seconds_before_release_end":
                seconds_before_end,
            "seconds_from_interval_midpoint":
                seconds_from_midpoint,
        })

    return scene_records


def load_existing(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            low_memory=False,
        )
    except Exception:
        return pd.DataFrame()


def save_checkpoint(
    summary,
    matches,
):
    SUMMARY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    matches.to_csv(
        MATCH_OUTPUT,
        index=False,
    )


def choose_best_scene(
    strict_scenes,
):
    if not strict_scenes:
        return None

    def sort_key(scene):
        collection_priority = (
            0
            if scene[
                "collection_name"
            ] == "S2_SR_HARMONIZED"
            else 1
        )

        cloud = scene[
            "cloudy_pixel_percentage"
        ]

        if not np.isfinite(cloud):
            cloud = 999.0

        midpoint_distance = abs(
            scene[
                "seconds_from_interval_midpoint"
            ]
        )

        return (
            collection_priority,
            cloud,
            midpoint_distance,
        )

    return sorted(
        strict_scenes,
        key=sort_key,
    )[0]


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            INPUT
        )

    initialize_earth_engine()

    intervals = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required_columns = [
        "release_interval_id",
        "site",
        "lat",
        "lon",
        "release_start_utc",
        "release_end_utc",
        "release_rate_kg_h",
        "emission_bin",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in intervals.columns
    ]

    if missing_columns:
        raise KeyError(
            "缺少必要欄位："
            + ", ".join(
                missing_columns
            )
        )

    intervals[
        "release_start_utc"
    ] = pd.to_datetime(
        intervals[
            "release_start_utc"
        ],
        errors="coerce",
        utc=True,
    )

    intervals[
        "release_end_utc"
    ] = pd.to_datetime(
        intervals[
            "release_end_utc"
        ],
        errors="coerce",
        utc=True,
    )

    intervals["lat"] = pd.to_numeric(
        intervals["lat"],
        errors="coerce",
    )

    intervals["lon"] = pd.to_numeric(
        intervals["lon"],
        errors="coerce",
    )

    intervals[
        "release_rate_kg_h"
    ] = pd.to_numeric(
        intervals[
            "release_rate_kg_h"
        ],
        errors="coerce",
    )

    intervals = intervals.dropna(
        subset=[
            "release_start_utc",
            "release_end_utc",
            "lat",
            "lon",
            "release_rate_kg_h",
        ]
    ).copy()

    intervals = intervals[
        intervals[
            "release_rate_kg_h"
        ].gt(0)
        & intervals[
            "release_rate_kg_h"
        ].lt(1000)
    ].copy()

    intervals = intervals.sort_values(
        [
            "release_start_utc",
            "site",
            "release_interval_id",
        ]
    ).reset_index(drop=True)

    summary = load_existing(
        SUMMARY_OUTPUT
    )

    matches = load_existing(
        MATCH_OUTPUT
    )

    if not summary.empty:
        completed_ids = set(
            summary.loc[
                summary[
                    "search_status"
                ].isin([
                    "success",
                    "no_match",
                ]),
                "release_interval_id",
            ].astype(str)
        )
    else:
        completed_ids = set()

    print("=" * 112)
    print("SENTINEL-2 EXACT LOW-EMISSION OVERLAP SEARCH")
    print("=" * 112)

    print(
        "\nIntervals to search:",
        len(intervals),
    )

    print(
        "Completed checkpoints:",
        len(completed_ids),
    )

    for number, interval in (
        intervals.iterrows()
    ):
        interval_id = str(
            interval[
                "release_interval_id"
            ]
        )

        if interval_id in completed_ids:
            print(
                f"[{number + 1}/{len(intervals)}] "
                f"{interval_id}: checkpoint complete"
            )

            continue

        start_time = interval[
            "release_start_utc"
        ]

        end_time = interval[
            "release_end_utc"
        ]

        point = ee.Geometry.Point([
            float(interval["lon"]),
            float(interval["lat"]),
        ])

        print(
            f"[{number + 1}/{len(intervals)}] "
            f"{interval_id} | "
            f"{interval['site']} | "
            f"{interval['release_rate_kg_h']:.3f} kg/h",
            flush=True,
        )

        interval_scene_records = []
        search_errors = []

        for (
            collection_name,
            collection_id,
        ) in COLLECTIONS:
            try:
                records = (
                    fetch_collection_scenes(
                        collection_name=
                            collection_name,
                        collection_id=
                            collection_id,
                        point=
                            point,
                        start_time=
                            start_time,
                        end_time=
                            end_time,
                    )
                )

                interval_scene_records.extend(
                    records
                )

            except Exception as error:
                search_errors.append(
                    f"{collection_name}: {error}"
                )

        strict_scenes = [
            scene
            for scene
            in interval_scene_records
            if (
                scene[
                    "strict_overlap"
                ]
                and scene[
                    "source_inside_footprint"
                ]
            )
        ]

        for scene in (
            interval_scene_records
        ):
            scene.update({
                "release_interval_id":
                    interval_id,
                "site":
                    interval["site"],
                "lat":
                    interval["lat"],
                "lon":
                    interval["lon"],
                "release_start_utc":
                    start_time,
                "release_end_utc":
                    end_time,
                "release_duration_minutes":
                    interval.get(
                        "release_duration_minutes"
                    ),
                "release_rate_kg_h":
                    interval[
                        "release_rate_kg_h"
                    ],
                "release_rate_source":
                    interval.get(
                        "release_rate_source"
                    ),
                "emission_bin":
                    interval[
                        "emission_bin"
                    ],
            })

        best_scene = choose_best_scene(
            strict_scenes
        )

        sr_count = sum(
            scene[
                "collection_name"
            ] == "S2_SR_HARMONIZED"
            for scene
            in interval_scene_records
        )

        toa_count = sum(
            scene[
                "collection_name"
            ] == "S2_TOA_HARMONIZED"
            for scene
            in interval_scene_records
        )

        strict_unique_keys = {
            (
                str(
                    scene[
                        "acquisition_time_utc"
                    ]
                ),
                str(
                    scene["mgrs_tile"]
                ),
            )
            for scene in strict_scenes
        }

        if search_errors and (
            len(search_errors)
            == len(COLLECTIONS)
        ):
            search_status = "failed"

        elif best_scene is None:
            search_status = "no_match"

        else:
            search_status = "success"

        summary_row = {
            "release_interval_id":
                interval_id,
            "site":
                interval["site"],
            "lat":
                interval["lat"],
            "lon":
                interval["lon"],
            "release_start_utc":
                start_time,
            "release_end_utc":
                end_time,
            "release_duration_minutes":
                interval.get(
                    "release_duration_minutes"
                ),
            "release_rate_kg_h":
                interval[
                    "release_rate_kg_h"
                ],
            "release_rate_source":
                interval.get(
                    "release_rate_source"
                ),
            "emission_bin":
                interval[
                    "emission_bin"
                ],
            "sr_scene_count":
                sr_count,
            "toa_scene_count":
                toa_count,
            "strict_scene_count":
                len(strict_unique_keys),
            "has_exact_s2_overlap":
                best_scene is not None,
            "search_status":
                search_status,
            "search_error":
                " | ".join(
                    search_errors
                ),
            "best_collection":
                (
                    best_scene[
                        "collection_name"
                    ]
                    if best_scene
                    else ""
                ),
            "best_scene_id":
                (
                    best_scene["scene_id"]
                    if best_scene
                    else ""
                ),
            "best_acquisition_time_utc":
                (
                    best_scene[
                        "acquisition_time_utc"
                    ]
                    if best_scene
                    else pd.NaT
                ),
            "best_cloudy_pixel_percentage":
                (
                    best_scene[
                        "cloudy_pixel_percentage"
                    ]
                    if best_scene
                    else np.nan
                ),
            "best_seconds_from_release_start":
                (
                    best_scene[
                        "seconds_from_release_start"
                    ]
                    if best_scene
                    else np.nan
                ),
            "best_seconds_before_release_end":
                (
                    best_scene[
                        "seconds_before_release_end"
                    ]
                    if best_scene
                    else np.nan
                ),
        }

        if not summary.empty:
            summary = summary[
                summary[
                    "release_interval_id"
                ].astype(str).ne(
                    interval_id
                )
            ].copy()

        summary = pd.concat(
            [
                summary,
                pd.DataFrame([
                    summary_row
                ]),
            ],
            ignore_index=True,
            sort=False,
        )

        if not matches.empty:
            matches = matches[
                matches[
                    "release_interval_id"
                ].astype(str).ne(
                    interval_id
                )
            ].copy()

        if interval_scene_records:
            matches = pd.concat(
                [
                    matches,
                    pd.DataFrame(
                        interval_scene_records
                    ),
                ],
                ignore_index=True,
                sort=False,
            )

        summary = summary.sort_values(
            [
                "release_start_utc",
                "release_interval_id",
            ]
        ).reset_index(drop=True)

        if not matches.empty:
            matches = matches.sort_values(
                [
                    "release_start_utc",
                    "release_interval_id",
                    "collection_name",
                    "acquisition_time_utc",
                ]
            ).reset_index(drop=True)

        save_checkpoint(
            summary,
            matches,
        )

        print(
            f"  SR={sr_count}, "
            f"TOA={toa_count}, "
            f"strict={len(strict_unique_keys)}, "
            f"status={search_status}",
            flush=True,
        )

        time.sleep(
            REQUEST_SLEEP_SECONDS
        )

    exact = summary[
        summary[
            "has_exact_s2_overlap"
        ].eq(True)
    ].copy()

    print("\n" + "=" * 112)
    print("EXACT LOW-EMISSION SENTINEL-2 RESULTS")
    print("=" * 112)

    print(
        "\nIntervals searched:",
        len(summary),
    )

    print(
        "Exact-overlap intervals:",
        len(exact),
    )

    print("\nSearch status:")
    print(
        summary[
            "search_status"
        ].value_counts(
            dropna=False
        )
    )

    print("\nExact overlaps by site:")
    print(
        exact["site"]
        .value_counts(
            dropna=False
        )
    )

    print("\nExact overlaps by emission bin:")
    print(
        exact["emission_bin"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print("\nBest collection:")
    print(
        exact[
            "best_collection"
        ].value_counts(
            dropna=False
        )
    )

    if not exact.empty:
        print(
            "\nExact-overlap release-rate range:"
        )

        print(
            "Minimum:",
            exact[
                "release_rate_kg_h"
            ].min(),
        )

        print(
            "Median:",
            exact[
                "release_rate_kg_h"
            ].median(),
        )

        print(
            "Maximum:",
            exact[
                "release_rate_kg_h"
            ].max(),
        )

        print("\nExact matches:")
        print(
            exact[
                [
                    "release_interval_id",
                    "site",
                    "release_rate_kg_h",
                    "emission_bin",
                    "release_start_utc",
                    "release_end_utc",
                    "best_acquisition_time_utc",
                    "best_collection",
                    "best_cloudy_pixel_percentage",
                ]
            ].to_string(
                index=False,
            )
        )

    print("\nSaved:")
    print(SUMMARY_OUTPUT)
    print(MATCH_OUTPUT)


if __name__ == "__main__":
    main()
