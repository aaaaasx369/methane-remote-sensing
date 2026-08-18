from pathlib import Path
from math import radians, sin, cos, asin, sqrt
import os
import time

import ee
import numpy as np
import pandas as pd


EVENT_INPUT = Path(
    "outputs/10_final_events_for_gee.csv"
)

INTERVAL_INPUT = Path(
    "outputs/309_all_exact_release_intervals_for_s2.csv"
)

CANDIDATE_OUTPUT = Path(
    "outputs/500_s5p_ch4_event_orbit_candidates_v1.csv"
)

EVENT_OUTPUT = Path(
    "outputs/501_s5p_ch4_event_availability_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/502_s5p_ch4_event_availability_report_v1.txt"
)


COLLECTION_ID = (
    "COPERNICUS/S5P/OFFL/L3_CH4"
)

CH4_BAND = (
    "CH4_column_volume_mixing_ratio_"
    "dry_air_bias_corrected"
)

UNCERTAINTY_BAND = (
    "CH4_column_volume_mixing_ratio_"
    "dry_air_uncertainty"
)

GRID_SCALE_M = 1113.2

# 用 10 km 半徑判斷 source 周圍是否有有效 XCH4。
LOCAL_RADIUS_M = 10000

# event 前後 24 小時搜尋 S5P orbit。
NEAR_TIME_HOURS = 24

# exact release interval 與 event 的位置容許距離。
INTERVAL_DISTANCE_LIMIT_M = 1000

S5P_OUTAGE_START = pd.Timestamp(
    "2022-07-26T00:00:00Z"
)

S5P_OUTAGE_END = pd.Timestamp(
    "2022-09-01T00:00:00Z"
)


def find_column(frame, candidates, name):
    for column in candidates:
        if column in frame.columns:
            return column

    raise KeyError(
        f"Cannot find {name}. Tried: "
        + ", ".join(candidates)
    )


def parse_label(value):
    if pd.isna(value):
        return np.nan

    try:
        number = float(value)

        if np.isclose(number, 1.0):
            return 1.0

        if np.isclose(number, 0.0):
            return 0.0

    except (TypeError, ValueError):
        pass

    return np.nan


def haversine_m(
    lat1,
    lon1,
    lat2,
    lon2,
):
    radius_m = 6371000.0

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    value = (
        sin(delta_lat / 2.0) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(delta_lon / 2.0) ** 2
    )

    return (
        2.0
        * radius_m
        * asin(sqrt(value))
    )


def to_utc_iso(timestamp):
    timestamp = pd.Timestamp(timestamp)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "UTC"
        )
    else:
        timestamp = timestamp.tz_convert(
            "UTC"
        )

    return timestamp.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def prepare_events():
    frame = pd.read_csv(
        EVENT_INPUT,
        low_memory=False,
    )

    event_id_column = find_column(
        frame,
        ["event_id"],
        "event ID",
    )

    time_column = find_column(
        frame,
        [
            "datetime_utc",
            "event_time_utc",
        ],
        "event datetime",
    )

    latitude_column = find_column(
        frame,
        ["lat", "latitude"],
        "latitude",
    )

    longitude_column = find_column(
        frame,
        ["lon", "longitude"],
        "longitude",
    )

    label_column = find_column(
        frame,
        ["true_release", "label"],
        "release label",
    )

    frame["_event_id"] = (
        frame[event_id_column]
        .astype(str)
        .str.strip()
    )

    frame["_event_time"] = pd.to_datetime(
        frame[time_column],
        errors="coerce",
        utc=True,
    )

    frame["_latitude"] = pd.to_numeric(
        frame[latitude_column],
        errors="coerce",
    )

    frame["_longitude"] = pd.to_numeric(
        frame[longitude_column],
        errors="coerce",
    )

    frame["_label"] = (
        frame[label_column]
        .map(parse_label)
    )

    return frame


def prepare_intervals():
    frame = pd.read_csv(
        INTERVAL_INPUT,
        low_memory=False,
    )

    frame["_start"] = pd.to_datetime(
        frame["release_start_utc"],
        errors="coerce",
        utc=True,
    )

    frame["_end"] = pd.to_datetime(
        frame["release_end_utc"],
        errors="coerce",
        utc=True,
    )

    frame["_latitude"] = pd.to_numeric(
        frame["lat"],
        errors="coerce",
    )

    frame["_longitude"] = pd.to_numeric(
        frame["lon"],
        errors="coerce",
    )

    if "release_rate_kg_h" in frame.columns:
        frame["_release_rate"] = pd.to_numeric(
            frame["release_rate_kg_h"],
            errors="coerce",
        )
    else:
        frame["_release_rate"] = np.nan

    frame = frame.dropna(
        subset=[
            "_start",
            "_end",
            "_latitude",
            "_longitude",
        ]
    ).copy()

    frame = frame[
        frame["_end"]
        >= frame["_start"]
    ].copy()

    return frame


def find_matching_intervals(
    event_time,
    latitude,
    longitude,
    intervals,
):
    if (
        pd.isna(event_time)
        or pd.isna(latitude)
        or pd.isna(longitude)
    ):
        return pd.DataFrame()

    candidates = intervals[
        intervals["_start"].le(
            event_time
        )
        & intervals["_end"].ge(
            event_time
        )
    ].copy()

    if candidates.empty:
        return candidates

    candidates["_distance_m"] = [
        haversine_m(
            latitude,
            longitude,
            row["_latitude"],
            row["_longitude"],
        )
        for _, row in candidates.iterrows()
    ]

    candidates = candidates[
        candidates["_distance_m"].le(
            INTERVAL_DISTANCE_LIMIT_M
        )
    ].copy()

    return candidates


def orbit_inside_any_interval(
    orbit_time,
    intervals,
):
    if (
        pd.isna(orbit_time)
        or intervals.empty
    ):
        return False

    return bool(
        (
            intervals["_start"].le(
                orbit_time
            )
            & intervals["_end"].ge(
                orbit_time
            )
        ).any()
    )


def classify_event_status(
    raw_orbit_count,
    valid_orbit_count,
    exact_candidate_count,
):
    if exact_candidate_count > 0:
        return (
            "valid_local_coverage_with_"
            "orbit_timestamp_in_release_interval"
        )

    if valid_orbit_count > 0:
        return (
            "valid_local_coverage_near_time_only"
        )

    if raw_orbit_count > 0:
        return (
            "orbit_asset_found_but_no_valid_"
            "local_ch4_retrieval"
        )

    return (
        "no_s5p_ch4_orbit_found_"
        "in_search_window"
    )


def query_event_orbits(
    collection,
    latitude,
    longitude,
    search_start,
    search_end,
):
    point = ee.Geometry.Point(
        [longitude, latitude]
    )

    region = point.buffer(
        LOCAL_RADIUS_M
    )

    collection_filtered = (
        collection
        .filterDate(
            to_utc_iso(search_start),
            to_utc_iso(search_end),
        )
        .filterBounds(region)
        .sort("system:time_start")
    )

    orbit_count = int(
        collection_filtered.size().getInfo()
    )

    if orbit_count == 0:
        return []

    reducer = (
        ee.Reducer.count()
        .combine(
            reducer2=ee.Reducer.mean(),
            sharedInputs=True,
        )
        .combine(
            reducer2=ee.Reducer.median(),
            sharedInputs=True,
        )
        .combine(
            reducer2=ee.Reducer.minMax(),
            sharedInputs=True,
        )
    )

    image_list = collection_filtered.toList(
        orbit_count
    )

    def image_to_feature(item):
        image = ee.Image(item)

        statistics = (
            image.select(
                [
                    CH4_BAND,
                    UNCERTAINTY_BAND,
                ]
            )
            .reduceRegion(
                reducer=reducer,
                geometry=region,
                scale=GRID_SCALE_M,
                bestEffort=True,
                maxPixels=1000000,
            )
        )

        properties = ee.Dictionary({
            "system_index":
                image.get("system:index"),

            "system_time_start":
                image.get(
                    "system:time_start"
                ),

            "orbit":
                image.get("ORBIT"),

            "product_id":
                image.get("PRODUCT_ID"),

            "processing_status":
                image.get(
                    "PROCESSING_STATUS"
                ),

            "product_quality":
                image.get(
                    "PRODUCT_QUALITY"
                ),

            "processor_version":
                image.get(
                    "PROCESSOR_VERSION"
                ),

            "algorithm_version":
                image.get(
                    "ALGORITHM_VERSION"
                ),

            "spatial_resolution":
                image.get(
                    "SPATIAL_RESOLUTION"
                ),
        })

        return ee.Feature(
            None,
            properties.combine(
                statistics,
                overwrite=True,
            ),
        )

    feature_collection = ee.FeatureCollection(
        image_list.map(
            image_to_feature
        )
    )

    result = feature_collection.getInfo()

    return [
        feature.get(
            "properties",
            {}
        )
        for feature in result.get(
            "features",
            []
        )
    ]


def main():
    if not EVENT_INPUT.exists():
        raise FileNotFoundError(
            EVENT_INPUT
        )

    if not INTERVAL_INPUT.exists():
        raise FileNotFoundError(
            INTERVAL_INPUT
        )

    project = os.environ.get(
        "EE_PROJECT",
        "methane-release-gee",
    )

    print("=" * 115)
    print(
        "SENTINEL-5P OFFL CH4 "
        "CONTROLLED-RELEASE AVAILABILITY AUDIT"
    )
    print("=" * 115)

    print("\nEarth Engine project:")
    print(project)

    ee.Initialize(
        project=project
    )

    collection = ee.ImageCollection(
        COLLECTION_ID
    )

    events = prepare_events()
    intervals = prepare_intervals()

    print("\nInput events:", len(events))
    print(
        "Exact interval evidence rows:",
        len(intervals),
    )

    event_records = []
    candidate_records = []

    ch4_count_key = (
        CH4_BAND + "_count"
    )

    ch4_mean_key = (
        CH4_BAND + "_mean"
    )

    ch4_median_key = (
        CH4_BAND + "_median"
    )

    ch4_min_key = (
        CH4_BAND + "_min"
    )

    ch4_max_key = (
        CH4_BAND + "_max"
    )

    uncertainty_mean_key = (
        UNCERTAINTY_BAND + "_mean"
    )

    for number, (_, event) in enumerate(
        events.iterrows(),
        start=1,
    ):
        event_id = event["_event_id"]
        event_time = event["_event_time"]
        latitude = event["_latitude"]
        longitude = event["_longitude"]
        label = event["_label"]

        print(
            f"\n[{number:03d}/{len(events):03d}] "
            f"{event_id}"
        )

        base = event.to_dict()

        base.update({
            "s5p_event_id":
                event_id,

            "s5p_event_time_utc":
                event_time,

            "s5p_latitude":
                latitude,

            "s5p_longitude":
                longitude,

            "s5p_true_release":
                label,

            "s5p_query_success":
                False,

            "s5p_query_error":
                "",
        })

        if (
            pd.isna(event_time)
            or pd.isna(latitude)
            or pd.isna(longitude)
        ):
            base.update({
                "s5p_query_error":
                    "missing_event_time_or_location",

                "exact_interval_match_count":
                    0,

                "raw_orbit_candidate_count":
                    0,

                "valid_local_coverage_orbit_count":
                    0,

                "orbit_timestamp_exact_candidate_count":
                    0,

                "s5p_ch4_status":
                    "invalid_event_metadata",
            })

            event_records.append(base)
            continue

        matched_intervals = (
            find_matching_intervals(
                event_time,
                latitude,
                longitude,
                intervals,
            )
        )

        interval_count = len(
            matched_intervals
        )

        if interval_count > 0:
            release_start = (
                matched_intervals["_start"]
                .min()
            )

            release_end = (
                matched_intervals["_end"]
                .max()
            )

            release_rate_median = (
                matched_intervals[
                    "_release_rate"
                ].median()
            )

            search_start = (
                release_start
                - pd.Timedelta(
                    hours=NEAR_TIME_HOURS
                )
            )

            search_end = (
                release_end
                + pd.Timedelta(
                    hours=NEAR_TIME_HOURS
                )
            )

        else:
            release_start = pd.NaT
            release_end = pd.NaT
            release_rate_median = np.nan

            search_start = (
                event_time
                - pd.Timedelta(
                    hours=NEAR_TIME_HOURS
                )
            )

            search_end = (
                event_time
                + pd.Timedelta(
                    hours=NEAR_TIME_HOURS
                )
            )

        in_official_outage = bool(
            event_time >= S5P_OUTAGE_START
            and event_time < S5P_OUTAGE_END
        )

        base.update({
            "exact_interval_match_count":
                interval_count,

            "exact_release_start_utc":
                release_start,

            "exact_release_end_utc":
                release_end,

            "exact_release_rate_median_kg_h":
                release_rate_median,

            "s5p_search_start_utc":
                search_start,

            "s5p_search_end_utc":
                search_end,

            "event_in_s5p_2022_outage":
                in_official_outage,
        })

        try:
            orbit_properties = query_event_orbits(
                collection=collection,
                latitude=latitude,
                longitude=longitude,
                search_start=search_start,
                search_end=search_end,
            )

            raw_orbit_count = len(
                orbit_properties
            )

            valid_orbit_count = 0
            exact_candidate_count = 0

            nearest_time_difference = np.nan
            nearest_valid_orbit_time = pd.NaT

            for properties in orbit_properties:
                orbit_time = pd.to_datetime(
                    properties.get(
                        "system_time_start"
                    ),
                    unit="ms",
                    errors="coerce",
                    utc=True,
                )

                local_cell_count = int(
                    properties.get(
                        ch4_count_key
                    )
                    or 0
                )

                valid_local_coverage = (
                    local_cell_count > 0
                )

                if valid_local_coverage:
                    valid_orbit_count += 1

                timestamp_inside_interval = (
                    valid_local_coverage
                    and orbit_inside_any_interval(
                        orbit_time,
                        matched_intervals,
                    )
                )

                if timestamp_inside_interval:
                    exact_candidate_count += 1

                time_difference_hours = (
                    abs(
                        (
                            orbit_time
                            - event_time
                        ).total_seconds()
                    )
                    / 3600.0
                    if pd.notna(orbit_time)
                    else np.nan
                )

                if (
                    valid_local_coverage
                    and pd.notna(
                        time_difference_hours
                    )
                    and (
                        pd.isna(
                            nearest_time_difference
                        )
                        or time_difference_hours
                        < nearest_time_difference
                    )
                ):
                    nearest_time_difference = (
                        time_difference_hours
                    )

                    nearest_valid_orbit_time = (
                        orbit_time
                    )

                candidate_records.append({
                    "event_id":
                        event_id,

                    "event_time_utc":
                        event_time,

                    "true_release":
                        label,

                    "latitude":
                        latitude,

                    "longitude":
                        longitude,

                    "exact_interval_match_count":
                        interval_count,

                    "exact_release_start_utc":
                        release_start,

                    "exact_release_end_utc":
                        release_end,

                    "s5p_system_index":
                        properties.get(
                            "system_index"
                        ),

                    "s5p_orbit":
                        properties.get(
                            "orbit"
                        ),

                    "s5p_product_id":
                        properties.get(
                            "product_id"
                        ),

                    "s5p_orbit_time_utc":
                        orbit_time,

                    "time_difference_hours":
                        time_difference_hours,

                    "orbit_timestamp_inside_"
                    "release_interval":
                        timestamp_inside_interval,

                    "valid_local_ch4_coverage":
                        valid_local_coverage,

                    "local_gridded_cell_count":
                        local_cell_count,

                    "local_ch4_mean_ppb":
                        properties.get(
                            ch4_mean_key
                        ),

                    "local_ch4_median_ppb":
                        properties.get(
                            ch4_median_key
                        ),

                    "local_ch4_min_ppb":
                        properties.get(
                            ch4_min_key
                        ),

                    "local_ch4_max_ppb":
                        properties.get(
                            ch4_max_key
                        ),

                    "local_uncertainty_mean_ppb":
                        properties.get(
                            uncertainty_mean_key
                        ),

                    "processing_status":
                        properties.get(
                            "processing_status"
                        ),

                    "product_quality":
                        properties.get(
                            "product_quality"
                        ),

                    "processor_version":
                        properties.get(
                            "processor_version"
                        ),

                    "algorithm_version":
                        properties.get(
                            "algorithm_version"
                        ),

                    "spatial_resolution":
                        properties.get(
                            "spatial_resolution"
                        ),
                })

            status = classify_event_status(
                raw_orbit_count,
                valid_orbit_count,
                exact_candidate_count,
            )

            base.update({
                "s5p_query_success":
                    True,

                "raw_orbit_candidate_count":
                    raw_orbit_count,

                "valid_local_coverage_orbit_count":
                    valid_orbit_count,

                "orbit_timestamp_exact_candidate_count":
                    exact_candidate_count,

                "nearest_valid_orbit_time_utc":
                    nearest_valid_orbit_time,

                "nearest_valid_orbit_time_"
                "difference_hours":
                    nearest_time_difference,

                "s5p_ch4_status":
                    status,
            })

            print(
                "  raw orbit assets:",
                raw_orbit_count,
            )

            print(
                "  valid local coverage:",
                valid_orbit_count,
            )

            print(
                "  orbit-time exact candidates:",
                exact_candidate_count,
            )

        except Exception as error:
            print(
                "  ERROR:",
                error,
            )

            base.update({
                "s5p_query_error":
                    str(error),

                "raw_orbit_candidate_count":
                    0,

                "valid_local_coverage_orbit_count":
                    0,

                "orbit_timestamp_exact_candidate_count":
                    0,

                "s5p_ch4_status":
                    "s5p_earth_engine_query_error",
            })

        event_records.append(base)

        pd.DataFrame(
            event_records
        ).to_csv(
            EVENT_OUTPUT,
            index=False,
        )

        pd.DataFrame(
            candidate_records
        ).to_csv(
            CANDIDATE_OUTPUT,
            index=False,
        )

        time.sleep(0.15)

    event_result = pd.DataFrame(
        event_records
    )

    candidate_result = pd.DataFrame(
        candidate_records
    )

    event_result.to_csv(
        EVENT_OUTPUT,
        index=False,
    )

    candidate_result.to_csv(
        CANDIDATE_OUTPUT,
        index=False,
    )

    successful_queries = int(
        event_result[
            "s5p_query_success"
        ].fillna(False).sum()
    )

    events_with_orbit_assets = int(
        event_result[
            "raw_orbit_candidate_count"
        ].gt(0).sum()
    )

    events_with_valid_coverage = int(
        event_result[
            "valid_local_coverage_orbit_count"
        ].gt(0).sum()
    )

    exact_orbit_candidates = int(
        event_result[
            "orbit_timestamp_exact_candidate_count"
        ].gt(0).sum()
    )

    outage_events = int(
        event_result[
            "event_in_s5p_2022_outage"
        ].fillna(False).sum()
    )

    status_summary = (
        event_result[
            "s5p_ch4_status"
        ]
        .value_counts(
            dropna=False
        )
    )

    label_coverage_summary = (
        event_result.groupby(
            [
                "s5p_true_release",
                "s5p_ch4_status",
            ],
            dropna=False,
        )
        .size()
    )

    report_lines = [
        "=" * 115,
        "SENTINEL-5P OFFL CH4 AVAILABILITY AUDIT V1",
        "=" * 115,
        "",
        f"Input controlled-release events: {len(event_result)}",
        (
            "Successful Earth Engine event queries: "
            f"{successful_queries}"
        ),
        (
            "Events with orbit assets in +/-24 h window: "
            f"{events_with_orbit_assets}"
        ),
        (
            "Events with valid local XCH4 coverage: "
            f"{events_with_valid_coverage}"
        ),
        (
            "Events with valid coverage and orbit timestamp "
            "inside release interval: "
            f"{exact_orbit_candidates}"
        ),
        (
            "Events during official 2022 S5P CH4 outage: "
            f"{outage_events}"
        ),
        (
            "Total candidate orbit rows: "
            f"{len(candidate_result)}"
        ),
        "",
        "S5P event status:",
        status_summary.to_string(),
        "",
        "Status by original release label:",
        label_coverage_summary.to_string(),
        "",
        "Important:",
        (
            "The Earth Engine system:time_start value is an "
            "orbit/product timestamp, not guaranteed to be the "
            "exact pixel acquisition time at the source."
        ),
        (
            "Orbit timestamps inside short release intervals are "
            "therefore screening candidates, not confirmed exact "
            "temporal overlaps."
        ),
        (
            "The local cell count refers to the approximately "
            "1.1 km Earth Engine grid. It does not represent "
            "independent native TROPOMI soundings."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 115)
    print("SENTINEL-5P CH4 AVAILABILITY SUMMARY")
    print("=" * 115)

    print(
        "\nInput controlled-release events:",
        len(event_result),
    )

    print(
        "Successful Earth Engine event queries:",
        successful_queries,
    )

    print(
        "Events with orbit assets in +/-24 h window:",
        events_with_orbit_assets,
    )

    print(
        "Events with valid local XCH4 coverage:",
        events_with_valid_coverage,
    )

    print(
        "Events with valid coverage and orbit timestamp "
        "inside release interval:",
        exact_orbit_candidates,
    )

    print(
        "Events during official 2022 S5P CH4 outage:",
        outage_events,
    )

    print(
        "Total candidate orbit rows:",
        len(candidate_result),
    )

    print("\nS5P event status:")
    print(status_summary)

    print("\nStatus by original release label:")
    print(label_coverage_summary)

    print("\nSaved:")
    print(CANDIDATE_OUTPUT)
    print(EVENT_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
