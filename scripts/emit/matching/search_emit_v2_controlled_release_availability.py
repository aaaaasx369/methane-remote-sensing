from pathlib import Path
from math import radians, sin, cos, asin, sqrt
import time

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


EVENT_INPUT = Path(
    "outputs/10_final_events_for_gee.csv"
)

INTERVAL_INPUT = Path(
    "outputs/309_all_exact_release_intervals_for_s2.csv"
)

COLLECTION_OUTPUT = Path(
    "outputs/478_emit_v2_collection_validation_v1.csv"
)

GRANULE_OUTPUT = Path(
    "outputs/479_emit_v2_event_granule_matches_v1.csv"
)

EVENT_OUTPUT = Path(
    "outputs/480_emit_v2_event_availability_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/481_emit_v2_event_availability_report_v1.txt"
)


CMR_COLLECTION_URL = (
    "https://cmr.earthdata.nasa.gov/"
    "search/collections.json"
)

CMR_GRANULE_URL = (
    "https://cmr.earthdata.nasa.gov/"
    "search/granules.json"
)

PRODUCTS = {
    "enhancement": {
        "short_name": "EMITL2BCH4ENH",
        "version": "002",
        "spatial_mode": "point",
    },
    "plume": {
        "short_name": "EMITL2BCH4PLM",
        "version": "002",
        "spatial_mode": "bbox",
    },
}

MAX_SOURCE_DISTANCE_M = 1000.0
PLUME_SEARCH_RADIUS_KM = 5.0
NEAR_TIME_HOURS = 24
PAGE_SIZE = 2000
REQUEST_DELAY_SECONDS = 0.25


def create_session():
    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(
        max_retries=retry
    )

    session.mount(
        "https://",
        adapter,
    )

    session.headers.update({
        "User-Agent":
            "methane-release-project/"
            "emit-v2-availability-audit",
        "Accept":
            "application/json",
    })

    return session


def find_column(frame, candidates, description):
    for column in candidates:
        if column in frame.columns:
            return column

    raise KeyError(
        f"Cannot find {description}. Tried: "
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

    text = str(value).strip().lower()

    if text in {
        "true",
        "positive",
        "release",
        "released",
        "yes",
    }:
        return 1.0

    if text in {
        "false",
        "negative",
        "no_release",
        "no release",
        "no",
    }:
        return 0.0

    return np.nan


def haversine_m(lat1, lon1, lat2, lon2):
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


def isoformat_utc(value):
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "UTC"
        )
    else:
        timestamp = timestamp.tz_convert(
            "UTC"
        )

    return (
        timestamp.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )


def make_bbox(
    latitude,
    longitude,
    radius_km,
):
    latitude_delta = (
        radius_km / 111.32
    )

    longitude_scale = max(
        np.cos(
            np.radians(latitude)
        ),
        0.01,
    )

    longitude_delta = (
        radius_km
        / (
            111.32
            * longitude_scale
        )
    )

    west = longitude - longitude_delta
    east = longitude + longitude_delta
    south = latitude - latitude_delta
    north = latitude + latitude_delta

    return (
        west,
        south,
        east,
        north,
    )


def validate_collections(session):
    records = []

    for product_type, config in PRODUCTS.items():
        parameters = {
            "short_name":
                config["short_name"],

            "version":
                config["version"],

            "page_size":
                20,
        }

        response = session.get(
            CMR_COLLECTION_URL,
            params=parameters,
            timeout=60,
        )

        response.raise_for_status()

        entries = (
            response.json()
            .get("feed", {})
            .get("entry", [])
        )

        records.append({
            "product_type":
                product_type,

            "short_name":
                config["short_name"],

            "version":
                config["version"],

            "collection_match_count":
                len(entries),

            "collection_found":
                len(entries) > 0,

            "collection_ids":
                " || ".join(
                    str(entry.get("id", ""))
                    for entry in entries
                ),

            "collection_titles":
                " || ".join(
                    str(entry.get("title", ""))
                    for entry in entries
                ),
        })

    validation = pd.DataFrame(records)

    validation.to_csv(
        COLLECTION_OUTPUT,
        index=False,
    )

    missing = validation[
        ~validation["collection_found"]
    ]

    if not missing.empty:
        raise RuntimeError(
            "EMIT V2 collection validation failed:\n"
            + missing.to_string(index=False)
        )

    return validation


def query_cmr_granules(
    session,
    short_name,
    version,
    start_time,
    end_time,
    latitude,
    longitude,
    spatial_mode,
):
    parameters = {
        "short_name":
            short_name,

        "version":
            version,

        "temporal":
            (
                f"{isoformat_utc(start_time)},"
                f"{isoformat_utc(end_time)}"
            ),

        "page_size":
            PAGE_SIZE,

        "page_num":
            1,
    }

    if spatial_mode == "point":
        parameters["point"] = (
            f"{longitude},{latitude}"
        )

    elif spatial_mode == "bbox":
        west, south, east, north = (
            make_bbox(
                latitude,
                longitude,
                PLUME_SEARCH_RADIUS_KM,
            )
        )

        parameters["bounding_box"] = (
            f"{west},{south},"
            f"{east},{north}"
        )

    else:
        raise ValueError(
            f"Unknown spatial mode: "
            f"{spatial_mode}"
        )

    all_entries = []

    while True:
        response = session.get(
            CMR_GRANULE_URL,
            params=parameters,
            timeout=90,
        )

        response.raise_for_status()

        entries = (
            response.json()
            .get("feed", {})
            .get("entry", [])
        )

        if not entries:
            break

        all_entries.extend(entries)

        if len(entries) < PAGE_SIZE:
            break

        parameters["page_num"] += 1

    return all_entries


def extract_links(entry):
    links = entry.get(
        "links",
        [],
    )

    data_links = []
    metadata_links = []

    for link in links:
        href = str(
            link.get("href", "")
        )

        relation = str(
            link.get("rel", "")
        ).lower()

        title = str(
            link.get("title", "")
        ).lower()

        if not href:
            continue

        if (
            "browse" in relation
            or "browse" in title
        ):
            continue

        if (
            "data#" in relation
            or href.lower().endswith(
                (
                    ".nc",
                    ".h5",
                    ".hdf",
                    ".zip",
                )
            )
        ):
            data_links.append(href)

        else:
            metadata_links.append(href)

    return (
        data_links,
        metadata_links,
    )


def granule_overlaps_interval(
    granule_start,
    granule_end,
    interval_start,
    interval_end,
):
    if (
        pd.isna(granule_start)
        or pd.isna(interval_start)
        or pd.isna(interval_end)
    ):
        return False

    if pd.isna(granule_end):
        granule_end = granule_start

    return bool(
        granule_start <= interval_end
        and granule_end >= interval_start
    )


def prepare_events():
    events = pd.read_csv(
        EVENT_INPUT,
        low_memory=False,
    )

    event_id_column = find_column(
        events,
        ["event_id"],
        "event ID",
    )

    event_time_column = find_column(
        events,
        [
            "datetime_utc",
            "event_time_utc",
        ],
        "event time",
    )

    latitude_column = find_column(
        events,
        ["lat", "latitude"],
        "latitude",
    )

    longitude_column = find_column(
        events,
        ["lon", "longitude"],
        "longitude",
    )

    label_column = find_column(
        events,
        ["true_release", "label"],
        "label",
    )

    events["_event_id"] = (
        events[event_id_column]
        .astype(str)
        .str.strip()
    )

    events["_event_time"] = (
        pd.to_datetime(
            events[event_time_column],
            errors="coerce",
            utc=True,
        )
    )

    events["_latitude"] = (
        pd.to_numeric(
            events[latitude_column],
            errors="coerce",
        )
    )

    events["_longitude"] = (
        pd.to_numeric(
            events[longitude_column],
            errors="coerce",
        )
    )

    events["_label"] = (
        events[label_column]
        .map(parse_label)
    )

    return events


def prepare_intervals():
    intervals = pd.read_csv(
        INTERVAL_INPUT,
        low_memory=False,
    )

    intervals["_start"] = (
        pd.to_datetime(
            intervals[
                "release_start_utc"
            ],
            errors="coerce",
            utc=True,
        )
    )

    intervals["_end"] = (
        pd.to_datetime(
            intervals[
                "release_end_utc"
            ],
            errors="coerce",
            utc=True,
        )
    )

    intervals["_latitude"] = (
        pd.to_numeric(
            intervals["lat"],
            errors="coerce",
        )
    )

    intervals["_longitude"] = (
        pd.to_numeric(
            intervals["lon"],
            errors="coerce",
        )
    )

    if "release_rate_kg_h" in intervals.columns:
        intervals["_rate"] = (
            pd.to_numeric(
                intervals[
                    "release_rate_kg_h"
                ],
                errors="coerce",
            )
        )
    else:
        intervals["_rate"] = np.nan

    intervals = intervals.dropna(
        subset=[
            "_start",
            "_end",
            "_latitude",
            "_longitude",
        ]
    ).copy()

    intervals = intervals[
        intervals["_end"]
        >= intervals["_start"]
    ].copy()

    return intervals


def find_exact_release_evidence(
    event,
    intervals,
):
    event_time = event["_event_time"]
    latitude = event["_latitude"]
    longitude = event["_longitude"]

    if (
        pd.isna(event_time)
        or pd.isna(latitude)
        or pd.isna(longitude)
    ):
        return pd.DataFrame()

    time_candidates = intervals[
        (
            intervals["_start"]
            <= event_time
        )
        & (
            intervals["_end"]
            >= event_time
        )
    ].copy()

    if time_candidates.empty:
        return time_candidates

    distances = []

    for _, interval in (
        time_candidates.iterrows()
    ):
        distances.append(
            haversine_m(
                latitude,
                longitude,
                interval["_latitude"],
                interval["_longitude"],
            )
        )

    time_candidates[
        "_distance_m"
    ] = distances

    return time_candidates[
        time_candidates["_distance_m"]
        <= MAX_SOURCE_DISTANCE_M
    ].copy()


def classify_event_status(row):
    enhancement_exact = int(
        row[
            "enhancement_exact_granule_count"
        ]
    )

    plume_exact = int(
        row[
            "plume_exact_granule_count"
        ]
    )

    enhancement_near = int(
        row[
            "enhancement_near_granule_count"
        ]
    )

    plume_near = int(
        row[
            "plume_near_granule_count"
        ]
    )

    if enhancement_exact > 0:
        if plume_exact > 0:
            return (
                "exact_emit_coverage_"
                "with_nearby_plume_product"
            )

        return (
            "exact_emit_coverage_"
            "without_nearby_plume_product"
        )

    if plume_exact > 0:
        return (
            "nearby_plume_product_"
            "but_source_point_not_in_"
            "enhancement_footprint"
        )

    if enhancement_near > 0:
        if plume_near > 0:
            return (
                "near_time_emit_coverage_"
                "and_nearby_plume_product"
            )

        return (
            "near_time_emit_coverage_"
            "without_nearby_plume_product"
        )

    if plume_near > 0:
        return (
            "near_time_nearby_plume_only"
        )

    return (
        "no_emit_v2_coverage_found_"
        "in_search_window"
    )


def main():
    if not EVENT_INPUT.exists():
        raise FileNotFoundError(
            EVENT_INPUT
        )

    if not INTERVAL_INPUT.exists():
        raise FileNotFoundError(
            INTERVAL_INPUT
        )

    session = create_session()

    print("=" * 115)
    print(
        "EMIT V2 CONTROLLED-RELEASE "
        "AVAILABILITY AUDIT"
    )
    print("=" * 115)

    validation = validate_collections(
        session
    )

    print("\nCollections validated:")
    print(
        validation[
            [
                "product_type",
                "short_name",
                "version",
                "collection_match_count",
            ]
        ].to_string(index=False)
    )

    events = prepare_events()
    intervals = prepare_intervals()

    print("\nInput event rows:", len(events))
    print(
        "Exact interval evidence rows:",
        len(intervals),
    )

    event_records = []
    granule_records = []

    for event_number, (_, event) in enumerate(
        events.iterrows(),
        start=1,
    ):
        event_id = event["_event_id"]
        event_time = event["_event_time"]
        latitude = event["_latitude"]
        longitude = event["_longitude"]
        label = event["_label"]

        print(
            f"\n[{event_number:03d}/"
            f"{len(events):03d}] "
            f"{event_id}"
        )

        base = event.to_dict()

        base.update({
            "event_id_emit":
                event_id,

            "event_time_utc_emit":
                event_time,

            "latitude_emit":
                latitude,

            "longitude_emit":
                longitude,

            "true_release_emit":
                label,

            "emit_query_success":
                False,

            "emit_query_error":
                "",
        })

        if (
            pd.isna(event_time)
            or pd.isna(latitude)
            or pd.isna(longitude)
        ):
            base.update({
                "emit_query_error":
                    "missing_event_time_or_location",

                "exact_interval_match_count":
                    0,

                "enhancement_exact_granule_count":
                    0,

                "enhancement_near_granule_count":
                    0,

                "plume_exact_granule_count":
                    0,

                "plume_near_granule_count":
                    0,

                "emit_v2_status":
                    "invalid_event_metadata",
            })

            event_records.append(base)
            continue

        exact_intervals = (
            find_exact_release_evidence(
                event,
                intervals,
            )
        )

        exact_interval_count = len(
            exact_intervals
        )

        if exact_interval_count > 0:
            exact_start = (
                exact_intervals["_start"]
                .max()
            )

            exact_end = (
                exact_intervals["_end"]
                .min()
            )

            if exact_start > exact_end:
                selected = (
                    exact_intervals
                    .sort_values(
                        [
                            "_distance_m",
                            "_start",
                        ]
                    )
                    .iloc[0]
                )

                exact_start = selected[
                    "_start"
                ]

                exact_end = selected[
                    "_end"
                ]

            exact_rate_median = (
                exact_intervals["_rate"]
                .median()
            )

            interval_status = (
                "exact_release_interval_found"
            )

            search_start = (
                exact_start
                - pd.Timedelta(
                    hours=NEAR_TIME_HOURS
                )
            )

            search_end = (
                exact_end
                + pd.Timedelta(
                    hours=NEAR_TIME_HOURS
                )
            )

        else:
            exact_start = pd.NaT
            exact_end = pd.NaT
            exact_rate_median = np.nan

            if label == 1.0:
                interval_status = (
                    "positive_without_exact_"
                    "interval_match"
                )

            elif label == 0.0:
                interval_status = (
                    "negative_event_no_exact_"
                    "release_interval"
                )

            else:
                interval_status = (
                    "unlabeled_event_no_exact_"
                    "release_interval"
                )

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

        base.update({
            "exact_interval_match_count":
                exact_interval_count,

            "release_interval_status":
                interval_status,

            "exact_release_start_utc":
                exact_start,

            "exact_release_end_utc":
                exact_end,

            "exact_release_rate_median_kg_h":
                exact_rate_median,

            "emit_search_start_utc":
                search_start,

            "emit_search_end_utc":
                search_end,
        })

        counts = {
            "enhancement_exact":
                0,

            "enhancement_near":
                0,

            "plume_exact":
                0,

            "plume_near":
                0,
        }

        try:
            for product_type, config in (
                PRODUCTS.items()
            ):
                entries = query_cmr_granules(
                    session=session,
                    short_name=(
                        config["short_name"]
                    ),
                    version=(
                        config["version"]
                    ),
                    start_time=search_start,
                    end_time=search_end,
                    latitude=latitude,
                    longitude=longitude,
                    spatial_mode=(
                        config["spatial_mode"]
                    ),
                )

                print(
                    f"  {product_type}: "
                    f"{len(entries)} "
                    f"granule(s)"
                )

                seen_ids = set()

                for entry in entries:
                    granule_id = str(
                        entry.get(
                            "id",
                            "",
                        )
                    )

                    if (
                        granule_id
                        and granule_id
                        in seen_ids
                    ):
                        continue

                    seen_ids.add(
                        granule_id
                    )

                    granule_start = (
                        pd.to_datetime(
                            entry.get(
                                "time_start"
                            ),
                            errors="coerce",
                            utc=True,
                        )
                    )

                    granule_end = (
                        pd.to_datetime(
                            entry.get(
                                "time_end"
                            ),
                            errors="coerce",
                            utc=True,
                        )
                    )

                    exact_overlap = (
                        exact_interval_count > 0
                        and
                        granule_overlaps_interval(
                            granule_start,
                            granule_end,
                            exact_start,
                            exact_end,
                        )
                    )

                    if exact_overlap:
                        temporal_class = (
                            "exact_release_overlap"
                        )

                        counts[
                            f"{product_type}_exact"
                        ] += 1

                    else:
                        temporal_class = (
                            "near_time_not_exact"
                        )

                        counts[
                            f"{product_type}_near"
                        ] += 1

                    data_links, metadata_links = (
                        extract_links(entry)
                    )

                    granule_records.append({
                        "event_id":
                            event_id,

                        "true_release":
                            label,

                        "event_time_utc":
                            event_time,

                        "latitude":
                            latitude,

                        "longitude":
                            longitude,

                        "product_type":
                            product_type,

                        "short_name":
                            config[
                                "short_name"
                            ],

                        "version":
                            config[
                                "version"
                            ],

                        "granule_id":
                            granule_id,

                        "granule_title":
                            entry.get(
                                "title",
                                "",
                            ),

                        "granule_producer_id":
                            entry.get(
                                "producer_granule_id",
                                "",
                            ),

                        "granule_time_start":
                            granule_start,

                        "granule_time_end":
                            granule_end,

                        "temporal_match_class":
                            temporal_class,

                        "exact_release_overlap":
                            exact_overlap,

                        "exact_release_start_utc":
                            exact_start,

                        "exact_release_end_utc":
                            exact_end,

                        "spatial_query_mode":
                            config[
                                "spatial_mode"
                            ],

                        "plume_search_radius_km":
                            (
                                PLUME_SEARCH_RADIUS_KM
                                if product_type
                                == "plume"
                                else np.nan
                            ),

                        "data_links":
                            " || ".join(
                                data_links
                            ),

                        "metadata_links":
                            " || ".join(
                                metadata_links
                            ),

                        "polygon_metadata":
                            str(
                                entry.get(
                                    "polygons",
                                    "",
                                )
                            ),

                        "box_metadata":
                            str(
                                entry.get(
                                    "boxes",
                                    "",
                                )
                            ),
                    })

            base.update({
                "emit_query_success":
                    True,

                "enhancement_exact_granule_count":
                    counts[
                        "enhancement_exact"
                    ],

                "enhancement_near_granule_count":
                    counts[
                        "enhancement_near"
                    ],

                "plume_exact_granule_count":
                    counts[
                        "plume_exact"
                    ],

                "plume_near_granule_count":
                    counts[
                        "plume_near"
                    ],
            })

            base[
                "emit_v2_status"
            ] = classify_event_status(
                base
            )

        except Exception as error:
            print(
                "  ERROR:",
                error,
            )

            base.update({
                "emit_query_error":
                    str(error),

                "enhancement_exact_granule_count":
                    0,

                "enhancement_near_granule_count":
                    0,

                "plume_exact_granule_count":
                    0,

                "plume_near_granule_count":
                    0,

                "emit_v2_status":
                    "emit_cmr_query_error",
            })

        event_records.append(base)

        pd.DataFrame(
            event_records
        ).to_csv(
            EVENT_OUTPUT,
            index=False,
        )

        pd.DataFrame(
            granule_records
        ).to_csv(
            GRANULE_OUTPUT,
            index=False,
        )

        time.sleep(
            REQUEST_DELAY_SECONDS
        )

    event_result = pd.DataFrame(
        event_records
    )

    granule_result = pd.DataFrame(
        granule_records
    )

    event_result.to_csv(
        EVENT_OUTPUT,
        index=False,
    )

    granule_result.to_csv(
        GRANULE_OUTPUT,
        index=False,
    )

    status_summary = (
        event_result[
            "emit_v2_status"
        ]
        .value_counts(
            dropna=False
        )
    )

    successful_queries = int(
        event_result[
            "emit_query_success"
        ].fillna(False).sum()
    )

    exact_enhancement_events = int(
        event_result[
            "enhancement_exact_granule_count"
        ].gt(0).sum()
    )

    exact_plume_events = int(
        event_result[
            "plume_exact_granule_count"
        ].gt(0).sum()
    )

    near_enhancement_events = int(
        event_result[
            "enhancement_near_granule_count"
        ].gt(0).sum()
    )

    near_plume_events = int(
        event_result[
            "plume_near_granule_count"
        ].gt(0).sum()
    )

    labeled = event_result[
        event_result[
            "true_release_emit"
        ].isin([0.0, 1.0])
    ].copy()

    report_lines = [
        "=" * 115,
        "EMIT V2 CONTROLLED-RELEASE AVAILABILITY AUDIT V1",
        "=" * 115,
        "",
        f"Input deduplicated events: {len(event_result)}",
        (
            "Successful CMR event queries: "
            f"{successful_queries}"
        ),
        (
            "Events with exact EMIT enhancement coverage: "
            f"{exact_enhancement_events}"
        ),
        (
            "Events with exact nearby plume product: "
            f"{exact_plume_events}"
        ),
        (
            "Events with near-time enhancement coverage: "
            f"{near_enhancement_events}"
        ),
        (
            "Events with near-time nearby plume product: "
            f"{near_plume_events}"
        ),
        (
            "Total matched granule rows: "
            f"{len(granule_result)}"
        ),
        "",
        "EMIT V2 status:",
        status_summary.to_string(),
        "",
        "Labeled-event summary:",
        (
            labeled.groupby(
                [
                    "true_release_emit",
                    "emit_v2_status",
                ],
                dropna=False,
            )
            .size()
            .to_string()
            if not labeled.empty
            else "No labeled events."
        ),
        "",
        "Interpretation:",
        (
            "Enhancement coverage without a plume product "
            "means EMIT covered the source, but no nearby "
            "plume complex product was identified. It is not "
            "automatically a true methane-negative result."
        ),
        (
            "The plume search uses a 5 km box around the "
            "source because the plume may be transported "
            "downwind from the release point."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 115)
    print("EMIT V2 AVAILABILITY SUMMARY")
    print("=" * 115)

    print(
        "\nInput deduplicated events:",
        len(event_result),
    )

    print(
        "Successful CMR event queries:",
        successful_queries,
    )

    print(
        "Events with exact EMIT enhancement coverage:",
        exact_enhancement_events,
    )

    print(
        "Events with exact nearby plume product:",
        exact_plume_events,
    )

    print(
        "Events with near-time enhancement coverage:",
        near_enhancement_events,
    )

    print(
        "Events with near-time nearby plume product:",
        near_plume_events,
    )

    print(
        "Total matched granule rows:",
        len(granule_result),
    )

    print("\nEMIT V2 status:")
    print(status_summary)

    print("\nSaved:")
    print(COLLECTION_OUTPUT)
    print(GRANULE_OUTPUT)
    print(EVENT_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
