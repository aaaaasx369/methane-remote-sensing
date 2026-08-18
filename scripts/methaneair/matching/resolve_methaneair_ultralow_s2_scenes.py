from pathlib import Path
import time

import ee
import numpy as np
import pandas as pd


PROJECT = "methane-release-gee"
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

INPUT = Path(
    "outputs/443_methaneair_s2_low_emission_candidate_audit_v1.csv"
)

EVENT_OUTPUT = Path(
    "outputs/445_methaneair_s2_below500_event_shortlist_v1.csv"
)

CANDIDATE_OUTPUT = Path(
    "outputs/446_methaneair_s2_below500_scene_candidates_v1.csv"
)

BEST_OUTPUT = Path(
    "outputs/447_methaneair_s2_below500_best_scene_per_event_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/448_methaneair_s2_below500_scene_resolution_report_v1.txt"
)

MAX_EMISSION_KG_H = 500.0
SEARCH_WINDOW_DAYS = 1


def initialize_earth_engine():
    try:
        ee.Initialize(project=PROJECT)
        print("Earth Engine initialized.")
    except Exception:
        print("Earth Engine authentication required.")
        ee.Authenticate()
        ee.Initialize(project=PROJECT)
        print("Earth Engine initialized after authentication.")


def query_event_scenes(row, max_attempts=3):
    event_id = str(row["event_id"])
    latitude = float(row["lat"])
    longitude = float(row["lon"])
    event_time = row["_event_time"]

    point = ee.Geometry.Point(
        [longitude, latitude]
    )

    ee_event_time = ee.Date(
        event_time.isoformat()
    )

    start = ee_event_time.advance(
        -SEARCH_WINDOW_DAYS,
        "day",
    )

    end = ee_event_time.advance(
        SEARCH_WINDOW_DAYS,
        "day",
    )

    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(point)
        .filterDate(start, end)
        .sort("system:time_start")
    )

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            collection_info = collection.getInfo()
            image_items = collection_info.get(
                "features",
                [],
            )
            break

        except Exception as error:
            last_error = error

            if attempt == max_attempts:
                raise

            print(
                f"  Retry {attempt}/{max_attempts}:",
                error,
            )

            time.sleep(3 * attempt)

    records = []

    for item in image_items:
        properties = item.get(
            "properties",
            {},
        )

        acquisition_ms = properties.get(
            "system:time_start"
        )

        if acquisition_ms is None:
            continue

        acquisition_time = pd.to_datetime(
            acquisition_ms,
            unit="ms",
            utc=True,
        )

        difference_hours = (
            acquisition_time - event_time
        ).total_seconds() / 3600.0

        system_index = properties.get(
            "system:index",
            ""
        )

        full_scene_id = item.get(
            "id",
            ""
        )

        if not full_scene_id and system_index:
            full_scene_id = (
                f"{S2_COLLECTION}/{system_index}"
            )

        cloud = pd.to_numeric(
            properties.get(
                "CLOUDY_PIXEL_PERCENTAGE",
                np.nan,
            ),
            errors="coerce",
        )

        records.append({
            "event_id":
                event_id,

            "event_time_utc":
                event_time.isoformat(),

            "emission_kg_hr":
                float(row["emission_kg_hr"]),

            "emission_tph":
                float(row["emission_tph"]),

            "latitude":
                latitude,

            "longitude":
                longitude,

            "flight_id":
                row.get("flight_id", ""),

            "plume_id":
                row.get("plume_id", ""),

            "basin":
                row.get("Basin", ""),

            "scene_id":
                full_scene_id,

            "system_index":
                system_index,

            "product_id":
                properties.get(
                    "PRODUCT_ID",
                    "",
                ),

            "mgrs_tile":
                properties.get(
                    "MGRS_TILE",
                    "",
                ),

            "spacecraft_name":
                properties.get(
                    "SPACECRAFT_NAME",
                    "",
                ),

            "acquisition_time_utc":
                acquisition_time.isoformat(),

            "time_difference_hours":
                float(difference_hours),

            "absolute_time_difference_hours":
                float(abs(difference_hours)),

            "same_utc_date":
                bool(
                    acquisition_time.date()
                    == event_time.date()
                ),

            "within_6_hours":
                bool(abs(difference_hours) <= 6),

            "within_12_hours":
                bool(abs(difference_hours) <= 12),

            "within_24_hours":
                bool(abs(difference_hours) <= 24),

            "scene_cloud_percentage":
                (
                    float(cloud)
                    if pd.notna(cloud)
                    else np.nan
                ),

            "search_window_days":
                SEARCH_WINDOW_DAYS,
        })

    return records


def main():
    initialize_earth_engine()

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    frame = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required_columns = [
        "event_id",
        "datetime_utc",
        "emission_kg_hr",
        "emission_tph",
        "lat",
        "lon",
        "already_in_patch_index",
    ]

    missing = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            "Missing columns: "
            + ", ".join(missing)
        )

    frame["emission_kg_hr"] = pd.to_numeric(
        frame["emission_kg_hr"],
        errors="coerce",
    )

    frame["emission_tph"] = pd.to_numeric(
        frame["emission_tph"],
        errors="coerce",
    )

    frame["lat"] = pd.to_numeric(
        frame["lat"],
        errors="coerce",
    )

    frame["lon"] = pd.to_numeric(
        frame["lon"],
        errors="coerce",
    )

    frame["_event_time"] = pd.to_datetime(
        frame["datetime_utc"],
        errors="coerce",
        utc=True,
    )

    already_downloaded = (
        frame["already_in_patch_index"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    shortlist = frame[
        frame["emission_kg_hr"].gt(0)
        & frame["emission_kg_hr"].lt(
            MAX_EMISSION_KG_H
        )
        & ~already_downloaded
    ].copy()

    shortlist = shortlist.dropna(
        subset=[
            "event_id",
            "_event_time",
            "lat",
            "lon",
        ]
    )

    shortlist = shortlist.drop_duplicates(
        subset=["event_id"],
        keep="first",
    )

    shortlist = shortlist.sort_values(
        [
            "emission_kg_hr",
            "datetime_utc",
            "event_id",
        ]
    ).reset_index(drop=True)

    shortlist.drop(
        columns=["_event_time"]
    ).to_csv(
        EVENT_OUTPUT,
        index=False,
    )

    # Restore parsed time after saving the clean event table.
    shortlist["_event_time"] = pd.to_datetime(
        shortlist["datetime_utc"],
        errors="coerce",
        utc=True,
    )

    print("=" * 105)
    print("METHANEAIR BELOW-500 KG/H S2 SCENE RESOLUTION")
    print("=" * 105)

    print("\nEvents to query:", len(shortlist))

    candidate_records = []
    event_status_records = []

    for number, (_, row) in enumerate(
        shortlist.iterrows(),
        start=1,
    ):
        print(
            f"\n[{number:02d}/{len(shortlist):02d}] "
            f"{row['event_id']} | "
            f"{row['emission_kg_hr']:.1f} kg/h"
        )

        try:
            records = query_event_scenes(row)

            print(
                "  Candidate scenes:",
                len(records),
            )

            candidate_records.extend(records)

            event_status_records.append({
                "event_id":
                    row["event_id"],

                "query_status":
                    "success",

                "candidate_scene_count":
                    len(records),

                "query_error":
                    "",
            })

        except Exception as error:
            print("  ERROR:", error)

            event_status_records.append({
                "event_id":
                    row["event_id"],

                "query_status":
                    "error",

                "candidate_scene_count":
                    0,

                "query_error":
                    str(error),
            })

        # Save progress after every event.
        pd.DataFrame(
            candidate_records
        ).to_csv(
            CANDIDATE_OUTPUT,
            index=False,
        )

        time.sleep(1)

    candidates = pd.DataFrame(
        candidate_records
    )

    event_status = pd.DataFrame(
        event_status_records
    )

    if candidates.empty:
        best = shortlist[
            [
                "event_id",
                "datetime_utc",
                "emission_kg_hr",
                "emission_tph",
                "lat",
                "lon",
            ]
        ].copy()

        best["resolution_status"] = (
            "no_candidate_scene"
        )

    else:
        candidates[
            "scene_cloud_percentage"
        ] = pd.to_numeric(
            candidates[
                "scene_cloud_percentage"
            ],
            errors="coerce",
        )

        candidates[
            "_cloud_for_sort"
        ] = candidates[
            "scene_cloud_percentage"
        ].fillna(999)

        candidates = candidates.sort_values(
            [
                "event_id",
                "absolute_time_difference_hours",
                "_cloud_for_sort",
                "scene_id",
            ]
        ).reset_index(drop=True)

        candidates[
            "candidate_rank_for_event"
        ] = (
            candidates.groupby(
                "event_id"
            )
            .cumcount()
            + 1
        )

        candidates[
            "selected_best_for_event"
        ] = candidates[
            "candidate_rank_for_event"
        ].eq(1)

        candidates.drop(
            columns=["_cloud_for_sort"],
            inplace=True,
        )

        candidates.to_csv(
            CANDIDATE_OUTPUT,
            index=False,
        )

        best = candidates[
            candidates[
                "selected_best_for_event"
            ]
        ].copy()

        best[
            "resolution_status"
        ] = "best_candidate_resolved"

        missing_events = shortlist[
            ~shortlist["event_id"].isin(
                best["event_id"]
            )
        ].copy()

        if not missing_events.empty:
            unresolved = pd.DataFrame({
                "event_id":
                    missing_events["event_id"],

                "event_time_utc":
                    missing_events[
                        "datetime_utc"
                    ],

                "emission_kg_hr":
                    missing_events[
                        "emission_kg_hr"
                    ],

                "emission_tph":
                    missing_events[
                        "emission_tph"
                    ],

                "latitude":
                    missing_events["lat"],

                "longitude":
                    missing_events["lon"],

                "resolution_status":
                    "no_candidate_scene",
            })

            best = pd.concat(
                [best, unresolved],
                ignore_index=True,
                sort=False,
            )

    best = best.merge(
        event_status,
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    if "scene_id" in best.columns:
        scene_counts = (
            best["scene_id"]
            .dropna()
            .value_counts()
        )

        best[
            "selected_scene_event_count"
        ] = best["scene_id"].map(
            scene_counts
        )

        best[
            "selected_scene_shared_by_events"
        ] = (
            best[
                "selected_scene_event_count"
            ].fillna(0).gt(1)
        )

    best.to_csv(
        BEST_OUTPUT,
        index=False,
    )

    resolved = best[
        best["resolution_status"].eq(
            "best_candidate_resolved"
        )
    ].copy()

    unique_scene_count = (
        resolved["scene_id"].nunique()
        if "scene_id" in resolved.columns
        else 0
    )

    same_day_count = (
        int(
            resolved["same_utc_date"]
            .fillna(False)
            .sum()
        )
        if "same_utc_date" in resolved.columns
        else 0
    )

    within_6h_count = (
        int(
            resolved["within_6_hours"]
            .fillna(False)
            .sum()
        )
        if "within_6_hours" in resolved.columns
        else 0
    )

    within_12h_count = (
        int(
            resolved["within_12_hours"]
            .fillna(False)
            .sum()
        )
        if "within_12_hours" in resolved.columns
        else 0
    )

    duplicated_scene_groups = (
        int(
            (
                resolved[
                    "scene_id"
                ].value_counts() > 1
            ).sum()
        )
        if not resolved.empty
        else 0
    )

    report_lines = [
        "=" * 105,
        "METHANEAIR BELOW-500 KG/H S2 SCENE RESOLUTION V1",
        "=" * 105,
        "",
        (
            "Important: these are candidate temporal matches. "
            "They are not yet locked methane-positive labels."
        ),
        "",
        f"Shortlisted unique events: {len(shortlist)}",
        (
            "Events with at least one S2 candidate: "
            f"{len(resolved)}"
        ),
        (
            "Events without an S2 candidate: "
            f"{len(shortlist) - len(resolved)}"
        ),
        (
            "Total candidate scene rows: "
            f"{len(candidates)}"
        ),
        (
            "Unique selected S2 scenes: "
            f"{unique_scene_count}"
        ),
        (
            "Selected scenes shared by multiple events: "
            f"{duplicated_scene_groups}"
        ),
        (
            "Selected matches on same UTC date: "
            f"{same_day_count}"
        ),
        (
            "Selected matches within 6 hours: "
            f"{within_6h_count}"
        ),
        (
            "Selected matches within 12 hours: "
            f"{within_12h_count}"
        ),
    ]

    if not resolved.empty:
        report_lines.extend([
            "",
            "Absolute time difference in hours:",
            resolved[
                "absolute_time_difference_hours"
            ].describe().to_string(),
            "",
            "Selected-scene cloud percentage:",
            resolved[
                "scene_cloud_percentage"
            ].describe().to_string(),
            "",
            "Selected scene reuse counts:",
            resolved[
                "scene_id"
            ].value_counts().head(20).to_string(),
        ])

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 105)
    print("RESOLUTION SUMMARY")
    print("=" * 105)

    print(
        "\nShortlisted unique events:",
        len(shortlist),
    )

    print(
        "Events with S2 candidates:",
        len(resolved),
    )

    print(
        "Events without S2 candidates:",
        len(shortlist) - len(resolved),
    )

    print(
        "Total candidate scene rows:",
        len(candidates),
    )

    print(
        "Unique selected S2 scenes:",
        unique_scene_count,
    )

    print(
        "Shared-scene groups:",
        duplicated_scene_groups,
    )

    print(
        "Same-date selected matches:",
        same_day_count,
    )

    print(
        "Selected matches within 6 hours:",
        within_6h_count,
    )

    print(
        "Selected matches within 12 hours:",
        within_12h_count,
    )

    print("\nSaved:")
    print(EVENT_OUTPUT)
    print(CANDIDATE_OUTPUT)
    print(BEST_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
