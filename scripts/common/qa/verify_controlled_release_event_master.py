from pathlib import Path
from math import radians, sin, cos, asin, sqrt

import numpy as np
import pandas as pd


EVENT_INPUT = Path(
    "outputs/10_final_events_for_gee.csv"
)

INTERVAL_INPUT = Path(
    "outputs/309_all_exact_release_intervals_for_s2.csv"
)

VERIFIED_OUTPUT = Path(
    "outputs/473_controlled_release_verified_event_master_v1.csv"
)

UNRESOLVED_OUTPUT = Path(
    "outputs/474_controlled_release_event_verification_issues_v1.csv"
)

MATCH_OUTPUT = Path(
    "outputs/475_controlled_release_event_interval_matches_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/476_controlled_release_event_verification_report_v1.txt"
)


# 允許不同資料來源之間有少量座標差異。
MAX_SOURCE_DISTANCE_M = 1000.0


def haversine_m(lat1, lon1, lat2, lon2):
    radius_m = 6371000.0

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    value = (
        sin(dlat / 2.0) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2.0) ** 2
    )

    return 2.0 * radius_m * asin(sqrt(value))


def parse_boolean(value):
    """Convert common Boolean, numeric, and text labels."""

    if pd.isna(value):
        return np.nan

    # Native Boolean values.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    # Numeric labels such as 1, 0, 1.0, and 0.0.
    try:
        numeric_value = float(value)

        if np.isclose(numeric_value, 1.0):
            return True

        if np.isclose(numeric_value, 0.0):
            return False

    except (TypeError, ValueError):
        pass

    text = str(value).strip().lower()

    true_values = {
        "true",
        "yes",
        "y",
        "positive",
        "release",
        "released",
        "true_release",
        "release_present",
        "plume",
        "plume_present",
    }

    false_values = {
        "false",
        "no",
        "n",
        "negative",
        "no_release",
        "no release",
        "release_absent",
        "no plume",
        "plume_absent",
    }

    if text in true_values:
        return True

    if text in false_values:
        return False

    return np.nan


def find_column(frame, candidates, description):
    for column in candidates:
        if column in frame.columns:
            return column

    raise KeyError(
        f"Cannot find {description}. Tried: "
        + ", ".join(candidates)
    )


def calculate_time_gap_minutes(
    event_time,
    release_start,
    release_end,
):
    if release_start <= event_time <= release_end:
        return 0.0

    if event_time < release_start:
        return (
            release_start - event_time
        ).total_seconds() / 60.0

    return (
        event_time - release_end
    ).total_seconds() / 60.0


def main():
    if not EVENT_INPUT.exists():
        raise FileNotFoundError(EVENT_INPUT)

    if not INTERVAL_INPUT.exists():
        raise FileNotFoundError(INTERVAL_INPUT)

    events = pd.read_csv(
        EVENT_INPUT,
        low_memory=False,
    )

    intervals = pd.read_csv(
        INTERVAL_INPUT,
        low_memory=False,
    )

    print("=" * 115)
    print("CONTROLLED-RELEASE EVENT MASTER VERIFICATION")
    print("=" * 115)

    print("\nEvent rows:", len(events))
    print("Exact interval rows:", len(intervals))

    event_id_column = find_column(
        events,
        ["event_id"],
        "event ID",
    )

    event_time_column = find_column(
        events,
        ["datetime_utc", "event_time_utc"],
        "event time",
    )

    event_latitude_column = find_column(
        events,
        ["lat", "latitude"],
        "event latitude",
    )

    event_longitude_column = find_column(
        events,
        ["lon", "longitude"],
        "event longitude",
    )

    label_column = find_column(
        events,
        ["true_release", "label"],
        "ground-truth label",
    )

    interval_start_column = find_column(
        intervals,
        ["release_start_utc"],
        "release interval start",
    )

    interval_end_column = find_column(
        intervals,
        ["release_end_utc"],
        "release interval end",
    )

    interval_latitude_column = find_column(
        intervals,
        ["lat", "latitude"],
        "interval latitude",
    )

    interval_longitude_column = find_column(
        intervals,
        ["lon", "longitude"],
        "interval longitude",
    )

    interval_rate_column = find_column(
        intervals,
        [
            "release_rate_kg_h",
            "emission_kg_hr",
        ],
        "release rate",
    )

    events["_event_id"] = (
        events[event_id_column]
        .astype(str)
        .str.strip()
    )

    events["_event_time"] = pd.to_datetime(
        events[event_time_column],
        errors="coerce",
        utc=True,
    )

    events["_latitude"] = pd.to_numeric(
        events[event_latitude_column],
        errors="coerce",
    )

    events["_longitude"] = pd.to_numeric(
        events[event_longitude_column],
        errors="coerce",
    )

    events["_true_release"] = events[
        label_column
    ].map(parse_boolean)

    intervals["_release_start"] = pd.to_datetime(
        intervals[interval_start_column],
        errors="coerce",
        utc=True,
    )

    intervals["_release_end"] = pd.to_datetime(
        intervals[interval_end_column],
        errors="coerce",
        utc=True,
    )

    intervals["_latitude"] = pd.to_numeric(
        intervals[interval_latitude_column],
        errors="coerce",
    )

    intervals["_longitude"] = pd.to_numeric(
        intervals[interval_longitude_column],
        errors="coerce",
    )

    intervals["_release_rate_kg_h"] = pd.to_numeric(
        intervals[interval_rate_column],
        errors="coerce",
    )

    intervals = intervals.dropna(
        subset=[
            "_release_start",
            "_release_end",
            "_latitude",
            "_longitude",
        ]
    ).copy()

    intervals = intervals[
        intervals["_release_end"]
        >= intervals["_release_start"]
    ].copy()

    verification_records = []
    match_records = []

    for number, (_, event) in enumerate(
        events.iterrows(),
        start=1,
    ):
        event_id = event["_event_id"]
        event_time = event["_event_time"]
        latitude = event["_latitude"]
        longitude = event["_longitude"]
        true_release = event["_true_release"]

        base = event.to_dict()

        base.update({
            "event_id_verified":
                event_id,

            "event_time_utc_verified":
                event_time,

            "latitude_verified":
                latitude,

            "longitude_verified":
                longitude,

            "true_release_verified":
                true_release,
        })

        if (
            pd.isna(event_time)
            or pd.isna(latitude)
            or pd.isna(longitude)
            or pd.isna(true_release)
        ):
            base.update({
                "verification_status":
                    "invalid_event_metadata",

                "exact_interval_match_count":
                    0,

                "ground_truth_verified":
                    False,

                "verification_issue":
                    "missing_time_location_or_label",
            })

            verification_records.append(base)
            continue

        # 先限制在事件前後一天，減少無關比較。
        time_candidates = intervals[
            (
                intervals["_release_start"]
                <= event_time + pd.Timedelta(days=1)
            )
            & (
                intervals["_release_end"]
                >= event_time - pd.Timedelta(days=1)
            )
        ].copy()

        candidate_records = []

        for interval_index, interval in (
            time_candidates.iterrows()
        ):
            distance_m = haversine_m(
                latitude,
                longitude,
                interval["_latitude"],
                interval["_longitude"],
            )

            if distance_m > MAX_SOURCE_DISTANCE_M:
                continue

            inside_interval = bool(
                interval["_release_start"]
                <= event_time
                <= interval["_release_end"]
            )

            time_gap_minutes = (
                calculate_time_gap_minutes(
                    event_time,
                    interval["_release_start"],
                    interval["_release_end"],
                )
            )

            record = {
                "event_id":
                    event_id,

                "event_time_utc":
                    event_time,

                "true_release":
                    true_release,

                "interval_source_row":
                    int(interval_index),

                "release_start_utc":
                    interval["_release_start"],

                "release_end_utc":
                    interval["_release_end"],

                "release_rate_kg_h":
                    interval[
                        "_release_rate_kg_h"
                    ],

                "interval_latitude":
                    interval["_latitude"],

                "interval_longitude":
                    interval["_longitude"],

                "source_distance_m":
                    distance_m,

                "event_inside_release_interval":
                    inside_interval,

                "time_gap_minutes":
                    time_gap_minutes,

                "source_file":
                    (
                        interval["source_file"]
                        if "source_file"
                        in interval.index
                        else ""
                    ),
            }

            candidate_records.append(record)
            match_records.append(record)

        candidates = pd.DataFrame(
            candidate_records
        )

        if candidates.empty:
            exact = candidates
        else:
            exact = candidates[
                candidates[
                    "event_inside_release_interval"
                ].eq(True)
            ].copy()

        exact_match_count = len(exact)

        if true_release is True:
            if exact_match_count > 0:
                status = (
                    "positive_exact_interval_verified"
                )

                verified = True
                issue = ""
            else:
                status = (
                    "positive_without_exact_interval_match"
                )

                verified = False
                issue = (
                    "reported_positive_but_no_exact_"
                    "release_interval_at_event_time"
                )

        else:
            if exact_match_count == 0:
                status = (
                    "negative_no_overlapping_release_verified"
                )

                verified = True
                issue = ""
            else:
                status = (
                    "negative_conflict_with_release_interval"
                )

                verified = False
                issue = (
                    "reported_negative_but_event_time_"
                    "overlaps_known_release"
                )

        base.update({
            "verification_status":
                status,

            "exact_interval_match_count":
                exact_match_count,

            "ground_truth_verified":
                verified,

            "verification_issue":
                issue,
        })

        if exact_match_count > 0:
            selected = exact.sort_values(
                [
                    "source_distance_m",
                    "release_start_utc",
                    "release_rate_kg_h",
                ],
                na_position="last",
            ).iloc[0]

            exact_rates = pd.to_numeric(
                exact["release_rate_kg_h"],
                errors="coerce",
            )

            base.update({
                "selected_release_start_utc":
                    selected[
                        "release_start_utc"
                    ],

                "selected_release_end_utc":
                    selected[
                        "release_end_utc"
                    ],

                "selected_release_rate_kg_h":
                    selected[
                        "release_rate_kg_h"
                    ],

                "median_matching_release_rate_kg_h":
                    exact_rates.median(),

                "minimum_matching_release_rate_kg_h":
                    exact_rates.min(),

                "maximum_matching_release_rate_kg_h":
                    exact_rates.max(),

                "selected_source_distance_m":
                    selected[
                        "source_distance_m"
                    ],

                "selected_interval_source_file":
                    selected["source_file"],
            })

        elif not candidates.empty:
            nearest = candidates.sort_values(
                [
                    "time_gap_minutes",
                    "source_distance_m",
                ]
            ).iloc[0]

            base.update({
                "nearest_release_time_gap_minutes":
                    nearest["time_gap_minutes"],

                "nearest_release_distance_m":
                    nearest["source_distance_m"],

                "nearest_release_start_utc":
                    nearest["release_start_utc"],

                "nearest_release_end_utc":
                    nearest["release_end_utc"],
            })

        verification_records.append(base)

        if number % 20 == 0:
            print(
                f"Verified {number}/{len(events)} events..."
            )

    verified = pd.DataFrame(
        verification_records
    )

    matches = pd.DataFrame(
        match_records
    )

    verified[
        "ground_truth_verified"
    ] = (
        verified[
            "ground_truth_verified"
        ].fillna(False)
    )

    verified[
        "multisensor_matching_ready"
    ] = verified[
        "ground_truth_verified"
    ]

    verified[
        "evaluation_group"
    ] = verified[
        "event_id_verified"
    ]

    verified.to_csv(
        VERIFIED_OUTPUT,
        index=False,
    )

    unresolved = verified[
        ~verified[
            "ground_truth_verified"
        ]
    ].copy()

    unresolved.to_csv(
        UNRESOLVED_OUTPUT,
        index=False,
    )

    matches.to_csv(
        MATCH_OUTPUT,
        index=False,
    )

    status_summary = (
        verified["verification_status"]
        .value_counts(dropna=False)
    )

    original_positive_count = int(
        verified[
            "true_release_verified"
        ].eq(True).sum()
    )

    original_negative_count = int(
        verified[
            "true_release_verified"
        ].eq(False).sum()
    )

    verified_count = int(
        verified[
            "ground_truth_verified"
        ].sum()
    )

    ready = verified[
        verified[
            "multisensor_matching_ready"
        ]
    ].copy()

    ready_positive_count = int(
        ready[
            "true_release_verified"
        ].eq(True).sum()
    )

    ready_negative_count = int(
        ready[
            "true_release_verified"
        ].eq(False).sum()
    )

    report_lines = [
        "=" * 115,
        "CONTROLLED-RELEASE EVENT MASTER VERIFICATION V1",
        "=" * 115,
        "",
        f"Input deduplicated events: {len(events)}",
        f"Original positive events: {original_positive_count}",
        f"Original negative events: {original_negative_count}",
        f"Verified events: {verified_count}",
        f"Unresolved/conflicting events: {len(unresolved)}",
        "",
        (
            "Events ready for multisensor matching: "
            f"{len(ready)}"
        ),
        (
            "Ready positive events: "
            f"{ready_positive_count}"
        ),
        (
            "Ready negative events: "
            f"{ready_negative_count}"
        ),
        "",
        "Verification status:",
        status_summary.to_string(),
        "",
        (
            "Matching rule: event location within "
            f"{MAX_SOURCE_DISTANCE_M:.0f} m and acquisition "
            "time inside an exact release interval."
        ),
        "",
        (
            "The 458-row interval inventory is used only as "
            "verification evidence. It is not treated as 458 "
            "independent release events."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 115)
    print("VERIFICATION SUMMARY")
    print("=" * 115)

    print(
        "\nInput deduplicated events:",
        len(events),
    )

    print(
        "Original positive events:",
        original_positive_count,
    )

    print(
        "Original negative events:",
        original_negative_count,
    )

    print(
        "Verified events:",
        verified_count,
    )

    print(
        "Unresolved/conflicting events:",
        len(unresolved),
    )

    print(
        "Events ready for multisensor matching:",
        len(ready),
    )

    print(
        "Ready positive events:",
        ready_positive_count,
    )

    print(
        "Ready negative events:",
        ready_negative_count,
    )

    print("\nVerification status:")
    print(status_summary)

    print("\nSaved:")
    print(VERIFIED_OUTPUT)
    print(UNRESOLVED_OUTPUT)
    print(MATCH_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
