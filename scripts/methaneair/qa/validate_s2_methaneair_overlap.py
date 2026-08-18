#!/usr/bin/env python3
"""Validate MethaneAIR–Sentinel-2 matches against release intervals.

This script does not evaluate the MethaneFuse model. It checks whether the
selected Sentinel-2 acquisition and the MethaneAIR time fall inside the same
controlled-release interval near the source coordinates.

Inputs
------
outputs/461_methaneair_s2_best_scene_match_v1.csv
outputs/309_all_exact_release_intervals_for_s2.csv

Outputs
-------
outputs/464_s2_methaneair_release_interval_candidates_v1.csv
outputs/465_s2_methaneair_best_validated_matches_v1.csv
outputs/466_s2_methaneair_strict_release_overlap_v1.csv
outputs/467_s2_methaneair_validation_summary_v1.csv
outputs/468_s2_methaneair_temporal_validation_v1.csv
outputs/469_s2_methaneair_validation_report_v1.txt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="methane_release_project root.",
    )
    parser.add_argument("--matches-csv", type=Path, default=None)
    parser.add_argument("--intervals-csv", type=Path, default=None)
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=10.0,
        help="Maximum source-to-release-site distance.",
    )
    parser.add_argument(
        "--temporal-buffer-hours",
        type=float,
        default=24.0,
        help=(
            "Keep nearby intervals whose edge is within this many hours of "
            "the Sentinel-2 or MethaneAIR anchor time."
        ),
    )
    return parser.parse_args()


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius_km = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    value = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(value))


def parse_time(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def inside(timestamp: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return bool(
        pd.notna(timestamp)
        and pd.notna(start)
        and pd.notna(end)
        and start <= timestamp <= end
    )


def distance_to_interval_hours(
    timestamp: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    if pd.isna(timestamp) or pd.isna(start) or pd.isna(end):
        return float("inf")
    if start <= timestamp <= end:
        return 0.0
    if timestamp < start:
        return (start - timestamp).total_seconds() / 3600.0
    return (timestamp - end).total_seconds() / 3600.0


def validation_class(
    s2_inside: bool,
    methaneair_inside: bool,
    same_interval: bool,
    same_day: bool,
    nearest_gap_hours: float,
) -> str:
    if s2_inside and methaneair_inside and same_interval:
        return "strict_both_sensors_inside_same_release"
    if s2_inside:
        return "strict_s2_inside_release"
    if methaneair_inside:
        return "methaneair_inside_release_s2_outside"
    if same_day and nearest_gap_hours <= 24:
        return "near_time_same_day_not_inside_release"
    if nearest_gap_hours <= 24:
        return "near_time_different_date_not_inside_release"
    return "no_nearby_release_interval"


def main() -> None:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    outputs = root / "outputs"
    matches_path = (
        args.matches_csv.expanduser().resolve()
        if args.matches_csv
        else outputs / "461_methaneair_s2_best_scene_match_v1.csv"
    )
    intervals_path = (
        args.intervals_csv.expanduser().resolve()
        if args.intervals_csv
        else outputs / "309_all_exact_release_intervals_for_s2.csv"
    )

    candidates_path = outputs / "464_s2_methaneair_release_interval_candidates_v1.csv"
    best_path = outputs / "465_s2_methaneair_best_validated_matches_v1.csv"
    strict_path = outputs / "466_s2_methaneair_strict_release_overlap_v1.csv"
    summary_path = outputs / "467_s2_methaneair_validation_summary_v1.csv"
    temporal_path = outputs / "468_s2_methaneair_temporal_validation_v1.csv"
    report_path = outputs / "469_s2_methaneair_validation_report_v1.txt"

    if not matches_path.exists():
        raise FileNotFoundError(f"Best-match CSV not found: {matches_path}")
    if not intervals_path.exists():
        raise FileNotFoundError(f"Release-interval CSV not found: {intervals_path}")

    matches = pd.read_csv(matches_path, low_memory=False)
    intervals = pd.read_csv(intervals_path, low_memory=False)

    required_match = {
        "source_row_id",
        "event_id",
        "lat",
        "lon",
        "methaneair_anchor_time_utc",
        "s2_time_utc",
        "same_utc_date",
        "search_status",
    }
    required_interval = {
        "release_interval_id",
        "release_start_utc",
        "release_end_utc",
        "release_rate_kg_h",
        "site",
        "lat",
        "lon",
    }
    missing_match = sorted(required_match - set(matches.columns))
    missing_interval = sorted(required_interval - set(intervals.columns))
    if missing_match:
        raise ValueError(f"Best-match CSV is missing columns: {missing_match}")
    if missing_interval:
        raise ValueError(f"Interval CSV is missing columns: {missing_interval}")

    matches["lat"] = pd.to_numeric(matches["lat"], errors="coerce")
    matches["lon"] = pd.to_numeric(matches["lon"], errors="coerce")
    intervals["lat"] = pd.to_numeric(intervals["lat"], errors="coerce")
    intervals["lon"] = pd.to_numeric(intervals["lon"], errors="coerce")
    intervals["release_rate_kg_h"] = pd.to_numeric(
        intervals["release_rate_kg_h"], errors="coerce"
    )
    intervals["release_start"] = pd.to_datetime(
        intervals["release_start_utc"], utc=True, errors="coerce"
    )
    intervals["release_end"] = pd.to_datetime(
        intervals["release_end_utc"], utc=True, errors="coerce"
    )

    intervals = intervals.dropna(
        subset=["lat", "lon", "release_start", "release_end"]
    ).copy()

    candidate_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    for _, match in matches.iterrows():
        base = match.to_dict()
        if str(match.get("search_status")) != "best_scene":
            best_rows.append(
                {
                    **base,
                    "validation_class": "no_s2_scene",
                    "release_candidate_count": 0,
                }
            )
            continue

        lat = float(match["lat"])
        lon = float(match["lon"])
        s2_time = parse_time(match["s2_time_utc"])
        methaneair_time = parse_time(match["methaneair_anchor_time_utc"])
        same_day = str(match.get("same_utc_date")).strip().lower() in {
            "true", "1", "yes"
        }

        current: list[dict[str, Any]] = []
        for _, interval in intervals.iterrows():
            distance = haversine_km(
                lat, lon, float(interval["lat"]), float(interval["lon"])
            )
            if distance > args.max_distance_km:
                continue

            start = interval["release_start"]
            end = interval["release_end"]
            s2_inside = inside(s2_time, start, end)
            methaneair_inside = inside(methaneair_time, start, end)
            s2_gap = distance_to_interval_hours(s2_time, start, end)
            methaneair_gap = distance_to_interval_hours(
                methaneair_time, start, end
            )
            nearest_gap = min(s2_gap, methaneair_gap)

            if (
                not s2_inside
                and not methaneair_inside
                and nearest_gap > args.temporal_buffer_hours
            ):
                continue

            same_interval = bool(s2_inside and methaneair_inside)
            category = validation_class(
                s2_inside=s2_inside,
                methaneair_inside=methaneair_inside,
                same_interval=same_interval,
                same_day=same_day,
                nearest_gap_hours=nearest_gap,
            )
            rank = {
                "strict_both_sensors_inside_same_release": 0,
                "strict_s2_inside_release": 1,
                "methaneair_inside_release_s2_outside": 2,
                "near_time_same_day_not_inside_release": 3,
                "near_time_different_date_not_inside_release": 4,
                "no_nearby_release_interval": 5,
            }[category]

            record = {
                **base,
                "release_interval_id": interval["release_interval_id"],
                "release_site": interval["site"],
                "release_start_utc": start.isoformat(),
                "release_end_utc": end.isoformat(),
                "release_rate_kg_h": interval["release_rate_kg_h"],
                "emission_bin": interval.get("emission_bin"),
                "distance_to_release_site_km": distance,
                "s2_inside_release_interval": s2_inside,
                "methaneair_inside_release_interval": methaneair_inside,
                "both_inside_same_release_interval": same_interval,
                "s2_distance_to_interval_hours": s2_gap,
                "methaneair_distance_to_interval_hours": methaneair_gap,
                "nearest_sensor_to_interval_hours": nearest_gap,
                "validation_class": category,
                "_rank": rank,
            }
            current.append(record)
            candidate_rows.append(record)

        if current:
            selected = min(
                current,
                key=lambda item: (
                    item["_rank"],
                    item["nearest_sensor_to_interval_hours"],
                    item["distance_to_release_site_km"],
                    str(item["release_interval_id"]),
                ),
            )
            selected = dict(selected)
            selected.pop("_rank", None)
            selected["release_candidate_count"] = len(current)
            best_rows.append(selected)
        else:
            best_rows.append(
                {
                    **base,
                    "validation_class": "no_nearby_release_interval",
                    "release_candidate_count": 0,
                    "release_interval_id": None,
                    "release_site": None,
                    "release_start_utc": None,
                    "release_end_utc": None,
                    "release_rate_kg_h": None,
                    "emission_bin": None,
                    "distance_to_release_site_km": None,
                    "s2_inside_release_interval": False,
                    "methaneair_inside_release_interval": False,
                    "both_inside_same_release_interval": False,
                    "s2_distance_to_interval_hours": None,
                    "methaneair_distance_to_interval_hours": None,
                    "nearest_sensor_to_interval_hours": None,
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    best = pd.DataFrame(best_rows)
    if "_rank" in candidates.columns:
        candidates = candidates.drop(columns=["_rank"])

    candidates.to_csv(candidates_path, index=False)
    best.to_csv(best_path, index=False)

    strict = best[
        best["validation_class"].isin(
            [
                "strict_both_sensors_inside_same_release",
                "strict_s2_inside_release",
            ]
        )
    ].copy()
    strict.to_csv(strict_path, index=False)

    summary = (
        best.groupby("validation_class", dropna=False)
        .agg(
            rows=("source_row_id", "size"),
            unique_events=("event_id", "nunique"),
            min_methaneair_emission_kg_hr=("emission_kg_hr", "min"),
            median_methaneair_emission_kg_hr=("emission_kg_hr", "median"),
            max_methaneair_emission_kg_hr=("emission_kg_hr", "max"),
            min_release_rate_kg_h=("release_rate_kg_h", "min"),
            median_release_rate_kg_h=("release_rate_kg_h", "median"),
            max_release_rate_kg_h=("release_rate_kg_h", "max"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    temporal_columns = [
        "source_row_id",
        "event_id",
        "flight_id",
        "emission_kg_hr",
        "lat",
        "lon",
        "methaneair_anchor_time_utc",
        "s2_scene_id",
        "s2_asset_id",
        "s2_time_utc",
        "absolute_time_difference_hours",
        "same_utc_date",
        "temporal_tier",
        "cloudy_pixel_percentage",
        "release_interval_id",
        "release_site",
        "release_start_utc",
        "release_end_utc",
        "release_rate_kg_h",
        "distance_to_release_site_km",
        "s2_inside_release_interval",
        "methaneair_inside_release_interval",
        "both_inside_same_release_interval",
        "validation_class",
    ]
    best[
        [column for column in temporal_columns if column in best.columns]
    ].to_csv(temporal_path, index=False)

    counts = best["validation_class"].value_counts(dropna=False)
    report = [
        "MethaneAIR–Sentinel-2 controlled-release validation",
        "=" * 72,
        f"Best-scene input: {matches_path}",
        f"Release intervals: {intervals_path}",
        f"Rows checked: {len(best)}",
        f"Maximum spatial distance: {args.max_distance_km:g} km",
        f"Temporal buffer: {args.temporal_buffer_hours:g} hours",
        "",
        "Validation-class counts:",
        counts.to_string(),
        "",
        "Important interpretation:",
        "- The MethaneAIR availability file contains observational detections.",
        "- The release-interval file mainly covers controlled releases at",
        "  Casa Grande and Ehrenberg, plus rows with unknown site names.",
        "- Therefore, no release match does not mean the MethaneAIR detection",
        "  was false; it means these two supplied tables do not establish a",
        "  controlled-release overlap.",
        "- Model agreement requires downloading the selected S2 scenes and",
        "  running MethaneFuse; this script only validates time and location.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print("\nCreated:")
    print(candidates_path)
    print(best_path)
    print(strict_path)
    print(summary_path)
    print(temporal_path)
    print(report_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
