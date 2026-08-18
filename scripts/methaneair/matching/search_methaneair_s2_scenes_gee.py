#!/usr/bin/env python3
"""Search real Sentinel-2 acquisitions near MethaneAIR detections.

Inputs
------
outputs/15_methaneair_s2_landsat_availability.csv

Outputs
-------
outputs/460_methaneair_s2_scene_candidates_v1.csv
outputs/461_methaneair_s2_best_scene_match_v1.csv
outputs/462_methaneair_s2_scene_search_summary_v1.csv
outputs/463_methaneair_s2_scene_search_report_v1.txt

The script searches metadata only. It does not download imagery and does not
claim that a same-day Sentinel-2 scene detected methane.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import ee
import numpy as np
import pandas as pd


S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="methane_release_project root.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Override the default availability CSV.",
    )
    parser.add_argument(
        "--ee-project",
        default=None,
        help="Earth Engine project. Defaults to EE_PROJECT or methane-release-gee.",
    )
    parser.add_argument(
        "--emission-min",
        type=float,
        default=None,
        help="Optional minimum MethaneAIR emission rate in kg/h.",
    )
    parser.add_argument(
        "--emission-max",
        type=float,
        default=None,
        help="Optional maximum MethaneAIR emission rate in kg/h.",
    )
    parser.add_argument(
        "--search-hours",
        type=float,
        default=24.0,
        help="Search this many hours before and after the MethaneAIR anchor time.",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=100.0,
        help="Maximum Sentinel-2 CLOUDY_PIXEL_PERCENTAGE.",
    )
    parser.add_argument(
        "--time-basis",
        choices=("coverage_midpoint", "datetime"),
        default="coverage_midpoint",
        help=(
            "Anchor the search on the MethaneAIR flight coverage midpoint "
            "or on datetime_utc."
        ),
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional row limit for a smoke test.",
    )
    parser.add_argument(
        "--only-input-s2-available",
        action="store_true",
        help="Only search rows whose existing s2_count_pm1day is greater than zero.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to prior outputs and skip source rows already processed.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.15,
        help="Pause between Earth Engine requests.",
    )
    return parser.parse_args()


def initialize_ee(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Run `earthengine authenticate` "
            f"and verify project={project!r}. Original error: {exc}"
        ) from exc


def utc_timestamp(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def iso_or_none(value: Any) -> str | None:
    timestamp = utc_timestamp(value)
    if pd.isna(timestamp):
        return None
    return timestamp.isoformat()


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def derive_anchor(row: pd.Series, basis: str) -> tuple[pd.Timestamp, str]:
    dt = utc_timestamp(row.get("datetime_utc"))
    start = utc_timestamp(row.get("time_coverage_start"))
    end = utc_timestamp(row.get("time_coverage_end"))

    if basis == "coverage_midpoint" and pd.notna(start) and pd.notna(end):
        return start + ((end - start) / 2), "coverage_midpoint"

    if pd.notna(dt):
        return dt, "datetime_utc"

    if pd.notna(start):
        return start, "time_coverage_start"

    raise ValueError(
        f"No usable MethaneAIR time for source row {row.get('system:index')!r}."
    )


def temporal_tier(
    anchor: pd.Timestamp,
    s2_time: pd.Timestamp,
    difference_hours: float,
) -> str:
    same_date = anchor.date() == s2_time.date()
    if not same_date:
        return "different_utc_date"
    if difference_hours <= 1:
        return "A_same_day_le_1h"
    if difference_hours <= 3:
        return "B_same_day_1_to_3h"
    if difference_hours <= 6:
        return "C_same_day_3_to_6h"
    return "D_same_day_6_to_24h"


def empty_best_record(base: dict[str, Any]) -> dict[str, Any]:
    return {
        **base,
        "candidate_count": 0,
        "s2_scene_id": None,
        "s2_asset_id": None,
        "s2_time_utc": None,
        "absolute_time_difference_hours": None,
        "same_utc_date": False,
        "s2_inside_methaneair_coverage": False,
        "temporal_tier": "no_scene",
        "cloudy_pixel_percentage": None,
        "mgrs_tile": None,
        "product_id": None,
        "spacecraft_name": None,
        "search_status": "no_scene",
        "search_error": None,
    }


def read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def write_combined(path: Path, previous: pd.DataFrame, new_rows: list[dict[str, Any]]) -> None:
    new_df = pd.DataFrame(new_rows)
    if previous.empty:
        combined = new_df
    elif new_df.empty:
        combined = previous
    else:
        combined = pd.concat([previous, new_df], ignore_index=True, sort=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    input_path = (
        args.input_csv.expanduser().resolve()
        if args.input_csv is not None
        else root / "outputs/15_methaneair_s2_landsat_availability.csv"
    )
    outputs = root / "outputs"
    candidates_path = outputs / "460_methaneair_s2_scene_candidates_v1.csv"
    best_path = outputs / "461_methaneair_s2_best_scene_match_v1.csv"
    summary_path = outputs / "462_methaneair_s2_scene_search_summary_v1.csv"
    report_path = outputs / "463_methaneair_s2_scene_search_report_v1.txt"

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    project = (
        args.ee_project
        or __import__("os").environ.get("EE_PROJECT")
        or "methane-release-gee"
    )
    initialize_ee(project)

    source = pd.read_csv(input_path, low_memory=False)
    required = {
        "system:index",
        "event_id",
        "flight_id",
        "datetime_utc",
        "time_coverage_start",
        "time_coverage_end",
        "emission_kg_hr",
        "lat",
        "lon",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    source["emission_kg_hr"] = pd.to_numeric(
        source["emission_kg_hr"], errors="coerce"
    )
    source["lat"] = pd.to_numeric(source["lat"], errors="coerce")
    source["lon"] = pd.to_numeric(source["lon"], errors="coerce")

    source = source.dropna(
        subset=["system:index", "lat", "lon", "emission_kg_hr"]
    ).copy()
    source["source_row_id"] = source["system:index"].astype(str)

    if args.emission_min is not None:
        source = source[source["emission_kg_hr"] >= args.emission_min].copy()
    if args.emission_max is not None:
        source = source[source["emission_kg_hr"] <= args.emission_max].copy()
    if args.only_input_s2_available and "s2_count_pm1day" in source.columns:
        count = pd.to_numeric(source["s2_count_pm1day"], errors="coerce").fillna(0)
        source = source[count > 0].copy()

    source = source.sort_values(
        ["emission_kg_hr", "datetime_utc", "source_row_id"]
    ).reset_index(drop=True)

    previous_candidates = read_existing(candidates_path) if args.resume else pd.DataFrame()
    previous_best = read_existing(best_path) if args.resume else pd.DataFrame()
    processed: set[str] = set()
    if not previous_best.empty and "source_row_id" in previous_best.columns:
        processed = set(previous_best["source_row_id"].astype(str))

    if processed:
        source = source[~source["source_row_id"].isin(processed)].copy()

    if args.max_events is not None:
        source = source.head(args.max_events).copy()

    print(f"Earth Engine project: {project}")
    print(f"Input rows selected: {len(source)}")
    print(f"Search window: ±{args.search_hours:g} hours")
    print(f"Time basis: {args.time_basis}")

    candidate_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    for position, (_, row) in enumerate(source.iterrows(), start=1):
        source_row_id = str(row["source_row_id"])
        anchor, anchor_basis = derive_anchor(row, args.time_basis)
        coverage_start = utc_timestamp(row.get("time_coverage_start"))
        coverage_end = utc_timestamp(row.get("time_coverage_end"))
        start = anchor - pd.Timedelta(hours=args.search_hours)
        end = anchor + pd.Timedelta(hours=args.search_hours)
        lat = float(row["lat"])
        lon = float(row["lon"])

        base = {
            "source_row_id": source_row_id,
            "event_id": row.get("event_id"),
            "flight_id": row.get("flight_id"),
            "plume_id": row.get("plume_id"),
            "label": row.get("label"),
            "label_type": row.get("label_type"),
            "ground_truth_type": row.get("ground_truth_type"),
            "source_dataset": row.get("source_dataset"),
            "emission_kg_hr": row.get("emission_kg_hr"),
            "lat": lat,
            "lon": lon,
            "methaneair_datetime_utc": iso_or_none(row.get("datetime_utc")),
            "methaneair_coverage_start_utc": iso_or_none(coverage_start),
            "methaneair_coverage_end_utc": iso_or_none(coverage_end),
            "methaneair_anchor_time_utc": anchor.isoformat(),
            "methaneair_time_basis": anchor_basis,
            "input_s2_count_pm1day": row.get("s2_count_pm1day"),
            "search_window_start_utc": start.isoformat(),
            "search_window_end_utc": end.isoformat(),
            "search_hours_each_side": args.search_hours,
            "max_cloud_filter": args.max_cloud,
        }

        print(
            f"[{position}/{len(source)}] {source_row_id} "
            f"event={row.get('event_id')} emission={row.get('emission_kg_hr'):.1f}"
        )

        try:
            point = ee.Geometry.Point([lon, lat])
            collection = (
                ee.ImageCollection(S2_COLLECTION)
                .filterBounds(point)
                .filterDate(start.isoformat(), end.isoformat())
                .filter(
                    ee.Filter.lte(
                        "CLOUDY_PIXEL_PERCENTAGE",
                        float(args.max_cloud),
                    )
                )
                .sort("system:time_start")
            )
            info = collection.getInfo()
            features = info.get("features", []) if isinstance(info, dict) else []

            current: list[dict[str, Any]] = []
            for feature in features:
                properties = feature.get("properties", {})
                raw_time = properties.get("system:time_start")
                s2_time = pd.to_datetime(raw_time, unit="ms", utc=True, errors="coerce")
                if pd.isna(s2_time):
                    continue
                difference_hours = abs((s2_time - anchor).total_seconds()) / 3600.0
                inside_coverage = bool(
                    pd.notna(coverage_start)
                    and pd.notna(coverage_end)
                    and coverage_start <= s2_time <= coverage_end
                )
                scene_id = properties.get("system:index") or feature.get("id")
                asset_id = feature.get("id")
                record = {
                    **base,
                    "candidate_count": len(features),
                    "s2_scene_id": scene_id,
                    "s2_asset_id": asset_id,
                    "s2_time_utc": s2_time.isoformat(),
                    "absolute_time_difference_hours": difference_hours,
                    "same_utc_date": anchor.date() == s2_time.date(),
                    "s2_inside_methaneair_coverage": inside_coverage,
                    "temporal_tier": temporal_tier(
                        anchor, s2_time, difference_hours
                    ),
                    "cloudy_pixel_percentage": properties.get(
                        "CLOUDY_PIXEL_PERCENTAGE"
                    ),
                    "mgrs_tile": properties.get("MGRS_TILE"),
                    "product_id": properties.get("PRODUCT_ID"),
                    "spacecraft_name": properties.get("SPACECRAFT_NAME"),
                    "search_status": "candidate",
                    "search_error": None,
                }
                current.append(record)
                candidate_rows.append(record)

            if current:
                best = min(
                    current,
                    key=lambda item: (
                        float(item["absolute_time_difference_hours"]),
                        safe_float(item["cloudy_pixel_percentage"]),
                        str(item["s2_scene_id"]),
                    ),
                )
                best = dict(best)
                best["search_status"] = "best_scene"
                best_rows.append(best)
                print(
                    f"  best={best['s2_scene_id']} "
                    f"Δt={best['absolute_time_difference_hours']:.2f} h "
                    f"tier={best['temporal_tier']}"
                )
            else:
                best_rows.append(empty_best_record(base))
                print("  no Sentinel-2 scene")

        except Exception as exc:
            failed = empty_best_record(base)
            failed["search_status"] = "failed"
            failed["search_error"] = f"{type(exc).__name__}: {exc}"
            best_rows.append(failed)
            print(f"  FAILED: {failed['search_error']}")

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    write_combined(candidates_path, previous_candidates, candidate_rows)
    write_combined(best_path, previous_best, best_rows)

    best = pd.read_csv(best_path, low_memory=False)
    if len(best):
        summary = (
            best.groupby(
                ["temporal_tier", "search_status"],
                dropna=False,
            )
            .agg(
                rows=("source_row_id", "size"),
                unique_events=("event_id", "nunique"),
                min_emission_kg_hr=("emission_kg_hr", "min"),
                median_emission_kg_hr=("emission_kg_hr", "median"),
                max_emission_kg_hr=("emission_kg_hr", "max"),
                median_time_difference_hours=(
                    "absolute_time_difference_hours",
                    "median",
                ),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame()
    summary.to_csv(summary_path, index=False)

    matched = best[best["search_status"].eq("best_scene")] if len(best) else best
    same_day = (
        int(matched["same_utc_date"].fillna(False).astype(bool).sum())
        if len(matched)
        else 0
    )
    inside_coverage = (
        int(
            matched["s2_inside_methaneair_coverage"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if len(matched)
        else 0
    )
    report = [
        "MethaneAIR–Sentinel-2 scene search",
        "=" * 72,
        f"Input: {input_path}",
        f"Earth Engine collection: {S2_COLLECTION}",
        f"Earth Engine project: {project}",
        f"Rows in accumulated best-match table: {len(best)}",
        f"Rows with a Sentinel-2 scene: {len(matched)}",
        f"Best scenes on the same UTC date: {same_day}",
        f"Best scenes inside MethaneAIR flight coverage: {inside_coverage}",
        "",
        "Temporal-tier counts:",
        (
            best["temporal_tier"].value_counts(dropna=False).to_string()
            if len(best)
            else "NONE"
        ),
        "",
        "Important interpretation:",
        "- This table establishes acquisition-time proximity only.",
        "- A same-day scene is not automatically a methane detection.",
        "- datetime_utc is often a flight-level time; the default anchor is",
        "  therefore the midpoint of time_coverage_start/time_coverage_end.",
        "- Controlled-release overlap must be checked separately.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n" + "\n".join(report))
    print("\nCreated:")
    print(candidates_path)
    print(best_path)
    print(summary_path)
    print(report_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
