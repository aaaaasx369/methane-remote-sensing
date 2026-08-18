from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sys

import ee
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/happydoraaa/methane_release_project")
INTERVALS_PATH = PROJECT_ROOT / "outputs/309_all_exact_release_intervals_for_s2.csv"

CANDIDATES_PATH = PROJECT_ROOT / "outputs/82_all_release_day_satellite_candidates.csv"
OVERLAPS_PATH = PROJECT_ROOT / "outputs/83_all_exact_satellite_release_overlaps.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs/83_all_exact_satellite_release_overlaps_summary.txt"

COLLECTIONS = {
    "Sentinel-2": {
        "collection_id": "COPERNICUS/S2_SR_HARMONIZED",
        "properties": [
            "system:index",
            "system:time_start",
            "CLOUDY_PIXEL_PERCENTAGE",
            "MGRS_TILE",
            "PRODUCT_ID",
            "SPACECRAFT_NAME",
        ],
    },
    "Landsat-8": {
        "collection_id": "LANDSAT/LC08/C02/T1_L2",
        "properties": [
            "system:index",
            "system:time_start",
            "CLOUD_COVER",
            "WRS_PATH",
            "WRS_ROW",
            "LANDSAT_PRODUCT_ID",
            "SPACECRAFT_ID",
        ],
    },
    "Landsat-9": {
        "collection_id": "LANDSAT/LC09/C02/T1_L2",
        "properties": [
            "system:index",
            "system:time_start",
            "CLOUD_COVER",
            "WRS_PATH",
            "WRS_ROW",
            "LANDSAT_PRODUCT_ID",
            "SPACECRAFT_ID",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search all Sentinel-2 and Landsat scenes acquired during "
            "the dates represented by the controlled-release intervals, "
            "then retain only exact timestamp overlaps."
        )
    )
    parser.add_argument(
        "--project",
        default="methane-release-gee",
        help="Google Earth Engine project ID.",
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=2000.0,
        help="Search radius around the controlled-release coordinate.",
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        choices=list(COLLECTIONS),
        default=list(COLLECTIONS),
        help="Sensors to search.",
    )
    return parser.parse_args()


def initialize_earth_engine(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine 初始化失敗。先執行 `earthengine authenticate`，"
            f"並確認 project={project!r} 可用。原始錯誤：{exc}"
        ) from exc


def load_intervals() -> pd.DataFrame:
    if not INTERVALS_PATH.exists():
        raise FileNotFoundError(f"找不到：{INTERVALS_PATH}")

    df = pd.read_csv(INTERVALS_PATH, low_memory=False)

    required = [
        "release_start_utc",
        "release_end_utc",
        "release_rate_kg_h",
        "site",
        "lat",
        "lon",
        "release_interval_id",
    ]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"309 檔案缺少欄位：{missing}")

    df["release_start_utc"] = pd.to_datetime(
        df["release_start_utc"], utc=True, errors="coerce"
    )
    df["release_end_utc"] = pd.to_datetime(
        df["release_end_utc"], utc=True, errors="coerce"
    )
    df["release_rate_kg_h"] = pd.to_numeric(
        df["release_rate_kg_h"], errors="coerce"
    )
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.dropna(
        subset=[
            "release_start_utc",
            "release_end_utc",
            "release_rate_kg_h",
            "site",
            "lat",
            "lon",
        ]
    ).copy()

    df = df[
        df["release_end_utc"] > df["release_start_utc"]
    ].copy()

    # Remove repeated copies of the same operation-log interval.
    df = df.sort_values(
        ["rate_priority", "release_duration_minutes"],
        ascending=[True, True],
        na_position="last",
    ).drop_duplicates(
        subset=[
            "site",
            "release_start_utc",
            "release_end_utc",
            "release_rate_kg_h",
            "lat",
            "lon",
        ],
        keep="first",
    )

    # One query is needed for every site/date represented by an interval.
    df["query_date"] = df["release_start_utc"].dt.strftime("%Y-%m-%d")

    return df.reset_index(drop=True)


def image_collection_metadata(
    collection_id: str,
    properties: list[str],
    latitude: float,
    longitude: float,
    date_text: str,
    buffer_m: float,
) -> list[dict]:
    start = ee.Date(date_text)
    end = start.advance(1, "day")

    geometry = ee.Geometry.Point([longitude, latitude]).buffer(buffer_m)

    collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(geometry)
        .filterDate(start, end)
        .sort("system:time_start")
    )

    image_list = collection.toList(collection.size())

    def to_dictionary(image_object):
        image = ee.Image(image_object)
        base = {
            "scene_id": image.id(),
            "collection_id": collection_id,
        }
        for prop in properties:
            base[prop] = image.get(prop)
        return ee.Dictionary(base)

    result = image_list.map(to_dictionary).getInfo()
    return result or []


def build_candidate_catalog(
    intervals: pd.DataFrame,
    sensors: list[str],
    buffer_m: float,
) -> pd.DataFrame:
    query_table = (
        intervals.groupby(["site", "query_date"], as_index=False)
        .agg(
            latitude=("lat", "median"),
            longitude=("lon", "median"),
            interval_count=("release_interval_id", "count"),
        )
        .sort_values(["site", "query_date"])
    )

    rows: list[dict] = []
    total_queries = len(query_table) * len(sensors)
    query_number = 0

    for _, query in query_table.iterrows():
        for sensor in sensors:
            query_number += 1
            config = COLLECTIONS[sensor]

            print(
                f"[{query_number}/{total_queries}] "
                f"{sensor} | {query['site']} | {query['query_date']}"
            )

            try:
                scenes = image_collection_metadata(
                    collection_id=config["collection_id"],
                    properties=config["properties"],
                    latitude=float(query["latitude"]),
                    longitude=float(query["longitude"]),
                    date_text=str(query["query_date"]),
                    buffer_m=buffer_m,
                )
            except Exception as exc:
                print(f"  QUERY ERROR: {type(exc).__name__}: {exc}")
                rows.append(
                    {
                        "sensor": sensor,
                        "site": query["site"],
                        "query_date": query["query_date"],
                        "query_latitude": query["latitude"],
                        "query_longitude": query["longitude"],
                        "query_status": f"error:{type(exc).__name__}",
                        "query_error": str(exc),
                    }
                )
                continue

            if not scenes:
                rows.append(
                    {
                        "sensor": sensor,
                        "site": query["site"],
                        "query_date": query["query_date"],
                        "query_latitude": query["latitude"],
                        "query_longitude": query["longitude"],
                        "query_status": "no_scene",
                    }
                )
                continue

            for scene in scenes:
                timestamp_ms = scene.get("system:time_start")
                acquisition_time = (
                    pd.to_datetime(timestamp_ms, unit="ms", utc=True)
                    if timestamp_ms is not None
                    else pd.NaT
                )

                row = {
                    "sensor": sensor,
                    "site": query["site"],
                    "query_date": query["query_date"],
                    "query_latitude": query["latitude"],
                    "query_longitude": query["longitude"],
                    "query_status": "scene_found",
                    "scene_id": scene.get("scene_id"),
                    "collection_id": scene.get("collection_id"),
                    "acquisition_time_utc": acquisition_time,
                }

                for prop in config["properties"]:
                    if prop == "system:time_start":
                        continue
                    row[prop.replace("system:", "system_")] = scene.get(prop)

                rows.append(row)

    candidates = pd.DataFrame(rows)

    if len(candidates):
        candidates = candidates.drop_duplicates(
            subset=[
                "sensor",
                "site",
                "scene_id",
                "acquisition_time_utc",
            ],
            keep="first",
        )

    return candidates


def build_exact_overlaps(
    intervals: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    scenes = candidates[
        candidates["query_status"] == "scene_found"
    ].copy()

    if scenes.empty:
        return pd.DataFrame()

    scenes["acquisition_time_utc"] = pd.to_datetime(
        scenes["acquisition_time_utc"], utc=True, errors="coerce"
    )
    scenes = scenes.dropna(subset=["acquisition_time_utc"])

    merged = scenes.merge(
        intervals,
        on=["site", "query_date"],
        how="inner",
        suffixes=("_scene", "_release"),
    )

    exact = merged[
        (merged["release_start_utc"] <= merged["acquisition_time_utc"])
        & (merged["acquisition_time_utc"] < merged["release_end_utc"])
    ].copy()

    if exact.empty:
        return exact

    exact["seconds_after_release_start"] = (
        exact["acquisition_time_utc"] - exact["release_start_utc"]
    ).dt.total_seconds()

    exact["seconds_before_release_end"] = (
        exact["release_end_utc"] - exact["acquisition_time_utc"]
    ).dt.total_seconds()

    exact["physical_release_gt"] = (
        exact["release_rate_kg_h"] > 0
    ).astype(int)

    exact["ground_truth_status"] = np.where(
        exact["physical_release_gt"] == 1,
        "confirmed_release",
        "confirmed_no_release",
    )

    exact = exact.drop_duplicates(
        subset=[
            "sensor",
            "scene_id",
            "release_interval_id",
        ],
        keep="first",
    ).sort_values(
        ["sensor", "site", "acquisition_time_utc"]
    )

    preferred = [
        "sensor",
        "site",
        "scene_id",
        "collection_id",
        "acquisition_time_utc",
        "release_interval_id",
        "release_start_utc",
        "release_end_utc",
        "seconds_after_release_start",
        "seconds_before_release_end",
        "release_rate_kg_h",
        "physical_release_gt",
        "ground_truth_status",
        "lat",
        "lon",
        "CLOUDY_PIXEL_PERCENTAGE",
        "CLOUD_COVER",
        "MGRS_TILE",
        "WRS_PATH",
        "WRS_ROW",
        "PRODUCT_ID",
        "LANDSAT_PRODUCT_ID",
        "SPACECRAFT_NAME",
        "SPACECRAFT_ID",
        "source_file",
        "source_sheet",
        "release_rate_source",
        "emission_bin",
        "strict_interval_candidate",
    ]

    existing = [column for column in preferred if column in exact.columns]
    remaining = [column for column in exact.columns if column not in existing]

    return exact[existing + remaining]


def write_summary(
    intervals: pd.DataFrame,
    candidates: pd.DataFrame,
    overlaps: pd.DataFrame,
    sensors: list[str],
) -> None:
    lines = [
        "Exact satellite / controlled-release overlap expansion",
        "=" * 72,
        f"Unique cleaned release intervals: {len(intervals)}",
        f"Sites with coordinates: {intervals['site'].nunique()}",
        f"Sensors searched: {', '.join(sensors)}",
        "",
        "Candidate scene status:",
        candidates["query_status"].value_counts(dropna=False).to_string(),
        "",
        f"Exact overlaps found: {len(overlaps)}",
    ]

    if len(overlaps):
        lines.extend(
            [
                "",
                "Exact overlaps by sensor:",
                overlaps["sensor"].value_counts().to_string(),
                "",
                "Exact overlaps by sensor and physical GT:",
                overlaps.groupby(
                    ["sensor", "ground_truth_status"]
                ).size().reset_index(name="rows").to_string(index=False),
                "",
                "Exact overlaps by site:",
                overlaps.groupby(
                    ["sensor", "site"]
                ).size().reset_index(name="rows").to_string(index=False),
                "",
                "Emission-rate summary by sensor:",
                overlaps.groupby("sensor")["release_rate_kg_h"]
                .describe()
                .to_string(),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "No exact timestamp overlaps were found.",
                "This is a valid result: a same-day scene is not sufficient",
                "unless its acquisition timestamp falls inside the release interval.",
            ]
        )

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


def main() -> None:
    args = parse_args()
    initialize_earth_engine(args.project)

    intervals = load_intervals()

    print(
        f"Cleaned intervals: {len(intervals)} | "
        f"sites: {intervals['site'].nunique()} | "
        f"site-days: {intervals[['site', 'query_date']].drop_duplicates().shape[0]}"
    )

    candidates = build_candidate_catalog(
        intervals=intervals,
        sensors=args.sensors,
        buffer_m=args.buffer_m,
    )
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(CANDIDATES_PATH, index=False)

    overlaps = build_exact_overlaps(intervals, candidates)
    overlaps.to_csv(OVERLAPS_PATH, index=False)

    write_summary(
        intervals=intervals,
        candidates=candidates,
        overlaps=overlaps,
        sensors=args.sensors,
    )

    print("\nCreated:")
    print(CANDIDATES_PATH)
    print(OVERLAPS_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
