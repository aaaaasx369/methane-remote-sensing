#!/usr/bin/env python3
"""
Download matched-reference Sentinel-2 negatives for the three selected
MethaneAIR sites.

Input:
  outputs/541_selected_methaneair_positive_manifest_v1.csv
  outputs/15_methaneair_s2_landsat_availability.csv

Output:
  outputs/544_methaneair_reference_negative_candidates_v1.csv
  outputs/545_methaneair_reference_negative_selected_v1.csv
  outputs/547_methaneair_reference_negative_manifest_v1.csv
  outputs/546_methaneair_reference_negative_report_v1.txt
  patches/s2_reference_negatives_v1/<site_id>/*.tif

Scientific meaning of label 0:
  no known MethaneAIR plume reference, not confirmed zero emission.
"""
from __future__ import annotations

import argparse
import io
import math
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import ee
import numpy as np
import pandas as pd
import rasterio
import requests

EARTH_RADIUS_KM = 6371.0088
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
LAT_ALIASES = ("lat", "latitude", "source_latitude", "source_lat")
LON_ALIASES = ("lon", "longitude", "lng", "source_longitude", "source_lon")
TIME_ALIASES = ("datetime_utc", "event_time_utc", "timestamp_utc", "acquisition_time_utc")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("/Users/happydoraaa/methane_release_project"))
    p.add_argument("--ee-project", default=os.environ.get("EE_PROJECT", ""))
    p.add_argument("--negatives-per-site", type=int, default=8)
    p.add_argument("--candidate-multiplier", type=int, default=5)
    p.add_argument("--year-offsets", nargs="+", type=int, default=[-2, -1, 0, 1, 2])
    p.add_argument("--season-window-days", type=int, default=50)
    p.add_argument("--exclude-days", type=int, default=14)
    p.add_argument("--local-event-radius-km", type=float, default=1.0)
    p.add_argument("--max-cloud-metadata", type=float, default=60.0)
    p.add_argument("--minimum-clear-fraction", type=float, default=0.80)
    p.add_argument("--patch-half-size-m", type=float, default=640.0)
    p.add_argument("--scale-m", type=float, default=20.0)
    p.add_argument("--search-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--sleep-seconds", type=float, default=0.3)
    return p.parse_args()


def first_col(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def initialize_ee(project: str) -> None:
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception as exc:
        raise SystemExit(
            "Earth Engine initialization failed.\n"
            "Run: earthengine authenticate\n"
            "Then: export EE_PROJECT='your-google-cloud-project-id'\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def circular_day_distance(a, b) -> int:
    da, db = pd.Timestamp(a).dayofyear, pd.Timestamp(b).dayofyear
    direct = abs(da - db)
    return int(min(direct, 366 - direct))


def get_clear_fraction(image: ee.Image, region: ee.Geometry, scale_m: float) -> float:
    scl = image.select("SCL")
    clear = (
        scl.neq(0).And(scl.neq(1)).And(scl.neq(3))
        .And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    )
    value = clear.rename("clear").reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=scale_m,
        maxPixels=1_000_000, bestEffort=True
    ).get("clear").getInfo()
    return float(value) if value is not None else np.nan


def search_site(site_id: str, lat: float, lon: float, positive_times, plume_times, args) -> list[dict]:
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(args.patch_half_size_m).bounds()
    unique = {}
    limit = max(args.negatives_per_site * args.candidate_multiplier, 20)

    for positive_time in positive_times:
        for offset in args.year_offsets:
            center = positive_time + pd.DateOffset(years=offset)
            start = center - pd.Timedelta(days=args.season_window_days)
            end = center + pd.Timedelta(days=args.season_window_days + 1)
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(region)
                .filterDate(start.isoformat(), end.isoformat())
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", args.max_cloud_metadata))
                .sort("CLOUDY_PIXEL_PERCENTAGE")
                .limit(limit)
            )

            def to_feature(image):
                return ee.Feature(None, {
                    "scene_id": image.get("system:index"),
                    "time_start": image.get("system:time_start"),
                    "cloud": image.get("CLOUDY_PIXEL_PERCENTAGE"),
                    "spacecraft": image.get("SPACECRAFT_NAME"),
                    "mgrs_tile": image.get("MGRS_TILE"),
                })

            info = ee.FeatureCollection(collection.map(to_feature)).getInfo()
            for feature in info.get("features", []):
                props = feature.get("properties", {})
                scene_id = str(props.get("scene_id", ""))
                if not scene_id or scene_id in unique:
                    continue
                scene_time = pd.to_datetime(props.get("time_start"), unit="ms", errors="coerce", utc=True)
                if pd.isna(scene_time):
                    continue
                nearest_plume_days = min(
                    [abs((scene_time - t).total_seconds()) / 86400 for t in plume_times],
                    default=np.inf,
                )
                if nearest_plume_days <= args.exclude_days:
                    continue
                image = ee.Image("COPERNICUS/S2_SR_HARMONIZED/" + scene_id)
                try:
                    clear_fraction = get_clear_fraction(image, region, args.scale_m)
                except Exception:
                    continue
                if not np.isfinite(clear_fraction) or clear_fraction < args.minimum_clear_fraction:
                    continue
                unique[scene_id] = {
                    "site_id": site_id, "latitude": lat, "longitude": lon,
                    "s2_scene_id": scene_id, "s2_acquisition_time_utc": scene_time,
                    "cloudy_pixel_percentage": props.get("cloud", np.nan),
                    "clear_fraction": clear_fraction,
                    "spacecraft": props.get("spacecraft", ""),
                    "mgrs_tile": props.get("mgrs_tile", ""),
                    "seasonal_distance_days": min(circular_day_distance(scene_time, t) for t in positive_times),
                    "nearest_known_plume_days": nearest_plume_days,
                    "patch_half_size_m": args.patch_half_size_m,
                }
    return list(unique.values())


def download_scene(scene_id: str, lat: float, lon: float, half_size_m: float, path: Path, scale_m: float, overwrite: bool):
    if path.exists() and not overwrite:
        try:
            with rasterio.open(path) as src:
                if src.count >= 6:
                    return True, "existing_valid_file"
        except Exception:
            pass
    region = ee.Geometry.Point([lon, lat]).buffer(half_size_m).bounds()
    image = ee.Image("COPERNICUS/S2_SR_HARMONIZED/" + scene_id).select(BANDS)
    url = image.getDownloadURL({
        "bands": BANDS, "region": region, "scale": scale_m,
        "format": "GEO_TIFF", "filePerBand": False,
    })
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = response.content
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if not names:
                return False, "zip_without_tif"
            path.write_bytes(zf.read(names[0]))
    else:
        path.write_bytes(content)
    try:
        with rasterio.open(path) as src:
            if src.count < 6 or src.width <= 0 or src.height <= 0:
                return False, "invalid_raster"
    except Exception as exc:
        return False, f"raster_open_failed:{type(exc).__name__}:{exc}"
    return True, "downloaded"


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    out = root / "outputs"
    positive_path = out / "541_selected_methaneair_positive_manifest_v1.csv"
    all_events_path = out / "15_methaneair_s2_landsat_availability.csv"
    for path in (positive_path, all_events_path):
        if not path.exists():
            raise SystemExit(f"Required file missing: {path}")
    initialize_ee(args.ee_project)

    positives = pd.read_csv(positive_path)
    events = pd.read_csv(all_events_path)
    lat_col, lon_col, time_col = first_col(events, LAT_ALIASES), first_col(events, LON_ALIASES), first_col(events, TIME_ALIASES)
    if any(v is None for v in (lat_col, lon_col, time_col)):
        raise SystemExit("All-event table lacks latitude, longitude, or time.")
    events["latitude_c"] = pd.to_numeric(events[lat_col], errors="coerce")
    events["longitude_c"] = pd.to_numeric(events[lon_col], errors="coerce")
    events["time_c"] = pd.to_datetime(events[time_col], errors="coerce", utc=True)
    events = events.dropna(subset=["latitude_c", "longitude_c", "time_c"])
    positives["event_time_utc"] = pd.to_datetime(positives["event_time_utc"], errors="coerce", utc=True)
    positives["s2_acquisition_time_utc"] = pd.to_datetime(positives["s2_acquisition_time_utc"], errors="coerce", utc=True)

    all_candidates = []
    site_rows = []
    for site_id, group in positives.groupby("site_id"):
        lat = float(group["latitude"].median())
        lon = float(group["longitude"].median())
        positive_times = list(group["s2_acquisition_time_utc"].combine_first(group["event_time_utc"]).dropna().drop_duplicates())
        local = events[events.apply(lambda r: haversine_km(lat, lon, r["latitude_c"], r["longitude_c"]) <= args.local_event_radius_km, axis=1)]
        plume_times = list(local["time_c"].dropna().drop_duplicates())
        rows = search_site(str(site_id), lat, lon, positive_times, plume_times, args)
        all_candidates.extend(rows)
        site_rows.append({
            "site_id": site_id, "latitude": lat, "longitude": lon,
            "positive_scenes": group["scene_id"].nunique(),
            "known_local_plume_dates": len(plume_times),
            "reference_candidates": len(rows),
        })
        print(f"{site_id}: {len(rows)} candidate reference scenes", flush=True)

    candidates = pd.DataFrame(all_candidates)
    candidate_path = out / "544_methaneair_reference_negative_candidates_v1.csv"
    candidates.to_csv(candidate_path, index=False)
    if candidates.empty:
        raise SystemExit("No reference candidates found. Try --minimum-clear-fraction 0.65 or a wider season window.")

    candidates = candidates.sort_values(
        ["site_id", "seasonal_distance_days", "cloudy_pixel_percentage", "clear_fraction"],
        ascending=[True, True, True, False], na_position="last"
    )
    selected = candidates.groupby("site_id", group_keys=False).head(args.negatives_per_site * args.candidate_multiplier).copy()
    selected["download_priority"] = selected.groupby("site_id").cumcount() + 1
    selected["selected_for_download"] = selected["download_priority"].le(args.negatives_per_site)
    selected_path = out / "545_methaneair_reference_negative_selected_v1.csv"
    selected.to_csv(selected_path, index=False)

    manifest_rows = []
    if not args.search_only:
        download_rows = selected[selected["selected_for_download"]]
        patch_root = root / "patches" / "s2_reference_negatives_v1"
        for position, (_, row) in enumerate(download_rows.iterrows(), start=1):
            site_id = str(row["site_id"])
            scene_id = str(row["s2_scene_id"])
            safe_scene = re.sub(r"[^A-Za-z0-9_.-]+", "_", scene_id)
            output_path = patch_root / site_id / f"{site_id}_{safe_scene}_label_0.tif"
            try:
                ok, status = download_scene(
                    scene_id, float(row["latitude"]), float(row["longitude"]),
                    float(row["patch_half_size_m"]), output_path, args.scale_m, args.overwrite
                )
            except Exception as exc:
                ok, status = False, f"{type(exc).__name__}:{exc}"
            record = row.to_dict()
            record.update({
                "sample_id": f"{site_id}_negative_{int(row['download_priority']):03d}",
                "scene_id": scene_id, "label": 0, "source_origin": "MethaneAIR",
                "ground_truth_type": "no_known_plume_reference",
                "benchmark_tier": "exploratory_external_source",
                "negative_confidence": "reference_only_not_confirmed_zero_emission",
                "patch_path": str(output_path.resolve()) if ok else "",
                "download_ok": ok, "download_status": status,
            })
            manifest_rows.append(record)
            print(f"[{position}/{len(download_rows)}] {site_id}: {status}", flush=True)
            time.sleep(args.sleep_seconds)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out / "547_methaneair_reference_negative_manifest_v1.csv"
    manifest.to_csv(manifest_path, index=False)

    summary = pd.DataFrame(site_rows)
    if not manifest.empty:
        downloaded = manifest.groupby("site_id").agg(
            requested=("sample_id", "size"), downloaded=("download_ok", "sum"),
            failed=("download_ok", lambda x: int((~x.astype(bool)).sum()))
        ).reset_index()
        summary = summary.merge(downloaded, on="site_id", how="left")
    report = [
        "=" * 112, "METHANEAIR REFERENCE-NEGATIVE REPORT V1", "=" * 112, "",
        f"Search-only: {args.search_only}", f"Desired negatives per site: {args.negatives_per_site}",
        f"Known-plume exclusion: ±{args.exclude_days} days", f"Minimum clear fraction: {args.minimum_clear_fraction}", "",
        "SITE SUMMARY", "-" * 112, summary.to_string(index=False), "",
        "LIMITATION", "-" * 112,
        "These label-0 scenes are no-known-plume references, not confirmed zero-emission ground truth.", "",
        "OUTPUTS", "-" * 112, str(candidate_path), str(selected_path), str(manifest_path),
    ]
    report_path = out / "546_methaneair_reference_negative_report_v1.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\nCreated:")
    for path in (candidate_path, selected_path, manifest_path, report_path):
        print(" ", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
