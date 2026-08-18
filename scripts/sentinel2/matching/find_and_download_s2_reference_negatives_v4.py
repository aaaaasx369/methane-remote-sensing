#!/usr/bin/env python3
"""
Search and download matched-reference Sentinel-2 patches (SCL fallback v3) for three selected
MethaneAIR observational plume sites.

Inputs
------
outputs/512_selected_methaneair_positive_manifest_v2.csv
outputs/15_methaneair_s2_landsat_availability.csv

Default sites
-------------
MethaneAIR_site_073
MethaneAIR_site_102
MethaneAIR_site_120

Scientific meaning of label 0
-----------------------------
These rows are "no known plume reference" images, not confirmed no-emission
ground truth. The script excludes known local MethaneAIR plume dates and matches
location, season, patch footprint, and image quality.

Outputs
-------
outputs/515_s2_negative_candidates_v1.csv
outputs/516_s2_negative_selected_v1.csv
outputs/517_s2_negative_manifest_v1.csv
outputs/518_s2_negative_download_report_v1.txt
patches/s2_matched_negatives_v1/<site_id>/*.tif
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import ee
except ImportError as exc:
    raise SystemExit(
        "Missing earthengine-api. Install it with:\n"
        "python -m pip install earthengine-api"
    ) from exc

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "Missing requests. Install it with:\npython -m pip install requests"
    ) from exc

try:
    import rasterio
except ImportError as exc:
    raise SystemExit(
        "Missing rasterio. Install it with:\npython -m pip install rasterio"
    ) from exc


EARTH_RADIUS_KM = 6371.0088
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

SITE_ALIASES = ("site_id", "site", "candidate_site_id")
EVENT_ALIASES = ("event_id_canonical", "event_id", "sample_id", "plume_id")
LAT_ALIASES = (
    "site_centroid_latitude", "latitude_canonical", "latitude", "lat"
)
LON_ALIASES = (
    "site_centroid_longitude", "longitude_canonical", "longitude", "lon", "lng"
)
EVENT_TIME_ALIASES = (
    "datetime_utc_canonical", "methaneair_time_utc", "datetime_utc",
    "event_time_utc"
)
S2_TIME_ALIASES = (
    "s2_acquisition_time_utc", "acquisition_time_utc", "s2_time_utc",
    "scene_time_utc"
)
SCENE_ALIASES = ("scene_id", "s2_scene_id", "system_index", "image_id")
PATH_ALIASES = (
    "resolved_patch_path", "patch_path", "relative_path", "file_path", "filename"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build matched-reference Sentinel-2 negatives for selected MethaneAIR sites."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/Users/happydoraaa/methane_release_project"),
    )
    parser.add_argument(
        "--positive-manifest",
        type=Path,
        default=Path("outputs/512_selected_methaneair_positive_manifest_v2.csv"),
    )
    parser.add_argument(
        "--all-events",
        type=Path,
        default=Path("outputs/15_methaneair_s2_landsat_availability.csv"),
    )
    parser.add_argument(
        "--ee-project",
        default=os.environ.get("EE_PROJECT", ""),
        help="Google Earth Engine Cloud project; defaults to EE_PROJECT.",
    )
    parser.add_argument(
        "--selected-sites",
        nargs="+",
        default=[
            "MethaneAIR_site_073",
            "MethaneAIR_site_102",
            "MethaneAIR_site_120",
        ],
    )
    parser.add_argument(
        "--negatives-per-site",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=5,
        help="Retain this many ranked candidates per desired negative before download.",
    )
    parser.add_argument(
        "--year-offsets",
        nargs="+",
        type=int,
        default=[-2, -1, 0, 1, 2],
        help="Search the same season in these year offsets around every positive date.",
    )
    parser.add_argument(
        "--season-window-days",
        type=int,
        default=50,
        help="Half-width of each same-season search window.",
    )
    parser.add_argument(
        "--exclude-days",
        type=int,
        default=14,
        help="Exclude scenes this close to any known local MethaneAIR plume event.",
    )
    parser.add_argument(
        "--local-event-radius-km",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-cloud-metadata",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--cloud-score-threshold",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--minimum-clear-fraction",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--default-patch-half-size-m",
        type=float,
        default=640.0,
        help="Used only when reference raster footprint cannot be inferred.",
    )
    parser.add_argument(
        "--scale-m",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Create candidate and selected tables without downloading GeoTIFFs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.3,
    )
    return parser.parse_args()


def absolute(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def first_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    lower = {str(column).strip().lower(): column for column in df.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    values = [lat1, lon1, lat2, lon2]
    if any(pd.isna(value) for value in values):
        return np.nan
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def circular_day_distance(day_a: int, day_b: int) -> int:
    delta = abs(int(day_a) - int(day_b))
    return min(delta, 366 - delta)


def parse_tile(value: object) -> str:
    match = re.search(r"(?:^|[_-])(T[0-9]{2}[A-Z]{3})(?:[_-]|$)", str(value))
    return match.group(1) if match else ""


def normalize_s2_scene_id(value: object) -> str:
    """Recover the original Sentinel-2 system:index after ImageCollection.merge().

    Earth Engine prefixes image IDs with strings such as ``1_`` and ``2_``
    when collections are merged repeatedly.  The true S2 index is the final
    timestamp/tile token, for example:
    20221008T175229_20221008T175614_T13TEE
    """
    text = str(value).strip()
    match = re.search(
        r"(\d{8}T\d{6}_\d{8}T\d{6}_T\d{2}[A-Z]{3})$",
        text,
    )
    return match.group(1) if match else text


def initialize_ee(project: str) -> None:
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:
        raise SystemExit(
            "Earth Engine initialization failed.\n"
            "Run:\n"
            "  earthengine authenticate\n"
            "  export EE_PROJECT='your-google-cloud-project-id'\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def canonicalize_positive(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "site_id": SITE_ALIASES,
        "event_id": EVENT_ALIASES,
        "latitude": LAT_ALIASES,
        "longitude": LON_ALIASES,
        "event_time_utc": EVENT_TIME_ALIASES,
        "s2_time_utc": S2_TIME_ALIASES,
        "scene_id": SCENE_ALIASES,
        "patch_path": PATH_ALIASES,
    }
    out = df.copy()
    for target, aliases in mapping.items():
        column = first_column(out, aliases)
        out[target] = out[column] if column is not None else np.nan

    required = ("site_id", "event_id", "latitude", "longitude")
    missing = [column for column in required if out[column].isna().all()]
    if missing:
        raise SystemExit(
            "Positive manifest lacks required data: "
            + ", ".join(missing)
            + f"\nAvailable columns: {list(df.columns)}"
        )

    out["site_id"] = normalize_text(out["site_id"])
    out["event_id"] = normalize_text(out["event_id"])
    out["scene_id"] = normalize_text(out["scene_id"]).fillna("")
    out["patch_path"] = normalize_text(out["patch_path"]).fillna("")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["event_time_utc"] = pd.to_datetime(
        out["event_time_utc"], errors="coerce", utc=True
    )
    out["s2_time_utc"] = pd.to_datetime(
        out["s2_time_utc"], errors="coerce", utc=True
    )
    return out


def canonicalize_events(df: pd.DataFrame) -> pd.DataFrame:
    event_col = first_column(df, EVENT_ALIASES)
    lat_col = first_column(df, ("lat", "latitude", "source_latitude"))
    lon_col = first_column(df, ("lon", "longitude", "lng", "source_longitude"))
    time_col = first_column(
        df, ("datetime_utc", "event_time_utc", "acquisition_time_utc")
    )
    if None in (event_col, lat_col, lon_col, time_col):
        raise SystemExit(
            "All-events table must contain event ID, latitude, longitude, and time."
        )

    out = pd.DataFrame(
        {
            "event_id": normalize_text(df[event_col]),
            "latitude": pd.to_numeric(df[lat_col], errors="coerce"),
            "longitude": pd.to_numeric(df[lon_col], errors="coerce"),
            "event_time_utc": pd.to_datetime(
                df[time_col], errors="coerce", utc=True
            ),
        }
    )
    return out.dropna().drop_duplicates().reset_index(drop=True)


def infer_patch_half_size_m(site_rows: pd.DataFrame, default_value: float) -> float:
    for value in site_rows["patch_path"].astype(str):
        path = Path(value).expanduser()
        if not path.exists():
            continue
        try:
            with rasterio.open(path) as src:
                x_resolution = abs(float(src.transform.a))
                y_resolution = abs(float(src.transform.e))
                width_m = src.width * x_resolution
                height_m = src.height * y_resolution
                half_size = max(width_m, height_m) / 2.0
                if 100 <= half_size <= 5000:
                    return float(half_size)
        except Exception:
            continue
    return float(default_value)


def build_local_event_dates(
    all_events: pd.DataFrame,
    latitude: float,
    longitude: float,
    radius_km: float,
) -> list[pd.Timestamp]:
    local = all_events.copy()
    local["distance_km"] = local.apply(
        lambda row: haversine_km(
            latitude, longitude, row["latitude"], row["longitude"]
        ),
        axis=1,
    )
    local = local[local["distance_km"] <= radius_km]
    return sorted(local["event_time_utc"].dropna().tolist())


def merge_date_windows(
    collection: ee.ImageCollection,
    positive_dates: list[pd.Timestamp],
    year_offsets: list[int],
    half_window_days: int,
) -> ee.ImageCollection:
    merged = ee.ImageCollection([])
    for positive_date in positive_dates:
        naive = positive_date.tz_convert("UTC").tz_localize(None)
        for offset in year_offsets:
            target_year = naive.year + int(offset)
            try:
                center = pd.Timestamp(
                    year=target_year,
                    month=naive.month,
                    day=naive.day,
                    tz="UTC",
                )
            except ValueError:
                center = pd.Timestamp(
                    year=target_year,
                    month=naive.month,
                    day=28,
                    tz="UTC",
                )
            start = center - pd.Timedelta(days=half_window_days)
            end = center + pd.Timedelta(days=half_window_days + 1)
            merged = merged.merge(
                collection.filterDate(start.isoformat(), end.isoformat())
            )
    return merged.distinct("system:index")


def search_site_candidates(
    site_id: str,
    latitude: float,
    longitude: float,
    positive_dates: list[pd.Timestamp],
    positive_scene_ids: set[str],
    local_event_dates: list[pd.Timestamp],
    half_size_m: float,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    point = ee.Geometry.Point([float(longitude), float(latitude)])
    region = point.buffer(float(half_size_m)).bounds()

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filter(
            ee.Filter.lte(
                "CLOUDY_PIXEL_PERCENTAGE", float(args.max_cloud_metadata)
            )
        )
    )
    # Use Sentinel-2's own SCL band instead of Cloud Score+ linking.
    # This avoids Python API compatibility problems with linkCollection().
    windowed = merge_date_windows(
        s2,
        positive_dates=positive_dates,
        year_offsets=args.year_offsets,
        half_window_days=args.season_window_days,
    )

    def image_to_feature(image):
        # Older Earth Engine Python clients may pass a generic ee.Element to
        # ImageCollection.map(). Cast it explicitly before image operations.
        image = ee.Image(image)
        scl = image.select(["SCL"])
        # Conservative usable surface classes: dark pixels (2), vegetation (4),
        # bare soil (5), and water (6). remap() avoids method dispatch on a
        # generic Element in older Python clients.
        usable = scl.remap(
            [2, 4, 5, 6],
            [1, 1, 1, 1],
            0,
        ).rename(["usable"])
        clear_fraction = usable.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=60,
            bestEffort=True,
            maxPixels=1_000_000,
        ).get("usable")
        return ee.Feature(
            None,
            {
                "scene_id": image.get("system:index"),
                "system_time_start": image.get("system:time_start"),
                "cloudy_pixel_percentage": image.get(
                    "CLOUDY_PIXEL_PERCENTAGE"
                ),
                "mgrs_tile": image.get("MGRS_TILE"),
                "spacecraft_name": image.get("SPACECRAFT_NAME"),
                "clear_fraction": clear_fraction,
            },
        )

    info = ee.FeatureCollection(windowed.map(image_to_feature)).getInfo()
    features = info.get("features", []) if isinstance(info, dict) else []

    rows = []
    positive_day_of_year = [date.dayofyear for date in positive_dates]

    for feature in features:
        props = feature.get("properties", {})
        raw_scene_id = str(props.get("scene_id", ""))
        scene_id = normalize_s2_scene_id(raw_scene_id)
        scene_time = pd.to_datetime(
            props.get("system_time_start"),
            unit="ms",
            errors="coerce",
            utc=True,
        )
        if pd.isna(scene_time):
            continue

        # Exclude scenes already used as positives.
        if scene_id in positive_scene_ids:
            continue

        # Exclude dates close to any known local MethaneAIR plume event.
        nearest_event_days = (
            min(
                abs((scene_time - event_time).total_seconds()) / 86400.0
                for event_time in local_event_dates
            )
            if local_event_dates
            else np.inf
        )
        if nearest_event_days <= args.exclude_days:
            continue

        clear_fraction = pd.to_numeric(
            props.get("clear_fraction"), errors="coerce"
        )
        if pd.isna(clear_fraction) or clear_fraction < args.minimum_clear_fraction:
            continue

        seasonal_distance_days = (
            min(
                circular_day_distance(
                    scene_time.dayofyear, positive_day
                )
                for positive_day in positive_day_of_year
            )
            if positive_day_of_year
            else np.nan
        )

        time_distance_days = (
            min(
                abs((scene_time - positive_time).total_seconds()) / 86400.0
                for positive_time in positive_dates
            )
            if positive_dates
            else np.nan
        )

        cloud_metadata = pd.to_numeric(
            props.get("cloudy_pixel_percentage"), errors="coerce"
        )

        # Lower score is better. Season matching is dominant, then local
        # clear fraction and scene cloud metadata.
        rank_score = (
            float(seasonal_distance_days)
            + 30.0 * (1.0 - float(clear_fraction))
            + 0.10 * float(cloud_metadata if pd.notna(cloud_metadata) else 100)
        )

        rows.append(
            {
                "site_id": site_id,
                "latitude": latitude,
                "longitude": longitude,
                "s2_scene_id": scene_id,
                "s2_time_utc": scene_time,
                "mgrs_tile": props.get("mgrs_tile", ""),
                "spacecraft_name": props.get("spacecraft_name", ""),
                "cloudy_pixel_percentage": cloud_metadata,
                "clear_fraction": float(clear_fraction),
                "nearest_known_local_plume_days": float(nearest_event_days),
                "seasonal_distance_days": int(seasonal_distance_days),
                "nearest_positive_time_days": float(time_distance_days),
                "candidate_rank_score": float(rank_score),
                "patch_half_size_m": half_size_m,
            }
        )

    return rows


def select_candidates(
    candidates: pd.DataFrame,
    per_site: int,
    multiplier: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    selected_parts = []
    retain_count = max(per_site, per_site * multiplier)

    for site_id, group in candidates.groupby("site_id"):
        group = (
            group.sort_values(
                [
                    "candidate_rank_score",
                    "clear_fraction",
                    "cloudy_pixel_percentage",
                ],
                ascending=[True, False, True],
            )
            .drop_duplicates(subset=["s2_scene_id"])
            .copy()
        )

        # Avoid choosing several scenes from essentially the same date.
        chosen = []
        chosen_dates: list[pd.Timestamp] = []
        for _, row in group.iterrows():
            scene_date = pd.Timestamp(row["s2_time_utc"])
            if any(
                abs((scene_date - prior).total_seconds()) < 4 * 86400
                for prior in chosen_dates
            ):
                continue
            chosen.append(row)
            chosen_dates.append(scene_date)
            if len(chosen) >= retain_count:
                break

        if chosen:
            selected_parts.append(pd.DataFrame(chosen))

    if not selected_parts:
        return pd.DataFrame(columns=candidates.columns)
    return pd.concat(selected_parts, ignore_index=True)


def download_scene(
    scene_id: str,
    latitude: float,
    longitude: float,
    half_size_m: float,
    output_path: Path,
    scale_m: float,
    cloud_threshold: float,
    overwrite: bool,
) -> tuple[bool, str]:
    if output_path.exists() and not overwrite:
        return True, "already_exists"

    scene_id = normalize_s2_scene_id(scene_id)

    matching = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filter(ee.Filter.eq("system:index", scene_id))
    )
    if int(matching.size().getInfo()) == 0:
        return False, f"Sentinel-2 scene not found after ID normalization: {scene_id}"

    image = ee.Image(matching.first())
    scl = image.select(["SCL"])
    clear_mask = scl.remap(
        [2, 4, 5, 6],
        [1, 1, 1, 1],
        0,
    ).rename(["clear_mask"])
    export_image = image.select(BANDS).updateMask(clear_mask).toFloat()

    region = (
        ee.Geometry.Point([float(longitude), float(latitude)])
        .buffer(float(half_size_m))
        .bounds()
    )

    params = {
        "name": output_path.stem,
        "bands": BANDS,
        "region": region,
        "scale": float(scale_m),
        "format": "GEO_TIFF",
        "filePerBand": False,
    }

    last_error = ""
    for attempt in range(1, 4):
        try:
            url = export_image.getDownloadURL(params)
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            content = response.content

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if content[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    tif_names = [
                        name for name in archive.namelist()
                        if name.lower().endswith((".tif", ".tiff"))
                    ]
                    if not tif_names:
                        raise RuntimeError("Download ZIP contains no GeoTIFF.")
                    tif_bytes = archive.read(tif_names[0])
                    output_path.write_bytes(tif_bytes)
            else:
                output_path.write_bytes(content)

            with rasterio.open(output_path) as src:
                if src.count < 6:
                    raise RuntimeError(
                        f"Downloaded raster has {src.count} bands, expected 6."
                    )
                array = src.read()
                valid = np.isfinite(array) & (array != 0)
                if not valid.any():
                    raise RuntimeError("Downloaded raster has no valid non-zero values.")

            return True, "downloaded"

        except Exception as exc:
            last_error = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            if output_path.exists():
                output_path.unlink()
            time.sleep(attempt * 2)

    return False, last_error


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    positive_path = absolute(root, args.positive_manifest)
    all_events_path = absolute(root, args.all_events)
    outputs = root / "outputs"
    patch_root = root / "patches" / "s2_matched_negatives_v1"
    outputs.mkdir(parents=True, exist_ok=True)
    patch_root.mkdir(parents=True, exist_ok=True)

    if not positive_path.exists():
        raise SystemExit(f"Positive manifest not found: {positive_path}")
    if not all_events_path.exists():
        raise SystemExit(f"All-events table not found: {all_events_path}")

    positive = canonicalize_positive(pd.read_csv(positive_path))
    all_events = canonicalize_events(pd.read_csv(all_events_path))

    positive = positive[
        positive["site_id"].astype(str).isin(args.selected_sites)
    ].copy()
    if positive.empty:
        raise SystemExit("No selected sites were found in the positive manifest.")

    initialize_ee(args.ee_project)

    candidate_rows = []
    site_config_rows = []

    for site_id, site_rows in positive.groupby("site_id"):
        latitude = float(site_rows["latitude"].median())
        longitude = float(site_rows["longitude"].median())

        positive_dates = sorted(
            set(
                site_rows["s2_time_utc"].dropna().tolist()
                or site_rows["event_time_utc"].dropna().tolist()
            )
        )
        if not positive_dates:
            raise SystemExit(f"{site_id}: no positive date could be determined.")

        positive_scene_ids = set(
            value for value in site_rows["scene_id"].astype(str)
            if value and value.lower() not in {"nan", "none", "<na>"}
        )
        half_size_m = infer_patch_half_size_m(
            site_rows, args.default_patch_half_size_m
        )
        local_event_dates = build_local_event_dates(
            all_events,
            latitude=latitude,
            longitude=longitude,
            radius_km=args.local_event_radius_km,
        )

        rows = search_site_candidates(
            site_id=site_id,
            latitude=latitude,
            longitude=longitude,
            positive_dates=positive_dates,
            positive_scene_ids=positive_scene_ids,
            local_event_dates=local_event_dates,
            half_size_m=half_size_m,
            args=args,
        )
        candidate_rows.extend(rows)
        site_config_rows.append(
            {
                "site_id": site_id,
                "latitude": latitude,
                "longitude": longitude,
                "positive_scene_count": len(positive_scene_ids),
                "positive_date_count": len(positive_dates),
                "known_local_event_count": len(local_event_dates),
                "patch_half_size_m": half_size_m,
                "candidate_count": len(rows),
            }
        )
        print(f"{site_id}: {len(rows)} acceptable candidate scenes", flush=True)

    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["site_id", "candidate_rank_score"]
        ).reset_index(drop=True)

    candidates_path = outputs / "515_s2_negative_candidates_v1.csv"
    candidates.to_csv(candidates_path, index=False)

    selected = select_candidates(
        candidates,
        per_site=args.negatives_per_site,
        multiplier=args.candidate_multiplier,
    )
    selected["download_priority"] = (
        selected.groupby("site_id").cumcount() + 1
        if not selected.empty
        else pd.Series(dtype=int)
    )
    selected["selected_for_download"] = (
        selected["download_priority"] <= args.negatives_per_site
        if not selected.empty
        else pd.Series(dtype=bool)
    )
    selected_path = outputs / "516_s2_negative_selected_v1.csv"
    selected.to_csv(selected_path, index=False)

    manifest_rows = []
    if not args.search_only and not selected.empty:
        download_rows = selected[selected["selected_for_download"]].copy()
        for index, row in download_rows.iterrows():
            site_id = str(row["site_id"])
            scene_id = str(row["s2_scene_id"])
            safe_scene = re.sub(r"[^A-Za-z0-9_.-]+", "_", scene_id)
            filename = f"{site_id}_{safe_scene}_label_0.tif"
            output_path = patch_root / site_id / filename

            success, status = download_scene(
                scene_id=scene_id,
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                half_size_m=float(row["patch_half_size_m"]),
                output_path=output_path,
                scale_m=args.scale_m,
                cloud_threshold=args.cloud_score_threshold,
                overwrite=args.overwrite,
            )
            record = row.to_dict()
            record.update(
                {
                    "sample_id": f"{site_id}_negative_{int(row['download_priority']):02d}",
                    "label": 0,
                    "source_origin": "MethaneAIR_reference",
                    "ground_truth_type": "no_known_plume_reference",
                    "negative_confidence": "reference_only_not_confirmed_zero_emission",
                    "patch_path": str(output_path.resolve()) if success else "",
                    "download_ok": success,
                    "download_status": status,
                }
            )
            manifest_rows.append(record)
            print(
                f"[{index + 1}/{len(download_rows)}] {site_id} "
                f"{scene_id}: {status}",
                flush=True,
            )
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    negative_manifest = pd.DataFrame(manifest_rows)
    negative_manifest_path = outputs / "517_s2_negative_manifest_v1.csv"
    negative_manifest.to_csv(negative_manifest_path, index=False)

    site_config = pd.DataFrame(site_config_rows)
    selected_summary = (
        selected.groupby("site_id")
        .agg(
            ranked_candidates=("s2_scene_id", "size"),
            selected_for_download=("selected_for_download", "sum"),
            median_clear_fraction=("clear_fraction", "median"),
            median_seasonal_distance_days=("seasonal_distance_days", "median"),
        )
        .reset_index()
        if not selected.empty
        else pd.DataFrame()
    )
    if not selected_summary.empty:
        site_config = site_config.merge(
            selected_summary, on="site_id", how="left"
        )

    if not negative_manifest.empty:
        downloaded_summary = (
            negative_manifest.groupby("site_id")
            .agg(
                downloaded_negatives=("download_ok", "sum"),
                failed_downloads=("download_ok", lambda values: int((~values).sum())),
            )
            .reset_index()
        )
        site_config = site_config.merge(
            downloaded_summary, on="site_id", how="left"
        )

    report_lines = [
        "=" * 112,
        "MATCHED-REFERENCE SENTINEL-2 NEGATIVE SEARCH V1",
        "=" * 112,
        "",
        f"Positive manifest: {positive_path}",
        f"All MethaneAIR events: {all_events_path}",
        f"Selected sites: {', '.join(args.selected_sites)}",
        f"Desired negatives per site: {args.negatives_per_site}",
        f"Known-plume exclusion: ±{args.exclude_days} days within {args.local_event_radius_km:g} km",
        f"Seasonal window: ±{args.season_window_days} days at year offsets {args.year_offsets}",
        f"Minimum clear fraction: {args.minimum_clear_fraction:g}",
        f"SCL usable-surface mask; legacy threshold argument retained but unused: {args.cloud_score_threshold:g}",
        f"Search-only mode: {args.search_only}",
        "",
        "SITE SUMMARY",
        "-" * 112,
        site_config.to_string(index=False),
        "",
        "IMPORTANT INTERPRETATION",
        "-" * 112,
        "1. These label-0 scenes are no-known-plume reference images.",
        "2. They are not confirmed zero-emission ground truth.",
        "3. Every selected scene is spatially matched to the same site.",
        "4. Scenes close to any known local MethaneAIR plume event are excluded.",
        "5. Selection favours similar season, high clear fraction, and low scene-cloud metadata.",
        "",
        "OUTPUTS",
        "-" * 112,
        str(candidates_path),
        str(selected_path),
        str(negative_manifest_path),
        str(patch_root),
    ]
    report_path = outputs / "518_s2_negative_download_report_v1.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nCreated:")
    for path in (
        candidates_path,
        selected_path,
        negative_manifest_path,
        report_path,
    ):
        print(path)

    if args.search_only:
        print("\nSearch-only mode complete. Review output 516 before downloading.")
    else:
        print("\nDownload stage complete. Review output 518.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
