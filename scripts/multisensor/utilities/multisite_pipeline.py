#!/usr/bin/env python3
"""
Multisite Sentinel-2 workflow: Steps 1–4

Commands
--------
python multisite_pipeline.py master   --project-root ...
python multisite_pipeline.py audit    --project-root ... [--use-ee]
python multisite_pipeline.py matches  --project-root ... [--use-ee-landsat]
python multisite_pipeline.py features --project-root ...
python multisite_pipeline.py all      --project-root ...

Outputs
-------
outputs/36_multisite_s2_master_table.csv
outputs/37_multisite_s2_availability.csv
outputs/38_cross_sensor_temporal_matches.csv
outputs/39_multisite_s2_features.csv
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject, transform

EPS = 1e-9
EARTH_RADIUS_KM = 6371.0088
BANDS = ("b2", "b3", "b4", "b8", "b11", "b12")

LABEL = ("label", "final_label", "classification_label", "target")
SITE = ("site_id", "site", "site_name", "facility", "location", "release_site")
SOURCE = ("ground_truth_source", "source_origin", "source_group", "source_dataset",
          "dataset_group", "campaign_id", "provenance", "label_source", "data_source")
SCENE = ("scene_id", "s2_scene_id", "system_index", "system:index",
         "image_id", "product_id", "granule_id")
SAMPLE = ("sample_id", "event_id", "observation_id", "patch_id", "id")
TIME = ("acquisition_time_utc", "s2_acquisition_time_utc", "s2_time_utc",
        "datetime_utc", "acquisition_datetime", "scene_time_utc",
        "timestamp_utc", "event_time_utc", "datetime")
PATHS = ("image_path", "resolved_patch_path", "patch_path", "relative_path",
         "file_path", "filepath", "tif_path", "filename")
EMISSION = ("emission_rate_kg_hr", "release_rate_kg_h", "emission_kg_hr",
            "emission_kg_h", "emission_rate_kg_h",
            "matched_positive_release_rate_kg_h")
LAT = ("source_latitude", "latitude", "lat", "site_latitude", "source_lat")
LON = ("source_longitude", "longitude", "lon", "lng", "site_longitude", "source_lon")
EVENT = ("event_id", "release_id", "observation_id", "plume_id", "sample_id")


def first_col(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    mapping = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in mapping:
            return mapping[alias.lower()]
    return None


def norm_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def norm_site(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", norm_text(value).lower())
    stop = {"release", "releases", "stack", "stacks", "site", "facility",
            "controlled", "methane", "source", "test", "station"}
    return " ".join(token for token in text.split() if token not in stop)


def parse_label(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    out = out.mask(text.isin(["positive", "tp", "plume", "release", "yes", "true"]), 1)
    out = out.mask(text.isin(["negative", "tn", "no plume", "no_release", "no", "false"]), 0)
    return out


def resolve_path(root: Path, value: object) -> str:
    text = norm_text(value)
    if not text:
        return ""
    raw = Path(text).expanduser()
    guesses = [
        raw, root / raw, root / "outputs" / raw, root / "patches" / raw,
        root / "images" / raw, root / "data" / raw, root / "downloads" / raw,
        root / "outputs" / raw.name, root / "patches" / raw.name,
        root / "images" / raw.name,
    ]
    for path in guesses:
        if path.exists() and path.is_file():
            return str(path.resolve())
    for folder_name in ("patches", "images", "data", "downloads", "outputs"):
        folder = root / folder_name
        if folder.exists():
            matches = list(folder.rglob(raw.name))
            if len(matches) == 1:
                return str(matches[0].resolve())
    return ""


def haversine(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return np.nan
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def read_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read().astype("float64")
        profile = {
            "transform": src.transform, "crs": src.crs, "nodata": src.nodata,
            "height": src.height, "width": src.width, "bounds": src.bounds,
        }
    if arr.shape[0] < 6:
        raise ValueError("Expected at least 6 bands in B2,B3,B4,B8,B11,B12 order")
    arr = arr[:6]
    if profile["nodata"] is not None:
        arr[arr == profile["nodata"]] = np.nan
    return arr, profile


def valid(arr: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(arr), axis=0) & np.any(arr != 0, axis=0)


def radial_masks(h: int, w: int):
    yy, xx = np.indices((h, w))
    cy, cx = (h - 1) / 2, (w - 1) / 2
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    scale = float(min(h, w))
    return d <= 0.18 * scale, (d >= 0.30 * scale) & (d <= 0.48 * scale)


def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = a + b
    out = np.full_like(a, np.nan, dtype="float64")
    good = np.isfinite(a) & np.isfinite(b) & (np.abs(den) > EPS)
    out[good] = (a[good] - b[good]) / den[good]
    return out


def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype="float64")
    good = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > EPS)
    out[good] = a[good] / b[good]
    return out


def stats(values: np.ndarray, prefix: str) -> dict:
    values = values[np.isfinite(values)]
    if not len(values):
        return {prefix + k: np.nan for k in ("_mean", "_median", "_std", "_p10", "_p90")}
    return {
        prefix + "_mean": float(np.mean(values)),
        prefix + "_median": float(np.median(values)),
        prefix + "_std": float(np.std(values)),
        prefix + "_p10": float(np.percentile(values, 10)),
        prefix + "_p90": float(np.percentile(values, 90)),
    }


def raster_center(path: Path):
    with rasterio.open(path) as src:
        x = (src.bounds.left + src.bounds.right) / 2
        y = (src.bounds.bottom + src.bounds.top) / 2
        if src.crs is None:
            return np.nan, np.nan
        lon, lat = transform(src.crs, CRS.from_epsg(4326), [x], [y])
        return float(lat[0]), float(lon[0])


def quick_background(path: Path):
    try:
        arr, _ = read_raster(path)
        mask = valid(arr)
        _, bg = radial_masks(arr.shape[1], arr.shape[2])
        mask &= bg
        values = nd(arr[3], arr[2])[mask]
        med, sd = float(np.nanmedian(values)), float(np.nanstd(values))
        if med >= 0.30:
            klass = "vegetated"
        elif med >= 0.10:
            klass = "mixed_vegetation"
        elif sd >= 0.12:
            klass = "low_vegetation_heterogeneous"
        else:
            klass = "low_vegetation_uniform"
        return klass, med, sd
    except Exception:
        return "unresolved", np.nan, np.nan


def preferred_manifest(root: Path, explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path
        raise SystemExit("Manifest not found: " + str(path))
    for name in ("548_five_site_multisource_manifest_v1.csv",
                 "530_five_site_master_manifest_v3.csv",
                 "500_multisource_canonical_table_v2.csv"):
        path = root / "outputs" / name
        if path.exists():
            return path
    raise SystemExit("Final five-site manifest not found in outputs/")


def canonical_manifest(raw: pd.DataFrame, root: Path) -> pd.DataFrame:
    lc, sc, pc = first_col(raw, LABEL), first_col(raw, SITE), first_col(raw, PATHS)
    if not all((lc, sc, pc)):
        raise SystemExit("Manifest requires label, site and image-path columns")
    smc, sec = first_col(raw, SAMPLE), first_col(raw, SCENE)
    soc, tc = first_col(raw, SOURCE), first_col(raw, TIME)
    ec, evc = first_col(raw, EMISSION), first_col(raw, EVENT)
    lac, loc = first_col(raw, LAT), first_col(raw, LON)

    out = pd.DataFrame(index=raw.index)
    out["sample_id"] = raw[smc].astype(str) if smc else ["sample_%05d" % i for i in range(len(raw))]
    out["site_id"] = raw[sc].astype(str)
    out["site_normalized"] = out["site_id"].map(norm_site)
    out["scene_id"] = raw[sec].astype(str) if sec else out["sample_id"]
    out["event_id"] = raw[evc].astype(str) if evc else ""
    out["ground_truth_source"] = raw[soc].astype(str) if soc else "unresolved_source"
    out["label"] = parse_label(raw[lc])
    out["acquisition_time_utc"] = pd.to_datetime(raw[tc], errors="coerce", utc=True) if tc else pd.NaT
    out["emission_rate_kg_hr"] = pd.to_numeric(raw[ec], errors="coerce") if ec else np.nan
    out["image_path"] = raw[pc].map(lambda x: resolve_path(root, x))
    out["source_latitude"] = pd.to_numeric(raw[lac], errors="coerce") if lac else np.nan
    out["source_longitude"] = pd.to_numeric(raw[loc], errors="coerce") if loc else np.nan

    for i, row in out.iterrows():
        if (pd.isna(row["source_latitude"]) or pd.isna(row["source_longitude"])) and row["image_path"]:
            try:
                lat, lon = raster_center(Path(row["image_path"]))
                out.at[i, "source_latitude"], out.at[i, "source_longitude"] = lat, lon
            except Exception:
                pass

    out = out[out["label"].isin([0, 1]) & out["site_normalized"].ne("")
              & out["image_path"].astype(str).str.len().gt(0)].copy()
    out["label"] = out["label"].astype(int)
    out["satellite"] = "Sentinel-2"
    out["campaign_id"] = (
        out["ground_truth_source"].astype(str).str.replace(r"\s+", "_", regex=True)
        + "_" + out["acquisition_time_utc"].dt.year.astype("Int64").astype(str)
    )
    out["ground_truth_type"] = np.where(
        out["site_id"].str.startswith("MethaneAIR_site_"),
        np.where(out["label"].eq(1), "observational_plume_positive",
                 "no_known_plume_reference"),
        "controlled_release_status",
    )
    return out.reset_index(drop=True)


def discover_interval_table(root: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        p = p if p.is_absolute() else root / p
        return p if p.exists() else None
    p = root / "outputs" / "309_all_exact_release_intervals_for_s2.csv"
    if p.exists():
        return p
    for p in (root / "outputs").glob("*.csv"):
        try:
            cols = [str(c).lower() for c in pd.read_csv(p, nrows=1).columns]
        except Exception:
            continue
        if any("release_start" in c or "interval_start" in c for c in cols) and \
           any("release_end" in c or "interval_end" in c for c in cols):
            return p
    return None


def canonical_intervals(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    raw = pd.read_csv(path)
    start = first_col(raw, ("release_start_utc", "interval_start_utc", "start_utc", "release_start"))
    end = first_col(raw, ("release_end_utc", "interval_end_utc", "end_utc", "release_end"))
    if not start or not end:
        return pd.DataFrame()
    sc, lac, loc, ec = first_col(raw, SITE), first_col(raw, LAT), first_col(raw, LON), first_col(raw, EMISSION)
    out = pd.DataFrame(index=raw.index)
    out["interval_id"] = ["interval_%05d" % i for i in range(len(raw))]
    out["site_normalized"] = raw[sc].map(norm_site) if sc else ""
    out["latitude"] = pd.to_numeric(raw[lac], errors="coerce") if lac else np.nan
    out["longitude"] = pd.to_numeric(raw[loc], errors="coerce") if loc else np.nan
    out["release_start_utc"] = pd.to_datetime(raw[start], errors="coerce", utc=True)
    out["release_end_utc"] = pd.to_datetime(raw[end], errors="coerce", utc=True)
    out["interval_emission"] = pd.to_numeric(raw[ec], errors="coerce") if ec else np.nan
    return out[out["release_start_utc"].notna() & out["release_end_utc"].notna()].reset_index(drop=True)


def match_interval(row: pd.Series, intervals: pd.DataFrame, radius_km: float):
    empty = {
        "release_interval_id": "", "release_start_utc": pd.NaT,
        "release_end_utc": pd.NaT, "release_interval_match": "not_found",
        "acquisition_inside_release_interval": False, "interval_emission": np.nan,
    }
    if intervals.empty or pd.isna(row["acquisition_time_utc"]):
        return empty
    c = intervals[intervals["site_normalized"].ne("")
                  & intervals["site_normalized"].eq(row["site_normalized"])].copy()
    if c.empty and pd.notna(row["source_latitude"]) and pd.notna(row["source_longitude"]):
        d = intervals.copy()
        d["distance"] = d.apply(lambda x: haversine(
            row["source_latitude"], row["source_longitude"], x["latitude"], x["longitude"]), axis=1)
        c = d[d["distance"] <= radius_km].copy()
    if c.empty:
        return empty
    c["inside"] = (c["release_start_utc"] <= row["acquisition_time_utc"]) & \
                  (c["release_end_utc"] >= row["acquisition_time_utc"])
    c["gap"] = c.apply(lambda x: 0 if x["inside"] else min(
        abs((row["acquisition_time_utc"] - x["release_start_utc"]).total_seconds()),
        abs((row["acquisition_time_utc"] - x["release_end_utc"]).total_seconds())), axis=1)
    x = c.sort_values(["inside", "gap"], ascending=[False, True]).iloc[0]
    return {
        "release_interval_id": x["interval_id"], "release_start_utc": x["release_start_utc"],
        "release_end_utc": x["release_end_utc"],
        "release_interval_match": "inside" if x["inside"] else "nearest_same_site",
        "acquisition_inside_release_interval": bool(x["inside"]),
        "interval_emission": x["interval_emission"],
    }


def discover_wind_table(root: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        p = p if p.is_absolute() else root / p
        return p if p.exists() else None
    scored = []
    for p in (root / "outputs").glob("*.csv"):
        try:
            cols = [str(c).lower() for c in pd.read_csv(p, nrows=1).columns]
        except Exception:
            continue
        if any("wind_speed" in c or c == "windspeed" for c in cols) and \
           any("wind_direction" in c or "wind_dir" in c for c in cols) and \
           any("time" in c or "datetime" in c for c in cols):
            scored.append((int("wind" in p.name.lower()), p))
    return sorted(scored, reverse=True)[0][1] if scored else None


def canonical_wind(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    raw = pd.read_csv(path)
    sp = first_col(raw, ("wind_speed", "wind_speed_m_s", "windspeed", "wind_m_s"))
    dr = first_col(raw, ("wind_direction", "wind_direction_deg", "wind_dir", "wind_bearing"))
    tc = first_col(raw, ("wind_time_utc", "datetime_utc", "timestamp_utc", "time_utc", "datetime"))
    if not all((sp, dr, tc)):
        return pd.DataFrame()
    sc, lac, loc = first_col(raw, SITE), first_col(raw, LAT), first_col(raw, LON)
    out = pd.DataFrame(index=raw.index)
    out["site_normalized"] = raw[sc].map(norm_site) if sc else ""
    out["latitude"] = pd.to_numeric(raw[lac], errors="coerce") if lac else np.nan
    out["longitude"] = pd.to_numeric(raw[loc], errors="coerce") if loc else np.nan
    out["wind_time_utc"] = pd.to_datetime(raw[tc], errors="coerce", utc=True)
    out["wind_speed"] = pd.to_numeric(raw[sp], errors="coerce")
    out["wind_direction"] = pd.to_numeric(raw[dr], errors="coerce")
    return out.dropna(subset=["wind_time_utc", "wind_speed", "wind_direction"]).reset_index(drop=True)


def match_wind(row: pd.Series, wind: pd.DataFrame, max_hours: float, radius_km: float):
    empty = {"wind_time_utc": pd.NaT, "wind_speed": np.nan, "wind_direction": np.nan,
             "wind_time_difference_hours": np.nan, "wind_match_status": "not_found"}
    if wind.empty or pd.isna(row["acquisition_time_utc"]):
        return empty
    c = wind[wind["site_normalized"].ne("")
             & wind["site_normalized"].eq(row["site_normalized"])].copy()
    status = "site_time"
    if c.empty and pd.notna(row["source_latitude"]) and pd.notna(row["source_longitude"]):
        d = wind.copy()
        d["distance"] = d.apply(lambda x: haversine(
            row["source_latitude"], row["source_longitude"], x["latitude"], x["longitude"]), axis=1)
        c = d[d["distance"] <= radius_km].copy()
        status = "coordinate_time"
    if c.empty:
        return empty
    c["dt"] = (c["wind_time_utc"] - row["acquisition_time_utc"]).abs().dt.total_seconds() / 3600
    x = c.sort_values("dt").iloc[0]
    if x["dt"] > max_hours:
        return empty
    return {"wind_time_utc": x["wind_time_utc"], "wind_speed": x["wind_speed"],
            "wind_direction": x["wind_direction"],
            "wind_time_difference_hours": x["dt"], "wind_match_status": status}


def cmd_master(args):
    root = Path(args.project_root).expanduser().resolve()
    outdir = root / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = preferred_manifest(root, args.manifest)
    master = canonical_manifest(pd.read_csv(manifest), root)

    int_path = discover_interval_table(root, args.release_intervals)
    intervals = canonical_intervals(int_path)
    master = pd.concat([master, pd.DataFrame([
        match_interval(row, intervals, args.spatial_match_km) for _, row in master.iterrows()
    ])], axis=1)
    master["emission_rate_kg_hr"] = master["interval_emission"].combine_first(master["emission_rate_kg_hr"])

    wind_path = discover_wind_table(root, args.wind_table)
    wind = canonical_wind(wind_path)
    master = pd.concat([master, pd.DataFrame([
        match_wind(row, wind, args.wind_max_hours, args.spatial_match_km)
        for _, row in master.iterrows()
    ])], axis=1)

    bg = [quick_background(Path(p)) for p in master["image_path"]]
    master[["background_class", "background_ndvi_median", "background_ndvi_std"]] = pd.DataFrame(bg, index=master.index)

    path = outdir / "36_multisite_s2_master_table.csv"
    master.to_csv(path, index=False)
    summary = master.groupby(["site_id", "ground_truth_source"], dropna=False).agg(
        rows=("sample_id", "size"), positive=("label", lambda x: int((x == 1).sum())),
        negative=("label", lambda x: int((x == 0).sum())),
        release_interval_matches=("acquisition_inside_release_interval", "sum"),
        wind_matches=("wind_speed", lambda x: int(x.notna().sum())),
        scenes=("scene_id", "nunique")).reset_index()
    summary.to_csv(outdir / "36_multisite_s2_master_summary.csv", index=False)
    (outdir / "36_multisite_s2_master_report.txt").write_text(
        "MULTISITE S2 MASTER TABLE\n" + "=" * 90 + "\n"
        + "Manifest: " + str(manifest) + "\n"
        + "Release interval table: " + str(int_path or "not found") + "\n"
        + "Wind table: " + str(wind_path or "not found") + "\n"
        + "Rows: %d\nSites: %d\nSources: %d\n\n" % (
            len(master), master["site_id"].nunique(), master["ground_truth_source"].nunique())
        + summary.to_string(index=False)
        + "\n\nMissing interval/wind values are kept missing rather than guessed.\n",
        encoding="utf-8")
    print("Created", path)


# ------------------------------ STEP 2 ---------------------------------------

def local_audit(row: pd.Series) -> dict:
    result = {
        "image_exists": False, "raster_read_ok": False, "band_count": np.nan,
        "width": np.nan, "height": np.nan, "crs": "", "b11_complete": False,
        "b12_complete": False, "valid_pixel_fraction": np.nan,
        "invalid_pixel_fraction": np.nan, "all_zero": np.nan, "has_nan": np.nan,
        "local_audit_error": "",
    }
    p = Path(str(row["image_path"]))
    result["image_exists"] = p.exists()
    if not p.exists():
        result["local_audit_error"] = "file_not_found"
        return result
    try:
        with rasterio.open(p) as src:
            result.update({
                "band_count": src.count, "width": src.width,
                "height": src.height, "crs": str(src.crs or ""),
            })
        arr, _ = read_raster(p)
        mask = valid(arr)
        result.update({
            "raster_read_ok": True,
            "b11_complete": bool(mask.any() and np.isfinite(arr[4][mask]).all()),
            "b12_complete": bool(mask.any() and np.isfinite(arr[5][mask]).all()),
            "valid_pixel_fraction": float(mask.mean()),
            "invalid_pixel_fraction": float(1 - mask.mean()),
            "all_zero": bool(np.all(np.nan_to_num(arr) == 0)),
            "has_nan": bool(np.isnan(arr).any()),
        })
    except Exception as exc:
        result["local_audit_error"] = type(exc).__name__ + ": " + str(exc)
    return result


def ee_init(project: str):
    import ee
    ee.Initialize(project=project) if project else ee.Initialize()
    return ee


def ee_find_s2(ee, row: pd.Series):
    cid = "COPERNICUS/S2_SR_HARMONIZED"
    scene = norm_text(row.get("scene_id", ""))
    if scene:
        try:
            image = ee.Image(cid + "/" + scene)
            if image.get("system:time_start").getInfo() is not None:
                return image
        except Exception:
            pass
    t = pd.to_datetime(row["acquisition_time_utc"], errors="coerce", utc=True)
    lat = pd.to_numeric(pd.Series([row["source_latitude"]]), errors="coerce").iloc[0]
    lon = pd.to_numeric(pd.Series([row["source_longitude"]]), errors="coerce").iloc[0]
    if pd.isna(t) or pd.isna(lat) or pd.isna(lon):
        return None
    coll = (ee.ImageCollection(cid)
            .filterBounds(ee.Geometry.Point([float(lon), float(lat)]))
            .filterDate((t - pd.Timedelta(hours=2)).isoformat(),
                        (t + pd.Timedelta(hours=2)).isoformat())
            .sort("system:time_start"))
    return ee.Image(coll.first()) if coll.size().getInfo() else None


def ee_scl_audit(ee, row: pd.Series, scale: float, half_size: float) -> dict:
    empty = {
        "ee_scene_found": False, "ee_scene_id": "",
        "ee_cloudy_pixel_percentage": np.nan,
        "ee_scl_cloud_shadow_fraction": np.nan,
        "ee_scl_cloud_fraction": np.nan,
        "ee_scl_cirrus_fraction": np.nan,
        "ee_scl_snow_fraction": np.nan,
        "ee_scl_invalid_fraction": np.nan,
        "ee_audit_error": "",
    }
    try:
        image = ee_find_s2(ee, row)
        if image is None:
            empty["ee_audit_error"] = "scene_not_found"
            return empty
        lat, lon = float(row["source_latitude"]), float(row["source_longitude"])
        region = ee.Geometry.Point([lon, lat]).buffer(half_size).bounds()
        scl = image.select("SCL")
        masks = {
            "ee_scl_cloud_shadow_fraction": scl.eq(3),
            "ee_scl_cloud_fraction": scl.eq(8).Or(scl.eq(9)),
            "ee_scl_cirrus_fraction": scl.eq(10),
            "ee_scl_snow_fraction": scl.eq(11),
            "ee_scl_invalid_fraction": scl.eq(0).Or(scl.eq(1)),
        }
        stack = ee.Image.cat([m.rename(name) for name, m in masks.items()])
        values = stack.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=scale,
            maxPixels=1_000_000, bestEffort=True).getInfo() or {}
        result = empty.copy()
        result.update(values)
        result.update({
            "ee_scene_found": True,
            "ee_scene_id": image.get("system:index").getInfo(),
            "ee_cloudy_pixel_percentage": image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo(),
        })
        return result
    except Exception as exc:
        empty["ee_audit_error"] = type(exc).__name__ + ": " + str(exc)
        return empty


def cmd_audit(args):
    root = Path(args.project_root).expanduser().resolve()
    outdir = root / "outputs"
    master_path = outdir / "36_multisite_s2_master_table.csv"
    if not master_path.exists():
        raise SystemExit("Run the master step first")
    master = pd.read_csv(master_path)
    result = pd.concat([master, pd.DataFrame([
        local_audit(row) for _, row in master.iterrows()
    ])], axis=1)

    if args.use_ee:
        ee = ee_init(args.ee_project)
        rows = []
        for i, (_, row) in enumerate(result.iterrows(), 1):
            rows.append(ee_scl_audit(ee, row, args.ee_scale_m, args.ee_half_size_m))
            if i % 10 == 0 or i == len(result):
                print("Earth Engine SCL audit %d/%d" % (i, len(result)), flush=True)
        result = pd.concat([result, pd.DataFrame(rows)], axis=1)
    else:
        for c in ("ee_scene_found", "ee_scene_id", "ee_cloudy_pixel_percentage",
                  "ee_scl_cloud_shadow_fraction", "ee_scl_cloud_fraction",
                  "ee_scl_cirrus_fraction", "ee_scl_snow_fraction",
                  "ee_scl_invalid_fraction", "ee_audit_error"):
            result[c] = np.nan if "fraction" in c or "percentage" in c else ""

    result["usable_local_image"] = (
        result["raster_read_ok"].fillna(False)
        & result["b11_complete"].fillna(False)
        & result["b12_complete"].fillna(False)
        & result["valid_pixel_fraction"].fillna(0).ge(0.80)
        & ~result["all_zero"].fillna(True)
    )
    result["usable_after_cloud_snow_filter"] = result["usable_local_image"]
    if args.use_ee:
        result["usable_after_cloud_snow_filter"] &= (
            result["ee_scl_cloud_fraction"].fillna(1).le(0.20)
            & result["ee_scl_snow_fraction"].fillna(1).le(0.05)
        )

    path = outdir / "37_multisite_s2_availability.csv"
    result.to_csv(path, index=False)
    summary = result.groupby("site_id", dropna=False).agg(
        rows=("sample_id", "size"),
        positive=("label", lambda x: int((x == 1).sum())),
        matched_negative=("label", lambda x: int((x == 0).sum())),
        readable=("raster_read_ok", "sum"),
        b11_complete=("b11_complete", "sum"),
        b12_complete=("b12_complete", "sum"),
        inside_release_interval=("acquisition_inside_release_interval", "sum"),
        usable_local=("usable_local_image", "sum"),
        usable_after_cloud_snow=("usable_after_cloud_snow_filter", "sum"),
        median_valid_fraction=("valid_pixel_fraction", "median"),
        median_cloud_fraction=("ee_scl_cloud_fraction", "median"),
        median_snow_fraction=("ee_scl_snow_fraction", "median"),
    ).reset_index()
    summary.to_csv(outdir / "37_multisite_s2_availability_summary.csv", index=False)
    (outdir / "37_multisite_s2_availability_report.txt").write_text(
        "MULTISITE SENTINEL-2 AVAILABILITY AUDIT\n" + "=" * 100 + "\n"
        + "Earth Engine SCL audit: %s\nRows: %d\nUsable local: %d\n"
        % (args.use_ee, len(result), int(result["usable_local_image"].sum()))
        + "Usable after cloud/snow: %d\n\n"
        % int(result["usable_after_cloud_snow_filter"].sum())
        + summary.to_string(index=False)
        + "\n\nLocal band-order assumption: B2,B3,B4,B8,B11,B12.\n"
        + "Cloud/snow require --use-ee because local six-band patches do not contain SCL.\n",
        encoding="utf-8")
    print("Created", path)


# ------------------------------ STEP 3 ---------------------------------------

TARGET_SENSORS = ("MethaneAIR", "Landsat", "GHGSat", "PRISMA")


def sensor_name(value: object) -> str:
    text = str(value).lower()
    if "methaneair" in text:
        return "MethaneAIR"
    if "landsat" in text or re.search(r"\bl[89]\b", text):
        return "Landsat"
    if "ghgsat" in text:
        return "GHGSat"
    if "prisma" in text:
        return "PRISMA"
    return ""


def sensor_files(root: Path, extras: Sequence[str]):
    patterns = ("methaneair", "landsat", "ghgsat", "prisma", "multisatellite", "historical")
    files = []
    for p in sorted((root / "outputs").glob("*.csv")):
        if any(x in p.name.lower() for x in patterns) and p.stat().st_size <= 100 * 1024 * 1024:
            files.append(p)
    for value in extras:
        p = Path(value)
        p = p if p.is_absolute() else root / p
        if p.exists():
            files.append(p)
    return list(dict.fromkeys(files))


def canonical_sensor_table(path: Path) -> pd.DataFrame:
    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    tc = first_col(raw, TIME)
    if tc is None:
        candidates = [c for c in raw.columns
                      if "acquisition" in str(c).lower() and "time" in str(c).lower()]
        tc = candidates[0] if candidates else None
    if tc is None:
        return pd.DataFrame()
    sens = first_col(raw, ("satellite", "sensor", "platform", "spacecraft", "instrument"))
    sc, sec, ec = first_col(raw, SITE), first_col(raw, SCENE), first_col(raw, EVENT)
    lac, loc = first_col(raw, LAT), first_col(raw, LON)
    out = pd.DataFrame(index=raw.index)
    out["sensor"] = raw[sens].map(sensor_name) if sens else sensor_name(path.name)
    out["sensor_time_utc"] = pd.to_datetime(raw[tc], errors="coerce", utc=True)
    out["site_id"] = raw[sc].astype(str) if sc else ""
    out["site_normalized"] = out["site_id"].map(norm_site)
    out["scene_id"] = raw[sec].astype(str) if sec else ""
    out["event_id"] = raw[ec].astype(str) if ec else ""
    out["latitude"] = pd.to_numeric(raw[lac], errors="coerce") if lac else np.nan
    out["longitude"] = pd.to_numeric(raw[loc], errors="coerce") if loc else np.nan
    out["source_file"] = str(path)
    return out[out["sensor"].isin(TARGET_SENSORS) & out["sensor_time_utc"].notna()].copy()


def ee_landsat_inventory(ee, master: pd.DataFrame, max_hours: float) -> pd.DataFrame:
    rows = []
    unique = master[["site_id", "source_latitude", "source_longitude",
                     "acquisition_time_utc"]].drop_duplicates()
    for _, s in unique.iterrows():
        t = pd.to_datetime(s["acquisition_time_utc"], errors="coerce", utc=True)
        lat = pd.to_numeric(pd.Series([s["source_latitude"]]), errors="coerce").iloc[0]
        lon = pd.to_numeric(pd.Series([s["source_longitude"]]), errors="coerce").iloc[0]
        if pd.isna(t) or pd.isna(lat) or pd.isna(lon):
            continue
        point = ee.Geometry.Point([float(lon), float(lat)])
        for cid in ("LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"):
            coll = (ee.ImageCollection(cid).filterBounds(point)
                    .filterDate((t - pd.Timedelta(hours=max_hours)).isoformat(),
                                (t + pd.Timedelta(hours=max_hours)).isoformat())
                    .sort("system:time_start").limit(10))
            info = coll.getInfo()
            for f in info.get("features", []):
                prop = f.get("properties", {})
                rows.append({
                    "sensor": "Landsat",
                    "sensor_time_utc": pd.to_datetime(
                        prop.get("system:time_start"), unit="ms", errors="coerce", utc=True),
                    "site_id": s["site_id"], "site_normalized": norm_site(s["site_id"]),
                    "scene_id": f.get("id", ""), "event_id": "",
                    "latitude": lat, "longitude": lon,
                    "source_file": "EarthEngine:" + cid,
                })
    return pd.DataFrame(rows)


def linked(s2: pd.Series, target: pd.Series, radius: float):
    if norm_text(s2.get("event_id", "")) and norm_text(target.get("event_id", "")) \
       and str(s2["event_id"]) == str(target["event_id"]):
        return True, "event_id_exact", 0.0
    if s2["site_normalized"] and target["site_normalized"] \
       and s2["site_normalized"] == target["site_normalized"]:
        return True, "site_exact", 0.0
    d = haversine(s2["source_latitude"], s2["source_longitude"],
                  target["latitude"], target["longitude"])
    return (True, "coordinate", d) if np.isfinite(d) and d <= radius else (False, "unlinked", d)


def match_tier(hours: float, same_date: bool, both_inside: bool) -> str:
    if both_inside:
        return "strict_same_release_interval"
    if same_date and hours <= 1:
        return "same_day_le_1h"
    if same_date and hours <= 3:
        return "same_day_1_to_3h"
    if same_date and hours <= 6:
        return "same_day_3_to_6h"
    if same_date and hours <= 24:
        return "same_day_6_to_24h"
    return "different_utc_date_le_24h"


def cmd_matches(args):
    root = Path(args.project_root).expanduser().resolve()
    outdir = root / "outputs"
    master_path = outdir / "36_multisite_s2_master_table.csv"
    if not master_path.exists():
        raise SystemExit("Run the master step first")
    master = pd.read_csv(master_path)
    for c in ("acquisition_time_utc", "release_start_utc", "release_end_utc"):
        master[c] = pd.to_datetime(master[c], errors="coerce", utc=True)
    master["site_normalized"] = master["site_id"].map(norm_site)

    files = sensor_files(root, args.extra_table)
    tables = [canonical_sensor_table(p) for p in files]
    tables = [x for x in tables if not x.empty]
    if args.use_ee_landsat:
        tables.append(ee_landsat_inventory(ee_init(args.ee_project), master, args.max_hours))
    inventory = pd.concat(tables, ignore_index=True, sort=False) if tables else pd.DataFrame(
        columns=["sensor", "sensor_time_utc", "site_id", "site_normalized",
                 "scene_id", "event_id", "latitude", "longitude", "source_file"])
    inventory = inventory.drop_duplicates(
        subset=["sensor", "sensor_time_utc", "scene_id", "site_normalized"])
    inventory.to_csv(outdir / "38_cross_sensor_inventory.csv", index=False)

    rows = []
    for _, s2 in master.iterrows():
        for sensor in TARGET_SENSORS:
            c = inventory[inventory["sensor"].eq(sensor)].copy()
            if c.empty or pd.isna(s2["acquisition_time_utc"]):
                rows.append({"s2_sample_id": s2["sample_id"], "s2_site_id": s2["site_id"],
                             "target_sensor": sensor, "match_found": False,
                             "match_status": "no_sensor_records"})
                continue
            c["absolute_time_difference_hours"] = (
                c["sensor_time_utc"] - s2["acquisition_time_utc"]
            ).abs().dt.total_seconds() / 3600
            c = c[c["absolute_time_difference_hours"] <= args.max_hours].copy()
            good = []
            for _, target in c.iterrows():
                ok, method, distance = linked(s2, target, args.spatial_match_km)
                if ok:
                    d = target.to_dict()
                    d.update({"link_method": method, "distance_km": distance})
                    good.append(d)
            if not good:
                rows.append({"s2_sample_id": s2["sample_id"], "s2_site_id": s2["site_id"],
                             "target_sensor": sensor, "match_found": False,
                             "match_status": "no_spatiotemporal_match"})
                continue
            x = pd.DataFrame(good).sort_values(
                ["absolute_time_difference_hours", "distance_km"]).iloc[0]
            tt = pd.to_datetime(x["sensor_time_utc"], utc=True)
            same = tt.date() == s2["acquisition_time_utc"].date()
            both = bool(pd.notna(s2["release_start_utc"]) and pd.notna(s2["release_end_utc"])
                        and s2["release_start_utc"] <= s2["acquisition_time_utc"] <= s2["release_end_utc"]
                        and s2["release_start_utc"] <= tt <= s2["release_end_utc"])
            rows.append({
                "s2_sample_id": s2["sample_id"], "s2_site_id": s2["site_id"],
                "s2_scene_id": s2["scene_id"], "s2_time_utc": s2["acquisition_time_utc"],
                "s2_label": s2["label"], "release_start_utc": s2["release_start_utc"],
                "release_end_utc": s2["release_end_utc"],
                "target_sensor": sensor, "target_scene_id": x["scene_id"],
                "target_time_utc": tt, "target_source_file": x["source_file"],
                "match_found": True, "match_status": "matched",
                "link_method": x["link_method"], "distance_km": x["distance_km"],
                "signed_time_difference_hours": (
                    tt - s2["acquisition_time_utc"]).total_seconds() / 3600,
                "absolute_time_difference_hours": x["absolute_time_difference_hours"],
                "same_utc_date": same, "both_inside_same_release_interval": both,
                "temporal_match_tier": match_tier(
                    x["absolute_time_difference_hours"], same, both),
            })
    result = pd.DataFrame(rows)
    path = outdir / "38_cross_sensor_temporal_matches.csv"
    result.to_csv(path, index=False)
    found = result[result["match_found"].fillna(False)].copy()
    summary = found.groupby(["target_sensor", "temporal_match_tier"], dropna=False).agg(
        matched_pairs=("s2_sample_id", "size"),
        unique_s2_scenes=("s2_scene_id", "nunique"),
        unique_target_scenes=("target_scene_id", "nunique"),
        median_time_difference_hours=("absolute_time_difference_hours", "median")
    ).reset_index() if not found.empty else pd.DataFrame()
    summary.to_csv(outdir / "38_cross_sensor_temporal_summary.csv", index=False)
    (outdir / "38_cross_sensor_temporal_report.txt").write_text(
        "CROSS-SENSOR TEMPORAL MATCHING\n" + "=" * 100 + "\n"
        + "Input CSV tables scanned: %d\nInventory rows: %d\nMatched pairs: %d\n\n"
        % (len(files), len(inventory), len(found))
        + (summary.to_string(index=False) if not summary.empty else "No matches found.")
        + "\n\nMatching requires event ID, site, or coordinate linkage; time-only coincidence is rejected.\n"
        + "GHGSat and PRISMA depend on local tables or --extra-table.\n",
        encoding="utf-8")
    print("Created", path)


# ------------------------------ STEP 4 ---------------------------------------

def circular_day(a, b):
    ta = pd.to_datetime(a, errors="coerce", utc=True)
    tb = pd.to_datetime(b, errors="coerce", utc=True)
    if pd.isna(ta) or pd.isna(tb):
        return np.inf
    d = abs(int(ta.dayofyear) - int(tb.dayofyear))
    return float(min(d, 366 - d))


def choose_reference(index: int, row: pd.Series, master: pd.DataFrame):
    same = master[(master["site_id"] == row["site_id"]) & (master.index != index)].copy()
    if same.empty:
        return None
    preferred = same[same["label"] == 0].copy()
    if preferred.empty:
        preferred = same
    preferred["seasonal_distance_days"] = preferred["acquisition_time_utc"].map(
        lambda x: circular_day(row["acquisition_time_utc"], x))
    preferred["absolute_date_distance_days"] = (
        preferred["acquisition_time_utc"] - row["acquisition_time_utc"]
    ).abs().dt.total_seconds() / 86400
    return preferred.sort_values(
        ["seasonal_distance_days", "absolute_date_distance_days"]).iloc[0]


def wind_region_masks(h: int, w: int, wind_from: object):
    value = pd.to_numeric(pd.Series([wind_from]), errors="coerce").iloc[0]
    empty = np.zeros((h, w), dtype=bool)
    if pd.isna(value):
        return empty, empty
    down_deg = (float(value) + 180) % 360
    theta = math.radians(down_deg)
    yy, xx = np.indices((h, w))
    cy, cx = (h - 1) / 2, (w - 1) / 2
    dx, dy_north = xx - cx, cy - yy
    along = dx * math.sin(theta) + dy_north * math.cos(theta)
    cross = dx * math.cos(theta) - dy_north * math.sin(theta)
    scale = float(min(h, w))
    down = ((along >= 0.05 * scale) & (along <= 0.45 * scale)
            & (np.abs(cross) <= 0.14 * scale))
    up = ((along <= -0.05 * scale) & (along >= -0.45 * scale)
          & (np.abs(cross) <= 0.14 * scale))
    return down, up


def reproject_reference(path: Path, target_profile: dict):
    src_arr, src = read_raster(path)
    h, w = target_profile["height"], target_profile["width"]
    if (src["crs"] == target_profile["crs"]
        and src["transform"] == target_profile["transform"]
        and src_arr.shape[1:] == (h, w)):
        return src_arr
    out = np.full((6, h, w), np.nan, dtype="float64")
    for i in range(6):
        reproject(
            source=src_arr[i], destination=out[i],
            src_transform=src["transform"], src_crs=src["crs"],
            dst_transform=target_profile["transform"], dst_crs=target_profile["crs"],
            resampling=Resampling.bilinear,
            src_nodata=np.nan, dst_nodata=np.nan)
    return out


def extract_features(row: pd.Series, reference):
    result = {
        "feature_read_ok": False, "feature_error": "",
        "temporal_reference_sample_id": "", "temporal_reference_label": np.nan,
        "temporal_reference_type": "",
        "temporal_reference_seasonal_distance_days": np.nan,
        "temporal_reference_absolute_distance_days": np.nan,
        "temporal_reference_uses_known_label": False,
    }
    try:
        arr, profile = read_raster(Path(row["image_path"]))
        mask = valid(arr)
        center_raw, bg_raw = radial_masks(arr.shape[1], arr.shape[2])
        center, bg = center_raw & mask, bg_raw & mask
        down_raw, up_raw = wind_region_masks(arr.shape[1], arr.shape[2],
                                              row.get("wind_direction", np.nan))
        down, up = down_raw & mask, up_raw & mask
        if not mask.any() or not center.any() or not bg.any():
            raise ValueError("No valid whole-patch, center, or background pixels")

        arrays = {}
        for i, name in enumerate(BANDS):
            x = arr[i]
            arrays[name] = x
            result.update(stats(x[mask], name))
            result.update(stats(x[center], name + "_center"))
            result.update(stats(x[bg], name + "_background"))
            result[name + "_center_minus_background"] = (
                float(np.nanmedian(x[center])) - float(np.nanmedian(x[bg])))
            result.update(stats(x[down], name + "_downwind"))
            result.update(stats(x[up], name + "_upwind"))
            result[name + "_downwind_minus_upwind"] = (
                float(np.nanmedian(x[down])) - float(np.nanmedian(x[up]))
                if down.any() and up.any() else np.nan)

        b2, b3, b4, b8, b11, b12 = [arrays[x] for x in BANDS]
        derived = {
            "ndvi": nd(b8, b4),
            "swir_nd_b11_b12": nd(b11, b12),
            "swir_ratio_b12_b11": ratio(b12, b11),
            "swir_difference_b11_minus_b12": b11 - b12,
        }
        for name, x in derived.items():
            result.update(stats(x[mask], name))
            result.update(stats(x[center], name + "_center"))
            result.update(stats(x[bg], name + "_background"))
            result[name + "_center_minus_background"] = (
                float(np.nanmedian(x[center])) - float(np.nanmedian(x[bg])))
            result.update(stats(x[down], name + "_downwind"))
            result.update(stats(x[up], name + "_upwind"))
            result[name + "_downwind_minus_upwind"] = (
                float(np.nanmedian(x[down])) - float(np.nanmedian(x[up]))
                if down.any() and up.any() else np.nan)

        invalid_fraction = float(1 - mask.mean())
        b11_cv = float(np.nanstd(b11[bg])) / (abs(float(np.nanmean(b11[bg]))) + EPS)
        b12_cv = float(np.nanstd(b12[bg])) / (abs(float(np.nanmean(b12[bg]))) + EPS)
        ndvi_sd = min(1.0, float(np.nanstd(derived["ndvi"][bg])))
        heterogeneity = min(1.0, (b11_cv + b12_cv) / 2)
        quality = max(0.0, min(1.0, 1 - invalid_fraction - 0.5 * heterogeneity - 0.5 * ndvi_sd))
        result.update({
            "valid_pixel_fraction_feature": float(mask.mean()),
            "center_valid_pixel_fraction": float(mask[center_raw].mean()),
            "background_valid_pixel_fraction": float(mask[bg_raw].mean()),
            "background_b11_cv": b11_cv, "background_b12_cv": b12_cv,
            "background_quality_score_heuristic": quality,
            "wind_features_available": bool(down.any() and up.any()),
        })

        if reference is not None:
            ref = reproject_reference(Path(reference["image_path"]), profile)
            common = mask & valid(ref)
            common_center, common_bg = common & center_raw, common & bg_raw
            if common.any():
                diff = arr - ref
                for i, name in enumerate(BANDS):
                    x = diff[i]
                    prefix = "temporal_" + name + "_difference"
                    result.update(stats(x[common], prefix))
                    result.update(stats(x[common_center], prefix + "_center"))
                    result.update(stats(x[common_bg], prefix + "_background"))
                    result[prefix + "_center_minus_background"] = (
                        float(np.nanmedian(x[common_center]))
                        - float(np.nanmedian(x[common_bg]))
                        if common_center.any() and common_bg.any() else np.nan)
                target_ndvi, ref_ndvi = nd(arr[3], arr[2]), nd(ref[3], ref[2])
                target_ratio, ref_ratio = ratio(arr[5], arr[4]), ratio(ref[5], ref[4])
                for name, x in {
                    "temporal_ndvi_difference": target_ndvi - ref_ndvi,
                    "temporal_swir_ratio_difference": target_ratio - ref_ratio,
                }.items():
                    result.update(stats(x[common], name))
                    result.update(stats(x[common_center], name + "_center"))
                    result.update(stats(x[common_bg], name + "_background"))

            t1 = pd.to_datetime(row["acquisition_time_utc"], utc=True)
            t2 = pd.to_datetime(reference["acquisition_time_utc"], utc=True)
            result.update({
                "temporal_reference_sample_id": reference["sample_id"],
                "temporal_reference_label": reference["label"],
                "temporal_reference_type": (
                    "matched_negative_reference" if int(reference["label"]) == 0
                    else "nearest_same_site_scene"),
                "temporal_reference_seasonal_distance_days": circular_day(t1, t2),
                "temporal_reference_absolute_distance_days":
                    abs((t1 - t2).total_seconds()) / 86400,
                "temporal_reference_uses_known_label": True,
            })

        result["feature_read_ok"] = True
    except Exception as exc:
        result["feature_error"] = type(exc).__name__ + ": " + str(exc)
    return result


def cmd_features(args):
    root = Path(args.project_root).expanduser().resolve()
    outdir = root / "outputs"
    master_path = outdir / "36_multisite_s2_master_table.csv"
    if not master_path.exists():
        raise SystemExit("Run the master step first")
    master = pd.read_csv(master_path)
    master["acquisition_time_utc"] = pd.to_datetime(
        master["acquisition_time_utc"], errors="coerce", utc=True)
    rows = []
    for i, row in master.iterrows():
        record = row.to_dict()
        record.update(extract_features(row, choose_reference(i, row, master)))
        rows.append(record)
        if (i + 1) % 10 == 0 or i + 1 == len(master):
            print("Feature extraction %d/%d" % (i + 1, len(master)), flush=True)
    result = pd.DataFrame(rows)
    path = outdir / "39_multisite_s2_features.csv"
    result.to_csv(path, index=False)
    summary = result.groupby("site_id", dropna=False).agg(
        rows=("sample_id", "size"), feature_success=("feature_read_ok", "sum"),
        positive=("label", lambda x: int((x == 1).sum())),
        negative=("label", lambda x: int((x == 0).sum())),
        temporal_reference_available=("temporal_reference_sample_id",
                                      lambda x: int(x.astype(str).str.len().gt(0).sum())),
        wind_feature_available=("wind_features_available", "sum"),
        median_background_quality=("background_quality_score_heuristic", "median"),
    ).reset_index()
    summary.to_csv(outdir / "39_multisite_s2_features_summary.csv", index=False)
    (outdir / "39_multisite_s2_features_report.txt").write_text(
        "MULTISITE SENTINEL-2 FEATURE EXTRACTION\n" + "=" * 100 + "\n"
        + "Rows: %d\nFeature success: %d\n\n"
        % (len(result), int(result["feature_read_ok"].sum()))
        + summary.to_string(index=False)
        + "\n\nFeature groups: B11/B12, source center, background ring, NDVI, "
          "temporal differences, wind-aligned downwind/upwind, background quality.\n"
        + "Temporal references use known label-0 rows from the same site, so these "
          "features are site-calibrated rather than strict zero-shot features.\n"
        + "Wind direction is interpreted as meteorological FROM direction.\n",
        encoding="utf-8")
    print("Created", path)


# ------------------------------ CLI ------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Run multisite Sentinel-2 steps 1–4")
    sub = p.add_subparsers(dest="command", required=True)

    def root_arg(x):
        x.add_argument("--project-root",
                       default="/Users/happydoraaa/methane_release_project")

    m = sub.add_parser("master")
    root_arg(m)
    m.add_argument("--manifest")
    m.add_argument("--release-intervals")
    m.add_argument("--wind-table")
    m.add_argument("--wind-max-hours", type=float, default=12)
    m.add_argument("--spatial-match-km", type=float, default=10)
    m.set_defaults(func=cmd_master)

    a = sub.add_parser("audit")
    root_arg(a)
    a.add_argument("--use-ee", action="store_true")
    a.add_argument("--ee-project", default=os.environ.get("EE_PROJECT", ""))
    a.add_argument("--ee-scale-m", type=float, default=20)
    a.add_argument("--ee-half-size-m", type=float, default=640)
    a.set_defaults(func=cmd_audit)

    c = sub.add_parser("matches")
    root_arg(c)
    c.add_argument("--max-hours", type=float, default=24)
    c.add_argument("--spatial-match-km", type=float, default=15)
    c.add_argument("--extra-table", action="append", default=[])
    c.add_argument("--use-ee-landsat", action="store_true")
    c.add_argument("--ee-project", default=os.environ.get("EE_PROJECT", ""))
    c.set_defaults(func=cmd_matches)

    f = sub.add_parser("features")
    root_arg(f)
    f.set_defaults(func=cmd_features)

    allp = sub.add_parser("all")
    root_arg(allp)
    allp.add_argument("--manifest")
    allp.add_argument("--release-intervals")
    allp.add_argument("--wind-table")
    allp.add_argument("--wind-max-hours", type=float, default=12)
    allp.add_argument("--spatial-match-km", type=float, default=10)
    allp.add_argument("--use-ee", action="store_true")
    allp.add_argument("--use-ee-landsat", action="store_true")
    allp.add_argument("--ee-project", default=os.environ.get("EE_PROJECT", ""))
    allp.add_argument("--ee-scale-m", type=float, default=20)
    allp.add_argument("--ee-half-size-m", type=float, default=640)
    allp.add_argument("--max-hours", type=float, default=24)
    allp.add_argument("--extra-table", action="append", default=[])
    allp.set_defaults(func=None)
    return p


def main():
    args = build_parser().parse_args()
    if args.command == "all":
        cmd_master(args)
        cmd_audit(args)
        cmd_matches(args)
        cmd_features(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
