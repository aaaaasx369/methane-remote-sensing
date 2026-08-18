#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete the remaining multisite workflow:
1. Audit and match GHGSat/PRISMA.
2. Create strict-zero-shot and site-calibrated feature views.
3. Audit and match numerical wind records.
4. Compute wind-aligned Sentinel-2 features.

Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio

TIME_ALIASES = (
    "acquisition_time_utc", "acquisition_time", "datetime_utc", "datetime",
    "timestamp_utc", "timestamp", "time_utc", "overpass_time_utc",
    "overpass_time", "observation_time_utc", "observation_time",
    "scene_time_utc", "sensing_time_utc", "sensing_time",
    "time_coverage_start", "start_time_utc", "start_time",
)
SITE_ALIASES = ("site_id", "site", "site_name", "facility", "location", "release_site")
LAT_ALIASES = ("latitude", "lat", "source_latitude", "site_latitude", "latitude_canonical")
LON_ALIASES = ("longitude", "lon", "lng", "source_longitude", "site_longitude", "longitude_canonical")
EVENT_ALIASES = ("event_id", "release_id", "observation_id", "plume_id", "source_event_id")
SCENE_ALIASES = ("scene_id", "image_id", "product_id", "granule_id", "system:index", "system_index")
SENSOR_ALIASES = ("sensor", "satellite", "platform", "instrument", "spacecraft")
PATH_ALIASES = ("image_path", "patch_path", "resolved_patch_path", "patch_path_raw", "relative_path", "file_path", "filename")

SPEED_ALIASES = ("wind_speed_m_s", "wind_speed_ms", "wind_speed", "windspeed", "wind_m_s", "ws_m_s", "ws")
DIRECTION_ALIASES = ("wind_direction_deg", "wind_direction_degrees", "wind_direction", "wind_dir_deg", "wind_dir", "wind_bearing")
U_ALIASES = ("u10", "u_10m", "wind_u", "u_wind", "eastward_wind", "u_component")
V_ALIASES = ("v10", "v_10m", "wind_v", "v_wind", "northward_wind", "v_component")

EARTH_RADIUS_KM = 6371.0088
EPS = 1e-9


def first_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    mapping = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in mapping:
            return mapping[alias.lower()]
    return None


def normalize_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def normalize_site(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", normalize_text(value).lower())
    stop = {"release", "releases", "stack", "stacks", "site", "facility", "controlled", "methane", "source", "test", "station"}
    return " ".join(token for token in text.split() if token not in stop)


def sensor_name(value: object) -> str:
    text = str(value).lower()
    if "ghgsat" in text:
        return "GHGSat"
    if "prisma" in text:
        return "PRISMA"
    return ""


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def resolve(root: Path, explicit: Optional[Path], default_name: str) -> Path:
    path = explicit if explicit is not None else Path("outputs") / default_name
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise SystemExit("找不到檔案: %s" % path)
    return path


def sensor_files(root: Path, extras) -> list:
    files = [
        p for p in sorted((root / "outputs").glob("*.csv"))
        if "ghgsat" in p.name.lower() or "prisma" in p.name.lower()
    ]
    for value in extras:
        p = Path(value)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            files.append(p)
    return list(dict.fromkeys(files))


def sensor_audit(root: Path, extras) -> pd.DataFrame:
    records = []
    for path in sensor_files(root, extras):
        df = read_csv(path)
        if df is None:
            continue
        t = first_column(df, TIME_ALIASES)
        s = first_column(df, SENSOR_ALIASES)
        site = first_column(df, SITE_ALIASES)
        lat = first_column(df, LAT_ALIASES)
        lon = first_column(df, LON_ALIASES)
        event = first_column(df, EVENT_ALIASES)
        scene = first_column(df, SCENE_ALIASES)
        records.append({
            "source_file": str(path),
            "rows": len(df),
            "sensor_from_filename": sensor_name(path.name),
            "sensor_column": s or "",
            "time_column": t or "",
            "valid_times": int(pd.to_datetime(df[t], errors="coerce", utc=True).notna().sum()) if t else 0,
            "site_column": site or "",
            "latitude_column": lat or "",
            "longitude_column": lon or "",
            "event_column": event or "",
            "scene_column": scene or "",
        })
    out = pd.DataFrame(records)
    out.to_csv(root / "outputs/43_ghgsat_prisma_inventory_audit.csv", index=False)
    report = [
        "GHGSAT / PRISMA INVENTORY AUDIT",
        "=" * 100,
        "Candidate CSV tables: %d" % len(out),
        "",
        out.to_string(index=False) if not out.empty else "No GHGSat/PRISMA candidate tables found.",
    ]
    (root / "outputs/43_ghgsat_prisma_inventory_audit_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return out


def canonical_sensor(path: Path) -> pd.DataFrame:
    raw = read_csv(path)
    if raw is None or raw.empty:
        return pd.DataFrame()
    t = first_column(raw, TIME_ALIASES)
    if t is None:
        return pd.DataFrame()
    s = first_column(raw, SENSOR_ALIASES)
    site = first_column(raw, SITE_ALIASES)
    lat = first_column(raw, LAT_ALIASES)
    lon = first_column(raw, LON_ALIASES)
    event = first_column(raw, EVENT_ALIASES)
    scene = first_column(raw, SCENE_ALIASES)

    out = pd.DataFrame({
        "sensor": raw[s].map(sensor_name) if s else sensor_name(path.name),
        "target_time_utc": pd.to_datetime(raw[t], errors="coerce", utc=True),
        "target_site_id": raw[site].astype(str) if site else "",
        "target_event_id": raw[event].astype(str) if event else "",
        "target_scene_id": raw[scene].astype(str) if scene else "",
        "target_latitude": pd.to_numeric(raw[lat], errors="coerce") if lat else np.nan,
        "target_longitude": pd.to_numeric(raw[lon], errors="coerce") if lon else np.nan,
        "target_source_file": str(path),
        "target_source_row": np.arange(len(raw)),
    })
    out["target_site_normalized"] = out["target_site_id"].map(normalize_site)
    return out[out["sensor"].isin(["GHGSat", "PRISMA"]) & out["target_time_utc"].notna()].copy()


def sensor_match(root: Path, master_path: Path, extras, max_hours: float, spatial_km: float):
    master = pd.read_csv(master_path, low_memory=False)
    master["acquisition_time_utc"] = pd.to_datetime(master["acquisition_time_utc"], errors="coerce", utc=True)
    master["site_normalized"] = master["site_id"].map(normalize_site)

    inv_parts = [canonical_sensor(p) for p in sensor_files(root, extras)]
    inv_parts = [x for x in inv_parts if not x.empty]
    inventory = pd.concat(inv_parts, ignore_index=True) if inv_parts else pd.DataFrame()
    if not inventory.empty:
        inventory = inventory.drop_duplicates(
            subset=["sensor", "target_time_utc", "target_scene_id", "target_event_id", "target_site_normalized", "target_latitude", "target_longitude"]
        )
    inventory.to_csv(root / "outputs/43_ghgsat_prisma_inventory.csv", index=False)

    rows = []
    for _, s2 in master.iterrows():
        for sensor in ["GHGSat", "PRISMA"]:
            candidates = inventory[inventory["sensor"].eq(sensor)].copy() if not inventory.empty else pd.DataFrame()
            base = {
                "s2_sample_id": s2.get("sample_id", ""),
                "s2_site_id": s2.get("site_id", ""),
                "s2_scene_id": s2.get("scene_id", ""),
                "s2_time_utc": s2.get("acquisition_time_utc"),
                "s2_label": s2.get("label"),
                "target_sensor": sensor,
                "match_found": False,
                "match_status": "",
            }
            if candidates.empty:
                base["match_status"] = "no_sensor_records"
                rows.append(base)
                continue

            candidates["absolute_time_difference_hours"] = (
                candidates["target_time_utc"] - s2["acquisition_time_utc"]
            ).abs().dt.total_seconds() / 3600
            candidates = candidates[candidates["absolute_time_difference_hours"] <= max_hours].copy()

            linked = []
            for _, target in candidates.iterrows():
                method = ""
                distance = np.nan
                s2_event = normalize_text(s2.get("event_id", ""))
                target_event = normalize_text(target.get("target_event_id", ""))
                if s2_event and target_event and s2_event == target_event:
                    method = "event_id_exact"
                    distance = 0.0
                elif s2["site_normalized"] and s2["site_normalized"] == target["target_site_normalized"]:
                    method = "site_exact"
                    distance = 0.0
                else:
                    distance = haversine_km(
                        s2.get("source_latitude"), s2.get("source_longitude"),
                        target.get("target_latitude"), target.get("target_longitude"),
                    )
                    if np.isfinite(distance) and distance <= spatial_km:
                        method = "coordinate"
                if method:
                    rec = target.to_dict()
                    rec["link_method"] = method
                    rec["distance_km"] = distance
                    linked.append(rec)

            if not linked:
                base["match_status"] = "no_spatiotemporal_match"
                rows.append(base)
                continue

            selected = pd.DataFrame(linked).sort_values(
                ["absolute_time_difference_hours", "distance_km"]
            ).iloc[0]
            base.update(selected.to_dict())
            base["match_found"] = True
            base["match_status"] = "matched"
            base["signed_time_difference_hours"] = (
                selected["target_time_utc"] - s2["acquisition_time_utc"]
            ).total_seconds() / 3600
            rows.append(base)

    out = pd.DataFrame(rows)
    out.to_csv(root / "outputs/44_ghgsat_prisma_temporal_matches.csv", index=False)
    found = out[out["match_found"].fillna(False)]
    summary = (
        found.groupby("target_sensor")
        .agg(
            matched_pairs=("s2_sample_id", "size"),
            unique_s2_scenes=("s2_scene_id", "nunique"),
            unique_target_scenes=("target_scene_id", "nunique"),
            median_time_difference_hours=("absolute_time_difference_hours", "median"),
        )
        .reset_index()
        if not found.empty else pd.DataFrame()
    )
    summary.to_csv(root / "outputs/44_ghgsat_prisma_temporal_summary.csv", index=False)
    text = [
        "GHGSAT / PRISMA TEMPORAL MATCHING",
        "=" * 100,
        "Inventory rows: %d" % len(inventory),
        "Matched pairs: %d" % len(found),
        "",
        summary.to_string(index=False) if not summary.empty else "No valid matches found.",
        "",
        "Time-only coincidence is rejected.",
    ]
    (root / "outputs/44_ghgsat_prisma_temporal_report.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


def feature_views(root: Path, feature_path: Path):
    df = pd.read_csv(feature_path, low_memory=False)
    temporal = [c for c in df if c.lower().startswith("temporal_") or "temporal_reference" in c.lower()]
    wind = [c for c in df if c.lower().startswith("wind_") or "downwind" in c.lower() or "upwind" in c.lower()]
    strict = df.drop(columns=list(dict.fromkeys(temporal + wind)), errors="ignore")
    calibrated = df.drop(columns=wind, errors="ignore")
    strict.to_csv(root / "outputs/40_multisite_s2_features_strict_zero_shot.csv", index=False)
    calibrated.to_csv(root / "outputs/41_multisite_s2_features_site_calibrated.csv", index=False)
    text = [
        "FEATURE VIEWS",
        "=" * 100,
        "Rows: %d" % len(df),
        "Original columns: %d" % len(df.columns),
        "Temporal columns: %d" % len(temporal),
        "Wind-related columns: %d" % len(wind),
        "Strict zero-shot columns: %d" % len(strict.columns),
        "Site-calibrated columns: %d" % len(calibrated.columns),
    ]
    (root / "outputs/40_41_feature_views_report.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


def wind_audit(root: Path) -> pd.DataFrame:
    records = []
    for path in sorted((root / "outputs").glob("*.csv")):
        df = read_csv(path)
        if df is None or df.empty:
            continue
        t = first_column(df, TIME_ALIASES)
        site = first_column(df, SITE_ALIASES)
        lat = first_column(df, LAT_ALIASES)
        lon = first_column(df, LON_ALIASES)
        speed = first_column(df, SPEED_ALIASES)
        direction = first_column(df, DIRECTION_ALIASES)
        u = first_column(df, U_ALIASES)
        v = first_column(df, V_ALIASES)
        if not ((speed and direction) or (u and v)):
            continue
        valid_times = int(pd.to_datetime(df[t], errors="coerce", utc=True).notna().sum()) if t else 0
        valid_speed = int(pd.to_numeric(df[speed], errors="coerce").notna().sum()) if speed else 0
        valid_direction = int(pd.to_numeric(df[direction], errors="coerce").notna().sum()) if direction else 0
        valid_u = int(pd.to_numeric(df[u], errors="coerce").notna().sum()) if u else 0
        valid_v = int(pd.to_numeric(df[v], errors="coerce").notna().sum()) if v else 0
        score = 3 * valid_times + 2 * valid_speed + 2 * valid_direction + valid_u + valid_v
        if site:
            score += len(df)
        if lat and lon:
            score += len(df)
        records.append({
            "source_file": str(path), "rows": len(df),
            "time_column": t or "", "valid_times": valid_times,
            "site_column": site or "", "latitude_column": lat or "", "longitude_column": lon or "",
            "speed_column": speed or "", "valid_speed": valid_speed,
            "direction_column": direction or "", "valid_direction": valid_direction,
            "u_column": u or "", "valid_u": valid_u,
            "v_column": v or "", "valid_v": valid_v,
            "score": score,
        })
    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["score", "valid_times", "rows"], ascending=False)
    out.to_csv(root / "outputs/45_wind_table_audit.csv", index=False)
    text = [
        "WIND TABLE AUDIT",
        "=" * 100,
        "Candidate numerical wind tables: %d" % len(out),
        "",
        out.head(30).to_string(index=False) if not out.empty else "No valid numerical wind table found.",
    ]
    (root / "outputs/45_wind_table_audit_report.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    return out


def canonical_wind(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    t = first_column(raw, TIME_ALIASES)
    site = first_column(raw, SITE_ALIASES)
    lat = first_column(raw, LAT_ALIASES)
    lon = first_column(raw, LON_ALIASES)
    speed = first_column(raw, SPEED_ALIASES)
    direction = first_column(raw, DIRECTION_ALIASES)
    ucol = first_column(raw, U_ALIASES)
    vcol = first_column(raw, V_ALIASES)
    if not t:
        raise SystemExit("Wind table 沒有可辨識的時間欄位: %s" % path)

    out = pd.DataFrame({
        "wind_time_utc": pd.to_datetime(raw[t], errors="coerce", utc=True),
        "wind_site_id": raw[site].astype(str) if site else "",
        "wind_latitude": pd.to_numeric(raw[lat], errors="coerce") if lat else np.nan,
        "wind_longitude": pd.to_numeric(raw[lon], errors="coerce") if lon else np.nan,
        "wind_source_row": np.arange(len(raw)),
    })
    out["wind_site_normalized"] = out["wind_site_id"].map(normalize_site)

    if speed and direction:
        out["wind_speed_m_s"] = pd.to_numeric(raw[speed], errors="coerce")
        out["wind_direction_from_deg"] = pd.to_numeric(raw[direction], errors="coerce")
        out["wind_derivation"] = "direct"
    elif ucol and vcol:
        u = pd.to_numeric(raw[ucol], errors="coerce")
        v = pd.to_numeric(raw[vcol], errors="coerce")
        out["wind_speed_m_s"] = np.sqrt(u ** 2 + v ** 2)
        out["wind_direction_from_deg"] = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
        out["wind_derivation"] = "from_uv"
    else:
        raise SystemExit("Wind table 沒有數值 speed/direction 或 u/v。")

    out["wind_source_file"] = str(path)
    out = out.dropna(subset=["wind_time_utc", "wind_speed_m_s", "wind_direction_from_deg"])
    return out[out["wind_speed_m_s"].ge(0) & out["wind_direction_from_deg"].between(0, 360)].copy()


def wind_match(root: Path, master_path: Path, wind_table: Path, max_hours: float, spatial_km: float):
    master = pd.read_csv(master_path, low_memory=False)
    master["acquisition_time_utc"] = pd.to_datetime(master["acquisition_time_utc"], errors="coerce", utc=True)
    master["site_normalized"] = master["site_id"].map(normalize_site)
    wind = canonical_wind(wind_table)
    wind.to_csv(root / "outputs/46_canonical_wind_inventory.csv", index=False)

    rows = []
    for _, sample in master.iterrows():
        base = {
            "sample_id": sample.get("sample_id", ""), "site_id": sample.get("site_id", ""),
            "scene_id": sample.get("scene_id", ""), "acquisition_time_utc": sample.get("acquisition_time_utc"),
            "wind_match_found": False, "wind_match_method": "",
            "wind_time_utc": pd.NaT, "wind_time_difference_hours": np.nan,
            "wind_distance_km": np.nan, "wind_speed_m_s": np.nan,
            "wind_direction_from_deg": np.nan, "wind_derivation": "",
            "wind_source_file": str(wind_table), "wind_source_row": np.nan,
        }
        same = wind[wind["wind_site_normalized"].ne("") & wind["wind_site_normalized"].eq(sample["site_normalized"])].copy()
        method = "site_exact"
        if same.empty:
            wind2 = wind.copy()
            wind2["distance_km"] = wind2.apply(
                lambda r: haversine_km(
                    sample.get("source_latitude"), sample.get("source_longitude"),
                    r.get("wind_latitude"), r.get("wind_longitude"),
                ), axis=1
            )
            same = wind2[wind2["distance_km"] <= spatial_km].copy()
            method = "coordinate"
        if same.empty:
            rows.append(base)
            continue
        same["time_difference_hours"] = (
            same["wind_time_utc"] - sample["acquisition_time_utc"]
        ).abs().dt.total_seconds() / 3600
        sort_cols = ["time_difference_hours"] + (["distance_km"] if "distance_km" in same else [])
        selected = same.sort_values(sort_cols).iloc[0]
        if selected["time_difference_hours"] > max_hours:
            rows.append(base)
            continue
        base.update({
            "wind_match_found": True,
            "wind_match_method": method,
            "wind_time_utc": selected["wind_time_utc"],
            "wind_time_difference_hours": selected["time_difference_hours"],
            "wind_distance_km": selected.get("distance_km", 0.0),
            "wind_speed_m_s": selected["wind_speed_m_s"],
            "wind_direction_from_deg": selected["wind_direction_from_deg"],
            "wind_derivation": selected["wind_derivation"],
            "wind_source_row": selected["wind_source_row"],
        })
        rows.append(base)

    out = pd.DataFrame(rows)
    out.to_csv(root / "outputs/46_multisite_wind_matches.csv", index=False)
    summary = out.groupby("site_id").agg(
        rows=("sample_id", "size"),
        wind_matches=("wind_match_found", "sum"),
        median_time_difference_hours=("wind_time_difference_hours", "median"),
        median_wind_speed_m_s=("wind_speed_m_s", "median"),
    ).reset_index()
    summary.to_csv(root / "outputs/46_multisite_wind_summary.csv", index=False)
    text = [
        "MULTISITE WIND MATCHING",
        "=" * 100,
        "Wind table: %s" % wind_table,
        "Valid wind records: %d" % len(wind),
        "Matched S2 rows: %d / %d" % (int(out["wind_match_found"].sum()), len(out)),
        "",
        summary.to_string(index=False),
    ]
    (root / "outputs/46_multisite_wind_report.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


def valid_mask(array: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(array), axis=0) & np.any(array != 0, axis=0)


def nd(a, b):
    out = np.full_like(a, np.nan, dtype=float)
    good = np.isfinite(a) & np.isfinite(b) & (np.abs(a + b) > EPS)
    out[good] = (a[good] - b[good]) / (a[good] + b[good])
    return out


def ratio(a, b):
    out = np.full_like(a, np.nan, dtype=float)
    good = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > EPS)
    out[good] = a[good] / b[good]
    return out


def wind_masks(h, w, direction_from_deg):
    theta = math.radians((float(direction_from_deg) + 180) % 360)
    yy, xx = np.indices((h, w))
    cy, cx = (h - 1) / 2, (w - 1) / 2
    dx, dy_north = xx - cx, cy - yy
    along = dx * math.sin(theta) + dy_north * math.cos(theta)
    cross = dx * math.cos(theta) - dy_north * math.sin(theta)
    scale = float(min(h, w))
    down = (along >= 0.05 * scale) & (along <= 0.45 * scale) & (np.abs(cross) <= 0.14 * scale)
    up = (along <= -0.05 * scale) & (along >= -0.45 * scale) & (np.abs(cross) <= 0.14 * scale)
    return down, up


def meddiff(values, down, up):
    a = values[down & np.isfinite(values)]
    b = values[up & np.isfinite(values)]
    return float(np.median(a) - np.median(b)) if len(a) and len(b) else np.nan


def wind_features(root: Path, feature_path: Path):
    match_path = root / "outputs/46_multisite_wind_matches.csv"
    if not match_path.exists():
        raise SystemExit("請先執行 wind-match。")
    features = pd.read_csv(feature_path, low_memory=False)
    matches = pd.read_csv(match_path, low_memory=False)
    join = [c for c in ["sample_id", "site_id", "scene_id"] if c in features and c in matches]
    if not join:
        raise SystemExit("Feature table 與 wind table 沒有共同鍵。")
    add_cols = join + [c for c in [
        "wind_match_found", "wind_time_utc", "wind_time_difference_hours",
        "wind_speed_m_s", "wind_direction_from_deg", "wind_derivation",
        "wind_source_file", "wind_source_row",
    ] if c in matches]
    out = features.merge(matches[add_cols].drop_duplicates(join), on=join, how="left", validate="one_to_one")
    image_col = first_column(out, PATH_ALIASES)
    if not image_col:
        raise SystemExit("Feature table 找不到 image path。")

    extra = []
    for _, row in out.iterrows():
        rec = {
            "wind_feature_success": False, "wind_feature_error": "",
            "wind_b11_downwind_minus_upwind": np.nan,
            "wind_b12_downwind_minus_upwind": np.nan,
            "wind_ndvi_downwind_minus_upwind": np.nan,
            "wind_swir_ratio_downwind_minus_upwind": np.nan,
            "wind_swir_nd_downwind_minus_upwind": np.nan,
        }
        matched = str(row.get("wind_match_found", "")).lower() in {"true", "1", "yes"}
        direction = pd.to_numeric(pd.Series([row.get("wind_direction_from_deg")]), errors="coerce").iloc[0]
        if not matched or pd.isna(direction):
            extra.append(rec)
            continue
        try:
            with rasterio.open(Path(str(row[image_col]))) as src:
                array = src.read().astype(float)
                nodata = src.nodata
            if array.shape[0] < 6:
                raise ValueError("Expected B2,B3,B4,B8,B11,B12.")
            array = array[:6]
            if nodata is not None:
                array[array == nodata] = np.nan
            valid = valid_mask(array)
            down, up = wind_masks(array.shape[1], array.shape[2], direction)
            down, up = down & valid, up & valid
            b2, b3, b4, b8, b11, b12 = array
            rec.update({
                "wind_feature_success": bool(down.any() and up.any()),
                "wind_b11_downwind_minus_upwind": meddiff(b11, down, up),
                "wind_b12_downwind_minus_upwind": meddiff(b12, down, up),
                "wind_ndvi_downwind_minus_upwind": meddiff(nd(b8, b4), down, up),
                "wind_swir_ratio_downwind_minus_upwind": meddiff(ratio(b12, b11), down, up),
                "wind_swir_nd_downwind_minus_upwind": meddiff(nd(b11, b12), down, up),
            })
        except Exception as exc:
            rec["wind_feature_error"] = type(exc).__name__ + ": " + str(exc)
        extra.append(rec)

    out = pd.concat([out, pd.DataFrame(extra)], axis=1)
    out.to_csv(root / "outputs/47_multisite_s2_features_with_wind.csv", index=False)
    summary = out.groupby("site_id").agg(
        rows=("sample_id", "size"),
        wind_feature_success=("wind_feature_success", "sum"),
    ).reset_index()
    summary.to_csv(root / "outputs/47_multisite_s2_features_with_wind_summary.csv", index=False)
    text = [
        "WIND-ALIGNED SENTINEL-2 FEATURES",
        "=" * 100,
        "Rows: %d" % len(out),
        "Wind feature success: %d" % int(out["wind_feature_success"].sum()),
        "",
        summary.to_string(index=False),
    ]
    (root / "outputs/47_multisite_s2_features_with_wind_report.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["sensor-audit", "sensor-match", "feature-views", "wind-audit", "wind-match", "wind-features", "all"])
    p.add_argument("--project-root", type=Path, default=Path("/Users/happydoraaa/methane_release_project"))
    p.add_argument("--master", type=Path)
    p.add_argument("--features", type=Path)
    p.add_argument("--sensor-table", action="append", default=[])
    p.add_argument("--sensor-max-hours", type=float, default=24.0)
    p.add_argument("--sensor-spatial-km", type=float, default=15.0)
    p.add_argument("--wind-table", type=Path)
    p.add_argument("--wind-max-hours", type=float, default=12.0)
    p.add_argument("--wind-spatial-km", type=float, default=25.0)
    return p.parse_args()


def main():
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    master = resolve(root, args.master, "36_multisite_s2_master_table.csv")
    features = resolve(root, args.features, "39_multisite_s2_features.csv")
    commands = [args.command] if args.command != "all" else [
        "sensor-audit", "sensor-match", "feature-views", "wind-audit"
    ]
    audit = None
    for cmd in commands:
        print("\n### %s ###" % cmd.upper())
        if cmd == "sensor-audit":
            sensor_audit(root, args.sensor_table)
        elif cmd == "sensor-match":
            sensor_match(root, master, args.sensor_table, args.sensor_max_hours, args.sensor_spatial_km)
        elif cmd == "feature-views":
            feature_views(root, features)
        elif cmd == "wind-audit":
            audit = wind_audit(root)
        elif cmd == "wind-match":
            audit = wind_audit(root) if audit is None else audit
            if args.wind_table is not None:
                table = args.wind_table if args.wind_table.is_absolute() else root / args.wind_table
            elif not audit.empty:
                table = Path(audit.iloc[0]["source_file"])
            else:
                raise SystemExit("找不到有效 numerical wind table。")
            wind_match(root, master, table, args.wind_max_hours, args.wind_spatial_km)
        elif cmd == "wind-features":
            wind_features(root, features)


if __name__ == "__main__":
    main()
