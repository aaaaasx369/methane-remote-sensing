#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_methanesat_156_far_temporal_negatives.py

Search ALL 156 current MethaneSAT positive samples for SAME-LOCATION,
FAR-IN-TIME negative controls.

Primary design
--------------
Positive pool:
    111 old model-ready L4point positives
  + 45 conservatively unique new L4point positives
  = 156 expected positives

Candidate negative must:
  * cover the EXACT same positive latitude/longitude in MethaneSAT L3 XCH4
  * be a DIFFERENT L3 collection
  * have |delta time| >= 90 days
  * preferably >= 180 days; >= 300 days is "year-like"
  * pass a 480 m source-centered XCH4 valid-pixel QA
  * have NO MethaneSAT L4 point source within the configured radius
  * have NO known positive/release record in the master inventory near that
    location/time

Negative evidence tiers
-----------------------
1. CONFIRMED_NO_EMISSION_EXTERNAL
   Explicit master-inventory no-release / zero-release ground truth matches
   the candidate in space/time.

2. METHANESAT_AREA_LIKELY_NONEMITTING
   Same candidate collection has L4 area product and the local
   l4_retained_emitter band equals 0.

3. METHANESAT_AREA_BELOW_NOISE_PROXY
   l4_retained_emitter is unavailable, but local mean_flux is <= the
   collection's flux_noise_floor_kg_hr. This is kept separate because it is
   weaker than an explicit retained-emitter=0 flag.

4. STRONG_NO_DETECTION_ONLY
   Good L3 image + no nearby L4 point + no known positive/release record,
   but no independent zero-emission evidence.

IMPORTANT:
The script NEVER silently calls tier 2/3/4 "confirmed zero emission".

Resume safety
-------------
Each positive gets an atomic per-positive checkpoint JSON.
If interrupted, rerun the SAME command. Completed positives are skipped.
No imagery is downloaded by this script.

Official EE assets
------------------
L3:
  projects/edf-methanesat-ee/assets/public-preview/L3concentration
L4 point:
  projects/edf-methanesat-ee/assets/public-preview/L4point
L4 area v2:
  projects/edf-methanesat-ee/assets/public-preview/L4area_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import ee
except ImportError as exc:
    raise SystemExit(
        "Missing earthengine-api.\n"
        "Install with:\n"
        "  python -m pip install earthengine-api\n"
    ) from exc


L3_ASSET = "projects/edf-methanesat-ee/assets/public-preview/L3concentration"
L4_POINT_ASSET = "projects/edf-methanesat-ee/assets/public-preview/L4point"
L4_AREA_ASSET = "projects/edf-methanesat-ee/assets/public-preview/L4area_v2"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--old-inventory",
        default="~/Downloads/MethaneSAT_222_inventory.csv",
        help="Existing 222-row inventory. label=1 rows should yield 111 old positives.",
    )
    p.add_argument(
        "--new-positives",
        default=(
            "/Volumes/engg-leung/dora lin/MethaneSAT_MethaneFuse/"
            "04_l4point_refresh/13_valid_new_positive_unique_conservative.csv"
        ),
        help="45 conservatively unique new positives from the L4 refresh audit.",
    )
    p.add_argument(
        "--master",
        default="~/Downloads/All_Inventory_V3_MethaneSAT_AVIRIS3_EMIT.csv",
        help="Master inventory CSV used for known positive/release and confirmed zero-release checks.",
    )
    p.add_argument(
        "--out",
        default="~/methane_release_project/methanesat_156_far_temporal_negative_search",
    )
    p.add_argument(
        "--checkpoint-dir",
        default="~/methane_release_project/methanesat_156_far_temporal_negative_checkpoints",
    )
    p.add_argument(
        "--project",
        default="methane-release-gee",
    )

    p.add_argument("--min-abs-days", type=float, default=90.0)
    p.add_argument("--preferred-days", type=float, default=180.0)
    p.add_argument("--year-like-days", type=float, default=300.0)

    p.add_argument(
        "--crop-half-m",
        type=float,
        default=240.0,
        help="480m x 480m QA region around the exact positive coordinate.",
    )
    p.add_argument(
        "--qa-scale-m",
        type=float,
        default=45.0,
    )
    p.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.50,
    )

    p.add_argument(
        "--l4-point-reject-km",
        type=float,
        default=10.0,
        help="Conservative: any L4 point source within this radius rejects the candidate.",
    )
    p.add_argument(
        "--master-radius-km",
        type=float,
        default=2.0,
    )
    p.add_argument(
        "--master-time-hours",
        type=float,
        default=24.0,
    )

    p.add_argument(
        "--max-candidates-per-positive",
        type=int,
        default=0,
        help="0 = keep all eligible candidates. Otherwise retain top N per positive.",
    )
    p.add_argument(
        "--expected-old",
        type=int,
        default=111,
    )
    p.add_argument(
        "--expected-new",
        type=int,
        default=45,
    )
    p.add_argument(
        "--expected-total",
        type=int,
        default=156,
    )
    p.add_argument(
        "--restart",
        action="store_true",
        help="Clear ONLY search checkpoints/output CSVs; does not delete any previously downloaded imagery.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def initialize_ee(project: str):
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed.\n"
            "Run:\n"
            "  earthengine authenticate\n"
            f"Then rerun with --project {project!r}\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def norm_collection(v: Any) -> str:
    s = str(v if v is not None else "").strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    if s[:1].lower() == "c":
        s = s[1:]
    return s.upper()


def clean_str(v: Any) -> str:
    s = str(v if v is not None else "").strip()
    return "" if s.lower() in {"", "nan", "none", "null"} else s


def parse_notes_value(notes: str, key: str) -> str:
    m = re.search(
        rf"\b{re.escape(key)}\s*=\s*([^;,\s]+)",
        str(notes),
        flags=re.I,
    )
    return m.group(1).strip() if m else ""


def to_dt(v: Any) -> pd.Timestamp:
    if v is None:
        return pd.NaT
    try:
        x = pd.to_datetime(v, utc=True, errors="coerce")
    except Exception:
        return pd.NaT
    if isinstance(x, pd.DatetimeIndex):
        return x[0] if len(x) else pd.NaT
    return x


def combine_date_time(date_v: Any, time_v: Any = "") -> pd.Timestamp:
    d = clean_str(date_v)
    t = clean_str(time_v)
    if not d:
        return pd.NaT
    if t:
        return to_dt(f"{d} {t}")
    return to_dt(d)


def haversine_km(lat1, lon1, lat2, lon2):
    try:
        vals = [float(lat1), float(lon1), float(lat2), float(lon2)]
    except Exception:
        return np.nan
    if any(not np.isfinite(x) for x in vals):
        return np.nan

    lat1, lon1, lat2, lon2 = vals
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def scalar(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.isoformat()
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def write_json_atomic(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_csv_atomic(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def positive_key(source: str, sample_id: str, collection_id: str, lat: float, lon: float):
    text = "|".join([
        source,
        clean_str(sample_id),
        norm_collection(collection_id),
        f"{float(lat):.7f}",
        f"{float(lon):.7f}",
    ])
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def checkpoint_name(pos_id: str):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", pos_id)[:120] + ".json"


# ---------------------------------------------------------------------
# Positive inputs
# ---------------------------------------------------------------------

def load_old_positives(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    lc = {str(c).strip().lower(): c for c in df.columns}

    for name in ["latitude", "longitude", "label", "notes"]:
        if name not in lc:
            raise ValueError(
                f"Old inventory missing column {name!r}. Columns={list(df.columns)}"
            )

    label = pd.to_numeric(df[lc["label"]], errors="coerce")
    x = df[label.eq(1)].copy()

    scene_col = lc.get("scene/observation id")
    date_col = lc.get("date")
    time_col = lc.get("utc time")

    out = pd.DataFrame({
        "positive_source": "OLD_111",
        "positive_sample_id": (
            x[scene_col].fillna("").astype(str).values
            if scene_col else [f"OLD_POS_{i:04d}" for i in range(len(x))]
        ),
        "latitude": pd.to_numeric(x[lc["latitude"]], errors="coerce").values,
        "longitude": pd.to_numeric(x[lc["longitude"]], errors="coerce").values,
        "positive_date_inventory": (
            x[date_col].fillna("").astype(str).values
            if date_col else ""
        ),
        "positive_time_inventory": (
            x[time_col].fillna("").astype(str).values
            if time_col else ""
        ),
        "notes": x[lc["notes"]].fillna("").astype(str).values,
    })

    out["positive_collection_id"] = out["notes"].map(
        lambda s: norm_collection(parse_notes_value(s, "collection_id"))
    )
    out["positive_plume_id"] = out["notes"].map(
        lambda s: clean_str(parse_notes_value(s, "plume_id"))
    )
    out["positive_target_id"] = out["notes"].map(
        lambda s: clean_str(parse_notes_value(s, "target_id"))
    )

    out = out[
        out["latitude"].notna()
        & out["longitude"].notna()
        & out["positive_collection_id"].str.len().gt(0)
    ].copy()

    return out.reset_index(drop=True)


def first_existing_col(df: pd.DataFrame, names):
    cols = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None


def load_new_positives(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    cid_col = first_existing_col(df, ["collection_id"])
    lat_col = first_existing_col(df, ["latitude", "lat"])
    lon_col = first_existing_col(df, ["longitude", "lon"])
    plume_col = first_existing_col(df, ["plume_id"])
    target_col = first_existing_col(df, ["target_id"])
    date_col = first_existing_col(df, ["date"])
    fid_col = first_existing_col(df, ["feature_id", "refreshed_feature_id", "row_id"])
    tif_col = first_existing_col(df, ["expected_tif", "output_tif"])

    required = {"collection_id": cid_col, "latitude": lat_col, "longitude": lon_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            f"New-positive CSV missing {missing}. Columns={list(df.columns)}"
        )

    out = pd.DataFrame({
        "positive_source": "NEW_45",
        "latitude": pd.to_numeric(df[lat_col], errors="coerce"),
        "longitude": pd.to_numeric(df[lon_col], errors="coerce"),
        "positive_collection_id": df[cid_col].map(norm_collection),
        "positive_plume_id": (
            df[plume_col].fillna("").astype(str) if plume_col else ""
        ),
        "positive_target_id": (
            df[target_col].fillna("").astype(str) if target_col else ""
        ),
        "positive_date_inventory": (
            df[date_col].fillna("").astype(str) if date_col else ""
        ),
        "positive_time_inventory": "",
        "positive_local_tif": (
            df[tif_col].fillna("").astype(str) if tif_col else ""
        ),
    })

    if fid_col:
        ids = df[fid_col].fillna("").astype(str)
    else:
        ids = pd.Series([""] * len(df), index=df.index)

    built = []
    for i, r in out.iterrows():
        fid = clean_str(ids.iloc[i])
        if not fid:
            fid = positive_key(
                "NEW_45",
                f"new_{i}",
                r["positive_collection_id"],
                r["latitude"],
                r["longitude"],
            )
        built.append(f"MSAT_NEW_{fid}")
    out["positive_sample_id"] = built

    out["notes"] = ""
    out = out[
        out["latitude"].notna()
        & out["longitude"].notna()
        & out["positive_collection_id"].str.len().gt(0)
    ].copy()

    return out.reset_index(drop=True)


def assemble_positives(old_path: Path, new_path: Path, args) -> pd.DataFrame:
    old = load_old_positives(old_path)
    new = load_new_positives(new_path)

    print("Loaded old positives:", len(old))
    print("Loaded new positives:", len(new))

    if len(old) != args.expected_old:
        raise RuntimeError(
            f"Expected {args.expected_old} old positives, got {len(old)}.\n"
            "Stop here rather than silently building the wrong benchmark."
        )
    if len(new) != args.expected_new:
        raise RuntimeError(
            f"Expected {args.expected_new} new positives, got {len(new)}.\n"
            "Stop here rather than silently building the wrong benchmark."
        )

    pos = pd.concat([old, new], ignore_index=True, sort=False)

    pos["positive_id"] = [
        positive_key(
            str(r["positive_source"]),
            str(r["positive_sample_id"]),
            str(r["positive_collection_id"]),
            float(r["latitude"]),
            float(r["longitude"]),
        )
        for _, r in pos.iterrows()
    ]

    if pos["positive_id"].duplicated().any():
        dup = pos[pos["positive_id"].duplicated(False)]
        raise RuntimeError(
            "Duplicate positive IDs after assembling old+new positives:\n"
            + dup[["positive_sample_id", "positive_collection_id", "latitude", "longitude"]]
            .to_string(index=False)
        )

    if len(pos) != args.expected_total:
        raise RuntimeError(
            f"Expected total {args.expected_total}, got {len(pos)}."
        )

    return pos.reset_index(drop=True)


# ---------------------------------------------------------------------
# Master inventory
# ---------------------------------------------------------------------

def load_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Master inventory not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(path, sheet_name="All_Inventory")
        except Exception as exc:
            raise RuntimeError(
                f"Could not read All_Inventory from {path}: {type(exc).__name__}: {exc}\n"
                "Use the local CSV copy instead."
            ) from exc
    else:
        raise ValueError(f"Unsupported master format: {path}")

    lc = {str(c).strip().lower(): c for c in df.columns}
    needed = ["latitude", "longitude", "date", "label", "label type"]
    missing = [x for x in needed if x not in lc]
    if missing:
        raise ValueError(
            f"Master inventory missing {missing}. Columns={list(df.columns)}"
        )

    out = pd.DataFrame()
    out["lat"] = pd.to_numeric(df[lc["latitude"]], errors="coerce")
    out["lon"] = pd.to_numeric(df[lc["longitude"]], errors="coerce")
    out["date"] = df[lc["date"]].fillna("").astype(str)
    out["time"] = (
        df[lc["utc time"]].fillna("").astype(str)
        if "utc time" in lc else ""
    )
    out["label"] = pd.to_numeric(df[lc["label"]], errors="coerce")
    out["label_type"] = df[lc["label type"]].fillna("").astype(str)
    out["modality"] = (
        df[lc["ground truth modality"]].fillna("").astype(str)
        if "ground truth modality" in lc else ""
    )
    out["source_dataset"] = (
        df[lc["source dataset"]].fillna("").astype(str)
        if "source dataset" in lc else ""
    )
    out["site"] = (
        df[lc["site"]].fillna("").astype(str)
        if "site" in lc else ""
    )
    out["record_id"] = (
        df[lc["scene/observation id"]].fillna("").astype(str)
        if "scene/observation id" in lc
        else [f"master_{i}" for i in range(len(df))]
    )

    out["dt"] = [
        combine_date_time(d, t)
        for d, t in zip(out["date"], out["time"])
    ]
    out = out[
        out["lat"].notna()
        & out["lon"].notna()
        & out["dt"].notna()
        & out["label"].isin([0, 1])
    ].copy()

    # Explicit zero-release/no-release ground truth only.
    lt = out["label_type"].str.lower()
    mod = out["modality"].str.lower()
    out["is_explicit_confirmed_zero"] = (
        out["label"].eq(0)
        & (
            lt.str.contains("confirmed no-release", regex=False)
            | lt.str.contains("zero-release", regex=False)
            | lt.str.contains("no release", regex=False)
        )
        & mod.str.contains("controlled release", regex=False)
    )

    # Any known positive/release/plume evidence is a conflict.
    out["is_known_positive"] = out["label"].eq(1)

    return out.reset_index(drop=True)


def master_matches(master: pd.DataFrame, lat: float, lon: float, dt: pd.Timestamp,
                   radius_km: float, time_hours: float):
    if pd.isna(dt) or master.empty:
        return [], []

    # Cheap coordinate box first.
    deg_lat = radius_km / 111.0
    coslat = max(0.2, math.cos(math.radians(lat)))
    deg_lon = radius_km / (111.0 * coslat)

    sub = master[
        master["lat"].between(lat - deg_lat, lat + deg_lat)
        & master["lon"].between(lon - deg_lon, lon + deg_lon)
    ].copy()

    if sub.empty:
        return [], []

    delta_h = (sub["dt"] - dt).abs().dt.total_seconds() / 3600.0
    sub = sub[delta_h <= time_hours].copy()
    if sub.empty:
        return [], []

    dists = [
        haversine_km(lat, lon, r["lat"], r["lon"])
        for _, r in sub.iterrows()
    ]
    sub["distance_km"] = dists
    sub["delta_hours"] = [
        abs((r["dt"] - dt).total_seconds()) / 3600.0
        for _, r in sub.iterrows()
    ]
    sub = sub[sub["distance_km"] <= radius_km]

    positive = sub[sub["is_known_positive"]].to_dict("records")
    confirmed_zero = sub[sub["is_explicit_confirmed_zero"]].to_dict("records")
    return positive, confirmed_zero


# ---------------------------------------------------------------------
# Live L4 point catalogue, cached locally
# ---------------------------------------------------------------------

def fetch_live_l4_points() -> pd.DataFrame:
    fc = ee.FeatureCollection(L4_POINT_ASSET)
    info = fc.getInfo()

    rows = []
    for f in info.get("features", []):
        props = f.get("properties") or {}
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") or [np.nan, np.nan]
        rows.append({
            "collection_id": norm_collection(props.get("collection_id")),
            "plume_id": clean_str(props.get("plume_id")),
            "target_id": clean_str(props.get("target_id")),
            "flux": props.get("flux"),
            "lat": pd.to_numeric(coords[1] if len(coords) > 1 else np.nan, errors="coerce"),
            "lon": pd.to_numeric(coords[0] if len(coords) > 0 else np.nan, errors="coerce"),
        })
    return pd.DataFrame(rows)


def nearby_l4_points(l4: pd.DataFrame, collection_id: str, lat: float, lon: float,
                     radius_km: float):
    cid = norm_collection(collection_id)
    g = l4[l4["collection_id"].eq(cid)]
    hits = []
    for _, r in g.iterrows():
        d = haversine_km(lat, lon, r["lat"], r["lon"])
        if np.isfinite(d) and d <= radius_km:
            hits.append({
                "plume_id": r["plume_id"],
                "distance_km": float(d),
                "flux": scalar(r["flux"]),
                "target_id": r["target_id"],
            })
    hits.sort(key=lambda x: x["distance_km"])
    return hits


# ---------------------------------------------------------------------
# Earth Engine L3 search
# ---------------------------------------------------------------------

def resolve_l3_image_by_collection(collection_id: str, point: ee.Geometry):
    cid = norm_collection(collection_id)
    base = ee.ImageCollection(L3_ASSET).filterBounds(point)

    for candidate in [f"c{cid}", cid]:
        ic = base.filter(ee.Filter.eq("collection_id", candidate))
        n = int(ic.size().getInfo())
        if n > 0:
            return ee.Image(ic.first()), candidate
    return None, ""


def image_time_from_props(props: dict):
    for k in ["time_coverage_start", "time_start"]:
        if k in props and props[k] not in [None, ""]:
            x = to_dt(props[k])
            if not pd.isna(x):
                return x

    # system:time_start is milliseconds since epoch.
    ms = props.get("system:time_start")
    if ms not in [None, ""]:
        try:
            return pd.to_datetime(float(ms), unit="ms", utc=True)
        except Exception:
            pass
    return pd.NaT


def resolve_positive_time(row: pd.Series):
    point = ee.Geometry.Point([float(row["longitude"]), float(row["latitude"])])
    img, matched_cid = resolve_l3_image_by_collection(
        row["positive_collection_id"],
        point,
    )

    if img is not None:
        props = img.toDictionary([
            "collection_id",
            "target_id",
            "time_coverage_start",
            "time_coverage_end",
            "system:time_start",
            "system:index",
        ]).getInfo()
        dt = image_time_from_props(props)
        if not pd.isna(dt):
            return dt, matched_cid, props

    # Fallback to inventory date/time only if L3 metadata time could not resolve.
    dt = combine_date_time(
        row.get("positive_date_inventory", ""),
        row.get("positive_time_inventory", ""),
    )
    return dt, matched_cid, {}


def search_l3_candidates(row: pd.Series, positive_dt: pd.Timestamp, args):
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(args.crop_half_m).bounds()

    def add_qa(img):
        x = img.select("XCH4")
        # Fraction of pixels with a valid, unmasked XCH4 value in the exact
        # source-centered 480m region.
        valid = x.mask().reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=args.qa_scale_m,
            bestEffort=True,
            maxPixels=100000,
        ).get("XCH4")
        return img.set("local_valid_fraction", valid)

    ic = ee.ImageCollection(L3_ASSET).filterBounds(point).map(add_qa)
    info = ic.getInfo()

    rows = []
    for item in info.get("features", []):
        props = item.get("properties") or {}
        cid = norm_collection(props.get("collection_id"))
        if not cid:
            continue

        dt = image_time_from_props(props)
        if pd.isna(dt) or pd.isna(positive_dt):
            abs_days = np.nan
            signed_days = np.nan
        else:
            signed_days = (dt - positive_dt).total_seconds() / 86400.0
            abs_days = abs(signed_days)

        vf = pd.to_numeric(props.get("local_valid_fraction"), errors="coerce")

        rows.append({
            "candidate_collection_id": cid,
            "candidate_collection_id_raw": clean_str(props.get("collection_id")),
            "candidate_target_id": clean_str(props.get("target_id")),
            "candidate_time_start": None if pd.isna(dt) else dt.isoformat(),
            "candidate_time_end": clean_str(props.get("time_coverage_end")),
            "candidate_system_index": clean_str(props.get("system:index") or item.get("id")),
            "signed_delta_days": scalar(signed_days),
            "abs_delta_days": scalar(abs_days),
            "local_valid_fraction": scalar(vf),
        })

    # One logical row per candidate collection at this point.
    df = pd.DataFrame(rows)
    if len(df):
        df = (
            df.sort_values(
                ["candidate_collection_id", "local_valid_fraction"],
                ascending=[True, False],
                na_position="last",
            )
            .drop_duplicates("candidate_collection_id")
            .reset_index(drop=True)
        )
    return df


# ---------------------------------------------------------------------
# L4 area evidence
# ---------------------------------------------------------------------

def inspect_l4_area(collection_id: str, lat: float, lon: float, cache: dict):
    key = (
        norm_collection(collection_id),
        round(float(lat), 6),
        round(float(lon), 6),
    )
    if key in cache:
        return cache[key]

    cid = key[0]
    point = ee.Geometry.Point([lon, lat])
    base = ee.ImageCollection(L4_AREA_ASSET).filterBounds(point)

    chosen = None
    matched_cid = ""
    for variant in [f"c{cid}", cid]:
        ic = base.filter(ee.Filter.eq("collection_id", variant))
        try:
            n = int(ic.size().getInfo())
        except Exception:
            n = 0
        if n > 0:
            chosen = ee.Image(ic.first())
            matched_cid = variant
            break

    if chosen is None:
        out = {
            "l4area_available": False,
            "l4area_collection_id": "",
            "l4area_retained_emitter": None,
            "l4area_mean_flux": None,
            "l4area_lower_bound_flux": None,
            "l4area_upper_bound_flux": None,
            "l4area_flux_noise_floor": None,
            "l4area_evidence": "NO_L4AREA_PRODUCT",
        }
        cache[key] = out
        return out

    try:
        bands = chosen.bandNames().getInfo()
        props = chosen.toDictionary([
            "collection_id",
            "processing_id",
            "flux_noise_floor_kg_hr",
            "time_coverage_start",
            "time_coverage_end",
            "two_step_core",
        ]).getInfo()

        band_values = {}
        wanted = [
            b for b in [
                "l4_retained_emitter",
                "mean_flux",
                "lower_bound_flux",
                "upper_bound_flux",
            ] if b in bands
        ]

        if wanted:
            vals = chosen.select(wanted).reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point.buffer(100).bounds(),
                scale=4000,
                bestEffort=True,
                maxPixels=10000,
            ).getInfo()
            band_values.update(vals or {})

        retained = pd.to_numeric(
            band_values.get("l4_retained_emitter"), errors="coerce"
        )
        mean_flux = pd.to_numeric(
            band_values.get("mean_flux"), errors="coerce"
        )
        lb = pd.to_numeric(
            band_values.get("lower_bound_flux"), errors="coerce"
        )
        ub = pd.to_numeric(
            band_values.get("upper_bound_flux"), errors="coerce"
        )
        noise = pd.to_numeric(
            props.get("flux_noise_floor_kg_hr"), errors="coerce"
        )

        if np.isfinite(retained):
            if int(round(float(retained))) == 0:
                evidence = "L4AREA_RETAINED_EMITTER_0"
            elif int(round(float(retained))) == 1:
                evidence = "L4AREA_RETAINED_EMITTER_1"
            else:
                evidence = "L4AREA_RETAINED_EMITTER_OTHER"
        elif np.isfinite(mean_flux) and np.isfinite(noise):
            if float(mean_flux) <= float(noise):
                evidence = "L4AREA_MEAN_AT_OR_BELOW_NOISE_FLOOR"
            else:
                evidence = "L4AREA_MEAN_ABOVE_NOISE_FLOOR"
        else:
            evidence = "L4AREA_PRESENT_BUT_LOCAL_STATUS_UNRESOLVED"

        out = {
            "l4area_available": True,
            "l4area_collection_id": norm_collection(
                props.get("collection_id") or matched_cid
            ),
            "l4area_processing_id": clean_str(props.get("processing_id")),
            "l4area_two_step_core": clean_str(props.get("two_step_core")),
            "l4area_retained_emitter": scalar(retained),
            "l4area_mean_flux": scalar(mean_flux),
            "l4area_lower_bound_flux": scalar(lb),
            "l4area_upper_bound_flux": scalar(ub),
            "l4area_flux_noise_floor": scalar(noise),
            "l4area_evidence": evidence,
        }
    except Exception as exc:
        out = {
            "l4area_available": True,
            "l4area_collection_id": cid,
            "l4area_retained_emitter": None,
            "l4area_mean_flux": None,
            "l4area_lower_bound_flux": None,
            "l4area_upper_bound_flux": None,
            "l4area_flux_noise_floor": None,
            "l4area_evidence": "L4AREA_QUERY_ERROR",
            "l4area_error": f"{type(exc).__name__}: {exc}",
        }

    cache[key] = out
    return out


# ---------------------------------------------------------------------
# Candidate classification / ranking
# ---------------------------------------------------------------------

def time_tier(abs_days, args):
    if not np.isfinite(abs_days):
        return "TIME_UNKNOWN", 0
    if abs_days >= args.year_like_days:
        return "YEAR_LIKE_300D_PLUS", 3
    if abs_days >= args.preferred_days:
        return "PREFERRED_180D_PLUS", 2
    if abs_days >= args.min_abs_days:
        return "MINIMUM_90D_PLUS", 1
    return "TOO_CLOSE", 0


def evidence_rank(label):
    return {
        "CONFIRMED_NO_EMISSION_EXTERNAL": 4,
        "METHANESAT_AREA_LIKELY_NONEMITTING": 3,
        "METHANESAT_AREA_BELOW_NOISE_PROXY": 2,
        "STRONG_NO_DETECTION_ONLY": 1,
    }.get(label, 0)


def classify_candidate(pos: pd.Series, cand: dict, l4points: pd.DataFrame,
                       master: pd.DataFrame, area_cache: dict, args):
    rec = {
        "positive_id": pos["positive_id"],
        "positive_source": pos["positive_source"],
        "positive_sample_id": pos["positive_sample_id"],
        "positive_collection_id": norm_collection(pos["positive_collection_id"]),
        "positive_time": pos["positive_time"],
        "latitude": float(pos["latitude"]),
        "longitude": float(pos["longitude"]),
        **cand,
    }

    cid = norm_collection(cand["candidate_collection_id"])
    dt = to_dt(cand["candidate_time_start"])
    abs_days = pd.to_numeric(cand["abs_delta_days"], errors="coerce")
    vf = pd.to_numeric(cand["local_valid_fraction"], errors="coerce")

    rec["same_positive_collection"] = (
        cid == norm_collection(pos["positive_collection_id"])
    )

    tlabel, trank = time_tier(abs_days, args)
    rec["time_tier"] = tlabel
    rec["time_rank"] = trank

    # Basic timing / QA rejection.
    if rec["same_positive_collection"]:
        rec["candidate_status"] = "REJECT_SAME_POSITIVE_COLLECTION"
        rec["negative_evidence_tier"] = ""
        return rec

    if not np.isfinite(abs_days) or abs_days < args.min_abs_days:
        rec["candidate_status"] = "REJECT_TOO_CLOSE_LT_90D"
        rec["negative_evidence_tier"] = ""
        return rec

    if not np.isfinite(vf) or vf < args.min_valid_fraction:
        rec["candidate_status"] = "REJECT_L3_QA"
        rec["negative_evidence_tier"] = ""
        return rec

    # L4 point source exclusion.
    nearby = nearby_l4_points(
        l4points,
        cid,
        float(pos["latitude"]),
        float(pos["longitude"]),
        args.l4_point_reject_km,
    )
    rec["nearby_l4_point_count"] = len(nearby)
    rec["nearest_l4_point_km"] = (
        nearby[0]["distance_km"] if nearby else None
    )
    rec["nearest_l4_point_plume_id"] = (
        nearby[0]["plume_id"] if nearby else ""
    )
    rec["nearest_l4_point_flux"] = (
        nearby[0]["flux"] if nearby else None
    )

    if nearby:
        rec["candidate_status"] = "REJECT_L4_POINT_WITHIN_RADIUS"
        rec["negative_evidence_tier"] = ""
        return rec

    # Master inventory cross-check.
    known_pos, confirmed_zero = master_matches(
        master,
        float(pos["latitude"]),
        float(pos["longitude"]),
        dt,
        args.master_radius_km,
        args.master_time_hours,
    )
    rec["master_known_positive_matches"] = len(known_pos)
    rec["master_confirmed_zero_matches"] = len(confirmed_zero)
    rec["master_confirmed_zero_ids"] = "|".join(
        clean_str(x.get("record_id")) for x in confirmed_zero
    )

    if known_pos:
        rec["candidate_status"] = "REJECT_MASTER_KNOWN_POSITIVE"
        rec["negative_evidence_tier"] = ""
        return rec

    # Strongest evidence: explicit external confirmed zero-release.
    if confirmed_zero:
        rec["candidate_status"] = "ELIGIBLE"
        rec["negative_evidence_tier"] = "CONFIRMED_NO_EMISSION_EXTERNAL"
        rec["negative_evidence_rank"] = 4
        rec.update({
            "l4area_available": None,
            "l4area_evidence": "NOT_REQUIRED_EXTERNAL_CONFIRMED_ZERO",
        })
        return rec

    # MethaneSAT area-product evidence.
    area = inspect_l4_area(
        cid,
        float(pos["latitude"]),
        float(pos["longitude"]),
        area_cache,
    )
    rec.update(area)

    ae = area.get("l4area_evidence", "")

    if ae == "L4AREA_RETAINED_EMITTER_1":
        rec["candidate_status"] = "REJECT_L4AREA_EMITTER"
        rec["negative_evidence_tier"] = ""
        return rec

    if ae == "L4AREA_RETAINED_EMITTER_0":
        tier = "METHANESAT_AREA_LIKELY_NONEMITTING"
    elif ae == "L4AREA_MEAN_AT_OR_BELOW_NOISE_FLOOR":
        tier = "METHANESAT_AREA_BELOW_NOISE_PROXY"
    else:
        tier = "STRONG_NO_DETECTION_ONLY"

    rec["candidate_status"] = "ELIGIBLE"
    rec["negative_evidence_tier"] = tier
    rec["negative_evidence_rank"] = evidence_rank(tier)
    return rec


def choose_best_per_positive(eligible: pd.DataFrame):
    if eligible.empty:
        return eligible.copy()

    x = eligible.copy()
    for c in ["negative_evidence_rank", "time_rank", "abs_delta_days", "local_valid_fraction"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.sort_values(
        [
            "positive_id",
            "negative_evidence_rank",
            "time_rank",
            "abs_delta_days",
            "local_valid_fraction",
        ],
        ascending=[True, False, False, False, False],
        na_position="last",
    )

    return x.drop_duplicates("positive_id", keep="first").reset_index(drop=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    old_path = Path(args.old_inventory).expanduser()
    new_path = Path(args.new_positives).expanduser()
    master_path = Path(args.master).expanduser()
    outdir = Path(args.out).expanduser()
    ckpt_dir = Path(args.checkpoint_dir).expanduser()

    if args.restart:
        if ckpt_dir.exists():
            import shutil
            shutil.rmtree(ckpt_dir)
        if outdir.exists():
            # Search outputs only; never touch any imagery directories.
            for p in outdir.glob("*.csv"):
                p.unlink(missing_ok=True)
            for p in outdir.glob("*.md"):
                p.unlink(missing_ok=True)

    outdir.mkdir(parents=True, exist_ok=True)
    per_pos_dir = ckpt_dir / "per_positive"
    per_pos_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("METHANESAT 156 — FAR SAME-LOCATION TEMPORAL NEGATIVE SEARCH")
    print("=" * 88)
    print("Rule: exact location, different collection, |Δt| >= 90 d")
    print("Preference: >=180 d; year-like: >=300 d")
    print("No imagery is downloaded in this phase.")
    print()

    positives = assemble_positives(old_path, new_path, args)
    master = load_master(master_path)

    initialize_ee(args.project)

    print("Fetching current official L4 point catalogue...")
    l4points = fetch_live_l4_points()
    print("Current L4 point rows:", len(l4points))
    print()

    # Resolve positive L3 acquisition times once.
    resolved_rows = []
    for i, r in positives.iterrows():
        dt, matched_cid, props = resolve_positive_time(r)
        rr = r.to_dict()
        rr["positive_time"] = None if pd.isna(dt) else dt.isoformat()
        rr["positive_l3_matched_collection_id"] = norm_collection(
            props.get("collection_id") or matched_cid
        )
        rr["positive_l3_target_id"] = clean_str(props.get("target_id"))
        rr["positive_l3_system_index"] = clean_str(props.get("system:index"))
        resolved_rows.append(rr)

        if (i + 1) % 25 == 0 or i + 1 == len(positives):
            print(f"Resolved positive times: {i+1}/{len(positives)}")

    positives = pd.DataFrame(resolved_rows)

    missing_pos_time = positives["positive_time"].fillna("").eq("")
    if missing_pos_time.any():
        bad = positives[missing_pos_time]
        raise RuntimeError(
            f"{len(bad)} positives have no resolvable acquisition time.\n"
            "Stop rather than silently apply wrong temporal gaps.\n"
            + bad[
                ["positive_sample_id", "positive_collection_id", "latitude", "longitude"]
            ].to_string(index=False)
        )

    write_csv_atomic(positives, outdir / "00_all_156_positives.csv")

    area_cache = {}
    errors = []

    for i, pos in positives.iterrows():
        pos_id = str(pos["positive_id"])
        ckpt = per_pos_dir / checkpoint_name(pos_id)

        if ckpt.exists():
            print(
                f"[{i+1}/{len(positives)}] {pos['positive_sample_id']} "
                "SKIP checkpoint"
            )
            continue

        print()
        print("-" * 88)
        print(f"[{i+1}/{len(positives)}] {pos['positive_sample_id']}")
        print(
            "collection:",
            pos["positive_collection_id"],
            "| time:",
            pos["positive_time"],
        )
        print(
            "location:",
            f"{float(pos['latitude']):.6f}",
            f"{float(pos['longitude']):.6f}",
        )

        try:
            positive_dt = to_dt(pos["positive_time"])
            raw = search_l3_candidates(pos, positive_dt, args)
            print("L3 observations covering exact point:", len(raw))

            classified = []
            for _, c in raw.iterrows():
                rec = classify_candidate(
                    pos,
                    c.to_dict(),
                    l4points,
                    master,
                    area_cache,
                    args,
                )
                classified.append(rec)

            eligible = [
                r for r in classified
                if r.get("candidate_status") == "ELIGIBLE"
            ]

            # Optional cap AFTER classification/ranking.
            if args.max_candidates_per_positive > 0 and eligible:
                tmp = pd.DataFrame(eligible)
                tmp = tmp.sort_values(
                    [
                        "negative_evidence_rank",
                        "time_rank",
                        "abs_delta_days",
                        "local_valid_fraction",
                    ],
                    ascending=[False, False, False, False],
                    na_position="last",
                ).head(args.max_candidates_per_positive)
                keep_keys = set(
                    zip(tmp["candidate_collection_id"], tmp["candidate_time_start"])
                )
                for r in classified:
                    if r.get("candidate_status") == "ELIGIBLE":
                        k = (
                            r.get("candidate_collection_id"),
                            r.get("candidate_time_start"),
                        )
                        if k not in keep_keys:
                            r["candidate_status"] = "ELIGIBLE_NOT_TOP_N"

            payload = {
                "positive": {
                    k: scalar(v) for k, v in pos.to_dict().items()
                },
                "candidates": [
                    {k: scalar(v) for k, v in r.items()}
                    for r in classified
                ],
                "completed_at_unix": time.time(),
            }
            write_json_atomic(ckpt, payload)

            counts = pd.Series(
                [r.get("candidate_status") for r in classified]
            ).value_counts()
            print("Candidate status:")
            for k, v in counts.items():
                print(f"  {k}: {v}")

            ev = pd.Series(
                [
                    r.get("negative_evidence_tier")
                    for r in classified
                    if r.get("candidate_status") == "ELIGIBLE"
                ]
            ).value_counts()
            if len(ev):
                print("Eligible evidence:")
                for k, v in ev.items():
                    print(f"  {k}: {v}")
            else:
                print("Eligible evidence: NONE")

        except KeyboardInterrupt:
            print("\nInterrupted. Completed positive checkpoints are safe.")
            raise
        except Exception as exc:
            err = {
                "positive_id": pos_id,
                "positive_sample_id": pos["positive_sample_id"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(err)
            print("ERROR:", err["error"])

            # Save error checkpoint too, so it is visible. It is NOT considered
            # complete for final aggregation and can be removed/retried.
            write_json_atomic(
                ckpt.with_suffix(".error.json"),
                err,
            )

    # Aggregate only successful checkpoints.
    all_rows = []
    completed_ids = set()

    for p in sorted(per_pos_dir.glob("*.json")):
        if p.name.endswith(".error.json"):
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        pos_obj = obj.get("positive", {})
        pid = clean_str(pos_obj.get("positive_id"))
        if pid:
            completed_ids.add(pid)
        all_rows.extend(obj.get("candidates", []))

    all_candidates = pd.DataFrame(all_rows)
    write_csv_atomic(
        all_candidates,
        outdir / "01_all_far_temporal_candidates.csv",
    )

    if len(all_candidates):
        eligible = all_candidates[
            all_candidates["candidate_status"].eq("ELIGIBLE")
        ].copy()
    else:
        eligible = pd.DataFrame()

    if len(eligible):
        eligible = eligible.sort_values(
            [
                "positive_id",
                "negative_evidence_rank",
                "time_rank",
                "abs_delta_days",
                "local_valid_fraction",
            ],
            ascending=[True, False, False, False, False],
            na_position="last",
        ).reset_index(drop=True)

    write_csv_atomic(
        eligible,
        outdir / "02_eligible_negative_candidates.csv",
    )

    best = choose_best_per_positive(eligible)
    write_csv_atomic(
        best,
        outdir / "03_best_one_negative_per_positive.csv",
    )

    confirmed = (
        eligible[
            eligible["negative_evidence_tier"].eq(
                "CONFIRMED_NO_EMISSION_EXTERNAL"
            )
        ].copy()
        if len(eligible) else pd.DataFrame()
    )
    write_csv_atomic(
        confirmed,
        outdir / "04_confirmed_no_emission_only.csv",
    )

    area_nonemit = (
        eligible[
            eligible["negative_evidence_tier"].isin([
                "METHANESAT_AREA_LIKELY_NONEMITTING",
                "METHANESAT_AREA_BELOW_NOISE_PROXY",
            ])
        ].copy()
        if len(eligible) else pd.DataFrame()
    )
    write_csv_atomic(
        area_nonemit,
        outdir / "05_l4area_nonemitting_or_below_noise.csv",
    )

    # Positive-level summary.
    summaries = []
    for _, pos in positives.iterrows():
        pid = pos["positive_id"]
        sub = (
            all_candidates[all_candidates["positive_id"].eq(pid)]
            if len(all_candidates) else pd.DataFrame()
        )
        eli = (
            sub[sub["candidate_status"].eq("ELIGIBLE")]
            if len(sub) else pd.DataFrame()
        )
        b = best[best["positive_id"].eq(pid)] if len(best) else pd.DataFrame()

        rec = {
            "positive_id": pid,
            "positive_source": pos["positive_source"],
            "positive_sample_id": pos["positive_sample_id"],
            "positive_collection_id": pos["positive_collection_id"],
            "positive_time": pos["positive_time"],
            "latitude": pos["latitude"],
            "longitude": pos["longitude"],
            "search_checkpoint_complete": pid in completed_ids,
            "all_candidate_rows": len(sub),
            "eligible_candidate_rows": len(eli),
            "confirmed_external_zero_count": (
                int(
                    eli["negative_evidence_tier"]
                    .eq("CONFIRMED_NO_EMISSION_EXTERNAL")
                    .sum()
                ) if len(eli) else 0
            ),
            "l4area_nonemitting_count": (
                int(
                    eli["negative_evidence_tier"]
                    .eq("METHANESAT_AREA_LIKELY_NONEMITTING")
                    .sum()
                ) if len(eli) else 0
            ),
            "l4area_below_noise_proxy_count": (
                int(
                    eli["negative_evidence_tier"]
                    .eq("METHANESAT_AREA_BELOW_NOISE_PROXY")
                    .sum()
                ) if len(eli) else 0
            ),
            "no_detection_only_count": (
                int(
                    eli["negative_evidence_tier"]
                    .eq("STRONG_NO_DETECTION_ONLY")
                    .sum()
                ) if len(eli) else 0
            ),
            "has_best_negative": len(b) > 0,
        }

        if len(b):
            rr = b.iloc[0]
            rec.update({
                "best_negative_collection_id": rr["candidate_collection_id"],
                "best_negative_time": rr["candidate_time_start"],
                "best_abs_delta_days": rr["abs_delta_days"],
                "best_time_tier": rr["time_tier"],
                "best_valid_fraction": rr["local_valid_fraction"],
                "best_negative_evidence_tier": rr["negative_evidence_tier"],
                "best_l4area_evidence": rr.get("l4area_evidence", ""),
            })

        summaries.append(rec)

    summary_df = pd.DataFrame(summaries)
    write_csv_atomic(
        summary_df,
        outdir / "06_positive_level_summary.csv",
    )

    no_pair = summary_df[~summary_df["has_best_negative"]].copy()
    write_csv_atomic(
        no_pair,
        outdir / "07_positives_without_eligible_negative.csv",
    )

    # Strict primary pair sets.
    strict_external = best[
        best["negative_evidence_tier"].eq("CONFIRMED_NO_EMISSION_EXTERNAL")
    ].copy() if len(best) else pd.DataFrame()
    write_csv_atomic(
        strict_external,
        outdir / "08_primary_pairs_external_confirmed_only.csv",
    )

    sensor_strong = best[
        best["negative_evidence_tier"].isin([
            "CONFIRMED_NO_EMISSION_EXTERNAL",
            "METHANESAT_AREA_LIKELY_NONEMITTING",
            "METHANESAT_AREA_BELOW_NOISE_PROXY",
        ])
    ].copy() if len(best) else pd.DataFrame()
    write_csv_atomic(
        sensor_strong,
        outdir / "09_primary_pairs_strongest_available.csv",
    )

    # Errors.
    if errors:
        write_csv_atomic(pd.DataFrame(errors), outdir / "10_errors.csv")
    else:
        write_csv_atomic(
            pd.DataFrame(columns=["positive_id", "positive_sample_id", "error"]),
            outdir / "10_errors.csv",
        )

    # Text summary.
    evidence_counts = (
        best["negative_evidence_tier"].value_counts()
        if len(best) else pd.Series(dtype=int)
    )
    time_counts = (
        best["time_tier"].value_counts()
        if len(best) else pd.Series(dtype=int)
    )

    lines = [
        "# MethaneSAT 156 far same-location temporal-negative search",
        "",
        "## Positive pool",
        f"- Old positives: {args.expected_old}",
        f"- New positives: {args.expected_new}",
        f"- Total positives: {len(positives)}",
        "",
        "## Search rule",
        "- Same exact positive coordinate",
        "- Different MethaneSAT L3 collection",
        f"- Minimum absolute temporal separation: {args.min_abs_days:g} days",
        f"- Preferred: >= {args.preferred_days:g} days",
        f"- Year-like: >= {args.year_like_days:g} days",
        f"- L3 local valid fraction >= {args.min_valid_fraction:.2f}",
        f"- Reject any L4 point source within {args.l4_point_reject_km:g} km",
        f"- Reject known positive/release records within {args.master_radius_km:g} km and ±{args.master_time_hours:g} h",
        "",
        "## Completion",
        f"- Positive checkpoints complete: {len(completed_ids)} / {len(positives)}",
        f"- Errors this run: {len(errors)}",
        "",
        "## Best one negative per positive",
        f"- Positives with an eligible pair: {len(best)} / {len(positives)}",
        f"- Positives without an eligible pair: {len(no_pair)}",
        "",
        "### Evidence tiers among selected best pairs",
    ]
    if len(evidence_counts):
        for k, v in evidence_counts.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- None")

    lines += ["", "### Temporal separation among selected best pairs"]
    if len(time_counts):
        for k, v in time_counts.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- None")

    lines += [
        "",
        "## Strict interpretation",
        f"- Externally confirmed no-emission pairs: {len(strict_external)}",
        f"- Strongest available pairs (external confirmed or L4-area below/noise/non-emitter): {len(sensor_strong)}",
        "",
        "Do NOT call STRONG_NO_DETECTION_ONLY rows 'confirmed zero emission'.",
        "The script intentionally keeps confirmed external zero-release, MethaneSAT area-product evidence, and no-detection-only evidence separate.",
    ]

    (outdir / "SUMMARY_156_FAR_TEMPORAL_NEGATIVES.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)
    print("Total positives                         :", len(positives))
    print("Completed positive searches             :", len(completed_ids))
    print("Positives with >=1 eligible negative    :", len(best))
    print("Positives without eligible negative     :", len(no_pair))
    print("Externally confirmed zero-emission pairs:", len(strict_external))
    print("Strongest available primary pairs       :", len(sensor_strong))
    print()
    print("Selected evidence tiers:")
    if len(evidence_counts):
        print(evidence_counts.to_string())
    else:
        print("NONE")
    print()
    print("Selected time tiers:")
    if len(time_counts):
        print(time_counts.to_string())
    else:
        print("NONE")
    print()
    print("Output:", outdir)
    print("Upload these first:")
    for fn in [
        "SUMMARY_156_FAR_TEMPORAL_NEGATIVES.md",
        "03_best_one_negative_per_positive.csv",
        "06_positive_level_summary.csv",
        "07_positives_without_eligible_negative.csv",
        "09_primary_pairs_strongest_available.csv",
        "10_errors.csv",
    ]:
        print(" ", outdir / fn)


if __name__ == "__main__":
    main()
