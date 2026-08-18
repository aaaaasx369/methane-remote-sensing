#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_methane_event_table.py

Purpose:
1. Download public data for:
   - 2023 Scientific Reports methane satellite single-blind test
   - 2024 AMT nine methane-sensing satellite systems test

2. Search all CSV / Excel / JSON files.

3. Produce:
   outputs/01_tabular_file_index.csv
   outputs/02_candidate_event_rows.csv
   outputs/03_event_table_draft.csv
   outputs/04_gee_check_events.js

4. The generated GEE JS file can be pasted into Google Earth Engine Code Editor
   to check whether Sentinel-2 / Landsat 8 / Landsat 9 / EMIT images exist
   at each event time and location.

Important:
This script creates a DRAFT event table. Because the repositories may contain
many analysis tables, you still need to inspect outputs/01_tabular_file_index.csv
and outputs/03_event_table_draft.csv.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


# ============================================================
# Basic paths
# ============================================================

ROOT = Path.cwd()
RAW_DIR = ROOT / "raw_data"
OUT_DIR = ROOT / "outputs"

RAW_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Data sources
# ============================================================

GIT_REPOS = {
    "2023_SatelliteTesting": "https://github.com/esherwin/SatelliteTesting.git",
    "2023_Controlled_Release_2021": "https://github.com/JSRuthe/Controlled_Release_2021.git",
}

ZENODO_2024_URL = (
    "https://zenodo.org/records/10149992/files/"
    "sahar-elabbadi%2FSU-Controlled-Releases-2022-publish.zip?download=1"
)

ZENODO_2024_ZIP = RAW_DIR / "SU-Controlled-Releases-2022-publish.zip"
ZENODO_2024_DIR = RAW_DIR / "2024_SU_Controlled_Releases"


# ============================================================
# Known fixed release locations from the papers
# PDF/paper coordinates are [latitude, longitude]
# GEE uses [longitude, latitude]
# ============================================================

KNOWN_SITE_INFO = {
    "2023": {
        "paper": "2023_Scientific_Reports",
        "lat": 33.630645,
        "lon": -114.489150,
        "site_name": "Ehrenberg_AZ_release_stack",
    },
    "2024": {
        "paper": "2024_AMT",
        "lat": 32.8218205,
        "lon": -111.7857730,
        "site_name": "Casa_Grande_AZ_release_stacks",
    },
}


# ============================================================
# Helper functions
# ============================================================

def run_command(cmd: List[str], cwd: Optional[Path] = None) -> bool:
    print(f"\n[RUN] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
        return True
    except Exception as e:
        print(f"[WARN] Command failed: {e}")
        return False


def has_command(name: str) -> bool:
    return shutil.which(name) is not None


def download_file(url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SKIP] Already downloaded: {out_path}")
        return

    print(f"[DOWNLOAD] {url}")
    print(f"[TO] {out_path}")

    with urllib.request.urlopen(url) as response:
        total = response.length or 0
        with open(out_path, "wb") as f:
            if total:
                with tqdm(total=total, unit="B", unit_scale=True) as pbar:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                shutil.copyfileobj(response, f)


def clone_repo(name: str, url: str) -> Path:
    dest = RAW_DIR / name

    if dest.exists():
        print(f"[SKIP] Repo already exists: {dest}")
        return dest

    if not has_command("git"):
        raise RuntimeError(
            "git is not installed. Please install git, or manually download the GitHub ZIP files."
        )

    ok = run_command(["git", "clone", url, str(dest)])
    if not ok:
        raise RuntimeError(f"Failed to clone {url}")

    return dest


def unzip_file(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[SKIP] Already extracted: {out_dir}")
        return

    out_dir.mkdir(exist_ok=True)
    print(f"[UNZIP] {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def download_all_sources() -> None:
    print("\n==============================")
    print("1. Downloading public sources")
    print("==============================")

    for name, url in GIT_REPOS.items():
        clone_repo(name, url)

    download_file(ZENODO_2024_URL, ZENODO_2024_ZIP)
    unzip_file(ZENODO_2024_ZIP, ZENODO_2024_DIR)


def find_tabular_files(root: Path) -> List[Path]:
    suffixes = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl"}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            files.append(p)
    return sorted(files)


def infer_paper_from_path(path: Path) -> str:
    s = str(path).lower()

    if "su-controlled" in s or "2022" in s or "2024" in s or "zenodo" in s:
        return "2024_AMT"

    if "satellitetesting" in s or "controlled_release_2021" in s or "2021" in s:
        return "2023_Scientific_Reports"

    return "unknown"


def infer_site_from_path(path: Path) -> Dict[str, object]:
    paper = infer_paper_from_path(path)
    if paper == "2024_AMT":
        return KNOWN_SITE_INFO["2024"]
    if paper == "2023_Scientific_Reports":
        return KNOWN_SITE_INFO["2023"]
    return {"paper": paper, "lat": None, "lon": None, "site_name": "unknown"}


def safe_read_csv(path: Path, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    seps = [None, ",", "\t", ";"]

    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(
                    path,
                    nrows=nrows,
                    encoding=enc,
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip",
                )
            except Exception:
                continue

    return None


def safe_read_excel(path: Path, nrows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    try:
        sheets = pd.read_excel(path, sheet_name=None, nrows=nrows)
        return sheets
    except Exception:
        return {}


def safe_read_json(path: Path, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_json(path, lines=True)
        if nrows is not None:
            df = df.head(nrows)
        return df
    except Exception:
        pass

    try:
        df = pd.read_json(path)
        if nrows is not None:
            df = df.head(nrows)
        return df
    except Exception:
        return None


def read_tabular_file(path: Path, nrows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    suffix = path.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        df = safe_read_csv(path, nrows=nrows)
        return {"default": df} if df is not None else {}

    if suffix in {".xlsx", ".xls"}:
        return safe_read_excel(path, nrows=nrows)

    if suffix in {".json", ".jsonl"}:
        df = safe_read_json(path, nrows=nrows)
        return {"default": df} if df is not None else {}

    return {}


# ============================================================
# Column detection
# ============================================================

KEYWORD_GROUPS = {
    "datetime": [
        "datetime", "date", "time", "utc", "timestamp",
        "overpass", "acquisition", "start", "end"
    ],
    "satellite": [
        "satellite", "sat", "sensor", "instrument", "platform",
        "constellation", "spacecraft"
    ],
    "team": [
        "team", "operator", "analyst", "group", "organization", "org"
    ],
    "emission": [
        "emission", "release", "flow", "rate", "metered",
        "methane", "ch4", "kg", "t/h", "tph", "kg/h", "kgh"
    ],
    "estimate": [
        "estimate", "estimated", "reported", "retrieval", "quantification",
        "prediction", "submitted"
    ],
    "wind": [
        "wind", "ws", "wd", "speed", "direction", "anemometer"
    ],
    "location": [
        "lat", "latitude", "lon", "lng", "longitude", "coord", "location"
    ],
    "quality": [
        "cloud", "filter", "filtered", "task", "tasked", "valid",
        "quality", "flag", "outcome"
    ],
}


def normalize_colname(c: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def column_matches(col: str, keywords: List[str]) -> bool:
    c = normalize_colname(col)
    return any(k.replace("/", "_").replace("-", "_").lower() in c for k in keywords)


def score_columns(columns: List[str]) -> Tuple[int, Dict[str, List[str]]]:
    matched = {}
    score = 0

    for group, kws in KEYWORD_GROUPS.items():
        cols = [c for c in columns if column_matches(c, kws)]
        if cols:
            matched[group] = cols
            score += 1

    return score, matched


def find_first_col(
    columns: List[str],
    include_keywords: List[str],
    exclude_keywords: Optional[List[str]] = None,
) -> Optional[str]:
    exclude_keywords = exclude_keywords or []

    candidates = []
    for c in columns:
        cn = normalize_colname(c)
        include = any(k.lower().replace("/", "_") in cn for k in include_keywords)
        exclude = any(k.lower().replace("/", "_") in cn for k in exclude_keywords)
        if include and not exclude:
            candidates.append(c)

    if not candidates:
        return None

    return candidates[0]


def parse_datetime_series(df: pd.DataFrame, columns: List[str]) -> Tuple[Optional[pd.Series], Optional[str]]:
    # Prefer columns with both date/time/utc/overpass in the name
    preferred_keywords = [
        ["datetime"],
        ["timestamp"],
        ["utc"],
        ["overpass"],
        ["acquisition"],
        ["date"],
        ["time"],
    ]

    for kws in preferred_keywords:
        for c in columns:
            cn = normalize_colname(c)
            if any(k in cn for k in kws):
                s = pd.to_datetime(df[c], errors="coerce", utc=True)
                if s.notna().sum() > 0:
                    return s, c

    # Try combining separate date and time columns
    date_col = find_first_col(columns, ["date"])
    time_col = find_first_col(columns, ["time", "utc", "overpass"])
    if date_col and time_col and date_col != time_col:
        combined = df[date_col].astype(str) + " " + df[time_col].astype(str)
        s = pd.to_datetime(combined, errors="coerce", utc=True)
        if s.notna().sum() > 0:
            return s, f"{date_col}+{time_col}"

    return None, None


def infer_numeric_col(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    for c in cols:
        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.notna().sum() > 0:
            return c
    return None


def infer_emission_tph(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.Series, Optional[str], str]:
    """
    Try to infer true metered emission rate.
    Returns:
    - emission_tph series
    - original column
    - note
    """

    true_like = []
    for c in columns:
        cn = normalize_colname(c)
        if any(k in cn for k in ["metered", "true", "release", "flow", "emission", "ch4", "methane"]):
            if not any(k in cn for k in ["estimate", "estimated", "submitted", "pred", "stage", "team"]):
                true_like.append(c)

    col = infer_numeric_col(df, true_like)
    if col is None:
        return pd.Series([pd.NA] * len(df)), None, "no emission column inferred"

    vals = pd.to_numeric(df[col], errors="coerce")
    cn = normalize_colname(col)

    # Unit inference
    if any(k in cn for k in ["kg", "kgh", "kg_h", "kgph"]):
        return vals / 1000.0, col, "converted from kg/h to t/h based on column name"

    if vals.dropna().max() > 50:
        return vals / 1000.0, col, "converted from likely kg/h to t/h because values are large"

    return vals, col, "assumed t/h"


def infer_lat_lon(df: pd.DataFrame, columns: List[str], path: Path) -> Tuple[pd.Series, pd.Series, str, str]:
    lat_col = find_first_col(columns, ["lat", "latitude"])
    lon_col = find_first_col(columns, ["lon", "lng", "longitude"])

    if lat_col and lon_col:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
        return lat, lon, lat_col, lon_col

    site = infer_site_from_path(path)
    lat = pd.Series([site["lat"]] * len(df))
    lon = pd.Series([site["lon"]] * len(df))
    return lat, lon, "known_site_constant", "known_site_constant"


# ============================================================
# Build file index
# ============================================================

def build_tabular_file_index(tabular_files: List[Path]) -> pd.DataFrame:
    rows = []

    print("\n==============================")
    print("2. Scanning tabular files")
    print("==============================")

    for path in tqdm(tabular_files):
        sheets = read_tabular_file(path, nrows=25)

        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue

            columns = list(df.columns)
            score, matched = score_columns(columns)

            rows.append({
                "path": str(path),
                "relative_path": str(path.relative_to(ROOT)),
                "sheet": sheet_name,
                "paper_guess": infer_paper_from_path(path),
                "n_preview_rows": len(df),
                "n_columns": len(columns),
                "score": score,
                "matched_groups": json.dumps(matched, ensure_ascii=False),
                "columns": json.dumps([str(c) for c in columns], ensure_ascii=False),
            })

    index_df = pd.DataFrame(rows).sort_values(
        ["score", "relative_path"], ascending=[False, True]
    )

    out_path = OUT_DIR / "01_tabular_file_index.csv"
    index_df.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")

    return index_df


# ============================================================
# Extract candidate rows
# ============================================================

def build_candidate_rows(index_df: pd.DataFrame, min_score: int = 3) -> pd.DataFrame:
    print("\n==============================")
    print("3. Extracting candidate event rows")
    print("==============================")

    candidate_rows = []

    candidates = index_df[index_df["score"] >= min_score].copy()

    for _, row in tqdm(candidates.iterrows(), total=len(candidates)):
        path = Path(row["path"])
        sheet = row["sheet"]

        sheets = read_tabular_file(path, nrows=None)
        if sheet not in sheets:
            continue

        df = sheets[sheet]
        if df is None or df.empty:
            continue

        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        columns = list(df.columns)

        score, matched = score_columns(columns)
        keep_cols = set()

        for group_cols in matched.values():
            keep_cols.update(group_cols)

        keep_cols = [c for c in columns if c in keep_cols]

        if not keep_cols:
            continue

        temp = df[keep_cols].copy()
        temp.insert(0, "source_file", str(path.relative_to(ROOT)))
        temp.insert(1, "source_sheet", sheet)
        temp.insert(2, "paper_guess", infer_paper_from_path(path))
        temp.insert(3, "score", score)

        candidate_rows.append(temp)

    if candidate_rows:
        out_df = pd.concat(candidate_rows, ignore_index=True, sort=False)
    else:
        out_df = pd.DataFrame()

    out_path = OUT_DIR / "02_candidate_event_rows.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")

    return out_df


# ============================================================
# Build event table draft
# ============================================================

def build_event_table_draft(index_df: pd.DataFrame, min_score: int = 3) -> pd.DataFrame:
    print("\n==============================")
    print("4. Building draft event table")
    print("==============================")

    rows = []
    candidates = index_df[index_df["score"] >= min_score].copy()

    for _, row in tqdm(candidates.iterrows(), total=len(candidates)):
        path = Path(row["path"])
        sheet = row["sheet"]

        sheets = read_tabular_file(path, nrows=None)
        if sheet not in sheets:
            continue

        df = sheets[sheet]
        if df is None or df.empty:
            continue

        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        columns = list(df.columns)

        dt_series, dt_col = parse_datetime_series(df, columns)
        if dt_series is None:
            continue

        lat_series, lon_series, lat_col, lon_col = infer_lat_lon(df, columns, path)

        satellite_col = find_first_col(
            columns,
            ["satellite", "instrument", "platform", "sensor", "constellation", "sat"],
        )

        team_col = find_first_col(
            columns,
            ["team", "operator", "analyst", "group", "organization", "org"],
        )

        wind_speed_col = find_first_col(
            columns,
            ["wind_speed", "windspeed", "ws", "wind"],
            exclude_keywords=["direction", "dir", "wd"],
        )

        wind_dir_col = find_first_col(
            columns,
            ["wind_direction", "winddir", "wd", "direction"],
        )

        emission_tph, emission_col, emission_note = infer_emission_tph(df, columns)

        for i in range(len(df)):
            dt = dt_series.iloc[i]
            if pd.isna(dt):
                continue

            lat = lat_series.iloc[i] if i < len(lat_series) else pd.NA
            lon = lon_series.iloc[i] if i < len(lon_series) else pd.NA

            if pd.isna(lat) or pd.isna(lon):
                continue

            satellite = df[satellite_col].iloc[i] if satellite_col else ""
            team = df[team_col].iloc[i] if team_col else ""

            e_tph = emission_tph.iloc[i] if i < len(emission_tph) else pd.NA
            true_release = ""
            if pd.notna(e_tph):
                true_release = int(float(e_tph) > 0)

            wind_speed = ""
            if wind_speed_col:
                wind_speed = df[wind_speed_col].iloc[i]

            wind_dir = ""
            if wind_dir_col:
                wind_dir = df[wind_dir_col].iloc[i]

            site = infer_site_from_path(path)

            rows.append({
                "paper": site["paper"],
                "site_name": site["site_name"],
                "datetime_utc": pd.Timestamp(dt).isoformat(),
                "lat": float(lat),
                "lon": float(lon),
                "satellite": str(satellite),
                "team": str(team),
                "true_release": true_release,
                "emission_tph": e_tph,
                "wind_speed": wind_speed,
                "wind_direction": wind_dir,
                "datetime_source_col": dt_col,
                "lat_source_col": lat_col,
                "lon_source_col": lon_col,
                "emission_source_col": emission_col,
                "emission_note": emission_note,
                "source_file": str(path.relative_to(ROOT)),
                "source_sheet": sheet,
            })

    draft = pd.DataFrame(rows)

    if not draft.empty:
        # Clean
        draft["datetime_utc"] = pd.to_datetime(draft["datetime_utc"], errors="coerce", utc=True)
        draft = draft.dropna(subset=["datetime_utc", "lat", "lon"])

        # Remove exact duplicates
        subset_cols = [
            "paper", "datetime_utc", "lat", "lon",
            "satellite", "team", "emission_tph", "source_file"
        ]
        draft = draft.drop_duplicates(subset=subset_cols)

        # Sort
        draft = draft.sort_values(["paper", "datetime_utc", "satellite", "team"])

        # Convert datetime to ISO string
        draft["datetime_utc"] = draft["datetime_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    out_path = OUT_DIR / "03_event_table_draft.csv"
    draft.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")

    return draft


# ============================================================
# Generate GEE JavaScript
# ============================================================

def js_string(s: object) -> str:
    if pd.isna(s):
        return "''"
    return json.dumps(str(s))


def generate_gee_js(draft: pd.DataFrame, max_events: int = 500) -> Path:
    print("\n==============================")
    print("5. Generating GEE JavaScript")
    print("==============================")

    out_path = OUT_DIR / "04_gee_check_events.js"

    if draft.empty:
        out_path.write_text(
            "// No draft events were generated. Please inspect 01_tabular_file_index.csv.\n",
            encoding="utf-8",
        )
        print(f"[SAVED] {out_path}")
        return out_path

    gee_df = draft.copy()
    gee_df = gee_df.dropna(subset=["datetime_utc", "lat", "lon"])
    gee_df = gee_df.head(max_events)

    features_js = []

    for idx, r in gee_df.iterrows():
        lon = float(r["lon"])
        lat = float(r["lat"])

        props = {
            "paper": r.get("paper", ""),
            "site_name": r.get("site_name", ""),
            "datetime_utc": r.get("datetime_utc", ""),
            "satellite_from_paper": r.get("satellite", ""),
            "team": r.get("team", ""),
            "true_release": r.get("true_release", ""),
            "emission_tph": "" if pd.isna(r.get("emission_tph", pd.NA)) else str(r.get("emission_tph")),
            "source_file": r.get("source_file", ""),
        }

        props_js = json.dumps(props, ensure_ascii=False)

        feature = (
            f"  ee.Feature(ee.Geometry.Point([{lon}, {lat}]), {props_js})"
        )
        features_js.append(feature)

    features_joined = ",\\n".join(features_js)

    js_code = f"""
// Auto-generated by build_methane_event_table.py
// Paste this file into Google Earth Engine Code Editor.
// It checks whether Sentinel-2 / Landsat 8 / Landsat 9 / EMIT L2A images exist
// near each controlled release event time and location.

var events = ee.FeatureCollection([
{features_joined}
]);

Map.centerObject(events, 8);
Map.addLayer(events, {{color: 'red'}}, 'controlled release events');

var S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');
var L9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2');
var EMIT = ee.ImageCollection('NASA/EMIT/L2A/RFL');

// Change this window if you want stricter matching.
// For exact overpass matching, try 3 hours or 6 hours.
// For initial availability checking, 1 day is safer.
var WINDOW_HOURS = 24;

function firstTimeOrNone(collection) {{
  return ee.Algorithms.If(
    collection.size().gt(0),
    ee.Date(collection.first().get('system:time_start')).format('YYYY-MM-dd HH:mm:ss'),
    'none'
  );
}}

function checkAvailability(feature) {{
  var point = feature.geometry();
  var t = ee.Date(feature.get('datetime_utc'));
  var start = t.advance(-WINDOW_HOURS, 'hour');
  var end = t.advance(WINDOW_HOURS, 'hour');

  var s2 = S2.filterBounds(point).filterDate(start, end)
    .sort('CLOUDY_PIXEL_PERCENTAGE');

  var l8 = L8.filterBounds(point).filterDate(start, end)
    .sort('CLOUD_COVER');

  var l9 = L9.filterBounds(point).filterDate(start, end)
    .sort('CLOUD_COVER');

  var emit = EMIT.filterBounds(point).filterDate(start, end);

  return feature.set({{
    search_start: start.format('YYYY-MM-dd HH:mm:ss'),
    search_end: end.format('YYYY-MM-dd HH:mm:ss'),

    s2_count: s2.size(),
    l8_count: l8.size(),
    l9_count: l9.size(),
    emit_l2a_count: emit.size(),

    s2_first_time: firstTimeOrNone(s2),
    l8_first_time: firstTimeOrNone(l8),
    l9_first_time: firstTimeOrNone(l9),
    emit_l2a_first_time: firstTimeOrNone(emit),

    s2_first_cloud: ee.Algorithms.If(
      s2.size().gt(0),
      ee.Image(s2.first()).get('CLOUDY_PIXEL_PERCENTAGE'),
      'none'
    ),

    l8_first_cloud: ee.Algorithms.If(
      l8.size().gt(0),
      ee.Image(l8.first()).get('CLOUD_COVER'),
      'none'
    ),

    l9_first_cloud: ee.Algorithms.If(
      l9.size().gt(0),
      ee.Image(l9.first()).get('CLOUD_COVER'),
      'none'
    )
  }});
}}

var checked = events.map(checkAvailability);

print('Checked availability table', checked);
print('Number of events', checked.size());

// Export availability table to Google Drive as CSV.
Export.table.toDrive({{
  collection: checked,
  description: 'controlled_release_satellite_availability',
  fileFormat: 'CSV'
}});

// Optional visualization:
// Show first available Sentinel-2 image around the first event.
var firstEvent = ee.Feature(events.first());
var firstPoint = firstEvent.geometry();
var firstTime = ee.Date(firstEvent.get('datetime_utc'));
var s2FirstCollection = S2
  .filterBounds(firstPoint)
  .filterDate(firstTime.advance(-WINDOW_HOURS, 'hour'), firstTime.advance(WINDOW_HOURS, 'hour'))
  .sort('CLOUDY_PIXEL_PERCENTAGE');

var s2First = ee.Image(s2FirstCollection.first());

Map.addLayer(
  s2First,
  {{bands: ['B12', 'B11', 'B8'], min: 0, max: 4000}},
  'First available S2 SWIR composite',
  false
);
"""

    out_path.write_text(js_code, encoding="utf-8")
    print(f"[SAVED] {out_path}")

    return out_path


# ============================================================
# Manual template
# ============================================================

def write_manual_template() -> None:
    template_path = OUT_DIR / "manual_events_template.csv"
    rows = [
        {
            "paper": "2023_Scientific_Reports",
            "event_id": "example_2023",
            "datetime_utc": "2021-10-27T18:35:00Z",
            "lat": 33.630645,
            "lon": -114.489150,
            "satellite": "Sentinel-2",
            "true_release": 1,
            "emission_tph": 3.5,
            "note": "example from paper figure, replace with repository values",
        },
        {
            "paper": "2024_AMT",
            "event_id": "example_2024",
            "datetime_utc": "2022-10-10T18:00:00Z",
            "lat": 32.8218205,
            "lon": -111.7857730,
            "satellite": "",
            "true_release": "",
            "emission_tph": "",
            "note": "replace with repository values",
        },
    ]
    pd.DataFrame(rows).to_csv(template_path, index=False)
    print(f"[SAVED] {template_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    download_all_sources()

    tabular_files = find_tabular_files(RAW_DIR)

    print("\n==============================")
    print("Found tabular files")
    print("==============================")
    print(f"Total tabular files: {len(tabular_files)}")

    index_df = build_tabular_file_index(tabular_files)
    build_candidate_rows(index_df, min_score=3)
    draft = build_event_table_draft(index_df, min_score=3)
    generate_gee_js(draft, max_events=500)
    write_manual_template()

    print("\nDONE.")
    print("\nPlease check these files:")
    print(f"1. {OUT_DIR / '01_tabular_file_index.csv'}")
    print(f"2. {OUT_DIR / '02_candidate_event_rows.csv'}")
    print(f"3. {OUT_DIR / '03_event_table_draft.csv'}")
    print(f"4. {OUT_DIR / '04_gee_check_events.js'}")
    print(f"5. {OUT_DIR / 'manual_events_template.csv'}")
    print("\nNext step:")
    print("Open outputs/04_gee_check_events.js and paste it into Google Earth Engine Code Editor.")


if __name__ == "__main__":
    main()