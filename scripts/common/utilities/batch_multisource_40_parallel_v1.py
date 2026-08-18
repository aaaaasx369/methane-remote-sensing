from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import json
import math
import time
import traceback

import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

HOME = Path.home()

PROJECT = (
    HOME
    / "methane_release_project"
)

LAB_ROOT = Path(
    "/Volumes/engg-leung/dora lin"
)

INPUT = (
    PROJECT
    / "candidate_negative_validation"
    / "pilot_10_positive_40_candidates_s2qa.csv"
)

OUTDIR = (
    PROJECT
    / "candidate_negative_validation"
    / "parallel_multisource_40"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# MethaneAIR matching
# ------------------------------------------------------------

METHANEAIR_RADIUS_KM = 10.0
METHANEAIR_TIME_DAYS = 14


# ------------------------------------------------------------
# File discovery
# ------------------------------------------------------------

LOCAL_MAXDEPTH = 7
LAB_MAXDEPTH = 5

FIND_TIMEOUT_SEC = 240


# ------------------------------------------------------------
# EMIT
# ------------------------------------------------------------

EMIT_COLLECTION = (
    "C3242680113-LPCLOUD"
)

CMR_URL = (
    "https://cmr.earthdata.nasa.gov/"
    "search/granules.json"
)


# ------------------------------------------------------------
# TROPOMI
# ------------------------------------------------------------

EE_PROJECT = "methane-release-gee"

TROPOMI_COLLECTION = (
    "COPERNICUS/S5P/OFFL/L3_CH4"
)

TROPOMI_CH4_BAND = (
    "CH4_column_volume_mixing_ratio_dry_air_bias_corrected"
)

TROPOMI_UNC_BAND = (
    "CH4_column_volume_mixing_ratio_dry_air_uncertainty"
)

# Regional context only
TROPOMI_SOURCE_RADIUS_M = 20_000

TROPOMI_BG_INNER_M = 30_000
TROPOMI_BG_OUTER_M = 70_000

# Approximate native CH4 footprint scale.
TROPOMI_SCALE_M = 7000

TROPOMI_WORKERS = 4
EMIT_WORKERS = 8

MAX_RETRIES = 4


# ============================================================
# LOAD 40 CANDIDATES
# ============================================================

print("=" * 110)
print("40-CANDIDATE PARALLEL MULTI-SOURCE VALIDATION")
print("=" * 110)

if not INPUT.exists():
    raise FileNotFoundError(
        f"Missing input:\n{INPUT}"
    )

candidates = pd.read_csv(
    INPUT,
    low_memory=False
)

if len(candidates) != 40:
    raise RuntimeError(
        f"Expected 40 rows, found {len(candidates)}"
    )

required = [
    "Pilot Candidate ID",
    "Pilot Parent Number",
    "Source Positive Record ID",
    "Site",
    "Latitude",
    "Longitude",
    "Date",
    "Resolved Offset Days",
    "Cloud/Snow QA Pass",
]

missing = [
    c
    for c in required
    if c not in candidates.columns
]

if missing:
    raise RuntimeError(
        "Missing columns:\n"
        +
        "\n".join(missing)
    )

candidates["_candidate_datetime"] = (
    pd.to_datetime(
        candidates["Date"],
        errors="coerce",
        utc=True,
    )
)

if candidates[
    "_candidate_datetime"
].isna().any():

    raise RuntimeError(
        "Some candidate dates are invalid."
    )


print("\nCandidates :", len(candidates))
print(
    "S2 QA pass :",
    int(
        candidates[
            "Cloud/Snow QA Pass"
        ].eq("pass").sum()
    )
)

print(
    "Exact S2   :",
    int(
        candidates[
            "Model Image Available"
        ].eq("yes").sum()
    )
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def mounted_lab():
    try:
        p = subprocess.run(
            ["mount"],
            capture_output=True,
            text=True,
            timeout=20,
        )

        return (
            "/Volumes/engg-leung"
            in p.stdout
        )

    except Exception:
        return False


def decode_timeout_output(x):
    if x is None:
        return ""

    if isinstance(x, bytes):
        return x.decode(
            "utf-8",
            errors="replace",
        )

    return str(x)


def haversine_vector(
    lat0,
    lon0,
    lat,
    lon,
):
    R = 6371.0088

    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    lat = np.radians(
        np.asarray(lat, dtype=float)
    )

    lon = np.radians(
        np.asarray(lon, dtype=float)
    )

    dlat = lat - lat0
    dlon = lon - lon0

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat0)
        *
        np.cos(lat)
        *
        np.sin(dlon / 2) ** 2
    )

    return (
        2
        *
        R
        *
        np.arcsin(np.sqrt(a))
    )


def normalize_col(s):
    return (
        str(s)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_col(
    columns,
    exact_names,
):
    cmap = {
        normalize_col(c): c
        for c in columns
    }

    for name in exact_names:
        n = normalize_col(name)

        if n in cmap:
            return cmap[n]

    for name in exact_names:
        n = normalize_col(name)

        for key, original in cmap.items():
            if n in key:
                return original

    return None


# ============================================================
# FILE DISCOVERY
#
# Read-only.
# No files are copied/deleted.
# ============================================================

FILE_PATTERNS = [
    "*methaneair*",
    "*methane_air*",
    "*pcannon*",
    "*emit*",
    "*methanesat*",
    "*tropomi*",
    "*s5p*",
    "*ch4enh*",
    "*ch4plm*",
    "*ch4sens*",
    "*ch4uncert*",
]


def discover_sensor_files(
    root,
    maxdepth,
    label,
):

    root = Path(root)

    if not root.exists():
        return {
            "label": label,
            "status": "root_missing",
            "files": [],
        }

    if (
        label == "LAB"
        and not mounted_lab()
    ):
        return {
            "label": label,
            "status": "smb_not_mounted",
            "files": [],
        }

    cmd = [
        "find",
        str(root),
        "-maxdepth",
        str(maxdepth),
        "-type",
        "f",
        "(",
    ]

    for i, pat in enumerate(
        FILE_PATTERNS
    ):

        if i > 0:
            cmd.append("-o")

        cmd.extend(
            [
                "-iname",
                pat,
            ]
        )

    cmd.extend(
        [
            ")",
            "-print",
        ]
    )

    status = "complete"
    output = ""

    try:

        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FIND_TIMEOUT_SEC,
        )

        output = (
            p.stdout
            or ""
        )

        if p.returncode != 0:
            status = (
                "find_returncode_"
                +
                str(p.returncode)
            )

    except subprocess.TimeoutExpired as e:

        status = "timeout_partial"

        output = (
            decode_timeout_output(
                e.stdout
            )
        )

    except Exception as e:

        status = (
            "error:"
            +
            repr(e)
        )

    files = []

    seen = set()

    for line in output.splitlines():

        line = line.strip()

        if not line:
            continue

        if line in seen:
            continue

        seen.add(line)
        files.append(
            Path(line)
        )

    return {
        "label": label,
        "status": status,
        "files": files,
    }


# ============================================================
# METHANEAIR CATALOG MATCHING
#
# Batch matches ALL 40 candidates against each discovered
# MethaneAIR/pcannon CSV.
# ============================================================

def source_datetime_series(df):

    full_col = find_col(
        df.columns,
        [
            "datetime",
            "timestamp",
            "acquisition_time_utc",
            "observation_time_utc",
            "time_coverage_start",
            "scene_datetime",
        ]
    )

    if full_col is not None:

        return pd.to_datetime(
            df[full_col],
            errors="coerce",
            utc=True,
        )

    date_col = find_col(
        df.columns,
        [
            "date",
            "observation_date",
            "acquisition_date",
        ]
    )

    if date_col is None:
        return None

    utc_col = find_col(
        df.columns,
        [
            "utc_time",
            "time_utc",
        ]
    )

    if utc_col is not None:

        text = (
            df[date_col]
            .astype("string")
            .fillna("")
            .str.strip()
            +
            " "
            +
            df[utc_col]
            .astype("string")
            .fillna("")
            .str.strip()
        )

    else:

        text = (
            df[date_col]
            .astype("string")
        )

    return pd.to_datetime(
        text,
        errors="coerce",
        utc=True,
    )


def match_methaneair_files(
    discovery,
):

    label = discovery[
        "label"
    ]

    files = discovery[
        "files"
    ]

    methaneair_csvs = []

    for p in files:

        name = p.name.lower()

        if p.suffix.lower() != ".csv":
            continue

        if (
            "methaneair" in name
            or "methane_air" in name
            or "pcannon" in name
        ):

            methaneair_csvs.append(
                p
            )

    matches = []

    file_errors = []

    for file_path in methaneair_csvs:

        try:

            df = pd.read_csv(
                file_path,
                low_memory=False,
            )

        except Exception as e:

            file_errors.append({
                "origin": label,
                "file": str(file_path),
                "error": repr(e),
            })

            continue

        lat_col = find_col(
            df.columns,
            [
                "latitude",
                "lat",
                "plume_latitude",
            ]
        )

        lon_col = find_col(
            df.columns,
            [
                "longitude",
                "lon",
                "lng",
                "plume_longitude",
            ]
        )

        if (
            lat_col is None
            or lon_col is None
        ):
            continue

        src_time = (
            source_datetime_series(
                df
            )
        )

        if src_time is None:
            continue

        src_lat = pd.to_numeric(
            df[lat_col],
            errors="coerce",
        )

        src_lon = pd.to_numeric(
            df[lon_col],
            errors="coerce",
        )

        valid = (
            src_lat.notna()
            &
            src_lon.notna()
            &
            src_time.notna()
        )

        if not valid.any():
            continue

        temp = pd.DataFrame({
            "_source_row":
                df.index,

            "_src_lat":
                src_lat,

            "_src_lon":
                src_lon,

            "_src_time":
                src_time,
        })

        temp = temp.loc[
            valid
        ].copy()

        id_col = find_col(
            df.columns,
            [
                "record_id",
                "plume_id",
                "source_id",
                "observation_id",
                "id",
            ]
        )

        for _, cand in (
            candidates.iterrows()
        ):

            cand_time = (
                cand[
                    "_candidate_datetime"
                ]
            )

            dt_days = (
                temp["_src_time"]
                -
                cand_time
            ).dt.total_seconds() / 86400

            temporal_mask = (
                dt_days.abs()
                <=
                METHANEAIR_TIME_DAYS
            )

            if not temporal_mask.any():
                continue

            sub = (
                temp.loc[
                    temporal_mask
                ]
                .copy()
            )

            sub[
                "_time_diff_days"
            ] = (
                dt_days.loc[
                    temporal_mask
                ]
            )

            distances = (
                haversine_vector(
                    float(
                        cand["Latitude"]
                    ),
                    float(
                        cand["Longitude"]
                    ),
                    sub["_src_lat"],
                    sub["_src_lon"],
                )
            )

            sub[
                "_distance_km"
            ] = distances

            sub = sub[
                sub["_distance_km"]
                <=
                METHANEAIR_RADIUS_KM
            ]

            if len(sub) == 0:
                continue

            for _, hit in (
                sub.iterrows()
            ):

                source_idx = int(
                    hit["_source_row"]
                )

                source_id = None

                if id_col is not None:

                    try:
                        source_id = (
                            df.loc[
                                source_idx,
                                id_col
                            ]
                        )
                    except Exception:
                        pass

                matches.append({
                    "Pilot Candidate ID":
                        cand[
                            "Pilot Candidate ID"
                        ],

                    "Site":
                        cand["Site"],

                    "Candidate Date":
                        cand["Date"],

                    "Origin":
                        label,

                    "Source File":
                        str(file_path),

                    "Source Row":
                        source_idx,

                    "Source Record ID":
                        source_id,

                    "Source Datetime UTC":
                        hit["_src_time"],

                    "Source Latitude":
                        hit["_src_lat"],

                    "Source Longitude":
                        hit["_src_lon"],

                    "Distance km":
                        hit["_distance_km"],

                    "Time Difference days":
                        hit["_time_diff_days"],
                })

    return (
        pd.DataFrame(matches),
        pd.DataFrame(file_errors),
        methaneair_csvs,
    )


# ============================================================
# LOCAL TASK
# ============================================================

def run_local_task():

    print(
        "\n[LOCAL] starting file discovery..."
    )

    discovery = (
        discover_sensor_files(
            PROJECT,
            LOCAL_MAXDEPTH,
            "LOCAL",
        )
    )

    print(
        "[LOCAL] discovery:",
        discovery["status"],
        "| files:",
        len(discovery["files"]),
    )

    matches, errors, ma_files = (
        match_methaneair_files(
            discovery
        )
    )

    print(
        "[LOCAL] MethaneAIR CSVs:",
        len(ma_files),
        "| matches:",
        len(matches),
    )

    return {
        "discovery": discovery,
        "matches": matches,
        "errors": errors,
        "methaneair_files": ma_files,
    }


# ============================================================
# LAB TASK
# ============================================================

def run_lab_task():

    print(
        "\n[LAB] starting SMB discovery..."
    )

    discovery = (
        discover_sensor_files(
            LAB_ROOT,
            LAB_MAXDEPTH,
            "LAB",
        )
    )

    print(
        "[LAB] discovery:",
        discovery["status"],
        "| files:",
        len(discovery["files"]),
    )

    matches, errors, ma_files = (
        match_methaneair_files(
            discovery
        )
    )

    print(
        "[LAB] MethaneAIR CSVs:",
        len(ma_files),
        "| matches:",
        len(matches),
    )

    return {
        "discovery": discovery,
        "matches": matches,
        "errors": errors,
        "methaneair_files": ma_files,
    }


# ============================================================
# EMIT CMR
# ============================================================

def cmr_query(
    lat,
    lon,
    start,
    end,
):

    params = {
        "collection_concept_id":
            EMIT_COLLECTION,

        # CMR expects lon,lat
        "point":
            f"{lon},{lat}",

        "temporal":
            f"{start},{end}",

        "page_size":
            100,
    }

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            r = requests.get(
                CMR_URL,
                params=params,
                timeout=60,
                headers={
                    "User-Agent":
                    "MethaneFuse-negative-validation/1.0"
                },
            )

            r.raise_for_status()

            return (
                r.json()
                .get("feed", {})
                .get("entry", [])
            )

        except Exception as e:

            last_error = e

            time.sleep(
                attempt
            )

    raise RuntimeError(
        repr(last_error)
    )


def emit_one(row):

    cid = (
        row["Pilot Candidate ID"]
    )

    lat = float(
        row["Latitude"]
    )

    lon = float(
        row["Longitude"]
    )

    d = pd.Timestamp(
        row["_candidate_datetime"]
    )

    exact_start = (
        d.strftime(
            "%Y-%m-%dT00:00:00Z"
        )
    )

    exact_end = (
        d.strftime(
            "%Y-%m-%dT23:59:59Z"
        )
    )

    ctx_start = (
        d
        -
        pd.Timedelta(days=3)
    ).strftime(
        "%Y-%m-%dT00:00:00Z"
    )

    ctx_end = (
        d
        +
        pd.Timedelta(days=3)
    ).strftime(
        "%Y-%m-%dT23:59:59Z"
    )

    try:

        exact = cmr_query(
            lat,
            lon,
            exact_start,
            exact_end,
        )

        context = cmr_query(
            lat,
            lon,
            ctx_start,
            ctx_end,
        )

        return {
            "Pilot Candidate ID":
                cid,

            "EMIT Exact Date Count":
                len(exact),

            "EMIT +/-3d Count":
                len(context),

            "EMIT Exact Titles":
                " | ".join(
                    str(x.get("title"))
                    for x in exact
                ),

            "EMIT +/-3d Titles":
                " | ".join(
                    str(x.get("title"))
                    for x in context
                ),

            "EMIT Coverage Status":
                (
                    "exact_date"
                    if len(exact) > 0
                    else
                    (
                        "context_only"
                        if len(context) > 0
                        else
                        "none"
                    )
                ),

            "EMIT Query Error":
                "",
        }

    except Exception as e:

        return {
            "Pilot Candidate ID":
                cid,

            "EMIT Exact Date Count":
                pd.NA,

            "EMIT +/-3d Count":
                pd.NA,

            "EMIT Exact Titles":
                "",

            "EMIT +/-3d Titles":
                "",

            "EMIT Coverage Status":
                "query_error",

            "EMIT Query Error":
                repr(e),
        }


def run_emit_batch():

    print(
        "\n[EMIT] starting 40-candidate CMR batch..."
    )

    rows = []

    with ThreadPoolExecutor(
        max_workers=EMIT_WORKERS
    ) as pool:

        futures = [
            pool.submit(
                emit_one,
                row,
            )
            for _, row
            in candidates.iterrows()
        ]

        done = 0

        for f in as_completed(
            futures
        ):

            rows.append(
                f.result()
            )

            done += 1

            if (
                done % 10 == 0
                or done == 40
            ):
                print(
                    f"[EMIT] {done}/40 complete"
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TROPOMI
# ============================================================

def tropomi_collection(
    ee,
    point,
    start,
    end,
):

    return (
        ee.ImageCollection(
            TROPOMI_COLLECTION
        )
        .filterDate(
            start,
            end,
        )
        .filterBounds(
            point
        )
    )


def tropomi_stats(
    ee,
    collection,
    point,
):

    count = int(
        collection.size().getInfo()
    )

    if count == 0:

        return {
            "count": 0,
            "regional_median": None,
            "background_median": None,
            "delta_ppb": None,
            "uncertainty_median": None,
            "uncertainty_x2": None,
            "valid_pixels": 0,
            "times": [],
        }

    image = (
        collection
        .select(
            [
                TROPOMI_CH4_BAND,
                TROPOMI_UNC_BAND,
            ]
        )
        .median()
    )

    source_region = (
        point.buffer(
            TROPOMI_SOURCE_RADIUS_M
        )
    )

    background_region = (
        point
        .buffer(
            TROPOMI_BG_OUTER_M
        )
        .difference(
            point.buffer(
                TROPOMI_BG_INNER_M
            )
        )
    )

    source_stats = (
        image.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=source_region,
            scale=TROPOMI_SCALE_M,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    bg_stats = (
        image
        .select(
            TROPOMI_CH4_BAND
        )
        .reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=background_region,
            scale=TROPOMI_SCALE_M,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    pixel_count = (
        image
        .select(
            TROPOMI_CH4_BAND
        )
        .reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=source_region,
            scale=TROPOMI_SCALE_M,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    times = (
        collection
        .aggregate_array(
            "system:time_start"
        )
        .getInfo()
    )

    regional = (
        source_stats.get(
            TROPOMI_CH4_BAND
        )
    )

    uncertainty = (
        source_stats.get(
            TROPOMI_UNC_BAND
        )
    )

    background = (
        bg_stats.get(
            TROPOMI_CH4_BAND
        )
    )

    valid_pixels = (
        pixel_count.get(
            TROPOMI_CH4_BAND
        )
        or 0
    )

    delta = None

    if (
        regional is not None
        and background is not None
    ):

        delta = (
            float(regional)
            -
            float(background)
        )

    return {
        "count":
            count,

        "regional_median":
            regional,

        "background_median":
            background,

        "delta_ppb":
            delta,

        "uncertainty_median":
            uncertainty,

        "uncertainty_x2":
            (
                None
                if uncertainty is None
                else
                2.0
                *
                float(uncertainty)
            ),

        "valid_pixels":
            valid_pixels,

        "times":
            times,
    }


def tropomi_one(
    ee,
    row,
):

    cid = (
        row["Pilot Candidate ID"]
    )

    lat = float(
        row["Latitude"]
    )

    lon = float(
        row["Longitude"]
    )

    date = pd.Timestamp(
        row[
            "_candidate_datetime"
        ]
    )

    d0 = date.strftime(
        "%Y-%m-%d"
    )

    d1 = (
        date
        +
        pd.Timedelta(days=1)
    ).strftime(
        "%Y-%m-%d"
    )

    c0 = (
        date
        -
        pd.Timedelta(days=1)
    ).strftime(
        "%Y-%m-%d"
    )

    c1 = (
        date
        +
        pd.Timedelta(days=2)
    ).strftime(
        "%Y-%m-%d"
    )

    point = ee.Geometry.Point(
        [
            lon,
            lat,
        ]
    )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            exact_collection = (
                tropomi_collection(
                    ee,
                    point,
                    d0,
                    d1,
                )
            )

            context_collection = (
                tropomi_collection(
                    ee,
                    point,
                    c0,
                    c1,
                )
            )

            exact = tropomi_stats(
                ee,
                exact_collection,
                point,
            )

            context = tropomi_stats(
                ee,
                context_collection,
                point,
            )

            exact_times = []

            for x in exact["times"]:

                try:
                    exact_times.append(
                        str(
                            pd.to_datetime(
                                x,
                                unit="ms",
                                utc=True,
                            )
                        )
                    )
                except Exception:
                    pass

            context_times = []

            for x in context["times"]:

                try:
                    context_times.append(
                        str(
                            pd.to_datetime(
                                x,
                                unit="ms",
                                utc=True,
                            )
                        )
                    )
                except Exception:
                    pass

            return {
                "Pilot Candidate ID":
                    cid,

                "TROPOMI Exact Date Image Count":
                    exact["count"],

                "TROPOMI Exact Regional Median ppb":
                    exact["regional_median"],

                "TROPOMI Exact Background Median ppb":
                    exact["background_median"],

                "TROPOMI Exact Delta ppb":
                    exact["delta_ppb"],

                "TROPOMI Exact Uncertainty Median ppb":
                    exact["uncertainty_median"],

                "TROPOMI Exact Uncertainty x2 ppb":
                    exact["uncertainty_x2"],

                "TROPOMI Exact Valid Pixels":
                    exact["valid_pixels"],

                "TROPOMI Exact Datetimes UTC":
                    " | ".join(
                        exact_times
                    ),

                "TROPOMI +/-1d Image Count":
                    context["count"],

                "TROPOMI +/-1d Regional Median ppb":
                    context["regional_median"],

                "TROPOMI +/-1d Background Median ppb":
                    context["background_median"],

                "TROPOMI +/-1d Delta ppb":
                    context["delta_ppb"],

                "TROPOMI +/-1d Valid Pixels":
                    context["valid_pixels"],

                "TROPOMI +/-1d Datetimes UTC":
                    " | ".join(
                        context_times
                    ),

                "TROPOMI Evidence Role":
                    (
                        "regional_support_only"
                        if (
                            exact["valid_pixels"] > 0
                            or
                            context["valid_pixels"] > 0
                        )
                        else
                        "no_valid_data"
                    ),

                "TROPOMI Query Error":
                    "",
            }

        except Exception as e:

            last_error = e

            time.sleep(
                attempt * 2
            )

    return {
        "Pilot Candidate ID":
            cid,

        "TROPOMI Exact Date Image Count":
            pd.NA,

        "TROPOMI Exact Regional Median ppb":
            pd.NA,

        "TROPOMI Exact Background Median ppb":
            pd.NA,

        "TROPOMI Exact Delta ppb":
            pd.NA,

        "TROPOMI Exact Uncertainty Median ppb":
            pd.NA,

        "TROPOMI Exact Uncertainty x2 ppb":
            pd.NA,

        "TROPOMI Exact Valid Pixels":
            pd.NA,

        "TROPOMI Exact Datetimes UTC":
            "",

        "TROPOMI +/-1d Image Count":
            pd.NA,

        "TROPOMI +/-1d Regional Median ppb":
            pd.NA,

        "TROPOMI +/-1d Background Median ppb":
            pd.NA,

        "TROPOMI +/-1d Delta ppb":
            pd.NA,

        "TROPOMI +/-1d Valid Pixels":
            pd.NA,

        "TROPOMI +/-1d Datetimes UTC":
            "",

        "TROPOMI Evidence Role":
            "query_error",

        "TROPOMI Query Error":
            repr(last_error),
    }


def run_tropomi_batch():

    print(
        "\n[TROPOMI] initializing Earth Engine..."
    )

    try:

        import ee

        ee.Initialize(
            project=EE_PROJECT
        )

    except Exception as e:

        print(
            "[TROPOMI] Earth Engine init failed:",
            repr(e)
        )

        rows = []

        for _, row in (
            candidates.iterrows()
        ):

            rows.append({
                "Pilot Candidate ID":
                    row[
                        "Pilot Candidate ID"
                    ],

                "TROPOMI Evidence Role":
                    "ee_init_error",

                "TROPOMI Query Error":
                    repr(e),
            })

        return pd.DataFrame(
            rows
        )

    print(
        "[TROPOMI] Earth Engine ready."
    )

    rows = []

    with ThreadPoolExecutor(
        max_workers=TROPOMI_WORKERS
    ) as pool:

        futures = [
            pool.submit(
                tropomi_one,
                ee,
                row,
            )
            for _, row
            in candidates.iterrows()
        ]

        done = 0

        for f in as_completed(
            futures
        ):

            rows.append(
                f.result()
            )

            done += 1

            if (
                done % 10 == 0
                or done == 40
            ):

                print(
                    f"[TROPOMI] {done}/40 complete"
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# LAUNCH ALL FOUR BRANCHES AT ONCE
# ============================================================

print("\n" + "=" * 110)
print("STARTING FOUR PARALLEL BRANCHES")
print("=" * 110)

print(
    "\n1. LOCAL existing-data search"
)

print(
    "2. LAB SMB existing-data search"
)

print(
    "3. EMIT CMR batch"
)

print(
    "4. TROPOMI Earth Engine batch"
)


with ThreadPoolExecutor(
    max_workers=4
) as outer:

    futures = {
        outer.submit(
            run_local_task
        ):
            "LOCAL",

        outer.submit(
            run_lab_task
        ):
            "LAB",

        outer.submit(
            run_emit_batch
        ):
            "EMIT",

        outer.submit(
            run_tropomi_batch
        ):
            "TROPOMI",
    }

    results = {}

    for future in as_completed(
        futures
    ):

        name = futures[
            future
        ]

        try:

            results[name] = (
                future.result()
            )

            print(
                f"\n[{name}] BRANCH COMPLETE"
            )

        except Exception as e:

            results[name] = {
                "branch_error":
                    repr(e),

                "traceback":
                    traceback.format_exc(),
            }

            print(
                f"\n[{name}] BRANCH FAILED:",
                repr(e),
            )


# ============================================================
# SAVE EXISTING FILE INVENTORIES
# ============================================================

local_result = results.get(
    "LOCAL",
    {}
)

lab_result = results.get(
    "LAB",
    {}
)


def save_discovery(
    result,
    filename,
):

    discovery = result.get(
        "discovery",
        {}
    )

    files = discovery.get(
        "files",
        []
    )

    with open(
        OUTDIR / filename,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "STATUS: "
            +
            str(
                discovery.get(
                    "status"
                )
            )
            +
            "\n"
        )

        f.write(
            "COUNT: "
            +
            str(len(files))
            +
            "\n\n"
        )

        for p in files:
            f.write(
                str(p)
                +
                "\n"
            )


if isinstance(
    local_result,
    dict
):
    save_discovery(
        local_result,
        "local_existing_sensor_files.txt",
    )


if isinstance(
    lab_result,
    dict
):
    save_discovery(
        lab_result,
        "lab_existing_sensor_files.txt",
    )


# ============================================================
# COMBINE METHANEAIR MATCHES
# ============================================================

ma_frames = []

for key in [
    "LOCAL",
    "LAB",
]:

    result = results.get(
        key,
        {}
    )

    if not isinstance(
        result,
        dict
    ):
        continue

    frame = result.get(
        "matches"
    )

    if (
        isinstance(
            frame,
            pd.DataFrame,
        )
        and len(frame)
    ):

        ma_frames.append(
            frame
        )


if ma_frames:

    methaneair_matches = pd.concat(
        ma_frames,
        ignore_index=True,
    )

    methaneair_matches = (
        methaneair_matches
        .sort_values(
            [
                "Pilot Candidate ID",
                "Distance km",
                "Time Difference days",
            ],
            na_position="last",
        )
    )

else:

    methaneair_matches = (
        pd.DataFrame()
    )


METHANEAIR_OUT = (
    OUTDIR
    /
    "methaneair_matches_all_40.csv"
)

methaneair_matches.to_csv(
    METHANEAIR_OUT,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# BUILD ONE ROW PER CANDIDATE METHANEAIR SUMMARY
# ============================================================

ma_summary_rows = []

for _, cand in (
    candidates.iterrows()
):

    cid = cand[
        "Pilot Candidate ID"
    ]

    if len(
        methaneair_matches
    ):

        sub = (
            methaneair_matches[
                methaneair_matches[
                    "Pilot Candidate ID"
                ]
                ==
                cid
            ]
        )

    else:

        sub = pd.DataFrame()

    ma_summary_rows.append({
        "Pilot Candidate ID":
            cid,

        "MethaneAIR +/-14d <=10km Match Count":
            len(sub),

        "MethaneAIR Independent Evidence":
            (
                "records_found"
                if len(sub) > 0
                else
                "none_found"
            ),
    })


ma_summary = pd.DataFrame(
    ma_summary_rows
)


# ============================================================
# MERGE EMIT + TROPOMI
# ============================================================

emit_df = results.get(
    "EMIT"
)

trop_df = results.get(
    "TROPOMI"
)


combined = (
    candidates
    .drop(
        columns=[
            "_candidate_datetime"
        ],
        errors="ignore",
    )
    .merge(
        ma_summary,
        on="Pilot Candidate ID",
        how="left",
        validate="one_to_one",
    )
)


if isinstance(
    emit_df,
    pd.DataFrame,
):

    combined = combined.merge(
        emit_df,
        on="Pilot Candidate ID",
        how="left",
        validate="one_to_one",
    )


if isinstance(
    trop_df,
    pd.DataFrame,
):

    combined = combined.merge(
        trop_df,
        on="Pilot Candidate ID",
        how="left",
        validate="one_to_one",
    )


# ============================================================
# CONSERVATIVE MULTI-SOURCE STATUS
#
# TROPOMI DOES NOT UPGRADE A CANDIDATE TO A
# HIGH-RES VALIDATED NEGATIVE.
# ============================================================

def provisional_status(row):

    if (
        row.get(
            "Cloud/Snow QA Pass"
        )
        !=
        "pass"
    ):

        return (
            "S2_GATE_NOT_PASS"
        )

    emit_status = row.get(
        "EMIT Coverage Status"
    )

    methaneair_count = row.get(
        "MethaneAIR +/-14d <=10km Match Count"
    )

    if emit_status == "exact_date":

        return (
            "HIGH_RES_COVERAGE_REQUIRES_EMIT_SIGNAL_QA"
        )

    try:

        if int(
            methaneair_count
        ) > 0:

            return (
                "METHANEAIR_RECORDS_REQUIRE_CLASSIFICATION"
            )

    except Exception:
        pass

    trop_role = row.get(
        "TROPOMI Evidence Role"
    )

    if (
        trop_role
        ==
        "regional_support_only"
    ):

        return (
            "U_UNKNOWN_WITH_TROPOMI_CONTEXT"
        )

    return "U_UNKNOWN"


combined[
    "Parallel Validation Status"
] = combined.apply(
    provisional_status,
    axis=1,
)


# ============================================================
# SAVE
# ============================================================

COMBINED_CSV = (
    OUTDIR
    /
    "pilot_40_parallel_multisource_audit.csv"
)

COMBINED_XLSX = (
    OUTDIR
    /
    "pilot_40_parallel_multisource_audit.xlsx"
)


combined.to_csv(
    COMBINED_CSV,
    index=False,
    encoding="utf-8-sig",
)


# Excel safety
excel_out = combined.copy()

for col in excel_out.columns:

    if isinstance(
        excel_out[col].dtype,
        pd.DatetimeTZDtype,
    ):

        excel_out[col] = (
            excel_out[col]
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )


excel_out.to_excel(
    COMBINED_XLSX,
    index=False,
    engine="openpyxl",
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 110)
print("PARALLEL 40-CANDIDATE SUMMARY")
print("=" * 110)

print(
    "\nRows:",
    len(combined)
)

print(
    "\nS2 QA:"
)

print(
    combined[
        "Cloud/Snow QA Pass"
    ]
    .value_counts(
        dropna=False
    )
)


if (
    "EMIT Coverage Status"
    in combined.columns
):

    print(
        "\nEMIT:"
    )

    print(
        combined[
            "EMIT Coverage Status"
        ]
        .value_counts(
            dropna=False
        )
    )


print(
    "\nMethaneAIR:"
)

print(
    combined[
        "MethaneAIR Independent Evidence"
    ]
    .value_counts(
        dropna=False
    )
)


if (
    "TROPOMI Evidence Role"
    in combined.columns
):

    print(
        "\nTROPOMI:"
    )

    print(
        combined[
            "TROPOMI Evidence Role"
        ]
        .value_counts(
            dropna=False
        )
    )


print(
    "\nPARALLEL VALIDATION STATUS:"
)

print(
    combined[
        "Parallel Validation Status"
    ]
    .value_counts(
        dropna=False
    )
)


print("\n" + "=" * 110)
print("OUTPUTS")
print("=" * 110)

print(
    "\nCombined CSV:"
)

print(
    COMBINED_CSV
)

print(
    "\nCombined XLSX:"
)

print(
    COMBINED_XLSX
)

print(
    "\nMethaneAIR matches:"
)

print(
    METHANEAIR_OUT
)

print(
    "\nLocal existing-file inventory:"
)

print(
    OUTDIR
    /
    "local_existing_sensor_files.txt"
)

print(
    "\nLab existing-file inventory:"
)

print(
    OUTDIR
    /
    "lab_existing_sensor_files.txt"
)


print("\n✅ 40 candidates processed as a batch")
print("✅ Mac local search ran independently")
print("✅ Lab SMB search ran independently")
print("✅ EMIT metadata search ran independently")
print("✅ TROPOMI regional audit ran independently")
print("✅ No image files downloaded")
print("✅ No existing files modified or deleted")

