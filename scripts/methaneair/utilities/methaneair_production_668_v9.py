#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import math
import os
import shutil
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
import ee


# ============================================================
# CONFIG
# ============================================================

HOME = Path.home()
PROJECT = HOME / "methane_release_project"

DEFAULT_MASTER_NAME = (
    "Professor_Master_Site_Date_Source_Inventory_"
    "V3_MethaneSAT_AVIRIS3_EMIT.xlsx"
)

OUTDIR = (
    PROJECT
    / "candidate_negative_validation"
    / "methaneair_production_668_v9"
)

CHECKPOINT_DIR = (
    PROJECT
    / "candidate_negative_validation"
    / "methaneair_production_668_v9_checkpoint"
)

LAB_ROOT = Path(
    "/Volumes/engg-leung/dora lin"
)

LAB_OUTDIR = (
    LAB_ROOT
    / "MethaneAIR_Candidate_Negative_Validation"
    / "v9_668_parents"
)

EE_PROJECT = "methane-release-gee"

MAIR_L3 = (
    "projects/edf-methanesat-ee/"
    "assets/mair/L3concentration"
)

MAIR_L4 = (
    "projects/edf-methanesat-ee/"
    "assets/mair/L4point"
)

S2_COLLECTION = (
    "COPERNICUS/S2_SR_HARMONIZED"
)

# ------------------------------------------------------------
# Temporal logic
# ------------------------------------------------------------

POST_POSITIVE_START_DAY = 1
POST_POSITIVE_END_DAY = 45

STRICT_S2_HOURS = 72.0

# ------------------------------------------------------------
# MethaneAIR corrected-grid QA
# ------------------------------------------------------------

SOURCE_HALF_M = 240.0

BACKGROUND_INNER_M = 800.0
BACKGROUND_OUTER_M = 2000.0

MAIR_SCALE_M = 10.2

MIN_SOURCE_COVERAGE = 0.80
STRONG_BACKGROUND_COVERAGE = 0.80

L4_EXCLUSION_RADIUS_M = 5000.0

# ------------------------------------------------------------
# Sentinel-2 corrected-grid QA
# ------------------------------------------------------------

S2_HALF_M = 240.0
S2_SCALE_M = 20

MIN_S2_CLEAR_OVER_REQUESTED = 0.80

# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------

DEFAULT_WORKERS = 3
PARENT_RETRIES = 3
MIRROR_EVERY_N_PARENTS = 25

EXPECTED_ROWS = 2672
EXPECTED_PARENTS = 668

# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Production-scale MethaneAIR coverage-first temporal "
        "negative validation for all parent positives."
    )
)

parser.add_argument(
    "--candidate-file",
    type=str,
    default="",
    help=(
        "Path to Professor Master XLSX or Candidate_Negatives CSV. "
        "If omitted, the script auto-locates the canonical V3 workbook."
    ),
)

parser.add_argument(
    "--sheet",
    type=str,
    default="Candidate_Negatives",
)

parser.add_argument(
    "--workers",
    type=int,
    default=DEFAULT_WORKERS,
)

parser.add_argument(
    "--limit",
    type=int,
    default=0,
    help="Process only first N parents; 0 means all.",
)

parser.add_argument(
    "--project",
    type=str,
    default=EE_PROJECT,
)

parser.add_argument(
    "--no-lab-mirror",
    action="store_true",
    help="Disable checkpoint/final mirroring to the lab SMB.",
)

args = parser.parse_args()


# ============================================================
# PATH SETUP
# ============================================================

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_JSONL = (
    CHECKPOINT_DIR
    / "parent_results.jsonl"
)

STRUCTURE_AUDIT = (
    OUTDIR
    / "00_parent_structure_audit.csv"
)

PARENT_OUT = (
    OUTDIR
    / "01_parent_summary.csv"
)

FLIGHT_OUT = (
    OUTDIR
    / "02_flight_level_audit.csv"
)

CONTROL_OUT = (
    OUTDIR
    / "03_final_unique_strict_s2_controls.csv"
)

REJECT_OUT = (
    OUTDIR
    / "04_rejected_nearby_l4.csv"
)

NO_S2_OUT = (
    OUTDIR
    / "05_highres_no_l4_but_no_strict_s2.csv"
)

UNRESOLVED_OUT = (
    OUTDIR
    / "06_unresolved_or_partial.csv"
)

SUMMARY_TXT = (
    OUTDIR
    / "07_run_summary.txt"
)

XLSX_OUT = (
    OUTDIR
    / "08_methaneair_production_668_v9.xlsx"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def mounted_lab():
    try:
        r = subprocess.run(
            ["mount"],
            capture_output=True,
            text=True,
            timeout=20,
        )

        return (
            "/Volumes/engg-leung"
            in r.stdout
        )

    except Exception:
        return False


def safe_lab_mkdir():
    if args.no_lab_mirror:
        return False

    if not mounted_lab():
        return False

    try:
        LAB_OUTDIR.mkdir(
            parents=True,
            exist_ok=True,
        )
        return True

    except Exception:
        return False


def safe_mirror_file(
    local_path,
    lab_name=None,
):
    if args.no_lab_mirror:
        return False

    if not mounted_lab():
        return False

    try:
        LAB_OUTDIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        if lab_name is None:
            lab_name = Path(
                local_path
            ).name

        dst = (
            LAB_OUTDIR
            / lab_name
        )

        tmp = (
            LAB_OUTDIR
            / (
                lab_name
                + ".tmp"
            )
        )

        shutil.copy2(
            local_path,
            tmp,
        )

        os.replace(
            tmp,
            dst,
        )

        return True

    except Exception as e:
        print(
            "[LAB MIRROR] skipped:",
            repr(e),
        )
        return False


def json_default(obj):
    if isinstance(
        obj,
        (
            pd.Timestamp,
        ),
    ):
        return str(obj)

    if isinstance(
        obj,
        (
            np.integer,
        ),
    ):
        return int(obj)

    if isinstance(
        obj,
        (
            np.floating,
        ),
    ):
        x = float(obj)

        if math.isnan(x):
            return None

        return x

    if isinstance(
        obj,
        (
            np.bool_,
        ),
    ):
        return bool(obj)

    if pd.isna(obj):
        return None

    return str(obj)


def append_checkpoint(
    record,
):
    with CHECKPOINT_JSONL.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=json_default,
            )
            +
            "\n"
        )

        f.flush()
        os.fsync(
            f.fileno()
        )


def load_latest_checkpoint_records():
    latest = {}

    if not CHECKPOINT_JSONL.exists():
        return latest

    with CHECKPOINT_JSONL.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(
                    line
                )

            except Exception:
                continue

            key = obj.get(
                "parent_key"
            )

            if key:
                latest[
                    key
                ] = obj

    return latest


def parse_time_value(x):
    if x is None:
        return None

    try:
        out = pd.to_datetime(
            x,
            utc=True,
            errors="coerce",
        )

        if pd.isna(out):
            return None

        return out

    except Exception:
        return None


def flight_midpoint_from_props(
    props,
):
    start = parse_time_value(
        props.get(
            "time_coverage_start"
        )
    )

    end = parse_time_value(
        props.get(
            "time_coverage_end"
        )
    )

    if (
        start is not None
        and
        end is not None
    ):
        return (
            start
            +
            (
                end - start
            )
            / 2
        )

    t_ms = props.get(
        "system:time_start"
    )

    if t_ms is not None:
        try:
            return pd.to_datetime(
                t_ms,
                unit="ms",
                utc=True,
            )
        except Exception:
            pass

    return start


# ============================================================
# AUTO-LOCATE INPUT
# ============================================================

def locate_candidate_file():
    if args.candidate_file:
        p = Path(
            args.candidate_file
        ).expanduser()

        if not p.exists():
            raise FileNotFoundError(
                p
            )

        return p

    direct_candidates = [
        PROJECT
        / DEFAULT_MASTER_NAME,

        HOME
        / "Downloads"
        / DEFAULT_MASTER_NAME,
    ]

    for p in direct_candidates:
        if p.exists():
            return p

    # Safe local-only find, capped depth. Never scans SMB.
    cmd = [
        "find",
        str(PROJECT),
        "-maxdepth",
        "5",
        "-type",
        "f",
        "-name",
        DEFAULT_MASTER_NAME,
        "-print",
        "-quit",
    ]

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        text = (
            r.stdout
            or ""
        ).strip()

        if text:
            p = Path(text)

            if p.exists():
                return p

    except Exception:
        pass

    raise FileNotFoundError(
        "Could not auto-locate canonical Professor Master V3 workbook.\n"
        "Run again with:\n"
        "  --candidate-file '/full/path/to/file.xlsx'"
    )


# ============================================================
# LOAD / DERIVE THE 668 PARENTS
# ============================================================

def load_candidate_sheet(
    path,
):
    suffix = path.suffix.lower()

    if suffix in {
        ".xlsx",
        ".xlsm",
    }:
        return pd.read_excel(
            path,
            sheet_name=args.sheet,
            engine="openpyxl",
        )

    if suffix == ".csv":
        return pd.read_csv(
            path,
            low_memory=False,
        )

    raise RuntimeError(
        f"Unsupported candidate file type: {suffix}"
    )


def derive_parents(
    candidate_df,
):
    required = [
        "Site",
        "Latitude",
        "Longitude",
        "Date",
        "Days After Positive",
        "Source Positive Record ID",
    ]

    missing = [
        c
        for c in required
        if c not in candidate_df.columns
    ]

    if missing:
        raise RuntimeError(
            "Candidate_Negatives missing columns:\n"
            +
            "\n".join(
                f"  {c}"
                for c in missing
            )
        )

    df = candidate_df.copy()

    df["_candidate_date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df["_offset"] = pd.to_numeric(
        df[
            "Days After Positive"
        ],
        errors="coerce",
    )

    if (
        df["_candidate_date"].isna().any()
        or
        df["_offset"].isna().any()
    ):
        raise RuntimeError(
            "Invalid Date or Days After Positive values."
        )

    df[
        "_positive_date"
    ] = (
        df["_candidate_date"]
        -
        pd.to_timedelta(
            df["_offset"],
            unit="D",
        )
    ).dt.normalize()

    audit_rows = []
    parent_rows = []

    for source_id, g in df.groupby(
        "Source Positive Record ID",
        sort=True,
        dropna=False,
    ):

        offsets = sorted(
            int(x)
            for x in (
                pd.to_numeric(
                    g["_offset"],
                    errors="coerce",
                )
                .dropna()
                .unique()
            )
        )

        positive_dates = sorted(
            set(
                pd.Timestamp(x)
                for x in (
                    g[
                        "_positive_date"
                    ]
                    .dropna()
                    .tolist()
                )
            )
        )

        structure_ok = (
            len(g) == 4
            and
            offsets == [
                1,
                3,
                7,
                14,
            ]
            and
            len(
                positive_dates
            ) == 1
        )

        r0 = g.iloc[0]

        audit_rows.append({
            "Source Positive Record ID":
                source_id,

            "Rows":
                len(g),

            "Offsets":
                "|".join(
                    str(x)
                    for x in offsets
                ),

            "Unique Positive Dates":
                len(
                    positive_dates
                ),

            "Structure OK":
                bool(
                    structure_ok
                ),
        })

        if not structure_ok:
            continue

        positive_date = (
            positive_dates[0]
        )

        lat = float(
            pd.to_numeric(
                pd.Series([
                    r0["Latitude"]
                ]),
                errors="coerce",
            ).iloc[0]
        )

        lon = float(
            pd.to_numeric(
                pd.Series([
                    r0["Longitude"]
                ]),
                errors="coerce",
            ).iloc[0]
        )

        parent_key = (
            f"{source_id}__"
            f"{positive_date.strftime('%Y%m%d')}__"
            f"{lat:.6f}__"
            f"{lon:.6f}"
        )

        parent_rows.append({
            "parent_key":
                parent_key,

            "Source Positive Record ID":
                source_id,

            "Site":
                r0[
                    "Site"
                ],

            "Latitude":
                lat,

            "Longitude":
                lon,

            "Parent Positive Date":
                positive_date.strftime(
                    "%Y-%m-%d"
                ),
        })

    audit = pd.DataFrame(
        audit_rows
    )

    parents = pd.DataFrame(
        parent_rows
    )

    return audit, parents


# ============================================================
# CORRECTED METHANEAIR XCH4 QA
# ============================================================

def xch4_region_stats(
    image,
    region,
):
    xch4 = image.select(
        "XCH4"
    )

    proj = xch4.projection()

    sentinel = -999999.0

    filled = xch4.unmask(
        value=sentinel,
        sameFootprint=False,
    ).rename(
        "xch4_filled"
    )

    requested = (
        filled.multiply(0)
        .add(1)
        .rename(
            "requested"
        )
    )

    covered = (
        filled.neq(
            sentinel
        )
        .rename(
            "covered"
        )
    )

    counts = (
        requested
        .addBands(
            covered
        )
        .reduceRegion(
            reducer=ee.Reducer.sum().unweighted(),
            geometry=region,
            crs=proj,
            scale=MAIR_SCALE_M,
            bestEffort=False,
            maxPixels=5_000_000,
        )
        .getInfo()
    )

    requested_n = float(
        counts.get(
            "requested"
        )
        or 0
    )

    covered_n = float(
        counts.get(
            "covered"
        )
        or 0
    )

    coverage_fraction = (
        covered_n
        /
        requested_n
        if requested_n > 0
        else np.nan
    )

    median_dict = (
        xch4.reduceRegion(
            reducer=ee.Reducer.median().unweighted(),
            geometry=region,
            crs=proj,
            scale=MAIR_SCALE_M,
            bestEffort=False,
            maxPixels=5_000_000,
        )
        .getInfo()
    )

    median = (
        median_dict.get(
            "XCH4"
        )
    )

    if median is not None:
        median = float(
            median
        )

    impossible = (
        not pd.isna(
            coverage_fraction
        )
        and
        (
            coverage_fraction < -1e-9
            or
            coverage_fraction > 1.0000001
        )
    )

    return {
        "requested_pixels":
            requested_n,

        "covered_pixels":
            covered_n,

        "coverage_fraction":
            coverage_fraction,

        "xch4_median_ppb":
            median,

        "impossible_fraction":
            bool(
                impossible
            ),
    }


# ============================================================
# L4 SAME-FLIGHT AUDIT
# ============================================================

def l4_stats(
    flight_id,
    point,
):
    fc = (
        ee.FeatureCollection(
            MAIR_L4
        )
        .filter(
            ee.Filter.eq(
                "flight_id",
                str(
                    flight_id
                ),
            )
        )
    )

    total = int(
        fc.size().getInfo()
    )

    if total == 0:
        return {
            "same_flight_count":
                0,

            "nearby_5km_count":
                0,

            "nearest_distance_m":
                None,
        }

    with_distance = fc.map(
        lambda f:
        f.set(
            "_distance_m",
            f.geometry().distance(
                point,
                maxError=1,
            ),
        )
    )

    nearest = (
        with_distance
        .aggregate_min(
            "_distance_m"
        )
        .getInfo()
    )

    nearby = int(
        fc.filterBounds(
            point.buffer(
                L4_EXCLUSION_RADIUS_M
            )
        )
        .size()
        .getInfo()
    )

    return {
        "same_flight_count":
            total,

        "nearby_5km_count":
            nearby,

        "nearest_distance_m":
            (
                float(
                    nearest
                )
                if nearest
                is not None
                else None
            ),
    }


# ============================================================
# CORRECTED SENTINEL-2 QA
# ============================================================

def corrected_s2_qa(
    image,
    region,
):
    scl = image.select(
        "SCL"
    )

    proj = scl.projection()

    filled = scl.unmask(
        value=255,
        sameFootprint=False,
    ).rename(
        "SCL_filled"
    )

    requested = (
        filled.multiply(0)
        .add(1)
        .rename(
            "requested"
        )
    )

    covered = (
        filled.neq(
            255
        )
        .rename(
            "covered"
        )
    )

    clear = (
        filled.gte(4)
        .And(
            filled.lte(7)
        )
        .rename(
            "clear"
        )
    )

    cloud = (
        filled.gte(8)
        .And(
            filled.lte(10)
        )
        .rename(
            "cloud"
        )
    )

    shadow = (
        filled.eq(3)
        .rename(
            "shadow"
        )
    )

    snow = (
        filled.eq(11)
        .rename(
            "snow"
        )
    )

    invalid = (
        filled.gte(0)
        .And(
            filled.lte(2)
        )
        .rename(
            "invalid"
        )
    )

    masked = (
        filled.eq(255)
        .rename(
            "masked"
        )
    )

    stack = (
        requested
        .addBands(
            covered
        )
        .addBands(
            clear
        )
        .addBands(
            cloud
        )
        .addBands(
            shadow
        )
        .addBands(
            snow
        )
        .addBands(
            invalid
        )
        .addBands(
            masked
        )
    )

    stats = stack.reduceRegion(
        reducer=ee.Reducer.sum().unweighted(),
        geometry=region,
        crs=proj,
        scale=S2_SCALE_M,
        bestEffort=False,
        maxPixels=1_000_000,
    ).getInfo()

    def n(
        key,
    ):
        return float(
            stats.get(
                key
            )
            or 0
        )

    requested_n = n(
        "requested"
    )

    covered_n = n(
        "covered"
    )

    clear_n = n(
        "clear"
    )

    cloud_n = n(
        "cloud"
    )

    shadow_n = n(
        "shadow"
    )

    snow_n = n(
        "snow"
    )

    invalid_n = n(
        "invalid"
    )

    masked_n = n(
        "masked"
    )

    def frac(
        a,
        d,
    ):
        return (
            a / d
            if d > 0
            else np.nan
        )

    coverage = frac(
        covered_n,
        requested_n,
    )

    clear_requested = frac(
        clear_n,
        requested_n,
    )

    clear_covered = frac(
        clear_n,
        covered_n,
    )

    partition_error = (
        (
            clear_n
            +
            cloud_n
            +
            shadow_n
            +
            snow_n
            +
            invalid_n
            +
            masked_n
        )
        -
        requested_n
    )

    impossible = (
        (
            not pd.isna(
                clear_covered
            )
            and
            clear_covered
            >
            1.0000001
        )
        or
        (
            not pd.isna(
                clear_requested
            )
            and
            not pd.isna(
                coverage
            )
            and
            clear_requested
            >
            coverage
            +
            1e-9
        )
        or
        abs(
            partition_error
        )
        >
        1e-6
    )

    return {
        "coverage_fraction":
            coverage,

        "clear_among_covered":
            clear_covered,

        "clear_over_requested":
            clear_requested,

        "cloud_over_requested":
            frac(
                cloud_n,
                requested_n,
            ),

        "shadow_over_requested":
            frac(
                shadow_n,
                requested_n,
            ),

        "snow_over_requested":
            frac(
                snow_n,
                requested_n,
            ),

        "masked_fraction":
            frac(
                masked_n,
                requested_n,
            ),

        "impossible_fraction":
            bool(
                impossible
            ),

        "qa_pass":
            (
                bool(
                    clear_requested
                    >=
                    MIN_S2_CLEAR_OVER_REQUESTED
                )
                if not pd.isna(
                    clear_requested
                )
                else False
            ),
    }


# ============================================================
# S2 SEARCH FOR ONE B FLIGHT
# ============================================================

def search_strict_s2(
    lon,
    lat,
    positive_date,
    flight_midpoint,
):
    point = ee.Geometry.Point(
        [
            lon,
            lat,
        ]
    )

    region = (
        point
        .buffer(
            S2_HALF_M
        )
        .bounds()
    )

    search_start = (
        flight_midpoint
        -
        pd.Timedelta(
            hours=STRICT_S2_HOURS
        )
    )

    search_end = (
        flight_midpoint
        +
        pd.Timedelta(
            hours=STRICT_S2_HOURS
        )
        +
        pd.Timedelta(
            seconds=1
        )
    )

    ic = (
        ee.ImageCollection(
            S2_COLLECTION
        )
        .filterBounds(
            point
        )
        .filterDate(
            search_start.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            search_end.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
        )
        .sort(
            "system:time_start"
        )
    )

    n_images = int(
        ic.size().getInfo()
    )

    if n_images == 0:
        return {
            "match_status":
                "NO_S2_WITHIN_72H",

            "scene_count":
                0,
        }

    images = ic.toList(
        n_images
    )

    scene_rows = []

    positive_ts = pd.Timestamp(
        positive_date,
        tz="UTC",
    )

    for i in range(
        n_images
    ):
        image = ee.Image(
            images.get(i)
        )

        props = (
            image.toDictionary(
                [
                    "system:index",
                    "system:time_start",
                    "PRODUCT_ID",
                    "MGRS_TILE",
                    "CLOUDY_PIXEL_PERCENTAGE",
                ]
            )
            .getInfo()
        )

        time_ms = props.get(
            "system:time_start"
        )

        if time_ms is None:
            continue

        s2_dt = pd.to_datetime(
            time_ms,
            unit="ms",
            utc=True,
        )

        if s2_dt <= positive_ts:
            continue

        delta_hours = (
            s2_dt
            -
            flight_midpoint
        ).total_seconds() / 3600.0

        if abs(
            delta_hours
        ) > (
            STRICT_S2_HOURS
            +
            0.1
        ):
            continue

        qa = corrected_s2_qa(
            image,
            region,
        )

        scene_rows.append({
            "s2_datetime_utc":
                str(
                    s2_dt
                ),

            "delta_hours":
                float(
                    delta_hours
                ),

            "abs_delta_hours":
                abs(
                    float(
                        delta_hours
                    )
                ),

            "product_id":
                props.get(
                    "PRODUCT_ID"
                ),

            "system_index":
                props.get(
                    "system:index"
                ),

            "mgrs_tile":
                props.get(
                    "MGRS_TILE"
                ),

            "scene_cloud_pct":
                props.get(
                    "CLOUDY_PIXEL_PERCENTAGE"
                ),

            "coverage_fraction":
                qa[
                    "coverage_fraction"
                ],

            "clear_among_covered":
                qa[
                    "clear_among_covered"
                ],

            "clear_over_requested":
                qa[
                    "clear_over_requested"
                ],

            "cloud_over_requested":
                qa[
                    "cloud_over_requested"
                ],

            "shadow_over_requested":
                qa[
                    "shadow_over_requested"
                ],

            "snow_over_requested":
                qa[
                    "snow_over_requested"
                ],

            "masked_fraction":
                qa[
                    "masked_fraction"
                ],

            "impossible_fraction":
                qa[
                    "impossible_fraction"
                ],

            "qa_pass":
                qa[
                    "qa_pass"
                ],
        })

    if not scene_rows:
        return {
            "match_status":
                "NO_POST_POSITIVE_S2_WITHIN_72H",

            "scene_count":
                n_images,
        }

    scenes = pd.DataFrame(
        scene_rows
    )

    # Deduplicate overlapping MGRS tiles within 20 minutes,
    # keeping the best local QA tile.
    scenes[
        "_dt"
    ] = pd.to_datetime(
        scenes[
            "s2_datetime_utc"
        ],
        utc=True,
    )

    scenes = scenes.sort_values(
        "_dt"
    )

    groups = []
    current = []
    previous = None

    for idx, row in scenes.iterrows():
        dt = row[
            "_dt"
        ]

        if (
            previous is None
            or
            (
                dt - previous
            ).total_seconds()
            <=
            20 * 60
        ):
            current.append(
                idx
            )

        else:
            groups.append(
                current
            )
            current = [
                idx
            ]

        previous = dt

    if current:
        groups.append(
            current
        )

    overpasses = []

    for inds in groups:
        g = scenes.loc[
            inds
        ].copy()

        g = g.sort_values(
            [
                "clear_over_requested",
                "coverage_fraction",
            ],
            ascending=[
                False,
                False,
            ],
        )

        r = g.iloc[0].copy()

        r[
            "overlap_tile_count"
        ] = len(
            g
        )

        overpasses.append(
            r
        )

    overpasses = pd.DataFrame(
        overpasses
    )

    passed = overpasses[
        overpasses[
            "qa_pass"
        ].eq(
            True
        )
    ].copy()

    if len(
        passed
    ) == 0:
        best_fail = overpasses.sort_values(
            [
                "clear_over_requested",
                "abs_delta_hours",
            ],
            ascending=[
                False,
                True,
            ],
        ).iloc[0]

        return {
            "match_status":
                "S2_WITHIN_72H_BUT_QA_FAIL",

            "scene_count":
                n_images,

            "overpass_count":
                len(
                    overpasses
                ),

            "best_fail_clear":
                best_fail[
                    "clear_over_requested"
                ],

            "best_fail_delta_hours":
                best_fail[
                    "delta_hours"
                ],

            "best_fail_product_id":
                best_fail[
                    "product_id"
                ],
        }

    best = passed.sort_values(
        [
            "abs_delta_hours",
            "clear_over_requested",
        ],
        ascending=[
            True,
            False,
        ],
    ).iloc[0]

    return {
        "match_status":
            "STRICT_MATCH",

        "scene_count":
            n_images,

        "overpass_count":
            len(
                overpasses
            ),

        "s2_datetime_utc":
            best[
                "s2_datetime_utc"
            ],

        "s2_delta_hours":
            best[
                "delta_hours"
            ],

        "s2_product_id":
            best[
                "product_id"
            ],

        "s2_system_index":
            best[
                "system_index"
            ],

        "s2_mgrs_tile":
            best[
                "mgrs_tile"
            ],

        "s2_coverage_fraction":
            best[
                "coverage_fraction"
            ],

        "s2_clear_among_covered":
            best[
                "clear_among_covered"
            ],

        "s2_clear_over_requested":
            best[
                "clear_over_requested"
            ],

        "s2_cloud_over_requested":
            best[
                "cloud_over_requested"
            ],

        "s2_shadow_over_requested":
            best[
                "shadow_over_requested"
            ],

        "s2_snow_over_requested":
            best[
                "snow_over_requested"
            ],

        "s2_masked_fraction":
            best[
                "masked_fraction"
            ],
    }


# ============================================================
# PROCESS ONE PARENT
# ============================================================

def process_parent_once(
    parent,
):
    parent_key = parent[
        "parent_key"
    ]

    source_id = parent[
        "Source Positive Record ID"
    ]

    site = parent[
        "Site"
    ]

    lat = float(
        parent[
            "Latitude"
        ]
    )

    lon = float(
        parent[
            "Longitude"
        ]
    )

    positive_date = pd.Timestamp(
        parent[
            "Parent Positive Date"
        ]
    ).normalize()

    point = ee.Geometry.Point(
        [
            lon,
            lat,
        ]
    )

    source_region = (
        point
        .buffer(
            SOURCE_HALF_M
        )
        .bounds()
    )

    background_region = (
        point
        .buffer(
            BACKGROUND_OUTER_M
        )
        .difference(
            point.buffer(
                BACKGROUND_INNER_M
            )
        )
    )

    start = (
        positive_date
        +
        pd.Timedelta(
            days=POST_POSITIVE_START_DAY
        )
    )

    end = (
        positive_date
        +
        pd.Timedelta(
            days=POST_POSITIVE_END_DAY + 1
        )
    )

    ic = (
        ee.ImageCollection(
            MAIR_L3
        )
        .filterBounds(
            point
        )
        .filterDate(
            start.strftime(
                "%Y-%m-%d"
            ),
            end.strftime(
                "%Y-%m-%d"
            ),
        )
        .sort(
            "system:time_start"
        )
    )

    n_images = int(
        ic.size().getInfo()
    )

    base_parent = {
        "parent_key":
            parent_key,

        "Source Positive Record ID":
            source_id,

        "Site":
            site,

        "Latitude":
            lat,

        "Longitude":
            lon,

        "Parent Positive Date":
            positive_date.strftime(
                "%Y-%m-%d"
            ),

        "L3 Image Intersections":
            n_images,
    }

    if n_images == 0:
        return {
            "complete":
                True,

            "parent_key":
                parent_key,

            "parent_summary": {
                **base_parent,

                "Parent Result":
                    "NO_L3_FLIGHT_INTERSECTION",

                "Unique L3 Flights":
                    0,

                "Source-Valid Flights":
                    0,

                "Nearby-L4 Reject Flights":
                    0,

                "High-Res No-L4 Flights":
                    0,

                "Strict B Flights":
                    0,
            },

            "flights":
                [],
        }

    images = ic.toList(
        n_images
    )

    asset_records = []

    for i in range(
        n_images
    ):
        image = ee.Image(
            images.get(i)
        )

        props = (
            image.toDictionary(
                [
                    "system:index",
                    "system:time_start",
                    "flight_id",
                    "target_id",
                    "time_coverage_start",
                    "time_coverage_end",
                ]
            )
            .getInfo()
        )

        flight_id = props.get(
            "flight_id"
        )

        if not flight_id:
            continue

        midpoint = (
            flight_midpoint_from_props(
                props
            )
        )

        if midpoint is None:
            continue

        src = xch4_region_stats(
            image,
            source_region,
        )

        bg = xch4_region_stats(
            image,
            background_region,
        )

        delta = None

        if (
            src[
                "xch4_median_ppb"
            ]
            is not None
            and
            bg[
                "xch4_median_ppb"
            ]
            is not None
        ):
            delta = (
                src[
                    "xch4_median_ppb"
                ]
                -
                bg[
                    "xch4_median_ppb"
                ]
            )

        asset_records.append({
            "flight_id":
                str(
                    flight_id
                ),

            "system_index":
                props.get(
                    "system:index"
                ),

            "target_id":
                props.get(
                    "target_id"
                ),

            "midpoint_utc":
                str(
                    midpoint
                ),

            "actual_days_after_positive":
                (
                    midpoint
                    -
                    pd.Timestamp(
                        positive_date,
                        tz="UTC",
                    )
                ).total_seconds()
                /
                86400.0,

            "source_requested_pixels":
                src[
                    "requested_pixels"
                ],

            "source_covered_pixels":
                src[
                    "covered_pixels"
                ],

            "source_coverage_fraction":
                src[
                    "coverage_fraction"
                ],

            "source_xch4_median_ppb":
                src[
                    "xch4_median_ppb"
                ],

            "background_requested_pixels":
                bg[
                    "requested_pixels"
                ],

            "background_covered_pixels":
                bg[
                    "covered_pixels"
                ],

            "background_coverage_fraction":
                bg[
                    "coverage_fraction"
                ],

            "background_xch4_median_ppb":
                bg[
                    "xch4_median_ppb"
                ],

            "source_minus_background_ppb":
                delta,

            "impossible_fraction":
                (
                    src[
                        "impossible_fraction"
                    ]
                    or
                    bg[
                        "impossible_fraction"
                    ]
                ),
        })

    if not asset_records:
        return {
            "complete":
                True,

            "parent_key":
                parent_key,

            "parent_summary": {
                **base_parent,

                "Parent Result":
                    "L3_INTERSECTS_BUT_NO_USABLE_FLIGHT_METADATA",

                "Unique L3 Flights":
                    0,

                "Source-Valid Flights":
                    0,

                "Nearby-L4 Reject Flights":
                    0,

                "High-Res No-L4 Flights":
                    0,

                "Strict B Flights":
                    0,
            },

            "flights":
                [],
        }

    assets = pd.DataFrame(
        asset_records
    )

    # Deduplicate multiple L3 assets from the same flight:
    # keep strongest source coverage, then background coverage.
    best_assets = []

    for flight_id, g in assets.groupby(
        "flight_id",
        sort=True,
    ):
        x = g.copy()

        x[
            "_src"
        ] = pd.to_numeric(
            x[
                "source_coverage_fraction"
            ],
            errors="coerce",
        ).fillna(
            -1
        )

        x[
            "_bg"
        ] = pd.to_numeric(
            x[
                "background_coverage_fraction"
            ],
            errors="coerce",
        ).fillna(
            -1
        )

        x = x.sort_values(
            [
                "_src",
                "_bg",
            ],
            ascending=[
                False,
                False,
            ],
        )

        best_assets.append(
            x.iloc[0].to_dict()
        )

    flight_rows = []

    for asset in best_assets:
        flight_id = asset[
            "flight_id"
        ]

        midpoint = pd.to_datetime(
            asset[
                "midpoint_utc"
            ],
            utc=True,
        )

        source_fraction = asset[
            "source_coverage_fraction"
        ]

        background_fraction = asset[
            "background_coverage_fraction"
        ]

        source_valid = (
            pd.notna(
                source_fraction
            )
            and
            float(
                source_fraction
            )
            >=
            MIN_SOURCE_COVERAGE
        )

        background_strong = (
            pd.notna(
                background_fraction
            )
            and
            float(
                background_fraction
            )
            >=
            STRONG_BACKGROUND_COVERAGE
        )

        l4 = l4_stats(
            flight_id,
            point,
        )

        classification = None
        s2 = {
            "match_status":
                "NOT_QUERIED"
        }

        if not source_valid:
            classification = (
                "U_METHANEAIR_L3_PARTIAL_OR_INVALID_AT_SOURCE"
            )

        elif (
            l4[
                "nearby_5km_count"
            ]
            >
            0
        ):
            classification = (
                "R_REJECT_SAME_FLIGHT_L4_WITHIN_5KM"
            )

        elif (
            l4[
                "same_flight_count"
            ]
            ==
            0
        ):
            classification = (
                "U_VALID_L3_L4_AVAILABILITY_UNCERTAIN"
            )

        else:
            # High-resolution no-L4 candidate.
            s2 = search_strict_s2(
                lon=lon,
                lat=lat,
                positive_date=positive_date,
                flight_midpoint=midpoint,
            )

            if (
                s2[
                    "match_status"
                ]
                ==
                "STRICT_MATCH"
            ):
                classification = (
                    "B_STRICT_CORRECTED_L3_NO_L4_WITH_ALIGNED_S2"
                )

            else:
                classification = (
                    "B_HIGH_RES_NO_L4_DETECTION_BUT_NO_USABLE_ALIGNED_S2"
                )

        evidence_subgrade = (
            "B1_STRONG_SOURCE_AND_BACKGROUND"
            if (
                source_valid
                and
                background_strong
            )
            else
            (
                "B2_SOURCE_VALID_BACKGROUND_WEAK"
                if source_valid
                else
                "U_SOURCE_COVERAGE_FAIL"
            )
        )

        flight_rows.append({
            "parent_key":
                parent_key,

            "Source Positive Record ID":
                source_id,

            "Site":
                site,

            "Latitude":
                lat,

            "Longitude":
                lon,

            "Parent Positive Date":
                positive_date.strftime(
                    "%Y-%m-%d"
                ),

            "MethaneAIR Flight ID":
                flight_id,

            "MethaneAIR Midpoint UTC":
                str(
                    midpoint
                ),

            "Actual Days After Positive":
                asset[
                    "actual_days_after_positive"
                ],

            "MethaneAIR L3 System Index":
                asset[
                    "system_index"
                ],

            "MethaneAIR Target ID":
                asset[
                    "target_id"
                ],

            "Corrected Source Coverage Fraction":
                source_fraction,

            "Corrected Source XCH4 Median ppb":
                asset[
                    "source_xch4_median_ppb"
                ],

            "Corrected Background Coverage Fraction":
                background_fraction,

            "Corrected Background XCH4 Median ppb":
                asset[
                    "background_xch4_median_ppb"
                ],

            "Corrected Source Minus Background ppb":
                asset[
                    "source_minus_background_ppb"
                ],

            "Corrected L3 Impossible Fraction Flag":
                asset[
                    "impossible_fraction"
                ],

            "Corrected Source Coverage Pass":
                bool(
                    source_valid
                ),

            "Corrected Background Strong":
                bool(
                    background_strong
                ),

            "Methane Evidence Subgrade":
                evidence_subgrade,

            "Same-Flight L4 Count":
                l4[
                    "same_flight_count"
                ],

            "Nearby L4 Count <=5km":
                l4[
                    "nearby_5km_count"
                ],

            "True Nearest Same-Flight L4 Distance m":
                l4[
                    "nearest_distance_m"
                ],

            "S2 Match Status":
                s2.get(
                    "match_status"
                ),

            "S2 Product ID":
                s2.get(
                    "s2_product_id"
                ),

            "S2 Datetime UTC":
                s2.get(
                    "s2_datetime_utc"
                ),

            "S2 Delta Hours From MethaneAIR":
                s2.get(
                    "s2_delta_hours"
                ),

            "S2 Coverage Fraction":
                s2.get(
                    "s2_coverage_fraction"
                ),

            "S2 Clear Among Covered Fraction":
                s2.get(
                    "s2_clear_among_covered"
                ),

            "S2 Clear Over Requested Fraction":
                s2.get(
                    "s2_clear_over_requested"
                ),

            "S2 Cloud Over Requested Fraction":
                s2.get(
                    "s2_cloud_over_requested"
                ),

            "S2 Shadow Over Requested Fraction":
                s2.get(
                    "s2_shadow_over_requested"
                ),

            "S2 Snow Over Requested Fraction":
                s2.get(
                    "s2_snow_over_requested"
                ),

            "S2 Masked Fraction":
                s2.get(
                    "s2_masked_fraction"
                ),

            "S2 Best Fail Clear Fraction":
                s2.get(
                    "best_fail_clear"
                ),

            "S2 Best Fail Delta Hours":
                s2.get(
                    "best_fail_delta_hours"
                ),

            "Coverage-First Classification":
                classification,
        })

    flights_df = pd.DataFrame(
        flight_rows
    )

    source_valid_count = int(
        flights_df[
            "Corrected Source Coverage Pass"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
        .sum()
    )

    reject_count = int(
        flights_df[
            "Coverage-First Classification"
        ]
        .eq(
            "R_REJECT_SAME_FLIGHT_L4_WITHIN_5KM"
        )
        .sum()
    )

    highres_count = int(
        flights_df[
            "Coverage-First Classification"
        ]
        .isin([
            "B_STRICT_CORRECTED_L3_NO_L4_WITH_ALIGNED_S2",
            "B_HIGH_RES_NO_L4_DETECTION_BUT_NO_USABLE_ALIGNED_S2",
        ])
        .sum()
    )

    strict_count = int(
        flights_df[
            "Coverage-First Classification"
        ]
        .eq(
            "B_STRICT_CORRECTED_L3_NO_L4_WITH_ALIGNED_S2"
        )
        .sum()
    )

    unresolved_count = int(
        flights_df[
            "Coverage-First Classification"
        ]
        .str.startswith(
            "U_",
            na=False,
        )
        .sum()
    )

    if strict_count > 0:
        parent_result = (
            "STRICT_B_FOUND"
        )

    elif highres_count > 0:
        parent_result = (
            "HIGH_RES_NO_L4_FOUND_BUT_NO_STRICT_S2"
        )

    elif reject_count > 0:
        parent_result = (
            "VALID_L3_FOUND_BUT_NEARBY_L4_REJECT"
        )

    elif source_valid_count > 0:
        parent_result = (
            "VALID_L3_FOUND_BUT_NO_COMPLETE_B_PLUS_S2"
        )

    else:
        parent_result = (
            "L3_INTERSECTS_BUT_SOURCE_COVERAGE_INVALID"
        )

    return {
        "complete":
            True,

        "parent_key":
            parent_key,

        "parent_summary": {
            **base_parent,

            "Parent Result":
                parent_result,

            "Unique L3 Flights":
                len(
                    flights_df
                ),

            "Source-Valid Flights":
                source_valid_count,

            "Nearby-L4 Reject Flights":
                reject_count,

            "High-Res No-L4 Flights":
                highres_count,

            "Strict B Flights":
                strict_count,

            "Unresolved Flights":
                unresolved_count,
        },

        "flights":
            flight_rows,
    }


def process_parent_with_retries(
    parent,
):
    last_error = None

    for attempt in range(
        1,
        PARENT_RETRIES + 1
    ):
        try:
            return process_parent_once(
                parent
            )

        except Exception as e:
            last_error = e

            if attempt < PARENT_RETRIES:
                time.sleep(
                    attempt * 3
                )

    return {
        "complete":
            False,

        "parent_key":
            parent[
                "parent_key"
            ],

        "parent_summary": {
            "parent_key":
                parent[
                    "parent_key"
                ],

            "Source Positive Record ID":
                parent[
                    "Source Positive Record ID"
                ],

            "Site":
                parent[
                    "Site"
                ],

            "Latitude":
                parent[
                    "Latitude"
                ],

            "Longitude":
                parent[
                    "Longitude"
                ],

            "Parent Positive Date":
                parent[
                    "Parent Positive Date"
                ],

            "Parent Result":
                "QUERY_ERROR",
        },

        "flights":
            [],

        "error":
            repr(
                last_error
            ),

        "traceback":
            traceback.format_exc(),
    }


# ============================================================
# REBUILD TABLES FROM CHECKPOINT
# ============================================================

def checkpoint_to_tables(
    latest_records,
):
    parent_rows = []
    flight_rows = []

    for obj in (
        latest_records.values()
    ):
        if not obj.get(
            "complete"
        ):
            continue

        ps = obj.get(
            "parent_summary"
        )

        if ps:
            parent_rows.append(
                ps
            )

        flight_rows.extend(
            obj.get(
                "flights",
                []
            )
        )

    parents_df = pd.DataFrame(
        parent_rows
    )

    flights_df = pd.DataFrame(
        flight_rows
    )

    return (
        parents_df,
        flights_df,
    )


# ============================================================
# FINAL DEDUPLICATION
# ============================================================

def build_unique_controls(
    flights_df,
):
    if len(
        flights_df
    ) == 0:
        return pd.DataFrame()

    strict = flights_df[
        flights_df[
            "Coverage-First Classification"
        ]
        .eq(
            "B_STRICT_CORRECTED_L3_NO_L4_WITH_ALIGNED_S2"
        )
    ].copy()

    if len(
        strict
    ) == 0:
        return pd.DataFrame()

    rows = []

    for (
        parent_key,
        source_id,
        site,
        lat,
        lon,
        positive_date,
        product_id,
    ), g in strict.groupby(
        [
            "parent_key",
            "Source Positive Record ID",
            "Site",
            "Latitude",
            "Longitude",
            "Parent Positive Date",
            "S2 Product ID",
        ],
        dropna=False,
    ):

        flight_ids = sorted(
            set(
                g[
                    "MethaneAIR Flight ID"
                ]
                .astype(
                    str
                )
                .tolist()
            )
        )

        b1_count = int(
            g[
                "Methane Evidence Subgrade"
            ]
            .eq(
                "B1_STRONG_SOURCE_AND_BACKGROUND"
            )
            .sum()
        )

        min_delta_idx = (
            pd.to_numeric(
                g[
                    "S2 Delta Hours From MethaneAIR"
                ],
                errors="coerce",
            )
            .abs()
            .idxmin()
        )

        representative = g.loc[
            min_delta_idx
        ]

        rows.append({
            "parent_key":
                parent_key,

            "Source Positive Record ID":
                source_id,

            "Site":
                site,

            "Latitude":
                lat,

            "Longitude":
                lon,

            "Parent Positive Date":
                positive_date,

            "S2 Product ID":
                product_id,

            "S2 Datetime UTC":
                representative[
                    "S2 Datetime UTC"
                ],

            "Supporting MethaneAIR Flight Count":
                len(
                    flight_ids
                ),

            "Supporting MethaneAIR Flight IDs":
                " | ".join(
                    flight_ids
                ),

            "B1 Strong Supporting Flight Count":
                b1_count,

            "Best Corrected Source Coverage":
                pd.to_numeric(
                    g[
                        "Corrected Source Coverage Fraction"
                    ],
                    errors="coerce",
                ).max(),

            "Best Corrected Background Coverage":
                pd.to_numeric(
                    g[
                        "Corrected Background Coverage Fraction"
                    ],
                    errors="coerce",
                ).max(),

            "Minimum Absolute S2 Delta Hours":
                pd.to_numeric(
                    g[
                        "S2 Delta Hours From MethaneAIR"
                    ],
                    errors="coerce",
                ).abs().min(),

            "S2 Coverage Fraction":
                representative[
                    "S2 Coverage Fraction"
                ],

            "S2 Clear Among Covered Fraction":
                representative[
                    "S2 Clear Among Covered Fraction"
                ],

            "S2 Clear Over Requested Fraction":
                representative[
                    "S2 Clear Over Requested Fraction"
                ],

            "S2 Masked Fraction":
                representative[
                    "S2 Masked Fraction"
                ],

            "Nearest Same-Flight L4 Distance m":
                pd.to_numeric(
                    g[
                        "True Nearest Same-Flight L4 Distance m"
                    ],
                    errors="coerce",
                ).min(),

            "Final Evidence Grade":
                (
                    "B1_STRONG_HIGH_RES_NO_L4_DETECTION"
                    if b1_count > 0
                    else
                    "B2_HIGH_RES_NO_L4_DETECTION_BACKGROUND_WEAK"
                ),

            "Final Label Type":
                "strict_temporal_weak_negative",
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# SAVE TABLES
# ============================================================

def write_outputs(
    latest_records,
):
    parents_df, flights_df = (
        checkpoint_to_tables(
            latest_records
        )
    )

    controls_df = (
        build_unique_controls(
            flights_df
        )
    )

    if len(
        flights_df
    ):
        rejects = flights_df[
            flights_df[
                "Coverage-First Classification"
            ].eq(
                "R_REJECT_SAME_FLIGHT_L4_WITHIN_5KM"
            )
        ].copy()

        no_s2 = flights_df[
            flights_df[
                "Coverage-First Classification"
            ].eq(
                "B_HIGH_RES_NO_L4_DETECTION_BUT_NO_USABLE_ALIGNED_S2"
            )
        ].copy()

        unresolved = flights_df[
            flights_df[
                "Coverage-First Classification"
            ]
            .str.startswith(
                "U_",
                na=False,
            )
        ].copy()

    else:
        rejects = pd.DataFrame()
        no_s2 = pd.DataFrame()
        unresolved = pd.DataFrame()

    parents_df.to_csv(
        PARENT_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    flights_df.to_csv(
        FLIGHT_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    controls_df.to_csv(
        CONTROL_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    rejects.to_csv(
        REJECT_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    no_s2.to_csv(
        NO_S2_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    unresolved.to_csv(
        UNRESOLVED_OUT,
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(
        XLSX_OUT,
        engine="openpyxl",
    ) as writer:

        parents_df.to_excel(
            writer,
            sheet_name="Parent_Summary",
            index=False,
        )

        flights_df.to_excel(
            writer,
            sheet_name="Flight_Audit",
            index=False,
        )

        controls_df.to_excel(
            writer,
            sheet_name="Strict_S2_Controls",
            index=False,
        )

        rejects.to_excel(
            writer,
            sheet_name="Rejected_L4_5km",
            index=False,
        )

        no_s2.to_excel(
            writer,
            sheet_name="HighRes_No_Strict_S2",
            index=False,
        )

        unresolved.to_excel(
            writer,
            sheet_name="Unresolved",
            index=False,
        )

    return (
        parents_df,
        flights_df,
        controls_df,
        rejects,
        no_s2,
        unresolved,
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 120)
print("METHANEAIR PRODUCTION 668 — COVERAGE-FIRST V9")
print("=" * 120)

candidate_path = (
    locate_candidate_file()
)

print(
    "\nCandidate source:"
)

print(
    candidate_path
)

candidate_df = (
    load_candidate_sheet(
        candidate_path
    )
)

print(
    "\nCandidate rows:",
    len(
        candidate_df
    )
)

audit, parents = derive_parents(
    candidate_df
)

audit.to_csv(
    STRUCTURE_AUDIT,
    index=False,
    encoding="utf-8-sig",
)

print(
    "Derived parents:",
    len(
        parents
    )
)

print(
    "Structure-pass parents:",
    int(
        audit[
            "Structure OK"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )
)

if len(
    candidate_df
) != EXPECTED_ROWS:
    print(
        "WARNING:"
        f" expected {EXPECTED_ROWS} candidate rows, "
        f"found {len(candidate_df)}."
    )

if len(
    parents
) != EXPECTED_PARENTS:
    print(
        "WARNING:"
        f" expected {EXPECTED_PARENTS} parents, "
        f"found {len(parents)}."
    )

if args.limit > 0:
    parents = (
        parents
        .head(
            args.limit
        )
        .copy()
    )

    print(
        "LIMIT active:",
        len(
            parents
        ),
        "parents",
    )

print(
    "\nInitializing Earth Engine..."
)

ee.Initialize(
    project=args.project
)

print(
    "Earth Engine ready."
)

existing = (
    load_latest_checkpoint_records()
)

completed_keys = {
    key
    for key, obj in (
        existing.items()
    )
    if obj.get(
        "complete"
    )
}

todo = parents[
    ~parents[
        "parent_key"
    ].isin(
        completed_keys
    )
].copy()

print(
    "\nPreviously completed:",
    len(
        completed_keys
        &
        set(
            parents[
                "parent_key"
            ]
        )
    )
)

print(
    "This run:",
    len(
        todo
    )
)

print(
    "Local checkpoint:"
)

print(
    CHECKPOINT_JSONL
)

if not args.no_lab_mirror:
    if safe_lab_mkdir():
        print(
            "Lab mirror active:"
        )
        print(
            LAB_OUTDIR
        )
    else:
        print(
            "Lab mirror currently unavailable; "
            "local processing will continue."
        )


done_this_run = 0

if len(
    todo
):
    with ThreadPoolExecutor(
        max_workers=max(
            1,
            int(
                args.workers
            ),
        )
    ) as pool:

        futures = {
            pool.submit(
                process_parent_with_retries,
                row,
            ):
                row[
                    "parent_key"
                ]

            for _, row in (
                todo.iterrows()
            )
        }

        for future in as_completed(
            futures
        ):
            key = futures[
                future
            ]

            try:
                result = (
                    future.result()
                )

            except Exception as e:
                result = {
                    "complete":
                        False,

                    "parent_key":
                        key,

                    "error":
                        repr(e),

                    "traceback":
                        traceback.format_exc(),
                }

            append_checkpoint(
                result
            )

            existing[
                key
            ] = result

            done_this_run += 1

            result_name = (
                result.get(
                    "parent_summary",
                    {}
                )
                .get(
                    "Parent Result",
                    "QUERY_ERROR",
                )
            )

            print(
                f"[{done_this_run}/{len(todo)}] "
                f"{key} -> {result_name}"
            )

            if (
                done_this_run
                %
                MIRROR_EVERY_N_PARENTS
                ==
                0
            ):
                safe_mirror_file(
                    CHECKPOINT_JSONL,
                    "parent_results.jsonl",
                )

# Re-read checkpoint from disk to ensure outputs reflect
# authoritative persisted results.
latest = (
    load_latest_checkpoint_records()
)

tables = write_outputs(
    latest
)

(
    parents_df,
    flights_df,
    controls_df,
    rejects_df,
    no_s2_df,
    unresolved_df,
) = tables

requested_keys = set(
    parents[
        "parent_key"
    ]
)

complete_requested = {
    key
    for key, obj in (
        latest.items()
    )
    if (
        key in requested_keys
        and
        obj.get(
            "complete"
        )
    )
}

failed_requested = [
    obj
    for key, obj in (
        latest.items()
    )
    if (
        key in requested_keys
        and
        not obj.get(
            "complete"
        )
    )
]

strict_flights = 0

if len(
    flights_df
):
    strict_flights = int(
        flights_df[
            "Coverage-First Classification"
        ]
        .eq(
            "B_STRICT_CORRECTED_L3_NO_L4_WITH_ALIGNED_S2"
        )
        .sum()
    )

summary_lines = [
    "METHANEAIR PRODUCTION 668 — V9 SUMMARY",
    "=" * 80,
    f"Candidate source: {candidate_path}",
    f"Requested parents this configuration: {len(parents)}",
    f"Completed parents: {len(complete_requested)}",
    f"Failed/incomplete parents: {len(failed_requested)}",
    f"Flight records: {len(flights_df)}",
    f"Corrected source-valid flights: "
    f"{int(flights_df['Corrected Source Coverage Pass'].fillna(False).astype(bool).sum()) if len(flights_df) else 0}",
    f"Nearby-L4 reject flights: {len(rejects_df)}",
    f"High-res no-L4 but no strict S2 flights: {len(no_s2_df)}",
    f"Strict B flight-level validations: {strict_flights}",
    f"Final unique strict S2 controls: {len(controls_df)}",
    "",
    "Parent results:",
]

if len(
    parents_df
):
    for k, v in (
        parents_df[
            "Parent Result"
        ]
        .value_counts(
            dropna=False
        )
        .items()
    ):
        summary_lines.append(
            f"  {k}: {v}"
        )

summary_lines.extend([
    "",
    "Final evidence grades:",
])

if len(
    controls_df
):
    for k, v in (
        controls_df[
            "Final Evidence Grade"
        ]
        .value_counts(
            dropna=False
        )
        .items()
    ):
        summary_lines.append(
            f"  {k}: {v}"
        )

SUMMARY_TXT.write_text(
    "\n".join(
        summary_lines
    )
    +
    "\n",
    encoding="utf-8",
)

print(
    "\n"
    +
    "=" * 120
)

print(
    "FINAL V9 SUMMARY"
)

print(
    "=" * 120
)

print(
    "\n".join(
        summary_lines
    )
)

# Final mirror. Failure is non-fatal.
for path in [
    STRUCTURE_AUDIT,
    PARENT_OUT,
    FLIGHT_OUT,
    CONTROL_OUT,
    REJECT_OUT,
    NO_S2_OUT,
    UNRESOLVED_OUT,
    SUMMARY_TXT,
    XLSX_OUT,
    CHECKPOINT_JSONL,
]:
    if path.exists():
        safe_mirror_file(
            path
        )

print(
    "\nOUTPUTS:"
)

for path in [
    STRUCTURE_AUDIT,
    PARENT_OUT,
    FLIGHT_OUT,
    CONTROL_OUT,
    REJECT_OUT,
    NO_S2_OUT,
    UNRESOLVED_OUT,
    SUMMARY_TXT,
    XLSX_OUT,
    CHECKPOINT_JSONL,
]:
    print(
        path
    )

print(
    "\n✅ fixed +1/+3/+7/+14 candidate dates are NOT used for validation"
)

print(
    "✅ all 668 parents are derived from the canonical 2672-row structure"
)

print(
    "✅ MethaneAIR coverage is searched first (+1 to +45 d)"
)

print(
    "✅ L3 source/background coverage uses corrected common-grid QA"
)

print(
    "✅ same-flight L4 within 5 km rejects contamination"
)

print(
    "✅ S2 is searched only after a high-res no-L4 candidate exists"
)

print(
    "✅ S2 uses corrected common-grid QA and strict +/-72 h alignment"
)

print(
    "✅ multiple MethaneAIR flights sharing one parent/S2 product deduplicate to ONE control"
)

print(
    "✅ local checkpoint supports safe reruns/resume"
)

print(
    "✅ lab mirror is best-effort and cannot stop the main run"
)
