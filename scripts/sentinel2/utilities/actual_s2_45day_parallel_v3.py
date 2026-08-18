from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import traceback
import math
import time

import numpy as np
import pandas as pd
import requests
import ee


# ============================================================
# CONFIG
# ============================================================

HOME = Path.home()
PROJECT = HOME / "methane_release_project"
LAB_ROOT = Path("/Volumes/engg-leung/dora lin")

INPUT = (
    PROJECT
    / "candidate_negative_validation"
    / "pilot_10_positive_40_candidates_s2qa.csv"
)

OUTDIR = (
    PROJECT
    / "candidate_negative_validation"
    / "actual_s2_45day_parallel_v3"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

EE_PROJECT = "methane-release-gee"

# ------------------------------------------------------------
# Actual-S2 temporal search
# ------------------------------------------------------------

SEARCH_START_DAY = 1
SEARCH_END_DAY = 45

TEMPORAL_BINS = [
    ("EARLY", 1, 10),
    ("MIDDLE", 11, 25),
    ("LATE", 26, 45),
]

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

HALF_PATCH_M = 240
S2_SCALE_M = 20

MIN_CLEAR_OVER_REQUESTED = 0.80

S2_PARENT_WORKERS = 3


# ------------------------------------------------------------
# EMIT
# ------------------------------------------------------------

EMIT_COLLECTION = "C3242680113-LPCLOUD"

CMR_URL = (
    "https://cmr.earthdata.nasa.gov/"
    "search/granules.json"
)

EMIT_WORKERS = 8


# ------------------------------------------------------------
# TROPOMI
# ------------------------------------------------------------

TROPOMI_COLLECTION = (
    "COPERNICUS/S5P/OFFL/L3_CH4"
)

TROPOMI_CH4_BAND = (
    "CH4_column_volume_mixing_ratio_dry_air_bias_corrected"
)

TROPOMI_UNC_BAND = (
    "CH4_column_volume_mixing_ratio_dry_air_uncertainty"
)

TROPOMI_SOURCE_RADIUS_M = 20_000
TROPOMI_BG_INNER_M = 30_000
TROPOMI_BG_OUTER_M = 70_000
TROPOMI_SCALE_M = 7000

TROPOMI_WORKERS = 4


# ------------------------------------------------------------
# Local / Lab metadata
# ------------------------------------------------------------

METADATA_PATTERNS = [
    "*methaneair*.csv",
    "*methane_air*.csv",
    "*pcannon*.csv",
    "*methaneair*.json",
    "*flight*.csv",
    "*coverage*.csv",
    "*retrieval*.csv",
]

LOCAL_MAXDEPTH = 7
LAB_MAXDEPTH = 5

LOCAL_FIND_TIMEOUT = 120
LAB_FIND_TIMEOUT = 240

METHANEAIR_RADIUS_KM = 10.0
METHANEAIR_CONTEXT_DAYS = 14

MAX_RETRIES = 4


# ============================================================
# INPUT
# ============================================================

print("=" * 110)
print("ACTUAL-S2 +45 DAY PARALLEL VALIDATION V3")
print("=" * 110)

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

pilot = pd.read_csv(
    INPUT,
    low_memory=False,
)

if len(pilot) != 40:
    raise RuntimeError(
        f"Expected 40 pilot rows; got {len(pilot)}"
    )

pilot["_nominal_date"] = pd.to_datetime(
    pilot["Date"],
    errors="coerce",
)

if pilot["_nominal_date"].isna().any():
    raise RuntimeError(
        "Invalid Date values in pilot."
    )


# ============================================================
# DERIVE THE 10 ORIGINAL POSITIVE DATES
# ============================================================

parent_rows = []

for parent_num, g in pilot.groupby(
    "Pilot Parent Number",
    sort=True,
):

    positive_dates = []

    for _, r in g.iterrows():

        positive_date = (
            pd.Timestamp(
                r["_nominal_date"]
            ).normalize()
            -
            pd.Timedelta(
                days=int(
                    r["Resolved Offset Days"]
                )
            )
        )

        positive_dates.append(
            positive_date
        )

    unique = sorted(
        set(positive_dates)
    )

    if len(unique) != 1:
        raise RuntimeError(
            f"Parent {parent_num}: "
            f"inconsistent positive dates {unique}"
        )

    r0 = g.iloc[0]

    parent_rows.append({
        "Pilot Parent Number":
            int(parent_num),

        "Source Positive Record ID":
            r0["Source Positive Record ID"],

        "Site":
            r0["Site"],

        "Latitude":
            float(r0["Latitude"]),

        "Longitude":
            float(r0["Longitude"]),

        "Parent Positive Date":
            unique[0],
    })


parents = pd.DataFrame(
    parent_rows
)

print("\nParents:", len(parents))

print(
    parents[
        [
            "Pilot Parent Number",
            "Site",
            "Parent Positive Date",
        ]
    ].to_string(index=False)
)


# ============================================================
# EARTH ENGINE
# ============================================================

print("\nInitializing Earth Engine...")

ee.Initialize(
    project=EE_PROJECT
)

print("Earth Engine ready.")


# ============================================================
# SENTINEL-2 LOCAL QA
# ============================================================

def s2_qa(
    image,
    region,
):

    scl = image.select("SCL")

    clear = (
        scl.eq(4)
        .Or(scl.eq(5))
        .Or(scl.eq(6))
        .Or(scl.eq(7))
        .rename("clear")
    )

    cloud = (
        scl.eq(8)
        .Or(scl.eq(9))
        .Or(scl.eq(10))
        .rename("cloud")
    )

    shadow = (
        scl.eq(3)
        .rename("shadow")
    )

    snow = (
        scl.eq(11)
        .rename("snow")
    )

    invalid = (
        scl.eq(0)
        .Or(scl.eq(1))
        .Or(scl.eq(2))
        .rename("invalid")
    )

    stack = (
        clear
        .addBands(cloud)
        .addBands(shadow)
        .addBands(snow)
        .addBands(invalid)
    )

    sums = (
        stack.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=S2_SCALE_M,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    valid_result = (
        scl.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=S2_SCALE_M,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    requested_result = (
        ee.Image.constant(1)
        .rename("requested")
        .reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=S2_SCALE_M,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    requested = float(
        requested_result.get("requested")
        or 0
    )

    valid = float(
        valid_result.get("SCL")
        or 0
    )

    clear_px = float(
        sums.get("clear")
        or 0
    )

    cloud_px = float(
        sums.get("cloud")
        or 0
    )

    shadow_px = float(
        sums.get("shadow")
        or 0
    )

    snow_px = float(
        sums.get("snow")
        or 0
    )

    invalid_px = float(
        sums.get("invalid")
        or 0
    )

    def div(a, b):
        return (
            a / b
            if b > 0
            else np.nan
        )

    valid_fraction = div(
        valid,
        requested,
    )

    clear_requested = div(
        clear_px,
        requested,
    )

    clear_valid = div(
        clear_px,
        valid,
    )

    masked_fraction = (
        1 - valid_fraction
        if not pd.isna(valid_fraction)
        else np.nan
    )

    return {
        "requested_px":
            requested,

        "valid_px":
            valid,

        "valid_fraction":
            valid_fraction,

        "clear_requested":
            clear_requested,

        "clear_valid":
            clear_valid,

        "cloud_requested":
            div(cloud_px, requested),

        "shadow_requested":
            div(shadow_px, requested),

        "snow_requested":
            div(snow_px, requested),

        "invalid_requested":
            div(invalid_px, requested),

        "masked_fraction":
            masked_fraction,

        "qa_pass":
            (
                not pd.isna(
                    clear_requested
                )
                and
                clear_requested
                >=
                MIN_CLEAR_OVER_REQUESTED
            ),
    }


# ============================================================
# SEARCH +1 ... +45 DAY S2
# ============================================================

def search_parent(
    parent,
):

    parent_num = int(
        parent["Pilot Parent Number"]
    )

    positive = pd.Timestamp(
        parent["Parent Positive Date"]
    ).normalize()

    lat = float(
        parent["Latitude"]
    )

    lon = float(
        parent["Longitude"]
    )

    point = ee.Geometry.Point(
        [lon, lat]
    )

    region = (
        point
        .buffer(HALF_PATCH_M)
        .bounds()
    )

    start = (
        positive
        +
        pd.Timedelta(
            days=SEARCH_START_DAY
        )
    )

    end = (
        positive
        +
        pd.Timedelta(
            days=SEARCH_END_DAY + 1
        )
    )

    collection = (
        ee.ImageCollection(
            S2_COLLECTION
        )
        .filterBounds(point)
        .filterDate(
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
        .sort("system:time_start")
    )

    n = int(
        collection.size().getInfo()
    )

    image_list = collection.toList(n)

    records = []

    for i in range(n):

        image = ee.Image(
            image_list.get(i)
        )

        props = (
            image.toDictionary([
                "system:index",
                "system:time_start",
                "PRODUCT_ID",
                "MGRS_TILE",
                "CLOUDY_PIXEL_PERCENTAGE",
            ])
            .getInfo()
        )

        time_ms = props.get(
            "system:time_start"
        )

        if time_ms is None:
            continue

        dt = pd.to_datetime(
            time_ms,
            unit="ms",
            utc=True,
        )

        dt_naive = (
            dt
            .tz_convert(None)
            .normalize()
        )

        offset = int(
            (
                dt_naive
                -
                positive
            ).days
        )

        if not (
            SEARCH_START_DAY
            <= offset
            <= SEARCH_END_DAY
        ):
            continue

        qa = s2_qa(
            image,
            region,
        )

        records.append({
            "Pilot Parent Number":
                parent_num,

            "Source Positive Record ID":
                parent[
                    "Source Positive Record ID"
                ],

            "Site":
                parent["Site"],

            "Latitude":
                lat,

            "Longitude":
                lon,

            "Parent Positive Date":
                positive.strftime(
                    "%Y-%m-%d"
                ),

            "Actual S2 Datetime UTC":
                str(dt),

            "Actual S2 Date":
                dt.strftime(
                    "%Y-%m-%d"
                ),

            "Actual Offset Days":
                offset,

            "S2 Product ID":
                props.get(
                    "PRODUCT_ID"
                ),

            "S2 System Index":
                props.get(
                    "system:index"
                ),

            "S2 MGRS Tile":
                props.get(
                    "MGRS_TILE"
                ),

            "S2 Scene Cloud Percentage":
                props.get(
                    "CLOUDY_PIXEL_PERCENTAGE"
                ),

            "S2 Valid SCL Fraction":
                qa[
                    "valid_fraction"
                ],

            "S2 Clear Over Requested Fraction":
                qa[
                    "clear_requested"
                ],

            "S2 Clear Among Valid Fraction":
                qa[
                    "clear_valid"
                ],

            "S2 Cloud Over Requested Fraction":
                qa[
                    "cloud_requested"
                ],

            "S2 Shadow Over Requested Fraction":
                qa[
                    "shadow_requested"
                ],

            "S2 Snow Over Requested Fraction":
                qa[
                    "snow_requested"
                ],

            "S2 Invalid Class Fraction":
                qa[
                    "invalid_requested"
                ],

            "S2 Masked Fraction":
                qa[
                    "masked_fraction"
                ],

            "S2 QA Pass":
                (
                    "pass"
                    if qa[
                        "qa_pass"
                    ]
                    else "fail"
                ),
        })

    return records


# ============================================================
# PHASE A
# ============================================================

print("\n" + "=" * 110)
print("PHASE A — +1 TO +45 DAY ACTUAL SENTINEL-2 SEARCH")
print("=" * 110)

scene_records = []

with ThreadPoolExecutor(
    max_workers=S2_PARENT_WORKERS
) as pool:

    futures = {
        pool.submit(
            search_parent,
            row,
        ):
            int(
                row[
                    "Pilot Parent Number"
                ]
            )

        for _, row in (
            parents.iterrows()
        )
    }

    for f in as_completed(
        futures
    ):

        parent_num = futures[f]

        try:

            recs = f.result()

            scene_records.extend(
                recs
            )

            print(
                f"[S2] parent {parent_num}: "
                f"{len(recs)} scenes"
            )

        except Exception as e:

            print(
                f"[S2] parent {parent_num} FAILED:",
                repr(e)
            )


scenes = pd.DataFrame(
    scene_records
)

SCENE_OUT = (
    OUTDIR
    / "01_all_s2_scenes_1to45d.csv"
)

scenes.to_csv(
    SCENE_OUT,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# DEDUP OVERLAPPING S2 TILES
# ============================================================

def collapse_overpasses(
    g,
):

    if len(g) == 0:
        return g

    x = g.copy()

    x["_dt"] = pd.to_datetime(
        x[
            "Actual S2 Datetime UTC"
        ],
        utc=True,
    )

    x = x.sort_values(
        "_dt"
    )

    groups = []

    current = []
    previous = None

    for idx, row in (
        x.iterrows()
    ):

        dt = row["_dt"]

        if (
            previous is None
            or
            (
                dt - previous
            ).total_seconds()
            <= 20 * 60
        ):

            current.append(idx)

        else:

            groups.append(current)
            current = [idx]

        previous = dt

    if current:
        groups.append(current)

    retained = []

    for inds in groups:

        gg = x.loc[
            inds
        ].copy()

        gg = gg.sort_values(
            [
                "S2 Clear Over Requested Fraction",
                "S2 Valid SCL Fraction",
            ],
            ascending=[
                False,
                False,
            ],
        )

        best = gg.iloc[0].copy()

        best[
            "Overlapping Tile Count"
        ] = len(gg)

        retained.append(
            best
        )

    out = pd.DataFrame(
        retained
    )

    return out.drop(
        columns=[
            "_dt"
        ],
        errors="ignore",
    )


overpass_frames = []

if len(scenes):

    for _, g in scenes.groupby(
        "Pilot Parent Number"
    ):

        overpass_frames.append(
            collapse_overpasses(g)
        )


overpasses = (
    pd.concat(
        overpass_frames,
        ignore_index=True,
    )
    if overpass_frames
    else pd.DataFrame()
)

OVERPASS_OUT = (
    OUTDIR
    / "02_s2_unique_overpasses_1to45d.csv"
)

overpasses.to_csv(
    OVERPASS_OUT,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SELECT ONE QA-PASS S2 PER EARLY/MIDDLE/LATE BIN
# ============================================================

selected_rows = []

for _, parent in parents.iterrows():

    parent_num = int(
        parent["Pilot Parent Number"]
    )

    if len(overpasses):

        pg = overpasses[
            overpasses[
                "Pilot Parent Number"
            ]
            ==
            parent_num
        ].copy()

    else:

        pg = pd.DataFrame()

    for label, day_min, day_max in (
        TEMPORAL_BINS
    ):

        if len(pg):

            q = pg[
                (
                    pg[
                        "Actual Offset Days"
                    ]
                    >= day_min
                )
                &
                (
                    pg[
                        "Actual Offset Days"
                    ]
                    <= day_max
                )
                &
                (
                    pg[
                        "S2 QA Pass"
                    ]
                    ==
                    "pass"
                )
            ].copy()

        else:

            q = pd.DataFrame()

        base = {
            "Actual Candidate ID":
                (
                    f"ACTUAL_P"
                    f"{parent_num:02d}_"
                    f"{label}"
                ),

            "Pilot Parent Number":
                parent_num,

            "Source Positive Record ID":
                parent[
                    "Source Positive Record ID"
                ],

            "Site":
                parent["Site"],

            "Latitude":
                parent["Latitude"],

            "Longitude":
                parent["Longitude"],

            "Parent Positive Date":
                pd.Timestamp(
                    parent[
                        "Parent Positive Date"
                    ]
                ).strftime(
                    "%Y-%m-%d"
                ),

            "Temporal Bin":
                label,

            "Bin Start Day":
                day_min,

            "Bin End Day":
                day_max,
        }

        if len(q) == 0:

            selected_rows.append({
                **base,

                "Selection Status":
                    "no_qa_pass_s2",

                "Actual S2 Date":
                    pd.NA,

                "Actual S2 Datetime UTC":
                    pd.NA,

                "Actual Offset Days":
                    pd.NA,

                "S2 Product ID":
                    pd.NA,

                "S2 Clear Over Requested Fraction":
                    pd.NA,

                "S2 Valid SCL Fraction":
                    pd.NA,

                "S2 Clear Among Valid Fraction":
                    pd.NA,
            })

            continue

        midpoint = (
            day_min + day_max
        ) / 2

        q[
            "_mid_error"
        ] = (
            q[
                "Actual Offset Days"
            ]
            -
            midpoint
        ).abs()

        # Primary = best local QA.
        # Secondary = best valid coverage.
        # Tertiary = closest to temporal-bin midpoint.
        q = q.sort_values(
            [
                "S2 Clear Over Requested Fraction",
                "S2 Valid SCL Fraction",
                "_mid_error",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )

        r = q.iloc[0]

        selected_rows.append({
            **base,

            "Selection Status":
                "selected",

            "Actual S2 Date":
                r[
                    "Actual S2 Date"
                ],

            "Actual S2 Datetime UTC":
                r[
                    "Actual S2 Datetime UTC"
                ],

            "Actual Offset Days":
                int(
                    r[
                        "Actual Offset Days"
                    ]
                ),

            "S2 Product ID":
                r[
                    "S2 Product ID"
                ],

            "S2 MGRS Tile":
                r[
                    "S2 MGRS Tile"
                ],

            "S2 Scene Cloud Percentage":
                r[
                    "S2 Scene Cloud Percentage"
                ],

            "S2 Clear Over Requested Fraction":
                r[
                    "S2 Clear Over Requested Fraction"
                ],

            "S2 Valid SCL Fraction":
                r[
                    "S2 Valid SCL Fraction"
                ],

            "S2 Clear Among Valid Fraction":
                r[
                    "S2 Clear Among Valid Fraction"
                ],

            "S2 Cloud Over Requested Fraction":
                r[
                    "S2 Cloud Over Requested Fraction"
                ],

            "S2 Shadow Over Requested Fraction":
                r[
                    "S2 Shadow Over Requested Fraction"
                ],

            "S2 Snow Over Requested Fraction":
                r[
                    "S2 Snow Over Requested Fraction"
                ],

            "S2 Masked Fraction":
                r[
                    "S2 Masked Fraction"
                ],
        })


selected = pd.DataFrame(
    selected_rows
)

SELECTED_OUT = (
    OUTDIR
    / "03_selected_early_middle_late_s2.csv"
)

selected.to_csv(
    SELECTED_OUT,
    index=False,
    encoding="utf-8-sig",
)


selected_valid = selected[
    selected[
        "Selection Status"
    ]
    ==
    "selected"
].copy()


print("\n" + "=" * 110)
print("ACTUAL S2 SELECTION SUMMARY")
print("=" * 110)

print(
    selected[
        "Selection Status"
    ]
    .value_counts(
        dropna=False
    )
)

print("\nBY TEMPORAL BIN:")

print(
    selected.groupby(
        "Temporal Bin"
    )[
        "Selection Status"
    ]
    .value_counts()
    .unstack(
        fill_value=0
    )
)

print(
    "\nSelected actual-S2 candidates:",
    len(selected_valid),
    "/ 30 possible"
)


# ============================================================
# EXISTING METADATA INDEX HELPERS
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


def decode_output(x):

    if x is None:
        return ""

    if isinstance(x, bytes):
        return x.decode(
            "utf-8",
            errors="replace",
        )

    return str(x)


def cached_paths(
    label,
):

    previous_root = (
        PROJECT
        / "candidate_negative_validation"
        / "parallel_multisource_40"
    )

    filename = (
        "local_existing_sensor_files.txt"
        if label == "LOCAL"
        else
        "lab_existing_sensor_files.txt"
    )

    p = (
        previous_root
        /
        filename
    )

    paths = []

    if not p.exists():
        return paths

    for line in p.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():

        line = line.strip()

        if (
            not line
            or line.startswith(
                "STATUS:"
            )
            or line.startswith(
                "COUNT:"
            )
        ):
            continue

        paths.append(
            Path(line)
        )

    return paths


def discover_metadata(
    root,
    maxdepth,
    timeout,
    label,
):

    old_paths = cached_paths(
        label
    )

    if (
        label == "LAB"
        and not mounted_lab()
    ):

        return {
            "status":
                "smb_not_mounted_using_cache",

            "paths":
                old_paths,
        }

    root = Path(root)

    if not root.exists():

        return {
            "status":
                "root_missing_using_cache",

            "paths":
                old_paths,
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

    for i, pattern in enumerate(
        METADATA_PATTERNS
    ):

        if i:
            cmd.append("-o")

        cmd.extend([
            "-iname",
            pattern,
        ])

    cmd.extend([
        ")",
        "-print",
    ])

    new_paths = []

    status = "complete"

    try:

        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = (
            r.stdout
            or ""
        )

        if r.returncode != 0:

            status = (
                f"returncode_{r.returncode}"
            )

    except subprocess.TimeoutExpired as e:

        status = "timeout_partial"

        output = decode_output(
            e.stdout
        )

    except Exception as e:

        status = (
            "error:"
            +
            repr(e)
        )

        output = ""

    for line in output.splitlines():

        line = line.strip()

        if line:

            new_paths.append(
                Path(line)
            )

    combined = []

    seen = set()

    for p in (
        old_paths
        +
        new_paths
    ):

        s = str(p)

        if s in seen:
            continue

        seen.add(s)

        combined.append(p)

    return {
        "status":
            status,

        "paths":
            combined,
    }


def norm(s):

    return (
        str(s)
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_col(
    cols,
    names,
):

    cmap = {
        norm(c): c
        for c in cols
    }

    for n in names:

        nn = norm(n)

        if nn in cmap:

            return cmap[nn]

    for n in names:

        nn = norm(n)

        for k, original in (
            cmap.items()
        ):

            if nn in k:

                return original

    return None


def haversine(
    lat0,
    lon0,
    lat,
    lon,
):

    R = 6371.0088

    p0 = np.radians(lat0)
    q0 = np.radians(lon0)

    p = np.radians(
        np.asarray(
            lat,
            dtype=float,
        )
    )

    q = np.radians(
        np.asarray(
            lon,
            dtype=float,
        )
    )

    a = (
        np.sin(
            (p - p0) / 2
        ) ** 2
        +
        np.cos(p0)
        *
        np.cos(p)
        *
        np.sin(
            (q - q0) / 2
        ) ** 2
    )

    return (
        2
        *
        R
        *
        np.arcsin(
            np.sqrt(a)
        )
    )


def parse_time(
    df,
):

    full = find_col(
        df.columns,
        [
            "datetime",
            "timestamp",
            "observation_time",
            "acquisition_time",
            "scene_datetime",
            "time_coverage_start",
        ]
    )

    if full is not None:

        return pd.to_datetime(
            df[full],
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

    time_col = find_col(
        df.columns,
        [
            "utc_time",
            "time_utc",
        ]
    )

    if time_col is None:

        return pd.to_datetime(
            df[date_col],
            errors="coerce",
            utc=True,
        )

    text = (
        df[date_col]
        .astype("string")
        .fillna("")
        .str.strip()
        +
        " "
        +
        df[time_col]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    return pd.to_datetime(
        text,
        errors="coerce",
        utc=True,
    )


def metadata_branch(
    root,
    maxdepth,
    timeout,
    label,
):

    discovery = discover_metadata(
        root,
        maxdepth,
        timeout,
        label,
    )

    print(
        f"\n[{label}] metadata discovery "
        f"{discovery['status']} | "
        f"{len(discovery['paths'])} paths"
    )

    inventory = (
        OUTDIR
        /
        f"04_{label.lower()}_metadata_inventory.txt"
    )

    with open(
        inventory,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"STATUS: {discovery['status']}\n"
        )

        f.write(
            f"COUNT: {len(discovery['paths'])}\n\n"
        )

        for p in (
            discovery["paths"]
        ):
            f.write(
                str(p)
                +
                "\n"
            )

    matches = []

    csv_files = [
        p
        for p in discovery["paths"]
        if p.suffix.lower()
        ==
        ".csv"
    ]

    for p in csv_files:

        try:

            df = pd.read_csv(
                p,
                low_memory=False,
            )

        except Exception:

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
            or
            lon_col is None
        ):

            continue

        times = parse_time(
            df
        )

        if times is None:

            continue

        lats = pd.to_numeric(
            df[lat_col],
            errors="coerce",
        )

        lons = pd.to_numeric(
            df[lon_col],
            errors="coerce",
        )

        valid = (
            lats.notna()
            &
            lons.notna()
            &
            times.notna()
        )

        if not valid.any():

            continue

        temp = pd.DataFrame({
            "_row":
                df.index,

            "_lat":
                lats,

            "_lon":
                lons,

            "_time":
                times,
        }).loc[
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
            selected_valid.iterrows()
        ):

            candidate_date = (
                pd.to_datetime(
                    cand[
                        "Actual S2 Date"
                    ],
                    utc=True,
                )
            )

            parent_date = (
                pd.to_datetime(
                    cand[
                        "Parent Positive Date"
                    ],
                    utc=True,
                )
            )

            dt_days = (
                temp["_time"]
                -
                candidate_date
            ).dt.total_seconds() / 86400

            mask = (
                dt_days.abs()
                <=
                METHANEAIR_CONTEXT_DAYS
            )

            if not mask.any():

                continue

            sub = temp.loc[
                mask
            ].copy()

            sub["_dt"] = (
                dt_days.loc[
                    mask
                ]
            )

            sub["_distance"] = (
                haversine(
                    float(
                        cand[
                            "Latitude"
                        ]
                    ),
                    float(
                        cand[
                            "Longitude"
                        ]
                    ),
                    sub["_lat"],
                    sub["_lon"],
                )
            )

            sub = sub[
                sub["_distance"]
                <=
                METHANEAIR_RADIUS_KM
            ]

            for _, hit in (
                sub.iterrows()
            ):

                source_idx = int(
                    hit["_row"]
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

                # Original parent positive itself is not
                # independent evidence.
                original_parent = (
                    str(source_id)
                    ==
                    str(
                        cand[
                            "Source Positive Record ID"
                        ]
                    )
                )

                parent_day_record = (
                    hit["_time"].date()
                    ==
                    parent_date.date()
                )

                if (
                    original_parent
                    or
                    parent_day_record
                ):

                    continue

                matches.append({
                    "Actual Candidate ID":
                        cand[
                            "Actual Candidate ID"
                        ],

                    "Origin":
                        label,

                    "Source File":
                        str(p),

                    "Source Record ID":
                        source_id,

                    "Source Datetime UTC":
                        str(
                            hit["_time"]
                        ),

                    "Distance km":
                        float(
                            hit[
                                "_distance"
                            ]
                        ),

                    "Time Difference days":
                        float(
                            hit["_dt"]
                        ),

                    "Same Candidate Day":
                        (
                            hit[
                                "_time"
                            ].date()
                            ==
                            candidate_date.date()
                        ),
                })

    return {
        "status":
            discovery[
                "status"
            ],

        "matches":
            pd.DataFrame(
                matches
            ),
    }


# ============================================================
# EMIT
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

        "point":
            f"{lon},{lat}",

        "temporal":
            f"{start},{end}",

        "page_size":
            100,
    }

    last = None

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
                    "MethaneFuse-validation-v3"
                },
            )

            r.raise_for_status()

            return (
                r.json()
                .get("feed", {})
                .get("entry", [])
            )

        except Exception as e:

            last = e

            time.sleep(
                attempt
            )

    raise RuntimeError(
        repr(last)
    )


def emit_one(
    row,
):

    d = pd.Timestamp(
        row[
            "Actual S2 Date"
        ]
    )

    lat = float(
        row["Latitude"]
    )

    lon = float(
        row["Longitude"]
    )

    exact = cmr_query(
        lat,
        lon,
        d.strftime(
            "%Y-%m-%dT00:00:00Z"
        ),
        d.strftime(
            "%Y-%m-%dT23:59:59Z"
        ),
    )

    context = cmr_query(
        lat,
        lon,
        (
            d
            -
            pd.Timedelta(days=3)
        ).strftime(
            "%Y-%m-%dT00:00:00Z"
        ),
        (
            d
            +
            pd.Timedelta(days=3)
        ).strftime(
            "%Y-%m-%dT23:59:59Z"
        ),
    )

    return {
        "Actual Candidate ID":
            row[
                "Actual Candidate ID"
            ],

        "EMIT Exact Date Count":
            len(exact),

        "EMIT +/-3d Count":
            len(context),

        "EMIT Exact Titles":
            " | ".join(
                str(
                    x.get("title")
                )
                for x in exact
            ),

        "EMIT +/-3d Titles":
            " | ".join(
                str(
                    x.get("title")
                )
                for x in context
            ),

        "EMIT Coverage Status":
            (
                "exact_date"
                if exact
                else
                (
                    "context_only"
                    if context
                    else
                    "none"
                )
            ),
    }


def emit_batch():

    rows = []

    with ThreadPoolExecutor(
        max_workers=EMIT_WORKERS
    ) as pool:

        futures = [
            pool.submit(
                emit_one,
                r,
            )
            for _, r in (
                selected_valid.iterrows()
            )
        ]

        done = 0

        for f in as_completed(
            futures
        ):

            try:

                rows.append(
                    f.result()
                )

            except Exception as e:

                print(
                    "[EMIT] error:",
                    repr(e)
                )

            done += 1

            if (
                done % 10 == 0
                or
                done
                ==
                len(
                    selected_valid
                )
            ):

                print(
                    f"[EMIT] "
                    f"{done}/"
                    f"{len(selected_valid)}"
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TROPOMI
# ============================================================

def tropomi_stats(
    collection,
    point,
):

    count = int(
        collection.size().getInfo()
    )

    if count == 0:

        return {
            "count": 0,
            "regional": None,
            "background": None,
            "delta": None,
            "uncertainty": None,
            "valid": 0,
        }

    image = (
        collection
        .select([
            TROPOMI_CH4_BAND,
            TROPOMI_UNC_BAND,
        ])
        .median()
    )

    source = (
        point.buffer(
            TROPOMI_SOURCE_RADIUS_M
        )
    )

    background = (
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

    src = (
        image.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=source,
            scale=TROPOMI_SCALE_M,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    bg = (
        image
        .select(
            TROPOMI_CH4_BAND
        )
        .reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=background,
            scale=TROPOMI_SCALE_M,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    cnt = (
        image
        .select(
            TROPOMI_CH4_BAND
        )
        .reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=source,
            scale=TROPOMI_SCALE_M,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    regional = src.get(
        TROPOMI_CH4_BAND
    )

    bkg = bg.get(
        TROPOMI_CH4_BAND
    )

    unc = src.get(
        TROPOMI_UNC_BAND
    )

    delta = None

    if (
        regional is not None
        and
        bkg is not None
    ):

        delta = (
            float(regional)
            -
            float(bkg)
        )

    return {
        "count":
            count,

        "regional":
            regional,

        "background":
            bkg,

        "delta":
            delta,

        "uncertainty":
            unc,

        "valid":
            int(
                cnt.get(
                    TROPOMI_CH4_BAND
                )
                or 0
            ),
    }


def tropomi_one(
    row,
):

    d = pd.Timestamp(
        row[
            "Actual S2 Date"
        ]
    )

    point = ee.Geometry.Point([
        float(
            row[
                "Longitude"
            ]
        ),
        float(
            row[
                "Latitude"
            ]
        ),
    ])

    exact_collection = (
        ee.ImageCollection(
            TROPOMI_COLLECTION
        )
        .filterBounds(point)
        .filterDate(
            d.strftime(
                "%Y-%m-%d"
            ),
            (
                d
                +
                pd.Timedelta(
                    days=1
                )
            ).strftime(
                "%Y-%m-%d"
            ),
        )
    )

    context_collection = (
        ee.ImageCollection(
            TROPOMI_COLLECTION
        )
        .filterBounds(point)
        .filterDate(
            (
                d
                -
                pd.Timedelta(
                    days=1
                )
            ).strftime(
                "%Y-%m-%d"
            ),
            (
                d
                +
                pd.Timedelta(
                    days=2
                )
            ).strftime(
                "%Y-%m-%d"
            ),
        )
    )

    a = tropomi_stats(
        exact_collection,
        point,
    )

    b = tropomi_stats(
        context_collection,
        point,
    )

    return {
        "Actual Candidate ID":
            row[
                "Actual Candidate ID"
            ],

        "TROPOMI Exact Image Count":
            a["count"],

        "TROPOMI Exact Regional Median ppb":
            a["regional"],

        "TROPOMI Exact Background Median ppb":
            a["background"],

        "TROPOMI Exact Delta ppb":
            a["delta"],

        "TROPOMI Exact Uncertainty ppb":
            a["uncertainty"],

        "TROPOMI Exact Valid Pixels":
            a["valid"],

        "TROPOMI +/-1d Image Count":
            b["count"],

        "TROPOMI +/-1d Delta ppb":
            b["delta"],

        "TROPOMI +/-1d Valid Pixels":
            b["valid"],

        "TROPOMI Evidence Role":
            (
                "regional_support_only"
                if (
                    a["valid"] > 0
                    or b["valid"] > 0
                )
                else
                "no_valid_data"
            ),
    }


def tropomi_batch():

    rows = []

    with ThreadPoolExecutor(
        max_workers=TROPOMI_WORKERS
    ) as pool:

        futures = [
            pool.submit(
                tropomi_one,
                r,
            )
            for _, r in (
                selected_valid.iterrows()
            )
        ]

        done = 0

        for f in as_completed(
            futures
        ):

            try:

                rows.append(
                    f.result()
                )

            except Exception as e:

                print(
                    "[TROPOMI] error:",
                    repr(e)
                )

            done += 1

            if (
                done % 10 == 0
                or
                done
                ==
                len(
                    selected_valid
                )
            ):

                print(
                    f"[TROPOMI] "
                    f"{done}/"
                    f"{len(selected_valid)}"
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PHASE B — RUN FOUR BRANCHES TOGETHER
# ============================================================

print("\n" + "=" * 110)
print("PHASE B — PARALLEL VALIDATION")
print("=" * 110)

print(
    "Candidates entering validation:",
    len(selected_valid)
)

with ThreadPoolExecutor(
    max_workers=4
) as outer:

    futures = {
        outer.submit(
            metadata_branch,
            PROJECT,
            LOCAL_MAXDEPTH,
            LOCAL_FIND_TIMEOUT,
            "LOCAL",
        ):
            "LOCAL",

        outer.submit(
            metadata_branch,
            LAB_ROOT,
            LAB_MAXDEPTH,
            LAB_FIND_TIMEOUT,
            "LAB",
        ):
            "LAB",

        outer.submit(
            emit_batch
        ):
            "EMIT",

        outer.submit(
            tropomi_batch
        ):
            "TROPOMI",
    }

    results = {}

    for f in as_completed(
        futures
    ):

        name = futures[f]

        try:

            results[name] = (
                f.result()
            )

            print(
                f"[{name}] COMPLETE"
            )

        except Exception as e:

            print(
                f"[{name}] FAILED:",
                repr(e)
            )

            results[name] = {
                "error":
                    repr(e),

                "trace":
                    traceback.format_exc(),
            }


# ============================================================
# COMBINE LOCAL + LAB CONTEXT
# ============================================================

context_frames = []

for name in [
    "LOCAL",
    "LAB",
]:

    r = results.get(
        name,
        {}
    )

    if not isinstance(
        r,
        dict
    ):

        continue

    frame = r.get(
        "matches"
    )

    if (
        isinstance(
            frame,
            pd.DataFrame
        )
        and len(frame)
    ):

        context_frames.append(
            frame
        )


context = (
    pd.concat(
        context_frames,
        ignore_index=True,
    )
    if context_frames
    else pd.DataFrame()
)

CONTEXT_OUT = (
    OUTDIR
    / "05_local_lab_context_matches.csv"
)

context.to_csv(
    CONTEXT_OUT,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SUMMARY PER SELECTED CANDIDATE
# ============================================================

context_summary = []

for _, r in (
    selected_valid.iterrows()
):

    cid = r[
        "Actual Candidate ID"
    ]

    if len(context):

        sub = context[
            context[
                "Actual Candidate ID"
            ]
            ==
            cid
        ]

    else:

        sub = pd.DataFrame()

    same_day = 0

    if (
        len(sub)
        and
        "Same Candidate Day"
        in sub.columns
    ):

        same_day = int(
            sub[
                "Same Candidate Day"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    context_summary.append({
        "Actual Candidate ID":
            cid,

        "Local/Lab Context Match Count":
            len(sub),

        "Local/Lab Same-Day Match Count":
            same_day,
    })


context_summary = pd.DataFrame(
    context_summary
)


# ============================================================
# MERGE
# ============================================================

final = selected.copy()

if len(
    context_summary
):

    final = final.merge(
        context_summary,
        on="Actual Candidate ID",
        how="left",
        validate="one_to_one",
    )


emit_df = results.get(
    "EMIT"
)

if isinstance(
    emit_df,
    pd.DataFrame
):

    final = final.merge(
        emit_df,
        on="Actual Candidate ID",
        how="left",
        validate="one_to_one",
    )


trop_df = results.get(
    "TROPOMI"
)

if isinstance(
    trop_df,
    pd.DataFrame
):

    final = final.merge(
        trop_df,
        on="Actual Candidate ID",
        how="left",
        validate="one_to_one",
    )


# ============================================================
# CONSERVATIVE STATUS
# ============================================================

def classify(
    row,
):

    if (
        row[
            "Selection Status"
        ]
        !=
        "selected"
    ):

        return (
            "NO_USABLE_S2_IN_BIN"
        )

    same_day = row.get(
        "Local/Lab Same-Day Match Count",
        0,
    )

    try:

        if (
            pd.notna(
                same_day
            )
            and
            int(
                same_day
            )
            > 0
        ):

            return (
                "REVIEW_SAME_DAY_KNOWN_RECORD"
            )

    except Exception:

        pass

    if (
        row.get(
            "EMIT Coverage Status"
        )
        ==
        "exact_date"
    ):

        return (
            "EMIT_SIGNAL_QA_REQUIRED"
        )

    if (
        row.get(
            "TROPOMI Evidence Role"
        )
        ==
        "regional_support_only"
    ):

        return (
            "U_UNKNOWN_WITH_TROPOMI_CONTEXT"
        )

    return "U_UNKNOWN"


final[
    "Validation Status"
] = final.apply(
    classify,
    axis=1,
)


# ============================================================
# SAVE FINAL
# ============================================================

FINAL_CSV = (
    OUTDIR
    / "06_actual_s2_45day_multisource_audit.csv"
)

FINAL_XLSX = (
    OUTDIR
    / "06_actual_s2_45day_multisource_audit.xlsx"
)

final.to_csv(
    FINAL_CSV,
    index=False,
    encoding="utf-8-sig",
)


excel_final = final.copy()

for col in (
    excel_final.columns
):

    if isinstance(
        excel_final[
            col
        ].dtype,
        pd.DatetimeTZDtype,
    ):

        excel_final[
            col
        ] = (
            excel_final[
                col
            ]
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )


excel_final.to_excel(
    FINAL_XLSX,
    index=False,
    engine="openpyxl",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 110)
print("V3 FINAL SUMMARY")
print("=" * 110)

print("\nS2 selection:")

print(
    final[
        "Selection Status"
    ]
    .value_counts(
        dropna=False
    )
)

print("\nBY TEMPORAL BIN:")

print(
    final.groupby(
        "Temporal Bin"
    )[
        "Selection Status"
    ]
    .value_counts()
    .unstack(
        fill_value=0
    )
)


valid = final[
    final[
        "Selection Status"
    ]
    ==
    "selected"
]


if (
    "EMIT Coverage Status"
    in final.columns
):

    print(
        "\nEMIT among selected:"
    )

    print(
        valid[
            "EMIT Coverage Status"
        ]
        .value_counts(
            dropna=False
        )
    )


if (
    "TROPOMI Evidence Role"
    in final.columns
):

    print(
        "\nTROPOMI among selected:"
    )

    print(
        valid[
            "TROPOMI Evidence Role"
        ]
        .value_counts(
            dropna=False
        )
    )


print(
    "\nVALIDATION STATUS:"
)

print(
    final[
        "Validation Status"
    ]
    .value_counts(
        dropna=False
    )
)


print("\n" + "=" * 110)
print("OUTPUTS")
print("=" * 110)

print(
    "\nAll +1–45 day S2 scenes:"
)

print(
    SCENE_OUT
)

print(
    "\nUnique S2 overpasses:"
)

print(
    OVERPASS_OUT
)

print(
    "\nSelected early/middle/late:"
)

print(
    SELECTED_OUT
)

print(
    "\nLocal/Lab matches:"
)

print(
    CONTEXT_OUT
)

print(
    "\nFinal audit:"
)

print(
    FINAL_CSV
)

print(
    FINAL_XLSX
)

print(
    "\n✅ up to 3 actual temporal controls per parent"
)

print(
    "✅ early / middle / late temporal diversity"
)

print(
    "✅ Mac + Lab metadata branches run concurrently"
)

print(
    "✅ previous file inventories reused as cache"
)

print(
    "✅ EMIT + TROPOMI run concurrently"
)

print(
    "✅ no imagery downloaded"
)

print(
    "✅ no existing data modified/deleted"
)

