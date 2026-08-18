from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import re
import traceback
import math
import time

import numpy as np
import pandas as pd
import ee


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
    / "actual_s2_45day_parallel_v3"
    / "06_actual_s2_45day_multisource_audit.csv"
)

OUTDIR = (
    PROJECT
    / "candidate_negative_validation"
    / "methaneair_2025_parallel_v4"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EARTH ENGINE / METHANEAIR 2025
# ============================================================

EE_PROJECT = "methane-release-gee"

MAIR_L3 = (
    "projects/edf-methanesat-ee/"
    "assets/mair/L3concentration"
)

MAIR_L4 = (
    "projects/edf-methanesat-ee/"
    "assets/mair/L4point"
)


# ============================================================
# COVERAGE / PLUME RULES
# ============================================================

# Source-centered patch for proving actual L3 coverage.
PATCH_HALF_SIZE_M = 240

# L3 is ~10 m; 20 m is enough for coverage audit
# and reduces EE computation.
L3_QA_SCALE_M = 20

# Require most of the MethaneFuse-sized patch to have
# valid MethaneAIR XCH4.
MIN_L3_VALID_FRACTION = 0.80

# Search L4 plume detections within 5 km of source.
L4_NEARBY_RADIUS_M = 5000

# Context search only. Does NOT generate exact-day
# high-resolution negative evidence.
CONTEXT_DAYS = 1

MAX_RETRIES = 4

GEE_WORKERS = 3


# ============================================================
# LOCAL / LAB DISCOVERY
# ============================================================

LOCAL_MAXDEPTH = 8
LAB_MAXDEPTH = 6

LOCAL_TIMEOUT_SEC = 180
LAB_TIMEOUT_SEC = 300

FILE_PATTERNS = [
    "*methaneair*",
    "*methane_air*",
    "*mair*",
    "*pcannon*",
    "*flight*",
    "*retrieval*",
    "*coverage*",
]

CANDIDATE_FILE_DATE_TOLERANCE_DAYS = 1


# ============================================================
# LOAD V3 RESULTS
# ============================================================

print("=" * 110)
print("METHANEAIR 2025 COVERAGE-FIRST VALIDATION V4")
print("=" * 110)

if not INPUT.exists():
    raise FileNotFoundError(
        f"Missing V3 audit:\n{INPUT}"
    )

v3 = pd.read_csv(
    INPUT,
    low_memory=False,
)

if "Selection Status" not in v3.columns:
    raise RuntimeError(
        "Missing Selection Status in V3 input."
    )

candidates = v3[
    v3["Selection Status"]
    ==
    "selected"
].copy()

print(
    "\nSelected actual-S2 candidates:",
    len(candidates)
)

if len(candidates) != 10:

    print(
        "WARNING: expected 10 selected candidates "
        f"from previous run, found {len(candidates)}."
    )

required = [
    "Actual Candidate ID",
    "Pilot Parent Number",
    "Source Positive Record ID",
    "Site",
    "Latitude",
    "Longitude",
    "Parent Positive Date",
    "Actual S2 Date",
    "Actual S2 Datetime UTC",
    "Actual Offset Days",
    "Temporal Bin",
]

missing = [
    c
    for c in required
    if c not in candidates.columns
]

if missing:

    raise RuntimeError(
        "Missing required columns:\n"
        +
        "\n".join(
            f"  {x}"
            for x in missing
        )
    )

candidates["_s2_date"] = pd.to_datetime(
    candidates["Actual S2 Date"],
    errors="coerce",
)

if candidates["_s2_date"].isna().any():
    raise RuntimeError(
        "Some Actual S2 Date values are invalid."
    )


print(
    "\nCandidate dates:"
)

print(
    candidates[
        [
            "Actual Candidate ID",
            "Site",
            "Temporal Bin",
            "Actual Offset Days",
            "Actual S2 Date",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# EARTH ENGINE INITIALIZATION
# ============================================================

print(
    "\nInitializing Earth Engine..."
)

try:

    ee.Initialize(
        project=EE_PROJECT
    )

except Exception as e:

    print(
        "\nEarth Engine initialization failed."
    )

    print(
        "If authentication expired, run:"
    )

    print(
        "  earthengine authenticate"
    )

    print(
        "\nOriginal error:"
    )

    print(
        repr(e)
    )

    raise

print(
    "Earth Engine ready."
)


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(x):

    try:

        if x is None:
            return None

        return float(x)

    except Exception:

        return None


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

    if isinstance(
        x,
        bytes
    ):

        return x.decode(
            "utf-8",
            errors="replace",
        )

    return str(x)


def norm(s):

    return (
        str(s)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_col(
    columns,
    names,
):

    cmap = {
        norm(c): c
        for c in columns
    }

    # exact first
    for name in names:

        n = norm(name)

        if n in cmap:

            return cmap[n]

    # substring fallback
    for name in names:

        n = norm(name)

        for k, original in (
            cmap.items()
        ):

            if n in k:

                return original

    return None


# ============================================================
# METHANEAIR L3 COVERAGE AUDIT
# ============================================================

def l3_patch_stats(
    image,
    point,
):

    region = (
        point
        .buffer(
            PATCH_HALF_SIZE_M
        )
        .bounds()
    )

    xch4 = image.select(
        "XCH4"
    )

    # Valid observed L3 pixels.
    valid_result = (
        xch4.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=L3_QA_SCALE_M,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    # Requested patch denominator.
    requested_result = (
        ee.Image.constant(1)
        .rename("requested")
        .reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=L3_QA_SCALE_M,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    # XCH4 median is context only.
    median_result = (
        xch4.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=region,
            scale=L3_QA_SCALE_M,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    valid_px = float(
        valid_result.get("XCH4")
        or 0
    )

    requested_px = float(
        requested_result.get(
            "requested"
        )
        or 0
    )

    valid_fraction = (
        valid_px
        /
        requested_px
        if requested_px > 0
        else np.nan
    )

    xch4_median = (
        median_result.get(
            "XCH4"
        )
    )

    return {
        "valid_px":
            valid_px,

        "requested_px":
            requested_px,

        "valid_fraction":
            valid_fraction,

        "xch4_median_ppb":
            xch4_median,
    }


# ============================================================
# L4 SAME-FLIGHT PLUME AUDIT
# ============================================================

def query_l4_for_flight(
    flight_id,
    point,
):

    all_flight = (
        ee.FeatureCollection(
            MAIR_L4
        )
        .filter(
            ee.Filter.eq(
                "flight_id",
                flight_id,
            )
        )
    )

    flight_count = int(
        all_flight.size().getInfo()
    )

    nearby = (
        all_flight.filterBounds(
            point.buffer(
                L4_NEARBY_RADIUS_M
            )
        )
    )

    nearby_count = int(
        nearby.size().getInfo()
    )

    plume_ids = []
    fluxes = []
    plume_records = []

    if nearby_count > 0:

        info = (
            nearby
            .limit(100)
            .getInfo()
        )

        for feature in (
            info.get(
                "features",
                []
            )
        ):

            props = feature.get(
                "properties",
                {}
            )

            geometry = feature.get(
                "geometry"
            )

            plume_id = props.get(
                "plume_id"
            )

            flux = props.get(
                "flux"
            )

            plume_ids.append(
                str(plume_id)
            )

            if flux is not None:
                fluxes.append(
                    flux
                )

            plume_records.append({
                "plume_id":
                    plume_id,

                "flux":
                    flux,

                "flux_sd":
                    props.get(
                        "flux_sd"
                    ),

                "flight_id":
                    props.get(
                        "flight_id"
                    ),

                "basin":
                    props.get(
                        "basin"
                    ),

                "time_coverage_start":
                    props.get(
                        "time_coverage_start"
                    ),

                "time_coverage_end":
                    props.get(
                        "time_coverage_end"
                    ),

                "geometry":
                    geometry,
            })

    return {
        "flight_l4_count":
            flight_count,

        "nearby_l4_count":
            nearby_count,

        "nearby_plume_ids":
            plume_ids,

        "nearby_fluxes":
            fluxes,

        "nearby_records":
            plume_records,
    }


# ============================================================
# ONE ACTUAL-S2 CANDIDATE → METHANEAIR
# ============================================================

def methaneair_one(
    row,
):

    cid = row[
        "Actual Candidate ID"
    ]

    lat = float(
        row["Latitude"]
    )

    lon = float(
        row["Longitude"]
    )

    d = pd.Timestamp(
        row["_s2_date"]
    ).normalize()

    start = d.strftime(
        "%Y-%m-%d"
    )

    end = (
        d
        +
        pd.Timedelta(days=1)
    ).strftime(
        "%Y-%m-%d"
    )

    context_start = (
        d
        -
        pd.Timedelta(
            days=CONTEXT_DAYS
        )
    ).strftime(
        "%Y-%m-%d"
    )

    context_end = (
        d
        +
        pd.Timedelta(
            days=CONTEXT_DAYS + 1
        )
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

            # =================================================
            # EXACT CALENDAR DATE L3 COVERAGE
            # =================================================

            exact_ic = (
                ee.ImageCollection(
                    MAIR_L3
                )
                .filterDate(
                    start,
                    end,
                )
                .filterBounds(
                    point
                )
            )

            exact_n = int(
                exact_ic.size().getInfo()
            )


            # =================================================
            # +/-1 DAY CONTEXT COVERAGE
            # =================================================

            context_ic = (
                ee.ImageCollection(
                    MAIR_L3
                )
                .filterDate(
                    context_start,
                    context_end,
                )
                .filterBounds(
                    point
                )
            )

            context_n = int(
                context_ic.size().getInfo()
            )


            exact_records = []

            valid_exact_records = []

            all_nearby_plumes = []

            for i in range(
                exact_n
            ):

                image = ee.Image(
                    exact_ic
                    .toList(
                        exact_n
                    )
                    .get(i)
                )

                props = (
                    image
                    .toDictionary([
                        "system:index",
                        "system:time_start",
                        "flight_id",
                        "target_id",
                        "time_coverage_start",
                        "time_coverage_end",
                        "processing_id",
                    ])
                    .getInfo()
                )

                flight_id = props.get(
                    "flight_id"
                )

                stats = (
                    l3_patch_stats(
                        image,
                        point,
                    )
                )

                valid_fraction = (
                    stats[
                        "valid_fraction"
                    ]
                )

                l3_valid = (
                    not pd.isna(
                        valid_fraction
                    )
                    and
                    valid_fraction
                    >=
                    MIN_L3_VALID_FRACTION
                )

                l4 = {
                    "flight_l4_count":
                        None,

                    "nearby_l4_count":
                        None,

                    "nearby_plume_ids":
                        [],

                    "nearby_fluxes":
                        [],

                    "nearby_records":
                        [],
                }

                if flight_id:

                    l4 = (
                        query_l4_for_flight(
                            flight_id,
                            point,
                        )
                    )

                record = {
                    "flight_id":
                        flight_id,

                    "target_id":
                        props.get(
                            "target_id"
                        ),

                    "system_index":
                        props.get(
                            "system:index"
                        ),

                    "time_coverage_start":
                        props.get(
                            "time_coverage_start"
                        ),

                    "time_coverage_end":
                        props.get(
                            "time_coverage_end"
                        ),

                    "processing_id":
                        props.get(
                            "processing_id"
                        ),

                    "l3_valid_fraction":
                        valid_fraction,

                    "l3_xch4_median_ppb":
                        stats[
                            "xch4_median_ppb"
                        ],

                    "l3_valid_coverage":
                        l3_valid,

                    "flight_l4_count":
                        l4[
                            "flight_l4_count"
                        ],

                    "nearby_l4_count":
                        l4[
                            "nearby_l4_count"
                        ],

                    "nearby_plume_ids":
                        l4[
                            "nearby_plume_ids"
                        ],

                    "nearby_fluxes":
                        l4[
                            "nearby_fluxes"
                        ],
                }

                exact_records.append(
                    record
                )

                if l3_valid:

                    valid_exact_records.append(
                        record
                    )

                if l4[
                    "nearby_records"
                ]:

                    for plume in (
                        l4[
                            "nearby_records"
                        ]
                    ):

                        plume_row = (
                            plume.copy()
                        )

                        plume_row[
                            "Actual Candidate ID"
                        ] = cid

                        plume_row[
                            "Candidate Site"
                        ] = row[
                            "Site"
                        ]

                        all_nearby_plumes.append(
                            plume_row
                        )


            # =================================================
            # CONSERVATIVE CLASSIFICATION
            # =================================================

            classification = (
                "NO_METHANEAIR_EXACT_COVERAGE"
            )

            evidence_grade = "U"

            reason = (
                "No exact-date MethaneAIR L3 "
                "image intersects source."
            )

            if exact_n > 0:

                if not valid_exact_records:

                    classification = (
                        "METHANEAIR_L3_PARTIAL_OR_"
                        "INVALID_COVERAGE"
                    )

                    reason = (
                        "MethaneAIR L3 intersects point, "
                        "but no exact-date image has "
                        f">={MIN_L3_VALID_FRACTION:.2f} "
                        "valid coverage over 480 m patch."
                    )

                else:

                    nearby_total = sum(
                        int(
                            x[
                                "nearby_l4_count"
                            ]
                            or 0
                        )
                        for x in valid_exact_records
                    )

                    flight_with_l4 = [
                        x
                        for x in valid_exact_records
                        if (
                            x[
                                "flight_l4_count"
                            ]
                            is not None
                            and
                            int(
                                x[
                                    "flight_l4_count"
                                ]
                            )
                            > 0
                        )
                    ]

                    if nearby_total > 0:

                        classification = (
                            "R_REJECT_METHANEAIR_"
                            "PLUME_NEARBY"
                        )

                        evidence_grade = "R"

                        reason = (
                            "Exact-date valid MethaneAIR "
                            "L3 coverage exists and same-flight "
                            "L4 plume detection is present "
                            f"within {L4_NEARBY_RADIUS_M/1000:.1f} km."
                        )

                    elif flight_with_l4:

                        classification = (
                            "B_HIGH_RES_NO_DETECTION_"
                            "CANDIDATE"
                        )

                        evidence_grade = "B"

                        reason = (
                            "Exact-date valid MethaneAIR L3 "
                            "coverage exists. The same flight "
                            "has L4 point-source detections "
                            "elsewhere, but none within "
                            f"{L4_NEARBY_RADIUS_M/1000:.1f} km "
                            "of candidate. Treat as "
                            "high-resolution no-detection, "
                            "not confirmed zero-emission."
                        )

                    else:

                        classification = (
                            "METHANEAIR_L3_COVERAGE_"
                            "L4_AVAILABILITY_UNCERTAIN"
                        )

                        evidence_grade = "U"

                        reason = (
                            "Exact-date valid MethaneAIR L3 "
                            "coverage exists, but no L4 features "
                            "were found for the same flight. "
                            "Because not all MethaneAIR products "
                            "are available for every flight, "
                            "absence of L4 cannot be interpreted "
                            "as no detection."
                        )


            max_valid_fraction = None

            if exact_records:

                vals = [
                    x[
                        "l3_valid_fraction"
                    ]
                    for x in exact_records
                    if (
                        x[
                            "l3_valid_fraction"
                        ]
                        is not None
                        and
                        not pd.isna(
                            x[
                                "l3_valid_fraction"
                            ]
                        )
                    )
                ]

                if vals:

                    max_valid_fraction = (
                        max(vals)
                    )


            flight_ids = sorted(
                set(
                    str(
                        x[
                            "flight_id"
                        ]
                    )
                    for x in exact_records
                    if x[
                        "flight_id"
                    ]
                )
            )


            l4_flight_counts = [
                x[
                    "flight_l4_count"
                ]
                for x in exact_records
                if (
                    x[
                        "flight_l4_count"
                    ]
                    is not None
                )
            ]


            nearby_count_total = sum(
                int(
                    x[
                        "nearby_l4_count"
                    ]
                    or 0
                )
                for x in exact_records
            )


            plume_ids = []

            for x in exact_records:

                plume_ids.extend(
                    x[
                        "nearby_plume_ids"
                    ]
                )


            return {
                "summary": {
                    "Actual Candidate ID":
                        cid,

                    "Site":
                        row[
                            "Site"
                        ],

                    "Actual S2 Date":
                        row[
                            "Actual S2 Date"
                        ],

                    "MethaneAIR Exact L3 Image Count":
                        exact_n,

                    "MethaneAIR +/-1d L3 Image Count":
                        context_n,

                    "MethaneAIR Valid Exact Flight Count":
                        len(
                            valid_exact_records
                        ),

                    "MethaneAIR Max L3 Valid Fraction":
                        max_valid_fraction,

                    "MethaneAIR Exact Flight IDs":
                        " | ".join(
                            flight_ids
                        ),

                    "MethaneAIR Same-Flight L4 Feature Counts":
                        " | ".join(
                            str(x)
                            for x in l4_flight_counts
                        ),

                    "MethaneAIR Nearby L4 Plume Count":
                        nearby_count_total,

                    "MethaneAIR Nearby L4 Plume IDs":
                        " | ".join(
                            sorted(
                                set(
                                    plume_ids
                                )
                            )
                        ),

                    "MethaneAIR Classification":
                        classification,

                    "MethaneAIR Evidence Grade":
                        evidence_grade,

                    "MethaneAIR Reason":
                        reason,

                    "MethaneAIR Query Error":
                        "",
                },

                "flight_records":
                    exact_records,

                "plumes":
                    all_nearby_plumes,
            }

        except Exception as e:

            last_error = e

            time.sleep(
                2 * attempt
            )


    return {
        "summary": {
            "Actual Candidate ID":
                cid,

            "Site":
                row[
                    "Site"
                ],

            "Actual S2 Date":
                row[
                    "Actual S2 Date"
                ],

            "MethaneAIR Exact L3 Image Count":
                pd.NA,

            "MethaneAIR +/-1d L3 Image Count":
                pd.NA,

            "MethaneAIR Valid Exact Flight Count":
                pd.NA,

            "MethaneAIR Max L3 Valid Fraction":
                pd.NA,

            "MethaneAIR Exact Flight IDs":
                "",

            "MethaneAIR Same-Flight L4 Feature Counts":
                "",

            "MethaneAIR Nearby L4 Plume Count":
                pd.NA,

            "MethaneAIR Nearby L4 Plume IDs":
                "",

            "MethaneAIR Classification":
                "QUERY_ERROR",

            "MethaneAIR Evidence Grade":
                "U",

            "MethaneAIR Reason":
                "Earth Engine query failed.",

            "MethaneAIR Query Error":
                repr(
                    last_error
                ),
        },

        "flight_records":
            [],

        "plumes":
            [],
    }


# ============================================================
# GEE METHANEAIR BATCH
# ============================================================

def run_methaneair_gee():

    print(
        "\n[METHANEAIR-GEE] "
        "starting 2025 L3/L4 coverage audit..."
    )

    summaries = []
    flight_rows = []
    plume_rows = []

    with ThreadPoolExecutor(
        max_workers=GEE_WORKERS
    ) as pool:

        futures = {
            pool.submit(
                methaneair_one,
                row,
            ):
                row[
                    "Actual Candidate ID"
                ]

            for _, row in (
                candidates.iterrows()
            )
        }

        done = 0

        for f in as_completed(
            futures
        ):

            cid = futures[f]

            try:

                result = f.result()

                summaries.append(
                    result[
                        "summary"
                    ]
                )

                for r in result[
                    "flight_records"
                ]:

                    rr = r.copy()

                    rr[
                        "Actual Candidate ID"
                    ] = cid

                    flight_rows.append(
                        rr
                    )

                plume_rows.extend(
                    result[
                        "plumes"
                    ]
                )

            except Exception as e:

                summaries.append({
                    "Actual Candidate ID":
                        cid,

                    "MethaneAIR Classification":
                        "BRANCH_ERROR",

                    "MethaneAIR Evidence Grade":
                        "U",

                    "MethaneAIR Query Error":
                        repr(e),
                })

            done += 1

            print(
                f"[METHANEAIR-GEE] "
                f"{done}/{len(candidates)}"
            )

    return {
        "summary":
            pd.DataFrame(
                summaries
            ),

        "flights":
            pd.DataFrame(
                flight_rows
            ),

        "plumes":
            pd.DataFrame(
                plume_rows
            ),
    }


# ============================================================
# LOCAL / LAB FILE INVENTORY CACHE
# ============================================================

def existing_cache_paths(
    label,
):

    paths = []

    cache_candidates = []

    v3_root = (
        PROJECT
        / "candidate_negative_validation"
        / "actual_s2_45day_parallel_v3"
    )

    old_root = (
        PROJECT
        / "candidate_negative_validation"
        / "parallel_multisource_40"
    )

    if label == "LOCAL":

        cache_candidates.extend([
            v3_root
            / "04_local_metadata_inventory.txt",

            old_root
            / "local_existing_sensor_files.txt",
        ])

    else:

        cache_candidates.extend([
            v3_root
            / "04_lab_metadata_inventory.txt",

            old_root
            / "lab_existing_sensor_files.txt",
        ])


    seen = set()

    for cache in (
        cache_candidates
    ):

        if not cache.exists():

            continue

        for line in cache.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines():

            line = line.strip()

            if (
                not line
                or
                line.startswith(
                    "STATUS:"
                )
                or
                line.startswith(
                    "COUNT:"
                )
            ):

                continue

            if line in seen:

                continue

            seen.add(line)

            paths.append(
                Path(line)
            )

    return paths


# ============================================================
# FILE DISCOVERY
# ============================================================

def discover_files(
    root,
    maxdepth,
    timeout_sec,
    label,
):

    cached = (
        existing_cache_paths(
            label
        )
    )

    if (
        label == "LAB"
        and
        not mounted_lab()
    ):

        return {
            "status":
                "smb_not_mounted_cache_only",

            "paths":
                cached,
        }

    root = Path(
        root
    )

    if not root.exists():

        return {
            "status":
                "root_missing_cache_only",

            "paths":
                cached,
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
        FILE_PATTERNS
    ):

        if i > 0:

            cmd.append(
                "-o"
            )

        cmd.extend([
            "-iname",
            pattern,
        ])

    cmd.extend([
        ")",
        "-print",
    ])


    output = ""

    status = "complete"

    try:

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        output = (
            result.stdout
            or ""
        )

        if result.returncode != 0:

            status = (
                "returncode_"
                +
                str(
                    result.returncode
                )
            )

    except subprocess.TimeoutExpired as e:

        status = (
            "timeout_partial"
        )

        output = (
            decode_output(
                e.stdout
            )
        )

    except Exception as e:

        status = (
            "error:"
            +
            repr(e)
        )


    fresh = []

    for line in (
        output.splitlines()
    ):

        line = (
            line.strip()
        )

        if line:

            fresh.append(
                Path(line)
            )


    combined = []

    seen = set()

    for p in (
        cached
        +
        fresh
    ):

        key = str(p)

        if key in seen:

            continue

        seen.add(
            key
        )

        combined.append(
            p
        )


    return {
        "status":
            status,

        "paths":
            combined,
    }


# ============================================================
# FILE DATE PARSING
# ============================================================

DATE_PATTERNS = [
    re.compile(
        r"(20\d{2})(\d{2})(\d{2})"
    ),

    re.compile(
        r"(20\d{2})[-_](\d{2})[-_](\d{2})"
    ),
]


def filename_dates(
    path,
):

    text = str(
        path
    )

    dates = []

    for pattern in (
        DATE_PATTERNS
    ):

        for match in (
            pattern.finditer(
                text
            )
        ):

            try:

                d = pd.Timestamp(
                    year=int(
                        match.group(1)
                    ),
                    month=int(
                        match.group(2)
                    ),
                    day=int(
                        match.group(3)
                    ),
                )

                dates.append(
                    d
                )

            except Exception:

                pass

    return sorted(
        set(
            dates
        )
    )


# ============================================================
# CONTENT-MATCH CSV METADATA
# ============================================================

def csv_candidate_matches(
    file_path,
    label,
):

    try:

        df = pd.read_csv(
            file_path,
            low_memory=False,
        )

    except Exception:

        return []


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

    time_col = find_col(
        df.columns,
        [
            "time_coverage_start",
            "datetime",
            "timestamp",
            "observation_time",
            "acquisition_time",
            "date",
        ]
    )

    if (
        lat_col is None
        or
        lon_col is None
        or
        time_col is None
    ):

        return []


    lats = pd.to_numeric(
        df[
            lat_col
        ],
        errors="coerce",
    )

    lons = pd.to_numeric(
        df[
            lon_col
        ],
        errors="coerce",
    )

    times = pd.to_datetime(
        df[
            time_col
        ],
        errors="coerce",
        utc=True,
    )

    valid = (
        lats.notna()
        &
        lons.notna()
        &
        times.notna()
    )

    if not valid.any():

        return []


    tmp = pd.DataFrame({
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


    records = []

    for _, cand in (
        candidates.iterrows()
    ):

        candidate_date = (
            pd.to_datetime(
                cand[
                    "Actual S2 Date"
                ],
                utc=True,
            )
        )

        time_diff = (
            tmp[
                "_time"
            ]
            -
            candidate_date
        ).dt.total_seconds() / 86400


        temporal = (
            time_diff.abs()
            <=
            CANDIDATE_FILE_DATE_TOLERANCE_DAYS
        )

        if not temporal.any():

            continue


        sub = tmp.loc[
            temporal
        ].copy()

        sub[
            "_time_diff"
        ] = (
            time_diff.loc[
                temporal
            ]
        )


        # vectorized Haversine
        R = 6371.0088

        lat0 = math.radians(
            float(
                cand[
                    "Latitude"
                ]
            )
        )

        lon0 = math.radians(
            float(
                cand[
                    "Longitude"
                ]
            )
        )

        lat_arr = np.radians(
            sub[
                "_lat"
            ].astype(
                float
            )
        )

        lon_arr = np.radians(
            sub[
                "_lon"
            ].astype(
                float
            )
        )

        dlat = (
            lat_arr
            -
            lat0
        )

        dlon = (
            lon_arr
            -
            lon0
        )

        a = (
            np.sin(
                dlat / 2
            ) ** 2
            +
            math.cos(
                lat0
            )
            *
            np.cos(
                lat_arr
            )
            *
            np.sin(
                dlon / 2
            ) ** 2
        )

        sub[
            "_distance_km"
        ] = (
            2
            *
            R
            *
            np.arcsin(
                np.sqrt(a)
            )
        )


        sub = sub[
            sub[
                "_distance_km"
            ]
            <= 10
        ]


        for _, hit in (
            sub.iterrows()
        ):

            records.append({
                "Actual Candidate ID":
                    cand[
                        "Actual Candidate ID"
                    ],

                "Origin":
                    label,

                "Match Type":
                    "CSV_CONTENT",

                "File":
                    str(
                        file_path
                    ),

                "Source Row":
                    int(
                        hit[
                            "_row"
                        ]
                    ),

                "Source Datetime UTC":
                    str(
                        hit[
                            "_time"
                        ]
                    ),

                "Distance km":
                    float(
                        hit[
                            "_distance_km"
                        ]
                    ),

                "Time Difference days":
                    float(
                        hit[
                            "_time_diff"
                        ]
                    ),
            })


    return records


# ============================================================
# LOCAL/LAB BRANCH
# ============================================================

def run_file_branch(
    root,
    maxdepth,
    timeout_sec,
    label,
):

    print(
        f"\n[{label}] "
        "starting MethaneAIR metadata/file audit..."
    )

    discovery = (
        discover_files(
            root,
            maxdepth,
            timeout_sec,
            label,
        )
    )

    print(
        f"[{label}] "
        f"status={discovery['status']} | "
        f"paths={len(discovery['paths'])}"
    )


    inventory_path = (
        OUTDIR
        /
        f"{label.lower()}_methaneair_file_inventory.txt"
    )

    with open(
        inventory_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "STATUS: "
            +
            str(
                discovery[
                    "status"
                ]
            )
            +
            "\n"
        )

        f.write(
            "COUNT: "
            +
            str(
                len(
                    discovery[
                        "paths"
                    ]
                )
            )
            +
            "\n\n"
        )

        for p in (
            discovery[
                "paths"
            ]
        ):

            f.write(
                str(p)
                +
                "\n"
            )


    filename_matches = []

    content_matches = []


    for p in (
        discovery[
            "paths"
        ]
    ):

        # ----------------------------------------------------
        # Filename/date-level matching
        # ----------------------------------------------------

        pdates = (
            filename_dates(
                p
            )
        )

        if pdates:

            for _, cand in (
                candidates.iterrows()
            ):

                cd = pd.Timestamp(
                    cand[
                        "_s2_date"
                    ]
                ).normalize()

                for fd in pdates:

                    diff = int(
                        (
                            fd
                            -
                            cd
                        ).days
                    )

                    if (
                        abs(diff)
                        <=
                        CANDIDATE_FILE_DATE_TOLERANCE_DAYS
                    ):

                        filename_matches.append({
                            "Actual Candidate ID":
                                cand[
                                    "Actual Candidate ID"
                                ],

                            "Origin":
                                label,

                            "Match Type":
                                "FILENAME_DATE",

                            "File":
                                str(p),

                            "File Date":
                                fd.strftime(
                                    "%Y-%m-%d"
                                ),

                            "Date Difference days":
                                diff,
                        })


        # ----------------------------------------------------
        # CSV content-level matching
        # ----------------------------------------------------

        if (
            p.suffix.lower()
            ==
            ".csv"
        ):

            content_matches.extend(
                csv_candidate_matches(
                    p,
                    label,
                )
            )


    return {
        "status":
            discovery[
                "status"
            ],

        "inventory":
            discovery[
                "paths"
            ],

        "filename_matches":
            pd.DataFrame(
                filename_matches
            ),

        "content_matches":
            pd.DataFrame(
                content_matches
            ),
    }


# ============================================================
# RUN THREE BRANCHES IN PARALLEL
# ============================================================

print(
    "\n"
    +
    "=" * 110
)

print(
    "STARTING PARALLEL METHANEAIR VALIDATION"
)

print(
    "=" * 110
)

print(
    "1. 2025 MethaneAIR GEE L3/L4"
)

print(
    "2. Mac local files/metadata"
)

print(
    "3. Lab SMB files/metadata"
)


with ThreadPoolExecutor(
    max_workers=3
) as outer:

    futures = {
        outer.submit(
            run_methaneair_gee
        ):
            "GEE",

        outer.submit(
            run_file_branch,
            PROJECT,
            LOCAL_MAXDEPTH,
            LOCAL_TIMEOUT_SEC,
            "LOCAL",
        ):
            "LOCAL",

        outer.submit(
            run_file_branch,
            LAB_ROOT,
            LAB_MAXDEPTH,
            LAB_TIMEOUT_SEC,
            "LAB",
        ):
            "LAB",
    }


    results = {}


    for f in as_completed(
        futures
    ):

        name = (
            futures[f]
        )

        try:

            results[
                name
            ] = (
                f.result()
            )

            print(
                f"\n[{name}] COMPLETE"
            )

        except Exception as e:

            print(
                f"\n[{name}] FAILED:",
                repr(e)
            )

            results[
                name
            ] = {
                "branch_error":
                    repr(e),

                "traceback":
                    traceback.format_exc(),
            }


# ============================================================
# SAVE GEE DETAILS
# ============================================================

gee_result = results.get(
    "GEE",
    {}
)

gee_summary = (
    gee_result.get(
        "summary"
    )
    if isinstance(
        gee_result,
        dict
    )
    else None
)


if not isinstance(
    gee_summary,
    pd.DataFrame
):

    gee_summary = pd.DataFrame()


gee_flights = (
    gee_result.get(
        "flights"
    )
    if isinstance(
        gee_result,
        dict
    )
    else None
)

if not isinstance(
    gee_flights,
    pd.DataFrame
):

    gee_flights = pd.DataFrame()


gee_plumes = (
    gee_result.get(
        "plumes"
    )
    if isinstance(
        gee_result,
        dict
    )
    else None
)

if not isinstance(
    gee_plumes,
    pd.DataFrame
):

    gee_plumes = pd.DataFrame()


GEE_SUMMARY_OUT = (
    OUTDIR
    /
    "01_methaneair_gee_candidate_summary.csv"
)

GEE_FLIGHTS_OUT = (
    OUTDIR
    /
    "02_methaneair_gee_flight_details.csv"
)

GEE_PLUMES_OUT = (
    OUTDIR
    /
    "03_methaneair_gee_nearby_l4_plumes.csv"
)


gee_summary.to_csv(
    GEE_SUMMARY_OUT,
    index=False,
    encoding="utf-8-sig",
)

gee_flights.to_csv(
    GEE_FLIGHTS_OUT,
    index=False,
    encoding="utf-8-sig",
)

gee_plumes.to_csv(
    GEE_PLUMES_OUT,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE LOCAL/LAB MATCHES
# ============================================================

file_match_frames = []


for label in [
    "LOCAL",
    "LAB",
]:

    result = results.get(
        label,
        {}
    )

    if not isinstance(
        result,
        dict
    ):

        continue


    for key in [
        "filename_matches",
        "content_matches",
    ]:

        frame = result.get(
            key
        )

        if (
            isinstance(
                frame,
                pd.DataFrame
            )
            and
            len(frame)
        ):

            file_match_frames.append(
                frame
            )


if file_match_frames:

    file_matches = pd.concat(
        file_match_frames,
        ignore_index=True,
        sort=False,
    )

else:

    file_matches = pd.DataFrame()


FILE_MATCH_OUT = (
    OUTDIR
    /
    "04_local_lab_candidate_file_matches.csv"
)

file_matches.to_csv(
    FILE_MATCH_OUT,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# ONE-ROW LOCAL/LAB SUMMARY
# ============================================================

local_lab_summary = []


for _, cand in (
    candidates.iterrows()
):

    cid = cand[
        "Actual Candidate ID"
    ]

    if len(
        file_matches
    ):

        sub = file_matches[
            file_matches[
                "Actual Candidate ID"
            ]
            ==
            cid
        ]

    else:

        sub = pd.DataFrame()


    local_n = 0
    lab_n = 0


    if len(sub):

        if "Origin" in sub.columns:

            local_n = int(
                sub[
                    "Origin"
                ]
                .eq(
                    "LOCAL"
                )
                .sum()
            )

            lab_n = int(
                sub[
                    "Origin"
                ]
                .eq(
                    "LAB"
                )
                .sum()
            )


    local_lab_summary.append({
        "Actual Candidate ID":
            cid,

        "Local MethaneAIR File/Metadata Matches":
            local_n,

        "Lab MethaneAIR File/Metadata Matches":
            lab_n,
    })


local_lab_summary = pd.DataFrame(
    local_lab_summary
)


# ============================================================
# FINAL MERGE
# ============================================================

final = (
    candidates
    .drop(
        columns=[
            "_s2_date"
        ],
        errors="ignore",
    )
    .copy()
)


if len(
    gee_summary
):

    merge_cols = [
        c
        for c in (
            gee_summary.columns
        )
        if (
            c
            ==
            "Actual Candidate ID"
            or
            c.startswith(
                "MethaneAIR"
            )
        )
    ]

    final = final.merge(
        gee_summary[
            merge_cols
        ],
        on="Actual Candidate ID",
        how="left",
        validate="one_to_one",
    )


final = final.merge(
    local_lab_summary,
    on="Actual Candidate ID",
    how="left",
    validate="one_to_one",
)


# ============================================================
# FINAL HIGH-RES STATUS
# ============================================================

def final_status(
    row,
):

    mair = row.get(
        "MethaneAIR Classification"
    )

    if (
        mair
        ==
        "R_REJECT_METHANEAIR_PLUME_NEARBY"
    ):

        return (
            "R_REJECT"
        )


    if (
        mair
        ==
        "B_HIGH_RES_NO_DETECTION_CANDIDATE"
    ):

        return (
            "B_HIGH_RES_NO_DETECTION_CANDIDATE"
        )


    if (
        mair
        ==
        "METHANEAIR_L3_COVERAGE_"
        "L4_AVAILABILITY_UNCERTAIN"
    ):

        return (
            "U_HIGH_RES_COVERAGE_"
            "BUT_L4_UNCERTAIN"
        )


    if (
        mair
        ==
        "METHANEAIR_L3_PARTIAL_OR_"
        "INVALID_COVERAGE"
    ):

        return (
            "U_PARTIAL_METHANEAIR_COVERAGE"
        )


    return (
        "U_NO_EXACT_HIGH_RES_METHANE_EVIDENCE"
    )


final[
    "High Resolution Validation Status"
] = final.apply(
    final_status,
    axis=1,
)


# ============================================================
# SAVE FINAL CSV/XLSX
# ============================================================

FINAL_CSV = (
    OUTDIR
    /
    "05_methaneair_2025_highres_validation.csv"
)

FINAL_XLSX = (
    OUTDIR
    /
    "05_methaneair_2025_highres_validation.xlsx"
)


final.to_csv(
    FINAL_CSV,
    index=False,
    encoding="utf-8-sig",
)


excel_final = (
    final.copy()
)


for col in (
    excel_final.columns
):

    if isinstance(
        excel_final[
            col
        ].dtype,
        pd.DatetimeTZDtype
    ):

        excel_final[
            col
        ] = (
            excel_final[
                col
            ]
            .dt.tz_convert(
                "UTC"
            )
            .dt.tz_localize(
                None
            )
        )


excel_final.to_excel(
    FINAL_XLSX,
    index=False,
    engine="openpyxl",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n"
    +
    "=" * 110
)

print(
    "METHANEAIR 2025 V4 SUMMARY"
)

print(
    "=" * 110
)


if (
    "MethaneAIR Classification"
    in final.columns
):

    print(
        "\nMethaneAIR classification:"
    )

    print(
        final[
            "MethaneAIR Classification"
        ]
        .value_counts(
            dropna=False
        )
    )


print(
    "\nHigh-resolution validation status:"
)

print(
    final[
        "High Resolution Validation Status"
    ]
    .value_counts(
        dropna=False
    )
)


if (
    "MethaneAIR Exact L3 Image Count"
    in final.columns
):

    exact_coverage = int(
        (
            pd.to_numeric(
                final[
                    "MethaneAIR Exact L3 Image Count"
                ],
                errors="coerce",
            )
            >
            0
        ).sum()
    )

    print(
        "\nCandidates with exact-date "
        "MethaneAIR L3 intersection:",
        f"{exact_coverage}/{len(final)}"
    )


if (
    "MethaneAIR Valid Exact Flight Count"
    in final.columns
):

    valid_coverage = int(
        (
            pd.to_numeric(
                final[
                    "MethaneAIR Valid Exact Flight Count"
                ],
                errors="coerce",
            )
            >
            0
        ).sum()
    )

    print(
        "Candidates with >=80% valid "
        "MethaneAIR L3 coverage:",
        f"{valid_coverage}/{len(final)}"
    )


print(
    "\nLocal file matches:"
)

print(
    final[
        "Local MethaneAIR File/Metadata Matches"
    ]
    .value_counts(
        dropna=False
    )
)


print(
    "\nLab file matches:"
)

print(
    final[
        "Lab MethaneAIR File/Metadata Matches"
    ]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# DISPLAY KEY ROWS
# ============================================================

display_cols = [
    "Actual Candidate ID",
    "Site",
    "Temporal Bin",
    "Actual Offset Days",
    "Actual S2 Date",

    "MethaneAIR Exact L3 Image Count",
    "MethaneAIR Valid Exact Flight Count",
    "MethaneAIR Max L3 Valid Fraction",
    "MethaneAIR Exact Flight IDs",
    "MethaneAIR Same-Flight L4 Feature Counts",
    "MethaneAIR Nearby L4 Plume Count",

    "MethaneAIR Classification",
    "High Resolution Validation Status",

    "Local MethaneAIR File/Metadata Matches",
    "Lab MethaneAIR File/Metadata Matches",
]

display_cols = [
    c
    for c in display_cols
    if c in final.columns
]


print(
    "\n"
    +
    "=" * 110
)

print(
    "CANDIDATE DETAILS"
)

print(
    "=" * 110
)


print(
    final[
        display_cols
    ].to_string(
        index=False
    )
)


print(
    "\n"
    +
    "=" * 110
)

print(
    "OUTPUTS"
)

print(
    "=" * 110
)


print(
    "\n1. GEE candidate summary:"
)

print(
    GEE_SUMMARY_OUT
)


print(
    "\n2. GEE flight details:"
)

print(
    GEE_FLIGHTS_OUT
)


print(
    "\n3. Nearby MethaneAIR L4 plumes:"
)

print(
    GEE_PLUMES_OUT
)


print(
    "\n4. Mac/Lab file matches:"
)

print(
    FILE_MATCH_OUT
)


print(
    "\n5. Final high-resolution audit:"
)

print(
    FINAL_CSV
)

print(
    FINAL_XLSX
)


print(
    "\n✅ 2025 MethaneAIR L3 coverage queried directly"
)

print(
    "✅ Same-flight L4 point-source detections queried"
)

print(
    "✅ Mac local file audit ran independently"
)

print(
    "✅ Lab SMB file audit ran independently"
)

print(
    "✅ Existing inventories reused as cache"
)

print(
    "✅ No MethaneAIR imagery downloaded"
)

print(
    "✅ No local/Lab files modified or deleted"
)

