from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import itertools
import math
import time
import traceback

import numpy as np
import pandas as pd
import requests
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
    / "pilot_10_positive_40_candidates_s2qa.csv"
)

OUTDIR = (
    PROJECT
    / "candidate_negative_validation"
    / "actual_s2_parallel_v2"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Target temporal anchors
# ------------------------------------------------------------

TARGET_OFFSETS = [
    1,
    3,
    7,
    14,
]

# Search all actual Sentinel-2 acquisitions from
# positive +1 through positive +17 days.
SEARCH_START_DAY = 1
SEARCH_END_DAY = 17

# An actual S2 acquisition can be assigned to an anchor
# if it is within +/-3 calendar days.
ANCHOR_TOLERANCE_DAYS = 3


# ------------------------------------------------------------
# Sentinel-2
# ------------------------------------------------------------

EE_PROJECT = "methane-release-gee"

S2_COLLECTION = (
    "COPERNICUS/S2_SR_HARMONIZED"
)

HALF_PATCH_M = 240
S2_SCALE_M = 20

MIN_CLEAR_REQUESTED = 0.80

S2_WORKERS = 4


# ------------------------------------------------------------
# MethaneAIR/local plume context
# ------------------------------------------------------------

METHANEAIR_RADIUS_KM = 10.0
METHANEAIR_CONTEXT_DAYS = 14

LOCAL_MAXDEPTH = 7
LAB_MAXDEPTH = 6

FIND_TIMEOUT_LOCAL = 120
FIND_TIMEOUT_LAB = 240


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

MAX_RETRIES = 4


# ============================================================
# LOAD ORIGINAL 40-ROW PILOT
# ============================================================

print("=" * 110)
print("ACTUAL-S2-FIRST PARALLEL VALIDATION V2")
print("=" * 110)

if not INPUT.exists():
    raise FileNotFoundError(
        f"Missing input:\n{INPUT}"
    )

pilot = pd.read_csv(
    INPUT,
    low_memory=False,
)

if len(pilot) != 40:
    raise RuntimeError(
        f"Expected 40 rows, found {len(pilot)}"
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
]

missing = [
    c
    for c in required
    if c not in pilot.columns
]

if missing:
    raise RuntimeError(
        "Missing columns:\n"
        +
        "\n".join(missing)
    )

pilot["_nominal_date"] = pd.to_datetime(
    pilot["Date"],
    errors="coerce",
)

if pilot["_nominal_date"].isna().any():
    raise RuntimeError(
        "Invalid nominal candidate dates."
    )


# ============================================================
# DERIVE 10 PARENT POSITIVES
# ============================================================

parent_rows = []

for parent_num, g in pilot.groupby(
    "Pilot Parent Number",
    sort=True,
):

    inferred_dates = []

    for _, r in g.iterrows():

        inferred = (
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

        inferred_dates.append(
            inferred
        )

    unique_dates = sorted(
        set(inferred_dates)
    )

    if len(unique_dates) != 1:

        raise RuntimeError(
            f"Parent {parent_num} produces "
            f"multiple inferred positive dates: "
            f"{unique_dates}"
        )

    r0 = g.iloc[0]

    parent_rows.append({
        "Pilot Parent Number":
            int(parent_num),

        "Source Positive Record ID":
            r0[
                "Source Positive Record ID"
            ],

        "Site":
            r0["Site"],

        "Latitude":
            float(
                r0["Latitude"]
            ),

        "Longitude":
            float(
                r0["Longitude"]
            ),

        "Parent Positive Date":
            unique_dates[0],
    })


parents = pd.DataFrame(
    parent_rows
)

if len(parents) != 10:
    raise RuntimeError(
        f"Expected 10 parents, found {len(parents)}"
    )


print("\nPARENTS:")
print(
    parents[
        [
            "Pilot Parent Number",
            "Source Positive Record ID",
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
# SENTINEL-2 QA
# ============================================================

def s2_local_qa(
    image,
    region,
):
    """
    Explicitly separate:

      requested patch pixels
      valid SCL pixels
      clear/cloud/shadow/snow pixels

    This fixes the denominator ambiguity from v1.
    """

    scl = image.select(
        "SCL"
    )

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

    invalid_class = (
        scl.eq(0)
        .Or(scl.eq(1))
        .Or(scl.eq(2))
        .rename("invalid_class")
    )

    stack = (
        clear
        .addBands(cloud)
        .addBands(shadow)
        .addBands(snow)
        .addBands(invalid_class)
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

    # Valid SCL count:
    # masked SCL pixels do NOT contribute.
    valid_dict = (
        scl.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=S2_SCALE_M,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    # Full requested patch pixel count.
    requested_dict = (
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

    valid_px = float(
        valid_dict.get("SCL")
        or 0
    )

    requested_px = float(
        requested_dict.get(
            "requested"
        )
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

    invalid_class_px = float(
        sums.get("invalid_class")
        or 0
    )

    valid_fraction = (
        valid_px / requested_px
        if requested_px > 0
        else np.nan
    )

    clear_among_valid = (
        clear_px / valid_px
        if valid_px > 0
        else np.nan
    )

    clear_over_requested = (
        clear_px / requested_px
        if requested_px > 0
        else np.nan
    )

    cloud_over_requested = (
        cloud_px / requested_px
        if requested_px > 0
        else np.nan
    )

    shadow_over_requested = (
        shadow_px / requested_px
        if requested_px > 0
        else np.nan
    )

    snow_over_requested = (
        snow_px / requested_px
        if requested_px > 0
        else np.nan
    )

    invalid_class_fraction = (
        invalid_class_px
        /
        requested_px
        if requested_px > 0
        else np.nan
    )

    masked_fraction = (
        1.0 - valid_fraction
        if not pd.isna(
            valid_fraction
        )
        else np.nan
    )

    qa_pass = (
        not pd.isna(
            clear_over_requested
        )
        and
        clear_over_requested
        >=
        MIN_CLEAR_REQUESTED
    )

    return {
        "requested_px":
            requested_px,

        "valid_scl_px":
            valid_px,

        "clear_px":
            clear_px,

        "valid_fraction":
            valid_fraction,

        "clear_among_valid":
            clear_among_valid,

        "clear_over_requested":
            clear_over_requested,

        "cloud_over_requested":
            cloud_over_requested,

        "shadow_over_requested":
            shadow_over_requested,

        "snow_over_requested":
            snow_over_requested,

        "invalid_class_fraction":
            invalid_class_fraction,

        "masked_fraction":
            masked_fraction,

        "qa_pass":
            bool(qa_pass),
    }


# ============================================================
# SEARCH ALL S2 FOR ONE PARENT
# ============================================================

def search_parent_s2(
    parent,
):

    parent_num = int(
        parent[
            "Pilot Parent Number"
        ]
    )

    positive_date = pd.Timestamp(
        parent[
            "Parent Positive Date"
        ]
    ).normalize()

    lat = float(
        parent["Latitude"]
    )

    lon = float(
        parent["Longitude"]
    )

    start = (
        positive_date
        +
        pd.Timedelta(
            days=SEARCH_START_DAY
        )
    )

    # filterDate end is exclusive.
    end = (
        positive_date
        +
        pd.Timedelta(
            days=SEARCH_END_DAY + 1
        )
    )

    point = ee.Geometry.Point(
        [
            lon,
            lat,
        ]
    )

    region = (
        point
        .buffer(
            HALF_PATCH_M
        )
        .bounds()
    )

    collection = (
        ee.ImageCollection(
            S2_COLLECTION
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

    n = int(
        collection.size().getInfo()
    )

    images = collection.toList(
        n
    )

    records = []

    for i in range(n):

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

        acquisition = pd.to_datetime(
            time_ms,
            unit="ms",
            utc=True,
        )

        acquisition_date = (
            acquisition
            .tz_convert(None)
            .normalize()
        )

        actual_offset_days = int(
            (
                acquisition_date
                -
                positive_date
            ).days
        )

        if not (
            SEARCH_START_DAY
            <=
            actual_offset_days
            <=
            SEARCH_END_DAY
        ):
            continue

        qa = s2_local_qa(
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
                positive_date.strftime(
                    "%Y-%m-%d"
                ),

            "Actual S2 Datetime UTC":
                str(acquisition),

            "Actual S2 Date":
                acquisition.strftime(
                    "%Y-%m-%d"
                ),

            "Actual Offset Calendar Days":
                actual_offset_days,

            "S2 System Index":
                props.get(
                    "system:index"
                ),

            "S2 Product ID":
                props.get(
                    "PRODUCT_ID"
                ),

            "S2 MGRS Tile":
                props.get(
                    "MGRS_TILE"
                ),

            "S2 Scene Cloud Percentage":
                props.get(
                    "CLOUDY_PIXEL_PERCENTAGE"
                ),

            "S2 Requested Pixels":
                qa[
                    "requested_px"
                ],

            "S2 Valid SCL Pixels":
                qa[
                    "valid_scl_px"
                ],

            "S2 Valid SCL Fraction":
                qa[
                    "valid_fraction"
                ],

            "S2 Clear Among Valid Fraction":
                qa[
                    "clear_among_valid"
                ],

            "S2 Clear Over Requested Fraction":
                qa[
                    "clear_over_requested"
                ],

            "S2 Cloud Over Requested Fraction":
                qa[
                    "cloud_over_requested"
                ],

            "S2 Shadow Over Requested Fraction":
                qa[
                    "shadow_over_requested"
                ],

            "S2 Snow Over Requested Fraction":
                qa[
                    "snow_over_requested"
                ],

            "S2 Invalid Class Fraction":
                qa[
                    "invalid_class_fraction"
                ],

            "S2 Masked Fraction":
                qa[
                    "masked_fraction"
                ],

            "S2 QA Pass":
                "pass"
                if qa["qa_pass"]
                else "fail",
        })

    return records


# ============================================================
# RUN 10 PARENTS IN PARALLEL
# ============================================================

print("\n" + "=" * 110)
print("PHASE A — SEARCHING ACTUAL SENTINEL-2 ACQUISITIONS")
print("=" * 110)

all_scene_records = []

with ThreadPoolExecutor(
    max_workers=S2_WORKERS
) as pool:

    future_map = {}

    for _, parent in (
        parents.iterrows()
    ):

        f = pool.submit(
            search_parent_s2,
            parent,
        )

        future_map[
            f
        ] = int(
            parent[
                "Pilot Parent Number"
            ]
        )

    for f in as_completed(
        future_map
    ):

        parent_num = (
            future_map[f]
        )

        try:

            records = f.result()

            all_scene_records.extend(
                records
            )

            print(
                f"[S2] parent {parent_num}: "
                f"{len(records)} scenes"
            )

        except Exception as e:

            print(
                f"[S2] parent {parent_num} FAILED:",
                repr(e)
            )


scene_inventory = pd.DataFrame(
    all_scene_records
)

SCENE_INVENTORY_OUT = (
    OUTDIR
    /
    "01_actual_s2_scene_inventory.csv"
)

scene_inventory.to_csv(
    SCENE_INVENTORY_OUT,
    index=False,
    encoding="utf-8-sig",
)


print(
    "\nTotal S2 scene records:",
    len(scene_inventory)
)

if len(scene_inventory):

    print(
        "S2 QA pass scenes:",
        int(
            scene_inventory[
                "S2 QA Pass"
            ].eq("pass").sum()
        )
    )


# ============================================================
# COLLAPSE OVERLAPPING TILE DUPLICATES INTO OVERPASSES
#
# Same overpass may appear as multiple MGRS tiles.
# Scenes within 20 minutes are treated as one overpass;
# retain best local-QA scene.
# ============================================================

def collapse_parent_overpasses(
    df,
):

    if len(df) == 0:
        return df.copy()

    work = df.copy()

    work["_dt"] = pd.to_datetime(
        work[
            "Actual S2 Datetime UTC"
        ],
        utc=True,
    )

    work = work.sort_values(
        "_dt"
    )

    groups = []

    current = []

    last_dt = None

    for idx, row in (
        work.iterrows()
    ):

        dt = row["_dt"]

        if (
            last_dt is None
            or
            (
                dt - last_dt
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

        last_dt = dt

    if current:
        groups.append(
            current
        )

    retained = []

    for group in groups:

        g = work.loc[
            group
        ].copy()

        g = g.sort_values(
            [
                "S2 Clear Over Requested Fraction",
                "S2 Valid SCL Fraction",
            ],
            ascending=[
                False,
                False,
            ],
        )

        best = g.iloc[0].copy()

        best[
            "S2 Overlap Scene Count"
        ] = len(g)

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

if len(scene_inventory):

    for parent_num, g in (
        scene_inventory.groupby(
            "Pilot Parent Number"
        )
    ):

        overpass_frames.append(
            collapse_parent_overpasses(
                g
            )
        )


if overpass_frames:

    overpasses = pd.concat(
        overpass_frames,
        ignore_index=True,
    )

else:

    overpasses = pd.DataFrame()


OVERPASS_OUT = (
    OUTDIR
    /
    "02_actual_s2_overpass_inventory.csv"
)

overpasses.to_csv(
    OVERPASS_OUT,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# OPTIMAL UNIQUE ASSIGNMENT TO +1/+3/+7/+14
#
# Goals in order:
#   1. maximize number of matched anchors
#   2. minimize total temporal error
#   3. maximize total clear fraction
#
# Same actual overpass cannot be reused twice for one parent.
# ============================================================

def assign_parent(
    parent_num,
    overpass_df,
):

    anchors = (
        TARGET_OFFSETS
    )

    if len(overpass_df):

        usable = (
            overpass_df[
                overpass_df[
                    "S2 QA Pass"
                ]
                ==
                "pass"
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

    else:

        usable = pd.DataFrame()

    options = {}

    for anchor in anchors:

        idxs = []

        if len(usable):

            for idx, r in (
                usable.iterrows()
            ):

                actual = int(
                    r[
                        "Actual Offset Calendar Days"
                    ]
                )

                error = abs(
                    actual
                    -
                    anchor
                )

                if (
                    error
                    <=
                    ANCHOR_TOLERANCE_DAYS
                ):

                    idxs.append(
                        idx
                    )

        options[
            anchor
        ] = idxs

    best = None

    def recurse(
        pos,
        used,
        mapping,
        matched,
        total_error,
        clear_sum,
    ):

        nonlocal best

        if pos == len(
            anchors
        ):

            score = (
                matched,
                -total_error,
                clear_sum,
            )

            if (
                best is None
                or
                score
                >
                best[
                    "score"
                ]
            ):

                best = {
                    "score":
                        score,

                    "mapping":
                        mapping.copy(),
                }

            return

        anchor = anchors[
            pos
        ]

        # unmatched option
        mapping[
            anchor
        ] = None

        recurse(
            pos + 1,
            used,
            mapping,
            matched,
            total_error,
            clear_sum,
        )

        for idx in options[
            anchor
        ]:

            if idx in used:
                continue

            r = usable.loc[
                idx
            ]

            actual = int(
                r[
                    "Actual Offset Calendar Days"
                ]
            )

            error = abs(
                actual
                -
                anchor
            )

            clear = float(
                r[
                    "S2 Clear Over Requested Fraction"
                ]
            )

            used.add(
                idx
            )

            mapping[
                anchor
            ] = idx

            recurse(
                pos + 1,
                used,
                mapping,
                matched + 1,
                total_error + error,
                clear_sum + clear,
            )

            used.remove(
                idx
            )

        mapping.pop(
            anchor,
            None,
        )

    recurse(
        0,
        set(),
        {},
        0,
        0,
        0.0,
    )

    rows = []

    for anchor in anchors:

        base = pilot[
            (
                pilot[
                    "Pilot Parent Number"
                ]
                ==
                parent_num
            )
            &
            (
                pilot[
                    "Resolved Offset Days"
                ]
                ==
                anchor
            )
        ]

        if len(base) != 1:

            raise RuntimeError(
                f"Cannot resolve original pilot row "
                f"for parent {parent_num}, "
                f"anchor {anchor}"
            )

        b = base.iloc[0]

        idx = (
            best["mapping"].get(
                anchor
            )
            if best
            else None
        )

        common = {
            "Pilot Candidate ID":
                b[
                    "Pilot Candidate ID"
                ],

            "Pilot Parent Number":
                parent_num,

            "Source Positive Record ID":
                b[
                    "Source Positive Record ID"
                ],

            "Site":
                b["Site"],

            "Latitude":
                float(
                    b["Latitude"]
                ),

            "Longitude":
                float(
                    b["Longitude"]
                ),

            "Parent Positive Date":
                (
                    pd.Timestamp(
                        b["_nominal_date"]
                    )
                    -
                    pd.Timedelta(
                        days=anchor
                    )
                ).strftime(
                    "%Y-%m-%d"
                ),

            "Target Offset Days":
                anchor,

            "Nominal Candidate Date":
                pd.Timestamp(
                    b[
                        "_nominal_date"
                    ]
                ).strftime(
                    "%Y-%m-%d"
                ),
        }

        if idx is None:

            rows.append({
                **common,

                "S2 Match Status":
                    "no_qa_pass_match_within_tolerance",

                "Actual S2 Datetime UTC":
                    pd.NA,

                "Actual S2 Date":
                    pd.NA,

                "Actual Offset Calendar Days":
                    pd.NA,

                "Offset Error Days":
                    pd.NA,

                "S2 Product ID":
                    pd.NA,

                "S2 MGRS Tile":
                    pd.NA,

                "S2 Clear Over Requested Fraction":
                    pd.NA,

                "S2 Valid SCL Fraction":
                    pd.NA,

                "S2 Clear Among Valid Fraction":
                    pd.NA,

                "S2 Masked Fraction":
                    pd.NA,
            })

            continue

        r = usable.loc[
            idx
        ]

        actual_offset = int(
            r[
                "Actual Offset Calendar Days"
            ]
        )

        rows.append({
            **common,

            "S2 Match Status":
                "matched",

            "Actual S2 Datetime UTC":
                r[
                    "Actual S2 Datetime UTC"
                ],

            "Actual S2 Date":
                r[
                    "Actual S2 Date"
                ],

            "Actual Offset Calendar Days":
                actual_offset,

            "Offset Error Days":
                actual_offset
                -
                anchor,

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

    return rows


assignment_rows = []

for parent_num in sorted(
    parents[
        "Pilot Parent Number"
    ].tolist()
):

    if len(overpasses):

        sub = overpasses[
            overpasses[
                "Pilot Parent Number"
            ]
            ==
            parent_num
        ].copy()

    else:

        sub = pd.DataFrame()

    assignment_rows.extend(
        assign_parent(
            parent_num,
            sub,
        )
    )


assigned = pd.DataFrame(
    assignment_rows
)

if len(assigned) != 40:

    raise RuntimeError(
        f"Expected 40 assignment rows, "
        f"found {len(assigned)}"
    )


ASSIGNMENT_OUT = (
    OUTDIR
    /
    "03_actual_s2_anchor_assignments.csv"
)

assigned.to_csv(
    ASSIGNMENT_OUT,
    index=False,
    encoding="utf-8-sig",
)


print("\n" + "=" * 110)
print("ACTUAL-S2 ASSIGNMENT SUMMARY")
print("=" * 110)

print(
    assigned[
        "S2 Match Status"
    ].value_counts(
        dropna=False
    )
)

print(
    "\nBY TARGET OFFSET:"
)

print(
    assigned.groupby(
        "Target Offset Days"
    )[
        "S2 Match Status"
    ]
    .value_counts()
    .unstack(
        fill_value=0
    )
)


# ============================================================
# ONLY MATCHED ACTUAL-S2 ROWS GO TO SENSOR VALIDATION
# ============================================================

matched = assigned[
    assigned[
        "S2 Match Status"
    ]
    ==
    "matched"
].copy()

print(
    "\nMatched actual-S2 candidates:",
    len(matched)
)


# ============================================================
# HELPERS FOR LOCAL / LAB METHANEAIR POSITIVE CONTEXT
# ============================================================

def mounted_lab():

    try:

        p = subprocess.run(
            [
                "mount"
            ],
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


def timeout_text(x):

    if x is None:
        return ""

    if isinstance(
        x,
        bytes,
    ):
        return x.decode(
            "utf-8",
            errors="replace",
        )

    return str(x)


def discover_methaneair_csvs(
    root,
    maxdepth,
    timeout_sec,
    label,
):

    root = Path(
        root
    )

    if (
        label == "LAB"
        and
        not mounted_lab()
    ):

        return (
            "smb_not_mounted",
            []
        )

    if not root.exists():

        return (
            "root_missing",
            []
        )

    cmd = [
        "find",
        str(root),
        "-maxdepth",
        str(maxdepth),
        "-type",
        "f",
        "(",
        "-iname",
        "*methaneair*.csv",
        "-o",
        "-iname",
        "*methane_air*.csv",
        "-o",
        "-iname",
        "*pcannon*.csv",
        ")",
        "-print",
    ]

    output = ""

    status = "complete"

    try:

        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        output = (
            p.stdout
            or ""
        )

        if p.returncode != 0:

            status = (
                f"returncode_{p.returncode}"
            )

    except subprocess.TimeoutExpired as e:

        status = (
            "timeout_partial"
        )

        output = timeout_text(
            e.stdout
        )

    files = []

    seen = set()

    for line in output.splitlines():

        line = line.strip()

        if (
            not line
            or
            line in seen
        ):
            continue

        seen.add(
            line
        )

        files.append(
            Path(line)
        )

    return (
        status,
        files,
    )


def norm(s):

    return (
        str(s)
        .strip()
        .lower()
        .replace(
            " ",
            "_",
        )
        .replace(
            "-",
            "_",
        )
    )


def find_col(
    cols,
    names,
):

    cmap = {
        norm(c): c
        for c in cols
    }

    for name in names:

        n = norm(
            name
        )

        if n in cmap:
            return cmap[n]

    for name in names:

        n = norm(
            name
        )

        for k, v in (
            cmap.items()
        ):

            if n in k:
                return v

    return None


def haversine_vector(
    lat0,
    lon0,
    lat,
    lon,
):

    R = 6371.0088

    p0 = np.radians(
        lat0
    )

    q0 = np.radians(
        lon0
    )

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

    dp = p - p0
    dq = q - q0

    a = (
        np.sin(
            dp / 2
        ) ** 2
        +
        np.cos(p0)
        *
        np.cos(p)
        *
        np.sin(
            dq / 2
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


def source_time_series(
    df,
):

    full = find_col(
        df.columns,
        [
            "datetime",
            "timestamp",
            "acquisition_time",
            "observation_time",
            "time_coverage_start",
            "scene_datetime",
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

    if time_col is not None:

        text = (
            df[
                date_col
            ]
            .astype(
                "string"
            )
            .fillna("")
            .str.strip()
            +
            " "
            +
            df[
                time_col
            ]
            .astype(
                "string"
            )
            .fillna("")
            .str.strip()
        )

    else:

        text = df[
            date_col
        ].astype(
            "string"
        )

    return pd.to_datetime(
        text,
        errors="coerce",
        utc=True,
    )


def run_existing_data_branch(
    root,
    maxdepth,
    timeout_sec,
    label,
):

    print(
        f"\n[{label}] searching MethaneAIR/plume CSVs..."
    )

    status, files = (
        discover_methaneair_csvs(
            root,
            maxdepth,
            timeout_sec,
            label,
        )
    )

    print(
        f"[{label}] discovery={status}, "
        f"files={len(files)}"
    )

    inventory_path = (
        OUTDIR
        /
        f"{label.lower()}_methaneair_csv_inventory.txt"
    )

    with open(
        inventory_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"STATUS: {status}\n"
        )

        f.write(
            f"COUNT: {len(files)}\n\n"
        )

        for p in files:
            f.write(
                str(p)
                +
                "\n"
            )

    records = []

    for file_path in files:

        try:

            df = pd.read_csv(
                file_path,
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

        times = (
            source_time_series(
                df
            )
        )

        if times is None:
            continue

        src_lat = pd.to_numeric(
            df[
                lat_col
            ],
            errors="coerce",
        )

        src_lon = pd.to_numeric(
            df[
                lon_col
            ],
            errors="coerce",
        )

        valid = (
            src_lat.notna()
            &
            src_lon.notna()
            &
            times.notna()
        )

        if not valid.any():
            continue

        temp = pd.DataFrame({
            "_row":
                df.index,

            "_lat":
                src_lat,

            "_lon":
                src_lon,

            "_time":
                times,
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
            matched.iterrows()
        ):

            actual_date = pd.to_datetime(
                cand[
                    "Actual S2 Date"
                ],
                utc=True,
            )

            dt_days = (
                temp[
                    "_time"
                ]
                -
                actual_date
            ).dt.total_seconds() / 86400

            temporal = (
                dt_days.abs()
                <=
                METHANEAIR_CONTEXT_DAYS
            )

            if not temporal.any():
                continue

            sub = (
                temp.loc[
                    temporal
                ]
                .copy()
            )

            sub[
                "_dt_days"
            ] = dt_days.loc[
                temporal
            ]

            sub[
                "_dist_km"
            ] = (
                haversine_vector(
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
                    sub[
                        "_lat"
                    ],
                    sub[
                        "_lon"
                    ],
                )
            )

            sub = sub[
                sub[
                    "_dist_km"
                ]
                <=
                METHANEAIR_RADIUS_KM
            ]

            for _, hit in (
                sub.iterrows()
            ):

                source_idx = int(
                    hit[
                        "_row"
                    ]
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

                # Avoid counting the original parent positive
                # as independent evidence when IDs are identical.
                if (
                    source_id is not None
                    and
                    str(source_id)
                    ==
                    str(
                        cand[
                            "Source Positive Record ID"
                        ]
                    )
                ):
                    continue

                records.append({
                    "Pilot Candidate ID":
                        cand[
                            "Pilot Candidate ID"
                        ],

                    "Origin":
                        label,

                    "Source File":
                        str(
                            file_path
                        ),

                    "Source Record ID":
                        source_id,

                    "Source Datetime UTC":
                        str(
                            hit[
                                "_time"
                            ]
                        ),

                    "Distance km":
                        float(
                            hit[
                                "_dist_km"
                            ]
                        ),

                    "Time Difference days":
                        float(
                            hit[
                                "_dt_days"
                            ]
                        ),

                    "Same Calendar Day":
                        (
                            hit[
                                "_time"
                            ].date()
                            ==
                            actual_date.date()
                        ),
                })

    return {
        "status":
            status,

        "files":
            files,

        "matches":
            pd.DataFrame(
                records
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
                    "MethaneFuse-negative-validation-v2"
                },
            )

            r.raise_for_status()

            return (
                r.json()
                .get(
                    "feed",
                    {}
                )
                .get(
                    "entry",
                    []
                )
            )

        except Exception as e:

            last_error = e

            time.sleep(
                attempt
            )

    raise RuntimeError(
        repr(
            last_error
        )
    )


def emit_one(
    row,
):

    cid = row[
        "Pilot Candidate ID"
    ]

    lat = float(
        row["Latitude"]
    )

    lon = float(
        row["Longitude"]
    )

    d = pd.Timestamp(
        row[
            "Actual S2 Date"
        ]
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
        pd.Timedelta(
            days=3
        )
    ).strftime(
        "%Y-%m-%dT00:00:00Z"
    )

    ctx_end = (
        d
        +
        pd.Timedelta(
            days=3
        )
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
                len(
                    exact
                ),

            "EMIT +/-3d Count":
                len(
                    context
                ),

            "EMIT Exact Titles":
                " | ".join(
                    str(
                        x.get(
                            "title"
                        )
                    )
                    for x in exact
                ),

            "EMIT +/-3d Titles":
                " | ".join(
                    str(
                        x.get(
                            "title"
                        )
                    )
                    for x in context
                ),

            "EMIT Coverage Status":
                (
                    "exact_date"
                    if len(
                        exact
                    ) > 0
                    else
                    (
                        "context_only"
                        if len(
                            context
                        ) > 0
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
        "\n[EMIT] validating actual S2 dates..."
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
            for _, row in (
                matched.iterrows()
            )
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
                or
                done == len(
                    matched
                )
            ):

                print(
                    f"[EMIT] "
                    f"{done}/{len(matched)}"
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
            "count":
                0,

            "regional":
                None,

            "background":
                None,

            "delta":
                None,

            "uncertainty":
                None,

            "valid_pixels":
                0,
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

    src = (
        image.reduceRegion(
            reducer=ee.Reducer.median(),
            geometry=source_region,
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
            geometry=background_region,
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
            geometry=source_region,
            scale=TROPOMI_SCALE_M,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .getInfo()
    )

    regional = src.get(
        TROPOMI_CH4_BAND
    )

    uncertainty = src.get(
        TROPOMI_UNC_BAND
    )

    background = bg.get(
        TROPOMI_CH4_BAND
    )

    delta = None

    if (
        regional is not None
        and
        background is not None
    ):

        delta = (
            float(
                regional
            )
            -
            float(
                background
            )
        )

    return {
        "count":
            count,

        "regional":
            regional,

        "background":
            background,

        "delta":
            delta,

        "uncertainty":
            uncertainty,

        "valid_pixels":
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

    cid = row[
        "Pilot Candidate ID"
    ]

    lat = float(
        row["Latitude"]
    )

    lon = float(
        row["Longitude"]
    )

    d = pd.Timestamp(
        row[
            "Actual S2 Date"
        ]
    )

    point = ee.Geometry.Point(
        [
            lon,
            lat,
        ]
    )

    exact = (
        ee.ImageCollection(
            TROPOMI_COLLECTION
        )
        .filterBounds(
            point
        )
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

    context = (
        ee.ImageCollection(
            TROPOMI_COLLECTION
        )
        .filterBounds(
            point
        )
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

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            a = tropomi_stats(
                exact,
                point,
            )

            b = tropomi_stats(
                context,
                point,
            )

            return {
                "Pilot Candidate ID":
                    cid,

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
                    a["valid_pixels"],

                "TROPOMI +/-1d Image Count":
                    b["count"],

                "TROPOMI +/-1d Delta ppb":
                    b["delta"],

                "TROPOMI +/-1d Valid Pixels":
                    b["valid_pixels"],

                "TROPOMI Evidence Role":
                    (
                        "regional_support_only"
                        if (
                            a[
                                "valid_pixels"
                            ]
                            > 0
                            or
                            b[
                                "valid_pixels"
                            ]
                            > 0
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
                2 * attempt
            )

    return {
        "Pilot Candidate ID":
            cid,

        "TROPOMI Evidence Role":
            "query_error",

        "TROPOMI Query Error":
            repr(
                last_error
            ),
    }


def run_tropomi_batch():

    print(
        "\n[TROPOMI] validating actual S2 dates..."
    )

    rows = []

    with ThreadPoolExecutor(
        max_workers=TROPOMI_WORKERS
    ) as pool:

        futures = [
            pool.submit(
                tropomi_one,
                row,
            )
            for _, row in (
                matched.iterrows()
            )
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
                or
                done == len(
                    matched
                )
            ):

                print(
                    f"[TROPOMI] "
                    f"{done}/{len(matched)}"
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PHASE B — FOUR PARALLEL BRANCHES
# ============================================================

print("\n" + "=" * 110)
print("PHASE B — PARALLEL MULTI-SOURCE VALIDATION")
print("=" * 110)

print(
    "Matched S2 candidates entering Phase B:",
    len(matched)
)

with ThreadPoolExecutor(
    max_workers=4
) as outer:

    futures = {
        outer.submit(
            run_existing_data_branch,
            PROJECT,
            LOCAL_MAXDEPTH,
            FIND_TIMEOUT_LOCAL,
            "LOCAL",
        ):
            "LOCAL",

        outer.submit(
            run_existing_data_branch,
            LAB_ROOT,
            LAB_MAXDEPTH,
            FIND_TIMEOUT_LAB,
            "LAB",
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

    for f in as_completed(
        futures
    ):

        name = futures[
            f
        ]

        try:

            results[
                name
            ] = f.result()

            print(
                f"\n[{name}] COMPLETE"
            )

        except Exception as e:

            results[
                name
            ] = {
                "branch_error":
                    repr(e),

                "traceback":
                    traceback.format_exc(),
            }

            print(
                f"\n[{name}] FAILED:",
                repr(e)
            )


# ============================================================
# SAVE LOCAL / LAB MATCHES
# ============================================================

context_frames = []

for name in [
    "LOCAL",
    "LAB",
]:

    result = results.get(
        name,
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
            pd.DataFrame
        )
        and
        len(frame)
    ):

        context_frames.append(
            frame
        )


if context_frames:

    methaneair_context = pd.concat(
        context_frames,
        ignore_index=True,
    )

else:

    methaneair_context = (
        pd.DataFrame()
    )


CONTEXT_OUT = (
    OUTDIR
    /
    "04_local_lab_methaneair_context_matches.csv"
)

methaneair_context.to_csv(
    CONTEXT_OUT,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# ONE-ROW-PER-CANDIDATE LOCAL/LAB SUMMARY
# ============================================================

context_summary = []

for _, row in (
    matched.iterrows()
):

    cid = row[
        "Pilot Candidate ID"
    ]

    if len(
        methaneair_context
    ):

        sub = (
            methaneair_context[
                methaneair_context[
                    "Pilot Candidate ID"
                ]
                ==
                cid
            ]
        )

    else:

        sub = pd.DataFrame()

    same_day = 0

    if (
        len(sub)
        and
        "Same Calendar Day"
        in sub.columns
    ):

        same_day = int(
            sub[
                "Same Calendar Day"
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
            .sum()
        )

    context_summary.append({
        "Pilot Candidate ID":
            cid,

        "MethaneAIR/Plume +/-14d Context Count":
            len(sub),

        "MethaneAIR/Plume Same-Day Count":
            same_day,
    })


context_summary = pd.DataFrame(
    context_summary
)


# ============================================================
# MERGE ACTUAL-S2 ASSIGNMENTS + SENSOR RESULTS
# ============================================================

final = assigned.copy()

if len(
    context_summary
):

    final = final.merge(
        context_summary,
        on="Pilot Candidate ID",
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
        on="Pilot Candidate ID",
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
        on="Pilot Candidate ID",
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
            "S2 Match Status"
        ]
        !=
        "matched"
    ):

        return (
            "NO_USABLE_ACTUAL_S2"
        )

    same_day_plume = row.get(
        "MethaneAIR/Plume Same-Day Count",
        0,
    )

    try:

        if (
            pd.notna(
                same_day_plume
            )
            and
            int(
                same_day_plume
            )
            > 0
        ):

            return (
                "REVIEW_SAME_DAY_KNOWN_PLUME"
            )

    except Exception:

        pass

    emit_status = row.get(
        "EMIT Coverage Status"
    )

    if (
        emit_status
        ==
        "exact_date"
    ):

        return (
            "HIGH_RES_COVERAGE_REQUIRES_EMIT_SIGNAL_QA"
        )

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

    return (
        "U_UNKNOWN"
    )


final[
    "Parallel Validation Status"
] = final.apply(
    classify,
    axis=1,
)


# ============================================================
# SAVE FINAL
# ============================================================

FINAL_CSV = (
    OUTDIR
    /
    "05_actual_s2_parallel_multisource_audit.csv"
)

FINAL_XLSX = (
    OUTDIR
    /
    "05_actual_s2_parallel_multisource_audit.xlsx"
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
        pd.DatetimeTZDtype,
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

print("\n" + "=" * 110)
print("FINAL V2 SUMMARY")
print("=" * 110)

print(
    "\nActual S2 anchor matching:"
)

print(
    final[
        "S2 Match Status"
    ]
    .value_counts(
        dropna=False
    )
)


print(
    "\nBY TARGET OFFSET:"
)

print(
    final.groupby(
        "Target Offset Days"
    )[
        "S2 Match Status"
    ]
    .value_counts()
    .unstack(
        fill_value=0
    )
)


if (
    "EMIT Coverage Status"
    in final.columns
):

    print(
        "\nEMIT among actual-S2 candidates:"
    )

    print(
        final.loc[
            final[
                "S2 Match Status"
            ]
            ==
            "matched",
            "EMIT Coverage Status",
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
        "\nTROPOMI among actual-S2 candidates:"
    )

    print(
        final.loc[
            final[
                "S2 Match Status"
            ]
            ==
            "matched",
            "TROPOMI Evidence Role",
        ]
        .value_counts(
            dropna=False
        )
    )


print(
    "\nFINAL STATUS:"
)

print(
    final[
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
    "\n1. All actual S2 scenes:"
)

print(
    SCENE_INVENTORY_OUT
)

print(
    "\n2. Deduplicated S2 overpasses:"
)

print(
    OVERPASS_OUT
)

print(
    "\n3. +1/+3/+7/+14 assignments:"
)

print(
    ASSIGNMENT_OUT
)

print(
    "\n4. Local/Lab methane context:"
)

print(
    CONTEXT_OUT
)

print(
    "\n5. Final multi-source audit:"
)

print(
    FINAL_CSV
)

print(
    FINAL_XLSX
)


print(
    "\n✅ Actual S2 acquisitions searched first"
)

print(
    "✅ Same S2 overpass cannot be reused "
    "for two anchors in the same parent"
)

print(
    "✅ +/-3-day anchor tolerance recorded explicitly"
)

print(
    "✅ Mac and Lab SMB branches run independently"
)

print(
    "✅ EMIT and TROPOMI run in parallel"
)

print(
    "✅ TROPOMI remains regional-support only"
)

print(
    "✅ No imagery downloaded"
)

print(
    "✅ No existing data modified or deleted"
)

