#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import numpy as np
import pandas as pd
import ee


# ============================================================
# CONFIG
# ============================================================

HOME = Path.home()
PROJECT = HOME / "methane_release_project"

V5_FLIGHTS = (
    PROJECT
    / "candidate_negative_validation"
    / "methaneair_coverage_first_v5"
    / "02_methaneair_flight_level_inventory.csv"
)

V6_OVERPASSES = (
    PROJECT
    / "candidate_negative_validation"
    / "methaneair_b5_s2_sensitivity_v6"
    / "03_unique_s2_overpasses_within_7d.csv"
)

OUTDIR = (
    PROJECT
    / "candidate_negative_validation"
    / "methaneair_s2_corrected_qa_v7"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

EE_PROJECT = "methane-release-gee"
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

B_CLASS = (
    "B_HIGH_RES_NO_L4_DETECTION_BUT_NO_USABLE_ALIGNED_S2"
)

HALF_PATCH_M = 240.0

# Sentinel-2 SCL native scale.
SCL_SCALE_M = 20

# Keep the original formal QA standard available.
EXISTING_CLEAR_OVER_REQUESTED_THRESHOLD = 0.80

# New diagnostic: separate coverage from cloud/scene quality.
SEPARATE_MIN_COVERAGE = 0.80
SEPARATE_MIN_CLEAR_AMONG_COVERED = 0.80

# Sensitivity only; not a formal threshold.
SENSITIVITY_CLEAR_OVER_REQUESTED = 0.75

STRICT_HOURS = 72.0
SECONDARY_HOURS = 120.0
EXPLORATORY_HOURS = 168.0

WORKERS = 4


# ============================================================
# LOAD INPUTS
# ============================================================

print("=" * 122)
print("SENTINEL-2 CORRECTED-GRID QA V7")
print("=" * 122)

for p in [V5_FLIGHTS, V6_OVERPASSES]:
    if not p.exists():
        raise FileNotFoundError(p)

v5 = pd.read_csv(V5_FLIGHTS, low_memory=False)
v6 = pd.read_csv(V6_OVERPASSES, low_memory=False)

b = v5[
    v5["Coverage-First Classification"].eq(B_CLASS)
].copy()

if len(b) != 5:
    print(f"WARNING: expected 5 B flights, found {len(b)}")

coord_cols = [
    "Pilot Parent Number",
    "MethaneAIR Flight ID",
    "Latitude",
    "Longitude",
]

missing = [
    c for c in coord_cols
    if c not in b.columns
]

if missing:
    raise RuntimeError(
        "Missing v5 coordinate columns:\n"
        + "\n".join(missing)
    )

needed_v6 = [
    "Pilot Parent Number",
    "MethaneAIR Flight ID",
    "MethaneAIR Midpoint UTC",
    "S2 Datetime UTC",
    "S2 Product ID",
    "S2 System Index",
    "S2 MGRS Tile",
    "S2 Delta Hours From MethaneAIR",
    "S2 Abs Delta Hours",
    "S2 Post Positive",
]

missing = [
    c for c in needed_v6
    if c not in v6.columns
]

if missing:
    raise RuntimeError(
        "Missing v6 overpass columns:\n"
        + "\n".join(missing)
    )

# v6 already carries the MethaneAIR evidence columns.
# Only merge fields that are actually missing from v6; otherwise pandas
# creates _x/_y suffixes and the canonical column names disappear.
join_keys = [
    "Pilot Parent Number",
    "MethaneAIR Flight ID",
]

candidate_add_cols = [
    "Latitude",
    "Longitude",
    "MethaneAIR Source Valid Fraction",
    "MethaneAIR Background Valid Fraction",
    "MethaneAIR Source Minus Background ppb",
    "MethaneAIR TRUE Nearest Same-Flight L4 Distance m",
]

add_cols = [
    c
    for c in candidate_add_cols
    if c in b.columns and c not in v6.columns
]

work = v6.merge(
    b[join_keys + add_cols],
    on=join_keys,
    how="inner",
    validate="many_to_one",
)

# Defensive check: required canonical fields must exist after merge.
required_after_merge = [
    "Latitude",
    "Longitude",
    "MethaneAIR Source Valid Fraction",
    "MethaneAIR Background Valid Fraction",
    "MethaneAIR Source Minus Background ppb",
]

missing_after_merge = [
    c
    for c in required_after_merge
    if c not in work.columns
]

if missing_after_merge:
    raise RuntimeError(
        "Required fields missing after v5/v6 merge:\n"
        + "\n".join(f"  {c}" for c in missing_after_merge)
        + "\n\nAvailable columns:\n"
        + "\n".join(f"  {c}" for c in work.columns)
    )

if len(work) == 0:
    raise RuntimeError(
        "No v6 overpasses matched the five v5 B flights."
    )

print("\nOverpass rows entering corrected QA:", len(work))
print(
    "Unique S2 products:",
    work["S2 Product ID"].nunique()
)


# ============================================================
# EARTH ENGINE
# ============================================================

print("\nInitializing Earth Engine...")
ee.Initialize(project=EE_PROJECT)
print("Earth Engine ready.")


# ============================================================
# CORRECTED QA
#
# Key change:
# - one SCL-derived image
# - unmask to sentinel class 255
# - all numerator/denominator bands share same projection/mask
# - one unweighted reducer
# - explicit SCL projection + 20 m scale
#
# Therefore class fractions cannot exceed coverage fraction,
# and clear-among-covered cannot exceed 1.
# ============================================================

def corrected_s2_qa(image, lon, lat):
    point = ee.Geometry.Point([lon, lat])

    region = (
        point
        .buffer(HALF_PATCH_M)
        .bounds()
    )

    scl = image.select("SCL")
    proj = scl.projection()

    # Fill all masked/out-of-footprint pixels with 255 and expand
    # the footprint so the requested 480 m patch has one common
    # categorical raster grid.
    filled = scl.unmask(
        value=255,
        sameFootprint=False,
    ).rename("SCL_filled")

    requested = (
        filled.multiply(0)
        .add(1)
        .rename("requested")
    )

    covered = (
        filled.neq(255)
        .rename("covered")
    )

    clear = (
        filled.gte(4)
        .And(filled.lte(7))
        .rename("clear")
    )

    cloud = (
        filled.gte(8)
        .And(filled.lte(10))
        .rename("cloud")
    )

    shadow = (
        filled.eq(3)
        .rename("shadow")
    )

    snow = (
        filled.eq(11)
        .rename("snow")
    )

    invalid = (
        filled.gte(0)
        .And(filled.lte(2))
        .rename("invalid")
    )

    masked = (
        filled.eq(255)
        .rename("masked")
    )

    # Catch any unexpected class not covered above.
    classified = (
        clear
        .Or(cloud)
        .Or(shadow)
        .Or(snow)
        .Or(invalid)
        .Or(masked)
    )

    other = (
        classified.Not()
        .rename("other")
    )

    stack = (
        requested
        .addBands(covered)
        .addBands(clear)
        .addBands(cloud)
        .addBands(shadow)
        .addBands(snow)
        .addBands(invalid)
        .addBands(masked)
        .addBands(other)
    )

    stats = stack.reduceRegion(
        reducer=ee.Reducer.sum().unweighted(),
        geometry=region,
        crs=proj,
        scale=SCL_SCALE_M,
        bestEffort=False,
        maxPixels=1_000_000,
    ).getInfo()

    def number(name):
        return float(stats.get(name) or 0)

    n_requested = number("requested")
    n_covered = number("covered")
    n_clear = number("clear")
    n_cloud = number("cloud")
    n_shadow = number("shadow")
    n_snow = number("snow")
    n_invalid = number("invalid")
    n_masked = number("masked")
    n_other = number("other")

    def frac(a, d):
        return a / d if d > 0 else np.nan

    coverage_fraction = frac(
        n_covered,
        n_requested,
    )

    clear_among_covered = frac(
        n_clear,
        n_covered,
    )

    clear_over_requested = frac(
        n_clear,
        n_requested,
    )

    class_sum = (
        n_clear
        + n_cloud
        + n_shadow
        + n_snow
        + n_invalid
        + n_masked
        + n_other
    )

    partition_error = (
        class_sum - n_requested
    )

    # Hard internal QA checks.
    impossible = (
        (
            not pd.isna(clear_among_covered)
            and clear_among_covered > 1.0000001
        )
        or
        (
            not pd.isna(clear_over_requested)
            and
            not pd.isna(coverage_fraction)
            and clear_over_requested > coverage_fraction + 1e-9
        )
        or
        abs(partition_error) > 1e-6
    )

    return {
        "Corrected Requested Pixel Count":
            n_requested,

        "Corrected Covered Pixel Count":
            n_covered,

        "Corrected Clear Pixel Count":
            n_clear,

        "Corrected Coverage Fraction":
            coverage_fraction,

        "Corrected Clear Among Covered Fraction":
            clear_among_covered,

        "Corrected Clear Over Requested Fraction":
            clear_over_requested,

        "Corrected Cloud Over Requested Fraction":
            frac(n_cloud, n_requested),

        "Corrected Shadow Over Requested Fraction":
            frac(n_shadow, n_requested),

        "Corrected Snow Over Requested Fraction":
            frac(n_snow, n_requested),

        "Corrected Invalid Class Fraction":
            frac(n_invalid, n_requested),

        "Corrected Masked Fraction":
            frac(n_masked, n_requested),

        "Corrected Other Fraction":
            frac(n_other, n_requested),

        "Corrected Partition Error Pixels":
            partition_error,

        "Corrected QA Impossible Fraction Flag":
            bool(impossible),

        # Existing formal standard, recalculated consistently.
        "Corrected Existing QA80 Pass":
            (
                bool(
                    clear_over_requested
                    >=
                    EXISTING_CLEAR_OVER_REQUESTED_THRESHOLD
                )
                if not pd.isna(clear_over_requested)
                else False
            ),

        # Diagnostic alternative. Do NOT silently replace the
        # formal standard with this rule.
        "Corrected Separate-Gate Pass":
            (
                bool(
                    coverage_fraction
                    >=
                    SEPARATE_MIN_COVERAGE
                    and
                    clear_among_covered
                    >=
                    SEPARATE_MIN_CLEAR_AMONG_COVERED
                )
                if (
                    not pd.isna(coverage_fraction)
                    and
                    not pd.isna(clear_among_covered)
                )
                else False
            ),

        "Corrected QA75 Sensitivity Pass":
            (
                bool(
                    clear_over_requested
                    >=
                    SENSITIVITY_CLEAR_OVER_REQUESTED
                )
                if not pd.isna(clear_over_requested)
                else False
            ),
    }


# ============================================================
# IMAGE FETCH
# ============================================================

def get_s2_image(row):
    system_index = str(
        row.get("S2 System Index")
        or ""
    ).strip()

    if (
        system_index
        and
        system_index.lower()
        not in {"nan", "none", "<na>"}
    ):
        return ee.Image(
            f"{S2_COLLECTION}/{system_index}"
        )

    product_id = str(
        row["S2 Product ID"]
    )

    image = (
        ee.ImageCollection(S2_COLLECTION)
        .filter(
            ee.Filter.eq(
                "PRODUCT_ID",
                product_id,
            )
        )
        .first()
    )

    return ee.Image(image)


# ============================================================
# UNIQUE PRODUCT/LOCATION QA CACHE
# ============================================================

def qa_key(row):
    return (
        str(row["S2 Product ID"]),
        round(float(row["Latitude"]), 7),
        round(float(row["Longitude"]), 7),
    )


unique_requests = (
    work[
        [
            "S2 Product ID",
            "S2 System Index",
            "Latitude",
            "Longitude",
        ]
    ]
    .drop_duplicates()
    .copy()
)

print(
    "Unique product/location QA requests:",
    len(unique_requests)
)


def qa_one(row):
    image = get_s2_image(row)

    qa = corrected_s2_qa(
        image,
        float(row["Longitude"]),
        float(row["Latitude"]),
    )

    qa.update({
        "S2 Product ID":
            row["S2 Product ID"],

        "Latitude":
            float(row["Latitude"]),

        "Longitude":
            float(row["Longitude"]),
    })

    return qa


# ============================================================
# LOCAL / LAB CACHE AUDITS
# ============================================================

def inventory_sources(label):
    roots = [
        (
            PROJECT
            / "candidate_negative_validation"
            / "methaneair_2025_parallel_v4"
        ),
        (
            PROJECT
            / "candidate_negative_validation"
            / "actual_s2_45day_parallel_v3"
        ),
        (
            PROJECT
            / "candidate_negative_validation"
            / "parallel_multisource_40"
        ),
    ]

    if label == "LOCAL":
        names = [
            "local_methaneair_file_inventory.txt",
            "04_local_metadata_inventory.txt",
            "local_existing_sensor_files.txt",
        ]
    else:
        names = [
            "lab_methaneair_file_inventory.txt",
            "04_lab_metadata_inventory.txt",
            "lab_existing_sensor_files.txt",
        ]

    sources = []
    paths = []
    seen = set()

    for root in roots:
        for name in names:
            p = root / name

            if not p.exists():
                continue

            sources.append(str(p))

            for line in p.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines():

                line = line.strip()

                if (
                    not line
                    or line.startswith("STATUS:")
                    or line.startswith("COUNT:")
                ):
                    continue

                if line in seen:
                    continue

                seen.add(line)
                paths.append(line)

    return sources, paths


def cache_audit(label):
    sources, paths = inventory_sources(label)

    methaneair_tokens = [
        "methaneair",
        "methane_air",
        "mair",
        "pcannon",
        "flight",
        "coverage",
        "retrieval",
    ]

    relevant = [
        p for p in paths
        if any(
            token in p.lower()
            for token in methaneair_tokens
        )
    ]

    mounted = pd.NA

    if label == "LAB":
        try:
            r = subprocess.run(
                ["mount"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            mounted = (
                "/Volumes/engg-leung"
                in r.stdout
            )
        except Exception:
            mounted = False

    return {
        "Origin":
            label,

        "Inventory Cache Files":
            " | ".join(sources),

        "Unique Cached Paths":
            len(paths),

        "MethaneAIR-Relevant Cached Paths":
            len(relevant),

        "Lab SMB Mounted":
            mounted,

        "Mode":
            "CACHE_READ_ONLY_NO_RECURSIVE_SCAN",
    }


# ============================================================
# RUN GEE QA + MAC CACHE + LAB CACHE TOGETHER
# ============================================================

def run_gee_qa():
    rows = []

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as pool:

        futures = [
            pool.submit(
                qa_one,
                row,
            )
            for _, row in (
                unique_requests.iterrows()
            )
        ]

        done = 0

        for f in as_completed(futures):
            rows.append(f.result())

            done += 1

            print(
                f"[GEE-QA] {done}/"
                f"{len(unique_requests)}"
            )

    return pd.DataFrame(rows)


print("\nStarting parallel branches:")
print("  1. Corrected GEE S2 QA")
print("  2. Mac metadata cache audit")
print("  3. Lab SMB metadata cache audit")

with ThreadPoolExecutor(
    max_workers=3
) as outer:

    futures = {
        outer.submit(run_gee_qa):
            "GEE",

        outer.submit(cache_audit, "LOCAL"):
            "LOCAL",

        outer.submit(cache_audit, "LAB"):
            "LAB",
    }

    results = {}

    for f in as_completed(futures):
        name = futures[f]

        try:
            results[name] = f.result()
            print(f"[{name}] BRANCH COMPLETE")

        except Exception as e:
            results[name] = {
                "Branch Error":
                    repr(e)
            }

            print(
                f"[{name}] BRANCH FAILED:",
                repr(e),
            )


corrected_cache = results.get("GEE")

if not isinstance(
    corrected_cache,
    pd.DataFrame,
):
    raise RuntimeError(
        "Corrected GEE QA branch failed."
    )

cache_rows = []

for label in ["LOCAL", "LAB"]:
    r = results.get(label)

    if isinstance(r, dict):
        cache_rows.append(r)

cache_df = pd.DataFrame(cache_rows)


# ============================================================
# MERGE CORRECTED QA BACK TO EVERY FLIGHT-OVERPASS PAIR
# ============================================================

corrected = work.merge(
    corrected_cache,
    on=[
        "S2 Product ID",
        "Latitude",
        "Longitude",
    ],
    how="left",
    validate="many_to_one",
)


# ============================================================
# CORRECTED TEMPORAL RESULT PER FLIGHT
# ============================================================

def tier(abs_hours):
    if abs_hours <= STRICT_HOURS:
        return "STRICT_LE_72H"

    if abs_hours <= SECONDARY_HOURS:
        return "SECONDARY_72_TO_120H"

    if abs_hours <= EXPLORATORY_HOURS:
        return "EXPLORATORY_120_TO_168H"

    return "OUTSIDE_7D"


corrected[
    "Corrected S2 Temporal Tier"
] = pd.to_numeric(
    corrected["S2 Abs Delta Hours"],
    errors="coerce",
).apply(
    lambda x:
        tier(float(x))
        if pd.notna(x)
        else pd.NA
)


flight_summary_rows = []

for (
    parent_num,
    flight_id,
), g in corrected.groupby(
    [
        "Pilot Parent Number",
        "MethaneAIR Flight ID",
    ],
    dropna=False,
):

    g = g.copy()

    # Formal temporal negatives still must be post-positive.
    post = g[
        g["S2 Post Positive"]
        .astype(str)
        .str.lower()
        .eq("true")
    ].copy()

    strict_qa80 = post[
        post[
            "Corrected Existing QA80 Pass"
        ].eq(True)
    ].copy()

    separate_gate = post[
        post[
            "Corrected Separate-Gate Pass"
        ].eq(True)
    ].copy()

    qa75 = post[
        post[
            "Corrected QA75 Sensitivity Pass"
        ].eq(True)
    ].copy()

    def choose(frame):
        if len(frame) == 0:
            return None

        ranks = {
            "STRICT_LE_72H": 0,
            "SECONDARY_72_TO_120H": 1,
            "EXPLORATORY_120_TO_168H": 2,
        }

        x = frame.copy()

        x["_rank"] = (
            x[
                "Corrected S2 Temporal Tier"
            ]
            .map(ranks)
            .fillna(9)
        )

        return (
            x.sort_values(
                [
                    "_rank",
                    "S2 Abs Delta Hours",
                    "Corrected Clear Over Requested Fraction",
                ],
                ascending=[
                    True,
                    True,
                    False,
                ],
            )
            .iloc[0]
        )

    best80 = choose(strict_qa80)
    bestsep = choose(separate_gate)
    best75 = choose(qa75)

    first = g.iloc[0]

    out = {
        "Pilot Parent Number":
            parent_num,

        "Site":
            first["Site"],

        "MethaneAIR Flight ID":
            flight_id,

        "MethaneAIR Midpoint UTC":
            first["MethaneAIR Midpoint UTC"],

        "MethaneAIR Source Valid Fraction":
            first[
                "MethaneAIR Source Valid Fraction"
            ],

        "MethaneAIR Background Valid Fraction":
            first[
                "MethaneAIR Background Valid Fraction"
            ],

        "MethaneAIR Source Minus Background ppb":
            first[
                "MethaneAIR Source Minus Background ppb"
            ],

        "Corrected Overpasses Within 7d":
            len(g),

        "Corrected Post-Positive Overpasses":
            len(post),

        "Corrected QA80-Pass Overpasses":
            len(strict_qa80),

        "Corrected Separate-Gate-Pass Overpasses":
            len(separate_gate),

        "Corrected QA75-Pass Overpasses":
            len(qa75),
    }

    if best80 is None:
        out.update({
            "Corrected Formal Result":
                "NO_QA80_S2_WITHIN_7D",

            "Corrected Formal S2 Product ID":
                pd.NA,

            "Corrected Formal Delta Hours":
                pd.NA,

            "Corrected Formal Clear Over Requested":
                pd.NA,

            "Corrected Formal Coverage Fraction":
                pd.NA,

            "Corrected Formal Clear Among Covered":
                pd.NA,
        })

    else:
        out.update({
            "Corrected Formal Result":
                best80[
                    "Corrected S2 Temporal Tier"
                ],

            "Corrected Formal S2 Product ID":
                best80["S2 Product ID"],

            "Corrected Formal Delta Hours":
                best80[
                    "S2 Delta Hours From MethaneAIR"
                ],

            "Corrected Formal Clear Over Requested":
                best80[
                    "Corrected Clear Over Requested Fraction"
                ],

            "Corrected Formal Coverage Fraction":
                best80[
                    "Corrected Coverage Fraction"
                ],

            "Corrected Formal Clear Among Covered":
                best80[
                    "Corrected Clear Among Covered Fraction"
                ],
        })

    if bestsep is None:
        out.update({
            "Separate-Gate Diagnostic Result":
                "NO_SEPARATE_GATE_PASS_WITHIN_7D",

            "Separate-Gate Best Product ID":
                pd.NA,

            "Separate-Gate Best Delta Hours":
                pd.NA,

            "Separate-Gate Best Coverage Fraction":
                pd.NA,

            "Separate-Gate Best Clear Among Covered":
                pd.NA,
        })

    else:
        out.update({
            "Separate-Gate Diagnostic Result":
                bestsep[
                    "Corrected S2 Temporal Tier"
                ],

            "Separate-Gate Best Product ID":
                bestsep["S2 Product ID"],

            "Separate-Gate Best Delta Hours":
                bestsep[
                    "S2 Delta Hours From MethaneAIR"
                ],

            "Separate-Gate Best Coverage Fraction":
                bestsep[
                    "Corrected Coverage Fraction"
                ],

            "Separate-Gate Best Clear Among Covered":
                bestsep[
                    "Corrected Clear Among Covered Fraction"
                ],
        })

    if best75 is None:
        out.update({
            "Corrected QA75 Sensitivity Result":
                "NO_QA75_S2_WITHIN_7D",

            "Corrected QA75 Best Product ID":
                pd.NA,

            "Corrected QA75 Best Delta Hours":
                pd.NA,

            "Corrected QA75 Best Clear Over Requested":
                pd.NA,
        })

    else:
        out.update({
            "Corrected QA75 Sensitivity Result":
                best75[
                    "Corrected S2 Temporal Tier"
                ],

            "Corrected QA75 Best Product ID":
                best75[
                    "S2 Product ID"
                ],

            "Corrected QA75 Best Delta Hours":
                best75[
                    "S2 Delta Hours From MethaneAIR"
                ],

            "Corrected QA75 Best Clear Over Requested":
                best75[
                    "Corrected Clear Over Requested Fraction"
                ],
        })

    flight_summary_rows.append(out)


flight_summary = pd.DataFrame(
    flight_summary_rows
)


# ============================================================
# UNIQUE S2 CONTROL SUMMARY
#
# Multiple MethaneAIR flights supporting one Sentinel-2 product
# still produce ONE model sample.
# ============================================================

post_corrected = corrected[
    corrected["S2 Post Positive"]
    .astype(str)
    .str.lower()
    .eq("true")
].copy()

control_rows = []

for (
    parent_num,
    site,
    product_id,
), g in post_corrected.groupby(
    [
        "Pilot Parent Number",
        "Site",
        "S2 Product ID",
    ],
    dropna=False,
):

    flight_ids = sorted(
        set(
            g[
                "MethaneAIR Flight ID"
            ]
            .astype(str)
            .tolist()
        )
    )

    control_rows.append({
        "Pilot Parent Number":
            parent_num,

        "Site":
            site,

        "S2 Product ID":
            product_id,

        "S2 Datetime UTC":
            g.iloc[0][
                "S2 Datetime UTC"
            ],

        "Supporting MethaneAIR Flight Count":
            len(flight_ids),

        "Supporting MethaneAIR Flight IDs":
            " | ".join(flight_ids),

        "Minimum Abs Delta Hours To Supporting Flight":
            pd.to_numeric(
                g["S2 Abs Delta Hours"],
                errors="coerce",
            ).min(),

        "Best MethaneAIR Source Valid Fraction":
            pd.to_numeric(
                g[
                    "MethaneAIR Source Valid Fraction"
                ],
                errors="coerce",
            ).max(),

        "Best MethaneAIR Background Valid Fraction":
            pd.to_numeric(
                g[
                    "MethaneAIR Background Valid Fraction"
                ],
                errors="coerce",
            ).max(),

        "Corrected Coverage Fraction":
            g.iloc[0][
                "Corrected Coverage Fraction"
            ],

        "Corrected Clear Among Covered Fraction":
            g.iloc[0][
                "Corrected Clear Among Covered Fraction"
            ],

        "Corrected Clear Over Requested Fraction":
            g.iloc[0][
                "Corrected Clear Over Requested Fraction"
            ],

        "Corrected Masked Fraction":
            g.iloc[0][
                "Corrected Masked Fraction"
            ],

        "Corrected Existing QA80 Pass":
            g.iloc[0][
                "Corrected Existing QA80 Pass"
            ],

        "Corrected Separate-Gate Pass":
            g.iloc[0][
                "Corrected Separate-Gate Pass"
            ],

        "Corrected QA75 Sensitivity Pass":
            g.iloc[0][
                "Corrected QA75 Sensitivity Pass"
            ],
    })


controls = pd.DataFrame(
    control_rows
)


# ============================================================
# INTERNAL CONSISTENCY AUDIT
# ============================================================

bad = corrected[
    corrected[
        "Corrected QA Impossible Fraction Flag"
    ].eq(True)
]

if len(bad):
    print(
        "\nWARNING: corrected QA still produced "
        f"{len(bad)} impossible rows."
    )
else:
    print(
        "\n✅ Corrected QA consistency:"
        " no fractions >1 and class partition closes."
    )


# ============================================================
# SAVE
# ============================================================

ROW_OUT = (
    OUTDIR
    / "01_corrected_overpass_qa.csv"
)

FLIGHT_OUT = (
    OUTDIR
    / "02_corrected_flight_summary.csv"
)

CONTROL_OUT = (
    OUTDIR
    / "03_unique_s2_control_summary.csv"
)

CACHE_OUT = (
    OUTDIR
    / "04_local_lab_cache_audit.csv"
)

XLSX_OUT = (
    OUTDIR
    / "05_corrected_s2_qa_v7.xlsx"
)

corrected.to_csv(
    ROW_OUT,
    index=False,
    encoding="utf-8-sig",
)

flight_summary.to_csv(
    FLIGHT_OUT,
    index=False,
    encoding="utf-8-sig",
)

controls.to_csv(
    CONTROL_OUT,
    index=False,
    encoding="utf-8-sig",
)

cache_df.to_csv(
    CACHE_OUT,
    index=False,
    encoding="utf-8-sig",
)

with pd.ExcelWriter(
    XLSX_OUT,
    engine="openpyxl",
) as writer:

    flight_summary.to_excel(
        writer,
        sheet_name="Flight_Summary",
        index=False,
    )

    controls.to_excel(
        writer,
        sheet_name="Unique_S2_Controls",
        index=False,
    )

    corrected.to_excel(
        writer,
        sheet_name="Corrected_Overpasses",
        index=False,
    )

    cache_df.to_excel(
        writer,
        sheet_name="Local_Lab_Cache",
        index=False,
    )


# ============================================================
# DISPLAY THE PRODUCT THAT SUPPORTED M06/M07/M08/M09
# ============================================================

focus = controls[
    controls[
        "Supporting MethaneAIR Flight Count"
    ]
    >= 2
].copy()


print("\n" + "=" * 122)
print("CORRECTED FORMAL QA80 RESULT")
print("=" * 122)

print(
    flight_summary[
        "Corrected Formal Result"
    ].value_counts(
        dropna=False
    )
)


print("\nSEPARATE COVERAGE/CLEAR GATE DIAGNOSTIC:")
print(
    flight_summary[
        "Separate-Gate Diagnostic Result"
    ].value_counts(
        dropna=False
    )
)


print("\nCORRECTED QA75 SENSITIVITY:")
print(
    flight_summary[
        "Corrected QA75 Sensitivity Result"
    ].value_counts(
        dropna=False
    )
)


print("\n" + "=" * 122)
print("MULTI-FLIGHT → UNIQUE S2 CONTROL")
print("=" * 122)

if len(focus):

    cols = [
        "Pilot Parent Number",
        "Site",
        "S2 Product ID",
        "S2 Datetime UTC",
        "Supporting MethaneAIR Flight Count",
        "Supporting MethaneAIR Flight IDs",
        "Minimum Abs Delta Hours To Supporting Flight",
        "Corrected Coverage Fraction",
        "Corrected Clear Among Covered Fraction",
        "Corrected Clear Over Requested Fraction",
        "Corrected Masked Fraction",
        "Corrected Existing QA80 Pass",
        "Corrected Separate-Gate Pass",
        "Corrected QA75 Sensitivity Pass",
    ]

    print(
        focus[cols]
        .to_string(index=False)
    )

else:
    print(
        "No S2 product is shared by multiple "
        "MethaneAIR flights."
    )


print("\n" + "=" * 122)
print("FLIGHT DETAILS")
print("=" * 122)

cols = [
    "Pilot Parent Number",
    "Site",
    "MethaneAIR Flight ID",
    "MethaneAIR Source Valid Fraction",
    "MethaneAIR Background Valid Fraction",
    "MethaneAIR Source Minus Background ppb",
    "Corrected Formal Result",
    "Separate-Gate Diagnostic Result",
    "Corrected QA75 Sensitivity Result",
]

print(
    flight_summary[
        cols
    ]
    .sort_values(
        [
            "Pilot Parent Number",
            "MethaneAIR Flight ID",
        ]
    )
    .to_string(index=False)
)


print("\nLOCAL/LAB CACHE AUDIT:")
print(
    cache_df.to_string(
        index=False
    )
)


print("\nOUTPUTS:")
print(ROW_OUT)
print(FLIGHT_OUT)
print(CONTROL_OUT)
print(CACHE_OUT)
print(XLSX_OUT)

print("\n✅ all S2 QA fractions now use one common SCL grid")
print("✅ all reducer inputs are unweighted and internally comparable")
print("✅ explicit SCL projection + 20 m scale")
print("✅ QA75 result is recalculated; old v6 QA75 result is superseded")
print("✅ multiple MethaneAIR flights sharing one S2 product count as ONE control")
print("✅ Mac and Lab metadata caches audited concurrently")
print("✅ Lab SMB is not recursively scanned")
