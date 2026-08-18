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

V7_FLIGHTS = (
    PROJECT
    / "candidate_negative_validation"
    / "methaneair_s2_corrected_qa_v7"
    / "02_corrected_flight_summary.csv"
)

OUTDIR = (
    PROJECT
    / "candidate_negative_validation"
    / "methaneair_l3_corrected_coverage_v8"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

EE_PROJECT = "methane-release-gee"

MAIR_L3 = (
    "projects/edf-methanesat-ee/"
    "assets/mair/L3concentration"
)

MAIR_L4 = (
    "projects/edf-methanesat-ee/"
    "assets/mair/L4point"
)

B_CLASS = (
    "B_HIGH_RES_NO_L4_DETECTION_BUT_NO_USABLE_ALIGNED_S2"
)

PATCH_HALF_M = 240.0

BG_INNER_M = 800.0
BG_OUTER_M = 2000.0

MAIR_SCALE_M = 10.2

MIN_SOURCE_COVERAGE = 0.80
STRONG_BACKGROUND_COVERAGE = 0.80

L4_RADIUS_M = 5000.0

WORKERS = 3


# ============================================================
# LOAD THE FIVE B FLIGHTS + V7 S2 RESULTS
# ============================================================

print("=" * 120)
print("METHANEAIR L3 CORRECTED-GRID COVERAGE V8")
print("=" * 120)

for p in [V5_FLIGHTS, V7_FLIGHTS]:
    if not p.exists():
        raise FileNotFoundError(p)

v5 = pd.read_csv(
    V5_FLIGHTS,
    low_memory=False,
)

v7 = pd.read_csv(
    V7_FLIGHTS,
    low_memory=False,
)

b = v5[
    v5[
        "Coverage-First Classification"
    ].eq(B_CLASS)
].copy()

if len(b) != 5:
    print(
        f"WARNING: expected 5 B flights, found {len(b)}"
    )

needed = [
    "Pilot Parent Number",
    "Source Positive Record ID",
    "Site",
    "Latitude",
    "Longitude",
    "Parent Positive Date",
    "MethaneAIR Flight ID",
    "MethaneAIR Midpoint UTC",
]

missing = [
    c for c in needed
    if c not in b.columns
]

if missing:
    raise RuntimeError(
        "Missing v5 columns:\n"
        + "\n".join(missing)
    )

v7_keep = [
    "Pilot Parent Number",
    "MethaneAIR Flight ID",
    "Corrected Formal Result",
    "Corrected Formal S2 Product ID",
    "Corrected Formal Delta Hours",
    "Corrected Formal Coverage Fraction",
    "Corrected Formal Clear Among Covered",
    "Corrected Formal Clear Over Requested",
]

v7_keep = [
    c for c in v7_keep
    if c in v7.columns
]

b = b.merge(
    v7[v7_keep],
    on=[
        "Pilot Parent Number",
        "MethaneAIR Flight ID",
    ],
    how="left",
    validate="one_to_one",
)


# ============================================================
# EARTH ENGINE
# ============================================================

print("\nInitializing Earth Engine...")
ee.Initialize(project=EE_PROJECT)
print("Earth Engine ready.")


# ============================================================
# CORRECTED XCH4 COVERAGE
#
# All numerator / denominator bands are derived from the SAME
# XCH4 raster, same projection, same mask expansion and same
# unweighted reducer.
# ============================================================

def xch4_region_stats(
    image,
    region,
):
    xch4 = image.select("XCH4")
    proj = xch4.projection()

    sentinel = -999999.0

    filled = xch4.unmask(
        value=sentinel,
        sameFootprint=False,
    ).rename("xch4_filled")

    requested = (
        filled.multiply(0)
        .add(1)
        .rename("requested")
    )

    covered = (
        filled.neq(sentinel)
        .rename("covered")
    )

    stack = (
        requested
        .addBands(covered)
    )

    counts = stack.reduceRegion(
        reducer=ee.Reducer.sum().unweighted(),
        geometry=region,
        crs=proj,
        scale=MAIR_SCALE_M,
        bestEffort=False,
        maxPixels=5_000_000,
    ).getInfo()

    requested_n = float(
        counts.get("requested")
        or 0
    )

    covered_n = float(
        counts.get("covered")
        or 0
    )

    coverage = (
        covered_n / requested_n
        if requested_n > 0
        else np.nan
    )

    median_dict = xch4.reduceRegion(
        reducer=ee.Reducer.median().unweighted(),
        geometry=region,
        crs=proj,
        scale=MAIR_SCALE_M,
        bestEffort=False,
        maxPixels=5_000_000,
    ).getInfo()

    median = median_dict.get("XCH4")

    if median is not None:
        median = float(median)

    impossible = (
        not pd.isna(coverage)
        and
        (
            coverage < -1e-9
            or
            coverage > 1.0000001
        )
    )

    return {
        "requested_pixels":
            requested_n,

        "covered_pixels":
            covered_n,

        "coverage_fraction":
            coverage,

        "xch4_median_ppb":
            median,

        "impossible_fraction":
            bool(impossible),
    }


# ============================================================
# L4 DISTANCE / COUNT
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
                str(flight_id),
            )
        )
    )

    total = int(
        fc.size().getInfo()
    )

    if total == 0:
        return {
            "same_flight_count": 0,
            "nearby_5km_count": 0,
            "nearest_distance_m": None,
        }

    with_dist = fc.map(
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
        with_dist
        .aggregate_min(
            "_distance_m"
        )
        .getInfo()
    )

    nearby = int(
        fc.filterBounds(
            point.buffer(
                L4_RADIUS_M
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
                float(nearest)
                if nearest is not None
                else None
            ),
    }


# ============================================================
# FIND THE L3 IMAGE FOR ONE FLIGHT
# ============================================================

def process_flight(
    row,
):
    flight_id = str(
        row[
            "MethaneAIR Flight ID"
        ]
    )

    lat = float(
        row["Latitude"]
    )

    lon = float(
        row["Longitude"]
    )

    point = ee.Geometry.Point(
        [
            lon,
            lat,
        ]
    )

    source_region = (
        point
        .buffer(
            PATCH_HALF_M
        )
        .bounds()
    )

    bg_region = (
        point
        .buffer(
            BG_OUTER_M
        )
        .difference(
            point.buffer(
                BG_INNER_M
            )
        )
    )

    ic = (
        ee.ImageCollection(
            MAIR_L3
        )
        .filter(
            ee.Filter.eq(
                "flight_id",
                flight_id,
            )
        )
        .filterBounds(
            source_region
        )
    )

    n = int(
        ic.size().getInfo()
    )

    if n == 0:
        return {
            "Pilot Parent Number":
                row[
                    "Pilot Parent Number"
                ],

            "Source Positive Record ID":
                row[
                    "Source Positive Record ID"
                ],

            "Site":
                row["Site"],

            "MethaneAIR Flight ID":
                flight_id,

            "Corrected L3 Image Count":
                0,

            "Corrected L3 Classification":
                "NO_L3_IMAGE_FOUND_FOR_FLIGHT",
        }

    images = ic.toList(n)

    candidates = []

    for i in range(n):
        image = ee.Image(
            images.get(i)
        )

        props = (
            image.toDictionary(
                [
                    "system:index",
                    "flight_id",
                    "target_id",
                    "time_coverage_start",
                    "time_coverage_end",
                ]
            )
            .getInfo()
        )

        src = xch4_region_stats(
            image,
            source_region,
        )

        bg = xch4_region_stats(
            image,
            bg_region,
        )

        delta = np.nan

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

        candidates.append({
            "system_index":
                props.get(
                    "system:index"
                ),

            "target_id":
                props.get(
                    "target_id"
                ),

            "source_requested":
                src[
                    "requested_pixels"
                ],

            "source_covered":
                src[
                    "covered_pixels"
                ],

            "source_fraction":
                src[
                    "coverage_fraction"
                ],

            "source_median":
                src[
                    "xch4_median_ppb"
                ],

            "background_requested":
                bg[
                    "requested_pixels"
                ],

            "background_covered":
                bg[
                    "covered_pixels"
                ],

            "background_fraction":
                bg[
                    "coverage_fraction"
                ],

            "background_median":
                bg[
                    "xch4_median_ppb"
                ],

            "delta":
                delta,

            "impossible":
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

    # If one flight has overlapping L3 assets, keep the one with
    # best source coverage, then best background coverage.
    best = sorted(
        candidates,
        key=lambda r: (
            -(
                r["source_fraction"]
                if pd.notna(
                    r["source_fraction"]
                )
                else -1
            ),
            -(
                r[
                    "background_fraction"
                ]
                if pd.notna(
                    r[
                        "background_fraction"
                    ]
                )
                else -1
            ),
        ),
    )[0]

    l4 = l4_stats(
        flight_id,
        point,
    )

    src_pass = (
        pd.notna(
            best[
                "source_fraction"
            ]
        )
        and
        best[
            "source_fraction"
        ]
        >=
        MIN_SOURCE_COVERAGE
    )

    bg_strong = (
        pd.notna(
            best[
                "background_fraction"
            ]
        )
        and
        best[
            "background_fraction"
        ]
        >=
        STRONG_BACKGROUND_COVERAGE
    )

    s2_strict = (
        str(
            row.get(
                "Corrected Formal Result"
            )
        )
        ==
        "STRICT_LE_72H"
    )

    if not src_pass:
        classification = (
            "U_CORRECTED_L3_SOURCE_COVERAGE_FAIL"
        )

    elif (
        l4[
            "nearby_5km_count"
        ]
        > 0
    ):
        classification = (
            "R_REJECT_L4_WITHIN_5KM"
        )

    elif (
        l4[
            "same_flight_count"
        ]
        > 0
        and
        s2_strict
    ):
        classification = (
            "B_STRICT_CORRECTED_L3_NO_L4_WITH_ALIGNED_S2"
        )

    elif (
        l4[
            "same_flight_count"
        ]
        > 0
    ):
        classification = (
            "B_CORRECTED_L3_NO_L4_BUT_NO_STRICT_S2"
        )

    else:
        classification = (
            "U_L4_AVAILABILITY_UNCERTAIN"
        )

    evidence_subgrade = (
        "B1_STRONG_SOURCE_AND_BACKGROUND"
        if (
            src_pass
            and
            bg_strong
        )
        else
        (
            "B2_SOURCE_VALID_BACKGROUND_WEAK"
            if src_pass
            else
            "U_SOURCE_COVERAGE_FAIL"
        )
    )

    return {
        "Pilot Parent Number":
            row[
                "Pilot Parent Number"
            ],

        "Source Positive Record ID":
            row[
                "Source Positive Record ID"
            ],

        "Site":
            row["Site"],

        "Latitude":
            lat,

        "Longitude":
            lon,

        "Parent Positive Date":
            row[
                "Parent Positive Date"
            ],

        "MethaneAIR Flight ID":
            flight_id,

        "MethaneAIR Midpoint UTC":
            row[
                "MethaneAIR Midpoint UTC"
            ],

        "Corrected L3 Image Count":
            n,

        "Corrected L3 System Index":
            best[
                "system_index"
            ],

        "Corrected Source Requested Pixels":
            best[
                "source_requested"
            ],

        "Corrected Source Covered Pixels":
            best[
                "source_covered"
            ],

        "Corrected Source Coverage Fraction":
            best[
                "source_fraction"
            ],

        "Corrected Source XCH4 Median ppb":
            best[
                "source_median"
            ],

        "Corrected Background Requested Pixels":
            best[
                "background_requested"
            ],

        "Corrected Background Covered Pixels":
            best[
                "background_covered"
            ],

        "Corrected Background Coverage Fraction":
            best[
                "background_fraction"
            ],

        "Corrected Background XCH4 Median ppb":
            best[
                "background_median"
            ],

        "Corrected Source Minus Background ppb":
            best[
                "delta"
            ],

        "Corrected L3 Impossible Fraction Flag":
            best[
                "impossible"
            ],

        "Corrected Source Coverage Pass":
            bool(
                src_pass
            ),

        "Corrected Background Strong":
            bool(
                bg_strong
            ),

        "Same-Flight L4 Count Recomputed":
            l4[
                "same_flight_count"
            ],

        "Nearby L4 Count <=5km Recomputed":
            l4[
                "nearby_5km_count"
            ],

        "True Nearest Same-Flight L4 Distance m":
            l4[
                "nearest_distance_m"
            ],

        "Corrected Formal Result":
            row.get(
                "Corrected Formal Result"
            ),

        "Corrected Formal S2 Product ID":
            row.get(
                "Corrected Formal S2 Product ID"
            ),

        "Corrected Formal Delta Hours":
            row.get(
                "Corrected Formal Delta Hours"
            ),

        "Corrected Formal S2 Coverage Fraction":
            row.get(
                "Corrected Formal Coverage Fraction"
            ),

        "Corrected Formal S2 Clear Among Covered":
            row.get(
                "Corrected Formal Clear Among Covered"
            ),

        "Corrected Formal S2 Clear Over Requested":
            row.get(
                "Corrected Formal Clear Over Requested"
            ),

        "Evidence Subgrade":
            evidence_subgrade,

        "Corrected L3 Classification":
            classification,
    }


# ============================================================
# CACHE-ONLY MAC / LAB AUDIT
# ============================================================

def cache_audit(
    label,
):
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

            sources.append(
                str(p)
            )

            for line in p.read_text(
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
                paths.append(line)

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

        "Cache Source Files":
            " | ".join(
                sources
            ),

        "Unique Cached Paths":
            len(paths),

        "Lab SMB Mounted":
            mounted,

        "Audit Mode":
            "CACHE_READ_ONLY_NO_RECURSIVE_SCAN",
    }


# ============================================================
# RUN GEE + MAC + LAB IN PARALLEL
# ============================================================

def run_gee():
    rows = []

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as pool:

        futures = [
            pool.submit(
                process_flight,
                row,
            )
            for _, row in (
                b.iterrows()
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

            print(
                f"[GEE-L3] "
                f"{done}/{len(b)}"
            )

    return pd.DataFrame(
        rows
    )


print("\nStarting parallel branches:")
print("  1. Corrected MethaneAIR L3 coverage")
print("  2. Mac cache audit")
print("  3. Lab SMB cache audit")

with ThreadPoolExecutor(
    max_workers=3
) as outer:

    futures = {
        outer.submit(
            run_gee
        ):
            "GEE",

        outer.submit(
            cache_audit,
            "LOCAL",
        ):
            "LOCAL",

        outer.submit(
            cache_audit,
            "LAB",
        ):
            "LAB",
    }

    results = {}

    for f in as_completed(
        futures
    ):
        name = futures[f]

        try:
            results[
                name
            ] = f.result()

            print(
                f"[{name}] BRANCH COMPLETE"
            )

        except Exception as e:
            results[
                name
            ] = {
                "Branch Error":
                    repr(e)
            }

            print(
                f"[{name}] BRANCH FAILED:",
                repr(e),
            )


corrected = results.get(
    "GEE"
)

if not isinstance(
    corrected,
    pd.DataFrame,
):
    raise RuntimeError(
        "Corrected L3 branch failed."
    )

cache_df = pd.DataFrame(
    [
        r
        for k, r in (
            results.items()
        )
        if (
            k in [
                "LOCAL",
                "LAB",
            ]
            and
            isinstance(
                r,
                dict,
            )
        )
    ]
)


# ============================================================
# UNIQUE STRICT S2 CONTROLS
# ============================================================

strict = corrected[
    corrected[
        "Corrected L3 Classification"
    ].eq(
        "B_STRICT_CORRECTED_L3_NO_L4_WITH_ALIGNED_S2"
    )
].copy()

control_rows = []

if len(strict):
    for (
        parent,
        site,
        product,
    ), g in strict.groupby(
        [
            "Pilot Parent Number",
            "Site",
            "Corrected Formal S2 Product ID",
        ],
        dropna=False,
    ):

        flight_ids = sorted(
            g[
                "MethaneAIR Flight ID"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        b1_count = int(
            g[
                "Evidence Subgrade"
            ]
            .eq(
                "B1_STRONG_SOURCE_AND_BACKGROUND"
            )
            .sum()
        )

        control_rows.append({
            "Pilot Parent Number":
                parent,

            "Site":
                site,

            "S2 Product ID":
                product,

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
                        "Corrected Formal Delta Hours"
                    ],
                    errors="coerce",
                ).abs().min(),

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

controls = pd.DataFrame(
    control_rows
)


# ============================================================
# OUTPUTS
# ============================================================

FLIGHT_OUT = (
    OUTDIR
    / "01_corrected_l3_flight_audit.csv"
)

CONTROL_OUT = (
    OUTDIR
    / "02_final_unique_strict_s2_controls.csv"
)

CACHE_OUT = (
    OUTDIR
    / "03_local_lab_cache_audit.csv"
)

XLSX_OUT = (
    OUTDIR
    / "04_corrected_l3_audit_v8.xlsx"
)

corrected.to_csv(
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

    corrected.to_excel(
        writer,
        sheet_name="Flight_Audit",
        index=False,
    )

    controls.to_excel(
        writer,
        sheet_name="Unique_Strict_Controls",
        index=False,
    )

    cache_df.to_excel(
        writer,
        sheet_name="Local_Lab_Cache",
        index=False,
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 120)
print("CORRECTED METHANEAIR L3 RESULT")
print("=" * 120)

print(
    corrected[
        "Corrected L3 Classification"
    ].value_counts(
        dropna=False
    )
)

print("\nEvidence subgrade:")
print(
    corrected[
        "Evidence Subgrade"
    ].value_counts(
        dropna=False
    )
)

print(
    "\nCorrected source coverage pass:",
    int(
        corrected[
            "Corrected Source Coverage Pass"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    ),
    "/",
    len(corrected),
)

print(
    "Corrected background strong:",
    int(
        corrected[
            "Corrected Background Strong"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    ),
    "/",
    len(corrected),
)

print(
    "Corrected strict B flights:",
    len(strict),
)

print(
    "Final unique strict S2 controls:",
    len(controls),
)


print("\n" + "=" * 120)
print("FLIGHT DETAILS")
print("=" * 120)

cols = [
    "Pilot Parent Number",
    "Site",
    "MethaneAIR Flight ID",
    "Corrected Source Coverage Fraction",
    "Corrected Background Coverage Fraction",
    "Corrected Source Minus Background ppb",
    "Same-Flight L4 Count Recomputed",
    "Nearby L4 Count <=5km Recomputed",
    "True Nearest Same-Flight L4 Distance m",
    "Corrected Formal Result",
    "Corrected Formal S2 Product ID",
    "Corrected Formal Delta Hours",
    "Evidence Subgrade",
    "Corrected L3 Classification",
]

print(
    corrected[
        cols
    ]
    .sort_values(
        [
            "Pilot Parent Number",
            "MethaneAIR Flight ID",
        ]
    )
    .to_string(
        index=False
    )
)


print("\n" + "=" * 120)
print("UNIQUE STRICT CONTROLS")
print("=" * 120)

if len(controls):
    print(
        controls.to_string(
            index=False
        )
    )
else:
    print(
        "No corrected strict controls survived."
    )


print("\nLOCAL/LAB CACHE AUDIT:")
print(
    cache_df.to_string(
        index=False
    )
)


print("\nOUTPUTS:")
print(FLIGHT_OUT)
print(CONTROL_OUT)
print(CACHE_OUT)
print(XLSX_OUT)

print("\n✅ MethaneAIR source/background coverage uses one common XCH4 grid")
print("✅ source/background coverage fractions are constrained to [0,1]")
print("✅ corrected v7 S2 QA is reused")
print("✅ repeated MethaneAIR flights supporting one S2 product deduplicate to one control")
print("✅ Mac + Lab caches audited concurrently")
print("✅ Lab SMB not recursively scanned")
