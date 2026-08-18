#!/usr/bin/env python3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import subprocess
import numpy as np
import pandas as pd
import ee

# ============================================================
# CONFIG
# ============================================================

HOME = Path.home()
PROJECT = HOME / "methane_release_project"

INPUT = (
    PROJECT
    / "candidate_negative_validation"
    / "methaneair_coverage_first_v5"
    / "02_methaneair_flight_level_inventory.csv"
)

OUTDIR = (
    PROJECT
    / "candidate_negative_validation"
    / "methaneair_b5_s2_sensitivity_v6"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

EE_PROJECT = "methane-release-gee"

MAIR_L4 = (
    "projects/edf-methanesat-ee/"
    "assets/mair/L4point"
)

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

# MethaneFuse-sized local QA region
HALF_PATCH_M = 240.0
S2_SCALE_M = 20

# Keep the primary image-quality standard unchanged.
PRIMARY_CLEAR_THRESHOLD = 0.80

# Sensitivity only; never upgrades a sample into the strict benchmark.
SENSITIVITY_CLEAR_THRESHOLD = 0.75

# Temporal tiers relative to MethaneAIR flight midpoint.
STRICT_HOURS = 72.0
SECONDARY_HOURS = 120.0
EXPLORATORY_HOURS = 168.0

# Query the whole +/- 7 d range once.
SEARCH_HOURS = EXPLORATORY_HOURS

# Used only to report nearby plume contamination.
NEARBY_L4_RADIUS_M = 5000.0

WORKERS = 3

B_CLASS = (
    "B_HIGH_RES_NO_L4_DETECTION_BUT_NO_USABLE_ALIGNED_S2"
)


# ============================================================
# LOAD THE FIVE B FLIGHTS
# ============================================================

print("=" * 118)
print("METHANEAIR B-FLIGHT SENTINEL-2 SENSITIVITY V6")
print("=" * 118)

if not INPUT.exists():
    raise FileNotFoundError(f"Missing v5 flight inventory:\n{INPUT}")

df = pd.read_csv(INPUT, low_memory=False)

required = [
    "Pilot Parent Number",
    "Source Positive Record ID",
    "Site",
    "Parent Positive Date",
    "MethaneAIR Flight ID",
    "MethaneAIR Midpoint UTC",
    "Actual Days After Positive",
    "MethaneAIR Source Valid Fraction",
    "MethaneAIR Source XCH4 Median ppb",
    "MethaneAIR Background Valid Fraction",
    "MethaneAIR Background XCH4 Median ppb",
    "MethaneAIR Source Minus Background ppb",
    "MethaneAIR Same-Flight L4 Count",
    "MethaneAIR Nearby L4 Count <=5km",
    "Coverage-First Classification",
    "Latitude",
    "Longitude",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(
        "Missing columns from v5 flight inventory:\n"
        + "\n".join(f"  {c}" for c in missing)
    )

b = df[
    df["Coverage-First Classification"].eq(B_CLASS)
].copy()

if len(b) == 0:
    raise RuntimeError("No B-class flights found in v5 inventory.")

b["_mair_mid"] = pd.to_datetime(
    b["MethaneAIR Midpoint UTC"],
    errors="coerce",
    utc=True,
)
b["_positive_date"] = pd.to_datetime(
    b["Parent Positive Date"],
    errors="coerce",
    utc=True,
)

if b["_mair_mid"].isna().any() or b["_positive_date"].isna().any():
    raise RuntimeError("Invalid MethaneAIR/positive datetime in B flights.")

print(f"\nB flights: {len(b)}")

print(
    b[
        [
            "Pilot Parent Number",
            "Site",
            "MethaneAIR Flight ID",
            "MethaneAIR Midpoint UTC",
            "Actual Days After Positive",
            "MethaneAIR Source Valid Fraction",
            "MethaneAIR Background Valid Fraction",
            "MethaneAIR Source Minus Background ppb",
        ]
    ].to_string(index=False)
)


# ============================================================
# EARTH ENGINE
# ============================================================

print("\nInitializing Earth Engine...")
ee.Initialize(project=EE_PROJECT)
print("Earth Engine ready.")


# ============================================================
# S2 QA
# ============================================================

def s2_qa(image, region):
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

    shadow = scl.eq(3).rename("shadow")
    snow = scl.eq(11).rename("snow")

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

    sums = stack.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=S2_SCALE_M,
        bestEffort=True,
        maxPixels=1_000_000,
    ).getInfo()

    valid_result = scl.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=region,
        scale=S2_SCALE_M,
        bestEffort=True,
        maxPixels=1_000_000,
    ).getInfo()

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

    requested = float(requested_result.get("requested") or 0)
    valid = float(valid_result.get("SCL") or 0)
    clear_px = float(sums.get("clear") or 0)
    cloud_px = float(sums.get("cloud") or 0)
    shadow_px = float(sums.get("shadow") or 0)
    snow_px = float(sums.get("snow") or 0)
    invalid_px = float(sums.get("invalid") or 0)

    def div(a, d):
        return a / d if d > 0 else np.nan

    return {
        "valid_fraction": div(valid, requested),
        "clear_requested": div(clear_px, requested),
        "clear_valid": div(clear_px, valid),
        "cloud_requested": div(cloud_px, requested),
        "shadow_requested": div(shadow_px, requested),
        "snow_requested": div(snow_px, requested),
        "invalid_requested": div(invalid_px, requested),
        "masked_fraction": (
            1.0 - div(valid, requested)
            if requested > 0
            else np.nan
        ),
    }


# ============================================================
# TRUE NEAREST SAME-FLIGHT L4 DISTANCE
# ============================================================

def l4_distance_stats(flight_id, point):
    fc = (
        ee.FeatureCollection(MAIR_L4)
        .filter(ee.Filter.eq("flight_id", flight_id))
    )

    total = int(fc.size().getInfo())

    if total == 0:
        return {
            "same_flight_count": 0,
            "within_5km_count": 0,
            "nearest_distance_m": None,
        }

    with_dist = fc.map(
        lambda f: f.set(
            "_distance_m",
            f.geometry().distance(point, maxError=1),
        )
    )

    nearest = with_dist.aggregate_min("_distance_m").getInfo()

    nearby = int(
        fc.filterBounds(
            point.buffer(NEARBY_L4_RADIUS_M)
        ).size().getInfo()
    )

    return {
        "same_flight_count": total,
        "within_5km_count": nearby,
        "nearest_distance_m": (
            float(nearest)
            if nearest is not None
            else None
        ),
    }


# ============================================================
# OVERPASS DEDUP
# ============================================================

def dedup_overpasses(scene_df):
    if len(scene_df) == 0:
        return scene_df.copy()

    x = scene_df.copy()
    x["_dt"] = pd.to_datetime(
        x["S2 Datetime UTC"],
        errors="coerce",
        utc=True,
    )
    x = x.sort_values("_dt")

    groups = []
    current = []
    previous = None

    for idx, row in x.iterrows():
        dt = row["_dt"]

        if (
            previous is None
            or (dt - previous).total_seconds() <= 20 * 60
        ):
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]

        previous = dt

    if current:
        groups.append(current)

    out = []

    for inds in groups:
        g = x.loc[inds].copy()

        # Retain the tile with the best local patch QA.
        g = g.sort_values(
            [
                "S2 Clear Over Requested Fraction",
                "S2 Valid SCL Fraction",
            ],
            ascending=[False, False],
        )

        r = g.iloc[0].copy()
        r["S2 Overlapping Tile Count"] = len(g)
        out.append(r)

    return pd.DataFrame(out).drop(
        columns=["_dt"],
        errors="ignore",
    )


# ============================================================
# TEMPORAL TIER
# ============================================================

def tier_for_hours(abs_hours):
    if abs_hours <= STRICT_HOURS:
        return "STRICT_LE_72H"

    if abs_hours <= SECONDARY_HOURS:
        return "SECONDARY_72_TO_120H"

    if abs_hours <= EXPLORATORY_HOURS:
        return "EXPLORATORY_120_TO_168H"

    return "OUTSIDE_7D"


# ============================================================
# ONE METHANEAIR FLIGHT
# ============================================================

def process_flight(row):
    parent_num = int(row["Pilot Parent Number"])
    flight_id = str(row["MethaneAIR Flight ID"])
    site = row["Site"]

    lat = float(row["Latitude"])
    lon = float(row["Longitude"])

    mair_mid = pd.Timestamp(row["_mair_mid"])
    positive_date = pd.Timestamp(row["_positive_date"])

    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(HALF_PATCH_M).bounds()

    start = (
        mair_mid
        - pd.Timedelta(hours=SEARCH_HOURS)
    ).strftime("%Y-%m-%dT%H:%M:%S")

    end = (
        mair_mid
        + pd.Timedelta(hours=SEARCH_HOURS)
        + pd.Timedelta(seconds=1)
    ).strftime("%Y-%m-%dT%H:%M:%S")

    ic = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(point)
        .filterDate(start, end)
        .sort("system:time_start")
    )

    n = int(ic.size().getInfo())
    lst = ic.toList(n)

    scene_rows = []

    for i in range(n):
        image = ee.Image(lst.get(i))

        props = image.toDictionary(
            [
                "system:index",
                "system:time_start",
                "PRODUCT_ID",
                "MGRS_TILE",
                "CLOUDY_PIXEL_PERCENTAGE",
            ]
        ).getInfo()

        t_ms = props.get("system:time_start")

        if t_ms is None:
            continue

        s2_dt = pd.to_datetime(
            t_ms,
            unit="ms",
            utc=True,
        )

        delta_hours = (
            s2_dt - mair_mid
        ).total_seconds() / 3600.0

        abs_delta_hours = abs(delta_hours)

        # Safety: collection date filtering should already enforce this.
        if abs_delta_hours > SEARCH_HOURS + 0.1:
            continue

        # Require the S2 acquisition to be after the original positive
        # event date. This prevents a "temporal negative" from reaching
        # backward to the positive/pre-positive side.
        post_positive = (
            s2_dt
            >
            positive_date
        )

        qa = s2_qa(
            image,
            region,
        )

        clear = qa["clear_requested"]

        scene_rows.append({
            "Pilot Parent Number":
                parent_num,

            "Source Positive Record ID":
                row["Source Positive Record ID"],

            "Site":
                site,

            "Parent Positive Date":
                row["Parent Positive Date"],

            "MethaneAIR Flight ID":
                flight_id,

            "MethaneAIR Midpoint UTC":
                str(mair_mid),

            "MethaneAIR Source Valid Fraction":
                row["MethaneAIR Source Valid Fraction"],

            "MethaneAIR Background Valid Fraction":
                row["MethaneAIR Background Valid Fraction"],

            "MethaneAIR Source Minus Background ppb":
                row["MethaneAIR Source Minus Background ppb"],

            "S2 Datetime UTC":
                str(s2_dt),

            "S2 Delta Hours From MethaneAIR":
                delta_hours,

            "S2 Abs Delta Hours":
                abs_delta_hours,

            "S2 Temporal Tier":
                tier_for_hours(abs_delta_hours),

            "S2 Post Positive":
                bool(post_positive),

            "S2 Product ID":
                props.get("PRODUCT_ID"),

            "S2 System Index":
                props.get("system:index"),

            "S2 MGRS Tile":
                props.get("MGRS_TILE"),

            "S2 Scene Cloud Percentage":
                props.get("CLOUDY_PIXEL_PERCENTAGE"),

            "S2 Valid SCL Fraction":
                qa["valid_fraction"],

            "S2 Clear Over Requested Fraction":
                clear,

            "S2 Clear Among Valid Fraction":
                qa["clear_valid"],

            "S2 Cloud Over Requested Fraction":
                qa["cloud_requested"],

            "S2 Shadow Over Requested Fraction":
                qa["shadow_requested"],

            "S2 Snow Over Requested Fraction":
                qa["snow_requested"],

            "S2 Masked Fraction":
                qa["masked_fraction"],

            "S2 Primary QA Pass":
                (
                    bool(clear >= PRIMARY_CLEAR_THRESHOLD)
                    if not pd.isna(clear)
                    else False
                ),

            "S2 0.75 Sensitivity Pass":
                (
                    bool(clear >= SENSITIVITY_CLEAR_THRESHOLD)
                    if not pd.isna(clear)
                    else False
                ),
        })

    scenes = pd.DataFrame(scene_rows)

    if len(scenes):
        overpasses = dedup_overpasses(scenes)
    else:
        overpasses = pd.DataFrame()

    # Only post-positive acquisitions can be selected.
    if len(overpasses):
        selectable = overpasses[
            overpasses["S2 Post Positive"].eq(True)
        ].copy()
    else:
        selectable = pd.DataFrame()

    primary = (
        selectable[
            selectable[
                "S2 Primary QA Pass"
            ].eq(True)
        ].copy()
        if len(selectable)
        else pd.DataFrame()
    )

    sensitivity75 = (
        selectable[
            selectable[
                "S2 0.75 Sensitivity Pass"
            ].eq(True)
        ].copy()
        if len(selectable)
        else pd.DataFrame()
    )

    # Primary choice:
    # 1) closest temporal tier
    # 2) closest time
    # 3) highest clear fraction
    tier_rank = {
        "STRICT_LE_72H": 0,
        "SECONDARY_72_TO_120H": 1,
        "EXPLORATORY_120_TO_168H": 2,
    }

    def select_best(frame):
        if len(frame) == 0:
            return None

        x = frame.copy()
        x["_tier_rank"] = x[
            "S2 Temporal Tier"
        ].map(tier_rank).fillna(9)

        x = x.sort_values(
            [
                "_tier_rank",
                "S2 Abs Delta Hours",
                "S2 Clear Over Requested Fraction",
            ],
            ascending=[True, True, False],
        )

        return x.iloc[0]

    best_primary = select_best(primary)
    best_075 = select_best(sensitivity75)

    l4 = l4_distance_stats(
        flight_id,
        point,
    )

    base = {
        "Pilot Parent Number":
            parent_num,

        "Source Positive Record ID":
            row["Source Positive Record ID"],

        "Site":
            site,

        "Parent Positive Date":
            row["Parent Positive Date"],

        "MethaneAIR Flight ID":
            flight_id,

        "MethaneAIR Midpoint UTC":
            str(mair_mid),

        "Actual Days After Positive":
            row["Actual Days After Positive"],

        "MethaneAIR Source Valid Fraction":
            row["MethaneAIR Source Valid Fraction"],

        "MethaneAIR Background Valid Fraction":
            row["MethaneAIR Background Valid Fraction"],

        "MethaneAIR Source Minus Background ppb":
            row["MethaneAIR Source Minus Background ppb"],

        "MethaneAIR Same-Flight L4 Count Recomputed":
            l4["same_flight_count"],

        "MethaneAIR Nearby L4 Count <=5km Recomputed":
            l4["within_5km_count"],

        "MethaneAIR TRUE Nearest Same-Flight L4 Distance m":
            l4["nearest_distance_m"],

        "S2 Overpasses Within 7d":
            len(overpasses),

        "S2 Post-Positive Overpasses Within 7d":
            len(selectable),

        "S2 Primary QA-Pass Overpasses Within 7d":
            len(primary),

        "S2 0.75 Sensitivity-Pass Overpasses Within 7d":
            len(sensitivity75),
    }

    if best_primary is None:
        base.update({
            "V6 Primary Result":
                "NO_QA80_S2_WITHIN_7D",

            "V6 Best S2 Temporal Tier":
                pd.NA,

            "V6 Best S2 Datetime UTC":
                pd.NA,

            "V6 Best S2 Delta Hours":
                pd.NA,

            "V6 Best S2 Clear Fraction":
                pd.NA,

            "V6 Best S2 Product ID":
                pd.NA,

            "V6 Strict Benchmark Eligible":
                False,

            "V6 Secondary Eligible":
                False,

            "V6 Exploratory Eligible":
                False,
        })

    else:
        tier = best_primary[
            "S2 Temporal Tier"
        ]

        base.update({
            "V6 Primary Result":
                (
                    "STRICT_MATCH"
                    if tier == "STRICT_LE_72H"
                    else
                    (
                        "SECONDARY_MATCH"
                        if tier == "SECONDARY_72_TO_120H"
                        else
                        "EXPLORATORY_MATCH"
                    )
                ),

            "V6 Best S2 Temporal Tier":
                tier,

            "V6 Best S2 Datetime UTC":
                best_primary[
                    "S2 Datetime UTC"
                ],

            "V6 Best S2 Delta Hours":
                best_primary[
                    "S2 Delta Hours From MethaneAIR"
                ],

            "V6 Best S2 Clear Fraction":
                best_primary[
                    "S2 Clear Over Requested Fraction"
                ],

            "V6 Best S2 Product ID":
                best_primary[
                    "S2 Product ID"
                ],

            "V6 Strict Benchmark Eligible":
                tier == "STRICT_LE_72H",

            "V6 Secondary Eligible":
                tier == "SECONDARY_72_TO_120H",

            "V6 Exploratory Eligible":
                tier == "EXPLORATORY_120_TO_168H",
        })

    if best_075 is None:
        base.update({
            "V6 0.75 Sensitivity Result":
                "NO_QA75_S2_WITHIN_7D",

            "V6 Best QA75 Delta Hours":
                pd.NA,

            "V6 Best QA75 Clear Fraction":
                pd.NA,

            "V6 Best QA75 Product ID":
                pd.NA,
        })

    else:
        base.update({
            "V6 0.75 Sensitivity Result":
                best_075[
                    "S2 Temporal Tier"
                ],

            "V6 Best QA75 Delta Hours":
                best_075[
                    "S2 Delta Hours From MethaneAIR"
                ],

            "V6 Best QA75 Clear Fraction":
                best_075[
                    "S2 Clear Over Requested Fraction"
                ],

            "V6 Best QA75 Product ID":
                best_075[
                    "S2 Product ID"
                ],
        })

    return {
        "summary": base,
        "scenes": scenes,
        "overpasses": overpasses,
    }


# ============================================================
# LOCAL / LAB CACHE AUDIT
#
# This sensitivity run does NOT recursively re-scan the SMB.
# It reads the already-created local inventory/cache files while
# the GEE S2 search is running. This preserves the dual-source
# workflow without reintroducing SMB timeout risk.
# ============================================================

def cached_inventory_paths(label):
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

    paths = []
    seen = set()
    source_files = []

    for root in roots:
        for name in names:
            f = root / name

            if not f.exists():
                continue

            source_files.append(str(f))

            for line in f.read_text(
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

    return source_files, paths


def cache_audit(label):
    source_files, paths = cached_inventory_paths(label)

    mount_active = None

    if label == "LAB":
        try:
            r = subprocess.run(
                ["mount"],
                capture_output=True,
                text=True,
                timeout=20,
            )

            mount_active = (
                "/Volumes/engg-leung"
                in r.stdout
            )

        except Exception:
            mount_active = False

    relevant = []

    tokens = [
        "methaneair",
        "methane_air",
        "mair",
        "pcannon",
        "flight",
        "coverage",
        "retrieval",
    ]

    for path in paths:
        low = path.lower()

        if any(
            token in low
            for token in tokens
        ):
            relevant.append(path)

    return {
        "Origin":
            label,

        "Cache Source Files":
            " | ".join(source_files),

        "Unique Cached Paths":
            len(paths),

        "MethaneAIR-Relevant Cached Paths":
            len(relevant),

        "Lab SMB Mount Active":
            (
                mount_active
                if label == "LAB"
                else pd.NA
            ),

        "Audit Mode":
            "LOCAL_CACHE_READ_ONLY",
    }


# ============================================================
# RUN FIVE FLIGHTS + LOCAL/LAB CACHE AUDITS IN PARALLEL
# ============================================================

def run_s2_batch():
    summary_rows = []
    scene_frames = []
    overpass_frames = []

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as pool:

        futures = {
            pool.submit(
                process_flight,
                row,
            ):
                (
                    int(row["Pilot Parent Number"]),
                    str(row["MethaneAIR Flight ID"]),
                )
            for _, row in b.iterrows()
        }

        done = 0

        for future in as_completed(futures):
            parent_num, flight_id = futures[future]

            try:
                result = future.result()

                summary_rows.append(
                    result["summary"]
                )

                if len(result["scenes"]):
                    scene_frames.append(
                        result["scenes"]
                    )

                if len(result["overpasses"]):
                    overpass_frames.append(
                        result["overpasses"]
                    )

            except Exception as e:
                summary_rows.append({
                    "Pilot Parent Number":
                        parent_num,

                    "MethaneAIR Flight ID":
                        flight_id,

                    "V6 Primary Result":
                        "QUERY_ERROR",

                    "V6 Error":
                        repr(e),
                })

            done += 1
            print(
                f"[V6-S2] {done}/{len(b)} complete"
            )

    summary = pd.DataFrame(
        summary_rows
    )

    all_scenes = (
        pd.concat(
            scene_frames,
            ignore_index=True,
        )
        if scene_frames
        else pd.DataFrame()
    )

    all_overpasses = (
        pd.concat(
            overpass_frames,
            ignore_index=True,
        )
        if overpass_frames
        else pd.DataFrame()
    )

    return summary, all_scenes, all_overpasses


print("\nStarting parallel branches:")
print("  1. GEE S2 +/-7 day sensitivity for five B flights")
print("  2. Mac local metadata cache audit")
print("  3. Lab SMB metadata cache audit")

with ThreadPoolExecutor(
    max_workers=3
) as outer:

    futures = {
        outer.submit(run_s2_batch):
            "S2",

        outer.submit(cache_audit, "LOCAL"):
            "LOCAL",

        outer.submit(cache_audit, "LAB"):
            "LAB",
    }

    branch_results = {}

    for future in as_completed(futures):
        name = futures[future]

        try:
            branch_results[name] = future.result()
            print(f"[{name}] BRANCH COMPLETE")

        except Exception as e:
            branch_results[name] = {
                "Branch Error":
                    repr(e)
            }
            print(
                f"[{name}] BRANCH FAILED:",
                repr(e),
            )


s2_result = branch_results.get("S2")

if (
    not isinstance(s2_result, tuple)
    or len(s2_result) != 3
):
    raise RuntimeError(
        "S2 branch did not return expected outputs."
    )

summary, all_scenes, all_overpasses = s2_result

cache_rows = []

for label in ["LOCAL", "LAB"]:
    result = branch_results.get(label)

    if isinstance(result, dict):
        row = {"Origin": label}
        row.update(result)
        cache_rows.append(row)

cache_df = pd.DataFrame(cache_rows)


# ============================================================
# CONSERVATIVE METHANE EVIDENCE SUBGRADE
# ============================================================

def methane_subgrade(row):
    src = pd.to_numeric(
        pd.Series([
            row.get(
                "MethaneAIR Source Valid Fraction"
            )
        ]),
        errors="coerce",
    ).iloc[0]

    bg = pd.to_numeric(
        pd.Series([
            row.get(
                "MethaneAIR Background Valid Fraction"
            )
        ]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(src):
        return "UNKNOWN"

    if (
        src >= 0.80
        and
        not pd.isna(bg)
        and
        bg >= 0.80
    ):
        return "B_STRONG_SOURCE_AND_BACKGROUND"

    if src >= 0.80:
        return "B_SOURCE_VALID_BACKGROUND_WEAK"

    return "U_SOURCE_COVERAGE_WEAK"


if len(summary):
    summary[
        "Methane Evidence Subgrade"
    ] = summary.apply(
        methane_subgrade,
        axis=1,
    )


# ============================================================
# OUTPUTS
# ============================================================

SUMMARY_OUT = (
    OUTDIR
    / "01_b5_flight_s2_sensitivity_summary.csv"
)

SCENE_OUT = (
    OUTDIR
    / "02_all_s2_scenes_within_7d.csv"
)

OVERPASS_OUT = (
    OUTDIR
    / "03_unique_s2_overpasses_within_7d.csv"
)

XLSX_OUT = (
    OUTDIR
    / "04_b5_flight_s2_sensitivity_v6.xlsx"
)

CACHE_OUT = (
    OUTDIR
    / "05_local_lab_cache_audit.csv"
)

summary.to_csv(
    SUMMARY_OUT,
    index=False,
    encoding="utf-8-sig",
)

all_scenes.to_csv(
    SCENE_OUT,
    index=False,
    encoding="utf-8-sig",
)

all_overpasses.to_csv(
    OVERPASS_OUT,
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

    summary.to_excel(
        writer,
        sheet_name="Flight_Summary",
        index=False,
    )

    all_overpasses.to_excel(
        writer,
        sheet_name="S2_Overpasses",
        index=False,
    )

    all_scenes.to_excel(
        writer,
        sheet_name="S2_All_Scenes",
        index=False,
    )

    cache_df.to_excel(
        writer,
        sheet_name="Local_Lab_Cache",
        index=False,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 118)
print("V6 RESULT")
print("=" * 118)

print("\nPrimary QA80 result:")
print(
    summary[
        "V6 Primary Result"
    ].value_counts(
        dropna=False
    )
)

print("\nMethane evidence subgrade:")
print(
    summary[
        "Methane Evidence Subgrade"
    ].value_counts(
        dropna=False
    )
)

print(
    "\nStrict benchmark eligible:",
    int(
        summary[
            "V6 Strict Benchmark Eligible"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    ),
    "/",
    len(summary),
)

print(
    "Secondary eligible:",
    int(
        summary[
            "V6 Secondary Eligible"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    ),
    "/",
    len(summary),
)

print(
    "Exploratory eligible:",
    int(
        summary[
            "V6 Exploratory Eligible"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    ),
    "/",
    len(summary),
)

print("\n0.75 threshold sensitivity only:")
print(
    summary[
        "V6 0.75 Sensitivity Result"
    ].value_counts(
        dropna=False
    )
)

display_cols = [
    "Pilot Parent Number",
    "Site",
    "MethaneAIR Flight ID",
    "MethaneAIR Source Valid Fraction",
    "MethaneAIR Background Valid Fraction",
    "MethaneAIR Source Minus Background ppb",
    "MethaneAIR Same-Flight L4 Count Recomputed",
    "MethaneAIR Nearby L4 Count <=5km Recomputed",
    "MethaneAIR TRUE Nearest Same-Flight L4 Distance m",
    "Methane Evidence Subgrade",
    "V6 Primary Result",
    "V6 Best S2 Temporal Tier",
    "V6 Best S2 Delta Hours",
    "V6 Best S2 Clear Fraction",
    "V6 0.75 Sensitivity Result",
    "V6 Best QA75 Delta Hours",
    "V6 Best QA75 Clear Fraction",
]

display_cols = [
    c
    for c in display_cols
    if c in summary.columns
]

print("\n" + "=" * 118)
print("FLIGHT DETAILS")
print("=" * 118)

print(
    summary[
        display_cols
    ]
    .sort_values(
        [
            "Pilot Parent Number",
            "MethaneAIR Flight ID",
        ]
    )
    .to_string(index=False)
)

print("\nOUTPUTS:")
print(SUMMARY_OUT)
print(SCENE_OUT)
print(OVERPASS_OUT)
print(XLSX_OUT)
print(CACHE_OUT)

print("\nLocal/Lab cache audit:")
if len(cache_df):
    print(
        cache_df.to_string(
            index=False
        )
    )

print("\n✅ S2 QA threshold 0.80 was NOT relaxed")
print("✅ +/-72 h remains the strict benchmark rule")
print("✅ 72–120 h is secondary only")
print("✅ 120–168 h is exploratory only")
print("✅ 0.75 is reported as sensitivity only")
print("✅ true nearest same-flight L4 distance is recomputed")
