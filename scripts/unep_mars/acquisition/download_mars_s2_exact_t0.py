import argparse
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ee
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

CSV = Path(
    "methanefuse_candidates/mars_s2_candidates.csv"
)

REPORT = Path(
    "methanefuse_candidates/mars_s2_exact_t0_resolver.csv"
)

COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

PROJECT_DEFAULT = "methane-release-gee"

DRIVE_FOLDER = "MARS_S2_EXACT_T0"

# MethaneUnion / MethaneFuse-compatible Sentinel-2 bands
BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12",
]

# Raw source crop
# 512 pixels × 20 m = 10.24 km
PATCH_PIXELS = 512
SCALE_M = 20

# Avoid flooding Earth Engine task queue
MAX_PENDING = 150


# ============================================================
# EARTH ENGINE
# ============================================================

def init_ee(project):
    print("Initializing Earth Engine...")
    print("Project:", project)

    try:
        ee.Initialize(project=project)

    except Exception:
        print("Earth Engine authentication required.")
        ee.Authenticate()
        ee.Initialize(project=project)

    print("Earth Engine initialized successfully.")


# ============================================================
# PARSE MARS SENTINEL-2 PRODUCT
# ============================================================

def parse_mars_tile(tile):
    """
    Example MARS product:

    S2B_MSIL1C_20210505T100029_N0300_R122_T32RNR_20210505T134857

    Returns:

    mission     = S2B
    sensing     = 20210505T100029
    baseline    = 0300
    orbit       = 122
    mgrs        = 32RNR
    generation  = 20210505T134857
    """

    if pd.isna(tile):
        return None

    tile = str(tile).strip()

    pattern = (
        r"^(S2[ABC])_MSIL1C_"
        r"(\d{8}T\d{6})_"
        r"N(\d{4})_"
        r"R(\d{3})_"
        r"T([0-9A-Z]{5})_"
        r"(\d{8}T\d{6})$"
    )

    m = re.match(pattern, tile)

    if not m:
        return None

    return {
        "mission": m.group(1),
        "sensing": m.group(2),
        "baseline": m.group(3),
        "orbit": int(m.group(4)),
        "mgrs": m.group(5),
        "generation": m.group(6),
    }


def spacecraft_name(mission):
    return {
        "S2A": "Sentinel-2A",
        "S2B": "Sentinel-2B",
        "S2C": "Sentinel-2C",
    }.get(mission)


# ============================================================
# RESOLVE MARS L1C -> EXACT GEE L2A
# ============================================================

def resolve_l2a(info):
    """
    Resolve MARS Sentinel-2 L1C product to the corresponding
    Sentinel-2 L2A product in:

        COPERNICUS/S2_SR_HARMONIZED

    IMPORTANT:
    We DO NOT search ±3 minutes around the sensing timestamp.

    Example:

    MARS sensing:
        2021-05-05 10:00:29

    GEE system:time_start:
        2021-05-05 10:13:08

    Therefore filterDate around ±3 minutes incorrectly removes
    the correct acquisition.

    Resolver:

        whole UTC day
            ↓
        exact MGRS tile
            ↓
        exact spacecraft
            ↓
        exact relative orbit
            ↓
        PRODUCT_ID contains exact MARS sensing timestamp
    """

    target_dt = datetime.strptime(
        info["sensing"],
        "%Y%m%dT%H%M%S"
    ).replace(tzinfo=timezone.utc)

    day_start = target_dt.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    day_end = day_start + timedelta(days=1)

    # --------------------------------------------------------
    # Step 1: whole UTC day
    # --------------------------------------------------------

    col = (
        ee.ImageCollection(COLLECTION)
        .filterDate(
            day_start.strftime("%Y-%m-%d"),
            day_end.strftime("%Y-%m-%d"),
        )
    )

    # --------------------------------------------------------
    # Step 2: exact MGRS tile
    # --------------------------------------------------------

    col = col.filter(
        ee.Filter.eq(
            "MGRS_TILE",
            info["mgrs"],
        )
    )

    # --------------------------------------------------------
    # Step 3: spacecraft
    # --------------------------------------------------------

    sc = spacecraft_name(info["mission"])

    if sc is not None:
        col = col.filter(
            ee.Filter.eq(
                "SPACECRAFT_NAME",
                sc,
            )
        )

    # --------------------------------------------------------
    # Step 4: relative orbit
    # --------------------------------------------------------

    col = col.filter(
        ee.Filter.eq(
            "SENSING_ORBIT_NUMBER",
            info["orbit"],
        )
    )

    # Number after date/tile/orbit filtering
    base_count = col.size().getInfo()

    if base_count == 0:
        return {
            "image": None,
            "product_id": None,
            "candidate_count": 0,
            "match_method": None,
            "system_time": None,
            "time_delta_sec": None,
            "all_candidates": [],
        }

    # --------------------------------------------------------
    # Get candidate product IDs
    # --------------------------------------------------------

    candidate_ids = col.aggregate_array(
        "PRODUCT_ID"
    ).getInfo()

    # --------------------------------------------------------
    # Step 5:
    # exact MARS sensing timestamp inside L2A PRODUCT_ID
    # --------------------------------------------------------

    exact_col = col.filter(
        ee.Filter.stringContains(
            "PRODUCT_ID",
            info["sensing"],
        )
    )

    exact_count = exact_col.size().getInfo()

    if exact_count > 0:

        exact_ids = exact_col.aggregate_array(
            "PRODUCT_ID"
        ).getInfo()

        target_id = exact_ids[0]

        image = ee.Image(
            exact_col.filter(
                ee.Filter.eq(
                    "PRODUCT_ID",
                    target_id,
                )
            ).first()
        )

        system_time_ms = image.get(
            "system:time_start"
        ).getInfo()

        if system_time_ms is not None:
            system_dt = datetime.fromtimestamp(
                system_time_ms / 1000,
                tz=timezone.utc,
            )

            delta_sec = abs(
                (system_dt - target_dt).total_seconds()
            )

            system_time = system_dt.isoformat()

        else:
            delta_sec = None
            system_time = None

        return {
            "image": image,
            "product_id": target_id,
            "candidate_count": base_count,
            "match_method": "EXACT_PRODUCT_TIMESTAMP",
            "system_time": system_time,
            "time_delta_sec": delta_sec,
            "all_candidates": candidate_ids,
        }

    # ========================================================
    # FALLBACK
    #
    # This should be rare.
    #
    # If exact PRODUCT_ID timestamp is unavailable, choose
    # candidate whose system:time_start is closest to MARS
    # sensing time.
    # ========================================================

    candidate_info = []

    for pid in candidate_ids:

        img = ee.Image(
            col.filter(
                ee.Filter.eq(
                    "PRODUCT_ID",
                    pid,
                )
            ).first()
        )

        system_time_ms = img.get(
            "system:time_start"
        ).getInfo()

        if system_time_ms is None:
            continue

        system_dt = datetime.fromtimestamp(
            system_time_ms / 1000,
            tz=timezone.utc,
        )

        delta_sec = abs(
            (system_dt - target_dt).total_seconds()
        )

        candidate_info.append({
            "product_id": pid,
            "system_time_ms": system_time_ms,
            "system_time": system_dt.isoformat(),
            "delta_sec": delta_sec,
        })

    if not candidate_info:
        return {
            "image": None,
            "product_id": None,
            "candidate_count": base_count,
            "match_method": None,
            "system_time": None,
            "time_delta_sec": None,
            "all_candidates": candidate_ids,
        }

    chosen = min(
        candidate_info,
        key=lambda x: x["delta_sec"],
    )

    target_id = chosen["product_id"]

    image = ee.Image(
        col.filter(
            ee.Filter.eq(
                "PRODUCT_ID",
                target_id,
            )
        ).first()
    )

    return {
        "image": image,
        "product_id": target_id,
        "candidate_count": base_count,
        "match_method": "NEAREST_SYSTEM_TIME_FALLBACK",
        "system_time": chosen["system_time"],
        "time_delta_sec": chosen["delta_sec"],
        "all_candidates": candidate_ids,
    }


# ============================================================
# EXPORT
# ============================================================

def export_image(
    image,
    lon,
    lat,
    plume_id,
):
    """
    Export a 10.24 km x 10.24 km source-centered Sentinel-2
    crop at 20 m resolution.

    Final model crops can be generated later after QA.
    """

    point = ee.Geometry.Point(
        [
            float(lon),
            float(lat),
        ]
    )

    half_size_m = (
        PATCH_PIXELS
        * SCALE_M
        / 2
    )

    region = (
        point
        .buffer(half_size_m)
        .bounds()
    )

    name = (
        f"MARS_{plume_id}_S2_t0"
    )

    out = (
        image
        .select(BANDS)
        .clip(region)
    )

    task = ee.batch.Export.image.toDrive(
        image=out,
        description=name,
        folder=DRIVE_FOLDER,
        fileNamePrefix=name,
        region=region,
        scale=SCALE_M,
        maxPixels=1e9,
        fileFormat="GeoTIFF",
    )

    task.start()

    return task


# ============================================================
# TASK QUEUE
# ============================================================

def pending_tasks():
    n = 0

    for task in ee.batch.Task.list():

        state = task.status().get(
            "state"
        )

        if state in (
            "READY",
            "RUNNING",
        ):
            n += 1

    return n


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Resolve MARS Sentinel-2 L1C detections "
            "to exact Sentinel-2 L2A products."
        )
    )

    parser.add_argument(
        "--project",
        default=PROJECT_DEFAULT,
        help=(
            "Google Cloud project used by Earth Engine. "
            f"Default: {PROJECT_DEFAULT}"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help=(
            "Number of MARS rows to process. "
            "Use 0 for all rows."
        ),
    )

    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Actually submit Google Drive exports. "
            "Without this flag, resolver only."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # EE
    # --------------------------------------------------------

    init_ee(args.project)

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    if not CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {CSV}"
        )

    df = pd.read_csv(CSV)

    print("\nInput rows:", len(df))

    if args.limit > 0:
        df = df.head(
            args.limit
        ).copy()

    print("Rows this run:", len(df))

    print()
    print("=" * 80)
    print(
        "MARS SENTINEL-2 EXACT L1C -> L2A RESOLVER"
    )
    print("=" * 80)

    print("Collection     :", COLLECTION)
    print("Cloud project  :", args.project)
    print("Rows           :", len(df))
    print("Submit exports :", args.submit)
    print("Drive folder   :", DRIVE_FOLDER)

    results = []

    # --------------------------------------------------------
    # Loop
    # --------------------------------------------------------

    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        plume_id = str(
            row["id_plume"]
        )

        tile = row.get(
            "tile"
        )

        lat = row.get(
            "lat"
        )

        lon = row.get(
            "lon"
        )

        print()
        print("-" * 80)
        print(
            f"[{position}/{len(df)}]"
        )
        print("Plume:", plume_id)

        rec = {
            "id_plume": plume_id,
            "source_name": row.get(
                "source_name",
                "",
            ),
            "mars_tile": tile,
            "lat": lat,
            "lon": lon,
            "event_time": row.get(
                "event_time",
                "",
            ),
            "status": "",
            "mission": "",
            "mars_sensing_time": "",
            "mars_orbit": "",
            "mars_mgrs": "",
            "resolved_product_id": "",
            "candidate_count": 0,
            "match_method": "",
            "gee_system_time": "",
            "time_delta_sec": "",
            "task_id": "",
            "error": "",
        }

        # ----------------------------------------------------
        # Basic metadata validation
        # ----------------------------------------------------

        if pd.isna(lat) or pd.isna(lon):
            rec["status"] = (
                "MISSING_COORDINATES"
            )

            results.append(rec)

            print(
                "  MISSING_COORDINATES"
            )

            pd.DataFrame(
                results
            ).to_csv(
                REPORT,
                index=False,
            )

            continue

        # ----------------------------------------------------
        # Parse MARS L1C
        # ----------------------------------------------------

        parsed = parse_mars_tile(
            tile
        )

        if parsed is None:

            rec["status"] = (
                "BAD_MARS_TILE"
            )

            results.append(rec)

            print(
                "  BAD_MARS_TILE:",
                tile,
            )

            pd.DataFrame(
                results
            ).to_csv(
                REPORT,
                index=False,
            )

            continue

        rec["mission"] = (
            parsed["mission"]
        )

        rec["mars_sensing_time"] = (
            parsed["sensing"]
        )

        rec["mars_orbit"] = (
            parsed["orbit"]
        )

        rec["mars_mgrs"] = (
            parsed["mgrs"]
        )

        print(
            "  MARS L1C:",
            tile,
        )

        print(
            "  Parsed   :",
            parsed["mission"],
            parsed["sensing"],
            f"R{parsed['orbit']:03d}",
            f"T{parsed['mgrs']}",
        )

        # ----------------------------------------------------
        # Resolve exact L2A
        # ----------------------------------------------------

        try:

            resolved = resolve_l2a(
                parsed
            )

            rec["candidate_count"] = (
                resolved[
                    "candidate_count"
                ]
            )

            if resolved["image"] is None:

                rec["status"] = (
                    "NO_L2A_MATCH"
                )

                print(
                    "  RESULT   : "
                    "NO_L2A_MATCH"
                )

                results.append(rec)

                pd.DataFrame(
                    results
                ).to_csv(
                    REPORT,
                    index=False,
                )

                continue

            image = resolved["image"]

            product_id = resolved[
                "product_id"
            ]

            method = resolved[
                "match_method"
            ]

            delta_sec = resolved[
                "time_delta_sec"
            ]

            system_time = resolved[
                "system_time"
            ]

            rec[
                "resolved_product_id"
            ] = product_id

            rec[
                "match_method"
            ] = method

            rec[
                "gee_system_time"
            ] = system_time

            rec[
                "time_delta_sec"
            ] = delta_sec

            rec["status"] = (
                "RESOLVED"
            )

            print(
                "  L2A      :",
                product_id,
            )

            print(
                "  Method   :",
                method,
            )

            print(
                "  candidates:",
                resolved[
                    "candidate_count"
                ],
            )

            print(
                "  GEE time :",
                system_time,
            )

            print(
                "  Δtime sec:",
                delta_sec,
            )

            # ------------------------------------------------
            # Optional export
            # ------------------------------------------------

            if args.submit:

                while (
                    pending_tasks()
                    >= MAX_PENDING
                ):

                    print(
                        "  Earth Engine "
                        "task queue full."
                    )

                    print(
                        "  Waiting 60 s..."
                    )

                    time.sleep(60)

                task = export_image(
                    image=image,
                    lon=lon,
                    lat=lat,
                    plume_id=plume_id,
                )

                rec["task_id"] = (
                    task.id
                )

                rec["status"] = (
                    "SUBMITTED"
                )

                print(
                    "  SUBMITTED:",
                    task.id,
                )

        except Exception as exc:

            rec["status"] = (
                "ERROR"
            )

            rec["error"] = repr(
                exc
            )

            print(
                "  ERROR:",
                repr(exc),
            )

        results.append(rec)

        # ----------------------------------------------------
        # Save continuously
        # ----------------------------------------------------

        pd.DataFrame(
            results
        ).to_csv(
            REPORT,
            index=False,
        )

        time.sleep(0.05)

    # ========================================================
    # SUMMARY
    # ========================================================

    out = pd.DataFrame(
        results
    )

    out.to_csv(
        REPORT,
        index=False,
    )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if len(out) == 0:
        print("No rows processed.")
        return

    print()
    print("STATUS")
    print(
        out["status"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    resolved_mask = (
        out[
            "resolved_product_id"
        ]
        .fillna("")
        .astype(str)
        .str.len()
        .gt(0)
    )

    n_resolved = int(
        resolved_mask.sum()
    )

    print()
    print(
        "Exact L2A resolved:",
        n_resolved,
        "/",
        len(out),
    )

    if (
        "match_method"
        in out.columns
    ):

        print()
        print("MATCH METHODS")

        x = (
            out.loc[
                resolved_mask,
                "match_method"
            ]
            .value_counts(
                dropna=False
            )
        )

        if len(x):
            print(
                x.to_string()
            )
        else:
            print("None")

    # --------------------------------------------------------
    # Unique scene count
    # --------------------------------------------------------

    if n_resolved:

        unique_scenes = (
            out.loc[
                resolved_mask,
                "resolved_product_id"
            ]
            .nunique()
        )

        print()
        print(
            "Unique L2A scenes:",
            unique_scenes,
        )

        print(
            "Resolved plume rows:",
            n_resolved,
        )

        duplicated_rows = (
            n_resolved
            - unique_scenes
        )

        print(
            "Repeated scene references:",
            duplicated_rows,
        )

    # --------------------------------------------------------
    # Time delta
    # --------------------------------------------------------

    delta = pd.to_numeric(
        out["time_delta_sec"],
        errors="coerce",
    )

    delta = delta.dropna()

    if len(delta):

        print()
        print(
            "GEE system time vs "
            "MARS sensing time"
        )

        print(
            "Minimum delta sec:",
            delta.min(),
        )

        print(
            "Median delta sec :",
            delta.median(),
        )

        print(
            "Maximum delta sec:",
            delta.max(),
        )

    print()
    print("Report:")
    print(REPORT)

    if args.submit:
        print()
        print(
            "Google Drive folder:"
        )
        print(DRIVE_FOLDER)


if __name__ == "__main__":
    main()
