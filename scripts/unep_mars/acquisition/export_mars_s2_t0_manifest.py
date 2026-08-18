import argparse
import time
from pathlib import Path

import ee
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT = "methane-release-gee"

COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

MANIFEST = Path(
    "methanefuse_candidates/"
    "mars_s2_export/"
    "04_exact_t0_export_manifest.csv"
)

REPORT_DIR = Path(
    "methanefuse_candidates/"
    "mars_s2_export/"
    "submission_reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DRIVE_FOLDER = "MARS_S2_EXACT_T0"

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

SCALE_M = 20

# Large source-centered raw crop
# 512 × 20 m = 10.24 km
PATCH_PIXELS = 512
HALF_SIZE_M = (
    PATCH_PIXELS
    * SCALE_M
    / 2
)


# ============================================================
# EARTH ENGINE
# ============================================================

def init_ee():

    print("Initializing Earth Engine...")
    print("Project:", PROJECT)

    try:
        ee.Initialize(
            project=PROJECT
        )

    except Exception:

        ee.Authenticate()

        ee.Initialize(
            project=PROJECT
        )

    print(
        "Earth Engine initialized."
    )


# ============================================================
# GET IMAGE FROM EXACT PRODUCT ID
# ============================================================

def get_exact_image(product_id):

    col = (
        ee.ImageCollection(
            COLLECTION
        )
        .filter(
            ee.Filter.eq(
                "PRODUCT_ID",
                product_id
            )
        )
    )

    n = col.size().getInfo()

    if n == 0:
        return None

    return ee.Image(
        col.first()
    )


# ============================================================
# EXPORT REGION
# ============================================================

def make_region(lon, lat):

    point = ee.Geometry.Point(
        [
            float(lon),
            float(lat),
        ]
    )

    region = (
        point
        .buffer(
            HALF_SIZE_M
        )
        .bounds()
    )

    return region


# ============================================================
# SUBMIT ONE EXPORT
# ============================================================

def submit_export(
    image,
    lon,
    lat,
    export_id,
):

    region = make_region(
        lon,
        lat,
    )

    filename = (
        f"{export_id}__t0"
    )

    out = (
        image
        .select(BANDS)
        .clip(region)
    )

    task = (
        ee.batch.Export.image.toDrive(
            image=out,

            description=filename,

            folder=DRIVE_FOLDER,

            fileNamePrefix=filename,

            region=region,

            scale=SCALE_M,

            maxPixels=1e9,

            fileFormat="GeoTIFF",
        )
    )

    task.start()

    return task


# ============================================================
# EXISTING TASK DESCRIPTIONS
# ============================================================

def get_existing_tasks():

    existing = {}

    print(
        "Reading existing Earth "
        "Engine tasks..."
    )

    for task in ee.batch.Task.list():

        status = task.status()

        desc = status.get(
            "description"
        )

        state = status.get(
            "state"
        )

        if desc:
            existing[desc] = state

    print(
        "Existing task descriptions:",
        len(existing)
    )

    return existing


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help=(
            "Starting manifest row "
            "(0-based)."
        )
    )

    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help=(
            "Number of exports "
            "to process."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve products only; "
            "do not submit exports."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # EE
    # --------------------------------------------------------

    init_ee()

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    if not MANIFEST.exists():

        raise FileNotFoundError(
            MANIFEST
        )

    df = pd.read_csv(
        MANIFEST
    )

    total = len(df)

    print()
    print("=" * 80)
    print("MARS S2 t0 EXPORT")
    print("=" * 80)

    print(
        "Total manifest rows:",
        total
    )

    start = max(
        0,
        args.start
    )

    if args.count <= 0:
        end = total
    else:
        end = min(
            start + args.count,
            total
        )

    batch = (
        df.iloc[
            start:end
        ]
        .copy()
    )

    print(
        "Batch:",
        start,
        "→",
        end - 1
    )

    print(
        "Rows:",
        len(batch)
    )

    print(
        "Dry run:",
        args.dry_run
    )

    print(
        "Drive folder:",
        DRIVE_FOLDER
    )

    existing = (
        get_existing_tasks()
        if not args.dry_run
        else {}
    )

    results = []

    # --------------------------------------------------------
    # Loop
    # --------------------------------------------------------

    for local_i, (_, row) in enumerate(
        batch.iterrows(),
        start=1
    ):

        manifest_index = (
            start
            + local_i
            - 1
        )

        export_id = str(
            row["export_id"]
        )

        product_id = str(
            row[
                "resolved_product_id"
            ]
        )

        lat = float(
            row["lat"]
        )

        lon = float(
            row["lon"]
        )

        task_description = (
            f"{export_id}__t0"
        )

        print()
        print("-" * 80)

        print(
            f"[{local_i}/{len(batch)}]"
        )

        print(
            "Manifest row:",
            manifest_index
        )

        print(
            "Export ID:",
            export_id
        )

        print(
            "Product:",
            product_id
        )

        print(
            "Center:",
            lat,
            lon
        )

        rec = {
            "manifest_index":
                manifest_index,

            "export_id":
                export_id,

            "resolved_product_id":
                product_id,

            "lat":
                lat,

            "lon":
                lon,

            "status":
                "",

            "task_id":
                "",

            "task_state":
                "",

            "error":
                "",
        }

        # ----------------------------------------------------
        # Don't duplicate existing submitted tasks
        # ----------------------------------------------------

        if (
            not args.dry_run
            and task_description
            in existing
        ):

            state = existing[
                task_description
            ]

            rec["status"] = (
                "ALREADY_EXISTS"
            )

            rec["task_state"] = state

            print(
                "Already exists:",
                state
            )

            results.append(
                rec
            )

            continue

        # ----------------------------------------------------
        # Resolve exact product
        # ----------------------------------------------------

        try:

            image = get_exact_image(
                product_id
            )

            if image is None:

                rec["status"] = (
                    "PRODUCT_NOT_FOUND"
                )

                print(
                    "PRODUCT_NOT_FOUND"
                )

                results.append(
                    rec
                )

                continue

            print(
                "Exact product found."
            )

            # ------------------------------------------------
            # Dry run
            # ------------------------------------------------

            if args.dry_run:

                rec["status"] = (
                    "READY"
                )

                print(
                    "READY"
                )

            # ------------------------------------------------
            # Submit
            # ------------------------------------------------

            else:

                task = submit_export(
                    image=image,
                    lon=lon,
                    lat=lat,
                    export_id=export_id,
                )

                rec["status"] = (
                    "SUBMITTED"
                )

                rec["task_id"] = (
                    task.id
                )

                rec["task_state"] = (
                    "READY"
                )

                existing[
                    task_description
                ] = "READY"

                print(
                    "SUBMITTED:",
                    task.id
                )

            results.append(
                rec
            )

        except Exception as exc:

            rec["status"] = (
                "ERROR"
            )

            rec["error"] = repr(
                exc
            )

            results.append(
                rec
            )

            print(
                "ERROR:",
                repr(exc)
            )

        # Save continuously
        report = pd.DataFrame(
            results
        )

        report_path = (
            REPORT_DIR
            /
            (
                f"batch_{start:05d}_"
                f"{end-1:05d}.csv"
            )
        )

        report.to_csv(
            report_path,
            index=False
        )

        time.sleep(
            0.15
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    out = pd.DataFrame(
        results
    )

    report_path = (
        REPORT_DIR
        /
        (
            f"batch_{start:05d}_"
            f"{end-1:05d}.csv"
        )
    )

    out.to_csv(
        report_path,
        index=False
    )

    print()
    print("=" * 80)
    print("BATCH SUMMARY")
    print("=" * 80)

    print(
        "Manifest rows:",
        start,
        "→",
        end - 1
    )

    print(
        "Rows processed:",
        len(out)
    )

    if len(out):

        print()
        print(
            out[
                "status"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print()
    print(
        "Report:",
        report_path
    )

    if not args.dry_run:

        print(
            "Google Drive folder:",
            DRIVE_FOLDER
        )


if __name__ == "__main__":
    main()
