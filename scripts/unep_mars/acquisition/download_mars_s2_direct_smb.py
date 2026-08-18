import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import ee
import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

PROJECT = "methane-release-gee"

COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

LOCAL_MANIFEST = Path(
    "methanefuse_candidates/"
    "mars_s2_export/"
    "04_exact_t0_export_manifest.csv"
)

# ------------------------------------------------------------
# LAB SMB
# ------------------------------------------------------------

SHARE_MOUNT = Path(
    "/Volumes/engg-leung"
)

DORA_ROOT = (
    SHARE_MOUNT /
    "dora lin"
)

CLOUD_ROOT = (
    DORA_ROOT /
    "UNEP_MARS" /
    "Sentinel2"
)

T0_DIR = (
    CLOUD_ROOT /
    "t0"
)

REPORT_DIR = (
    CLOUD_ROOT /
    "reports"
)

MANIFEST_DIR = (
    CLOUD_ROOT /
    "manifests"
)

# Small local report backup only.
# No TIFFs are stored here.
LOCAL_REPORT_DIR = Path(
    "methanefuse_candidates/"
    "mars_s2_export/"
    "direct_download_reports"
)

# ------------------------------------------------------------
# Sentinel-2 bands
# ------------------------------------------------------------

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

# 512 × 20 m ~= 10.24 km source-centered crop
PATCH_PIXELS = 512

HALF_SIZE_M = (
    PATCH_PIXELS
    * SCALE_M
    / 2
)

# HTTP
CHUNK_SIZE = 1024 * 1024  # 1 MB
CONNECT_TIMEOUT = 60
READ_TIMEOUT = 600

DEFAULT_RETRIES = 5

MIN_VALID_BYTES = 1024


# ============================================================
# CUSTOM ERROR
# ============================================================

class ShareUnavailableError(RuntimeError):
    pass


# ============================================================
# SMB SAFETY
# ============================================================

def ensure_share_available():
    """
    Critical safety check.

    We intentionally require /Volumes/engg-leung
    to be an actual mounted filesystem.

    If the SMB share disappears, STOP instead of creating
    a local replacement directory under /Volumes.
    """

    if not SHARE_MOUNT.exists():
        raise ShareUnavailableError(
            f"SMB mount does not exist: {SHARE_MOUNT}"
        )

    if not os.path.ismount(SHARE_MOUNT):
        raise ShareUnavailableError(
            f"{SHARE_MOUNT} exists but is NOT mounted. "
            "Reconnect the laboratory SMB share before continuing."
        )

    if not DORA_ROOT.exists():
        raise ShareUnavailableError(
            f"Cannot find: {DORA_ROOT}"
        )

    return True


def prepare_directories():
    """
    Only create folders AFTER verifying the SMB is mounted.
    """

    ensure_share_available()

    T0_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LOCAL_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
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
        print("Authentication required.")
        ee.Authenticate()

        ee.Initialize(
            project=PROJECT
        )

    print("Earth Engine initialized.")


# ============================================================
# EXACT IMAGE
# ============================================================

def get_exact_image(product_id):
    """
    Manifest already contains exact validated PRODUCT_ID.

    We therefore query that exact L2A product directly.
    """

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

    return ee.Image(
        col.first()
    )


# ============================================================
# REGION
# ============================================================

def make_region(lon, lat):

    point = ee.Geometry.Point(
        [
            float(lon),
            float(lat),
        ]
    )

    return (
        point
        .buffer(HALF_SIZE_M)
        .bounds()
    )


# ============================================================
# TIFF VALIDATION
# ============================================================

def valid_tiff(path):
    """
    Lightweight validation without requiring rasterio.

    TIFF headers:
      little endian: II*\x00
      big endian:    MM\x00*
    """

    path = Path(path)

    if not path.exists():
        return False

    try:
        if path.stat().st_size < MIN_VALID_BYTES:
            return False

        with path.open("rb") as f:
            header = f.read(4)

        return header in (
            b"II*\x00",
            b"MM\x00*",
        )

    except OSError:
        return False


# ============================================================
# DOWNLOAD URL
# ============================================================

def make_download_url(
    image,
    lon,
    lat,
):
    """
    Request one multi-band GeoTIFF.
    """

    region = make_region(
        lon,
        lat
    )

    selected = image.select(
        BANDS
    )

    params = {
        "bands": BANDS,
        "region": region,
        "scale": SCALE_M,
        "format": "GEO_TIFF",
    }

    return selected.getDownloadURL(
        params
    )


# ============================================================
# STREAM DOWNLOAD DIRECTLY TO SMB
# ============================================================

def download_one(
    image,
    lon,
    lat,
    destination,
    retries,
):
    destination = Path(
        destination
    )

    part = Path(
        str(destination) + ".part"
    )

    # --------------------------------------------------------
    # Already complete
    # --------------------------------------------------------

    ensure_share_available()

    if valid_tiff(destination):
        return {
            "status": "SKIPPED_EXISTING",
            "bytes": destination.stat().st_size,
            "attempts": 0,
        }

    # Existing but invalid final file
    if destination.exists():

        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        bad = destination.with_name(
            destination.name
            + f".bad_{stamp}"
        )

        destination.rename(
            bad
        )

    # Old interrupted temp file
    if part.exists():
        try:
            part.unlink()
        except OSError:
            pass

    last_error = None

    # --------------------------------------------------------
    # Retry
    # --------------------------------------------------------

    for attempt in range(
        1,
        retries + 1
    ):

        try:
            ensure_share_available()

            # Generate a fresh Earth Engine URL
            # for every retry.
            url = make_download_url(
                image=image,
                lon=lon,
                lat=lat,
            )

            ensure_share_available()

            with requests.get(
                url,
                stream=True,
                timeout=(
                    CONNECT_TIMEOUT,
                    READ_TIMEOUT
                ),
            ) as response:

                response.raise_for_status()

                with part.open(
                    "wb"
                ) as f:

                    for chunk in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):

                        if not chunk:
                            continue

                        # Recheck SMB periodically.
                        ensure_share_available()

                        f.write(
                            chunk
                        )

            ensure_share_available()

            # ------------------------------------------------
            # Validate temp file BEFORE rename
            # ------------------------------------------------

            if not valid_tiff(part):
                raise RuntimeError(
                    "Downloaded file does not look like a valid TIFF."
                )

            size = part.stat().st_size

            # Atomic-ish rename on same SMB volume
            part.replace(
                destination
            )

            return {
                "status": "DOWNLOADED",
                "bytes": size,
                "attempts": attempt,
            }

        except ShareUnavailableError:
            # DO NOT continue blindly if SMB disappeared.
            raise

        except Exception as exc:

            last_error = repr(
                exc
            )

            print(
                f"    attempt {attempt}/{retries} failed:",
                last_error
            )

            try:
                if part.exists():
                    part.unlink()
            except OSError:
                pass

            # Verify SMB before retrying.
            ensure_share_available()

            if attempt < retries:

                wait = min(
                    5 * (2 ** (attempt - 1)),
                    60
                )

                print(
                    f"    retrying in {wait}s..."
                )

                time.sleep(
                    wait
                )

    return {
        "status": "FAILED",
        "bytes": 0,
        "attempts": retries,
        "error": last_error,
    }


# ============================================================
# REPORT
# ============================================================

def save_report(
    results,
    cloud_report,
    local_report,
):
    df = pd.DataFrame(
        results
    )

    # Always preserve a tiny local CSV backup.
    df.to_csv(
        local_report,
        index=False
    )

    # Cloud report only if SMB is still alive.
    ensure_share_available()

    df.to_csv(
        cloud_report,
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Directly download exact UNEP MARS "
            "Sentinel-2 L2A t0 crops from Earth Engine "
            "to the University of Alberta SMB share."
        )
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting manifest row (0-based).",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help=(
            "Number of rows to process. "
            "Use 0 for all remaining rows."
        ),
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Pause between successful downloads.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not LOCAL_MANIFEST.exists():
        raise FileNotFoundError(
            LOCAL_MANIFEST
        )

    # --------------------------------------------------------
    # Must verify network mount BEFORE mkdir
    # --------------------------------------------------------

    prepare_directories()

    # --------------------------------------------------------
    # Keep a manifest copy on cloud
    # --------------------------------------------------------

    cloud_manifest = (
        MANIFEST_DIR /
        LOCAL_MANIFEST.name
    )

    shutil.copy2(
        LOCAL_MANIFEST,
        cloud_manifest
    )

    # --------------------------------------------------------
    # Earth Engine
    # --------------------------------------------------------

    init_ee()

    df = pd.read_csv(
        LOCAL_MANIFEST
    )

    total = len(df)

    start = max(
        0,
        args.start
    )

    if start >= total:
        raise ValueError(
            f"--start {start} >= manifest rows {total}"
        )

    if args.count <= 0:
        end = total
    else:
        end = min(
            start + args.count,
            total
        )

    batch = df.iloc[
        start:end
    ].copy()

    # --------------------------------------------------------
    # Report names
    # --------------------------------------------------------

    report_name = (
        f"direct_batch_"
        f"{start:05d}_"
        f"{end - 1:05d}.csv"
    )

    cloud_report = (
        REPORT_DIR /
        report_name
    )

    local_report = (
        LOCAL_REPORT_DIR /
        report_name
    )

    print()
    print("=" * 80)
    print("MARS SENTINEL-2 DIRECT SMB DOWNLOAD")
    print("=" * 80)

    print(
        "Manifest rows :",
        total
    )

    print(
        "Batch         :",
        start,
        "→",
        end - 1
    )

    print(
        "Rows          :",
        len(batch)
    )

    print(
        "Destination   :",
        T0_DIR
    )

    print(
        "Scale         :",
        SCALE_M,
        "m"
    )

    print(
        "Bands         :",
        len(BANDS)
    )

    print()

    results = []

    downloaded = 0
    skipped = 0
    failed = 0

    # ========================================================
    # LOOP
    # ========================================================

    for local_position, (_, row) in enumerate(
        batch.iterrows(),
        start=1
    ):

        manifest_index = (
            start
            + local_position
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

        if (
            "output_filename"
            in row.index
            and pd.notna(
                row["output_filename"]
            )
        ):
            filename = str(
                row["output_filename"]
            )
        else:
            filename = (
                f"{export_id}__t0.tif"
            )

        destination = (
            T0_DIR /
            filename
        )

        print("-" * 80)

        print(
            f"[{local_position}/{len(batch)}]"
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

        print(
            "Output:",
            destination
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

            "output_filename":
                filename,

            "output_path":
                str(destination),

            "status":
                "",

            "bytes":
                0,

            "attempts":
                0,

            "error":
                "",
        }

        try:
            ensure_share_available()

            # -----------------------------------------------
            # Existing valid TIFF -> no EE work needed
            # -----------------------------------------------

            if valid_tiff(
                destination
            ):

                size = (
                    destination
                    .stat()
                    .st_size
                )

                rec["status"] = (
                    "SKIPPED_EXISTING"
                )

                rec["bytes"] = size

                skipped += 1

                print(
                    "SKIPPED_EXISTING",
                    f"({size / 1024 / 1024:.2f} MB)"
                )

            else:

                print(
                    "Resolving exact product..."
                )

                image = get_exact_image(
                    product_id
                )

                print(
                    "Downloading directly to SMB..."
                )

                result = download_one(
                    image=image,
                    lon=lon,
                    lat=lat,
                    destination=destination,
                    retries=args.retries,
                )

                rec.update(
                    result
                )

                if (
                    result["status"]
                    == "DOWNLOADED"
                ):

                    downloaded += 1

                    print(
                        "DOWNLOADED",
                        f"({result['bytes'] / 1024 / 1024:.2f} MB)",
                        f"attempt={result['attempts']}",
                    )

                elif (
                    result["status"]
                    == "SKIPPED_EXISTING"
                ):

                    skipped += 1

                    print(
                        "SKIPPED_EXISTING"
                    )

                else:

                    failed += 1

                    print(
                        "FAILED:",
                        result.get(
                            "error",
                            ""
                        )
                    )

            results.append(
                rec
            )

            # -----------------------------------------------
            # Save progress after every row
            # -----------------------------------------------

            save_report(
                results,
                cloud_report,
                local_report
            )

            time.sleep(
                args.sleep
            )

        except ShareUnavailableError as exc:

            rec["status"] = (
                "SMB_DISCONNECTED"
            )

            rec["error"] = repr(
                exc
            )

            results.append(
                rec
            )

            # Local report should still survive.
            pd.DataFrame(
                results
            ).to_csv(
                local_report,
                index=False
            )

            print()
            print("=" * 80)
            print("SMB SHARE DISCONNECTED")
            print("=" * 80)

            print(
                exc
            )

            print()
            print(
                "Download stopped intentionally."
            )

            print(
                "Reconnect engg-leung and "
                "run the SAME command again."
            )

            print(
                "Existing valid TIFF files will be skipped."
            )

            sys.exit(
                2
            )

        except Exception as exc:

            failed += 1

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

            try:
                save_report(
                    results,
                    cloud_report,
                    local_report
                )

            except Exception:
                pd.DataFrame(
                    results
                ).to_csv(
                    local_report,
                    index=False
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    out = pd.DataFrame(
        results
    )

    save_report(
        results,
        cloud_report,
        local_report
    )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        "Rows processed :",
        len(out)
    )

    print(
        "Downloaded     :",
        downloaded
    )

    print(
        "Already existed:",
        skipped
    )

    print(
        "Failed         :",
        failed
    )

    if len(out):

        print()
        print("STATUS")

        print(
            out[
                "status"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

        successful = out[
            out["status"].isin(
                [
                    "DOWNLOADED",
                    "SKIPPED_EXISTING",
                ]
            )
        ]

        total_bytes = pd.to_numeric(
            successful["bytes"],
            errors="coerce"
        ).fillna(0).sum()

        print()
        print(
            "Valid TIFFs this batch:",
            len(successful)
        )

        print(
            "Total size:",
            f"{total_bytes / 1024**3:.3f} GiB"
        )

    print()
    print(
        "Cloud data:"
    )
    print(
        T0_DIR
    )

    print()
    print(
        "Cloud report:"
    )
    print(
        cloud_report
    )

    print()
    print(
        "Local report backup:"
    )
    print(
        local_report
    )


if __name__ == "__main__":
    main()
