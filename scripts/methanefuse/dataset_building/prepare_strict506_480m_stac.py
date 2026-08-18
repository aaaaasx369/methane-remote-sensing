#!/usr/bin/env python3

from pathlib import Path
import argparse
import time
import requests
import numpy as np
import pandas as pd
import rasterio

from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as warp_transform


ROOT = Path("/project/6002520/yunjung1/MethaneFuse")

INPUT = ROOT / "data/custom/methaneair_full_strict506_source_manifest.csv"

OUTPUT_ROOT = (
    ROOT / "data/custom/methaneair_full_strict506_480m"
)

OUTPUT_CSV = (
    ROOT / "data/custom/methaneair_full_strict506_480m_eval.csv"
)

AUDIT_CSV = (
    ROOT / "data/custom/methaneair_full_strict506_480m_audit.csv"
)

STAC_BASE = (
    "https://earth-search.aws.element84.com/v1/"
    "collections/sentinel-2-l2a/items"
)

# Exact MethaneFuse Sentinel-2 12-band ordering.
BANDS = [
    ("coastal", "B01"),
    ("blue", "B02"),
    ("green", "B03"),
    ("red", "B04"),
    ("rededge1", "B05"),
    ("rededge2", "B06"),
    ("rededge3", "B07"),
    ("nir", "B08"),
    ("nir08", "B8A"),
    ("nir09", "B09"),
    ("swir16", "B11"),
    ("swir22", "B12"),
]

PATCH_PIXELS = 48
PATCH_METERS = 480.0
PIXEL_SIZE = PATCH_METERS / PATCH_PIXELS   # 10 m
HALF = PATCH_METERS / 2.0

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "AWS_NO_SIGN_REQUEST": "YES",
}


def fetch_item(scene_id):
    url = f"{STAC_BASE}/{scene_id}"

    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def make_grid(item, lon, lat):
    """
    Use the native B02 CRS, matching the historical MethaneFuse
    preparation choice of the B2 projection.

    Create a 480 m × 480 m grid centered on the source:
    48 × 48 pixels = 10 m/pixel.
    """
    blue_href = item["assets"]["blue"]["href"]

    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(blue_href) as src:
            crs = src.crs

    xs, ys = warp_transform(
        "EPSG:4326",
        crs,
        [float(lon)],
        [float(lat)],
    )

    cx, cy = xs[0], ys[0]

    transform = from_origin(
        cx - HALF,
        cy + HALF,
        PIXEL_SIZE,
        PIXEL_SIZE,
    )

    return crs, transform


def read_band(href, crs, transform):
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as src:
            with WarpedVRT(
                src,
                crs=crs,
                transform=transform,
                width=PATCH_PIXELS,
                height=PATCH_PIXELS,
                resampling=Resampling.nearest,
                nodata=0,
            ) as vrt:
                arr = vrt.read(1)

    # Historical pipeline used unmask(0).toUint16().
    arr = np.nan_to_num(arr, nan=0, posinf=0, neginf=0)
    arr = np.clip(arr, 0, 65535).astype(np.uint16)

    return arr


def write_patch(scene_id, lon, lat, output_path):
    item = fetch_item(scene_id)

    if item["id"] != scene_id:
        raise RuntimeError(
            f"STAC ID mismatch: requested={scene_id} got={item['id']}"
        )

    missing = [
        key for key, _ in BANDS
        if key not in item.get("assets", {})
    ]

    if missing:
        raise RuntimeError(
            f"{scene_id}: missing assets {missing}"
        )

    crs, transform = make_grid(item, lon, lat)

    arrays = []

    for asset_key, band_name in BANDS:
        href = item["assets"][asset_key]["href"]

        arr = read_band(
            href,
            crs,
            transform,
        )

        arrays.append(arr)

    stack = np.stack(arrays, axis=0)

    if stack.shape != (12, 48, 48):
        raise RuntimeError(
            f"Unexpected shape: {stack.shape}"
        )

    if stack.dtype != np.uint16:
        raise RuntimeError(
            f"Unexpected dtype: {stack.dtype}"
        )

    if np.all(stack == 0):
        raise RuntimeError(
            "All-zero output"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile = {
        "driver": "GTiff",
        "height": 48,
        "width": 48,
        "count": 12,
        "dtype": "uint16",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "nodata": 0,
    }

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:

        for i, (arr, (_, band_name)) in enumerate(
            zip(arrays, BANDS),
            start=1,
        ):
            dst.write(arr, i)
            dst.set_band_description(i, band_name)

        dst.update_tags(
            source_scene_id=scene_id,
            source_datetime=item["properties"].get(
                "datetime", ""
            ),
            footprint_m="480",
            patch_pixels="48",
            input_format="MethaneFuse_s2_12band",
        )

    return item["properties"].get("datetime")


def validate(path):
    try:
        with rasterio.open(path) as src:
            arr = src.read()

            return (
                src.count == 12
                and src.width == 48
                and src.height == 48
                and arr.dtype == np.uint16
                and not np.all(arr == 0)
            )
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = ap.parse_args()

    df = pd.read_csv(INPUT)

    if args.limit > 0:
        df = df.head(args.limit).copy()

    eval_rows = []
    audit_rows = []

    for row_num, row in df.iterrows():
        rid = str(row["record_id"])

        print(
            f"\n[{row_num + 1}/{len(df)}] "
            f"{rid} label={row['label']}",
            flush=True,
        )

        paths = {}
        ok = True

        for slot in ["t0", "t90", "t360"]:

            scene_id = str(
                row[f"{slot}_scene_id"]
            )

            out = (
                OUTPUT_ROOT
                / rid
                / f"{rid}__{slot}.tif"
            )

            status = "pending"
            error = ""
            actual_datetime = ""

            try:
                if (
                    out.exists()
                    and not args.overwrite
                    and validate(out)
                ):
                    status = "reused_existing"

                else:
                    actual_datetime = write_patch(
                        scene_id=scene_id,
                        lon=float(row["longitude"]),
                        lat=float(row["latitude"]),
                        output_path=out,
                    )

                    status = "downloaded"

                if not validate(out):
                    raise RuntimeError(
                        "Output TIFF validation failed"
                    )

                paths[slot] = str(out.resolve())

            except Exception as exc:
                status = "failed"
                error = (
                    f"{type(exc).__name__}: {exc}"
                )
                ok = False

            audit_rows.append({
                "record_id": rid,
                "label": int(row["label"]),
                "site_id": row["site_id"],
                "slot": slot,
                "scene_id": scene_id,
                "manifest_scene_time_utc":
                    row[f"{slot}_scene_time_utc"],
                "stac_scene_time_utc":
                    actual_datetime,
                "output_path": str(out),
                "status": status,
                "error": error,
            })

            print(
                f"  {slot}: {status} "
                f"{scene_id}",
                flush=True,
            )

        if ok:
            eval_rows.append({
                "id": rid,
                "label": int(row["label"]),
                "s2_0_path": paths["t0"],
                "s2_90_path": paths["t90"],
                "s2_360_path": paths["t360"],
                "site": row["site_id"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "ground_truth_time_utc":
                    row["ground_truth_time_utc"],
                "t0_scene_id":
                    row["t0_scene_id"],
                "t90_scene_id":
                    row["t90_scene_id"],
                "t360_scene_id":
                    row["t360_scene_id"],
            })

        pd.DataFrame(audit_rows).to_csv(
            AUDIT_CSV,
            index=False,
        )

        pd.DataFrame(eval_rows).to_csv(
            OUTPUT_CSV,
            index=False,
        )

    eval_df = pd.DataFrame(eval_rows)
    audit_df = pd.DataFrame(audit_rows)

    print("\n==============================")
    print("DONE")
    print("==============================")
    print("Requested records:", len(df))
    print("Ready records:", len(eval_df))

    if len(eval_df):
        print("\nLabels:")
        print(
            eval_df["label"]
            .value_counts()
            .sort_index()
        )

    print("\nSlot status:")
    print(
        audit_df["status"]
        .value_counts(dropna=False)
    )

    print("\nCreated:")
    print(OUTPUT_CSV)
    print(AUDIT_CSV)


if __name__ == "__main__":
    main()
