#!/usr/bin/env python3
"""
Match ground-truth observations to Sentinel-2 L2A and download model-ready
patches from public Earth Search Cloud-Optimized GeoTIFFs.

Outputs:
  data/methaneair_full/sentinel2/
  data/methaneair_full/sentinel2_match_long.csv
  data/methaneair_full/sentinel2_temporal_manifest.csv

Each model input GeoTIFF contains six reflectance bands on a common 20 m grid:
  B02, B03, B04, B08, B11, B12

A separate SCL GeoTIFF is written for quality auditing.

Confirmed negative labels are downloaded from controlled/physical-release
records. MethaneAIR temporal negative candidates can also be downloaded, but
their label remains unconfirmed until exclusion checks are completed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio.warp import transform as warp_transform


STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
BANDS = [
    ("blue", "B02"),
    ("green", "B03"),
    ("red", "B04"),
    ("nir", "B08"),
    ("swir16", "B11"),
    ("swir22", "B12"),
]
CLEAR_SCL = {4, 5, 6, 7}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path("/project/6002520/yunjung1/MethaneFuse"),
    )
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--negative-candidates",
        type=Path,
        default=None,
    )
    p.add_argument("--include-candidate-negatives", action="store_true")
    p.add_argument("--patch-pixels", type=int, default=128)
    p.add_argument("--max-cloud-cover", type=float, default=80.0)
    p.add_argument("--qa-clear-threshold", type=float, default=0.8)
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--request-timeout", type=int, default=120)
    return p.parse_args()


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def utc_timestamp(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce", utc=True)


def search_s2(
    session: requests.Session,
    lon: float,
    lat: float,
    target: pd.Timestamp,
    window_days: int,
    max_cloud: float,
    timeout: int,
) -> list[dict[str, Any]]:
    start = (target - pd.Timedelta(days=window_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    end = (target + pd.Timedelta(days=window_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    body = {
        "collections": [COLLECTION],
        "intersects": {
            "type": "Point",
            "coordinates": [lon, lat],
        },
        "datetime": f"{start}/{end}",
        "limit": 100,
        "query": {
            "eo:cloud_cover": {
                "lte": max_cloud,
            }
        },
    }

    last_error = None
    for attempt in range(1, 5):
        try:
            response = session.post(
                STAC_SEARCH,
                json=body,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json().get("features", [])
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 15))

    raise RuntimeError(f"STAC search failed: {last_error}")


def select_item(
    items: list[dict[str, Any]],
    target: pd.Timestamp,
) -> dict[str, Any] | None:
    scored = []

    for item in items:
        dt = utc_timestamp(item.get("properties", {}).get("datetime"))
        if pd.isna(dt):
            continue

        cloud = item.get("properties", {}).get("eo:cloud_cover")
        try:
            cloud_number = float(cloud)
        except Exception:
            cloud_number = 100.0

        time_delta_hours = abs((dt - target).total_seconds()) / 3600.0
        score = time_delta_hours + cloud_number * 1.5
        scored.append((score, time_delta_hours, cloud_number, item))

    if not scored:
        return None

    scored.sort(key=lambda value: (value[0], value[1], value[2]))
    return scored[0][3]


def asset_scale_offset(asset: dict[str, Any]) -> tuple[float, float, Any]:
    bands = asset.get("raster:bands") or [{}]
    metadata = bands[0] if bands else {}
    return (
        float(metadata.get("scale", 1.0)),
        float(metadata.get("offset", 0.0)),
        metadata.get("nodata"),
    )


def read_to_grid(
    href: str,
    asset: dict[str, Any],
    crs,
    dst_transform,
    width: int,
    height: int,
    resampling: Resampling,
    output_float: bool,
) -> np.ndarray:
    with rasterio.open(href) as src:
        with WarpedVRT(
            src,
            crs=crs,
            transform=dst_transform,
            width=width,
            height=height,
            resampling=resampling,
        ) as vrt:
            array = vrt.read(1)

    scale, offset, nodata = asset_scale_offset(asset)

    if output_float:
        array = array.astype(np.float32)
        invalid = np.zeros(array.shape, dtype=bool)
        if nodata is not None:
            invalid |= array == float(nodata)
        invalid |= ~np.isfinite(array)
        array = array * scale + offset
        array[invalid] = np.nan
        return array

    return array.astype(np.uint8)


def download_patch(
    item: dict[str, Any],
    lon: float,
    lat: float,
    image_path: Path,
    scl_path: Path,
    patch_pixels: int,
) -> tuple[float, float]:
    assets = item.get("assets", {})
    required = [key for key, _ in BANDS] + ["scl"]
    missing = [key for key in required if key not in assets]
    if missing:
        raise KeyError(f"Missing STAC assets: {missing}")

    reference_asset = assets["swir16"]
    reference_href = reference_asset["href"]

    env_options = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
        "AWS_NO_SIGN_REQUEST": "YES",
    }

    with rasterio.Env(**env_options):
        with rasterio.open(reference_href) as reference:
            xs, ys = warp_transform(
                "EPSG:4326",
                reference.crs,
                [lon],
                [lat],
            )
            row, col = reference.index(xs[0], ys[0])
            half = patch_pixels // 2
            window = Window(
                col_off=col - half,
                row_off=row - half,
                width=patch_pixels,
                height=patch_pixels,
            )
            dst_transform = reference.window_transform(window)
            dst_crs = reference.crs

        reflectance = []
        for asset_key, band_name in BANDS:
            asset = assets[asset_key]
            array = read_to_grid(
                href=asset["href"],
                asset=asset,
                crs=dst_crs,
                dst_transform=dst_transform,
                width=patch_pixels,
                height=patch_pixels,
                resampling=Resampling.bilinear,
                output_float=True,
            )
            reflectance.append(array)

        scl_asset = assets["scl"]
        scl = read_to_grid(
            href=scl_asset["href"],
            asset=scl_asset,
            crs=dst_crs,
            dst_transform=dst_transform,
            width=patch_pixels,
            height=patch_pixels,
            resampling=Resampling.nearest,
            output_float=False,
        )

    image_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": patch_pixels,
        "width": patch_pixels,
        "count": len(BANDS),
        "dtype": "float32",
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "nodata": np.nan,
    }

    with rasterio.open(image_path, "w", **profile) as dst:
        for index, (array, (_, band_name)) in enumerate(
            zip(reflectance, BANDS),
            start=1,
        ):
            dst.write(array.astype(np.float32), index)
            dst.set_band_description(index, band_name)
        dst.update_tags(
            reflectance_scale_applied="true",
            band_order="B02,B03,B04,B08,B11,B12",
        )

    scl_profile = {
        "driver": "GTiff",
        "height": patch_pixels,
        "width": patch_pixels,
        "count": 1,
        "dtype": "uint8",
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "deflate",
        "tiled": True,
        "nodata": 0,
    }
    with rasterio.open(scl_path, "w", **scl_profile) as dst:
        dst.write(scl, 1)
        dst.set_band_description(1, "SCL")

    valid = scl > 0
    clear = np.isin(scl, list(CLEAR_SCL))
    valid_fraction = float(valid.mean())
    clear_fraction = (
        float(clear[valid].mean()) if np.any(valid) else 0.0
    )
    return clear_fraction, valid_fraction


def slot_specs(base_time: pd.Timestamp):
    return [
        ("t0", base_time, 3),
        ("t90", base_time - pd.Timedelta(days=90), 12),
        ("t360", base_time - pd.Timedelta(days=360), 12),
    ]


def load_records(args: argparse.Namespace, output_root: Path) -> pd.DataFrame:
    gt_path = (
        args.ground_truth.expanduser().resolve()
        if args.ground_truth
        else output_root / "ground_truth_confirmed.csv"
    )
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    confirmed = pd.read_csv(gt_path)
    confirmed["download_group"] = "confirmed"

    frames = [confirmed]
    if args.include_candidate_negatives:
        candidate_path = (
            args.negative_candidates.expanduser().resolve()
            if args.negative_candidates
            else output_root / "ground_truth_negative_candidates.csv"
        )
        if candidate_path.exists():
            candidates = pd.read_csv(candidate_path)
            candidates["download_group"] = "candidate_negative"
            frames.append(candidates)

    records = pd.concat(frames, ignore_index=True, sort=False)
    records["latitude"] = pd.to_numeric(records["latitude"], errors="coerce")
    records["longitude"] = pd.to_numeric(records["longitude"], errors="coerce")
    records["_time"] = pd.to_datetime(
        records["acquisition_time_utc"],
        errors="coerce",
        utc=True,
    )

    records = records[
        records["latitude"].notna()
        & records["longitude"].notna()
        & records["_time"].notna()
    ].copy()

    if args.max_records > 0:
        records = records.head(args.max_records)

    return records


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_root = project_root / "data" / "methaneair_full"
    download_root = output_root / "sentinel2"
    download_root.mkdir(parents=True, exist_ok=True)

    records = load_records(args, output_root)
    print(f"Records eligible for Sentinel-2 search: {len(records)}", flush=True)

    session = requests.Session()
    long_rows = []
    manifest_rows = []

    for number, (_, row) in enumerate(records.iterrows(), start=1):
        record_id = safe_name(row["record_id"])
        base_time = row["_time"]
        lon = float(row["longitude"])
        lat = float(row["latitude"])

        manifest = {
            "record_id": row["record_id"],
            "site_id": row.get("site_id"),
            "latitude": lat,
            "longitude": lon,
            "ground_truth_time_utc": base_time.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "label": row.get("label"),
            "proposed_label": row.get("proposed_label"),
            "label_status": row.get("label_status"),
            "ground_truth_source": row.get("ground_truth_source"),
            "download_group": row.get("download_group"),
        }

        for slot, target, window_days in slot_specs(base_time):
            image_path = download_root / f"{record_id}__{slot}.tif"
            scl_path = download_root / f"{record_id}__{slot}__scl.tif"

            result = {
                "record_id": row["record_id"],
                "slot": slot,
                "target_time_utc": target.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "window_days": window_days,
                "selected_item_id": pd.NA,
                "selected_datetime_utc": pd.NA,
                "scene_cloud_cover": pd.NA,
                "image_path": str(image_path),
                "scl_path": str(scl_path),
                "clear_fraction": pd.NA,
                "valid_fraction": pd.NA,
                "qa_pass": False,
                "status": "pending",
                "error": "",
            }

            try:
                if args.resume and image_path.exists() and scl_path.exists():
                    # The raster files already exist, but we still need to
                    # recover the source Sentinel-2 scene metadata.
                    items = search_s2(
                        session=session,
                        lon=lon,
                        lat=lat,
                        target=target,
                        window_days=window_days,
                        max_cloud=args.max_cloud_cover,
                        timeout=args.request_timeout,
                    )
                    item = select_item(items, target)

                    if item is not None:
                        properties = item.get("properties", {})
                        result["selected_item_id"] = item.get("id")
                        result["selected_datetime_utc"] = properties.get(
                            "datetime"
                        )
                        result["scene_cloud_cover"] = properties.get(
                            "eo:cloud_cover"
                        )
                    else:
                        result["error"] = (
                            "Existing raster files were found, but the "
                            "corresponding STAC scene could not be recovered."
                        )

                    with rasterio.open(scl_path) as src:
                        scl = src.read(1)

                    valid = scl > 0
                    clear = np.isin(scl, list(CLEAR_SCL))

                    clear_fraction = (
                        float(clear[valid].mean())
                        if np.any(valid)
                        else 0.0
                    )
                    valid_fraction = float(valid.mean())

                    result["clear_fraction"] = clear_fraction
                    result["valid_fraction"] = valid_fraction
                    result["qa_pass"] = (
                        clear_fraction >= args.qa_clear_threshold
                    )
                    result["status"] = "already_exists"
                else:
                    items = search_s2(
                        session=session,
                        lon=lon,
                        lat=lat,
                        target=target,
                        window_days=window_days,
                        max_cloud=args.max_cloud_cover,
                        timeout=args.request_timeout,
                    )
                    item = select_item(items, target)
                    if item is None:
                        result["status"] = "no_scene"
                    else:
                        properties = item.get("properties", {})
                        result["selected_item_id"] = item.get("id")
                        result["selected_datetime_utc"] = properties.get(
                            "datetime"
                        )
                        result["scene_cloud_cover"] = properties.get(
                            "eo:cloud_cover"
                        )

                        clear_fraction, valid_fraction = download_patch(
                            item=item,
                            lon=lon,
                            lat=lat,
                            image_path=image_path,
                            scl_path=scl_path,
                            patch_pixels=args.patch_pixels,
                        )
                        result["clear_fraction"] = clear_fraction
                        result["valid_fraction"] = valid_fraction
                        result["qa_pass"] = (
                            clear_fraction >= args.qa_clear_threshold
                        )
                        result["status"] = "downloaded"

            except Exception as exc:
                result["status"] = "error"
                result["error"] = f"{type(exc).__name__}: {exc}"

            long_rows.append(result)

            manifest[f"{slot}_path"] = (
                str(image_path)
                if result["status"] in {"downloaded", "already_exists"}
                else pd.NA
            )
            manifest[f"{slot}_scl_path"] = (
                str(scl_path)
                if result["status"] in {"downloaded", "already_exists"}
                else pd.NA
            )
            manifest[f"{slot}_scene_id"] = result["selected_item_id"]
            manifest[f"{slot}_scene_time_utc"] = result[
                "selected_datetime_utc"
            ]
            manifest[f"{slot}_clear_fraction"] = result["clear_fraction"]
            manifest[f"{slot}_qa_pass"] = result["qa_pass"]
            manifest[f"{slot}_status"] = result["status"]

        manifest["all_three_downloaded"] = all(
            manifest.get(f"{slot}_status")
            in {"downloaded", "already_exists"}
            for slot in ["t0", "t90", "t360"]
        )
        manifest["all_three_qa_pass"] = all(
            bool(manifest.get(f"{slot}_qa_pass"))
            for slot in ["t0", "t90", "t360"]
        )
        manifest_rows.append(manifest)

        print(
            f"Sentinel-2 {number}/{len(records)}: {record_id} | "
            f"t0={manifest.get('t0_status')} "
            f"t90={manifest.get('t90_status')} "
            f"t360={manifest.get('t360_status')}",
            flush=True,
        )

        if number % 10 == 0:
            pd.DataFrame(long_rows).to_csv(
                output_root / "sentinel2_match_long.partial.csv",
                index=False,
            )
            pd.DataFrame(manifest_rows).to_csv(
                output_root / "sentinel2_temporal_manifest.partial.csv",
                index=False,
            )

    long_df = pd.DataFrame(long_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    long_path = output_root / "sentinel2_match_long.csv"
    manifest_path = output_root / "sentinel2_temporal_manifest.csv"
    summary_path = output_root / "sentinel2_download_summary.csv"

    long_df.to_csv(long_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "records": len(manifest_df),
                "confirmed_records": int(
                    manifest_df["download_group"].eq("confirmed").sum()
                ),
                "candidate_negative_records": int(
                    manifest_df["download_group"].eq(
                        "candidate_negative"
                    ).sum()
                ),
                "all_three_downloaded": int(
                    manifest_df["all_three_downloaded"].sum()
                ),
                "all_three_qa_pass": int(
                    manifest_df["all_three_qa_pass"].sum()
                ),
                "t0_downloaded": int(
                    manifest_df["t0_status"].isin(
                        ["downloaded", "already_exists"]
                    ).sum()
                ),
                "t0_qa_pass": int(manifest_df["t0_qa_pass"].sum()),
            }
        ]
    )
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
