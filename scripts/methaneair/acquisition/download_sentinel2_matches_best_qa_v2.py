#!/usr/bin/env python3
"""
Robust Sentinel-2 matcher for MethaneAIR/controlled-release ground truth.

Improvements over v1
--------------------
1. Uses fallback time windows:
   t0   : ±3, ±7, ±14 days
   t90  : ±12, ±20 days
   t360 : ±12, ±20 days
2. Evaluates local SCL around the source for several candidate scenes.
3. Selects the scene with the best local clear fraction, rather than relying
   only on catalog-level cloud cover.
4. Writes candidate audit information and time offsets.
5. Uses a separate output directory and filenames, so v1 results are preserved.

Reflectance output bands on a common 20 m grid:
B02, B03, B04, B08, B11, B12
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Iterable

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

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "AWS_NO_SIGN_REQUEST": "YES",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/project/6002520/yunjung1/MethaneFuse"),
    )
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--negative-candidates", type=Path, default=None)
    parser.add_argument("--include-candidate-negatives", action="store_true")
    parser.add_argument("--patch-pixels", type=int, default=128)
    parser.add_argument("--max-cloud-cover", type=float, default=95.0)
    parser.add_argument("--qa-clear-threshold", type=float, default=0.8)
    parser.add_argument("--max-candidates-per-slot", type=int, default=10)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument(
        "--output-subdir",
        default="sentinel2_best_qa_v2",
    )
    return parser.parse_args()


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def utc_timestamp(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce", utc=True)


def slot_specs(base_time: pd.Timestamp):
    return [
        ("t0", base_time, [3, 7, 14]),
        ("t90", base_time - pd.Timedelta(days=90), [12, 20]),
        ("t360", base_time - pd.Timedelta(days=360), [12, 20]),
    ]


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


def deduplicate_items(
    items: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    seen = set()

    for item in items:
        item_id = item.get("id")
        if item_id in seen:
            continue
        seen.add(item_id)
        output.append(item)

    return output


def required_assets_available(item: dict[str, Any]) -> bool:
    assets = item.get("assets", {})
    required = [key for key, _ in BANDS] + ["scl"]
    return all(key in assets for key in required)


def asset_scale_offset(asset: dict[str, Any]) -> tuple[float, float, Any]:
    bands = asset.get("raster:bands") or [{}]
    metadata = bands[0] if bands else {}

    return (
        float(metadata.get("scale", 1.0)),
        float(metadata.get("offset", 0.0)),
        metadata.get("nodata"),
    )


def reference_grid(
    item: dict[str, Any],
    lon: float,
    lat: float,
    patch_pixels: int,
):
    assets = item["assets"]
    href = assets["swir16"]["href"]

    with rasterio.open(href) as reference:
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

        return (
            reference.crs,
            reference.window_transform(window),
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
    with rasterio.open(href) as source:
        with WarpedVRT(
            source,
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
        invalid = ~np.isfinite(array)

        if nodata is not None:
            invalid |= array == float(nodata)

        array = array * scale + offset
        array[invalid] = np.nan
        return array

    return array.astype(np.uint8)


def scl_metrics(
    item: dict[str, Any],
    lon: float,
    lat: float,
    patch_pixels: int,
) -> tuple[float, float, np.ndarray, Any, Any]:
    crs, transform = reference_grid(
        item=item,
        lon=lon,
        lat=lat,
        patch_pixels=patch_pixels,
    )

    asset = item["assets"]["scl"]
    scl = read_to_grid(
        href=asset["href"],
        asset=asset,
        crs=crs,
        dst_transform=transform,
        width=patch_pixels,
        height=patch_pixels,
        resampling=Resampling.nearest,
        output_float=False,
    )

    valid = scl > 0
    clear = np.isin(scl, list(CLEAR_SCL))

    valid_fraction = float(valid.mean())
    clear_fraction = (
        float(clear[valid].mean())
        if np.any(valid)
        else 0.0
    )

    return clear_fraction, valid_fraction, scl, crs, transform


def scene_cloud_cover(item: dict[str, Any]) -> float:
    value = item.get("properties", {}).get("eo:cloud_cover")
    try:
        return float(value)
    except Exception:
        return 100.0


def candidate_sort_key(
    audit: dict[str, Any],
    qa_threshold: float,
):
    """Prefer full usable coverage before image clearness.

    A candidate enters the highest-priority group only when both its
    local clear fraction and valid coverage fraction meet the threshold.
    Among fully usable scenes, the acquisition closest to the target
    time is preferred.
    """
    clear_ok = audit["clear_fraction"] >= qa_threshold
    valid_ok = audit["valid_fraction"] >= qa_threshold

    return (
        0 if (clear_ok and valid_ok) else 1,
        0 if valid_ok else 1,
        0 if clear_ok else 1,
        audit["time_delta_hours"],
        -audit["clear_fraction"],
        -audit["valid_fraction"],
        audit["scene_cloud_cover"],
    )


def evaluate_candidates(
    items: list[dict[str, Any]],
    lon: float,
    lat: float,
    target: pd.Timestamp,
    patch_pixels: int,
    qa_threshold: float,
    max_candidates: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    sortable = []

    for item in items:
        if not required_assets_available(item):
            continue

        dt = utc_timestamp(
            item.get("properties", {}).get("datetime")
        )
        if pd.isna(dt):
            continue

        sortable.append(
            (
                abs((dt - target).total_seconds()) / 3600.0,
                scene_cloud_cover(item),
                item,
            )
        )

    sortable.sort(key=lambda value: (value[0], value[1]))
    sortable = sortable[:max_candidates]

    audits = []

    with rasterio.Env(**GDAL_ENV):
        for time_delta_hours, cloud, item in sortable:
            audit = {
                "item_id": item.get("id"),
                "datetime_utc": item.get(
                    "properties", {}
                ).get("datetime"),
                "time_delta_hours": time_delta_hours,
                "scene_cloud_cover": cloud,
                "clear_fraction": 0.0,
                "valid_fraction": 0.0,
                "assessment_status": "pending",
                "assessment_error": "",
                "_item": item,
            }

            try:
                (
                    clear_fraction,
                    valid_fraction,
                    _,
                    _,
                    _,
                ) = scl_metrics(
                    item=item,
                    lon=lon,
                    lat=lat,
                    patch_pixels=patch_pixels,
                )

                audit["clear_fraction"] = clear_fraction
                audit["valid_fraction"] = valid_fraction
                audit["assessment_status"] = "assessed"

            except Exception as exc:
                audit["assessment_status"] = "error"
                audit["assessment_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )

            audits.append(audit)

    usable = [
        audit for audit in audits
        if audit["assessment_status"] == "assessed"
    ]

    if not usable:
        return None, audits

    usable.sort(
        key=lambda audit: candidate_sort_key(
            audit,
            qa_threshold=qa_threshold,
        )
    )

    return usable[0], audits


def write_patch(
    item: dict[str, Any],
    lon: float,
    lat: float,
    image_path: Path,
    scl_path: Path,
    patch_pixels: int,
) -> tuple[float, float]:
    assets = item["assets"]

    with rasterio.Env(**GDAL_ENV):
        (
            clear_fraction,
            valid_fraction,
            scl,
            crs,
            transform,
        ) = scl_metrics(
            item=item,
            lon=lon,
            lat=lat,
            patch_pixels=patch_pixels,
        )

        reflectance = []
        for asset_key, _ in BANDS:
            asset = assets[asset_key]

            array = read_to_grid(
                href=asset["href"],
                asset=asset,
                crs=crs,
                dst_transform=transform,
                width=patch_pixels,
                height=patch_pixels,
                resampling=Resampling.bilinear,
                output_float=True,
            )
            reflectance.append(array)

    image_path.parent.mkdir(parents=True, exist_ok=True)

    image_profile = {
        "driver": "GTiff",
        "height": patch_pixels,
        "width": patch_pixels,
        "count": len(BANDS),
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "nodata": np.nan,
    }

    with rasterio.open(image_path, "w", **image_profile) as destination:
        for index, (array, (_, band_name)) in enumerate(
            zip(reflectance, BANDS),
            start=1,
        ):
            destination.write(
                array.astype(np.float32),
                index,
            )
            destination.set_band_description(index, band_name)

        destination.update_tags(
            reflectance_scale_applied="true",
            band_order="B02,B03,B04,B08,B11,B12",
        )

    scl_profile = {
        "driver": "GTiff",
        "height": patch_pixels,
        "width": patch_pixels,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "nodata": 0,
    }

    with rasterio.open(scl_path, "w", **scl_profile) as destination:
        destination.write(scl, 1)
        destination.set_band_description(1, "SCL")

    return clear_fraction, valid_fraction


def load_records(
    args: argparse.Namespace,
    output_root: Path,
) -> pd.DataFrame:
    ground_truth_path = (
        args.ground_truth.expanduser().resolve()
        if args.ground_truth
        else output_root / "ground_truth_confirmed.csv"
    )

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {ground_truth_path}"
        )

    confirmed = pd.read_csv(ground_truth_path)
    confirmed["download_group"] = "confirmed"
    frames = [confirmed]

    if args.include_candidate_negatives:
        candidate_path = (
            args.negative_candidates.expanduser().resolve()
            if args.negative_candidates
            else output_root
            / "ground_truth_negative_candidates.csv"
        )

        if candidate_path.exists():
            candidates = pd.read_csv(candidate_path)
            candidates["download_group"] = "candidate_negative"
            frames.append(candidates)

    records = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    records["latitude"] = pd.to_numeric(
        records["latitude"],
        errors="coerce",
    )
    records["longitude"] = pd.to_numeric(
        records["longitude"],
        errors="coerce",
    )
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


def existing_scl_metrics(
    scl_path: Path,
) -> tuple[float, float]:
    with rasterio.open(scl_path) as source:
        scl = source.read(1)

    valid = scl > 0
    clear = np.isin(scl, list(CLEAR_SCL))

    return (
        float(clear[valid].mean())
        if np.any(valid)
        else 0.0,
        float(valid.mean()),
    )


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_root = project_root / "data" / "methaneair_full"
    download_root = output_root / args.output_subdir
    download_root.mkdir(parents=True, exist_ok=True)

    records = load_records(args, output_root)

    print(
        f"Records eligible for Sentinel-2 search: {len(records)}",
        flush=True,
    )

    session = requests.Session()
    long_rows = []
    candidate_rows = []
    manifest_rows = []

    for number, (_, row) in enumerate(
        records.iterrows(),
        start=1,
    ):
        safe_record_id = safe_name(row["record_id"])
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
            "ground_truth_source": row.get(
                "ground_truth_source"
            ),
            "download_group": row.get("download_group"),
        }

        for slot, target, windows in slot_specs(base_time):
            image_path = (
                download_root
                / f"{safe_record_id}__{slot}.tif"
            )
            scl_path = (
                download_root
                / f"{safe_record_id}__{slot}__scl.tif"
            )

            result = {
                "record_id": row["record_id"],
                "slot": slot,
                "target_time_utc": target.strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "search_windows_days": "|".join(
                    str(value) for value in windows
                ),
                "window_used_days": pd.NA,
                "candidate_count": 0,
                "candidates_assessed": 0,
                "selected_item_id": pd.NA,
                "selected_datetime_utc": pd.NA,
                "time_delta_hours": pd.NA,
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
                if (
                    args.resume
                    and image_path.exists()
                    and scl_path.exists()
                ):
                    (
                        clear_fraction,
                        valid_fraction,
                    ) = existing_scl_metrics(scl_path)

                    result["clear_fraction"] = clear_fraction
                    result["valid_fraction"] = valid_fraction
                    result["qa_pass"] = (
                        clear_fraction
                        >= args.qa_clear_threshold
                        and valid_fraction
                        >= args.qa_clear_threshold
                    )
                    result["status"] = "already_exists"

                else:
                    selected_audit = None
                    all_audits = []
                    all_items = []

                    for window_days in windows:
                        found = search_s2(
                            session=session,
                            lon=lon,
                            lat=lat,
                            target=target,
                            window_days=window_days,
                            max_cloud=args.max_cloud_cover,
                            timeout=args.request_timeout,
                        )

                        all_items = deduplicate_items(
                            [*all_items, *found]
                        )

                        (
                            selected_audit,
                            audits,
                        ) = evaluate_candidates(
                            items=all_items,
                            lon=lon,
                            lat=lat,
                            target=target,
                            patch_pixels=args.patch_pixels,
                            qa_threshold=args.qa_clear_threshold,
                            max_candidates=(
                                args.max_candidates_per_slot
                            ),
                        )

                        all_audits = audits
                        result["window_used_days"] = window_days

                        if (
                            selected_audit is not None
                            and selected_audit[
                                "clear_fraction"
                            ]
                            >= args.qa_clear_threshold
                            and selected_audit[
                                "valid_fraction"
                            ]
                            >= args.qa_clear_threshold
                        ):
                            break

                    result["candidate_count"] = len(all_items)
                    result["candidates_assessed"] = len(
                        all_audits
                    )

                    for audit in all_audits:
                        candidate_rows.append(
                            {
                                "record_id": row["record_id"],
                                "slot": slot,
                                "target_time_utc": target.strftime(
                                    "%Y-%m-%dT%H:%M:%SZ"
                                ),
                                "window_used_days": result[
                                    "window_used_days"
                                ],
                                **{
                                    key: value
                                    for key, value in audit.items()
                                    if key != "_item"
                                },
                            }
                        )

                    if selected_audit is None:
                        result["status"] = (
                            "no_scene"
                            if not all_items
                            else "no_usable_candidate"
                        )
                    else:
                        item = selected_audit["_item"]

                        result["selected_item_id"] = (
                            selected_audit["item_id"]
                        )
                        result["selected_datetime_utc"] = (
                            selected_audit["datetime_utc"]
                        )
                        result["time_delta_hours"] = (
                            selected_audit[
                                "time_delta_hours"
                            ]
                        )
                        result["scene_cloud_cover"] = (
                            selected_audit[
                                "scene_cloud_cover"
                            ]
                        )

                        (
                            clear_fraction,
                            valid_fraction,
                        ) = write_patch(
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
                            clear_fraction
                            >= args.qa_clear_threshold
                            and valid_fraction
                            >= args.qa_clear_threshold
                        )
                        result["status"] = "downloaded"

            except Exception as exc:
                result["status"] = "error"
                result["error"] = (
                    f"{type(exc).__name__}: {exc}"
                )

            long_rows.append(result)

            success = result["status"] in {
                "downloaded",
                "already_exists",
            }

            manifest[f"{slot}_path"] = (
                str(image_path) if success else pd.NA
            )
            manifest[f"{slot}_scl_path"] = (
                str(scl_path) if success else pd.NA
            )
            manifest[f"{slot}_scene_id"] = result[
                "selected_item_id"
            ]
            manifest[f"{slot}_scene_time_utc"] = result[
                "selected_datetime_utc"
            ]
            manifest[f"{slot}_time_delta_hours"] = result[
                "time_delta_hours"
            ]
            manifest[f"{slot}_window_used_days"] = result[
                "window_used_days"
            ]
            manifest[f"{slot}_clear_fraction"] = result[
                "clear_fraction"
            ]
            manifest[f"{slot}_qa_pass"] = result[
                "qa_pass"
            ]
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
            f"Sentinel-2 best-QA-v2 {number}/{len(records)}: "
            f"{safe_record_id} | "
            f"t0={manifest.get('t0_status')} "
            f"({manifest.get('t0_clear_fraction')}) "
            f"t90={manifest.get('t90_status')} "
            f"({manifest.get('t90_clear_fraction')}) "
            f"t360={manifest.get('t360_status')} "
            f"({manifest.get('t360_clear_fraction')})",
            flush=True,
        )

        if number % 5 == 0:
            pd.DataFrame(long_rows).to_csv(
                output_root
                / "sentinel2_match_long_best_qa_v2.partial.csv",
                index=False,
            )
            pd.DataFrame(candidate_rows).to_csv(
                output_root
                / "sentinel2_candidate_audit_best_qa_v2.partial.csv",
                index=False,
            )
            pd.DataFrame(manifest_rows).to_csv(
                output_root
                / "sentinel2_temporal_manifest_best_qa_v2.partial.csv",
                index=False,
            )

    long_df = pd.DataFrame(long_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    long_path = (
        output_root / "sentinel2_match_long_best_qa_v2.csv"
    )
    candidate_path = (
        output_root / "sentinel2_candidate_audit_best_qa_v2.csv"
    )
    manifest_path = (
        output_root
        / "sentinel2_temporal_manifest_best_qa_v2.csv"
    )
    summary_path = (
        output_root / "sentinel2_download_summary_best_qa_v2.csv"
    )

    long_df.to_csv(long_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "records": len(manifest_df),
                "confirmed_records": int(
                    manifest_df["download_group"]
                    .eq("confirmed")
                    .sum()
                ),
                "candidate_negative_records": int(
                    manifest_df["download_group"]
                    .eq("candidate_negative")
                    .sum()
                ),
                "all_three_downloaded": int(
                    manifest_df["all_three_downloaded"].sum()
                ),
                "all_three_qa_pass": int(
                    manifest_df["all_three_qa_pass"].sum()
                ),
                "t0_downloaded": int(
                    manifest_df["t0_status"]
                    .isin(["downloaded", "already_exists"])
                    .sum()
                ),
                "t0_qa_pass": int(
                    manifest_df["t0_qa_pass"].sum()
                ),
                "t90_qa_pass": int(
                    manifest_df["t90_qa_pass"].sum()
                ),
                "t360_qa_pass": int(
                    manifest_df["t360_qa_pass"].sum()
                ),
            }
        ]
    )

    summary.to_csv(summary_path, index=False)
    print("\n" + summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
