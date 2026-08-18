#!/usr/bin/env python3
"""Prepare the 17 MethaneAIR–Sentinel-2 P1 event-centered patches for MethaneFuse.

This script preserves the exact Sentinel-2 preparation logic used by
prepare_five_site_methanefuse_zero_shot.py:

- COPERNICUS/S2_SR_HARMONIZED
- 12 bands in MethaneFuse order
- t0 exact asset from the P1 manifest
- t-90 target with ±30-day search window
- t-360 target with ±45-day search window
- temporal candidates ranked by target-time difference, then cloud percentage,
  then asset ID
- 480 m x 480 m event-centered region
- 48 x 48 pixels
- raw SR DN values exported as uint16
- scene-specific B2 CRS
- nodata/unmasked pixels set to zero

Input
-----
outputs/475_methaneair_s2_p1_patch_manifest_v1.csv

Outputs
-------
outputs/481_methaneair_s2_p1_methanefuse_download_manifest_v1.csv
outputs/482_methaneair_s2_p1_methanefuse_summary_v1.txt
MethaneFuse/data/custom/methaneair_s2_p1_zero_shot_eval.csv
MethaneFuse/data/custom/methaneair_s2_p1_zero_shot_eval_scl80.csv
"""

from __future__ import annotations

import argparse
import io
import math
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import ee
import numpy as np
import pandas as pd
import tifffile


PROJECT_ROOT = Path("/Users/happydoraaa/methane_release_project")
METHANEFUSE_ROOT = Path("/Users/happydoraaa/MethaneFuse")

DEFAULT_INPUT_MANIFEST = (
    PROJECT_ROOT
    / "outputs/475_methaneair_s2_p1_patch_manifest_v1.csv"
)

DEFAULT_PATCH_ROOT = (
    PROJECT_ROOT
    / "methanefuse_input/s2_12band_methaneair_p1"
)

DEFAULT_DOWNLOAD_MANIFEST = (
    PROJECT_ROOT
    / "outputs/481_methaneair_s2_p1_methanefuse_download_manifest_v1.csv"
)

DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs/482_methaneair_s2_p1_methanefuse_summary_v1.txt"
)

DEFAULT_EVAL_CSV = (
    METHANEFUSE_ROOT
    / "data/custom/methaneair_s2_p1_zero_shot_eval.csv"
)

DEFAULT_EVAL_SCL80_CSV = (
    METHANEFUSE_ROOT
    / "data/custom/methaneair_s2_p1_zero_shot_eval_scl80.csv"
)

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

# Exact MethaneFuse s2_12band order.
S2_BANDS = [
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

FRAME_SPECS = {
    "t0": {
        "offset_days": 0,
        "half_window_days": 0,
    },
    "t90": {
        "offset_days": -90,
        "half_window_days": 30,
    },
    "t360": {
        "offset_days": -360,
        "half_window_days": 45,
    },
}

PATCH_PIXELS = 48
PATCH_METERS = 480
HALF_PATCH_METERS = PATCH_METERS / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default="methane-release-gee",
        help="Google Earth Engine project ID.",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=DEFAULT_INPUT_MANIFEST,
    )
    parser.add_argument(
        "--patch-root",
        type=Path,
        default=DEFAULT_PATCH_ROOT,
    )
    parser.add_argument(
        "--download-manifest",
        type=Path,
        default=DEFAULT_DOWNLOAD_MANIFEST,
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=DEFAULT_EVAL_CSV,
    )
    parser.add_argument(
        "--eval-scl80-csv",
        type=Path,
        default=DEFAULT_EVAL_SCL80_CSV,
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=80.0,
        help=(
            "Maximum scene CLOUDY_PIXEL_PERCENTAGE for t90/t360 searches. "
            "The exact t0 asset is never replaced."
        ),
    )
    parser.add_argument(
        "--scl-threshold",
        type=float,
        default=0.80,
        help="Minimum all-frame SCL clear fraction for the strict eval CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload valid existing TIFFs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for a smoke test.",
    )
    return parser.parse_args()


def initialize_ee(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine 初始化失敗。先執行 `earthengine authenticate`，"
            f"並確認 project={project!r}。原始錯誤：{exc}"
        ) from exc


def parse_bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def safe_name(text: Any) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    return value.strip("_")[:180]


def normalize_asset_id(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise ValueError(f"Invalid Sentinel-2 asset ID: {value!r}")
    if text.startswith("COPERNICUS/"):
        return text
    return f"{S2_COLLECTION}/{text}"


def millis_to_timestamp(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, unit="ms", utc=True)


def load_p1_manifest(
    input_path: Path,
    limit: int | None,
) -> pd.DataFrame:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"P1 manifest not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)

    required = [
        "sample_id",
        "scene_cluster_id",
        "event_id",
        "flight_id",
        "emission_kg_hr",
        "patch_center_lat",
        "patch_center_lon",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{input_path.name} 缺少必要欄位：{missing}\n"
            f"目前欄位：{df.columns.tolist()}"
        )

    asset_column = next(
        (
            column
            for column in ["t0_asset_id", "s2_asset_id"]
            if column in df.columns
        ),
        None,
    )
    time_column = next(
        (
            column
            for column in ["t0_time_utc", "s2_time_utc"]
            if column in df.columns
        ),
        None,
    )
    scene_column = next(
        (
            column
            for column in ["t0_scene_id", "s2_scene_id"]
            if column in df.columns
        ),
        None,
    )

    if asset_column is None:
        raise ValueError("P1 manifest 缺少 t0_asset_id 或 s2_asset_id。")
    if time_column is None:
        raise ValueError("P1 manifest 缺少 t0_time_utc 或 s2_time_utc。")

    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["event_id"] = df["event_id"].astype(str).str.strip()
    df["flight_id"] = df["flight_id"].astype(str).str.strip()
    df["scene_cluster_id"] = (
        df["scene_cluster_id"].astype(str).str.strip()
    )

    if df["sample_id"].duplicated().any():
        duplicates = (
            df.loc[df["sample_id"].duplicated(False), "sample_id"]
            .tolist()
        )
        raise ValueError(f"sample_id 必須唯一，重複值：{duplicates}")

    df["lat"] = pd.to_numeric(
        df["patch_center_lat"], errors="coerce"
    )
    df["lon"] = pd.to_numeric(
        df["patch_center_lon"], errors="coerce"
    )
    df["emission_kg_hr"] = pd.to_numeric(
        df["emission_kg_hr"], errors="coerce"
    )
    df["acquisition_time_utc"] = pd.to_datetime(
        df[time_column], utc=True, errors="coerce"
    )
    df["t0_asset_id_resolved"] = df[asset_column].map(
        normalize_asset_id
    )

    if scene_column is not None:
        df["scene_id"] = df[scene_column].astype(str)
    else:
        df["scene_id"] = (
            df["t0_asset_id_resolved"]
            .astype(str)
            .str.rsplit("/", n=1)
            .str[-1]
        )

    df = df.dropna(
        subset=[
            "lat",
            "lon",
            "emission_kg_hr",
            "acquisition_time_utc",
        ]
    ).copy()

    # These P1 rows are MethaneAIR-referenced observational positives.
    if "label" in df.columns:
        labels = pd.to_numeric(df["label"], errors="coerce")
        if labels.isna().any() or not labels.eq(1).all():
            raise ValueError(
                "P1 manifest 應全部為 label=1，實際包含其他值。"
            )
    df["label"] = 1
    df["label_provenance"] = "methaneair_observational_positive"
    df["site"] = "MethaneAIR_P1"
    df["ground_truth_source"] = (
        "MethaneAIR_observational_detection"
    )
    df["controlled_release_verified"] = False
    df["independent_acquisition_unit"] = df["scene_cluster_id"]
    df["external_eval_id"] = df["sample_id"].map(safe_name)

    df = df.sort_values(
        [
            "acquisition_time_utc",
            "scene_cluster_id",
            "event_id",
            "sample_id",
        ]
    ).reset_index(drop=True)

    if limit is not None:
        df = df.head(limit).copy()

    return df


def get_image_metadata(image: ee.Image) -> dict[str, Any]:
    metadata = ee.Dictionary(
        {
            "scene_id": image.get("system:index"),
            "asset_id": image.id(),
            "time_start": image.get("system:time_start"),
            "cloud_pct": image.get("CLOUDY_PIXEL_PERCENTAGE"),
            "mgrs_tile": image.get("MGRS_TILE"),
            "product_id": image.get("PRODUCT_ID"),
            "spacecraft_name": image.get("SPACECRAFT_NAME"),
        }
    ).getInfo()

    raw_asset_id = metadata.get("asset_id") or metadata.get("scene_id")
    if raw_asset_id:
        metadata["asset_id"] = normalize_asset_id(raw_asset_id)

    if metadata.get("time_start") is not None:
        metadata["acquisition_time_utc"] = millis_to_timestamp(
            metadata["time_start"]
        )
    else:
        metadata["acquisition_time_utc"] = pd.NaT

    return metadata


def load_exact_t0(row: pd.Series) -> tuple[ee.Image, dict[str, Any]]:
    asset_id = normalize_asset_id(row["t0_asset_id_resolved"])
    image = ee.Image(asset_id)

    # Force an Earth Engine request so nonexistent assets fail immediately.
    image.get("system:index").getInfo()
    metadata = get_image_metadata(image)

    requested_time = pd.to_datetime(
        row["acquisition_time_utc"], utc=True, errors="coerce"
    )
    actual_time = pd.to_datetime(
        metadata["acquisition_time_utc"], utc=True, errors="coerce"
    )

    difference_minutes = abs(
        (actual_time - requested_time).total_seconds()
    ) / 60.0

    # The asset itself is authoritative, but a large discrepancy indicates
    # that the manifest is internally inconsistent.
    if difference_minutes > 10.0:
        raise RuntimeError(
            "P1 manifest 的 t0 asset 與 t0 time 不一致："
            f" sample={row['sample_id']}, asset={asset_id}, "
            f"difference={difference_minutes:.2f} minutes"
        )

    metadata.update(
        {
            "candidate_count": 1,
            "target_time_utc": requested_time,
            "window_start_utc": requested_time,
            "window_end_utc": requested_time,
            "absolute_target_difference_seconds": (
                difference_minutes * 60.0
            ),
            "match_method": "exact_asset_from_p1_manifest",
        }
    )
    return image.select(S2_BANDS), metadata


def choose_temporal_image(
    latitude: float,
    longitude: float,
    t0: pd.Timestamp,
    offset_days: int,
    half_window_days: int,
    max_cloud: float,
) -> tuple[ee.Image | None, dict[str, Any]]:
    target = t0 + pd.Timedelta(days=offset_days)
    start = target - pd.Timedelta(days=half_window_days)
    end = target + pd.Timedelta(days=half_window_days + 1)

    point = ee.Geometry.Point([longitude, latitude])

    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(point)
        .filterDate(start.isoformat(), end.isoformat())
        .filter(
            ee.Filter.lte(
                "CLOUDY_PIXEL_PERCENTAGE",
                float(max_cloud),
            )
        )
    )

    size = int(collection.size().getInfo())
    search_info = {
        "target_time_utc": target,
        "window_start_utc": start,
        "window_end_utc": end,
        "candidate_count": size,
    }

    if size == 0:
        return None, search_info

    raw = (
        collection
        .select(S2_BANDS)
        .toList(collection.size())
        .map(
            lambda obj: ee.Dictionary(
                {
                    "asset_id": ee.Image(obj).id(),
                    "scene_id": ee.Image(obj).get("system:index"),
                    "time_start": ee.Image(obj).get("system:time_start"),
                    "cloud_pct": ee.Image(obj).get(
                        "CLOUDY_PIXEL_PERCENTAGE"
                    ),
                    "mgrs_tile": ee.Image(obj).get("MGRS_TILE"),
                    "product_id": ee.Image(obj).get("PRODUCT_ID"),
                    "spacecraft_name": ee.Image(obj).get(
                        "SPACECRAFT_NAME"
                    ),
                }
            )
        )
        .getInfo()
    )

    candidates: list[dict[str, Any]] = []
    for item in raw:
        timestamp = millis_to_timestamp(item["time_start"])
        cloud = item.get("cloud_pct")
        cloud = float(cloud) if cloud is not None else math.inf
        diff_seconds = abs((timestamp - target).total_seconds())

        candidates.append(
            {
                **item,
                "acquisition_time_utc": timestamp,
                "absolute_target_difference_seconds": diff_seconds,
                "_sort_key": (
                    diff_seconds,
                    cloud,
                    str(item["asset_id"]),
                ),
            }
        )

    selected = min(candidates, key=lambda row: row["_sort_key"])
    selected.pop("_sort_key", None)

    raw_asset_id = selected.get("asset_id") or selected.get("scene_id")
    if not raw_asset_id:
        raise RuntimeError(
            "Temporal S2 candidate has no asset ID or scene ID."
        )

    selected["asset_id"] = normalize_asset_id(raw_asset_id)
    image = ee.Image(selected["asset_id"]).select(S2_BANDS)

    search_info.update(selected)
    return image, search_info


def scl_clear_fraction(
    image: ee.Image,
    latitude: float,
    longitude: float,
) -> float:
    region = (
        ee.Geometry.Point([longitude, latitude])
        .buffer(HALF_PATCH_METERS)
        .bounds()
    )

    scl = image.select("SCL")

    clear = (
        scl.eq(4)
        .Or(scl.eq(5))
        .Or(scl.eq(6))
        .Or(scl.eq(7))
        .rename("clear")
    )

    value = (
        clear.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=20,
            bestEffort=True,
            maxPixels=1_000_000,
        )
        .get("clear")
        .getInfo()
    )

    return float(value) if value is not None else float("nan")


def validate_tiff(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tiff_exists": path.exists(),
        "tiff_valid": False,
        "tiff_shape": None,
        "tiff_dtype": None,
        "all_zero": None,
        "has_nan": None,
    }

    if not path.exists():
        return result

    try:
        array = tifffile.imread(path)
    except Exception as exc:
        result["validation_error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["tiff_shape"] = "x".join(str(value) for value in array.shape)
    result["tiff_dtype"] = str(array.dtype)
    result["all_zero"] = bool(np.all(array == 0))
    result["has_nan"] = bool(
        np.isnan(array).any()
        if np.issubdtype(array.dtype, np.floating)
        else False
    )

    expected_channel_shape = (
        array.ndim == 3
        and (
            array.shape[0] == len(S2_BANDS)
            or array.shape[-1] == len(S2_BANDS)
        )
    )
    expected_spatial_shape = (
        array.ndim == 3
        and sorted(array.shape)[-2:] == [48, 48]
    )
    expected_dtype = array.dtype == np.dtype("uint16")

    result["tiff_valid"] = bool(
        expected_channel_shape
        and expected_spatial_shape
        and expected_dtype
        and not result["all_zero"]
        and not result["has_nan"]
    )

    return result


def download_geotiff(
    image: ee.Image,
    latitude: float,
    longitude: float,
    output_path: Path,
    overwrite: bool,
    retries: int = 3,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        validation = validate_tiff(output_path)
        if validation["tiff_valid"]:
            return {
                "download_status": "reused_existing",
                **validation,
            }

    region = (
        ee.Geometry.Point([longitude, latitude])
        .buffer(HALF_PATCH_METERS)
        .bounds()
    )

    export_image = (
        image.select(S2_BANDS)
        .unmask(0)
        .toUint16()
    )

    crs = image.select("B2").projection().crs().getInfo()

    params = {
        "name": output_path.stem,
        "bands": S2_BANDS,
        "region": region.getInfo()["coordinates"],
        "dimensions": f"{PATCH_PIXELS}x{PATCH_PIXELS}",
        "crs": crs,
        "format": "GEO_TIFF",
        "filePerBand": False,
    }

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            url = export_image.getDownloadURL(params)
            with urlopen(url, timeout=180) as response:
                payload = response.read()

            output_path.parent.mkdir(parents=True, exist_ok=True)

            if payload[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    members = [
                        member
                        for member in archive.namelist()
                        if member.lower().endswith((".tif", ".tiff"))
                    ]
                    if not members:
                        raise RuntimeError(
                            "Earth Engine ZIP 中沒有 TIFF。"
                        )
                    with archive.open(members[0]) as source:
                        with output_path.open("wb") as target:
                            shutil.copyfileobj(source, target)
            else:
                output_path.write_bytes(payload)

            validation = validate_tiff(output_path)
            if not validation["tiff_valid"]:
                raise RuntimeError(
                    f"下載後 TIFF 驗證失敗：{validation}"
                )

            return {
                "download_status": "downloaded",
                **validation,
            }

        except Exception as exc:
            last_error = exc
            if output_path.exists():
                output_path.unlink()
            if attempt < retries:
                time.sleep(2.0 * attempt)

    return {
        "download_status": "failed",
        "download_error": (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "unknown"
        ),
        **validate_tiff(output_path),
    }


def main() -> None:
    args = parse_args()

    input_path = args.input_manifest.expanduser().resolve()
    patch_root = args.patch_root.expanduser().resolve()
    download_manifest_path = (
        args.download_manifest.expanduser().resolve()
    )
    summary_path = args.summary_path.expanduser().resolve()
    eval_csv_path = args.eval_csv.expanduser().resolve()
    eval_scl80_path = args.eval_scl80_csv.expanduser().resolve()

    initialize_ee(args.project)
    source = load_p1_manifest(input_path, args.limit)

    print(f"P1 manifest: {input_path}")
    print(f"Event-centered samples requested: {len(source)}")
    print(
        "Independent S2 acquisitions:",
        source["scene_cluster_id"].nunique(),
    )
    print(
        "Emission range:",
        f"{source['emission_kg_hr'].min():.3f}",
        "to",
        f"{source['emission_kg_hr'].max():.3f}",
        "kg/h",
    )

    manifest_rows: list[dict[str, Any]] = []

    for position, (_, row) in enumerate(source.iterrows(), start=1):
        sample_id = str(row["external_eval_id"])
        latitude = float(row["lat"])
        longitude = float(row["lon"])
        t0_time = row["acquisition_time_utc"]

        print(
            f"\n[{position}/{len(source)}] {sample_id}\n"
            f"  event={row['event_id']}\n"
            f"  flight={row['flight_id']}\n"
            f"  cluster={row['scene_cluster_id']}\n"
            f"  emission={row['emission_kg_hr']:.3f} kg/h"
        )

        base = row.to_dict()

        for frame_name, spec in FRAME_SPECS.items():
            output_path = (
                patch_root
                / sample_id
                / f"{sample_id}__{frame_name}.tif"
            )

            if frame_name == "t0":
                image, metadata = load_exact_t0(row)
            else:
                image, metadata = choose_temporal_image(
                    latitude=latitude,
                    longitude=longitude,
                    t0=t0_time,
                    offset_days=int(spec["offset_days"]),
                    half_window_days=int(spec["half_window_days"]),
                    max_cloud=float(args.max_cloud),
                )

            frame_record: dict[str, Any] = {
                **base,
                "frame": frame_name,
                "frame_output_path": str(output_path.resolve()),
                "frame_target_time_utc": metadata.get(
                    "target_time_utc"
                ),
                "frame_window_start_utc": metadata.get(
                    "window_start_utc"
                ),
                "frame_window_end_utc": metadata.get(
                    "window_end_utc"
                ),
                "frame_candidate_count": metadata.get(
                    "candidate_count"
                ),
                "frame_scene_id": metadata.get("scene_id"),
                "frame_asset_id": metadata.get("asset_id"),
                "frame_acquisition_time_utc": metadata.get(
                    "acquisition_time_utc"
                ),
                "frame_cloudy_pixel_percentage": metadata.get(
                    "cloud_pct"
                ),
                "frame_mgrs_tile": metadata.get("mgrs_tile"),
                "frame_product_id": metadata.get("product_id"),
                "frame_spacecraft_name": metadata.get(
                    "spacecraft_name"
                ),
                "frame_absolute_target_difference_seconds": (
                    metadata.get(
                        "absolute_target_difference_seconds"
                    )
                ),
                "frame_match_method": metadata.get("match_method"),
            }

            if image is None:
                frame_record.update(
                    {
                        "frame_status": "no_temporal_candidate",
                        "download_status": "not_attempted",
                        "tiff_valid": False,
                        "scl_clear_fraction": np.nan,
                    }
                )
                manifest_rows.append(frame_record)
                print(f"  {frame_name}: no temporal candidate")
                continue

            try:
                qa_asset_id = (
                    metadata.get("asset_id")
                    or normalize_asset_id(metadata.get("scene_id"))
                )
                clear_fraction = scl_clear_fraction(
                    ee.Image(qa_asset_id),
                    latitude,
                    longitude,
                )
            except Exception as exc:
                clear_fraction = float("nan")
                frame_record["scl_qa_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )

            download_info = download_geotiff(
                image=image,
                latitude=latitude,
                longitude=longitude,
                output_path=output_path,
                overwrite=args.overwrite,
            )

            frame_record.update(download_info)
            frame_record["scl_clear_fraction"] = clear_fraction
            frame_record["frame_status"] = (
                "ready"
                if download_info.get("tiff_valid")
                else "failed"
            )
            manifest_rows.append(frame_record)

            clear_text = (
                f"{clear_fraction:.3f}"
                if np.isfinite(clear_fraction)
                else "NA"
            )
            print(
                f"  {frame_name}: {frame_record['frame_status']} | "
                f"scene={metadata.get('scene_id')} | "
                f"cloud={metadata.get('cloud_pct')} | "
                f"SCL clear={clear_text}"
            )

    manifest = pd.DataFrame(manifest_rows)
    download_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(download_manifest_path, index=False)

    if manifest.empty:
        raise RuntimeError("No frame records were created.")

    manifest["tiff_valid_bool"] = (
        manifest["tiff_valid"].fillna(False).astype(bool)
    )

    ready_pivot = (
        manifest.groupby("sample_id")
        .agg(
            ready_frames=("tiff_valid_bool", "sum"),
            minimum_scl_clear_fraction=(
                "scl_clear_fraction",
                "min",
            ),
        )
        .reset_index()
    )

    frame_paths = (
        manifest[manifest["tiff_valid_bool"]]
        .pivot(
            index="sample_id",
            columns="frame",
            values="frame_output_path",
        )
        .reset_index()
    )

    eval_table = (
        source.merge(
            ready_pivot,
            on="sample_id",
            how="left",
        )
        .merge(
            frame_paths,
            on="sample_id",
            how="left",
        )
    )

    eval_table = eval_table[
        eval_table["ready_frames"].fillna(0).eq(3)
    ].copy()

    eval_table["id"] = eval_table["sample_id"].astype(str)
    eval_table["s2_0_path"] = eval_table["t0"]
    eval_table["s2_90_path"] = eval_table["t90"]
    eval_table["s2_360_path"] = eval_table["t360"]

    metadata_columns = [
        "id",
        "sample_id",
        "label",
        "label_provenance",
        "site",
        "s2_0_path",
        "s2_90_path",
        "s2_360_path",
        "scene_cluster_id",
        "independent_acquisition_unit",
        "scene_id",
        "t0_asset_id_resolved",
        "acquisition_time_utc",
        "event_id",
        "flight_id",
        "plume_id",
        "emission_kg_hr",
        "lat",
        "lon",
        "minimum_scl_clear_fraction",
        "ground_truth_source",
        "controlled_release_verified",
        "evaluation_role",
        "temporal_tier",
        "absolute_time_difference_hours",
        "patches_in_same_s2_scene",
    ]
    eval_table = eval_table[
        [
            column
            for column in metadata_columns
            if column in eval_table.columns
        ]
    ].copy()

    eval_csv_path.parent.mkdir(parents=True, exist_ok=True)
    eval_table.to_csv(eval_csv_path, index=False)

    strict = eval_table[
        pd.to_numeric(
            eval_table["minimum_scl_clear_fraction"],
            errors="coerce",
        ).ge(float(args.scl_threshold))
    ].copy()
    strict.to_csv(eval_scl80_path, index=False)

    expected_frames = len(source) * len(FRAME_SPECS)
    valid_frames = int(manifest["tiff_valid_bool"].sum())
    ready_samples = len(eval_table)
    ready_acquisitions = (
        eval_table["scene_cluster_id"].nunique()
        if len(eval_table)
        else 0
    )

    summary_lines = [
        "MethaneAIR–Sentinel-2 P1 MethaneFuse preparation",
        "=" * 76,
        f"Input manifest: {input_path}",
        f"Requested event-centered samples: {len(source)}",
        (
            "Requested independent S2 acquisitions: "
            f"{source['scene_cluster_id'].nunique()}"
        ),
        f"Expected temporal frames: {expected_frames}",
        f"Valid temporal frames: {valid_frames}",
        f"Samples with all three frames: {ready_samples}",
        f"Independent acquisitions represented: {ready_acquisitions}",
        (
            f"Strict SCL>={args.scl_threshold:.2f} samples: "
            f"{len(strict)}"
        ),
        "",
        "Frame download status:",
        manifest["download_status"]
        .value_counts(dropna=False)
        .to_string(),
        "",
        "Frame validity by temporal slot:",
        manifest.groupby("frame")["tiff_valid_bool"]
        .agg(["count", "sum"])
        .to_string(),
        "",
        "Ready-sample emission summary (kg/h):",
        (
            eval_table["emission_kg_hr"].describe().to_string()
            if len(eval_table)
            else "NONE"
        ),
        "",
        "Important:",
        "- All rows are MethaneAIR-referenced observational positives.",
        "- They are not controlled-release-verified positives.",
        "- Multiple event-centered patches may share one S2 acquisition.",
        "- Report both patch-level N and independent-acquisition N.",
        "- AUROC, FPR, specificity, and balanced accuracy are undefined",
        "  for this positive-only evaluation set.",
    ]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "\n".join(summary_lines))
    print("\nCreated:")
    print(download_manifest_path)
    print(eval_csv_path)
    print(eval_scl80_path)
    print(summary_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
