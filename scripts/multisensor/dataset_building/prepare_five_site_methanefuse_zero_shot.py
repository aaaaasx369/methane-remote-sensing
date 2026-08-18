from __future__ import annotations

import argparse
import io
import json
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

GROUND_TRUTH_CANDIDATES = [
    PROJECT_ROOT
    / "outputs/601_five_site_ground_truth_for_methanefuse.csv",
]

PATCH_ROOT = (
    PROJECT_ROOT
    / "methanefuse_input"
    / "s2_12band_five_site_zero_shot"
)

DOWNLOAD_MANIFEST = (
    PROJECT_ROOT
    / "outputs/600_five_site_methanefuse_download_manifest.csv"
)

EVAL_CSV = (
    METHANEFUSE_ROOT
    / "data/custom/five_site_zero_shot_eval.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs/600_five_site_methanefuse_summary.txt"
)

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"


LOCAL_PATCH_ASSET_OVERRIDES = {
    "MA_S2_patch_17": "COPERNICUS/S2_SR_HARMONIZED/20230903T160829_20230903T161849_T17SND",
    "MA_S2_patch_4": "COPERNICUS/S2_SR_HARMONIZED/20230730T160831_20230730T161947_T17SND",
    "MA_S2_patch_41": "COPERNICUS/S2_SR_HARMONIZED/20231002T173131_20231002T173646_T13SFR",
}


# MethaneFuse uses its s2_12band sensor configuration. Keep this exact order.
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
    parser = argparse.ArgumentParser(
        description=(
            "Download real 12-band Sentinel-2 t0/t-90/t-360 patches "
            "for the exact controlled-release scene-level ground truth "
            "and build a MethaneFuse wide-table evaluation CSV."
        )
    )
    parser.add_argument(
        "--project",
        default="methane-release-gee",
        help="Google Earth Engine project ID.",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=80.0,
        help=(
            "Maximum scene-level CLOUDY_PIXEL_PERCENTAGE for temporal "
            "reference searches. The exact t0 scene is never replaced."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload TIFFs even when an existing valid file is present.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for testing.",
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


def choose_ground_truth_path() -> Path:
    for path in GROUND_TRUTH_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "找不到 scene-level ground truth。檢查：\n"
        + "\n".join(str(path) for path in GROUND_TRUTH_CANDIDATES)
    )


def parse_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def safe_name(text: Any) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    return value.strip("_")[:180]


def load_ground_truth(limit: int | None) -> tuple[pd.DataFrame, Path]:
    path = choose_ground_truth_path()
    df = pd.read_csv(path, low_memory=False)

    required = [
        "sensor",
        "scene_id",
        "site",
        "acquisition_time_utc",
        "physical_release_gt",
        "lat",
        "lon",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} 缺少必要欄位：{missing}\n"
            f"目前欄位：{df.columns.tolist()}"
        )

    df = df[
        df["sensor"].astype(str).str.strip().eq("Sentinel-2")
    ].copy()

    if "binary_gt_usable" in df.columns:
        df = df[parse_bool(df["binary_gt_usable"])].copy()

    df["acquisition_time_utc"] = pd.to_datetime(
        df["acquisition_time_utc"],
        utc=True,
        errors="coerce",
    )
    df["physical_release_gt"] = pd.to_numeric(
        df["physical_release_gt"],
        errors="coerce",
    )
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.dropna(
        subset=[
            "scene_id",
            "acquisition_time_utc",
            "physical_release_gt",
            "lat",
            "lon",
        ]
    ).copy()

    df["physical_release_gt"] = df["physical_release_gt"].astype(int)

    if not set(df["physical_release_gt"].unique()).issubset({0, 1}):
        raise ValueError("physical_release_gt 必須只有 0/1。")

    # One model row per unique satellite scene.
    df = (
        df.sort_values(["scene_id", "acquisition_time_utc"])
        .drop_duplicates(subset=["scene_id"], keep="first")
        .reset_index(drop=True)
    )

    if limit is not None:
        df = df.head(limit).copy()

    df["external_eval_id"] = [
        f"five_site_s2_{index:03d}_{safe_name(scene_id)}"
        for index, scene_id in enumerate(df["scene_id"], start=1)
    ]

    return df, path


def image_asset_id(scene_id: str) -> str:
    text = str(scene_id).strip()
    if text.startswith("COPERNICUS/"):
        return text
    return f"{S2_COLLECTION}/{text}"


def millis_to_timestamp(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, unit="ms", utc=True)



# --- five-site t0 asset resolver ---
def resolve_t0_asset(row: pd.Series) -> dict:
    """
    Resolve the real Earth Engine Sentinel-2 t0 asset.

    Two-site controlled-release rows usually contain a real Earth Engine
    scene ID. MethaneAIR rows may instead contain a local patch identifier
    such as MA_S2_patch_102. In that case, find the nearest real S2 scene
    using acquisition time and source coordinates.
    """
    scene_id = str(row["scene_id"]).strip()

    if scene_id in LOCAL_PATCH_ASSET_OVERRIDES:
        asset_id = LOCAL_PATCH_ASSET_OVERRIDES[scene_id]
        image = ee.Image(asset_id)

        # Force validation that the asset exists.
        image.get("system:index").getInfo()

        metadata = get_image_metadata(image)

        requested_time = pd.to_datetime(
            row["acquisition_time_utc"],
            utc=True,
            errors="coerce",
        )

        resolved_time = pd.to_datetime(
            metadata["acquisition_time_utc"],
            utc=True,
            errors="coerce",
        )

        difference_minutes = abs(
            (
                resolved_time
                - requested_time
            ).total_seconds()
        ) / 60.0

        print(
            "[t0 override]",
            scene_id,
            "->",
            asset_id,
            f"({difference_minutes:.1f} min; nearest-scene match)",
        )

        return {
            "asset_id": asset_id,
            "image": image,
            "metadata": metadata,
            "match_method": "manual_nearest_scene_override",
            "time_difference_minutes": difference_minutes,
        }


    direct_candidates = []

    if scene_id.startswith("COPERNICUS/"):
        direct_candidates.append(scene_id)
    elif scene_id and not scene_id.lower().startswith("ma_s2_patch"):
        direct_candidates.append(
            f"{S2_COLLECTION}/{scene_id}"
        )

    direct_error = None

    for candidate in direct_candidates:
        try:
            image = ee.Image(candidate)

            # Force an Earth Engine request so a nonexistent image is caught.
            system_index = image.get("system:index").getInfo()

            if system_index:
                metadata = get_image_metadata(image)

                return {
                    "asset_id": candidate,
                    "image": image,
                    "metadata": metadata,
                    "match_method": "scene_id_exact",
                    "time_difference_minutes": 0.0,
                }

        except Exception as exc:
            direct_error = exc

    acquisition_time = pd.to_datetime(
        row["acquisition_time_utc"],
        utc=True,
        errors="coerce",
    )

    if pd.isna(acquisition_time):
        raise ValueError(
            f"無法解析 acquisition_time_utc："
            f"{row.get('acquisition_time_utc')}"
        )

    lat = float(row["lat"])
    lon = float(row["lon"])

    point = ee.Geometry.Point([lon, lat])
    target_ms = int(acquisition_time.timestamp() * 1000)

    selected_image = None
    selected_info = None

    # Normally the correct scene should be within minutes.
    # The second window is only a fallback for boundary/time-format issues.
    for window_hours in [12, 48]:
        start_time = (
            acquisition_time
            - pd.Timedelta(hours=window_hours)
        )

        end_time = (
            acquisition_time
            + pd.Timedelta(hours=window_hours)
        )

        collection = (
            ee.ImageCollection(S2_COLLECTION)
            .filterBounds(point)
            .filterDate(
                start_time.isoformat(),
                end_time.isoformat(),
            )
        )

        def add_time_difference(image):
            difference = (
                ee.Number(image.get("system:time_start"))
                .subtract(target_ms)
                .abs()
            )

            return image.set(
                "_time_difference_ms",
                difference,
            )

        collection = (
            collection
            .map(add_time_difference)
            .sort("_time_difference_ms")
        )

        count = int(collection.size().getInfo())

        if count == 0:
            continue

        selected_image = ee.Image(collection.first())

        selected_info = ee.Dictionary({
            "system_id": selected_image.get("system:id"),
            "system_index": selected_image.get("system:index"),
            "system_time_start": selected_image.get(
                "system:time_start"
            ),
            "product_id": selected_image.get("PRODUCT_ID"),
            "time_difference_ms": selected_image.get(
                "_time_difference_ms"
            ),
        }).getInfo()

        break

    if selected_image is None or selected_info is None:
        raise RuntimeError(
            "找不到符合 acquisition time 與座標的 Sentinel-2 t0 scene。"
            f" scene_id={scene_id},"
            f" time={acquisition_time},"
            f" lat={lat}, lon={lon}."
            f" Direct-load error={direct_error}"
        )

    system_index = selected_info.get("system_index")
    system_id = selected_info.get("system_id")

    if system_id:
        asset_id = str(system_id)
    elif system_index:
        asset_id = f"{S2_COLLECTION}/{system_index}"
    else:
        raise RuntimeError(
            "Earth Engine 找到影像，但沒有 system:id 或 system:index。"
        )

    difference_ms = float(
        selected_info.get("time_difference_ms") or 0
    )

    difference_minutes = difference_ms / 60000.0

    # A much larger difference would mean this is probably not the intended
    # acquisition and should not silently be treated as exact t0.
    if difference_minutes > 360:
        raise RuntimeError(
            "找到的 Sentinel-2 scene 與指定 acquisition time "
            f"相差 {difference_minutes:.1f} 分鐘，超過 6 小時。"
            f" Local scene_id={scene_id}, matched asset={asset_id}"
        )

    metadata = get_image_metadata(selected_image)

    print(
        "[t0 resolved]",
        scene_id,
        "->",
        asset_id,
        f"({difference_minutes:.1f} min)",
    )

    return {
        "asset_id": asset_id,
        "image": selected_image,
        "metadata": metadata,
        "match_method": "time_coordinate_match",
        "time_difference_minutes": difference_minutes,
    }


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

    raw_asset_id = (
        metadata.get("asset_id")
        or metadata.get("scene_id")
    )

    if raw_asset_id:
        metadata["asset_id"] = image_asset_id(
            str(raw_asset_id)
        )

    if metadata.get("time_start") is not None:
        metadata["acquisition_time_utc"] = millis_to_timestamp(
            metadata["time_start"]
        )
    else:
        metadata["acquisition_time_utc"] = pd.NaT

    return metadata


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

    candidates = []
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
                "_sort_key": (diff_seconds, cloud, str(item["asset_id"])),
            }
        )

    selected = min(candidates, key=lambda row: row["_sort_key"])
    selected.pop("_sort_key", None)

    raw_asset_id = (
        selected.get("asset_id")
        or selected.get("scene_id")
    )

    if not raw_asset_id:
        raise RuntimeError(
            "Temporal Sentinel-2 candidate has no asset ID "
            "or system:index."
        )

    selected["asset_id"] = image_asset_id(
        str(raw_asset_id)
    )

    image = ee.Image(
        selected["asset_id"]
    ).select(S2_BANDS)

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

    # Clear-like land/water classes: vegetation, bare soil, water,
    # and unclassified. This is a QA indicator, not a hard deletion rule.
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
    result = {
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

    result["tiff_valid"] = bool(
        expected_channel_shape
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

    # Keep raw Sentinel-2 surface-reflectance DN values as uint16.
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
    initialize_ee(args.project)

    gt, gt_path = load_ground_truth(args.limit)

    print(f"Ground truth: {gt_path}")
    print(f"Unique S2 scenes selected: {len(gt)}")
    print("\nLabel counts:")
    print(gt["physical_release_gt"].value_counts().sort_index())

    manifest_rows: list[dict[str, Any]] = []

    for position, (_, row) in enumerate(gt.iterrows(), start=1):
        scene_id = str(row["scene_id"])
        t0_time = row["acquisition_time_utc"]
        latitude = float(row["lat"])
        longitude = float(row["lon"])
        sample_id = str(row["external_eval_id"])

        print(
            f"\n[{position}/{len(gt)}] {sample_id}\n"
            f"  scene={scene_id}\n"
            f"  site={row['site']}\n"
            f"  label={row['physical_release_gt']}"
        )

        base = row.to_dict()
        frame_paths: dict[str, str] = {}
        row_ok = True

        for frame_name, spec in FRAME_SPECS.items():
            output_path = (
                PATCH_ROOT
                / sample_id
                / f"{sample_id}__{frame_name}.tif"
            )

            if frame_name == "t0":
                # Resolve every t0 scene before loading it.
                # Real S2 scene IDs load directly; MA_S2_patch IDs
                # are matched using acquisition time and coordinates.
                resolved_t0 = resolve_t0_asset(row)
                asset_id = resolved_t0["asset_id"]
                image = resolved_t0["image"].select(S2_BANDS)
                metadata = resolved_t0["metadata"]
                metadata["candidate_count"] = 1
                metadata["target_time_utc"] = t0_time
                metadata["window_start_utc"] = t0_time
                metadata["window_end_utc"] = t0_time
                metadata["absolute_target_difference_seconds"] = abs(
                    (
                        metadata["acquisition_time_utc"]
                        - t0_time
                    ).total_seconds()
                )
            else:
                image, metadata = choose_temporal_image(
                    latitude=latitude,
                    longitude=longitude,
                    t0=t0_time,
                    offset_days=int(spec["offset_days"]),
                    half_window_days=int(
                        spec["half_window_days"]
                    ),
                    max_cloud=float(args.max_cloud),
                )

            frame_record = {
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
                "frame_absolute_target_difference_seconds":
                    metadata.get(
                        "absolute_target_difference_seconds"
                    ),
            }

            if image is None:
                frame_record.update(
                    {
                        "frame_status": "no_temporal_candidate",
                        "download_status": "not_attempted",
                        "tiff_valid": False,
                    }
                )
                manifest_rows.append(frame_record)
                row_ok = False
                print(f"  {frame_name}: no temporal candidate")
                continue

            try:
                clear_fraction = scl_clear_fraction(
                    ee.Image(
                        metadata.get("asset_id")
                        or image_asset_id(
                            str(metadata.get("scene_id"))
                        )
                    ),
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

            if not download_info.get("tiff_valid"):
                row_ok = False

            if download_info.get("tiff_valid"):
                frame_paths[frame_name] = str(
                    output_path.resolve()
                )

            print(
                f"  {frame_name}: "
                f"{frame_record['frame_status']} | "
                f"scene={metadata.get('scene_id')} | "
                f"cloud={metadata.get('cloud_pct')} | "
                f"SCL clear={clear_fraction:.3f}"
                if np.isfinite(clear_fraction)
                else
                f"  {frame_name}: "
                f"{frame_record['frame_status']} | "
                f"scene={metadata.get('scene_id')} | "
                f"cloud={metadata.get('cloud_pct')} | "
                "SCL clear=NA"
            )

        base["all_three_frames_ready"] = row_ok
        base["s2_0_path"] = frame_paths.get("t0")
        base["s2_90_path"] = frame_paths.get("t90")
        base["s2_360_path"] = frame_paths.get("t360")

    manifest = pd.DataFrame(manifest_rows)
    DOWNLOAD_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(DOWNLOAD_MANIFEST, index=False)

    ready_pivot = (
        manifest.groupby("external_eval_id")
        .agg(
            ready_frames=("tiff_valid", "sum"),
            minimum_scl_clear_fraction=(
                "scl_clear_fraction",
                "min",
            ),
        )
        .reset_index()
    )

    frame_paths = (
        manifest[
            manifest["tiff_valid"].fillna(False).astype(bool)
        ]
        .pivot(
            index="external_eval_id",
            columns="frame",
            values="frame_output_path",
        )
        .reset_index()
    )

    eval_table = gt.merge(
        ready_pivot,
        on="external_eval_id",
        how="left",
    ).merge(
        frame_paths,
        on="external_eval_id",
        how="left",
    )

    eval_table = eval_table[
        eval_table["ready_frames"].fillna(0).eq(3)
    ].copy()

    rename = {
        "external_eval_id": "id",
        "physical_release_gt": "label",
        "t0": "s2_0_path",
        "t90": "s2_90_path",
        "t360": "s2_360_path",
    }

    # The standardized five-site source may already contain columns named
    # label, id, or s2_*_path. Drop those destination columns before
    # renaming to prevent duplicate column names.
    for source_column, destination_column in rename.items():
        if (
            source_column in eval_table.columns
            and destination_column in eval_table.columns
            and source_column != destination_column
        ):
            eval_table = eval_table.drop(
                columns=[destination_column]
            )

    # Prevent duplicate destination columns before renaming.
    # Example: the source table already contains "label", while
    # physical_release_gt is also about to be renamed to "label".
    for source_column, destination_column in rename.items():
        if (
            source_column in eval_table.columns
            and destination_column in eval_table.columns
            and source_column != destination_column
        ):
            eval_table = eval_table.drop(
                columns=[destination_column]
            )

    eval_table = eval_table.rename(columns=rename)

    duplicate_columns = (
        eval_table.columns[
            eval_table.columns.duplicated()
        ].tolist()
    )

    if duplicate_columns:
        raise ValueError(
            "eval_table 仍有重複欄位："
            f"{duplicate_columns}"
        )

    duplicate_columns = (
        eval_table.columns[
            eval_table.columns.duplicated()
        ].tolist()
    )

    if duplicate_columns:
        raise ValueError(
            "eval_table 仍有重複欄位："
            f"{duplicate_columns}"
        )

    # MethaneFuse loader and later per-site auditing need both IDs
    # and the distinction between physical-release and plume-reference labels.
    eval_table["sample_id"] = eval_table["id"].astype(str)

    if "label_provenance" not in eval_table.columns:
        provenance_map = {
            "Casa_Grande": "physical_release",
            "Ehrenberg": "physical_release",
            "MA_site_038": "plume_reference",
            "MA_site_043": "plume_reference",
            "MA_site_073": "plume_reference",
        }
        eval_table["label_provenance"] = (
            eval_table["site"].map(provenance_map)
        )

    required_eval_columns = [
        "id",
        "sample_id",
        "label",
        "label_provenance",
        "site",
        "s2_0_path",
        "s2_90_path",
        "s2_360_path",
    ]

    missing_eval_columns = [
        column
        for column in required_eval_columns
        if column not in eval_table.columns
    ]

    if missing_eval_columns:
        raise ValueError(
            "MethaneFuse evaluation table 缺少必要欄位："
            f"{missing_eval_columns}"
        )

    metadata_columns = [
        "id",
        "sample_id",
        "label",
        "label_provenance",
        "s2_0_path",
        "s2_90_path",
        "s2_360_path",
        "scene_id",
        "site",
        "acquisition_time_utc",
        "lat",
        "lon",
        "minimum_scl_clear_fraction",
        "consensus_release_rate_kg_h",
        "release_rates_all_kg_h",
        "emission_rate_usable",
        "emission_bin_usable",
        "consensus_emission_bin",
        "scene_ground_truth_status",
        "release_interval_id",
        "release_interval_ids_all",
    ]

    eval_table = eval_table[
        [
            column
            for column in metadata_columns
            if column in eval_table.columns
        ]
    ].copy()

    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    eval_table.to_csv(EVAL_CSV, index=False)

    summary_lines = [
        "MethaneFuse exact Sentinel-2 external evaluation preparation",
        "=" * 76,
        f"Ground-truth input: {gt_path}",
        f"Unique binary-GT S2 scenes requested: {len(gt)}",
        f"Scenes with all three real temporal frames: {len(eval_table)}",
        "",
        "Requested label counts:",
        gt["physical_release_gt"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string(),
        "",
        "Ready evaluation label counts:",
        (
            eval_table["label"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
            if len(eval_table)
            else "NONE"
        ),
        "",
        "Frame download status:",
        manifest["download_status"]
        .value_counts(dropna=False)
        .to_string(),
        "",
        "Frame validity by temporal slot:",
        manifest.groupby("frame")["tiff_valid"]
        .agg(["count", "sum"])
        .to_string(),
        "",
        "Important:",
        "- t0 is the exact controlled-release Sentinel-2 scene.",
        "- t90 and t360 are real pre-event Sentinel-2 scenes selected near",
        "  90 and 360 days before t0; the exact selected IDs are audited.",
        "- The output is suitable for an external binary evaluation after",
        "  inspecting cloud/SCL QA. It does not turn physical release ON/OFF",
        "  into a guaranteed visible-plume label.",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n" + "\n".join(summary_lines))
    print("\nCreated:")
    print(DOWNLOAD_MANIFEST)
    print(EVAL_CSV)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
