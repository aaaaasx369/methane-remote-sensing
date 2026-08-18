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
from typing import Any, Iterable
from urllib.request import urlopen

import ee
import numpy as np
import pandas as pd
import tifffile


PROJECT_ROOT = Path("/Users/happydoraaa/methane_release_project")
METHANEFUSE_ROOT = Path("/Users/happydoraaa/MethaneFuse")

GROUND_TRUTH_CANDIDATES = [
    PROJECT_ROOT / "outputs/108_landsat_high_emission_core_manifest.csv",
]

PATCH_ROOT = (
    PROJECT_ROOT
    / "methanefuse_input"
    / "l89_7band_high_emission_zero_shot"
)

DOWNLOAD_MANIFEST = (
    PROJECT_ROOT
    / "outputs/602_landsat89_methanefuse_download_manifest.csv"
)

EVAL_CSV = (
    METHANEFUSE_ROOT
    / "data/custom/landsat89_high_emission_zero_shot_eval.csv"
)

STRICT_EVAL_CSV = (
    METHANEFUSE_ROOT
    / "data/custom/landsat89_high_emission_zero_shot_eval_qa80.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs/602_landsat89_methanefuse_summary.txt"
)

L8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
L9_COLLECTION = "LANDSAT/LC09/C02/T1_L2"

SENSOR_COLLECTIONS = {
    "Landsat-8": L8_COLLECTION,
    "Landsat-9": L9_COLLECTION,
}

# MethaneFuse landsat89_7band order. Do not change this order.
L89_BANDS = [
    "SR_B1",  # Coastal aerosol
    "SR_B2",  # Blue
    "SR_B3",  # Green
    "SR_B4",  # Red
    "SR_B5",  # NIR
    "SR_B6",  # SWIR 1
    "SR_B7",  # SWIR 2
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

# 480 m checkpoint footprint at 30 m Landsat GSD.
PATCH_METERS = 480
PATCH_PIXELS = 16
HALF_PATCH_METERS = PATCH_METERS / 2.0

# QA_PIXEL bits 0–5: fill, dilated cloud, cirrus, cloud,
# cloud shadow, and snow. A pixel is locally clear when all are zero.
QA_BAD_BITS_MASK = sum(1 << bit for bit in range(6))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download real 7-band Landsat 8/9 t0/t-90/t-360 patches "
            "for the high-emission controlled-release manifest and build "
            "MethaneFuse wide-table evaluation CSVs."
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
            "Maximum scene-level CLOUD_COVER for t90/t360 searches. "
            "The resolved t0 scene is never replaced because of cloud cover."
        ),
    )
    parser.add_argument(
        "--min-clear",
        type=float,
        default=0.80,
        help=(
            "Minimum local QA clear fraction required in every frame for "
            "the strict sensitivity-analysis CSV. The raw ready CSV keeps "
            "all samples with three valid TIFFs."
        ),
    )
    parser.add_argument(
        "--temporal-sensor-mode",
        choices=["same", "same_then_any", "any"],
        default="same_then_any",
        help=(
            "Sensor rule for t90/t360: same=only the t0 spacecraft; "
            "same_then_any=prefer the same spacecraft then fall back to "
            "either Landsat-8/9; any=search both immediately."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload TIFFs even when a valid output already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional observation limit for a smoke test.",
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
        "找不到 Landsat ground-truth manifest。檢查：\n"
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


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() not in {"", "nan", "none", "null"}


def unique_nonempty(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    for value in values:
        if not nonempty(value):
            continue
        text = str(value).strip()
        if text not in output:
            output.append(text)
    return output


def normalize_sensor(value: Any) -> str:
    text = str(value).strip().lower().replace("_", "-")
    compact = re.sub(r"[^a-z0-9]", "", text)

    if compact in {"landsat8", "l8", "lc08"}:
        return "Landsat-8"
    if compact in {"landsat9", "l9", "lc09"}:
        return "Landsat-9"

    raise ValueError(f"無法辨識 Landsat sensor：{value!r}")


def normalize_site(value: Any) -> str:
    text = str(value).strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", text)

    if "casagrande" in compact:
        return "Casa_Grande"
    if "ehrenberg" in compact:
        return "Ehrenberg"

    return safe_name(value)


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse mixed ISO timestamps with and without fractional seconds."""
    try:
        return pd.to_datetime(
            series,
            utc=True,
            errors="coerce",
            format="mixed",
        )
    except (TypeError, ValueError):
        # Compatibility fallback for older pandas versions.
        return series.map(
            lambda value: pd.to_datetime(
                value,
                utc=True,
                errors="coerce",
            )
        )


def load_ground_truth(limit: int | None) -> tuple[pd.DataFrame, Path]:
    path = choose_ground_truth_path()
    df = pd.read_csv(path, low_memory=False)

    label_column = first_existing_column(
        df,
        ["final_scene_label", "label", "final_label"],
    )
    site_column = first_existing_column(
        df,
        ["site_key_normalized", "site_name_normalized", "site_name", "site_key"],
    )
    sensor_column = first_existing_column(
        df,
        ["landsat_sensor", "landsat_sensor_review", "sensor", "SENSOR_ID"],
    )
    time_column = first_existing_column(
        df,
        ["acquisition_time_utc", "landsat_image_time_utc", "datetime_utc"],
    )
    lat_column = first_existing_column(df, ["lat", "site_lat"])
    lon_column = first_existing_column(df, ["lon", "site_lon"])
    key_column = first_existing_column(
        df,
        ["scene_key", "overpass_id", "raster_group_id", "event_id"],
    )

    resolved = {
        "label": label_column,
        "site": site_column,
        "sensor": sensor_column,
        "time": time_column,
        "lat": lat_column,
        "lon": lon_column,
        "key": key_column,
    }
    missing = [name for name, column in resolved.items() if column is None]
    if missing:
        raise ValueError(
            f"{path.name} 無法解析必要欄位：{missing}\n"
            f"目前欄位：{df.columns.tolist()}"
        )

    assert label_column is not None
    assert site_column is not None
    assert sensor_column is not None
    assert time_column is not None
    assert lat_column is not None
    assert lon_column is not None
    assert key_column is not None

    df = df.copy()
    df["model_label"] = pd.to_numeric(df[label_column], errors="coerce")
    df["model_site"] = df[site_column].map(normalize_site)
    df["model_sensor"] = df[sensor_column].map(normalize_sensor)
    df["model_acquisition_time_utc"] = parse_datetime_series(
        df[time_column]
    )
    df["model_lat"] = pd.to_numeric(df[lat_column], errors="coerce")
    df["model_lon"] = pd.to_numeric(df[lon_column], errors="coerce")
    df["model_scene_key"] = df[key_column].astype(str).str.strip()

    df = df.dropna(
        subset=[
            "model_label",
            "model_acquisition_time_utc",
            "model_lat",
            "model_lon",
        ]
    ).copy()

    df["model_label"] = df["model_label"].astype(int)
    if not set(df["model_label"].unique()).issubset({0, 1}):
        raise ValueError("Landsat label 必須只有 0/1。")

    df = (
        df.sort_values(["model_scene_key", "model_acquisition_time_utc"])
        .drop_duplicates(subset=["model_scene_key"], keep="first")
        .reset_index(drop=True)
    )

    if limit is not None:
        df = df.head(limit).copy()

    df["external_eval_id"] = [
        f"landsat89_{index:03d}_{safe_name(scene_key)}"
        for index, scene_key in enumerate(df["model_scene_key"], start=1)
    ]

    return df, path


def millis_to_timestamp(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, unit="ms", utc=True)


def collection_for_sensor(sensor: str) -> str:
    try:
        return SENSOR_COLLECTIONS[sensor]
    except KeyError as exc:
        raise ValueError(f"不支援的 Landsat sensor：{sensor}") from exc


def infer_collection_from_index(index: str) -> str | None:
    text = str(index).strip().upper()
    if text.startswith("LC08_") or text.startswith("LC08/"):
        return L8_COLLECTION
    if text.startswith("LC09_") or text.startswith("LC09/"):
        return L9_COLLECTION
    return None


def image_asset_id(scene_id: str, sensor: str | None = None) -> str:
    text = str(scene_id).strip()
    if text.startswith("LANDSAT/"):
        return text

    collection = infer_collection_from_index(text)
    if collection is None and sensor is not None:
        collection = collection_for_sensor(sensor)
    if collection is None:
        raise ValueError(
            f"無法從 scene ID 推斷 Landsat collection：{scene_id!r}"
        )
    return f"{collection}/{text}"


def get_image_metadata(image: ee.Image) -> dict[str, Any]:
    metadata = ee.Dictionary(
        {
            "asset_id": image.get("system:id"),
            "scene_id": image.get("system:index"),
            "time_start": image.get("system:time_start"),
            "cloud_pct": image.get("CLOUD_COVER"),
            "cloud_land_pct": image.get("CLOUD_COVER_LAND"),
            "product_id": image.get("LANDSAT_PRODUCT_ID"),
            "legacy_scene_id": image.get("LANDSAT_SCENE_ID"),
            "spacecraft_id": image.get("SPACECRAFT_ID"),
            "wrs_path": image.get("WRS_PATH"),
            "wrs_row": image.get("WRS_ROW"),
            "processing_level": image.get("PROCESSING_LEVEL"),
        }
    ).getInfo()

    scene_id = metadata.get("scene_id")
    spacecraft = str(metadata.get("spacecraft_id") or "")
    sensor = (
        "Landsat-9"
        if "9" in spacecraft
        else "Landsat-8"
        if "8" in spacecraft
        else None
    )

    raw_asset_id = metadata.get("asset_id")
    if not raw_asset_id and scene_id:
        raw_asset_id = image_asset_id(str(scene_id), sensor=sensor)
    metadata["asset_id"] = raw_asset_id
    metadata["sensor"] = sensor

    if metadata.get("time_start") is not None:
        metadata["acquisition_time_utc"] = millis_to_timestamp(
            metadata["time_start"]
        )
    else:
        metadata["acquisition_time_utc"] = pd.NaT

    return metadata


def add_time_difference(
    collection: ee.ImageCollection,
    target: pd.Timestamp,
) -> ee.ImageCollection:
    target_ms = int(target.timestamp() * 1000)

    def mapper(image: ee.Image) -> ee.Image:
        difference = (
            ee.Number(image.get("system:time_start"))
            .subtract(target_ms)
            .abs()
        )
        return image.set("_time_difference_ms", difference)

    return collection.map(mapper)


def nearest_image_from_collection(
    collection: ee.ImageCollection,
    target: pd.Timestamp,
) -> tuple[ee.Image | None, int]:
    collection = add_time_difference(collection, target).sort(
        "_time_difference_ms"
    )
    count = int(collection.size().getInfo())
    if count == 0:
        return None, 0
    return ee.Image(collection.first()), count


def direct_asset_candidates(row: pd.Series) -> list[str]:
    candidates: list[str] = []

    # Already-complete Earth Engine asset IDs.
    for column in ["frame_asset_id", "asset_id"]:
        if column in row.index and nonempty(row.get(column)):
            text = str(row.get(column)).strip()
            if text.startswith("LANDSAT/"):
                candidates.append(text)

    collection_values = unique_nonempty(
        row.get(column)
        for column in [
            "gee_collection_id",
            "collection_id",
        ]
        if column in row.index
    )
    index_values = unique_nonempty(
        row.get(column)
        for column in [
            "gee_system:index",
            "system:index",
        ]
        if column in row.index
    )

    for index in index_values:
        if index.startswith("LANDSAT/"):
            candidates.append(index)
            continue
        inferred = infer_collection_from_index(index)
        if inferred:
            candidates.append(f"{inferred}/{index}")
        for collection in collection_values:
            if collection.startswith("LANDSAT/"):
                candidates.append(f"{collection.rstrip('/')}/{index}")

    return unique_nonempty(candidates)


def exact_property_candidates(row: pd.Series) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    for column in [
        "gee_LANDSAT_PRODUCT_ID",
        "LANDSAT_PRODUCT_ID",
        "landsat_product_id_normalized",
    ]:
        if column in row.index and nonempty(row.get(column)):
            pairs.append(("LANDSAT_PRODUCT_ID", str(row.get(column)).strip()))

    # Some rows use the product ID as scene_key.
    scene_key = str(row.get("model_scene_key", "")).strip()
    if re.match(r"^LC0[89]_L2", scene_key, flags=re.IGNORECASE):
        pairs.append(("LANDSAT_PRODUCT_ID", scene_key))

    for column in ["gee_LANDSAT_SCENE_ID", "LANDSAT_SCENE_ID"]:
        if column in row.index and nonempty(row.get(column)):
            pairs.append(("LANDSAT_SCENE_ID", str(row.get(column)).strip()))

    return list(dict.fromkeys(pairs))


def resolve_t0_asset(row: pd.Series) -> dict[str, Any]:
    target_time = pd.to_datetime(
        row["model_acquisition_time_utc"],
        utc=True,
        errors="coerce",
    )
    if pd.isna(target_time):
        raise ValueError(
            f"無法解析 acquisition time：{row.get('model_acquisition_time_utc')}"
        )

    sensor = str(row["model_sensor"])
    collection_id = collection_for_sensor(sensor)
    point = ee.Geometry.Point(
        [float(row["model_lon"]), float(row["model_lat"])]
    )

    direct_error: Exception | None = None
    for candidate in direct_asset_candidates(row):
        try:
            image = ee.Image(candidate)
            system_index = image.get("system:index").getInfo()
            if system_index:
                metadata = get_image_metadata(image)
                difference_minutes = abs(
                    (
                        metadata["acquisition_time_utc"] - target_time
                    ).total_seconds()
                ) / 60.0
                return {
                    "asset_id": metadata.get("asset_id") or candidate,
                    "image": image,
                    "metadata": metadata,
                    "match_method": "earth_engine_asset_exact",
                    "time_difference_minutes": difference_minutes,
                }
        except Exception as exc:
            direct_error = exc

    base_collection = ee.ImageCollection(collection_id).filterBounds(point)

    # Prefer exact product/legacy scene properties when available.
    for property_name, property_value in exact_property_candidates(row):
        try:
            matched = base_collection.filter(
                ee.Filter.eq(property_name, property_value)
            )
            image, count = nearest_image_from_collection(
                matched,
                target_time,
            )
            if image is not None:
                metadata = get_image_metadata(image)
                difference_minutes = abs(
                    (
                        metadata["acquisition_time_utc"] - target_time
                    ).total_seconds()
                ) / 60.0
                return {
                    "asset_id": metadata.get("asset_id"),
                    "image": image,
                    "metadata": metadata,
                    "match_method": f"property_exact:{property_name}",
                    "time_difference_minutes": difference_minutes,
                    "candidate_count": count,
                }
        except Exception as exc:
            direct_error = exc

    selected_image: ee.Image | None = None
    selected_count = 0

    # Rows without IDs are resolved from sensor + acquisition time + location.
    for window_hours in [12, 48]:
        start = target_time - pd.Timedelta(hours=window_hours)
        end = target_time + pd.Timedelta(hours=window_hours)
        collection = base_collection.filterDate(
            start.isoformat(),
            end.isoformat(),
        )
        selected_image, selected_count = nearest_image_from_collection(
            collection,
            target_time,
        )
        if selected_image is not None:
            break

    if selected_image is None:
        raise RuntimeError(
            "找不到符合 sensor、時間與座標的 Landsat t0 scene。"
            f" key={row.get('model_scene_key')}, sensor={sensor}, "
            f"time={target_time}, lat={row.get('model_lat')}, "
            f"lon={row.get('model_lon')}. Direct/property error={direct_error}"
        )

    metadata = get_image_metadata(selected_image)
    difference_minutes = abs(
        (metadata["acquisition_time_utc"] - target_time).total_seconds()
    ) / 60.0

    if difference_minutes > 360:
        raise RuntimeError(
            "找到的 Landsat scene 與指定 t0 相差 "
            f"{difference_minutes:.1f} 分鐘，超過 6 小時。"
        )

    return {
        "asset_id": metadata.get("asset_id"),
        "image": selected_image,
        "metadata": metadata,
        "match_method": "sensor_time_coordinate_match",
        "time_difference_minutes": difference_minutes,
        "candidate_count": selected_count,
    }


def collection_ids_for_temporal_search(
    preferred_sensor: str,
    mode: str,
) -> list[list[str]]:
    preferred = collection_for_sensor(preferred_sensor)
    both = [L8_COLLECTION, L9_COLLECTION]

    if mode == "same":
        return [[preferred]]
    if mode == "any":
        return [both]
    return [[preferred], both]


def merge_collections(collection_ids: list[str]) -> ee.ImageCollection:
    if not collection_ids:
        raise ValueError("collection_ids 不可為空。")

    collection = ee.ImageCollection(collection_ids[0])
    for collection_id in collection_ids[1:]:
        collection = collection.merge(ee.ImageCollection(collection_id))
    return collection


def choose_temporal_image(
    latitude: float,
    longitude: float,
    t0: pd.Timestamp,
    preferred_sensor: str,
    offset_days: int,
    half_window_days: int,
    max_cloud: float,
    sensor_mode: str,
) -> tuple[ee.Image | None, dict[str, Any]]:
    target = t0 + pd.Timedelta(days=offset_days)
    start = target - pd.Timedelta(days=half_window_days)
    end = target + pd.Timedelta(days=half_window_days + 1)
    point = ee.Geometry.Point([longitude, latitude])

    base_info = {
        "target_time_utc": target,
        "window_start_utc": start,
        "window_end_utc": end,
        "candidate_count": 0,
        "search_sensor_mode": sensor_mode,
    }

    chosen_collection_ids: list[str] | None = None
    chosen_collection: ee.ImageCollection | None = None
    chosen_count = 0

    for collection_ids in collection_ids_for_temporal_search(
        preferred_sensor,
        sensor_mode,
    ):
        collection = (
            merge_collections(collection_ids)
            .filterBounds(point)
            .filterDate(start.isoformat(), end.isoformat())
            .filter(ee.Filter.lte("CLOUD_COVER", float(max_cloud)))
        )
        count = int(collection.size().getInfo())
        if count > 0:
            chosen_collection_ids = collection_ids
            chosen_collection = collection
            chosen_count = count
            break

    if chosen_collection is None or chosen_collection_ids is None:
        return None, base_info

    raw = (
        chosen_collection
        .toList(chosen_collection.size())
        .map(
            lambda obj: ee.Dictionary(
                {
                    "asset_id": ee.Image(obj).get("system:id"),
                    "scene_id": ee.Image(obj).get("system:index"),
                    "time_start": ee.Image(obj).get("system:time_start"),
                    "cloud_pct": ee.Image(obj).get("CLOUD_COVER"),
                    "cloud_land_pct": ee.Image(obj).get("CLOUD_COVER_LAND"),
                    "product_id": ee.Image(obj).get("LANDSAT_PRODUCT_ID"),
                    "legacy_scene_id": ee.Image(obj).get("LANDSAT_SCENE_ID"),
                    "spacecraft_id": ee.Image(obj).get("SPACECRAFT_ID"),
                    "wrs_path": ee.Image(obj).get("WRS_PATH"),
                    "wrs_row": ee.Image(obj).get("WRS_ROW"),
                }
            )
        )
        .getInfo()
    )

    candidates: list[dict[str, Any]] = []
    for item in raw:
        if item.get("time_start") is None:
            continue
        timestamp = millis_to_timestamp(item["time_start"])
        cloud = item.get("cloud_pct")
        cloud_sort = float(cloud) if cloud is not None else math.inf
        diff_seconds = abs((timestamp - target).total_seconds())

        scene_id = item.get("scene_id")
        spacecraft = str(item.get("spacecraft_id") or "")
        sensor = (
            "Landsat-9"
            if "9" in spacecraft
            else "Landsat-8"
            if "8" in spacecraft
            else preferred_sensor
        )
        asset_id = item.get("asset_id")
        if not asset_id and scene_id:
            asset_id = image_asset_id(str(scene_id), sensor=sensor)

        candidates.append(
            {
                **item,
                "asset_id": asset_id,
                "sensor": sensor,
                "acquisition_time_utc": timestamp,
                "absolute_target_difference_seconds": diff_seconds,
                "_sort_key": (
                    diff_seconds,
                    cloud_sort,
                    str(asset_id),
                ),
            }
        )

    if not candidates:
        return None, base_info

    selected = min(candidates, key=lambda item: item["_sort_key"])
    selected.pop("_sort_key", None)

    if not selected.get("asset_id"):
        raise RuntimeError("Temporal Landsat candidate 沒有 asset ID。")

    search_info = {
        **base_info,
        **selected,
        "candidate_count": chosen_count,
        "search_collections": "|".join(chosen_collection_ids),
    }
    return ee.Image(str(selected["asset_id"])), search_info


def qa_clear_fraction(
    image: ee.Image,
    latitude: float,
    longitude: float,
) -> float:
    region = (
        ee.Geometry.Point([longitude, latitude])
        .buffer(HALF_PATCH_METERS)
        .bounds()
    )

    qa = image.select("QA_PIXEL")
    clear = (
        qa.bitwiseAnd(QA_BAD_BITS_MASK)
        .eq(0)
        .And(image.select("SR_B1").mask())
        .rename("clear")
        .unmask(0)
    )

    value = (
        clear.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=30,
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

    expected_channels = (
        array.ndim == 3
        and (
            array.shape[0] == len(L89_BANDS)
            or array.shape[-1] == len(L89_BANDS)
        )
    )

    result["tiff_valid"] = bool(
        expected_channels
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

    # Preserve Collection 2 raw SR DN values. Do not apply the
    # 0.0000275 scale and -0.2 offset before MethaneFuse ingestion.
    export_image = (
        image.select(L89_BANDS)
        .unmask(0)
        .toUint16()
    )

    crs = image.select("SR_B2").projection().crs().getInfo()

    params = {
        "name": output_path.stem,
        "bands": L89_BANDS,
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
                        raise RuntimeError("Earth Engine ZIP 中沒有 TIFF。")
                    with archive.open(members[0]) as source:
                        with output_path.open("wb") as target:
                            shutil.copyfileobj(source, target)
            else:
                output_path.write_bytes(payload)

            validation = validate_tiff(output_path)
            if not validation["tiff_valid"]:
                raise RuntimeError(f"下載後 TIFF 驗證失敗：{validation}")

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


def frame_metadata_record(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_target_time_utc": metadata.get("target_time_utc"),
        "frame_window_start_utc": metadata.get("window_start_utc"),
        "frame_window_end_utc": metadata.get("window_end_utc"),
        "frame_candidate_count": metadata.get("candidate_count"),
        "frame_scene_id": metadata.get("scene_id"),
        "frame_asset_id": metadata.get("asset_id"),
        "frame_acquisition_time_utc": metadata.get(
            "acquisition_time_utc"
        ),
        "frame_cloud_cover": metadata.get("cloud_pct"),
        "frame_cloud_cover_land": metadata.get("cloud_land_pct"),
        "frame_product_id": metadata.get("product_id"),
        "frame_legacy_scene_id": metadata.get("legacy_scene_id"),
        "frame_spacecraft_id": metadata.get("spacecraft_id"),
        "frame_sensor": metadata.get("sensor"),
        "frame_wrs_path": metadata.get("wrs_path"),
        "frame_wrs_row": metadata.get("wrs_row"),
        "frame_processing_level": metadata.get("processing_level"),
        "frame_absolute_target_difference_seconds": metadata.get(
            "absolute_target_difference_seconds"
        ),
        "frame_search_collections": metadata.get("search_collections"),
        "frame_search_sensor_mode": metadata.get("search_sensor_mode"),
        "frame_match_method": metadata.get("match_method"),
    }


def build_eval_table(
    gt: pd.DataFrame,
    manifest: pd.DataFrame,
    min_clear: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ready_pivot = (
        manifest.groupby("external_eval_id")
        .agg(
            ready_frames=("tiff_valid", "sum"),
            minimum_qa_clear_fraction=("qa_clear_fraction", "min"),
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

    eval_table["id"] = eval_table["external_eval_id"].astype(str)
    eval_table["sample_id"] = eval_table["id"]
    eval_table["label"] = eval_table["model_label"].astype(int)
    eval_table["site"] = eval_table["model_site"]
    eval_table["label_provenance"] = "physical_release"
    eval_table["l89_0_path"] = eval_table["t0"]
    eval_table["l89_90_path"] = eval_table["t90"]
    eval_table["l89_360_path"] = eval_table["t360"]

    columns = [
        "id",
        "sample_id",
        "label",
        "label_provenance",
        "site",
        "l89_0_path",
        "l89_90_path",
        "l89_360_path",
        "model_scene_key",
        "model_sensor",
        "model_acquisition_time_utc",
        "model_lat",
        "model_lon",
        "minimum_qa_clear_fraction",
        "release_rate_kg_h",
        "ch4_kgh_mean",
        "ch4_kgh_sigma",
        "ground_truth_type",
        "label_status",
        "label_confidence",
        "label_source",
    ]
    columns = [column for column in columns if column in eval_table.columns]
    raw_eval = eval_table[columns].copy()

    strict_eval = raw_eval[
        raw_eval["minimum_qa_clear_fraction"]
        .fillna(-math.inf)
        .ge(float(min_clear))
    ].copy()

    return raw_eval, strict_eval


def main() -> None:
    args = parse_args()
    initialize_ee(args.project)

    gt, gt_path = load_ground_truth(args.limit)

    print(f"Ground truth: {gt_path}")
    print(f"Unique Landsat observations selected: {len(gt)}")
    print("\nLabel counts:")
    print(gt["model_label"].value_counts().sort_index())
    print("\nSensor counts:")
    print(gt["model_sensor"].value_counts())
    print("\nSite counts:")
    print(gt["model_site"].value_counts())

    manifest_rows: list[dict[str, Any]] = []

    for position, (_, row) in enumerate(gt.iterrows(), start=1):
        t0_time = pd.to_datetime(
            row["model_acquisition_time_utc"],
            utc=True,
        )
        latitude = float(row["model_lat"])
        longitude = float(row["model_lon"])
        sample_id = str(row["external_eval_id"])
        preferred_sensor = str(row["model_sensor"])

        print(
            f"\n[{position}/{len(gt)}] {sample_id}\n"
            f"  key={row['model_scene_key']}\n"
            f"  site={row['model_site']}\n"
            f"  sensor={preferred_sensor}\n"
            f"  label={row['model_label']}"
        )

        base = row.to_dict()

        for frame_name, spec in FRAME_SPECS.items():
            output_path = (
                PATCH_ROOT
                / sample_id
                / f"{sample_id}__{frame_name}.tif"
            )

            if frame_name == "t0":
                resolved_t0 = resolve_t0_asset(row)
                image = resolved_t0["image"]
                metadata = dict(resolved_t0["metadata"])
                metadata.update(
                    {
                        "candidate_count": resolved_t0.get(
                            "candidate_count", 1
                        ),
                        "target_time_utc": t0_time,
                        "window_start_utc": t0_time,
                        "window_end_utc": t0_time,
                        "match_method": resolved_t0.get("match_method"),
                        "absolute_target_difference_seconds": abs(
                            (
                                metadata["acquisition_time_utc"] - t0_time
                            ).total_seconds()
                        ),
                    }
                )
            else:
                image, metadata = choose_temporal_image(
                    latitude=latitude,
                    longitude=longitude,
                    t0=t0_time,
                    preferred_sensor=preferred_sensor,
                    offset_days=int(spec["offset_days"]),
                    half_window_days=int(spec["half_window_days"]),
                    max_cloud=float(args.max_cloud),
                    sensor_mode=args.temporal_sensor_mode,
                )

            frame_record: dict[str, Any] = {
                **base,
                "frame": frame_name,
                "frame_output_path": str(output_path.resolve()),
                **frame_metadata_record(metadata),
            }

            if image is None:
                frame_record.update(
                    {
                        "frame_status": "no_temporal_candidate",
                        "download_status": "not_attempted",
                        "tiff_valid": False,
                        "qa_clear_fraction": float("nan"),
                    }
                )
                manifest_rows.append(frame_record)
                print(f"  {frame_name}: no temporal candidate")
                continue

            try:
                clear_fraction = qa_clear_fraction(
                    image,
                    latitude,
                    longitude,
                )
            except Exception as exc:
                clear_fraction = float("nan")
                frame_record["qa_error"] = (
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
            frame_record["qa_clear_fraction"] = clear_fraction
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
                f"sensor={metadata.get('sensor')} | "
                f"cloud={metadata.get('cloud_pct')} | "
                f"QA clear={clear_text}"
            )

    manifest = pd.DataFrame(manifest_rows)
    DOWNLOAD_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(DOWNLOAD_MANIFEST, index=False)

    raw_eval, strict_eval = build_eval_table(
        gt=gt,
        manifest=manifest,
        min_clear=float(args.min_clear),
    )

    EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    raw_eval.to_csv(EVAL_CSV, index=False)
    strict_eval.to_csv(STRICT_EVAL_CSV, index=False)

    summary_lines = [
        "MethaneFuse Landsat 8/9 external evaluation preparation",
        "=" * 76,
        f"Ground-truth input: {gt_path}",
        f"Unique Landsat observations requested: {len(gt)}",
        f"Observations with all three valid TIFFs: {len(raw_eval)}",
        (
            f"Strict observations with every frame QA clear >= "
            f"{float(args.min_clear):.2f}: {len(strict_eval)}"
        ),
        "",
        "Requested label counts:",
        gt["model_label"].value_counts(dropna=False).sort_index().to_string(),
        "",
        "Raw ready label counts:",
        (
            raw_eval["label"].value_counts(dropna=False).sort_index().to_string()
            if len(raw_eval)
            else "NONE"
        ),
        "",
        "Strict ready label counts:",
        (
            strict_eval["label"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
            if len(strict_eval)
            else "NONE"
        ),
        "",
        "Frame download status:",
        manifest["download_status"].value_counts(dropna=False).to_string(),
        "",
        "Frame validity by temporal slot:",
        manifest.groupby("frame")["tiff_valid"]
        .agg(["count", "sum"])
        .to_string(),
        "",
        "Frame QA clear fraction:",
        manifest.groupby("frame")["qa_clear_fraction"]
        .agg(["count", "mean", "min", "median", "max"])
        .to_string(),
        "",
        "Important:",
        "- t0 is resolved from the manifest using exact IDs when available,",
        "  otherwise by Landsat sensor + acquisition time + coordinates.",
        "- t90 and t360 are real pre-event Landsat scenes selected near 90",
        "  and 360 days before t0. same_then_any prefers the t0 spacecraft",
        "  and falls back to either Landsat-8/9 when needed.",
        "- TIFF band order is SR_B1...SR_B7 and values remain raw uint16 DN.",
        "- QA clear fraction excludes QA_PIXEL bits 0-5 locally over 480 m.",
        "- The raw CSV is the main coverage evaluation; the strict CSV is a",
        "  cloud/snow sensitivity analysis, not a replacement ground truth.",
    ]

    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "\n".join(summary_lines))
    print("\nCreated:")
    print(DOWNLOAD_MANIFEST)
    print(EVAL_CSV)
    print(STRICT_EVAL_CSV)
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
