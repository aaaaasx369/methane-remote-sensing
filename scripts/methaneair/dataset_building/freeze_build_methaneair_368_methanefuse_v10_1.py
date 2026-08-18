#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import ee
import numpy as np
import pandas as pd
import rasterio

HOME = Path.home()
PROJECT_ROOT = HOME / "methane_release_project"
METHANEFUSE_ROOT = HOME / "MethaneFuse"

SOURCE_V9 = (
    PROJECT_ROOT
    / "candidate_negative_validation"
    / "methaneair_production_668_v9"
    / "03_final_unique_strict_s2_controls.csv"
)

DATASET_NAME = "MethaneAIR_Validated_S2_Controls_368_v1"
LOCAL_ROOT = PROJECT_ROOT / DATASET_NAME
LOCAL_CANONICAL = LOCAL_ROOT / "canonical"
LOCAL_PATCH_ROOT = LOCAL_ROOT / "patches"
LOCAL_MANIFESTS = LOCAL_ROOT / "manifests"
LOCAL_CHECKPOINT = LOCAL_ROOT / "checkpoint"

LAB_SHARE = Path("/Volumes/engg-leung")
LAB_ROOT = LAB_SHARE / "dora lin" / DATASET_NAME

CANONICAL_CSV = LOCAL_CANONICAL / "00_canonical_368_controls.csv"
CANONICAL_SHA = LOCAL_CANONICAL / "00_canonical_368_controls.sha256"
FRAME_MANIFEST = LOCAL_MANIFESTS / "01_frame_manifest.csv"
SAMPLE_AUDIT = LOCAL_MANIFESTS / "02_sample_audit.csv"
MF_TECHNICAL_LOCAL = LOCAL_MANIFESTS / "03_methanefuse_all3_technical_local.csv"
MF_STRICT_LOCAL = LOCAL_MANIFESTS / "04_methanefuse_strictqa_local.csv"
MF_STRICT_LAB = LOCAL_MANIFESTS / "05_methanefuse_strictqa_lab.csv"
BUILD_SUMMARY = LOCAL_MANIFESTS / "06_build_summary.txt"
BUILD_XLSX = LOCAL_MANIFESTS / "07_build_audit.xlsx"
CHECKPOINT_JSONL = LOCAL_CHECKPOINT / "build_results.jsonl"

METHANEFUSE_EVAL_CSV = (
    METHANEFUSE_ROOT
    / "data"
    / "custom"
    / "methaneair_validated_368_strictqa_eval.csv"
)

EXPECTED_CONTROLS = 368
EXPECTED_GRADE_COUNTS = {
    "B1_STRONG_HIGH_RES_NO_L4_DETECTION": 337,
    "B2_HIGH_RES_NO_L4_DETECTION_BACKGROUND_WEAK": 31,
}

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_BANDS = [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B9", "B11", "B12",
]

PATCH_PIXELS = 48
PATCH_METERS = 480.0
HALF_PATCH_METERS = PATCH_METERS / 2.0

FRAME_SPECS = {
    "t0": {"offset_days": 0, "half_window_days": 0},
    "t90": {"offset_days": -90, "half_window_days": 30},
    "t360": {"offset_days": -360, "half_window_days": 45},
}

SCL_SCALE_M = 20
MIN_CLEAR_OVER_REQUESTED = 0.80
DEFAULT_MAX_SCENE_CLOUD = 100.0
OVERPASS_GROUP_MINUTES = 20
DOWNLOAD_RETRIES = 4
EE_QUERY_RETRIES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze 368 MethaneAIR-validated S2 controls, download exact "
            "12-band 48x48 t0/t-90/t-360 patches, and build MethaneFuse "
            "wide-table manifests."
        )
    )
    parser.add_argument("--project", default="methane-release-gee")
    parser.add_argument("--source", default=str(SOURCE_V9))
    parser.add_argument(
        "--limit", type=int, default=0,
        help="First N controls only; 0=all. Later full runs resume."
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--max-scene-cloud", type=float, default=DEFAULT_MAX_SCENE_CLOUD,
        help="Scene-level prefilter only; corrected local SCL is the QA gate."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refreeze", action="store_true")
    parser.add_argument("--allow-noncanonical-counts", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-lab-mirror", action="store_true")
    parser.add_argument("--mirror-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lab_mounted() -> bool:
    try:
        result = subprocess.run(
            ["mount"], capture_output=True, text=True, timeout=20
        )
        return str(LAB_SHARE) in result.stdout
    except Exception:
        return False


def relative_to_local_root(path: Path) -> Path:
    return path.resolve().relative_to(LOCAL_ROOT.resolve())


def lab_path_for(local_path: Path) -> Path:
    return LAB_ROOT / relative_to_local_root(local_path)


def safe_mirror_one(local_path: Path) -> dict[str, Any]:
    result = {
        "lab_mirror_status": "disabled_or_unavailable",
        "lab_path": "",
    }

    if not local_path.exists():
        result["lab_mirror_status"] = "local_missing"
        return result

    if not lab_mounted():
        return result

    lab_path = lab_path_for(local_path)
    result["lab_path"] = str(lab_path)

    try:
        lab_path.parent.mkdir(parents=True, exist_ok=True)

        if lab_path.exists():
            try:
                if lab_path.stat().st_size == local_path.stat().st_size:
                    result["lab_mirror_status"] = "reused_existing"
                    return result
            except Exception:
                pass

        tmp = Path(str(lab_path) + ".part")
        subprocess.run(
            ["cp", "-p", str(local_path), str(tmp)],
            check=True,
            timeout=90,
        )

        if tmp.stat().st_size != local_path.stat().st_size:
            raise RuntimeError("SMB copied size mismatch")

        subprocess.run(
            ["mv", "-f", str(tmp), str(lab_path)],
            check=True,
            timeout=30,
        )

        if lab_path.stat().st_size != local_path.stat().st_size:
            raise RuntimeError("SMB final size mismatch")

        result["lab_mirror_status"] = "mirrored"
        return result

    except Exception as exc:
        result["lab_mirror_status"] = (
            f"mirror_failed:{type(exc).__name__}:{exc}"
        )
        return result


def mirror_tree_only() -> None:
    if not LOCAL_ROOT.exists():
        raise FileNotFoundError(f"Local dataset root not found:\n{LOCAL_ROOT}")

    if not lab_mounted():
        raise RuntimeError(
            "Lab SMB is not mounted. Reconnect /Volumes/engg-leung first."
        )

    LAB_ROOT.mkdir(parents=True, exist_ok=True)

    print("Mirroring local dataset to lab SMB:")
    print(f"  local: {LOCAL_ROOT}")
    print(f"  lab  : {LAB_ROOT}")

    subprocess.run(
        ["rsync", "-a", "--partial", f"{LOCAL_ROOT}/", f"{LAB_ROOT}/"],
        check=True,
    )
    print("✅ mirror-only complete")


def json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        x = float(obj)
        return None if math.isnan(x) else x
    if isinstance(obj, np.bool_):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def append_checkpoint(record: dict[str, Any]) -> None:
    LOCAL_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_JSONL.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=json_default,
            )
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def load_checkpoint_latest() -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not CHECKPOINT_JSONL.exists():
        return latest

    with CHECKPOINT_JSONL.open(
        "r", encoding="utf-8", errors="ignore"
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            control_id = obj.get("control_id")
            if control_id:
                latest[str(control_id)] = obj
    return latest


def build_deterministic_canonical(source_path: Path) -> pd.DataFrame:
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    df = pd.read_csv(source_path, low_memory=False)

    required = [
        "Source Positive Record ID",
        "Site",
        "Latitude",
        "Longitude",
        "Parent Positive Date",
        "S2 Product ID",
        "S2 Datetime UTC",
        "Supporting MethaneAIR Flight Count",
        "Supporting MethaneAIR Flight IDs",
        "B1 Strong Supporting Flight Count",
        "Best Corrected Source Coverage",
        "Best Corrected Background Coverage",
        "Minimum Absolute S2 Delta Hours",
        "S2 Coverage Fraction",
        "S2 Clear Among Covered Fraction",
        "S2 Clear Over Requested Fraction",
        "S2 Masked Fraction",
        "Nearest Same-Flight L4 Distance m",
        "Final Evidence Grade",
        "Final Label Type",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "v9 strict-control CSV is missing required columns:\n"
            + "\n".join(f"  {c}" for c in missing)
        )

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["S2 Datetime UTC"] = pd.to_datetime(
        df["S2 Datetime UTC"], utc=True, errors="coerce"
    )
    df["Parent Positive Date"] = pd.to_datetime(
        df["Parent Positive Date"], errors="coerce"
    )

    if df[
        [
            "Latitude",
            "Longitude",
            "S2 Datetime UTC",
            "Parent Positive Date",
            "S2 Product ID",
        ]
    ].isna().any().any():
        raise RuntimeError("Canonical source contains invalid required values.")

    if df.duplicated(
        subset=[
            "Source Positive Record ID",
            "Latitude",
            "Longitude",
            "S2 Product ID",
        ]
    ).any():
        raise RuntimeError("Duplicate unique-control keys found in v9 output.")

    if not df["Final Label Type"].astype(str).eq(
        "strict_temporal_weak_negative"
    ).all():
        raise RuntimeError("Unexpected Final Label Type in v9 output.")

    df = df.sort_values(
        [
            "Site",
            "Parent Positive Date",
            "Source Positive Record ID",
            "S2 Datetime UTC",
            "S2 Product ID",
            "Latitude",
            "Longitude",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    df.insert(
        0,
        "control_id",
        [f"MAIRNEG_{i:04d}" for i in range(1, len(df) + 1)],
    )
    df.insert(1, "label", 0)
    df.insert(2, "label_provenance", df["Final Evidence Grade"].astype(str))
    df.insert(
        3,
        "ground_truth_type",
        "high_res_no_L4_detection_temporal_control",
    )
    df.insert(4, "dataset_name", DATASET_NAME)
    df.insert(5, "canonical_manifest_version", "v1")

    df["Parent Positive Date"] = (
        pd.to_datetime(df["Parent Positive Date"])
        .dt.strftime("%Y-%m-%d")
    )
    df["S2 Datetime UTC"] = (
        pd.to_datetime(df["S2 Datetime UTC"], utc=True)
        .dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )

    return df


def check_expected_canonical(
    df: pd.DataFrame,
    allow_noncanonical: bool,
) -> None:
    grade_counts = (
        df["Final Evidence Grade"].value_counts(dropna=False).to_dict()
    )

    problems = []

    if len(df) != EXPECTED_CONTROLS:
        problems.append(
            f"row count {len(df)} != expected {EXPECTED_CONTROLS}"
        )

    for grade, expected in EXPECTED_GRADE_COUNTS.items():
        actual = int(grade_counts.get(grade, 0))
        if actual != expected:
            problems.append(
                f"{grade}: {actual} != expected {expected}"
            )

    unexpected = set(grade_counts) - set(EXPECTED_GRADE_COUNTS)
    if unexpected:
        problems.append(
            "unexpected evidence grades: "
            + ", ".join(sorted(str(x) for x in unexpected))
        )

    if problems and not allow_noncanonical:
        raise RuntimeError(
            "Canonical source does not match frozen v9 result:\n"
            + "\n".join(f"  {p}" for p in problems)
            + "\nUse --allow-noncanonical-counts only if intentional."
        )

    if problems:
        print("WARNING: noncanonical counts allowed:")
        for p in problems:
            print(" ", p)


def freeze_or_reuse_canonical(
    source_path: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    LOCAL_CANONICAL.mkdir(parents=True, exist_ok=True)

    candidate = build_deterministic_canonical(source_path)
    check_expected_canonical(
        candidate,
        allow_noncanonical=args.allow_noncanonical_counts,
    )

    csv_bytes = candidate.to_csv(
        index=False,
        lineterminator="\n",
    ).encode("utf-8")
    candidate_sha = hashlib.sha256(csv_bytes).hexdigest()

    if CANONICAL_CSV.exists():
        existing_sha = sha256_file(CANONICAL_CSV)

        if existing_sha == candidate_sha:
            print("Canonical manifest already frozen and unchanged.")
            return pd.read_csv(CANONICAL_CSV, low_memory=False)

        if not args.refreeze:
            raise RuntimeError(
                "Existing canonical manifest differs from current v9 source.\n"
                f"Existing SHA256 : {existing_sha}\n"
                f"Current  SHA256 : {candidate_sha}\n"
                "Refusing to silently change the frozen dataset. "
                "Use --refreeze only if intentional."
            )

        print("WARNING: --refreeze supplied; replacing canonical manifest.")

    CANONICAL_CSV.write_bytes(csv_bytes)
    CANONICAL_SHA.write_text(
        f"{candidate_sha}  {CANONICAL_CSV.name}\n",
        encoding="utf-8",
    )

    if not args.no_lab_mirror:
        safe_mirror_one(CANONICAL_CSV)
        safe_mirror_one(CANONICAL_SHA)

    return candidate


def initialize_ee(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Run `earthengine authenticate` "
            f"if needed and confirm project={project!r}. Original error: {exc}"
        ) from exc


def retry_call(func, *args, retries=EE_QUERY_RETRIES, **kwargs):
    last = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last = exc
            if attempt < retries:
                time.sleep(attempt * 2)
    raise last


def resolve_s2_asset_id(item: dict[str, Any]) -> str:
    """Return a loadable full Sentinel-2 asset ID from candidate metadata."""
    system_index = str(item.get("system_index") or "").strip()
    raw_asset_id = str(item.get("asset_id") or "").strip()

    if system_index and system_index.lower() not in {"nan", "none", "<na>"}:
        return f"{S2_COLLECTION}/{system_index}"

    if raw_asset_id:
        if raw_asset_id.startswith("COPERNICUS/") or raw_asset_id.startswith("projects/"):
            return raw_asset_id
        return f"{S2_COLLECTION}/{raw_asset_id}"

    raise RuntimeError("Temporal S2 candidate has neither system:index nor asset_id.")


def image_metadata(image: ee.Image) -> dict[str, Any]:
    raw = ee.Dictionary(
        {
            "asset_id": image.id(),
            "system_index": image.get("system:index"),
            "time_start": image.get("system:time_start"),
            "cloud_pct": image.get("CLOUDY_PIXEL_PERCENTAGE"),
            "mgrs_tile": image.get("MGRS_TILE"),
            "product_id": image.get("PRODUCT_ID"),
            "spacecraft_name": image.get("SPACECRAFT_NAME"),
        }
    ).getInfo()

    if raw.get("time_start") is not None:
        raw["acquisition_time_utc"] = pd.to_datetime(
            raw["time_start"], unit="ms", utc=True
        )
    else:
        raw["acquisition_time_utc"] = pd.NaT

    return raw


def corrected_s2_qa(
    image: ee.Image,
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    point = ee.Geometry.Point([longitude, latitude])
    region = point.buffer(HALF_PATCH_METERS).bounds()

    scl = image.select("SCL")
    proj = scl.projection()

    filled = scl.unmask(
        value=255,
        sameFootprint=False,
    ).rename("SCL_filled")

    requested = filled.multiply(0).add(1).rename("requested")
    covered = filled.neq(255).rename("covered")
    clear = filled.gte(4).And(filled.lte(7)).rename("clear")
    cloud = filled.gte(8).And(filled.lte(10)).rename("cloud")
    shadow = filled.eq(3).rename("shadow")
    snow = filled.eq(11).rename("snow")
    invalid = filled.gte(0).And(filled.lte(2)).rename("invalid")
    masked = filled.eq(255).rename("masked")

    stack = (
        requested
        .addBands(covered)
        .addBands(clear)
        .addBands(cloud)
        .addBands(shadow)
        .addBands(snow)
        .addBands(invalid)
        .addBands(masked)
    )

    stats = stack.reduceRegion(
        reducer=ee.Reducer.sum().unweighted(),
        geometry=region,
        crs=proj,
        scale=SCL_SCALE_M,
        bestEffort=False,
        maxPixels=1_000_000,
    ).getInfo()

    def n(key: str) -> float:
        return float(stats.get(key) or 0)

    requested_n = n("requested")
    covered_n = n("covered")
    clear_n = n("clear")
    cloud_n = n("cloud")
    shadow_n = n("shadow")
    snow_n = n("snow")
    invalid_n = n("invalid")
    masked_n = n("masked")

    def frac(a: float, b: float) -> float:
        return a / b if b > 0 else float("nan")

    coverage = frac(covered_n, requested_n)
    clear_requested = frac(clear_n, requested_n)
    clear_covered = frac(clear_n, covered_n)

    partition_error = (
        clear_n + cloud_n + shadow_n + snow_n + invalid_n + masked_n
        - requested_n
    )

    impossible = (
        (
            np.isfinite(clear_covered)
            and clear_covered > 1.0000001
        )
        or (
            np.isfinite(clear_requested)
            and np.isfinite(coverage)
            and clear_requested > coverage + 1e-9
        )
        or abs(partition_error) > 1e-6
    )

    if impossible:
        raise RuntimeError(
            "Corrected SCL QA produced an impossible fraction/partition."
        )

    return {
        "coverage_fraction": coverage,
        "clear_among_covered_fraction": clear_covered,
        "clear_over_requested_fraction": clear_requested,
        "cloud_over_requested_fraction": frac(cloud_n, requested_n),
        "shadow_over_requested_fraction": frac(shadow_n, requested_n),
        "snow_over_requested_fraction": frac(snow_n, requested_n),
        "invalid_over_requested_fraction": frac(invalid_n, requested_n),
        "masked_fraction": frac(masked_n, requested_n),
        "qa_pass": bool(
            np.isfinite(clear_requested)
            and clear_requested >= MIN_CLEAR_OVER_REQUESTED
        ),
    }


def find_exact_t0_image(
    product_id: str,
    latitude: float,
    longitude: float,
) -> tuple[ee.Image, dict[str, Any]]:
    point = ee.Geometry.Point([longitude, latitude])
    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(point)
        .filter(ee.Filter.eq("PRODUCT_ID", str(product_id)))
    )

    size = int(retry_call(lambda: collection.size().getInfo()))
    if size == 0:
        raise RuntimeError(
            f"Exact t0 PRODUCT_ID not found at control location: {product_id}"
        )

    images = collection.toList(size)
    candidates = []

    for i in range(size):
        image = ee.Image(images.get(i))
        meta = retry_call(image_metadata, image)
        qa = retry_call(
            corrected_s2_qa,
            image,
            latitude,
            longitude,
        )
        candidates.append({"image": image, "meta": meta, "qa": qa})

    best = max(
        candidates,
        key=lambda x: (
            float(x["qa"]["clear_over_requested_fraction"]),
            float(x["qa"]["coverage_fraction"]),
        ),
    )

    return best["image"], {
        **best["meta"],
        **{f"qa_{k}": v for k, v in best["qa"].items()},
        "candidate_count": size,
    }


def get_temporal_candidate_metadata(
    latitude: float,
    longitude: float,
    t0_time: pd.Timestamp,
    offset_days: int,
    half_window_days: int,
    max_scene_cloud: float,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    target = t0_time + pd.Timedelta(days=offset_days)
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
                float(max_scene_cloud),
            )
        )
        .sort("system:time_start")
    )

    size = int(retry_call(lambda: collection.size().getInfo()))
    search_info = {
        "target_time_utc": target,
        "window_start_utc": start,
        "window_end_utc": end,
        "candidate_scene_count": size,
    }

    if size == 0:
        return [], search_info

    raw = retry_call(
        lambda: collection.toList(size)
        .map(
            lambda obj: ee.Dictionary(
                {
                    "asset_id": ee.Image(obj).id(),
                    "system_index": ee.Image(obj).get("system:index"),
                    "time_start": ee.Image(obj).get("system:time_start"),
                    "cloud_pct": ee.Image(obj).get(
                        "CLOUDY_PIXEL_PERCENTAGE"
                    ),
                    "mgrs_tile": ee.Image(obj).get("MGRS_TILE"),
                    "product_id": ee.Image(obj).get("PRODUCT_ID"),
                    "spacecraft_name": ee.Image(obj).get("SPACECRAFT_NAME"),
                }
            )
        )
        .getInfo()
    )

    candidates = []
    for item in raw:
        if item.get("time_start") is None:
            continue

        ts = pd.to_datetime(item["time_start"], unit="ms", utc=True)
        cloud = item.get("cloud_pct")
        cloud = float(cloud) if cloud is not None else math.inf

        resolved_asset_id = resolve_s2_asset_id(item)

        candidates.append(
            {
                **item,
                "raw_asset_id": item.get("asset_id"),
                "asset_id": resolved_asset_id,
                "resolved_asset_id": resolved_asset_id,
                "acquisition_time_utc": ts,
                "cloud_pct_numeric": cloud,
                "absolute_target_difference_seconds": abs(
                    (ts - target).total_seconds()
                ),
            }
        )

    candidates.sort(
        key=lambda row: (
            row["acquisition_time_utc"],
            row["cloud_pct_numeric"],
            str(row.get("asset_id")),
        )
    )

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous: pd.Timestamp | None = None

    for row in candidates:
        ts = row["acquisition_time_utc"]
        if (
            previous is None
            or (ts - previous).total_seconds()
            <= OVERPASS_GROUP_MINUTES * 60
        ):
            current.append(row)
        else:
            groups.append(current)
            current = [row]
        previous = ts

    if current:
        groups.append(current)

    def group_rank(group: list[dict[str, Any]]) -> tuple[Any, ...]:
        best_diff = min(
            float(x["absolute_target_difference_seconds"])
            for x in group
        )
        best_cloud = min(float(x["cloud_pct_numeric"]) for x in group)
        return (best_diff, best_cloud, str(group[0].get("asset_id")))

    groups.sort(key=group_rank)
    search_info["candidate_overpass_count"] = len(groups)
    return groups, search_info


def choose_temporal_image(
    latitude: float,
    longitude: float,
    t0_time: pd.Timestamp,
    offset_days: int,
    half_window_days: int,
    max_scene_cloud: float,
) -> tuple[ee.Image | None, dict[str, Any]]:
    groups, search_info = get_temporal_candidate_metadata(
        latitude=latitude,
        longitude=longitude,
        t0_time=t0_time,
        offset_days=offset_days,
        half_window_days=half_window_days,
        max_scene_cloud=max_scene_cloud,
    )

    if not groups:
        search_info["selection_reason"] = "no_temporal_candidate"
        return None, search_info

    best_fail = None
    qa_errors: list[str] = []

    for overpass_index, group in enumerate(groups, start=1):
        scored = []

        for item in group:
            try:
                resolved_asset_id = resolve_s2_asset_id(item)
                image = ee.Image(resolved_asset_id)
            except Exception as exc:
                error_text = (
                    f"asset_reconstruction_error | "
                    f"system_index={item.get('system_index')} | "
                    f"asset_id={item.get('asset_id')} | "
                    f"{type(exc).__name__}: {exc}"
                )
                qa_errors.append(error_text)
                scored.append({**item, "image": None, "qa_error": error_text, "qa": None})
                continue

            try:
                qa = retry_call(
                    corrected_s2_qa,
                    image,
                    latitude,
                    longitude,
                )
            except Exception as exc:
                error_text = (
                    f"qa_error | asset={resolved_asset_id} | "
                    f"{type(exc).__name__}: {exc}"
                )
                qa_errors.append(error_text)
                scored.append({**item, "image": image, "qa_error": error_text, "qa": None})
                continue

            scored.append(
                {
                    **item,
                    "asset_id": resolved_asset_id,
                    "resolved_asset_id": resolved_asset_id,
                    "image": image,
                    "qa": qa,
                }
            )

        valid_scored = [x for x in scored if x.get("qa") is not None]
        if not valid_scored:
            continue

        best_tile = max(
            valid_scored,
            key=lambda x: (
                float(x["qa"]["clear_over_requested_fraction"]),
                float(x["qa"]["coverage_fraction"]),
                -float(x["cloud_pct_numeric"]),
            ),
        )

        candidate_info = {
            **{
                k: v
                for k, v in best_tile.items()
                if k not in {"image", "qa"}
            },
            **{f"qa_{k}": v for k, v in best_tile["qa"].items()},
            "selected_overpass_index": overpass_index,
            "overlap_tile_count": len(group),
        }

        if best_tile["qa"]["qa_pass"]:
            search_info.update(candidate_info)
            search_info["selection_reason"] = (
                "nearest_overpass_with_corrected_local_QA80_pass"
            )
            return best_tile["image"], search_info

        if best_fail is None:
            best_fail = (best_tile["image"], candidate_info)
        else:
            current_key = (
                float(candidate_info["qa_clear_over_requested_fraction"]),
                -float(candidate_info["absolute_target_difference_seconds"]),
                -float(candidate_info["cloud_pct_numeric"]),
            )
            previous_key = (
                float(best_fail[1]["qa_clear_over_requested_fraction"]),
                -float(best_fail[1]["absolute_target_difference_seconds"]),
                -float(best_fail[1]["cloud_pct_numeric"]),
            )
            if current_key > previous_key:
                best_fail = (best_tile["image"], candidate_info)

    if best_fail is not None:
        image, candidate_info = best_fail
        search_info.update(candidate_info)
        search_info["selection_reason"] = (
            "no_QA80_overpass_found_best_corrected_local_QA_fail"
        )
        return image, search_info

    search_info["selection_reason"] = "all_candidate_QA_queries_failed"
    search_info["qa_error_count"] = len(qa_errors)
    search_info["qa_errors"] = " || ".join(qa_errors[:20])
    return None, search_info


def validate_tiff(path: Path) -> dict[str, Any]:
    result = {
        "tiff_exists": path.exists(),
        "tiff_valid": False,
        "tiff_width": None,
        "tiff_height": None,
        "tiff_band_count": None,
        "tiff_dtype": None,
        "tiff_all_zero": None,
        "tiff_size_bytes": None,
    }

    if not path.exists():
        return result

    try:
        result["tiff_size_bytes"] = path.stat().st_size

        with rasterio.open(path) as src:
            result["tiff_width"] = int(src.width)
            result["tiff_height"] = int(src.height)
            result["tiff_band_count"] = int(src.count)
            result["tiff_dtype"] = "|".join(src.dtypes)
            arr = src.read()

        all_zero = bool(np.all(arr == 0))
        result["tiff_all_zero"] = all_zero

        result["tiff_valid"] = bool(
            result["tiff_width"] == PATCH_PIXELS
            and result["tiff_height"] == PATCH_PIXELS
            and result["tiff_band_count"] == len(S2_BANDS)
            and all(
                dtype == "uint16"
                for dtype in result["tiff_dtype"].split("|")
            )
            and not all_zero
        )

    except Exception as exc:
        result["tiff_validation_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    return result


def download_geotiff(
    image: ee.Image,
    latitude: float,
    longitude: float,
    output_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        validation = validate_tiff(output_path)
        if validation["tiff_valid"]:
            return {"download_status": "reused_existing", **validation}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    region = (
        ee.Geometry.Point([longitude, latitude])
        .buffer(HALF_PATCH_METERS)
        .bounds()
    )

    export_image = image.select(S2_BANDS).unmask(0).toUint16()

    crs = retry_call(
        lambda: image.select("B2").projection().crs().getInfo()
    )

    params = {
        "name": output_path.stem,
        "bands": S2_BANDS,
        "region": region.getInfo()["coordinates"],
        "dimensions": f"{PATCH_PIXELS}x{PATCH_PIXELS}",
        "crs": crs,
        "format": "GEO_TIFF",
        "filePerBand": False,
    }

    part_path = Path(str(output_path) + ".part")
    last_error = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            if part_path.exists():
                part_path.unlink()

            url = retry_call(export_image.getDownloadURL, params)

            with urlopen(url, timeout=180) as response:
                payload = response.read()

            if payload[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    members = [
                        m
                        for m in archive.namelist()
                        if m.lower().endswith((".tif", ".tiff"))
                    ]
                    if not members:
                        raise RuntimeError(
                            "Earth Engine ZIP contains no TIFF."
                        )
                    with archive.open(members[0]) as src:
                        with part_path.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
            else:
                part_path.write_bytes(payload)

            validation = validate_tiff(part_path)
            if not validation["tiff_valid"]:
                raise RuntimeError(
                    f"Downloaded TIFF validation failed: {validation}"
                )

            os.replace(part_path, output_path)
            final_validation = validate_tiff(output_path)

            if not final_validation["tiff_valid"]:
                raise RuntimeError(
                    f"Final TIFF validation failed: {final_validation}"
                )

            return {
                "download_status": "downloaded",
                **final_validation,
            }

        except Exception as exc:
            last_error = exc
            try:
                if part_path.exists():
                    part_path.unlink()
            except Exception:
                pass
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(2 * attempt)

    return {
        "download_status": "failed",
        "download_error": (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "unknown"
        ),
        **validate_tiff(output_path),
    }


def process_control(
    row: pd.Series,
    args: argparse.Namespace,
) -> dict[str, Any]:
    control_id = str(row["control_id"])
    lat = float(row["Latitude"])
    lon = float(row["Longitude"])
    t0_product_id = str(row["S2 Product ID"])
    t0_expected_time = pd.to_datetime(
        row["S2 Datetime UTC"], utc=True
    )

    sample_dir = LOCAL_PATCH_ROOT / control_id

    base_summary = {
        "control_id": control_id,
        "label": 0,
        "label_provenance": row["label_provenance"],
        "ground_truth_type": row["ground_truth_type"],
        "site": row["Site"],
        "source_positive_record_id": row["Source Positive Record ID"],
        "parent_positive_date": row["Parent Positive Date"],
        "lat": lat,
        "lon": lon,
        "t0_product_id": t0_product_id,
        "t0_expected_datetime_utc": str(t0_expected_time),
        "supporting_methaneair_flight_count": row[
            "Supporting MethaneAIR Flight Count"
        ],
        "supporting_methaneair_flight_ids": row[
            "Supporting MethaneAIR Flight IDs"
        ],
        "final_evidence_grade": row["Final Evidence Grade"],
        "final_label_type": row["Final Label Type"],
    }

    frame_rows = []
    selected_paths: dict[str, str] = {}
    selected_lab_paths: dict[str, str] = {}
    all_technical = True
    all_qa = True
    retryable_failure = False

    try:
        t0_image, t0_meta = retry_call(
            find_exact_t0_image,
            t0_product_id,
            lat,
            lon,
        )
    except Exception as exc:
        return {
            "complete": False,
            "retryable": True,
            "control_id": control_id,
            "sample_summary": {
                **base_summary,
                "sample_result": "T0_QUERY_ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            },
            "frames": [],
            "traceback": traceback.format_exc(),
        }

    t0_actual_time = pd.to_datetime(
        t0_meta["acquisition_time_utc"], utc=True
    )

    for frame_name, spec in FRAME_SPECS.items():
        output_path = sample_dir / f"{control_id}__{frame_name}.tif"

        if frame_name == "t0":
            image = t0_image
            meta = dict(t0_meta)
            meta["target_time_utc"] = t0_expected_time
            meta["window_start_utc"] = t0_expected_time
            meta["window_end_utc"] = t0_expected_time
            meta["selection_reason"] = "exact_validated_v9_t0_PRODUCT_ID"
            meta["absolute_target_difference_seconds"] = abs(
                (t0_actual_time - t0_expected_time).total_seconds()
            )
        else:
            try:
                image, meta = retry_call(
                    choose_temporal_image,
                    latitude=lat,
                    longitude=lon,
                    t0_time=t0_actual_time,
                    offset_days=int(spec["offset_days"]),
                    half_window_days=int(spec["half_window_days"]),
                    max_scene_cloud=float(args.max_scene_cloud),
                )
            except Exception as exc:
                image = None
                meta = {
                    "selection_reason": "temporal_query_error",
                    "query_error": f"{type(exc).__name__}: {exc}",
                }
                retryable_failure = True

        frame = {
            **base_summary,
            "frame": frame_name,
            "frame_output_path": str(output_path.resolve()),
            "frame_target_time_utc": meta.get("target_time_utc"),
            "frame_window_start_utc": meta.get("window_start_utc"),
            "frame_window_end_utc": meta.get("window_end_utc"),
            "frame_candidate_scene_count": meta.get("candidate_scene_count"),
            "frame_candidate_overpass_count": meta.get(
                "candidate_overpass_count"
            ),
            "frame_selection_reason": meta.get("selection_reason"),
            "frame_asset_id": meta.get("asset_id"),
            "frame_system_index": meta.get("system_index"),
            "frame_product_id": meta.get("product_id"),
            "frame_acquisition_time_utc": meta.get(
                "acquisition_time_utc"
            ),
            "frame_cloudy_pixel_percentage": meta.get("cloud_pct"),
            "frame_mgrs_tile": meta.get("mgrs_tile"),
            "frame_spacecraft_name": meta.get("spacecraft_name"),
            "frame_absolute_target_difference_seconds": meta.get(
                "absolute_target_difference_seconds"
            ),
            "frame_overlap_tile_count": meta.get("overlap_tile_count"),
            "frame_qa_error_count": meta.get("qa_error_count"),
            "frame_qa_errors": meta.get("qa_errors"),
            "scl_coverage_fraction": meta.get("qa_coverage_fraction"),
            "scl_clear_among_covered_fraction": meta.get(
                "qa_clear_among_covered_fraction"
            ),
            "scl_clear_over_requested_fraction": meta.get(
                "qa_clear_over_requested_fraction"
            ),
            "scl_cloud_over_requested_fraction": meta.get(
                "qa_cloud_over_requested_fraction"
            ),
            "scl_shadow_over_requested_fraction": meta.get(
                "qa_shadow_over_requested_fraction"
            ),
            "scl_snow_over_requested_fraction": meta.get(
                "qa_snow_over_requested_fraction"
            ),
            "scl_masked_fraction": meta.get("qa_masked_fraction"),
            "scl_qa_pass": meta.get("qa_qa_pass"),
        }

        if image is None:
            selection_reason = str(frame.get("frame_selection_reason") or "")
            if selection_reason in {
                "temporal_query_error",
                "all_candidate_QA_queries_failed",
            }:
                retryable_failure = True

            frame.update(
                {
                    "frame_status": "no_image_selected",
                    "download_status": "not_attempted",
                    "tiff_valid": False,
                    "lab_mirror_status": "not_attempted",
                    "lab_path": "",
                }
            )
            frame_rows.append(frame)
            all_technical = False
            all_qa = False
            continue

        if frame.get("scl_qa_pass") is None:
            try:
                qa = retry_call(
                    corrected_s2_qa,
                    image,
                    lat,
                    lon,
                )
                frame.update(
                    {
                        "scl_coverage_fraction": qa["coverage_fraction"],
                        "scl_clear_among_covered_fraction": qa[
                            "clear_among_covered_fraction"
                        ],
                        "scl_clear_over_requested_fraction": qa[
                            "clear_over_requested_fraction"
                        ],
                        "scl_cloud_over_requested_fraction": qa[
                            "cloud_over_requested_fraction"
                        ],
                        "scl_shadow_over_requested_fraction": qa[
                            "shadow_over_requested_fraction"
                        ],
                        "scl_snow_over_requested_fraction": qa[
                            "snow_over_requested_fraction"
                        ],
                        "scl_masked_fraction": qa["masked_fraction"],
                        "scl_qa_pass": qa["qa_pass"],
                    }
                )
            except Exception as exc:
                frame["scl_qa_error"] = f"{type(exc).__name__}: {exc}"
                frame["scl_qa_pass"] = False
                retryable_failure = True

        download = download_geotiff(
            image=image,
            latitude=lat,
            longitude=lon,
            output_path=output_path,
            overwrite=args.overwrite,
        )
        frame.update(download)

        if download.get("tiff_valid"):
            selected_paths[frame_name] = str(output_path.resolve())

            if args.no_lab_mirror:
                mirror = {
                    "lab_mirror_status": "disabled",
                    "lab_path": "",
                }
            else:
                mirror = safe_mirror_one(output_path)

            frame.update(mirror)

            if (
                mirror.get("lab_path")
                and mirror.get("lab_mirror_status")
                in {"mirrored", "reused_existing"}
            ):
                selected_lab_paths[frame_name] = mirror["lab_path"]
        else:
            all_technical = False
            retryable_failure = True
            frame.setdefault("lab_mirror_status", "not_attempted")
            frame.setdefault("lab_path", "")

        if not bool(frame.get("scl_qa_pass")):
            all_qa = False

        frame["frame_status"] = (
            "ready" if bool(download.get("tiff_valid")) else "failed"
        )
        frame_rows.append(frame)

    all_technical = bool(all_technical and len(selected_paths) == 3)
    all_qa = bool(
        all_qa
        and len(frame_rows) == 3
        and all(bool(f.get("scl_qa_pass")) for f in frame_rows)
    )
    model_ready = bool(all_technical and all_qa)

    t0_regression = False
    for f in frame_rows:
        if f["frame"] == "t0":
            t0_regression = not bool(f.get("scl_qa_pass"))
            break

    clear_values = pd.to_numeric(
        pd.Series(
            [f.get("scl_clear_over_requested_fraction") for f in frame_rows]
        ),
        errors="coerce",
    ).dropna()

    minimum_clear = (
        float(clear_values.min()) if len(clear_values) else float("nan")
    )

    summary = {
        **base_summary,
        "ready_frames": int(
            sum(bool(f.get("tiff_valid")) for f in frame_rows)
        ),
        "all_three_technical_pass": all_technical,
        "all_three_qa_pass_corrected": all_qa,
        "strict_model_ready": model_ready,
        "minimum_scl_clear_fraction": minimum_clear,
        "t0_corrected_qa_regression": t0_regression,
        "s2_0_path": selected_paths.get("t0", ""),
        "s2_90_path": selected_paths.get("t90", ""),
        "s2_360_path": selected_paths.get("t360", ""),
        "lab_s2_0_path": selected_lab_paths.get("t0", ""),
        "lab_s2_90_path": selected_lab_paths.get("t90", ""),
        "lab_s2_360_path": selected_lab_paths.get("t360", ""),
        "all_three_lab_mirrored": bool(len(selected_lab_paths) == 3),
        "sample_result": (
            "STRICT_MODEL_READY"
            if model_ready
            else (
                "TECHNICAL_READY_QA_FAIL"
                if all_technical
                else "TECHNICAL_INCOMPLETE"
            )
        ),
    }

    complete = not retryable_failure

    if any(
        f.get("frame_status") == "no_image_selected"
        and f.get("frame_selection_reason") == "no_temporal_candidate"
        for f in frame_rows
    ):
        complete = True

    return {
        "complete": bool(complete),
        "retryable": bool(not complete),
        "control_id": control_id,
        "sample_summary": summary,
        "frames": frame_rows,
    }


def checkpoint_to_tables(
    latest: dict[str, dict[str, Any]],
    selected_control_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    frames = []

    for control_id, obj in latest.items():
        if (
            selected_control_ids is not None
            and control_id not in selected_control_ids
        ):
            continue

        summary = obj.get("sample_summary")
        if summary:
            summaries.append(summary)
        frames.extend(obj.get("frames", []))

    sample_df = pd.DataFrame(summaries)
    frame_df = pd.DataFrame(frames)

    if len(sample_df):
        sample_df = sample_df.sort_values("control_id").reset_index(drop=True)

    if len(frame_df):
        frame_order = pd.Categorical(
            frame_df["frame"],
            categories=["t0", "t90", "t360"],
            ordered=True,
        )
        frame_df = (
            frame_df.assign(_frame_order=frame_order)
            .sort_values(["control_id", "_frame_order"])
            .drop(columns=["_frame_order"])
            .reset_index(drop=True)
        )

    return sample_df, frame_df


def build_methanefuse_eval(
    sample_df: pd.DataFrame,
    use_lab_paths: bool,
    strict_qa_only: bool,
) -> pd.DataFrame:
    if len(sample_df) == 0:
        return pd.DataFrame()

    df = sample_df.copy()

    if strict_qa_only:
        df = df[
            df["strict_model_ready"].fillna(False).astype(bool)
        ].copy()
    else:
        df = df[
            df["all_three_technical_pass"].fillna(False).astype(bool)
        ].copy()

    if len(df) == 0:
        return pd.DataFrame()

    if use_lab_paths:
        df = df[
            df["all_three_lab_mirrored"].fillna(False).astype(bool)
        ].copy()
        path_map = {
            "lab_s2_0_path": "s2_0_path",
            "lab_s2_90_path": "s2_90_path",
            "lab_s2_360_path": "s2_360_path",
        }
    else:
        path_map = {
            "s2_0_path": "s2_0_path",
            "s2_90_path": "s2_90_path",
            "s2_360_path": "s2_360_path",
        }

    out = pd.DataFrame(
        {
            "id": df["control_id"].astype(str),
            "sample_id": df["control_id"].astype(str),
            "label": 0,
            "label_provenance": df["final_evidence_grade"].astype(str),
            "ground_truth_type": df["ground_truth_type"].astype(str),
            "site": df["site"].astype(str),
            "scene_id": df["t0_product_id"].astype(str),
            "acquisition_time_utc": df[
                "t0_expected_datetime_utc"
            ].astype(str),
            "lat": pd.to_numeric(df["lat"], errors="coerce"),
            "lon": pd.to_numeric(df["lon"], errors="coerce"),
            "minimum_scl_clear_fraction": pd.to_numeric(
                df["minimum_scl_clear_fraction"], errors="coerce"
            ),
            "source_positive_record_id": df[
                "source_positive_record_id"
            ].astype(str),
            "parent_positive_date": df["parent_positive_date"].astype(str),
            "supporting_methaneair_flight_count": df[
                "supporting_methaneair_flight_count"
            ],
            "supporting_methaneair_flight_ids": df[
                "supporting_methaneair_flight_ids"
            ].astype(str),
            "negative_control_definition": (
                "strict_temporal_weak_negative__"
                "MethaneAIR_L3_source_valid__"
                "same_flight_L4_no_high_emitter_within_5km__"
                "S2_within_72h"
            ),
        }
    )

    for source_col, destination_col in path_map.items():
        out[destination_col] = df[source_col].astype(str).values

    ordered = [
        "id",
        "sample_id",
        "label",
        "label_provenance",
        "ground_truth_type",
        "s2_0_path",
        "s2_90_path",
        "s2_360_path",
        "scene_id",
        "site",
        "acquisition_time_utc",
        "lat",
        "lon",
        "minimum_scl_clear_fraction",
        "source_positive_record_id",
        "parent_positive_date",
        "supporting_methaneair_flight_count",
        "supporting_methaneair_flight_ids",
        "negative_control_definition",
    ]

    return out[ordered].reset_index(drop=True)


def save_outputs(
    latest: dict[str, dict[str, Any]],
    selected_control_ids: set[str] | None,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    LOCAL_MANIFESTS.mkdir(parents=True, exist_ok=True)

    sample_df, frame_df = checkpoint_to_tables(
        latest,
        selected_control_ids=selected_control_ids,
    )

    frame_df.to_csv(FRAME_MANIFEST, index=False, encoding="utf-8-sig")
    sample_df.to_csv(SAMPLE_AUDIT, index=False, encoding="utf-8-sig")

    technical_local = build_methanefuse_eval(
        sample_df,
        use_lab_paths=False,
        strict_qa_only=False,
    )
    strict_local = build_methanefuse_eval(
        sample_df,
        use_lab_paths=False,
        strict_qa_only=True,
    )
    strict_lab = build_methanefuse_eval(
        sample_df,
        use_lab_paths=True,
        strict_qa_only=True,
    )

    technical_local.to_csv(
        MF_TECHNICAL_LOCAL,
        index=False,
        encoding="utf-8-sig",
    )
    strict_local.to_csv(
        MF_STRICT_LOCAL,
        index=False,
        encoding="utf-8-sig",
    )
    strict_lab.to_csv(
        MF_STRICT_LAB,
        index=False,
        encoding="utf-8-sig",
    )

    if METHANEFUSE_ROOT.exists():
        try:
            METHANEFUSE_EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(MF_STRICT_LOCAL, METHANEFUSE_EVAL_CSV)
        except Exception as exc:
            print(
                "WARNING: could not copy strict eval CSV into MethaneFuse repo:",
                repr(exc),
            )

    try:
        with pd.ExcelWriter(BUILD_XLSX, engine="openpyxl") as writer:
            sample_df.to_excel(writer, sheet_name="Sample_Audit", index=False)
            frame_df.to_excel(writer, sheet_name="Frame_Manifest", index=False)
            strict_local.to_excel(writer, sheet_name="Strict_Local", index=False)
            strict_lab.to_excel(writer, sheet_name="Strict_Lab", index=False)
    except Exception as exc:
        print("WARNING: XLSX audit write failed:", repr(exc))

    requested = len(sample_df)
    technical_n = (
        int(
            sample_df["all_three_technical_pass"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if len(sample_df)
        else 0
    )
    qa_n = (
        int(
            sample_df["all_three_qa_pass_corrected"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if len(sample_df)
        else 0
    )
    model_ready_n = (
        int(
            sample_df["strict_model_ready"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if len(sample_df)
        else 0
    )
    t0_regressions = (
        int(
            sample_df["t0_corrected_qa_regression"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if len(sample_df)
        else 0
    )

    lines = [
        f"{DATASET_NAME} BUILD SUMMARY",
        "=" * 84,
        f"Canonical source: {CANONICAL_CSV}",
        (
            "Canonical SHA256: "
            + (
                sha256_file(CANONICAL_CSV)
                if CANONICAL_CSV.exists()
                else "MISSING"
            )
        ),
        f"Rows represented in this run/output: {requested}",
        f"All-three technical pass: {technical_n}",
        f"All-three corrected-QA pass: {qa_n}",
        f"Strict model-ready: {model_ready_n}",
        f"t0 corrected-QA regressions: {t0_regressions}",
        f"Strict MethaneFuse local rows: {len(strict_local)}",
        f"Strict MethaneFuse lab-mirrored rows: {len(strict_lab)}",
        "",
        "Sample result:",
    ]

    if len(sample_df):
        lines.extend(
            "  " + line
            for line in sample_df["sample_result"]
            .value_counts(dropna=False)
            .to_string()
            .splitlines()
        )

    lines.extend(["", "Evidence grade among strict model-ready:"])

    if len(sample_df):
        ready = sample_df[
            sample_df["strict_model_ready"].fillna(False).astype(bool)
        ]
        if len(ready):
            lines.extend(
                "  " + line
                for line in ready["final_evidence_grade"]
                .value_counts(dropna=False)
                .to_string()
                .splitlines()
            )
        else:
            lines.append("  NONE")

    BUILD_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not args.no_lab_mirror:
        for path in [
            FRAME_MANIFEST,
            SAMPLE_AUDIT,
            MF_TECHNICAL_LOCAL,
            MF_STRICT_LOCAL,
            MF_STRICT_LAB,
            BUILD_SUMMARY,
            BUILD_XLSX,
            CANONICAL_CSV,
            CANONICAL_SHA,
            CHECKPOINT_JSONL,
        ]:
            if path.exists():
                safe_mirror_one(path)

    return sample_df, frame_df, strict_local


def main() -> None:
    args = parse_args()

    if args.mirror_only:
        mirror_tree_only()
        return

    source_path = Path(args.source).expanduser()

    for p in [
        LOCAL_CANONICAL,
        LOCAL_PATCH_ROOT,
        LOCAL_MANIFESTS,
        LOCAL_CHECKPOINT,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("METHANEAIR-VALIDATED S2 CONTROLS -> METHANEFUSE BUILD V10")
    print("=" * 96)

    canonical = freeze_or_reuse_canonical(source_path, args)

    print("\nCANONICAL FREEZE")
    print(f"Rows: {len(canonical)}")
    print(canonical["Final Evidence Grade"].value_counts(dropna=False).to_string())
    print(f"Manifest: {CANONICAL_CSV}")
    print(f"SHA256  : {sha256_file(CANONICAL_CSV)}")

    print("\nFORMAT")
    print(f"Bands: {','.join(S2_BANDS)}")
    print(f"Patch: {PATCH_PIXELS} x {PATCH_PIXELS} pixels")
    print(f"Footprint: {PATCH_METERS:.0f} m")
    print("Frames: t0 / t-90 / t-360")
    print(
        "Corrected SCL gate: clear/requested >= "
        f"{MIN_CLEAR_OVER_REQUESTED:.2f}"
    )

    raw_bytes = (
        len(canonical)
        * 3
        * PATCH_PIXELS
        * PATCH_PIXELS
        * len(S2_BANDS)
        * 2
    )
    print(
        "Approx raw pixel payload for 368 x 3 frames: "
        f"{raw_bytes / (1024**2):.1f} MiB "
        "(GeoTIFF overhead/compression excluded)"
    )

    if args.preflight_only:
        print("\nNo imagery queried or downloaded (--preflight-only).")
        return

    initialize_ee(args.project)

    selected = canonical.copy()
    if args.limit > 0:
        selected = selected.head(args.limit).copy()
        print(f"\nLIMIT active: {len(selected)} controls")

    selected_ids = set(selected["control_id"].astype(str))
    latest = load_checkpoint_latest()

    def checkpoint_needs_v10_1_repair(obj: dict[str, Any]) -> bool:
        frames = obj.get("frames", []) or []
        return any(
            str(frame.get("frame_selection_reason") or "")
            == "all_candidate_QA_queries_failed"
            for frame in frames
        )

    completed_ids = {
        cid
        for cid, obj in latest.items()
        if obj.get("complete") is True
        and not checkpoint_needs_v10_1_repair(obj)
    }

    repair_ids = {
        cid
        for cid, obj in latest.items()
        if checkpoint_needs_v10_1_repair(obj)
    }

    if repair_ids:
        print(
            f"Auto-retrying {len(repair_ids & selected_ids)} selected controls "
            "from the v10 temporal asset-ID/QA failure state."
        )

    todo = selected[
        ~selected["control_id"].astype(str).isin(completed_ids)
    ].copy()

    print(
        "\nPreviously complete in selected set: "
        f"{len(selected_ids & completed_ids)}"
    )
    print(f"This run: {len(todo)}")

    if not args.no_lab_mirror:
        print(
            "Lab mirror:",
            "ACTIVE"
            if lab_mounted()
            else "currently unavailable; local build continues",
        )
        print(f"Lab root: {LAB_ROOT}")

    done = 0

    if len(todo):
        with ThreadPoolExecutor(
            max_workers=max(1, int(args.workers))
        ) as pool:
            futures = {
                pool.submit(process_control, row, args): str(row["control_id"])
                for _, row in todo.iterrows()
            }

            for future in as_completed(futures):
                control_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "complete": False,
                        "retryable": True,
                        "control_id": control_id,
                        "sample_summary": {
                            "control_id": control_id,
                            "sample_result": "UNHANDLED_ERROR",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        "frames": [],
                        "traceback": traceback.format_exc(),
                    }

                append_checkpoint(result)
                latest[control_id] = result
                done += 1

                summary = result.get("sample_summary", {})
                print(
                    f"[{done}/{len(todo)}] {control_id} -> "
                    f"{summary.get('sample_result', 'UNKNOWN')}"
                )

                if done % 10 == 0:
                    save_outputs(
                        latest,
                        selected_control_ids=selected_ids,
                        args=args,
                    )

    latest = load_checkpoint_latest()
    sample_df, frame_df, strict_local = save_outputs(
        latest,
        selected_control_ids=selected_ids,
        args=args,
    )

    print("\n" + "=" * 96)
    print("FINAL BUILD SUMMARY")
    print("=" * 96)
    print(BUILD_SUMMARY.read_text(encoding="utf-8"))

    print("OUTPUTS")
    for path in [
        CANONICAL_CSV,
        CANONICAL_SHA,
        FRAME_MANIFEST,
        SAMPLE_AUDIT,
        MF_TECHNICAL_LOCAL,
        MF_STRICT_LOCAL,
        MF_STRICT_LAB,
        BUILD_SUMMARY,
        BUILD_XLSX,
        CHECKPOINT_JSONL,
    ]:
        print(path)

    if METHANEFUSE_EVAL_CSV.exists():
        print("\nMethaneFuse strict local eval CSV:")
        print(METHANEFUSE_EVAL_CSV)

    print("\nIMPORTANT")
    print("- Canonical population remains 368 validated t0 controls.")
    print(
        "- Model-ready count may be lower if t-90/t-360 references are "
        "missing or fail corrected QA."
    )
    print(
        "- B1/B2 provenance is preserved; these are no-detection weak "
        "controls, not Q=0."
    )
    print(
        "- Multiple MethaneAIR flights supporting one t0 are already "
        "deduplicated at control level."
    )
    print(
        "- Local files are authoritative; SMB mirror failure never "
        "invalidates local progress."
    )
    print("- Re-run the same command to resume; valid TIFFs are reused.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted by user. Existing local TIFFs/checkpoint are preserved."
        )
        raise
    except Exception as exc:
        print(
            f"\nERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
