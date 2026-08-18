#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_methanesat_120_paired_image_benchmark.py

Build the frozen MethaneSAT image-only paired benchmark from:

  03_best_one_negative_per_positive.csv

Expected design:
  120 L4-detected positive observations
  120 same-location, far-time temporal controls

For EACH pair this script downloads BOTH acquisitions again from the official
MethaneSAT L3 XCH4 Earth Engine collection using the exact same:
  - latitude / longitude
  - 480 m x 480 m source-centered crop
  - 45 m requested scale
  - XCH4 band only

This avoids mixing old NPZ preprocessing with newly downloaded controls.

Outputs
-------
<out>/
  raw_tif/
    positive/
    temporal_control/
  npz/
    samples/
    pairs/
  manifests/
    00_input_pairs_frozen.csv
    01_pair_build_audit.csv
    02_valid_pairs.csv
    03_sample_image_features.csv
    04_failed_pairs.csv
  SUMMARY_PAIRED_BENCHMARK.md

NPZ semantics
-------------
Each sample NPZ contains:
  xch4          : float32 [224, 224], raw XCH4 values (ppb-like product units)
  valid_mask    : uint8   [224, 224]
  binary_label  : int8 scalar (1 = L4-detected positive, 0 = temporal control)
  class_name    : "positive" or "temporal_control"
  pair_id
  collection_id
  latitude
  longitude
  acquisition_time
  control_evidence_tier

Each pair NPZ contains:
  positive_xch4
  temporal_control_xch4
  positive_valid_mask
  temporal_control_valid_mask
  pair_id
  latitude
  longitude
  positive_collection_id
  temporal_control_collection_id
  positive_time
  temporal_control_time
  abs_delta_days
  control_evidence_tier

IMPORTANT LABEL CAVEAT
----------------------
binary_label=0 means "same-site far-time temporal control", NOT externally
confirmed zero emission, unless control_evidence_tier explicitly says so.

Resume safety
-------------
- valid existing TIFF => skip
- new download => .part
- validate .part with rasterio
- fsync
- atomic os.replace to final TIFF
- NPZ files are written to .tmp and atomically renamed
- local JSONL checkpoint is fsync'd
- rerun the SAME command after interruption
- --restart-checkpoint clears only the local log; it never deletes TIFF/NPZ data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import ee
except ImportError as exc:
    raise SystemExit(
        "Missing earthengine-api.\n"
        "Install with:\n"
        "  python -m pip install earthengine-api\n"
    ) from exc

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "Missing requests.\n"
        "Install with:\n"
        "  python -m pip install requests\n"
    ) from exc

try:
    import rasterio
except ImportError as exc:
    raise SystemExit(
        "Missing rasterio.\n"
        "Install with:\n"
        "  python -m pip install rasterio\n"
    ) from exc


L3_ASSET = "projects/edf-methanesat-ee/assets/public-preview/L3concentration"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--pairs",
        default=(
            "~/methane_release_project/"
            "methanesat_156_far_temporal_negative_search/"
            "03_best_one_negative_per_positive.csv"
        ),
        help="Frozen one-control-per-positive manifest.",
    )
    p.add_argument(
        "--out",
        default=(
            "/Volumes/engg-leung/dora lin/"
            "MethaneSAT_MethaneFuse/05_paired_image_benchmark_120"
        ),
        help="Lab/server benchmark output root.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=(
            "~/methane_release_project/"
            "methanesat_120_paired_benchmark_checkpoints"
        ),
        help="LOCAL checkpoint directory; keep off SMB.",
    )
    p.add_argument("--project", default="methane-release-gee")

    p.add_argument(
        "--expected-pairs",
        type=int,
        default=120,
        help="Stop if the frozen manifest is not exactly this size.",
    )
    p.add_argument("--crop-half-m", type=float, default=240.0)
    p.add_argument("--scale-m", type=float, default=45.0)
    p.add_argument("--min-valid-fraction", type=float, default=0.50)
    p.add_argument("--npz-size", type=int, default=224)

    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--sleep", type=float, default=0.15)

    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional smoke test. 0 = all frozen pairs.",
    )
    p.add_argument(
        "--restart-checkpoint",
        action="store_true",
        help="Clear only local status JSONL. Existing valid TIFF/NPZ files remain.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def initialize_ee(project: str):
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed.\n"
            "Run:\n"
            "  earthengine authenticate\n"
            f"Then rerun with --project {project!r}\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def clean_str(v: Any) -> str:
    s = str(v if v is not None else "").strip()
    return "" if s.lower() in {"", "nan", "none", "null"} else s


def norm_collection(v: Any) -> str:
    s = clean_str(v)
    if s[:1].lower() == "c":
        s = s[1:]
    return s.upper()


def safe_name(v: Any) -> str:
    s = clean_str(v)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("._-") or "unknown"


def scalar(v: Any):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.isoformat()
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def stable_pair_id(row: pd.Series) -> str:
    text = "|".join([
        clean_str(row.get("positive_id")),
        norm_collection(row.get("positive_collection_id")),
        norm_collection(row.get("candidate_collection_id")),
        f"{float(row.get('latitude')):.8f}",
        f"{float(row.get('longitude')):.8f}",
    ])
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:14]


def ensure_output_mount(outdir: Path):
    if str(outdir).startswith("/Volumes/"):
        parts = outdir.parts
        if len(parts) < 3:
            raise RuntimeError(f"Invalid output path: {outdir}")
        mount_root = Path("/", "Volumes", parts[2])
        if not mount_root.exists():
            raise RuntimeError(
                f"SMB SHARE DISCONNECTED: {mount_root}\n"
                "Reconnect it and rerun the SAME command."
            )

    outdir.mkdir(parents=True, exist_ok=True)

    probe = outdir / ".write_test.tmp"
    try:
        with probe.open("wb") as f:
            f.write(b"ok")
            f.flush()
            os.fsync(f.fileno())
        probe.unlink()
    except Exception as exc:
        raise RuntimeError(
            f"Output is not writable: {outdir}\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc


def append_jsonl_fsync(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = {str(k): scalar(v) for k, v in rec.items()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    # np.savez_compressed appends ".npz" when given a string without .npz;
    # passing an open file handle avoids that surprise.
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "positive_id",
    "positive_sample_id",
    "positive_collection_id",
    "positive_time",
    "latitude",
    "longitude",
    "candidate_collection_id",
    "candidate_time_start",
    "abs_delta_days",
    "negative_evidence_tier",
]


def load_pairs(path: Path, expected_pairs: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Pair manifest not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Pair manifest missing required columns: {missing}\n"
            f"Columns: {list(df.columns)}"
        )

    if len(df) != expected_pairs:
        raise RuntimeError(
            f"Expected exactly {expected_pairs} frozen pairs, got {len(df)}.\n"
            "Stop rather than silently building a different benchmark."
        )

    if df["positive_id"].duplicated().any():
        dup = df[df["positive_id"].duplicated(False)]
        raise RuntimeError(
            "Frozen pair manifest has duplicate positive_id rows:\n"
            + dup[
                [
                    "positive_id",
                    "positive_sample_id",
                    "positive_collection_id",
                    "candidate_collection_id",
                ]
            ].to_string(index=False)
        )

    df = df.copy()
    df["positive_collection_id"] = df["positive_collection_id"].map(norm_collection)
    df["candidate_collection_id"] = df["candidate_collection_id"].map(norm_collection)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["abs_delta_days"] = pd.to_numeric(df["abs_delta_days"], errors="coerce")

    bad = df[
        df["latitude"].isna()
        | df["longitude"].isna()
        | df["positive_collection_id"].eq("")
        | df["candidate_collection_id"].eq("")
    ]
    if len(bad):
        raise RuntimeError(
            f"{len(bad)} frozen pairs have missing coordinate/collection fields."
        )

    too_close = df[df["abs_delta_days"] < 90]
    if len(too_close):
        raise RuntimeError(
            f"{len(too_close)} frozen pairs violate |delta time| >= 90 days."
        )

    same = df[
        df["positive_collection_id"].eq(df["candidate_collection_id"])
    ]
    if len(same):
        raise RuntimeError(
            f"{len(same)} frozen pairs use the same positive/control collection."
        )

    df["pair_id"] = [f"PAIR_{i+1:04d}_{stable_pair_id(r)}" for i, (_, r) in enumerate(df.iterrows())]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Earth Engine / download
# ---------------------------------------------------------------------

def get_l3_image(collection_id: str, lat: float, lon: float):
    cid = norm_collection(collection_id)
    point = ee.Geometry.Point([float(lon), float(lat)])
    base = ee.ImageCollection(L3_ASSET).filterBounds(point)

    for variant in [f"c{cid}", cid]:
        ic = base.filter(ee.Filter.eq("collection_id", variant))
        n = int(ic.size().getInfo())
        if n > 0:
            img = ee.Image(ic.first())
            props = img.toDictionary([
                "collection_id",
                "target_id",
                "time_coverage_start",
                "time_coverage_end",
                "system:time_start",
                "system:index",
            ]).getInfo()
            return img, variant, props

    raise RuntimeError(
        f"No L3 image found for collection {collection_id!r} at "
        f"({lat:.6f}, {lon:.6f})"
    )


def build_download_url(collection_id: str, lat: float, lon: float,
                       crop_half_m: float, scale_m: float):
    img, matched_cid, props = get_l3_image(collection_id, lat, lon)

    point = ee.Geometry.Point([float(lon), float(lat)])
    region = point.buffer(float(crop_half_m)).bounds()

    url = img.select("XCH4").getDownloadURL({
        "region": region,
        "scale": float(scale_m),
        "format": "GEO_TIFF",
    })
    return url, matched_cid, props


def stream_download(url: str, part: Path, timeout: int):
    if part.exists():
        part.unlink()

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        ctype = str(r.headers.get("Content-Type", "")).lower()
        if "text/html" in ctype or "application/json" in ctype:
            preview = r.content[:1000]
            raise RuntimeError(
                f"Unexpected download Content-Type={ctype}; "
                f"response preview={preview!r}"
            )

        with part.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
            f.flush()
            os.fsync(f.fileno())


# ---------------------------------------------------------------------
# TIFF validation / preprocessing
# ---------------------------------------------------------------------

def read_valid_array(path: Path):
    with rasterio.open(path) as ds:
        if ds.count < 1:
            raise RuntimeError("TIFF has no bands")

        raw = ds.read(1, masked=False).astype(np.float64)
        dmask = ds.dataset_mask() > 0

        valid = dmask & np.isfinite(raw)

        if ds.nodata is not None:
            valid &= raw != ds.nodata

        # Broad sanity: product should not contain non-positive XCH4 as valid.
        valid &= raw > 0

        arr = np.where(valid, raw, np.nan)

        meta = {
            "width": int(ds.width),
            "height": int(ds.height),
            "bands": int(ds.count),
            "dtype": str(ds.dtypes[0]),
            "crs": str(ds.crs) if ds.crs else "",
            "file_bytes": int(path.stat().st_size),
            "valid_pixel_fraction": float(valid.mean()) if valid.size else 0.0,
        }

        vals = arr[np.isfinite(arr)]
        if vals.size:
            meta.update({
                "xch4_min": float(np.min(vals)),
                "xch4_median": float(np.median(vals)),
                "xch4_mean": float(np.mean(vals)),
                "xch4_p95": float(np.percentile(vals, 95)),
                "xch4_max": float(np.max(vals)),
            })
        else:
            meta.update({
                "xch4_min": None,
                "xch4_median": None,
                "xch4_mean": None,
                "xch4_p95": None,
                "xch4_max": None,
            })

        return arr, valid, meta


def validate_tiff(path: Path, min_valid_fraction: float):
    if not path.exists() or path.stat().st_size <= 0:
        return False, {}, None, None

    try:
        arr, valid, meta = read_valid_array(path)
        ok = (
            arr.ndim == 2
            and arr.shape[0] >= 2
            and arr.shape[1] >= 2
            and meta["valid_pixel_fraction"] >= min_valid_fraction
            and np.isfinite(arr).any()
        )
        return bool(ok), meta, arr, valid
    except Exception:
        return False, {}, None, None


def resize_linear_axis(a: np.ndarray, new_n: int, axis: int):
    """Pure NumPy linear resize along one axis."""
    a = np.asarray(a, dtype=np.float64)

    if axis == 1:
        old_n = a.shape[1]
        if old_n == new_n:
            return a.copy()
        old_x = np.linspace(0.0, 1.0, old_n)
        new_x = np.linspace(0.0, 1.0, new_n)
        out = np.empty((a.shape[0], new_n), dtype=np.float64)
        for i in range(a.shape[0]):
            out[i] = np.interp(new_x, old_x, a[i])
        return out

    if axis == 0:
        old_n = a.shape[0]
        if old_n == new_n:
            return a.copy()
        old_x = np.linspace(0.0, 1.0, old_n)
        new_x = np.linspace(0.0, 1.0, new_n)
        out = np.empty((new_n, a.shape[1]), dtype=np.float64)
        for j in range(a.shape[1]):
            out[:, j] = np.interp(new_x, old_x, a[:, j])
        return out

    raise ValueError("axis must be 0 or 1")


def resize_2d(a: np.ndarray, size: int):
    return resize_linear_axis(
        resize_linear_axis(a, size, axis=1),
        size,
        axis=0,
    )


def resize_float_with_mask(arr: np.ndarray, valid: np.ndarray, size: int):
    """
    Weighted bilinear-like resize without SciPy.

    Missing pixels are not treated as zero XCH4:
      resize(data * weight) / resize(weight)
    """
    w = valid.astype(np.float64)
    filled = np.where(valid, arr, 0.0)

    num = resize_2d(filled, size)
    den = resize_2d(w, size)

    out_valid = den >= 0.50
    out = np.full((size, size), np.nan, dtype=np.float32)
    out[out_valid] = (num[out_valid] / den[out_valid]).astype(np.float32)

    return out, out_valid.astype(np.uint8)


def image_features(x: np.ndarray, valid_mask: np.ndarray, crop_size_m: float):
    valid = valid_mask.astype(bool) & np.isfinite(x)
    vals = x[valid]

    result = {
        "valid_fraction_224": float(valid.mean()) if valid.size else 0.0,
        "mean": None,
        "median": None,
        "p90": None,
        "p95": None,
        "p99": None,
        "max": None,
        "center_r60_mean": None,
        "ring_r120_220_median": None,
        "center_minus_ring": None,
    }

    if vals.size == 0:
        return result

    result.update({
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(np.max(vals)),
    })

    h, w = x.shape
    yy, xx = np.mgrid[0:h, 0:w]

    # Pixel-center coordinates mapped to the fixed physical crop.
    xm = ((xx + 0.5) / w - 0.5) * crop_size_m
    ym = ((yy + 0.5) / h - 0.5) * crop_size_m
    rr = np.sqrt(xm * xm + ym * ym)

    center = valid & (rr <= 60.0)
    ring = valid & (rr >= 120.0) & (rr <= 220.0)

    if center.any():
        result["center_r60_mean"] = float(np.mean(x[center]))
    if ring.any():
        result["ring_r120_220_median"] = float(np.median(x[ring]))

    if (
        result["center_r60_mean"] is not None
        and result["ring_r120_220_median"] is not None
    ):
        result["center_minus_ring"] = (
            result["center_r60_mean"]
            - result["ring_r120_220_median"]
        )

    return result


# ---------------------------------------------------------------------
# Build one sample
# ---------------------------------------------------------------------

def make_sample_paths(base_tif: Path, sample_npz_dir: Path,
                      pair_id: str, class_name: str,
                      collection_id: str):
    short = "POS" if class_name == "positive" else "CTRL"
    cid = safe_name(norm_collection(collection_id))
    tif = base_tif / f"{pair_id}__{short}__c{cid}.tif"
    npz = sample_npz_dir / f"{pair_id}__{short}__c{cid}.npz"
    return tif, npz


def build_sample(
    *,
    pair_id: str,
    class_name: str,
    binary_label: int,
    collection_id: str,
    acquisition_time: str,
    lat: float,
    lon: float,
    evidence_tier: str,
    tif_path: Path,
    npz_path: Path,
    args,
):
    # Existing final TIFF is accepted only after actual raster QA.
    ok, tmeta, arr, valid = validate_tiff(
        tif_path,
        args.min_valid_fraction,
    )

    download_status = "SKIPPED_VALID_EXISTING"

    if not ok:
        if tif_path.exists():
            bad = tif_path.with_suffix(
                tif_path.suffix + f".invalid.{int(time.time())}"
            )
            tif_path.rename(bad)

        last_error = ""
        matched_cid = ""
        ee_props = {}

        for attempt in range(1, args.retries + 1):
            part = tif_path.with_suffix(tif_path.suffix + ".part")
            try:
                url, matched_cid, ee_props = build_download_url(
                    collection_id,
                    lat,
                    lon,
                    args.crop_half_m,
                    args.scale_m,
                )

                stream_download(url, part, args.timeout)

                ok, tmeta, arr, valid = validate_tiff(
                    part,
                    args.min_valid_fraction,
                )
                if not ok:
                    raise RuntimeError(
                        "Downloaded TIFF failed raster/valid-fraction QA"
                    )

                os.replace(part, tif_path)
                download_status = "DOWNLOADED"
                last_error = ""
                break

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

                if part.exists():
                    try:
                        part.unlink()
                    except Exception:
                        pass

                if attempt < args.retries:
                    wait = min(60, 3 * (2 ** (attempt - 1)))
                    time.sleep(wait)

        if not ok:
            return {
                "sample_ok": False,
                "class_name": class_name,
                "binary_label": binary_label,
                "collection_id": norm_collection(collection_id),
                "acquisition_time": acquisition_time,
                "output_tif": str(tif_path),
                "output_npz": str(npz_path),
                "download_status": "FAILED",
                "error": last_error,
            }
    else:
        matched_cid = norm_collection(collection_id)
        ee_props = {}

    x224, mask224 = resize_float_with_mask(
        arr,
        valid,
        args.npz_size,
    )

    # Require the standardized image to remain meaningfully valid.
    std_valid_fraction = float(mask224.mean())
    if std_valid_fraction < args.min_valid_fraction:
        return {
            "sample_ok": False,
            "class_name": class_name,
            "binary_label": binary_label,
            "collection_id": norm_collection(collection_id),
            "acquisition_time": acquisition_time,
            "output_tif": str(tif_path),
            "output_npz": str(npz_path),
            "download_status": download_status,
            "error": (
                f"Standardized 224x224 valid fraction "
                f"{std_valid_fraction:.4f} < {args.min_valid_fraction:.4f}"
            ),
            **tmeta,
        }

    atomic_npz(
        npz_path,
        xch4=x224.astype(np.float32),
        valid_mask=mask224.astype(np.uint8),
        binary_label=np.array(binary_label, dtype=np.int8),
        class_name=np.array(class_name),
        pair_id=np.array(pair_id),
        collection_id=np.array(norm_collection(collection_id)),
        latitude=np.array(lat, dtype=np.float64),
        longitude=np.array(lon, dtype=np.float64),
        acquisition_time=np.array(clean_str(acquisition_time)),
        control_evidence_tier=np.array(clean_str(evidence_tier)),
    )

    features = image_features(
        x224,
        mask224,
        crop_size_m=2 * args.crop_half_m,
    )

    return {
        "sample_ok": True,
        "class_name": class_name,
        "binary_label": binary_label,
        "collection_id": norm_collection(collection_id),
        "matched_collection_id": norm_collection(matched_cid),
        "acquisition_time": acquisition_time,
        "output_tif": str(tif_path),
        "output_npz": str(npz_path),
        "download_status": download_status,
        "error": "",
        **tmeta,
        **{f"feature_{k}": v for k, v in features.items()},
        "ee_target_id": clean_str(ee_props.get("target_id")),
        "ee_time_coverage_start": clean_str(
            ee_props.get("time_coverage_start")
        ),
        "ee_system_index": clean_str(ee_props.get("system:index")),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    pair_path = Path(args.pairs).expanduser()
    outdir = Path(args.out).expanduser()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()

    pairs = load_pairs(pair_path, args.expected_pairs)

    # Freeze the full 120 before applying --limit.
    if args.limit > 0:
        work = pairs.head(args.limit).copy()
    else:
        work = pairs.copy()

    ensure_output_mount(outdir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    status_jsonl = checkpoint_dir / "build_status.jsonl"
    if args.restart_checkpoint and status_jsonl.exists():
        status_jsonl.unlink()
        print("[RESTART] local status log removed.")
        print("[RESTART] existing valid TIFF/NPZ files are NOT deleted.")

    initialize_ee(args.project)

    raw_pos_dir = outdir / "raw_tif" / "positive"
    raw_ctrl_dir = outdir / "raw_tif" / "temporal_control"
    sample_npz_dir = outdir / "npz" / "samples"
    pair_npz_dir = outdir / "npz" / "pairs"
    manifest_dir = outdir / "manifests"

    for d in [
        raw_pos_dir,
        raw_ctrl_dir,
        sample_npz_dir,
        pair_npz_dir,
        manifest_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    atomic_csv(
        pairs,
        manifest_dir / "00_input_pairs_frozen.csv",
    )

    print("=" * 88)
    print("METHANESAT 120-PAIR IMAGE-ONLY BENCHMARK BUILDER")
    print("=" * 88)
    print("Frozen pairs:", len(pairs))
    print("This run:", len(work))
    print("Output:", outdir)
    print("Checkpoint:", status_jsonl)
    print(
        "Processing: BOTH positive and temporal-control XCH4 are downloaded "
        "with the same 480 m / 45 m pipeline."
    )
    print()

    pair_audit = []
    feature_rows = []

    for i, row in work.iterrows():
        pair_id = row["pair_id"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        pos_cid = norm_collection(row["positive_collection_id"])
        ctrl_cid = norm_collection(row["candidate_collection_id"])
        evidence = clean_str(row["negative_evidence_tier"])

        print("-" * 88)
        print(f"[{i+1}/{len(work)}] {pair_id}")
        print("positive:", pos_cid)
        print("control :", ctrl_cid)
        print("delta d :", float(row["abs_delta_days"]))
        print("evidence:", evidence)

        pos_tif, pos_npz = make_sample_paths(
            raw_pos_dir,
            sample_npz_dir,
            pair_id,
            "positive",
            pos_cid,
        )
        ctrl_tif, ctrl_npz = make_sample_paths(
            raw_ctrl_dir,
            sample_npz_dir,
            pair_id,
            "temporal_control",
            ctrl_cid,
        )

        try:
            pos_result = build_sample(
                pair_id=pair_id,
                class_name="positive",
                binary_label=1,
                collection_id=pos_cid,
                acquisition_time=clean_str(row["positive_time"]),
                lat=lat,
                lon=lon,
                evidence_tier=evidence,
                tif_path=pos_tif,
                npz_path=pos_npz,
                args=args,
            )

            print(
                "  POS :",
                "PASS" if pos_result["sample_ok"] else "FAIL",
                pos_result.get("download_status"),
                f"valid={pos_result.get('valid_pixel_fraction')}",
            )

            ctrl_result = build_sample(
                pair_id=pair_id,
                class_name="temporal_control",
                binary_label=0,
                collection_id=ctrl_cid,
                acquisition_time=clean_str(row["candidate_time_start"]),
                lat=lat,
                lon=lon,
                evidence_tier=evidence,
                tif_path=ctrl_tif,
                npz_path=ctrl_npz,
                args=args,
            )

            print(
                "  CTRL:",
                "PASS" if ctrl_result["sample_ok"] else "FAIL",
                ctrl_result.get("download_status"),
                f"valid={ctrl_result.get('valid_pixel_fraction')}",
            )

            pair_ok = bool(
                pos_result["sample_ok"]
                and ctrl_result["sample_ok"]
            )

            pair_npz = pair_npz_dir / f"{pair_id}.npz"

            if pair_ok:
                with np.load(pos_npz, allow_pickle=False) as pz:
                    pos_x = pz["xch4"].copy()
                    pos_mask = pz["valid_mask"].copy()
                with np.load(ctrl_npz, allow_pickle=False) as nz:
                    ctrl_x = nz["xch4"].copy()
                    ctrl_mask = nz["valid_mask"].copy()

                atomic_npz(
                    pair_npz,
                    positive_xch4=pos_x.astype(np.float32),
                    temporal_control_xch4=ctrl_x.astype(np.float32),
                    positive_valid_mask=pos_mask.astype(np.uint8),
                    temporal_control_valid_mask=ctrl_mask.astype(np.uint8),
                    pair_id=np.array(pair_id),
                    latitude=np.array(lat, dtype=np.float64),
                    longitude=np.array(lon, dtype=np.float64),
                    positive_collection_id=np.array(pos_cid),
                    temporal_control_collection_id=np.array(ctrl_cid),
                    positive_time=np.array(clean_str(row["positive_time"])),
                    temporal_control_time=np.array(
                        clean_str(row["candidate_time_start"])
                    ),
                    abs_delta_days=np.array(
                        float(row["abs_delta_days"]),
                        dtype=np.float32,
                    ),
                    control_evidence_tier=np.array(evidence),
                )

            audit_row = {
                **row.to_dict(),
                "pair_id": pair_id,
                "pair_valid": pair_ok,
                "positive_tif": str(pos_tif),
                "temporal_control_tif": str(ctrl_tif),
                "positive_npz": str(pos_npz),
                "temporal_control_npz": str(ctrl_npz),
                "pair_npz": str(pair_npz) if pair_ok else "",
                "positive_status": (
                    "PASS" if pos_result["sample_ok"] else "FAIL"
                ),
                "control_status": (
                    "PASS" if ctrl_result["sample_ok"] else "FAIL"
                ),
                "positive_error": pos_result.get("error", ""),
                "control_error": ctrl_result.get("error", ""),
                "positive_valid_fraction": pos_result.get(
                    "valid_pixel_fraction"
                ),
                "control_valid_fraction": ctrl_result.get(
                    "valid_pixel_fraction"
                ),
                "positive_width": pos_result.get("width"),
                "positive_height": pos_result.get("height"),
                "control_width": ctrl_result.get("width"),
                "control_height": ctrl_result.get("height"),
            }
            pair_audit.append(audit_row)

            for result in [pos_result, ctrl_result]:
                feature_rows.append({
                    "pair_id": pair_id,
                    "positive_id": row["positive_id"],
                    "positive_sample_id": row["positive_sample_id"],
                    "latitude": lat,
                    "longitude": lon,
                    "abs_delta_days": row["abs_delta_days"],
                    "control_evidence_tier": evidence,
                    "class_name": result["class_name"],
                    "binary_label": result["binary_label"],
                    "collection_id": result["collection_id"],
                    "acquisition_time": result["acquisition_time"],
                    "sample_ok": result["sample_ok"],
                    "valid_pixel_fraction_raw": result.get(
                        "valid_pixel_fraction"
                    ),
                    "valid_fraction_224": result.get(
                        "feature_valid_fraction_224"
                    ),
                    "mean": result.get("feature_mean"),
                    "median": result.get("feature_median"),
                    "p90": result.get("feature_p90"),
                    "p95": result.get("feature_p95"),
                    "p99": result.get("feature_p99"),
                    "max": result.get("feature_max"),
                    "center_r60_mean": result.get(
                        "feature_center_r60_mean"
                    ),
                    "ring_r120_220_median": result.get(
                        "feature_ring_r120_220_median"
                    ),
                    "center_minus_ring": result.get(
                        "feature_center_minus_ring"
                    ),
                })

            append_jsonl_fsync(
                status_jsonl,
                {
                    "pair_id": pair_id,
                    "pair_valid": pair_ok,
                    "positive_collection_id": pos_cid,
                    "control_collection_id": ctrl_cid,
                    "positive_status": audit_row["positive_status"],
                    "control_status": audit_row["control_status"],
                    "positive_tif": str(pos_tif),
                    "control_tif": str(ctrl_tif),
                    "pair_npz": audit_row["pair_npz"],
                },
            )

            print("  PAIR:", "PASS" if pair_ok else "FAIL")

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            print("Completed final TIFF/NPZ files remain intact.")
            print("Rerun the SAME command to resume.")
            raise

        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print("  PAIR ERROR:", err)
            pair_audit.append({
                **row.to_dict(),
                "pair_id": pair_id,
                "pair_valid": False,
                "positive_tif": str(pos_tif),
                "temporal_control_tif": str(ctrl_tif),
                "positive_npz": str(pos_npz),
                "temporal_control_npz": str(ctrl_npz),
                "pair_npz": "",
                "positive_status": "UNKNOWN",
                "control_status": "UNKNOWN",
                "positive_error": err,
                "control_error": err,
            })
            append_jsonl_fsync(
                status_jsonl,
                {
                    "pair_id": pair_id,
                    "pair_valid": False,
                    "error": err,
                },
            )

        time.sleep(args.sleep)

    # -----------------------------------------------------------------
    # Final audit: do not trust session success alone.
    # Re-validate all expected pairs in the work set from disk.
    # -----------------------------------------------------------------

    final_rows = []

    for _, row in work.iterrows():
        pair_id = row["pair_id"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        pos_cid = norm_collection(row["positive_collection_id"])
        ctrl_cid = norm_collection(row["candidate_collection_id"])

        pos_tif, pos_npz = make_sample_paths(
            raw_pos_dir, sample_npz_dir, pair_id, "positive", pos_cid
        )
        ctrl_tif, ctrl_npz = make_sample_paths(
            raw_ctrl_dir, sample_npz_dir, pair_id, "temporal_control", ctrl_cid
        )
        pair_npz = pair_npz_dir / f"{pair_id}.npz"

        pos_ok, pos_meta, _, _ = validate_tiff(
            pos_tif, args.min_valid_fraction
        )
        ctrl_ok, ctrl_meta, _, _ = validate_tiff(
            ctrl_tif, args.min_valid_fraction
        )

        pos_npz_ok = pos_npz.exists() and pos_npz.stat().st_size > 0
        ctrl_npz_ok = ctrl_npz.exists() and ctrl_npz.stat().st_size > 0
        pair_npz_ok = pair_npz.exists() and pair_npz.stat().st_size > 0

        pair_valid = bool(
            pos_ok
            and ctrl_ok
            and pos_npz_ok
            and ctrl_npz_ok
            and pair_npz_ok
        )

        final_rows.append({
            **row.to_dict(),
            "pair_id": pair_id,
            "pair_valid": pair_valid,
            "positive_tif_valid": pos_ok,
            "control_tif_valid": ctrl_ok,
            "positive_npz_exists": pos_npz_ok,
            "control_npz_exists": ctrl_npz_ok,
            "pair_npz_exists": pair_npz_ok,
            "positive_valid_fraction": pos_meta.get(
                "valid_pixel_fraction"
            ),
            "control_valid_fraction": ctrl_meta.get(
                "valid_pixel_fraction"
            ),
            "positive_tif": str(pos_tif),
            "temporal_control_tif": str(ctrl_tif),
            "positive_npz": str(pos_npz),
            "temporal_control_npz": str(ctrl_npz),
            "pair_npz": str(pair_npz),
        })

    final_audit = pd.DataFrame(final_rows)
    valid_pairs = final_audit[final_audit["pair_valid"].eq(True)].copy()
    failed_pairs = final_audit[~final_audit["pair_valid"].eq(True)].copy()

    atomic_csv(
        final_audit,
        manifest_dir / "01_pair_build_audit.csv",
    )
    atomic_csv(
        valid_pairs,
        manifest_dir / "02_valid_pairs.csv",
    )

    features = pd.DataFrame(feature_rows)
    if len(features):
        atomic_csv(
            features,
            manifest_dir / "03_sample_image_features.csv",
        )
    else:
        atomic_csv(
            pd.DataFrame(columns=[
                "pair_id", "class_name", "binary_label", "collection_id"
            ]),
            manifest_dir / "03_sample_image_features.csv",
        )

    atomic_csv(
        failed_pairs,
        manifest_dir / "04_failed_pairs.csv",
    )

    # Summary
    evidence_counts = (
        valid_pairs["negative_evidence_tier"].value_counts()
        if len(valid_pairs) else pd.Series(dtype=int)
    )
    time_counts = (
        valid_pairs["time_tier"].value_counts()
        if len(valid_pairs) and "time_tier" in valid_pairs.columns
        else pd.Series(dtype=int)
    )

    lines = [
        "# MethaneSAT paired image-only benchmark",
        "",
        "## Frozen input",
        f"- Frozen pair manifest: {len(pairs)} pairs",
        f"- Processed in this run: {len(work)} pairs",
        "",
        "## Final disk audit",
        f"- Valid paired samples: {len(valid_pairs)} / {len(work)}",
        f"- Failed/incomplete pairs: {len(failed_pairs)}",
        f"- Positive TIFFs expected: {len(work)}",
        f"- Temporal-control TIFFs expected: {len(work)}",
        "",
        "## Image construction",
        f"- Exact same pair coordinate for positive/control",
        f"- Physical crop: {2*args.crop_half_m:.0f} m x {2*args.crop_half_m:.0f} m",
        f"- Earth Engine requested L3 XCH4 scale: {args.scale_m:g} m",
        f"- Raw TIFF minimum valid fraction: {args.min_valid_fraction:.2f}",
        f"- Standardized NPZ size: {args.npz_size} x {args.npz_size}",
        "- Positive and control are rebuilt through the SAME preprocessing path.",
        "",
        "## Control-label semantics",
        "- binary_label=0 means same-site far-time temporal control.",
        "- It does NOT mean externally confirmed zero emission unless the evidence tier explicitly says so.",
        "",
        "## Valid pair evidence tiers",
    ]

    if len(evidence_counts):
        for k, v in evidence_counts.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- None")

    lines += ["", "## Valid pair time tiers"]
    if len(time_counts):
        for k, v in time_counts.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- Not available")

    lines += [
        "",
        "## Next analysis",
        "Use manifests/03_sample_image_features.csv for the first paired image-only baseline.",
        "Do not use latitude, longitude, collection ID, target ID, date, flux, or L4 metadata as model inputs.",
    ]

    (outdir / "SUMMARY_PAIRED_BENCHMARK.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("FINAL DISK AUDIT")
    print("=" * 88)
    print("Valid pairs       :", f"{len(valid_pairs)}/{len(work)}")
    print("Failed/incomplete :", len(failed_pairs))
    print("Positive TIFFs    :", int(final_audit["positive_tif_valid"].sum()))
    print("Control TIFFs     :", int(final_audit["control_tif_valid"].sum()))
    print("Pair NPZ          :", int(final_audit["pair_npz_exists"].sum()))
    print()
    print("Output:", outdir)
    print("Important files:")
    for fn in [
        "SUMMARY_PAIRED_BENCHMARK.md",
        "manifests/01_pair_build_audit.csv",
        "manifests/02_valid_pairs.csv",
        "manifests/03_sample_image_features.csv",
        "manifests/04_failed_pairs.csv",
    ]:
        print(" ", outdir / fn)


if __name__ == "__main__":
    main()
