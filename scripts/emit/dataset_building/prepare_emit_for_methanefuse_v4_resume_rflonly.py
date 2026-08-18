#!/usr/bin/env python3
# EMIT -> MethaneFuse external-dataset adapter v4 (resume + RFL-only).
#
# Current MethaneFuse uses EMIT L2A surface reflectance-derived 16-band
# WV3-simulated inputs. CH4ENH/CH4PLM are used here for selection, anchoring,
# exclusion, and QA rather than as direct model inputs.

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import earthaccess
    import numpy as np
    import pandas as pd
    import rasterio
    import xarray as xr
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Run:")
    print("  python3 -m pip install -U earthaccess numpy pandas xarray netCDF4 rasterio")
    sys.exit(1)

WV3_BANDS = [
    "Coastal (MS7)", "Blue (MS4)", "Green (MS3)", "Yellow (MS6)",
    "Red (MS2)", "Red Edge (MS5)", "NIR1 (MS1)", "NIR2 (MS8)",
    "SWIR1", "SWIR2", "SWIR3", "SWIR4", "SWIR5", "SWIR6", "SWIR7", "SWIR8",
]
SRF_URL = (
    "https://raw.githubusercontent.com/yuyao-wang/MethaneUnion/main/"
    "preprocess_dataset_EMIT/WV3_VNIR_SWIR_response.csv"
)

SCENE_RE = re.compile(r"EMIT_L2B_CH4ENH_002_(\d{8}T\d{6})_(\d+)_(\d{3})")
CORE_RE = re.compile(r"(\d{8}T\d{6})_(\d+)_(\d{3})")

QUERY_M = 480.0
EMIT_GSD_M = 60.0
NATIVE_QUERY_PX = int(round(QUERY_M / EMIT_GSD_M))   # 8
TARGET_SIZE = 518
CONTEXT_PX = 128
NEG_WINDOWS = (30, 90, 180, 365)


def recursive_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from recursive_strings(k)
            yield from recursive_strings(v)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from recursive_strings(x)


def extract_scene_ids(obj: Any) -> set[str]:
    out = set()
    for s in recursive_strings(obj):
        for m in SCENE_RE.finditer(s):
            out.add(f"EMIT_L2B_CH4ENH_002_{m.group(1)}_{m.group(2)}_{m.group(3)}")
    if out:
        return out
    for s in recursive_strings(obj):
        for m in CORE_RE.finditer(s):
            out.add(f"EMIT_L2B_CH4ENH_002_{m.group(1)}_{m.group(2)}_{m.group(3)}")
    return out


def scene_dt(scene_id: str) -> datetime:
    m = SCENE_RE.search(scene_id)
    if not m:
        raise ValueError(f"Cannot parse CH4ENH scene ID: {scene_id}")
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def scene_core(scene_id: str) -> tuple[str, str, str]:
    m = SCENE_RE.search(scene_id)
    if not m:
        raise ValueError(scene_id)
    return m.group(1), m.group(2), m.group(3)


def granule_ur(g: Any) -> str:
    try:
        umm = g.get("umm", {})
        for k in ("GranuleUR", "GranuleUr"):
            if k in umm:
                return str(umm[k])
    except Exception:
        pass
    for s in recursive_strings(g):
        if "EMIT" in s:
            return s
    return str(g)


def result_scene_id(g: Any) -> Optional[str]:
    found = extract_scene_ids(g)
    if found:
        return sorted(found)[0]
    try:
        for u in g.data_links():
            found = extract_scene_ids(u)
            if found:
                return sorted(found)[0]
    except Exception:
        pass
    return None


def flatten_geometry_coords(obj: Any) -> list[tuple[float, float]]:
    coords = []

    def walk(x: Any):
        if isinstance(x, (list, tuple)):
            if (
                len(x) >= 2
                and isinstance(x[0], (int, float))
                and isinstance(x[1], (int, float))
            ):
                lon, lat = float(x[0]), float(x[1])
                if -180 <= lon <= 180 and -90 <= lat <= 90:
                    coords.append((lon, lat))
            else:
                for y in x:
                    walk(y)
        elif isinstance(x, dict):
            if "coordinates" in x:
                walk(x["coordinates"])
            else:
                for v in x.values():
                    walk(v)

    if isinstance(obj, dict) and "geometry" in obj:
        walk(obj["geometry"])
    elif isinstance(obj, dict) and obj.get("type") == "FeatureCollection":
        for f in obj.get("features", []):
            if isinstance(f, dict) and "geometry" in f:
                walk(f["geometry"])
    else:
        walk(obj)
    return coords


def find_latlon_pairs(obj: Any, path: str = "") -> list[tuple[int, float, float, str]]:
    out = []
    if isinstance(obj, dict):
        lat_keys = [k for k in obj if "lat" in str(k).lower()]
        lon_keys = [k for k in obj if "lon" in str(k).lower()]
        for lk in lat_keys:
            for ok in lon_keys:
                try:
                    lat = float(obj[lk])
                    lon = float(obj[ok])
                except Exception:
                    continue
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue
                keytext = f"{path}/{lk}/{ok}".lower()
                score = 1
                if "max" in keytext:
                    score += 5
                if "enhance" in keytext or "plume" in keytext:
                    score += 3
                if "source" in keytext:
                    score += 2
                out.append((score, lon, lat, keytext))
        for k, v in obj.items():
            out.extend(find_latlon_pairs(v, f"{path}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(find_latlon_pairs(v, f"{path}/{i}"))
    return out


def plume_anchor_from_json(fp: Path) -> tuple[float, float, str]:
    obj = json.loads(fp.read_text(encoding="utf-8"))
    explicit = find_latlon_pairs(obj)
    if explicit:
        explicit.sort(key=lambda x: x[0], reverse=True)
        _, lon, lat, keytext = explicit[0]
        return lon, lat, f"explicit_latlon:{keytext}"

    coords = flatten_geometry_coords(obj)
    if coords:
        lons = np.asarray([p[0] for p in coords], dtype=float)
        lats = np.asarray([p[1] for p in coords], dtype=float)
        return float(np.median(lons)), float(np.median(lats)), "plume_geometry_median"

    raise ValueError(f"No plume coordinate found in {fp.name}")


def find_plm_json(label_dir: Path, meta_name: str) -> Optional[Path]:
    meta_name = meta_name.strip()
    candidates = [
        label_dir / f"{meta_name}.json",
        label_dir / f"{meta_name.replace('CH4PLM_002_', 'CH4PLMMETA_002_')}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    m = re.search(r"(\d{8}T\d{6})_(\d{6})", meta_name)
    if m:
        hits = list(label_dir.glob(f"*{m.group(1)}*{m.group(2)}*.json"))
        if hits:
            return sorted(hits)[0]
    return None


def build_plume_scene_exclusion(all_json_dir: Path) -> set[str]:
    scenes = set()
    for fp in all_json_dir.glob("*.json"):
        try:
            scenes.update(extract_scene_ids(json.loads(fp.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return scenes


def search_exact_ch4enh_at_point(scene_id: str, lon: float, lat: float):
    dt = scene_dt(scene_id)
    hits = earthaccess.search_data(
        short_name="EMITL2BCH4ENH",
        version="002",
        point=(lon, lat),
        temporal=((dt - timedelta(minutes=10)).isoformat(),
                  (dt + timedelta(minutes=10)).isoformat()),
        count=50,
    )
    for g in hits:
        if result_scene_id(g) == scene_id:
            return g
    return None


def rematch_negative(
    positive_scene_id: str,
    current_negative_scene_id: str,
    lon: float,
    lat: float,
    plume_scene_ids: set[str],
    used_negatives: set[str],
):
    g0 = search_exact_ch4enh_at_point(current_negative_scene_id, lon, lat)
    if (
        g0 is not None
        and current_negative_scene_id not in plume_scene_ids
        and current_negative_scene_id not in used_negatives
    ):
        return current_negative_scene_id, "retained_current_negative"

    pos_dt = scene_dt(positive_scene_id)
    for days in NEG_WINDOWS:
        hits = earthaccess.search_data(
            short_name="EMITL2BCH4ENH",
            version="002",
            point=(lon, lat),
            temporal=((pos_dt - timedelta(days=days)).isoformat(),
                      (pos_dt + timedelta(days=days)).isoformat()),
            count=500,
        )
        cand = []
        for g in hits:
            sid = result_scene_id(g)
            if not sid or sid == positive_scene_id:
                continue
            if sid in plume_scene_ids or sid in used_negatives:
                continue
            try:
                dt = scene_dt(sid)
            except Exception:
                continue
            delta = abs((dt - pos_dt).total_seconds()) / 86400.0
            if delta < 0.5:
                continue
            cand.append((delta, sid))
        if cand:
            cand.sort(key=lambda x: x[0])
            return cand[0][1], f"rematched_within_{days}d"
    return None, "no_negative_at_corrected_anchor"



def alternative_negative_candidates(
    positive_scene_id: str,
    lon: float,
    lat: float,
    plume_scene_ids: set[str],
    blocked_scene_ids: set[str],
):
    """
    Yield additional same-location candidate negatives ordered by temporal
    distance to the positive acquisition.

    This is used after a candidate fails model-input QA (e.g. EMIT L2A crop is
    all missing/zero).  It excludes:
      - the positive scene itself
      - every published CH4PLM source scene
      - already-used/failing negative scenes
    """
    pos_dt = scene_dt(positive_scene_id)
    best = {}

    for days in NEG_WINDOWS:
        hits = earthaccess.search_data(
            short_name="EMITL2BCH4ENH",
            version="002",
            point=(lon, lat),
            temporal=((pos_dt - timedelta(days=days)).isoformat(),
                      (pos_dt + timedelta(days=days)).isoformat()),
            count=500,
        )
        for g in hits:
            sid = result_scene_id(g)
            if not sid or sid == positive_scene_id:
                continue
            if sid in plume_scene_ids or sid in blocked_scene_ids:
                continue
            try:
                dt = scene_dt(sid)
            except Exception:
                continue

            delta = abs((dt - pos_dt).total_seconds()) / 86400.0
            if delta < 0.5:
                continue

            prev = best.get(sid)
            item = (delta, days, sid)
            if prev is None or item < prev:
                best[sid] = item

    return sorted(best.values(), key=lambda x: (x[0], x[1], x[2]))


def is_retryable_negative_error(reason: str) -> bool:
    """
    Errors that mean "this candidate is not a usable model input at the plume
    anchor", so we should try the next negative rather than fail the pair.
    """
    tokens = (
        "EMIT_missing_ratio:",
        "no_exact_EMITL2ARFL_t0",
        "anchor_far_from_L2A_swath:",
        "unexpected_native_crop_shape:",
        "array dimension",
        "write_read_shape_mismatch:",
    )
    return any(t in reason for t in tokens)



def ensure_srf(out_dir: Path) -> Path:
    fp = out_dir / "WV3_VNIR_SWIR_response.csv"
    if not fp.exists():
        print(f"Downloading MethaneUnion WV3 SRF -> {fp}")
        urllib.request.urlretrieve(SRF_URL, fp)
    return fp


def search_exact_l2a(scene_id: str, lon: float, lat: float):
    dt = scene_dt(scene_id)
    ts, orbit, scene = scene_core(scene_id)
    hits = earthaccess.search_data(
        short_name="EMITL2ARFL",
        point=(lon, lat),
        temporal=((dt - timedelta(minutes=20)).isoformat(),
                  (dt + timedelta(minutes=20)).isoformat()),
        count=50,
    )
    target_core = f"{ts}_{orbit}_{scene}"
    for g in hits:
        if target_core in granule_ur(g):
            return g

    best = None
    best_sec = float("inf")
    for g in hits:
        m = re.search(r"(\d{8}T\d{6})", granule_ur(g))
        if not m:
            continue
        gdt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        sec = abs((gdt - dt).total_seconds())
        if sec < best_sec:
            best, best_sec = g, sec
    return best if best_sec <= 20 * 60 else None


def granule_datetime(g: Any) -> Optional[datetime]:
    m = re.search(r"(\d{8}T\d{6})", granule_ur(g))
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def search_l2a_near_target(lon: float, lat: float, target: datetime, t0: datetime):
    for halfwin in (30, 60, 120):
        hits = earthaccess.search_data(
            short_name="EMITL2ARFL",
            point=(lon, lat),
            temporal=((target - timedelta(days=halfwin)).isoformat(),
                      (target + timedelta(days=halfwin)).isoformat()),
            count=500,
        )
        scored = []
        for g in hits:
            dt = granule_datetime(g)
            if dt is None or dt >= t0 - timedelta(hours=12):
                continue
            scored.append((abs((dt-target).total_seconds()), dt, g))
        if scored:
            scored.sort(key=lambda x: x[0])
            return scored[0][2], scored[0][1]
    return None, None


def find_downloaded_rfl(raw_dir: Path, g: Any) -> Optional[Path]:
    ur = granule_ur(g)
    cores = re.findall(r"\d{8}T\d{6}_\d+_\d{3}", ur)
    candidates = list(raw_dir.glob("*.nc"))
    good = [
        p for p in candidates
        if "RFL" in p.name.upper()
        and "UNCERT" not in p.name.upper()
        and "MASK" not in p.name.upper()
    ]
    if cores:
        exactish = [p for p in good if cores[0] in p.name]
        if exactish:
            return sorted(exactish)[0]
    return None


def rfl_data_url(g: Any) -> Optional[str]:
    """
    Return ONLY the EMIT L2A reflectance NetCDF URL.
    Excludes RFLUNCERT and MASK so we do not download the full 3-file granule.
    """
    try:
        links = list(g.data_links())
    except Exception:
        links = []

    candidates = []
    for u in links:
        if not isinstance(u, str):
            continue
        name = u.upper()
        if not name.startswith("HTTP"):
            continue
        if not re.search(r"\.NC(?:\?.*)?$", u, flags=re.I):
            continue
        if "RFLUNCERT" in name or "MASK" in name:
            continue
        # The wanted file is EMIT_L2A_RFL_....nc
        if "EMIT_L2A_RFL_" in name:
            candidates.append(u)

    if not candidates:
        # Conservative fallback: any .nc link containing RFL but not UNCERT/MASK.
        for u in links:
            if not isinstance(u, str):
                continue
            name = u.upper()
            if (
                u.startswith("http")
                and re.search(r"\.NC(?:\?.*)?$", u, flags=re.I)
                and "RFL" in name
                and "RFLUNCERT" not in name
                and "MASK" not in name
            ):
                candidates.append(u)

    return sorted(set(candidates))[0] if candidates else None


def download_l2a(g: Any, raw_dir: Path) -> Path:
    """
    Download only the L2A RFL NetCDF needed for WV3 simulation.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = find_downloaded_rfl(raw_dir, g)
    if existing:
        return existing

    url = rfl_data_url(g)
    if url is None:
        raise RuntimeError(f"no_RFL_data_url:{granule_ur(g)}")

    print("     downloading RFL only")
    paths = earthaccess.download(
        [url],
        local_path=str(raw_dir),
        provider="LPCLOUD",
        threads=1,
    )

    existing = find_downloaded_rfl(raw_dir, g)
    if existing:
        return existing

    for p in paths or []:
        pp = Path(str(p))
        if (
            pp.exists()
            and pp.suffix.lower() == ".nc"
            and "RFL" in pp.name.upper()
            and "UNCERT" not in pp.name.upper()
            and "MASK" not in pp.name.upper()
        ):
            return pp

    raise FileNotFoundError(f"Could not locate downloaded L2A RFL for {granule_ur(g)}")


def load_srf_matrix(srf_csv: Path, emit_waves: np.ndarray) -> np.ndarray:
    df = pd.read_csv(srf_csv)
    waves = df["nm/Band"].to_numpy(dtype=float)
    mat = np.zeros((len(emit_waves), 16), dtype=np.float32)
    for i, b in enumerate(WV3_BANDS):
        vals = df[b].to_numpy(dtype=float)
        w = np.interp(emit_waves, waves, vals, left=0.0, right=0.0)
        mat[:, i] = (w / (w.sum() + 1e-12)).astype(np.float32)
    return mat


def resize_chw_linear(img: np.ndarray, out_size: int) -> np.ndarray:
    c, h, w = img.shape
    if h == out_size and w == out_size:
        return img.astype(np.float32, copy=False)
    ys = np.linspace(0, h - 1, out_size)
    xs = np.linspace(0, w - 1, out_size)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    out = np.empty((c, out_size, out_size), dtype=np.float32)
    for i in range(c):
        band = img[i].astype(np.float32, copy=False)
        a = band[y0[:, None], x0[None, :]]
        b = band[y0[:, None], x1[None, :]]
        c0 = band[y1[:, None], x0[None, :]]
        d = band[y1[:, None], x1[None, :]]
        out[i] = (
            a * (1-wx) * (1-wy) + b * wx * (1-wy)
            + c0 * (1-wx) * wy + d * wx * wy
        )
    return out


def centered_bounds(center: int, size: int, n: int) -> tuple[int, int]:
    if n < size:
        raise RuntimeError(f"array dimension {n} smaller than requested crop {size}")
    start = center - size // 2
    start = max(0, min(start, n - size))
    return start, start + size


def make_wv3_query_crop(
    rfl_path: Path,
    lon: float,
    lat: float,
    srf_csv: Path,
    out_tif: Path,
) -> dict[str, Any]:
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(rfl_path, group="location", engine="netcdf4") as loc:
        lats = np.asarray(loc["lat"].values, dtype=np.float64)
        lons = np.asarray(loc["lon"].values, dtype=np.float64)

    d2 = (lats-lat)**2 + ((lons-lon)*math.cos(math.radians(lat)))**2
    d2[~np.isfinite(d2)] = np.inf
    flat = int(np.argmin(d2))
    py, px = np.unravel_index(flat, d2.shape)
    nearest_deg = float(math.sqrt(d2[py, px]))
    if not math.isfinite(nearest_deg) or nearest_deg > 0.02:
        raise RuntimeError(f"anchor_far_from_L2A_swath:{nearest_deg:.6f}deg")

    h, w = lats.shape
    ctx = min(CONTEXT_PX, h, w)
    y0, y1 = centered_bounds(py, ctx, h)
    x0, x1 = centered_bounds(px, ctx, w)

    with xr.open_dataset(rfl_path, engine="netcdf4") as ds:
        r = np.asarray(ds["reflectance"][y0:y1, x0:x1, :].values, dtype=np.float32)
    with xr.open_dataset(rfl_path, group="sensor_band_parameters", engine="netcdf4") as sb:
        emit_waves = np.asarray(sb["wavelengths"].values, dtype=np.float64)

    srf = load_srf_matrix(srf_csv, emit_waves)
    sim = np.matmul(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0), srf)

    scaled = np.zeros_like(sim, dtype=np.float32)
    for b in range(16):
        bd = sim[:, :, b]
        valid = bd > 0
        if np.any(valid):
            lo, hi = np.percentile(bd[valid], [1, 99])
            final = ((bd-lo)/(hi-lo+1e-6))*6000.0 + 8000.0
            final[~valid] = 0.0
            scaled[:, :, b] = np.clip(final, 0, 65535)

    cy, cx = py-y0, px-x0
    qy0, qy1 = centered_bounds(cy, NATIVE_QUERY_PX, scaled.shape[0])
    qx0, qx1 = centered_bounds(cx, NATIVE_QUERY_PX, scaled.shape[1])
    crop = scaled[qy0:qy1, qx0:qx1, :].transpose(2, 0, 1)

    if crop.shape != (16, NATIVE_QUERY_PX, NATIVE_QUERY_PX):
        raise RuntimeError(f"unexpected_native_crop_shape:{crop.shape}")

    missing_ratio = float(np.mean((~np.isfinite(crop[0])) | (crop[0] == 0)))
    if missing_ratio > 0.25:
        raise RuntimeError(f"EMIT_missing_ratio:{missing_ratio:.3f}")

    up = resize_chw_linear(crop, TARGET_SIZE)
    up = np.clip(np.nan_to_num(up, nan=0.0), 0, 65535).astype(np.uint16)

    with rasterio.open(
        out_tif, "w", driver="GTiff", height=TARGET_SIZE, width=TARGET_SIZE,
        count=16, dtype="uint16", compress="deflate", BIGTIFF="IF_SAFER",
    ) as dst:
        dst.write(up)

    with rasterio.open(out_tif) as chk:
        got = chk.read()
    if got.shape != (16, TARGET_SIZE, TARGET_SIZE):
        raise RuntimeError(f"write_read_shape_mismatch:{got.shape}")

    return {"native_missing_ratio": missing_ratio, "nearest_deg": nearest_deg}


@dataclass
class Sample:
    sample_id: str
    pair_id: str
    role: str
    label: int
    scene_id: str
    t0: datetime
    lon: float
    lat: float
    plume_json: str
    anchor_method: str
    negative_match_method: str = ""


def build_samples(root: Path) -> tuple[list[Sample], list[dict[str, Any]]]:
    pair_csv = root/"emit_v2_pairs.csv"
    label_dir = root/"03_positive_plm_labels"
    all_json = root/"00_all_plm_json"
    plume_scenes = build_plume_scene_exclusion(all_json)
    print(f"Published-plume source-scene exclusion set: {len(plume_scenes)}")

    rows = list(csv.DictReader(pair_csv.open(newline="", encoding="utf-8")))
    samples, audit, used_neg = [], [], set()

    for i, row in enumerate(rows, 1):
        pair_id = row["pair_id"]
        pos_sid = row["positive_scene_id"]
        old_neg_sid = row["negative_scene_id"]
        plm_names = [x.strip() for x in row["positive_plm_granules"].split(";") if x.strip()]
        if not plm_names:
            audit.append({"pair_id": pair_id, "status": "FAIL", "reason": "no_plm_name"})
            continue

        plm_fp = find_plm_json(label_dir, plm_names[0])
        if plm_fp is None:
            audit.append({"pair_id": pair_id, "status": "FAIL", "reason": "plm_json_missing"})
            continue

        try:
            lon, lat, anchor_method = plume_anchor_from_json(plm_fp)
        except Exception as e:
            audit.append({"pair_id": pair_id, "status": "FAIL", "reason": f"anchor:{e}"})
            continue

        neg_sid, neg_method = rematch_negative(
            pos_sid, old_neg_sid, lon, lat, plume_scenes, used_neg
        )
        if neg_sid is None:
            audit.append({
                "pair_id": pair_id, "status": "FAIL",
                "reason": "no_candidate_negative_at_plume_anchor",
                "anchor_lon": lon, "anchor_lat": lat,
            })
            continue
        used_neg.add(neg_sid)

        samples.extend([
            Sample(
                f"{pair_id}_POS", pair_id, "positive", 1, pos_sid, scene_dt(pos_sid),
                lon, lat, str(plm_fp), anchor_method
            ),
            Sample(
                f"{pair_id}_NEG", pair_id, "candidate_negative", 0, neg_sid, scene_dt(neg_sid),
                lon, lat, str(plm_fp), anchor_method, neg_method
            ),
        ])
        audit.append({
            "pair_id": pair_id, "status": "PASS",
            "positive_scene_id": pos_sid,
            "old_negative_scene_id": old_neg_sid,
            "final_negative_scene_id": neg_sid,
            "negative_changed": int(neg_sid != old_neg_sid),
            "negative_match_method": neg_method,
            "anchor_lon": lon, "anchor_lat": lat,
            "anchor_method": anchor_method,
            "plume_json": str(plm_fp),
            "all_positive_plm_granules": ";".join(plm_names),
        })
        print(
            f"[pair {i:02d}/{len(rows)}] {pair_id} "
            f"anchor=({lat:.5f},{lon:.5f}) neg={neg_method}"
        )
    return samples, audit


def relpath(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def process_sample(s: Sample, out: Path, raw_dir: Path, srf_csv: Path, mode: str):
    sd = out/"samples"/s.sample_id
    sd.mkdir(parents=True, exist_ok=True)
    p0 = sd/"emit_t0.tif"

    g0 = search_exact_l2a(s.scene_id, s.lon, s.lat)
    if g0 is None:
        raise RuntimeError("no_exact_EMITL2ARFL_t0")
    nc0 = download_l2a(g0, raw_dir)
    qa0 = {"native_missing_ratio": ""}
    if not p0.exists():
        qa0 = make_wv3_query_crop(nc0, s.lon, s.lat, srf_csv, p0)

    t0_rfl_dt = granule_datetime(g0) or s.t0

    if mode == "smoke":
        p90 = p180 = p0
        t90_dt = t180_dt = t0_rfl_dt
        g90_name = g180_name = "SMOKE_REPEAT_T0"
    else:
        g90, t90_dt = search_l2a_near_target(
            s.lon, s.lat, s.t0-timedelta(days=90), s.t0
        )
        g180, t180_dt = search_l2a_near_target(
            s.lon, s.lat, s.t0-timedelta(days=180), s.t0
        )
        if g90 is None:
            raise RuntimeError("no_EMITL2ARFL_near_tminus90")
        if g180 is None:
            raise RuntimeError("no_EMITL2ARFL_near_tminus180")

        p90 = sd/"emit_tminus90.tif"
        p180 = sd/"emit_tminus180.tif"
        if not p90.exists():
            make_wv3_query_crop(download_l2a(g90, raw_dir), s.lon, s.lat, srf_csv, p90)
        if not p180.exists():
            make_wv3_query_crop(download_l2a(g180, raw_dir), s.lon, s.lat, srf_csv, p180)
        g90_name = granule_ur(g90)
        g180_name = granule_ur(g180)

    def days_before(dt):
        return abs((s.t0-dt).total_seconds())/86400.0 if dt else float("nan")

    return {
        "id": s.sample_id,
        "label": s.label,
        "latitude": f"{s.lat:.8f}",
        "longitude": f"{s.lon:.8f}",
        "emit_0_path": relpath(p0, out),
        "emit_90_path": relpath(p90, out),
        # MethaneUnion EMIT uses t0/-90/-180, while the current generic
        # MethaneFuse wide-table loader requires a *_360_path column.
        "emit_360_path": relpath(p180, out),
        "pair_id": s.pair_id,
        "scene_role": s.role,
        "source_ch4enh_scene_id": s.scene_id,
        "source_t0_utc": s.t0.isoformat(),
        "plume_anchor_json": Path(s.plume_json).name,
        "anchor_method": s.anchor_method,
        "negative_match_method": s.negative_match_method,
        "input_product": "EMITL2ARFL",
        "wv3_band_count": 16,
        "query_scale_m": 480,
        "native_emit_query_px": NATIVE_QUERY_PX,
        "output_size_px": TARGET_SIZE,
        "temporal_mode": mode,
        "actual_t90_days_before_t0": f"{days_before(t90_dt):.3f}",
        "actual_third_days_before_t0": f"{days_before(t180_dt):.3f}",
        "third_frame_semantics": (
            "t0_repeated" if mode == "smoke"
            else "EMIT_tminus180_aliased_to_emit_360_path"
        ),
        "t0_l2a_granule": granule_ur(g0),
        "t90_l2a_granule": g90_name,
        "third_l2a_granule": g180_name,
        "t0_native_missing_ratio": qa0.get("native_missing_ratio", ""),
        "negative_label_strength": (
            "" if s.label == 1
            else "candidate_negative_no_published_CH4PLM_source_scene"
        ),
    }



def cleanup_raw_l2a_cache(raw_dir: Path) -> tuple[int, int]:
    """Delete raw EMIT L2A cache files; model-ready TIFFs are kept."""
    if not raw_dir.exists():
        return 0, 0
    n = 0
    b = 0
    for p in sorted(raw_dir.rglob("*"), reverse=True):
        try:
            if p.is_file():
                try:
                    b += p.stat().st_size
                except OSError:
                    pass
                p.unlink()
                n += 1
            elif p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        except OSError:
            pass
    raw_dir.mkdir(parents=True, exist_ok=True)
    return n, b


def write_progress_csvs(
    out: Path,
    ok_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> None:
    """Write small recovery snapshots after each processed sample."""
    write_csv(out/"eval_emit_480m.partial.csv", ok_rows)
    write_csv(out/"qa_report.partial.csv", qa_rows)
    write_csv(out/"pair_anchor_audit.partial.csv", audit_rows)




def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_resume_state(out: Path):
    """
    Recover rows written by v3/v4 after each completed sample.

    PASS samples are never re-downloaded. Previous FAIL rows are retried.
    """
    eval_partial = out/"eval_emit_480m.partial.csv"
    qa_partial = out/"qa_report.partial.csv"

    ok = read_csv_rows(eval_partial)
    qa_all = read_csv_rows(qa_partial)

    pass_ids = {
        str(r.get("id", "")).strip()
        for r in qa_all
        if str(r.get("status", "")).upper() == "PASS"
    }

    # Keep only PASS QA rows. Previous failures are allowed another attempt.
    qa_pass = [
        r for r in qa_all
        if str(r.get("status", "")).upper() == "PASS"
    ]

    # Keep only eval rows whose sample really has PASS state and whose files exist.
    valid_ok = []
    valid_ids = set()
    for r in ok:
        sid = str(r.get("id", "")).strip()
        if sid not in pass_ids:
            continue

        paths_ok = True
        for c in ("emit_0_path", "emit_90_path", "emit_360_path"):
            v = str(r.get(c, "")).strip()
            if not v:
                paths_ok = False
                break
            p = Path(v)
            if not p.is_absolute():
                p = out/p
            if not p.exists():
                paths_ok = False
                break
            try:
                with rasterio.open(p) as ds:
                    if (ds.count, ds.height, ds.width) != (16, 518, 518):
                        paths_ok = False
                        break
            except Exception:
                paths_ok = False
                break

        if paths_ok:
            valid_ok.append(r)
            valid_ids.add(sid)

    qa_pass = [r for r in qa_pass if str(r.get("id", "")).strip() in valid_ids]

    return valid_ok, qa_pass, valid_ids



def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    for r in rows[1:]:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_rebaser(out: Path):
    code = r"""#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--dataset-root", required=True)
p.add_argument("--input", default="eval_emit_480m.csv")
p.add_argument("--output", default="eval_emit_480m_abs.csv")
a=p.parse_args()
root=Path(a.dataset_root).expanduser().resolve()
rows=list(csv.DictReader((root/a.input).open(newline="", encoding="utf-8")))
if not rows:
    raise RuntimeError("Input CSV has no rows")
for r in rows:
    for c in ("emit_0_path","emit_90_path","emit_360_path"):
        v=str(r.get(c,"")).strip()
        if v and not Path(v).is_absolute():
            r[c]=str((root/v).resolve())
with (root/a.output).open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(root/a.output)
"""
    fp = out/"make_paths_absolute.py"
    fp.write_text(code, encoding="utf-8")
    fp.chmod(0o755)


def write_readme(out: Path, mode: str, n_ok: int, n_fail: int):
    txt = f"""# EMIT external evaluation package for MethaneFuse

Mode: {mode}
Prepared samples: {n_ok}
Failed/skipped: {n_fail}
Query footprint: 480 m
EMIT native GSD: 60 m
Native query crop: 8 x 8 pixels
Model TIFF shape: 16 x 518 x 518
TIFF dtype: uint16

## Model input
EMITL2ARFL surface reflectance
-> 16 WorldView-3 SRF-simulated bands
-> 480 m crop
-> 518 x 518 TIFF

Band order:
{", ".join(WV3_BANDS)}

## Labels
Positive = published EMIT CH4PLM V2 plume anchor.
Candidate negative = same location on another CH4ENH acquisition excluded from
all published CH4PLM source-scene references available in this local V2 audit.
Candidate negatives are weak negatives, not proof of methane-free conditions.
If a candidate has invalid/missing L2A pixels at the plume anchor, this adapter
automatically tries the next same-location non-PLM candidate and records the retry.

## Temporal warning
smoke mode repeats the real t0 image in all three temporal path columns. It is
ONLY an inference plumbing test and must not be reported as a final temporal
benchmark.

temporal mode uses EMIT t0 / approximately t-90 / approximately t-180. The
current MethaneFuse generic wide-table loader requires the third column to be
named emit_360_path, so the actual EMIT t-180 image is intentionally stored
there. The true offset is recorded in actual_third_days_before_t0.

## After transferring this folder
Run:
  python make_paths_absolute.py --dataset-root /ABS/PATH/TO/{out.name}

Then from the MethaneFuse repo root:
  python scripts/eval/evaluate_classification.py \\
    --eval_csv /ABS/PATH/TO/{out.name}/eval_emit_480m_abs.csv \\
    --checkpoint checkpoints/stage2_classification_480m.pt \\
    --stage b \\
    --batch_size 16 \\
    --num_workers 8 \\
    --row_fusion_mode max \\
    --wv3_srf_csv /ABS/PATH/TO/{out.name}/WV3_VNIR_SWIR_response.csv \\
    --output_json results/eval/emit_external_480m.json

For a loader/inference check first, add:
  --max_eval_steps 1

## Send these
samples/
eval_emit_480m.csv
WV3_VNIR_SWIR_response.csv
make_paths_absolute.py
README_FOR_SENIOR.md
qa_report.csv
pair_anchor_audit.csv

raw_l2a/ is not needed by MethaneFuse after preprocessing, so it can be omitted
from the handoff archive to save transfer size.
"""
    (out/"README_FOR_SENIOR.md").write_text(txt, encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", default="emit_v2_posneg_100")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=("smoke","temporal"), default="smoke")
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument(
        "--cleanup-raw-l2a",
        action="store_true",
        help="Delete raw EMIT L2A cache after each sample conversion.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from eval/qa *.partial.csv and skip already-PASS samples.",
    )
    a=ap.parse_args()

    root=Path(a.root).expanduser().resolve()
    out=Path(a.out or f"emit_methanefuse_480m_{a.mode}").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw_dir=out/"raw_l2a"
    srf_csv=ensure_srf(out)

    print("="*78)
    print("EMIT -> MethaneFuse adapter")
    print("Input :",root)
    print("Output:",out)
    print("Mode  :",a.mode)
    print("="*78)

    print("\n[1/6] Earthdata login")
    try:
        earthaccess.login(strategy="netrc")
        print("Authenticated using ~/.netrc")
    except Exception:
        earthaccess.login(strategy="interactive", persist=False)
        print("Authenticated interactively for this run")

    print("\n[2/6] Correct plume anchors + verify/rematch negatives")
    samples,audit=build_samples(root)
    if a.limit_pairs is not None:
        pair_ids=sorted({s.pair_id for s in samples})[:a.limit_pairs]
        keep=set(pair_ids)
        samples=[s for s in samples if s.pair_id in keep]
        audit=[r for r in audit if r.get("pair_id") in keep]
    write_csv(out/"pair_anchor_audit.csv",audit)

    print(f"\n[3/6] Process {len(samples)} samples")

    if a.resume:
        ok, qa, completed_ids = load_resume_state(out)
        print(
            f"Resume state: {len(completed_ids)} already-PASS samples "
            f"({len(ok)} valid eval rows)"
        )
    else:
        ok, qa, completed_ids = [], [], set()

    # Needed for automatic replacement of model-invalid candidate negatives.
    plume_scene_ids = build_plume_scene_exclusion(root/"00_all_plm_json")
    pair_positive_scene = {
        s.pair_id: s.scene_id for s in samples if int(s.label) == 1
    }
    audit_by_pair = {r.get("pair_id"): r for r in audit}
    used_final_negatives = {
        str(r.get("source_ch4enh_scene_id", "")).strip()
        for r in ok
        if str(r.get("label", "")).strip() == "0"
        and str(r.get("source_ch4enh_scene_id", "")).strip()
    }

    # Restore model-QA final negatives into the freshly recomputed pair audit.
    for r in ok:
        if str(r.get("label", "")).strip() != "0":
            continue
        pair_id = str(r.get("pair_id", "")).strip()
        sid = str(r.get("source_ch4enh_scene_id", "")).strip()
        ar = audit_by_pair.get(pair_id)
        if ar is not None and sid:
            ar["final_negative_scene_id"] = sid
            ar["resumed_final_negative_scene_id"] = sid

    for i,s in enumerate(samples,1):
        if s.sample_id in completed_ids:
            print(f"[{i:03d}/{len(samples):03d}] {s.sample_id} -> RESUME SKIP PASS")
            continue

        print(f"[{i:03d}/{len(samples):03d}] {s.sample_id}")

        initial_scene_id = s.scene_id
        retry_count = 0
        retry_notes = []
        success = False
        final_row = None
        final_sample = s

        # If this is a negative, don't allow a scene already accepted for a
        # previous pair.  Positive rows are processed exactly once.
        initial_conflict = (
            int(s.label) == 0 and s.scene_id in used_final_negatives
        )

        if initial_conflict:
            initial_error = "duplicate_negative_scene"
            print(f"  -> RETRY: {initial_error}")
        else:
            try:
                final_row = process_sample(s,out,raw_dir,srf_csv,a.mode)
                success = True
            except Exception as e:
                initial_error = str(e)
                print(f"  -> initial candidate failed: {initial_error}")

        # Negative-only automatic retry.
        if (
            not success
            and int(s.label) == 0
            and (initial_conflict or is_retryable_negative_error(initial_error))
        ):
            pos_sid = pair_positive_scene.get(s.pair_id)
            blocked = set(used_final_negatives)
            blocked.add(initial_scene_id)

            alternatives = alternative_negative_candidates(
                positive_scene_id=pos_sid,
                lon=s.lon,
                lat=s.lat,
                plume_scene_ids=plume_scene_ids,
                blocked_scene_ids=blocked,
            )

            print(f"  -> {len(alternatives)} alternate negative candidates available")

            for delta_days, search_window, alt_sid in alternatives:
                retry_count += 1
                retry_notes.append(
                    f"retry{retry_count}:{alt_sid}:delta={delta_days:.3f}d"
                )

                # Remove any partial/stale model TIFF for this sample ID before
                # trying a different acquisition.
                shutil.rmtree(out/"samples"/s.sample_id, ignore_errors=True)

                alt = Sample(
                    sample_id=s.sample_id,
                    pair_id=s.pair_id,
                    role=s.role,
                    label=s.label,
                    scene_id=alt_sid,
                    t0=scene_dt(alt_sid),
                    lon=s.lon,
                    lat=s.lat,
                    plume_json=s.plume_json,
                    anchor_method=s.anchor_method,
                    negative_match_method=(
                        f"model_QA_retry_{retry_count}_within_{search_window}d"
                    ),
                )

                print(
                    f"  -> retry {retry_count}: {alt_sid} "
                    f"(Δ={delta_days:.1f} d)"
                )

                try:
                    row = process_sample(alt,out,raw_dir,srf_csv,a.mode)
                    final_row = row
                    final_sample = alt
                    success = True
                    print("  -> RETRY PASS")
                    break
                except Exception as e:
                    reason = str(e)
                    retry_notes.append(f"retry{retry_count}_fail:{reason}")
                    blocked.add(alt_sid)
                    print(f"     retry failed: {reason}")
                    # Keep trying on model-input/coverage QA failures.
                    if not is_retryable_negative_error(reason):
                        break

        if success:
            ok.append(final_row)
            if int(final_sample.label) == 0:
                used_final_negatives.add(final_sample.scene_id)

                # Update the pair audit so the handoff records the model-ready
                # negative actually used, not merely the first CH4ENH candidate.
                ar = audit_by_pair.get(final_sample.pair_id)
                if ar is not None:
                    ar["pre_model_qa_negative_scene_id"] = initial_scene_id
                    ar["final_negative_scene_id"] = final_sample.scene_id
                    ar["negative_changed_by_model_qa"] = int(
                        final_sample.scene_id != initial_scene_id
                    )
                    ar["negative_match_method"] = final_sample.negative_match_method
                    ar["model_qa_retry_count"] = retry_count

            qa.append({
                "id":final_sample.sample_id,
                "pair_id":final_sample.pair_id,
                "label":final_sample.label,
                "status":"PASS",
                "reason":"",
                "initial_scene_id":initial_scene_id,
                "final_scene_id":final_sample.scene_id,
                "retry_count":retry_count,
                "retry_notes":";".join(retry_notes),
            })

            if a.cleanup_raw_l2a:
                removed_n, removed_b = cleanup_raw_l2a_cache(raw_dir)
                if removed_n:
                    print(
                        f"  -> cleaned raw_l2a cache: {removed_n} files, "
                        f"{removed_b/1024**3:.2f} GiB"
                    )

            try:
                write_progress_csvs(out, ok, qa, audit)
            except OSError as progress_err:
                print(f"  -> WARNING: progress CSV write failed: {progress_err}")

            print("  -> PASS")
        else:
            # Clean any partial output that must not be handed to the model.
            shutil.rmtree(out/"samples"/s.sample_id, ignore_errors=True)
            final_reason = (
                retry_notes[-1] if retry_notes
                else initial_error
            )
            qa.append({
                "id":s.sample_id,
                "pair_id":s.pair_id,
                "label":s.label,
                "status":"FAIL",
                "reason":final_reason,
                "initial_scene_id":initial_scene_id,
                "final_scene_id":"",
                "retry_count":retry_count,
                "retry_notes":";".join(retry_notes),
            })

            if a.cleanup_raw_l2a:
                removed_n, removed_b = cleanup_raw_l2a_cache(raw_dir)
                if removed_n:
                    print(
                        f"  -> cleaned raw_l2a cache after fail: {removed_n} files, "
                        f"{removed_b/1024**3:.2f} GiB"
                    )

            try:
                write_progress_csvs(out, ok, qa, audit)
            except OSError as progress_err:
                print(f"  -> WARNING: progress CSV write failed: {progress_err}")

            print("  -> FINAL FAIL:", final_reason)

    # Re-write audit after any model-QA-driven negative replacements.
    write_csv(out/"pair_anchor_audit.csv",audit)

    print("\n[4/6] Write manifests")
    write_csv(out/"eval_emit_480m.csv",ok)
    write_csv(out/"qa_report.csv",qa)
    write_rebaser(out)

    print("\n[5/6] Integrity check")
    path_errors=shape_errors=0
    for r in ok:
        for c in ("emit_0_path","emit_90_path","emit_360_path"):
            p=out/r[c]
            if not p.exists():
                path_errors+=1
                continue
            with rasterio.open(p) as ds:
                if (ds.count,ds.height,ds.width)!=(16,518,518):
                    shape_errors+=1
    pos=sum(int(r["label"])==1 for r in ok)
    neg=sum(int(r["label"])==0 for r in ok)
    pairs_pos={r["pair_id"] for r in ok if int(r["label"])==1}
    pairs_neg={r["pair_id"] for r in ok if int(r["label"])==0}
    complete_pairs=len(pairs_pos & pairs_neg)
    print(f"Prepared rows: {len(ok)} positive={pos} negative={neg}")
    print(f"Complete pos/neg pairs: {complete_pairs}")
    print("Path errors :",path_errors)
    print("Shape errors:",shape_errors)

    print("\n[6/6] Handoff README")
    write_readme(out,a.mode,len(ok),len(qa)-len(ok))

    print("\nDONE")
    print("Dataset :",out)
    print("Eval CSV:",out/"eval_emit_480m.csv")
    if a.cleanup_raw_l2a:
        print("Raw L2A cache cleanup: ENABLED")
    print("L2A download mode: RFL ONLY")
    if a.resume:
        print("Resume mode: ENABLED")
    if a.mode=="smoke":
        print("WARNING: smoke mode repeats t0 in all three temporal slots.")
    else:
        print("NOTE: temporal mode aliases real EMIT t-180 into emit_360_path.")


if __name__=="__main__":
    main()
