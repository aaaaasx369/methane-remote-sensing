#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
refresh_methanesat_l4point_and_download_missing.py

Refresh the LIVE MethaneSAT L4 point-source catalogue from Google Earth Engine,
compare it against the user's existing 111 positive rows, and download ONLY
clearly missing positive L3 XCH4 crops.

Design goals
------------
- Uses the live official MethaneSAT L4 point-source FeatureCollection.
- Compares current L4 features to old positives primarily by:
    1) same collection_id + plume_id, when old plume_id is usable;
    2) same collection_id + very-near coordinates.
- Keeps ambiguous same-scene matches separate instead of silently calling them new.
- Downloads only CLEAR_NEW_FEATURE rows by default.
- Resume-safe:
    * valid existing TIFF -> SKIP
    * download into .part
    * rasterio validation
    * fsync
    * atomic rename
    * local JSONL checkpoint with fsync
- Never deletes previously downloaded data.
- Preserves L4 provenance: collection_id, plume_id, flux, flux_sd, target_id, date.

Official assets
---------------
L4 point:
  projects/edf-methanesat-ee/assets/public-preview/L4point

L3 XCH4:
  projects/edf-methanesat-ee/assets/public-preview/L3concentration
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


L4_ASSET = "projects/edf-methanesat-ee/assets/public-preview/L4point"
L3_ASSET = "projects/edf-methanesat-ee/assets/public-preview/L3concentration"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--old-inventory",
        default="~/Downloads/MethaneSAT_222_inventory.csv",
        help="Existing 222-row inventory; label=1 rows are the old positive set.",
    )
    p.add_argument(
        "--out",
        default="/Volumes/engg-leung/dora lin/MethaneSAT_MethaneFuse/04_l4point_refresh",
        help="Network/lab output directory.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default="~/methane_release_project/methanesat_l4point_refresh_checkpoints",
        help="LOCAL checkpoint directory; keep off SMB.",
    )
    p.add_argument(
        "--project",
        default="methane-release-gee",
        help="Earth Engine Cloud project.",
    )
    p.add_argument(
        "--crop-half-m",
        type=float,
        default=240.0,
        help="240 m => 480 m x 480 m crop around the L4 point.",
    )
    p.add_argument(
        "--scale-m",
        type=float,
        default=45.0,
        help="L3 XCH4 download scale.",
    )
    p.add_argument(
        "--coordinate-match-m",
        type=float,
        default=150.0,
        help="Same collection + distance <= this is considered already represented.",
    )
    p.add_argument(
        "--ambiguous-match-m",
        type=float,
        default=1000.0,
        help="Same collection + distance in (coordinate-match, this] is flagged ambiguous.",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=5,
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=180,
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.15,
    )
    p.add_argument(
        "--catalog-only",
        action="store_true",
        help="Create current catalogue + exact diff only; do not download missing positive crops.",
    )
    p.add_argument(
        "--include-ambiguous",
        action="store_true",
        help="Also download AMBIGUOUS_SAME_COLLECTION rows. Default is safer: clear-new only.",
    )
    p.add_argument(
        "--restart-checkpoint",
        action="store_true",
        help="Clear only local status log. Existing TIFFs remain and still validate/skip.",
    )
    return p.parse_args()


def initialize_ee(project: str):
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed.\n"
            "Run: earthengine authenticate\n"
            f"Then rerun with --project {project!r}\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def norm_collection(v: Any) -> str:
    s = str(v if v is not None else "").strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    if s[:1].lower() == "c":
        s = s[1:]
    return s.upper()


def norm_plume(v: Any) -> str:
    s = str(v if v is not None else "").strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    # Old inventory sometimes serialized integer-like IDs as "6.0".
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return s


def parse_notes_value(notes: str, key: str) -> str:
    m = re.search(
        rf"\b{re.escape(key)}\s*=\s*([^;,\s]+)",
        str(notes),
        flags=re.I,
    )
    return m.group(1).strip() if m else ""


def haversine_m(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return np.nan
    r = 6371008.8
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_old_positives(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Old inventory not found: {path}")

    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}

    required = ["latitude", "longitude", "label", "notes"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(
            f"Old inventory missing columns {missing}. Columns={list(df.columns)}"
        )

    x = pd.DataFrame()
    x["old_row_index"] = np.arange(len(df))
    x["latitude"] = pd.to_numeric(df[cols["latitude"]], errors="coerce")
    x["longitude"] = pd.to_numeric(df[cols["longitude"]], errors="coerce")
    x["label"] = pd.to_numeric(df[cols["label"]], errors="coerce")
    x["notes"] = df[cols["notes"]].fillna("").astype(str)

    sid_col = cols.get("scene/observation id")
    date_col = cols.get("date")
    time_col = cols.get("utc time")

    x["old_sample_id"] = (
        df[sid_col].fillna("").astype(str)
        if sid_col is not None
        else [f"old_{i:06d}" for i in range(len(df))]
    )
    x["old_date"] = (
        df[date_col].fillna("").astype(str)
        if date_col is not None else ""
    )
    x["old_utc_time"] = (
        df[time_col].fillna("").astype(str)
        if time_col is not None else ""
    )

    x["collection_id"] = x["notes"].map(
        lambda s: norm_collection(parse_notes_value(s, "collection_id"))
    )
    x["plume_id"] = x["notes"].map(
        lambda s: norm_plume(parse_notes_value(s, "plume_id"))
    )

    x = x[
        x["label"].eq(1)
        & x["latitude"].notna()
        & x["longitude"].notna()
        & x["collection_id"].str.len().gt(0)
    ].copy()

    # Remove literal duplicate rows only.
    x = x.drop_duplicates(
        subset=["old_sample_id", "collection_id", "latitude", "longitude"]
    ).reset_index(drop=True)
    return x


def fetch_live_l4_catalogue() -> pd.DataFrame:
    fc = ee.FeatureCollection(L4_ASSET)
    n = int(fc.size().getInfo())
    info = fc.getInfo()

    rows = []
    for i, f in enumerate(info.get("features", [])):
        p = f.get("properties", {}) or {}
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None]

        lon = coords[0] if len(coords) > 0 else None
        lat = coords[1] if len(coords) > 1 else None

        rows.append({
            "live_index": i,
            "collection_id": norm_collection(p.get("collection_id")),
            "date": p.get("date"),
            "flux": p.get("flux"),
            "flux_sd": p.get("flux_sd"),
            "processing_id": p.get("processing_id"),
            "plume_id": norm_plume(p.get("plume_id")),
            "plume_id_in_scene": p.get("plume_id_in_scene"),
            "region": p.get("region"),
            "target_id": p.get("target_id"),
            "latitude": pd.to_numeric(lat, errors="coerce"),
            "longitude": pd.to_numeric(lon, errors="coerce"),
        })

    out = pd.DataFrame(rows)
    if len(out) != n:
        print(f"[WARN] fc.size={n}, features returned={len(out)}")
    return out


def compare_live_to_old(live: pd.DataFrame, old: pd.DataFrame,
                        coord_match_m: float,
                        ambiguous_match_m: float) -> pd.DataFrame:
    old_by_collection = {
        cid: g.copy()
        for cid, g in old.groupby("collection_id")
    }

    rows = []
    for _, r in live.iterrows():
        rec = r.to_dict()
        cid = r["collection_id"]
        plume = r["plume_id"]

        g = old_by_collection.get(cid, pd.DataFrame())

        rec.update({
            "old_match_status": "",
            "old_match_method": "",
            "old_match_sample_id": "",
            "old_match_distance_m": np.nan,
            "old_match_plume_id": "",
            "old_same_collection_count": int(len(g)),
        })

        if g.empty:
            rec["old_match_status"] = "CLEAR_NEW_FEATURE"
            rec["old_match_method"] = "collection_not_in_old_positive_set"
            rows.append(rec)
            continue

        # Method 1: same collection + exact normalized plume_id, when plume_id exists.
        if plume:
            exact = g[g["plume_id"].eq(plume)].copy()
            if len(exact) == 1:
                rr = exact.iloc[0]
                d = haversine_m(
                    r["latitude"], r["longitude"],
                    rr["latitude"], rr["longitude"]
                )
                rec["old_match_status"] = "ALREADY_HAVE"
                rec["old_match_method"] = "same_collection_exact_plume_id"
                rec["old_match_sample_id"] = rr["old_sample_id"]
                rec["old_match_distance_m"] = d
                rec["old_match_plume_id"] = rr["plume_id"]
                rows.append(rec)
                continue

        # Method 2: nearest old positive in the same collection.
        distances = np.array([
            haversine_m(
                r["latitude"], r["longitude"],
                rr["latitude"], rr["longitude"]
            )
            for _, rr in g.iterrows()
        ], dtype=float)

        finite_mask = np.isfinite(distances)
        if not finite_mask.any():
            rec["old_match_status"] = "AMBIGUOUS_SAME_COLLECTION"
            rec["old_match_method"] = "same_collection_but_no_usable_old_coordinates"
            rows.append(rec)
            continue

        jpos = int(np.nanargmin(distances))
        rr = g.iloc[jpos]
        d = float(distances[jpos])

        rec["old_match_sample_id"] = rr["old_sample_id"]
        rec["old_match_distance_m"] = d
        rec["old_match_plume_id"] = rr["plume_id"]

        if d <= coord_match_m:
            rec["old_match_status"] = "ALREADY_HAVE"
            rec["old_match_method"] = f"same_collection_coordinate_within_{coord_match_m:.0f}m"
        elif d <= ambiguous_match_m:
            rec["old_match_status"] = "AMBIGUOUS_SAME_COLLECTION"
            rec["old_match_method"] = (
                f"same_collection_nearest_old_{d:.1f}m;"
                "could_be_revised_or_distinct_nearby_plume"
            )
        else:
            rec["old_match_status"] = "CLEAR_NEW_FEATURE"
            rec["old_match_method"] = (
                f"same_collection_but_nearest_old_{d:.1f}m"
            )

        rows.append(rec)

    return pd.DataFrame(rows)


def ensure_output_mount(outdir: Path):
    if str(outdir).startswith("/Volumes/"):
        parts = outdir.parts
        mount_root = Path("/", "Volumes", parts[2])
        if not mount_root.exists():
            raise RuntimeError(
                f"NETWORK SHARE DISCONNECTED: {mount_root}\n"
                "Reconnect it and rerun the SAME command. Local checkpoints remain safe."
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
            f"OUTPUT NOT WRITABLE: {outdir}\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc


def atomic_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl_fsync(path: Path, rec: dict):
    safe = {}
    for k, v in rec.items():
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = None if np.isnan(v) else float(v)
        elif isinstance(v, (np.bool_,)):
            v = bool(v)
        else:
            try:
                if pd.isna(v):
                    v = None
            except Exception:
                pass
        safe[str(k)] = v

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def stable_feature_id(row: pd.Series) -> str:
    parts = [
        norm_collection(row.get("collection_id")),
        norm_plume(row.get("plume_id")),
        str(row.get("plume_id_in_scene", "")),
        f"{float(row.get('latitude')):.8f}",
        f"{float(row.get('longitude')):.8f}",
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def sanitize(v):
    s = str(v if v is not None else "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("._-") or "unknown"


def output_filename(row: pd.Series) -> str:
    cid = sanitize(norm_collection(row["collection_id"]))
    plume = sanitize(norm_plume(row.get("plume_id")) or row.get("plume_id_in_scene", "plume"))
    fid = stable_feature_id(row)
    return f"MSAT_L4REFRESH__c{cid}__p{plume}__{fid}.tif"


def validate_tiff(path: Path) -> tuple[bool, dict]:
    meta = {
        "file_bytes": path.stat().st_size if path.exists() else 0,
        "width": None,
        "height": None,
        "bands": None,
        "dtype": None,
        "crs": None,
        "valid_pixel_fraction": None,
        "xch4_min": None,
        "xch4_median": None,
        "xch4_max": None,
    }
    if not path.exists() or path.stat().st_size <= 0:
        return False, meta

    try:
        with rasterio.open(path) as ds:
            meta["width"] = ds.width
            meta["height"] = ds.height
            meta["bands"] = ds.count
            meta["dtype"] = ds.dtypes[0] if ds.count else None
            meta["crs"] = str(ds.crs) if ds.crs else ""

            if ds.count < 1 or ds.width < 2 or ds.height < 2:
                return False, meta

            a = ds.read(1, masked=True).astype("float64")
            vals = a.compressed()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                meta["valid_pixel_fraction"] = 0.0
                return False, meta

            meta["valid_pixel_fraction"] = float(vals.size / a.size)
            meta["xch4_min"] = float(np.min(vals))
            meta["xch4_median"] = float(np.median(vals))
            meta["xch4_max"] = float(np.max(vals))
            return True, meta

    except Exception:
        return False, meta


def get_l3_image(collection_id: str) -> tuple[ee.Image, str]:
    cid = norm_collection(collection_id)
    variants = [f"c{cid}", cid]
    base = ee.ImageCollection(L3_ASSET)

    for candidate in variants:
        ic = base.filter(ee.Filter.eq("collection_id", candidate))
        n = int(ic.size().getInfo())
        if n > 0:
            return ee.Image(ic.first()), candidate

    raise RuntimeError(
        f"No L3 image found for collection_id={collection_id!r}; tried {variants}"
    )


def make_download_url(row: pd.Series, half_m: float, scale_m: float):
    img, l3_cid = get_l3_image(row["collection_id"])
    point = ee.Geometry.Point([
        float(row["longitude"]),
        float(row["latitude"]),
    ])
    region = point.buffer(half_m).bounds()

    url = img.select("XCH4").getDownloadURL({
        "region": region,
        "scale": scale_m,
        "format": "GEO_TIFF",
    })
    return url, l3_cid


def stream_download(url: str, part: Path, timeout: int):
    if part.exists():
        part.unlink()

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        ctype = str(r.headers.get("Content-Type", "")).lower()
        if "text/html" in ctype or "application/json" in ctype:
            preview = r.content[:1000]
            raise RuntimeError(
                f"Unexpected Content-Type={ctype}; response={preview!r}"
            )

        with part.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
            f.flush()
            os.fsync(f.fileno())


def main():
    args = parse_args()

    old_path = Path(args.old_inventory).expanduser().resolve()
    outdir = Path(args.out).expanduser()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    initialize_ee(args.project)

    print("=" * 82)
    print("METHANESAT LIVE L4 POINT REFRESH")
    print("=" * 82)

    old = load_old_positives(old_path)
    print("Old model-ready positive rows:", len(old))
    print("Old positive collections:", old["collection_id"].nunique())

    live = fetch_live_l4_catalogue()
    print("Live L4 point features:", len(live))
    print("Live L4 collections:", live["collection_id"].nunique())
    print()

    ensure_output_mount(outdir)
    outdir = outdir.resolve()

    atomic_csv(live, outdir / "00_live_l4point_catalogue.csv")
    atomic_csv(old, outdir / "01_old_positive_inventory.csv")

    diff = compare_live_to_old(
        live,
        old,
        coord_match_m=args.coordinate_match_m,
        ambiguous_match_m=args.ambiguous_match_m,
    )
    atomic_csv(diff, outdir / "02_l4point_diff.csv")

    print("Diff classes:")
    print(diff["old_match_status"].value_counts().to_string())
    print()

    new = diff[diff["old_match_status"].eq("CLEAR_NEW_FEATURE")].copy()
    ambiguous = diff[
        diff["old_match_status"].eq("AMBIGUOUS_SAME_COLLECTION")
    ].copy()
    already = diff[diff["old_match_status"].eq("ALREADY_HAVE")].copy()

    atomic_csv(new, outdir / "03_clear_new_positive_features.csv")
    atomic_csv(ambiguous, outdir / "04_ambiguous_same_collection_features.csv")
    atomic_csv(already, outdir / "05_already_have_features.csv")

    if args.catalog_only:
        download_rows = pd.DataFrame()
        print("CATALOG ONLY: no imagery downloads requested.")
    else:
        download_rows = new.copy()
        if args.include_ambiguous:
            download_rows = pd.concat(
                [download_rows, ambiguous],
                ignore_index=True,
            )
        download_rows = download_rows.reset_index(drop=True)

    tif_dir = outdir / "new_positive_tif"
    tif_dir.mkdir(parents=True, exist_ok=True)

    status_jsonl = checkpoint_dir / "download_status.jsonl"
    if args.restart_checkpoint and status_jsonl.exists():
        status_jsonl.unlink()
        print("[RESTART] Local status log cleared.")
        print("[RESTART] Existing TIFFs are never deleted.")

    session = []

    for i, row in download_rows.iterrows():
        final = tif_dir / output_filename(row)
        part = final.with_suffix(final.suffix + ".part")

        prefix = (
            f"[{i+1}/{len(download_rows)}] "
            f"c{norm_collection(row['collection_id'])} "
            f"plume={row.get('plume_id')}"
        )

        valid_existing, meta = validate_tiff(final)
        if valid_existing:
            print(prefix, "SKIP valid existing")
            rec = {
                **row.to_dict(),
                "download_status": "SKIPPED_VALID_EXISTING",
                "output_tif": str(final),
                **meta,
            }
            append_jsonl_fsync(status_jsonl, rec)
            session.append(rec)
            continue

        if final.exists():
            bad = final.with_suffix(final.suffix + f".invalid.{int(time.time())}")
            final.rename(bad)
            print(prefix, "existing invalid TIFF preserved as", bad.name)

        success = False
        last_error = ""
        meta = {}
        l3_matched_cid = ""

        for attempt in range(1, args.retries + 1):
            try:
                print(prefix, f"download attempt {attempt}/{args.retries}")
                url, l3_matched_cid = make_download_url(
                    row,
                    half_m=args.crop_half_m,
                    scale_m=args.scale_m,
                )
                stream_download(url, part, args.timeout)

                valid, meta = validate_tiff(part)
                if not valid:
                    raise RuntimeError("Downloaded .part failed raster validation")

                os.replace(part, final)
                success = True
                break

            except KeyboardInterrupt:
                print("\nInterrupted. Final validated TIFFs remain safe.")
                raise

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                print("    ERROR:", last_error)
                if part.exists():
                    try:
                        part.unlink()
                    except Exception:
                        pass

                if attempt < args.retries:
                    wait = min(60, 3 * (2 ** (attempt - 1)))
                    print(f"    retrying in {wait}s...")
                    time.sleep(wait)

        if success:
            print("    SAVED:", final)
            rec = {
                **row.to_dict(),
                "download_status": "DOWNLOADED",
                "l3_matched_collection_id": l3_matched_cid,
                "output_tif": str(final),
                **meta,
            }
            append_jsonl_fsync(status_jsonl, rec)
            session.append(rec)
            time.sleep(args.sleep)
        else:
            rec = {
                **row.to_dict(),
                "download_status": "FAILED",
                "l3_matched_collection_id": l3_matched_cid,
                "output_tif": str(final),
                "error": last_error,
            }
            append_jsonl_fsync(status_jsonl, rec)
            session.append(rec)

    # Final disk audit of expected new downloads.
    audit_rows = []
    for _, row in download_rows.iterrows():
        final = tif_dir / output_filename(row)
        valid, meta = validate_tiff(final)
        audit_rows.append({
            **row.to_dict(),
            "feature_id": stable_feature_id(row),
            "expected_tif": str(final),
            "final_exists": final.exists(),
            "final_valid": bool(valid),
            **meta,
        })

    audit = pd.DataFrame(audit_rows)
    atomic_csv(audit, outdir / "06_new_positive_download_audit.csv")

    if len(audit):
        good = audit[audit["final_valid"].eq(True)].copy()
        failed = audit[~audit["final_valid"].eq(True)].copy()
    else:
        good = audit.copy()
        failed = audit.copy()

    atomic_csv(good, outdir / "07_new_positive_downloaded.csv")
    atomic_csv(failed, outdir / "08_new_positive_missing_or_invalid.csv")

    # Current catalogue manifest with local availability status.
    current = diff.copy()
    current["old_positive_available"] = current["old_match_status"].eq("ALREADY_HAVE")
    current["new_download_expected"] = current["old_match_status"].eq("CLEAR_NEW_FEATURE")
    if args.include_ambiguous:
        current["new_download_expected"] |= current["old_match_status"].eq(
            "AMBIGUOUS_SAME_COLLECTION"
        )

    path_map = {}
    if len(audit):
        for _, r in audit.iterrows():
            path_map[stable_feature_id(r)] = (
                r["expected_tif"],
                bool(r["final_valid"]),
            )

    current["refreshed_feature_id"] = current.apply(stable_feature_id, axis=1)
    current["new_local_tif"] = current["refreshed_feature_id"].map(
        lambda x: path_map.get(x, ("", False))[0]
    )
    current["new_local_tif_valid"] = current["refreshed_feature_id"].map(
        lambda x: path_map.get(x, ("", False))[1]
    )
    atomic_csv(current, outdir / "09_current_l4point_positive_manifest.csv")

    counts = diff["old_match_status"].value_counts().to_dict()
    summary = [
        "# MethaneSAT live L4 point-source refresh",
        "",
        "## Catalogue",
        f"- Old model-ready positive rows: {len(old)}",
        f"- Old positive collections: {old['collection_id'].nunique()}",
        f"- Live official L4 point features: {len(live)}",
        f"- Live official L4 collections: {live['collection_id'].nunique()}",
        "",
        "## Diff",
    ]
    for k in sorted(counts):
        summary.append(f"- {k}: {counts[k]}")
    summary += [
        f"- Clear-new collections: {new['collection_id'].nunique() if len(new) else 0}",
        f"- Ambiguous collections: {ambiguous['collection_id'].nunique() if len(ambiguous) else 0}",
        "",
        "## Download",
        f"- Requested new positive crops: {len(download_rows)}",
        f"- Valid new positive TIFFs on disk: {len(good)}",
        f"- Missing/invalid new TIFFs: {len(failed)}",
        "",
        "## Matching policy",
        f"- Same collection + exact usable plume_id => ALREADY_HAVE.",
        f"- Same collection + coordinate distance <= {args.coordinate_match_m:.0f} m => ALREADY_HAVE.",
        f"- Same collection + distance <= {args.ambiguous_match_m:.0f} m but > {args.coordinate_match_m:.0f} m => AMBIGUOUS.",
        "- CLEAR_NEW_FEATURE is downloaded by default.",
        "- Ambiguous rows are not downloaded unless --include-ambiguous is passed.",
        "",
        "## Resume policy",
        "- Existing valid TIFFs are skipped.",
        "- Downloads use .part + raster validation + fsync + atomic rename.",
        "- Local checkpoints are separate from SMB.",
        "- Rerun the exact same command after interruption.",
    ]
    (outdir / "SUMMARY_L4POINT_REFRESH.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )

    print()
    print("=" * 82)
    print("FINAL")
    print("=" * 82)
    print("Live L4 features        :", len(live))
    print("Already represented     :", len(already))
    print("Clear new features      :", len(new))
    print("Ambiguous same-collection:", len(ambiguous))
    print("Valid new TIFFs         :", len(good))
    print("Failed/missing          :", len(failed))
    print()
    print("Output:", outdir)
    print("Upload these first:")
    for fn in [
        "SUMMARY_L4POINT_REFRESH.md",
        "02_l4point_diff.csv",
        "03_clear_new_positive_features.csv",
        "04_ambiguous_same_collection_features.csv",
        "06_new_positive_download_audit.csv",
    ]:
        print(" ", outdir / fn)


if __name__ == "__main__":
    main()
