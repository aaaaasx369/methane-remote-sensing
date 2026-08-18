#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_methanesat_raw_local_background.py

MethaneSAT raw-L3 local-background audit.

Goal
----
Use the ACTUAL raw MethaneSAT L3 GeoTIFFs and the master inventory to test whether
the 480 m source/negative crop is locally enhanced relative to a 1–3 km background
annulus in the SAME L3 collection.

This is stricter than comparing already-cropped NPZ samples because it:
  1) returns to the original raw L3 XCH4 image;
  2) uses the exact sample coordinates from the master inventory;
  3) matches each sample to its collection_id in the raw TIFF filename;
  4) computes local background from the same acquisition/collection;
  5) compares positive and negative samples globally and within collection.

Expected inventory sheet:
    MethaneSAT_222

Expected columns:
    Latitude, Longitude, Label, Scene/Observation ID, Notes

Notes must contain:
    collection_id=...
and usually:
    target_id=...
    distance_to_nearest_L4_m=...

Outputs
-------
00_raw_l3_inventory.csv
01_sample_local_background_metrics.csv
02_label_diagnostics.csv
03_within_collection_diagnostics.csv
04_collection_pair_summary.csv
SUMMARY_METHANESAT_RAW_LOCAL.md

Important
---------
This script does NOT assume "61 TIFFs" or "222 samples". It counts what it actually
finds and reports all unmatched inventory rows / collections.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy as transform_xy
from rasterio.warp import transform as warp_transform
from scipy.stats import mannwhitneyu, rankdata, wilcoxon


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--l3-dir",
        default="/Volumes/engg-leung/dora lin/MethaneSAT_L3_L4_positive_61",
        help="Directory containing raw MethaneSAT L3 GeoTIFFs.",
    )
    p.add_argument(
        "--inventory",
        default="",
        help=(
            "Path to professor master inventory XLSX/CSV. If omitted, the script "
            "searches common roots for Professor_Master_Site_Date_Source_Inventory*.xlsx."
        ),
    )
    p.add_argument(
        "--inventory-sheet",
        default="MethaneSAT_222",
        help="Excel sheet containing the MethaneSAT model-sample inventory.",
    )
    p.add_argument(
        "--out",
        default="~/methane_release_project/methanesat_raw_local_background",
    )
    p.add_argument(
        "--source-half-size-m",
        type=float,
        default=240.0,
        help="Half-width of source/negative square; 240 m gives a 480x480 m crop.",
    )
    p.add_argument(
        "--annulus-inner-m",
        type=float,
        default=1000.0,
        help="Inner radius of same-scene local background annulus.",
    )
    p.add_argument(
        "--annulus-outer-m",
        type=float,
        default=3000.0,
        help="Outer radius of same-scene local background annulus.",
    )
    p.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Additional root to search for the master inventory; repeatable.",
    )
    return p.parse_args()


def find_inventory(explicit: str, roots: list[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Inventory not found: {p}")
        return p

    search_roots = [
        Path.home() / "methane_release_project",
        Path("/Volumes/engg-leung/dora lin"),
    ]
    search_roots.extend(Path(x).expanduser() for x in roots)

    candidates = []
    patterns = [
        "Professor_Master_Site_Date_Source_Inventory_V3_MethaneSAT_AVIRIS3_EMIT.xlsx",
        "Professor_Master_Site_Date_Source_Inventory_V2_MethaneSAT.xlsx",
        "*MethaneSAT*.xlsx",
    ]

    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for pat in patterns:
            for p in root.rglob(pat):
                rp = str(p.resolve())
                if rp not in seen:
                    seen.add(rp)
                    candidates.append(p.resolve())

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-find a MethaneSAT master inventory XLSX. "
            "Re-run with --inventory /full/path/to/file.xlsx"
        )

    # Prefer V3, then V2, then newest modified.
    def score(p: Path):
        name = p.name.lower()
        v3 = int("_v3_" in name)
        v2 = int("_v2_" in name)
        return (v3, v2, p.stat().st_mtime)

    candidates.sort(key=score, reverse=True)
    print("Inventory candidates:")
    for c in candidates[:10]:
        print(" ", c)
    print("Using:", candidates[0])
    return candidates[0]


def read_inventory(path: Path, sheet: str) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_csv(path)

    # Normalize column lookup without destroying originals.
    lower = {str(c).strip().lower(): c for c in df.columns}

    required = ["latitude", "longitude", "label", "notes"]
    missing = [x for x in required if x not in lower]
    if missing:
        raise ValueError(
            f"Inventory is missing required columns {missing}. "
            f"Columns={list(df.columns)}"
        )

    out = pd.DataFrame()
    out["latitude"] = pd.to_numeric(df[lower["latitude"]], errors="coerce")
    out["longitude"] = pd.to_numeric(df[lower["longitude"]], errors="coerce")
    out["label"] = pd.to_numeric(df[lower["label"]], errors="coerce")
    out["notes"] = df[lower["notes"]].fillna("").astype(str)

    scene_col = lower.get("scene/observation id")
    site_col = lower.get("site")
    out["sample_id"] = (
        df[scene_col].astype(str) if scene_col is not None
        else pd.Series([f"row_{i:06d}" for i in range(len(df))])
    )
    out["site"] = (
        df[site_col].astype(str) if site_col is not None
        else ""
    )

    def grab(pattern: str, s: str):
        m = re.search(pattern, s, flags=re.I)
        return m.group(1) if m else ""

    out["collection_id"] = out["notes"].map(
        lambda s: grab(r"\bcollection_id\s*=\s*([A-Za-z0-9]+)", s)
    )
    out["target_id"] = out["notes"].map(
        lambda s: grab(r"\btarget_id\s*=\s*([A-Za-z0-9_.-]+)", s)
    )
    out["distance_to_nearest_L4_m"] = pd.to_numeric(
        out["notes"].map(
            lambda s: grab(
                r"\bdistance_to_nearest_L4_m\s*=\s*([0-9eE+.\-]+)", s
            )
        ),
        errors="coerce",
    )

    out = out[
        out["label"].isin([0, 1])
        & out["latitude"].notna()
        & out["longitude"].notna()
        & out["collection_id"].astype(str).str.len().gt(0)
    ].copy()

    out["label"] = out["label"].astype(int)

    # The master sheet can occasionally contain repeated rendered copies.
    out = out.drop_duplicates(
        subset=[
            "sample_id", "latitude", "longitude", "label", "collection_id"
        ]
    ).reset_index(drop=True)

    return out


def parse_tiff_collection_id(name: str) -> str:
    m = re.search(r"_(c[0-9A-F]+)_", name, flags=re.I)
    return m.group(1) if m else ""


def inventory_raw_tiffs(l3_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(l3_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".tif", ".tiff"}:
            continue
        cid = parse_tiff_collection_id(p.name)
        rows.append({
            "collection_id": cid,
            "path": str(p.resolve()),
            "filename": p.name,
            "size_bytes": p.stat().st_size,
        })
    return pd.DataFrame(rows)


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * R * np.arcsin(np.sqrt(a))


def local_xy_m(lat, lon, lat0, lon0):
    # Accurate enough for <= 3 km diagnostics.
    dy = (lat - lat0) * 110540.0
    dx = (lon - lon0) * 111320.0 * np.cos(np.radians(lat0))
    return dx, dy


def source_to_pixel(ds, lat, lon):
    if ds.crs is None:
        raise ValueError("GeoTIFF has no CRS.")

    if str(ds.crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
        xs, ys = [lon], [lat]
    else:
        xs, ys = warp_transform("EPSG:4326", ds.crs, [lon], [lat])

    row, col = ds.index(xs[0], ys[0])
    return int(row), int(col)


def pixel_resolution_m(ds, row, col):
    # Pixel center and one-pixel neighbors -> WGS84 -> haversine.
    coords = []
    for rr, cc in [(row, col), (row, col + 1), (row + 1, col)]:
        x, y = transform_xy(ds.transform, rr, cc, offset="center")
        coords.append((x, y))

    xs = [z[0] for z in coords]
    ys = [z[1] for z in coords]

    if str(ds.crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
        lons, lats = xs, ys
    else:
        lons, lats = warp_transform(ds.crs, "EPSG:4326", xs, ys)

    xres = float(haversine_m(lats[0], lons[0], lats[1], lons[1]))
    yres = float(haversine_m(lats[0], lons[0], lats[2], lons[2]))
    return xres, yres


def read_local_window(ds, lat, lon, outer_m):
    row0, col0 = source_to_pixel(ds, lat, lon)
    xres, yres = pixel_resolution_m(ds, row0, col0)

    pix_m = max(min(xres, yres), 1.0)
    half = int(math.ceil((outer_m + 300.0) / pix_m))

    window = Window(
        col_off=col0 - half,
        row_off=row0 - half,
        width=2 * half + 1,
        height=2 * half + 1,
    )

    arr = ds.read(
        1,
        window=window,
        boundless=True,
        masked=True,
    ).astype("float64")

    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    arr = np.asarray(arr, dtype=float)

    # Convert window pixel centers to lon/lat.
    h, w = arr.shape
    global_rows = np.arange(row0 - half, row0 + half + 1)
    global_cols = np.arange(col0 - half, col0 + half + 1)
    rr, cc = np.meshgrid(global_rows, global_cols, indexing="ij")

    xs, ys = transform_xy(
        ds.transform,
        rr.ravel(),
        cc.ravel(),
        offset="center",
    )
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if str(ds.crs).upper() in {"EPSG:4326", "OGC:CRS84"}:
        lons = xs
        lats = ys
    else:
        lons, lats = warp_transform(
            ds.crs,
            "EPSG:4326",
            xs.tolist(),
            ys.tolist(),
        )
        lons = np.asarray(lons)
        lats = np.asarray(lats)

    lats = lats.reshape(h, w)
    lons = lons.reshape(h, w)

    dx, dy = local_xy_m(lats, lons, lat, lon)
    dist = np.sqrt(dx ** 2 + dy ** 2)

    return arr, dx, dy, dist, xres, yres


def robust_mad(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    med = np.median(v)
    return float(np.median(np.abs(v - med)))


def stats(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }
    return {
        "n": len(v),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "max": float(np.max(v)),
    }


def metrics_for_sample(
    ds,
    lat,
    lon,
    half_size_m,
    annulus_inner_m,
    annulus_outer_m,
):
    arr, dx, dy, dist, xres, yres = read_local_window(
        ds, lat, lon, annulus_outer_m
    )

    finite = np.isfinite(arr)

    source_square = (
        finite
        & (np.abs(dx) <= half_size_m)
        & (np.abs(dy) <= half_size_m)
    )

    source_circle = finite & (dist <= half_size_m)

    annulus = (
        finite
        & (dist >= annulus_inner_m)
        & (dist <= annulus_outer_m)
    )

    sq = stats(arr[source_square])
    ci = stats(arr[source_circle])
    bg = stats(arr[annulus])

    bg_vals = arr[annulus]
    bg_vals = bg_vals[np.isfinite(bg_vals)]
    bg_mad = robust_mad(bg_vals)
    sigma = 1.4826 * bg_mad if np.isfinite(bg_mad) else np.nan

    out = {
        "pixel_res_x_m": xres,
        "pixel_res_y_m": yres,
        "source_square_n": sq["n"],
        "source_circle_n": ci["n"],
        "annulus_n": bg["n"],
        "source_square_mean": sq["mean"],
        "source_square_median": sq["median"],
        "source_square_p95": sq["p95"],
        "source_square_p99": sq["p99"],
        "source_circle_mean": ci["mean"],
        "source_circle_median": ci["median"],
        "source_circle_p95": ci["p95"],
        "annulus_mean": bg["mean"],
        "annulus_median": bg["median"],
        "annulus_std": bg["std"],
        "annulus_mad": bg_mad,
    }

    for prefix, source in [
        ("square", sq),
        ("circle", ci),
    ]:
        for stat_name in ["mean", "median", "p95", "p99"]:
            if stat_name not in source:
                continue
            val = source[stat_name]
            contrast = (
                val - bg["median"]
                if np.isfinite(val) and np.isfinite(bg["median"])
                else np.nan
            )
            out[f"{prefix}_{stat_name}_minus_annulus_median"] = contrast
            out[f"{prefix}_{stat_name}_z"] = (
                contrast / sigma
                if np.isfinite(contrast) and np.isfinite(sigma) and sigma > 0
                else np.nan
            )

    return out


def auc_positive_high(y, score):
    d = pd.DataFrame({
        "y": pd.to_numeric(y, errors="coerce"),
        "s": pd.to_numeric(score, errors="coerce"),
    }).dropna()

    if d["y"].nunique() < 2:
        return np.nan

    y = d["y"].astype(int).to_numpy()
    s = d["s"].to_numpy(float)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())

    ranks = rankdata(s, method="average")
    u = ranks[y == 1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def diagnostic_features(df):
    exclude = {
        "label", "latitude", "longitude", "distance_to_nearest_L4_m",
        "pixel_res_x_m", "pixel_res_y_m",
        "source_square_n", "source_circle_n", "annulus_n",
    }
    features = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if any(
                token in c for token in [
                    "source_square_", "source_circle_", "annulus_",
                    "_minus_annulus_", "_z"
                ]
            ):
                features.append(c)
    return features


def label_diagnostics(df):
    rows = []
    for f in diagnostic_features(df):
        d = df[["label", f]].copy()
        d["label"] = pd.to_numeric(d["label"], errors="coerce")
        d[f] = pd.to_numeric(d[f], errors="coerce")
        d = d.dropna()

        if d["label"].nunique() < 2:
            continue

        pos = d.loc[d.label == 1, f].to_numpy()
        neg = d.loc[d.label == 0, f].to_numpy()
        auc = auc_positive_high(d.label, d[f])

        try:
            p = mannwhitneyu(pos, neg, alternative="two-sided").pvalue
        except Exception:
            p = np.nan

        rows.append({
            "feature": f,
            "n": len(d),
            "n_positive": len(pos),
            "n_negative": len(neg),
            "positive_median": float(np.median(pos)),
            "negative_median": float(np.median(neg)),
            "raw_auc_positive_high": auc,
            "orientation_free_auc": max(auc, 1 - auc),
            "mannwhitney_p": p,
        })

    return pd.DataFrame(rows)


def within_collection(df):
    rows = []
    features = diagnostic_features(df)

    for f in features:
        pair_diffs = []
        sample_rows = []

        for cid, g in df.groupby("collection_id"):
            gg = g[["label", f]].copy()
            gg["label"] = pd.to_numeric(gg["label"], errors="coerce")
            gg[f] = pd.to_numeric(gg[f], errors="coerce")
            gg = gg.dropna()

            pos = gg.loc[gg.label == 1, f]
            neg = gg.loc[gg.label == 0, f]
            if not len(pos) or not len(neg):
                continue

            pair_diffs.append({
                "collection_id": cid,
                "positive_mean": float(pos.mean()),
                "negative_mean": float(neg.mean()),
                "difference_pos_minus_neg": float(pos.mean() - neg.mean()),
            })

            tmp = gg.copy()
            tmp["collection_id"] = cid
            tmp["scene_demeaned"] = tmp[f] - tmp[f].mean()
            sample_rows.append(tmp)

        pairs = pd.DataFrame(pair_diffs)
        if not len(pairs):
            continue

        dd = pd.concat(sample_rows, ignore_index=True)
        auc = auc_positive_high(dd["label"], dd["scene_demeaned"])
        dif = pairs["difference_pos_minus_neg"].to_numpy(float)

        try:
            wp = (
                wilcoxon(dif, alternative="two-sided").pvalue
                if len(dif) >= 5 and np.any(dif != 0)
                else np.nan
            )
        except Exception:
            wp = np.nan

        rows.append({
            "feature": f,
            "mixed_label_collections": len(pairs),
            "sample_rows_in_mixed_collections": len(dd),
            "fraction_collections_positive_gt_negative": float(np.mean(dif > 0)),
            "median_collection_pos_minus_neg": float(np.median(dif)),
            "wilcoxon_p": wp,
            "collection_demeaned_raw_auc": auc,
            "collection_demeaned_orientation_free_auc": (
                max(auc, 1 - auc) if np.isfinite(auc) else np.nan
            ),
        })

    return pd.DataFrame(rows)


def collection_pair_summary(df):
    rows = []
    for cid, g in df.groupby("collection_id"):
        pos = g[g.label == 1]
        neg = g[g.label == 0]
        rows.append({
            "collection_id": cid,
            "n": len(g),
            "n_positive": len(pos),
            "n_negative": len(neg),
            "has_both_labels": bool(len(pos) and len(neg)),
            "raw_tiff_path": (
                g["raw_tiff_path"].dropna().iloc[0]
                if g["raw_tiff_path"].notna().any()
                else ""
            ),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    l3_dir = Path(args.l3_dir).expanduser().resolve()
    if not l3_dir.exists():
        raise FileNotFoundError(f"L3 directory does not exist: {l3_dir}")

    outdir = Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    inventory_path = find_inventory(args.inventory, args.search_root)
    inv = read_inventory(inventory_path, args.inventory_sheet)

    raw = inventory_raw_tiffs(l3_dir)
    raw.to_csv(outdir / "00_raw_l3_inventory.csv", index=False)

    print("\nACTUAL RAW L3:")
    print(" TIFFs:", len(raw))
    print(" Collections:", raw["collection_id"].replace("", np.nan).nunique())

    # If duplicate raw TIFFs map to same collection, keep them in the raw inventory
    # but use the first one for metrics and report the duplication.
    path_by_collection = (
        raw[raw.collection_id.astype(str).str.len() > 0]
        .drop_duplicates("collection_id")
        .set_index("collection_id")["path"]
        .to_dict()
    )

    inv["raw_tiff_path"] = inv["collection_id"].map(path_by_collection)
    inv["raw_tiff_found"] = inv["raw_tiff_path"].notna()

    print("\nACTUAL MASTER INVENTORY:")
    print(" rows:", len(inv))
    print(" positive:", int((inv.label == 1).sum()))
    print(" negative:", int((inv.label == 0).sum()))
    print(" collections:", inv.collection_id.nunique())
    print(" rows with matched raw TIFF:", int(inv.raw_tiff_found.sum()))

    rows = []
    grouped = inv.groupby("collection_id", sort=True)

    for gi, (cid, g) in enumerate(grouped, 1):
        tif = path_by_collection.get(cid)
        print(
            f"[{gi}/{inv.collection_id.nunique()}] "
            f"{cid} rows={len(g)} tif={'YES' if tif else 'NO'}"
        )

        if not tif:
            for _, r in g.iterrows():
                rec = r.to_dict()
                rec.update({
                    "status": "FAIL",
                    "error": "raw_tiff_not_found_for_collection",
                })
                rows.append(rec)
            continue

        try:
            with rasterio.open(tif) as ds:
                for _, r in g.iterrows():
                    rec = r.to_dict()
                    rec["status"] = "PASS"
                    rec["error"] = ""
                    try:
                        rec.update(
                            metrics_for_sample(
                                ds,
                                float(r.latitude),
                                float(r.longitude),
                                args.source_half_size_m,
                                args.annulus_inner_m,
                                args.annulus_outer_m,
                            )
                        )
                    except Exception as exc:
                        rec["status"] = "FAIL"
                        rec["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(rec)

        except Exception as exc:
            for _, r in g.iterrows():
                rec = r.to_dict()
                rec.update({
                    "status": "FAIL",
                    "error": f"raster_open_failed: {type(exc).__name__}: {exc}",
                })
                rows.append(rec)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        outdir / "01_sample_local_background_metrics.csv",
        index=False,
    )

    good = metrics[metrics.status.eq("PASS")].copy()

    diag = label_diagnostics(good)
    diag.to_csv(outdir / "02_label_diagnostics.csv", index=False)

    wc = within_collection(good)
    wc.to_csv(outdir / "03_within_collection_diagnostics.csv", index=False)

    cps = collection_pair_summary(metrics)
    cps.to_csv(outdir / "04_collection_pair_summary.csv", index=False)

    lines = []
    lines.append("# MethaneSAT raw-L3 local-background audit")
    lines.append("")
    lines.append("## Actual data")
    lines.append(f"- Raw L3 TIFFs discovered: {len(raw)}")
    lines.append(
        f"- Unique TIFF collection IDs: "
        f"{raw['collection_id'].replace('', np.nan).nunique()}"
    )
    lines.append(f"- Master inventory rows used: {len(inv)}")
    lines.append(f"- Positive: {int((inv.label == 1).sum())}")
    lines.append(f"- Negative: {int((inv.label == 0).sum())}")
    lines.append(f"- Inventory collections: {inv.collection_id.nunique()}")
    lines.append(f"- Rows matched to raw L3 TIFF: {int(inv.raw_tiff_found.sum())}")
    lines.append(f"- Pixel-successful rows: {len(good)}")
    lines.append(f"- Failed rows: {int((metrics.status != 'PASS').sum())}")
    lines.append("")
    lines.append("## Geometry")
    lines.append(
        f"- Source/negative region: {2*args.source_half_size_m:.0f} m x "
        f"{2*args.source_half_size_m:.0f} m square"
    )
    lines.append(
        f"- Local background annulus: "
        f"{args.annulus_inner_m/1000:.1f}–"
        f"{args.annulus_outer_m/1000:.1f} km"
    )
    lines.append("")
    lines.append("## Strongest global label diagnostics")
    if len(diag):
        for _, z in diag.sort_values(
            ["orientation_free_auc", "mannwhitney_p"],
            ascending=[False, True],
        ).head(12).iterrows():
            lines.append(
                f"- {z.feature}: raw AUC={z.raw_auc_positive_high:.3f}, "
                f"orientation-free={z.orientation_free_auc:.3f}, "
                f"p={z.mannwhitney_p:.3g}, "
                f"pos median={z.positive_median:.3f}, "
                f"neg median={z.negative_median:.3f}"
            )

    lines.append("")
    lines.append("## Strongest within-collection diagnostics")
    if len(wc):
        for _, z in wc.sort_values(
            "collection_demeaned_orientation_free_auc",
            ascending=False,
        ).head(12).iterrows():
            lines.append(
                f"- {z.feature}: mixed collections={int(z.mixed_label_collections)}, "
                f"collection-demeaned raw AUC={z.collection_demeaned_raw_auc:.3f}, "
                f"orientation-free={z.collection_demeaned_orientation_free_auc:.3f}, "
                f"positive>negative collections="
                f"{z.fraction_collections_positive_gt_negative:.3f}, "
                f"Wilcoxon p={z.wilcoxon_p:.3g}"
            )

    lines.append("")
    lines.append("## Interpretation rule")
    lines.append(
        "- Strong positive source-minus-annulus contrasts that persist after "
        "within-collection control support localized XCH4 enhancement."
    )
    lines.append(
        "- If raw/global XCH4 separates labels but source-minus-annulus contrasts "
        "collapse, large-scale spatial background remains a plausible confounder."
    )
    lines.append(
        "- This is a confounding audit, not a calibrated flux retrieval."
    )

    (outdir / "SUMMARY_METHANESAT_RAW_LOCAL.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\nDONE:", outdir)
    print("Upload these:")
    for fn in [
        "SUMMARY_METHANESAT_RAW_LOCAL.md",
        "00_raw_l3_inventory.csv",
        "01_sample_local_background_metrics.csv",
        "02_label_diagnostics.csv",
        "03_within_collection_diagnostics.csv",
        "04_collection_pair_summary.csv",
    ]:
        print(" ", outdir / fn)


if __name__ == "__main__":
    main()
