#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_methanesat_confounds.py

Purpose
-------
Audit the ACTUAL local MethaneSAT model-ready NPZ package before doing any new training.

Main research question:
    Are positive/negative differences driven by localized methane enhancement,
    or by scene/background/domain differences?

The script does NOT assume the old remembered sample counts are correct.
It scans the paths you give it, discovers MethaneSAT NPZ packages, inspects the
actual NPZ files, and reports what is truly present now.

What it checks
--------------
1) Exact local MethaneSAT package(s), NPZ count, labels, array schema.
2) Whether positives and negatives share the same L3 scene / collection.
3) XCH4 background statistics versus center-localized contrast statistics.
4) Global label AUROC and same-scene / scene-demeaned AUROC where possible.
5) Negative distance-to-source metadata if it exists.

Outputs
-------
00_detected_packages.csv
01_methanesat_sample_inventory.csv
02_label_diagnostics.csv
03_scene_pair_diagnostics.csv
04_array_schema_summary.csv
SUMMARY_METHANESAT_AUDIT.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, wilcoxon


SKIP_DIR_NAMES = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    "Library", ".Trash", ".cache", "Caches",
}

LABEL_KEYS = [
    "label", "y", "target", "class", "class_label",
    "plume_label", "binary_label",
]

SCENE_KEYS = [
    "collection_id", "source_collection_id", "l3_collection_id",
    "scene_id", "l3_scene_id", "scene", "granule_id",
    "l3_filename", "source_l3_filename", "l3_file", "source_l3_file",
]

SAMPLE_KEYS = [
    "sample_id", "id", "sample", "external_eval_id",
]

LAT_KEYS = ["lat", "latitude", "center_lat", "source_lat"]
LON_KEYS = ["lon", "longitude", "center_lon", "source_lon"]

DISTANCE_KEYS = [
    "nearest_source_distance_km",
    "distance_to_nearest_source_km",
    "distance_to_source_km",
    "negative_distance_km",
    "source_distance_km",
]


def safe_scalar(x: Any) -> Any:
    """Convert small NPZ scalar/object values to Python scalars/strings."""
    if isinstance(x, np.ndarray):
        if x.shape == ():
            try:
                return safe_scalar(x.item())
            except Exception:
                return str(x)
        if x.size == 1:
            try:
                return safe_scalar(x.reshape(-1)[0])
            except Exception:
                return str(x)
        return x
    if isinstance(x, (np.generic,)):
        return x.item()
    return x


def flatten_obj(obj: Any, prefix: str = "", out: dict | None = None) -> dict:
    if out is None:
        out = {}

    obj = safe_scalar(obj)

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flatten_obj(v, key, out)
        return out

    if isinstance(obj, (list, tuple)):
        if len(obj) <= 20:
            for i, v in enumerate(obj):
                key = f"{prefix}[{i}]"
                flatten_obj(v, key, out)
        else:
            out[prefix] = f"<sequence length={len(obj)}>"
        return out

    if isinstance(obj, np.ndarray):
        out[prefix] = f"<array shape={obj.shape} dtype={obj.dtype}>"
        return out

    out[prefix] = obj
    return out


def normalized_meta(meta: dict) -> dict:
    result = {}
    for k, v in meta.items():
        lk = str(k).lower()
        result[lk] = v
        # Also expose final component of nested keys, e.g. metadata.label -> label.
        tail = re.split(r"[.\[\]]+", lk)[-1]
        if tail and tail not in result:
            result[tail] = v
    return result


def first_value(meta: dict, keys: list[str]) -> tuple[Any, str]:
    nm = normalized_meta(meta)

    # Exact match first.
    for key in keys:
        if key.lower() in nm:
            v = nm[key.lower()]
            if v is not None and str(v).strip() not in {"", "nan", "None"}:
                return v, key

    # Suffix / contains fallback for nested metadata.
    for actual_key, v in nm.items():
        for key in keys:
            kk = key.lower()
            if actual_key.endswith("." + kk) or actual_key == kk:
                if v is not None and str(v).strip() not in {"", "nan", "None"}:
                    return v, actual_key

    return None, ""


def parse_label(meta: dict, path: Path) -> tuple[float, str]:
    v, source = first_value(meta, LABEL_KEYS)
    if v is not None:
        try:
            f = float(v)
            if f in (0.0, 1.0):
                return f, f"metadata:{source}"
        except Exception:
            s = str(v).strip().lower()
            if s in {"positive", "pos", "plume", "true", "yes"}:
                return 1.0, f"metadata:{source}"
            if s in {"negative", "neg", "no_plume", "background", "false", "no"}:
                return 0.0, f"metadata:{source}"

    text = str(path).lower()
    # Strong path signals only; avoid interpreting "posneg" as a label.
    if re.search(r"(^|[/_.-])(negative|neg)([/_.-]|$)", text):
        return 0.0, "path"
    if re.search(r"(^|[/_.-])(positive|pos)([/_.-]|$)", text):
        return 1.0, "path"

    # Common filename suffixes.
    stem = path.stem.lower()
    if re.search(r"(?:^|[_-])neg(?:ative)?(?:[_-]|$)", stem):
        return 0.0, "filename"
    if re.search(r"(?:^|[_-])pos(?:itive)?(?:[_-]|$)", stem):
        return 1.0, "filename"

    return np.nan, ""


def canonical_scene_id(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return ""
    s = Path(s).name
    # Remove extension but keep the original collection token.
    s = re.sub(r"\.(tif|tiff|nc|npz|npy)$", "", s, flags=re.I)
    return s


def infer_scene(meta: dict, path: Path) -> tuple[str, str]:
    v, source = first_value(meta, SCENE_KEYS)
    if v is not None:
        return canonical_scene_id(v), f"metadata:{source}"

    # Conservative filename fallback: strip only obvious POS/NEG/sample suffixes.
    stem = path.stem
    fallback = re.sub(
        r"(?i)(?:[_-](?:pos|positive|neg|negative))(?:[_-]?\d+)?$",
        "",
        stem,
    )
    if fallback != stem:
        return fallback, "filename_fallback"

    return "", ""


def infer_sample_id(meta: dict, path: Path) -> tuple[str, str]:
    v, source = first_value(meta, SAMPLE_KEYS)
    if v is not None:
        return str(v), f"metadata:{source}"
    return path.stem, "filename"


def find_npz_files(roots: list[Path]) -> list[Path]:
    found = []
    seen = set()

    for root in roots:
        root = root.expanduser()
        if not root.exists():
            print(f"[WARN] root does not exist: {root}")
            continue

        if root.is_file() and root.suffix.lower() == ".npz":
            if "methanesat" in str(root).lower():
                found.append(root.resolve())
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune expensive/irrelevant dirs.
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIR_NAMES
                and not d.startswith(".")
            ]

            low_dir = dirpath.lower()
            for fn in filenames:
                if not fn.lower().endswith(".npz"):
                    continue
                p = Path(dirpath) / fn
                if "methanesat" not in str(p).lower():
                    continue
                rp = str(p.resolve())
                if rp not in seen:
                    seen.add(rp)
                    found.append(p.resolve())

    return sorted(found)


def package_root_for(p: Path) -> Path:
    """
    Identify a logical package root.
    Prefer an ancestor whose name contains methanesat and where descendants
    include samples/NPZs. Otherwise use parent.
    """
    candidates = []
    for anc in [p.parent, *p.parents]:
        if "methanesat" in anc.name.lower():
            candidates.append(anc)
        if len(candidates) >= 3:
            break

    # Choose the nearest MethaneSAT-named ancestor.
    return candidates[0] if candidates else p.parent


def select_array(npz: np.lib.npyio.NpzFile) -> tuple[str, np.ndarray | None]:
    priorities = ["ch4", "xch4", "methane", "image", "data", "patch"]

    for k in priorities:
        if k in npz.files:
            try:
                a = np.asarray(npz[k])
                if a.ndim >= 2 and np.issubdtype(a.dtype, np.number):
                    return k, a
            except Exception:
                pass

    # Fallback: largest numeric >=2D array by element count.
    choices = []
    for k in npz.files:
        try:
            a = np.asarray(npz[k])
        except Exception:
            continue
        if a.ndim >= 2 and np.issubdtype(a.dtype, np.number):
            choices.append((a.size, k, a))
    if choices:
        _, k, a = max(choices, key=lambda z: z[0])
        return k, a

    return "", None


def extract_image_2d(a: np.ndarray) -> tuple[np.ndarray | None, str, list[float]]:
    a = np.asarray(a, dtype=float)

    if a.ndim == 2:
        frac = float(np.isfinite(a).mean())
        return a, "2d", [frac]

    if a.ndim == 3:
        # Candidate channel-first.
        if a.shape[0] <= 10:
            fracs = [float(np.isfinite(a[i]).mean()) for i in range(a.shape[0])]
            idx = int(np.nanargmax(fracs))
            return a[idx], f"channel_first[{idx}]", fracs

        # Candidate channel-last.
        if a.shape[-1] <= 10:
            fracs = [float(np.isfinite(a[..., i]).mean()) for i in range(a.shape[-1])]
            idx = int(np.nanargmax(fracs))
            return a[..., idx], f"channel_last[{idx}]", fracs

    return None, "unsupported", []


def robust_mad(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    med = np.median(v)
    return float(np.median(np.abs(v - med)))


def image_features(img: np.ndarray) -> dict:
    img = np.asarray(img, dtype=float)
    finite = np.isfinite(img)

    out = {
        "finite_fraction": float(finite.mean()),
        "height": int(img.shape[0]),
        "width": int(img.shape[1]),
    }

    vals = img[finite]
    if len(vals) < 20:
        for k in [
            "global_mean", "global_median", "global_std", "global_p95", "global_p99",
            "center_median", "center_p95", "ring_median", "ring_mad",
            "center_minus_ring_median", "center_minus_ring_p95",
            "center_z_median", "center_z_p95",
        ]:
            out[k] = np.nan
        return out

    h, w = img.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.indices((h, w))

    # Dimensionless normalized radius:
    # r=1 is approximately the nearest edge from center.
    scale = min(h, w) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max(scale, 1)

    center = finite & (rr <= 0.25)
    ring = finite & (rr >= 0.60) & (rr <= 0.90)

    cv = img[center]
    rv = img[ring]

    out.update({
        "global_mean": float(np.mean(vals)),
        "global_median": float(np.median(vals)),
        "global_std": float(np.std(vals)),
        "global_p95": float(np.percentile(vals, 95)),
        "global_p99": float(np.percentile(vals, 99)),
    })

    if len(cv) >= 10 and len(rv) >= 20:
        rmed = float(np.median(rv))
        rmad = robust_mad(rv)
        sigma = 1.4826 * rmad + 1e-12

        cmed = float(np.median(cv))
        cp95 = float(np.percentile(cv, 95))

        out.update({
            "center_median": cmed,
            "center_p95": cp95,
            "ring_median": rmed,
            "ring_mad": rmad,
            "center_minus_ring_median": cmed - rmed,
            "center_minus_ring_p95": cp95 - rmed,
            "center_z_median": (cmed - rmed) / sigma,
            "center_z_p95": (cp95 - rmed) / sigma,
        })
    else:
        for k in [
            "center_median", "center_p95", "ring_median", "ring_mad",
            "center_minus_ring_median", "center_minus_ring_p95",
            "center_z_median", "center_z_p95",
        ]:
            out[k] = np.nan

    return out


def read_npz_record(path: Path, package_root: Path) -> dict:
    rec = {
        "path": str(path),
        "package_root": str(package_root),
        "filename": path.name,
        "status": "PASS",
        "error": "",
    }

    try:
        # Local user-owned scientific files; allow_pickle is needed for saved metadata dicts.
        with np.load(path, allow_pickle=True) as z:
            rec["npz_keys"] = "|".join(z.files)

            meta = {}
            for k in z.files:
                try:
                    a = z[k]
                except Exception:
                    continue

                # Flatten only scalar/small object metadata; don't flatten image arrays.
                if np.asarray(a).ndim == 0 or np.asarray(a).dtype == object:
                    try:
                        flatten_obj(a, k, meta)
                    except Exception:
                        pass

            label, label_method = parse_label(meta, path)
            rec["label"] = label
            rec["label_method"] = label_method

            sample_id, sample_method = infer_sample_id(meta, path)
            rec["sample_id"] = sample_id
            rec["sample_id_method"] = sample_method

            scene_id, scene_method = infer_scene(meta, path)
            rec["scene_id"] = scene_id
            rec["scene_id_method"] = scene_method

            lat, lat_src = first_value(meta, LAT_KEYS)
            lon, lon_src = first_value(meta, LON_KEYS)
            try:
                rec["lat"] = float(lat) if lat is not None else np.nan
            except Exception:
                rec["lat"] = np.nan
            try:
                rec["lon"] = float(lon) if lon is not None else np.nan
            except Exception:
                rec["lon"] = np.nan

            dist, dist_src = first_value(meta, DISTANCE_KEYS)
            try:
                rec["nearest_source_distance_km"] = (
                    float(dist) if dist is not None else np.nan
                )
                rec["distance_metadata_key"] = dist_src
            except Exception:
                rec["nearest_source_distance_km"] = np.nan
                rec["distance_metadata_key"] = ""

            array_key, arr = select_array(z)
            rec["array_key"] = array_key

            if arr is None:
                raise ValueError("No numeric >=2D array found in NPZ.")

            rec["array_shape"] = "x".join(map(str, arr.shape))
            rec["array_dtype"] = str(arr.dtype)

            img, channel_method, channel_fracs = extract_image_2d(arr)
            rec["image_channel_method"] = channel_method
            rec["channel_finite_fractions"] = json.dumps(channel_fracs)

            if img is None:
                raise ValueError(
                    f"Could not convert array {array_key} shape={arr.shape} to one 2D image."
                )

            rec.update(image_features(img))

    except Exception as exc:
        rec["status"] = "FAIL"
        rec["error"] = f"{type(exc).__name__}: {exc}"

    return rec


def auc_positive_high(y, score) -> float:
    d = pd.DataFrame({
        "y": pd.to_numeric(y, errors="coerce"),
        "s": pd.to_numeric(score, errors="coerce"),
    }).dropna()

    if d["y"].nunique() < 2:
        return np.nan

    yv = d["y"].astype(int).to_numpy()
    sv = d["s"].to_numpy(float)
    n1 = int((yv == 1).sum())
    n0 = int((yv == 0).sum())

    ranks = rankdata(sv, method="average")
    u = ranks[yv == 1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def label_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "global_mean", "global_median", "global_std", "global_p95", "global_p99",
        "center_median", "center_p95", "ring_median", "ring_mad",
        "center_minus_ring_median", "center_minus_ring_p95",
        "center_z_median", "center_z_p95",
    ]

    rows = []
    for f in features:
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


def scene_pair_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For mixed-label scenes, compare positive and negative rows within the same scene.
    This is the key confounding test because scene/background is controlled.
    """
    d0 = df.copy()
    d0 = d0[d0["scene_id"].astype(str).str.len() > 0].copy()

    mixed_scenes = []
    for sid, g in d0.groupby("scene_id"):
        labels = set(pd.to_numeric(g["label"], errors="coerce").dropna().astype(int))
        if labels == {0, 1}:
            mixed_scenes.append(sid)

    d0 = d0[d0.scene_id.isin(mixed_scenes)].copy()

    features = [
        "global_median", "global_p95", "ring_median",
        "center_minus_ring_median", "center_minus_ring_p95",
        "center_z_median", "center_z_p95",
    ]

    rows = []
    for f in features:
        per_scene = []
        for sid, g in d0.groupby("scene_id"):
            gg = g[["label", f]].copy()
            gg["label"] = pd.to_numeric(gg["label"], errors="coerce")
            gg[f] = pd.to_numeric(gg[f], errors="coerce")
            gg = gg.dropna()

            pos = gg.loc[gg.label == 1, f]
            neg = gg.loc[gg.label == 0, f]
            if len(pos) and len(neg):
                per_scene.append({
                    "scene_id": sid,
                    "positive_mean": float(pos.mean()),
                    "negative_mean": float(neg.mean()),
                    "difference_pos_minus_neg": float(pos.mean() - neg.mean()),
                })

        ps = pd.DataFrame(per_scene)
        if len(ps) == 0:
            continue

        dif = ps["difference_pos_minus_neg"].to_numpy(float)
        fraction_positive = float(np.mean(dif > 0))

        try:
            if np.any(dif != 0) and len(dif) >= 5:
                wp = wilcoxon(dif, alternative="two-sided").pvalue
            else:
                wp = np.nan
        except Exception:
            wp = np.nan

        # Scene-demean sample rows, then compute AUC.
        dd = d0[["scene_id", "label", f]].copy()
        dd[f] = pd.to_numeric(dd[f], errors="coerce")
        dd["label"] = pd.to_numeric(dd["label"], errors="coerce")
        dd = dd.dropna()
        dd["scene_demeaned"] = dd[f] - dd.groupby("scene_id")[f].transform("mean")
        auc = auc_positive_high(dd["label"], dd["scene_demeaned"])

        rows.append({
            "feature": f,
            "mixed_label_scenes": len(ps),
            "sample_rows_in_mixed_scenes": len(dd),
            "fraction_scenes_positive_gt_negative": fraction_positive,
            "median_scene_pos_minus_neg": float(np.median(dif)),
            "wilcoxon_p": wp,
            "scene_demeaned_raw_auc": auc,
            "scene_demeaned_orientation_free_auc": (
                max(auc, 1 - auc) if np.isfinite(auc) else np.nan
            ),
        })

    return pd.DataFrame(rows)


def array_schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "package_root", "array_key", "array_shape", "array_dtype",
        "image_channel_method", "channel_finite_fractions",
        "npz_keys",
    ]
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.DataFrame()

    return (
        df.groupby(present, dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="+",
        default=["~/methane_release_project"],
        help=(
            "Roots to scan. Example: --roots ~/methane_release_project "
            "\"/Volumes/engg-leung/dora lin\""
        ),
    )
    ap.add_argument(
        "--data-dir",
        default="",
        help=(
            "Optional exact MethaneSAT package. If provided, only this directory "
            "is scanned."
        ),
    )
    ap.add_argument(
        "--out",
        default="~/methane_release_project/methanesat_confounds_audit",
    )
    args = ap.parse_args()

    outdir = Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.data_dir:
        roots = [Path(args.data_dir).expanduser()]
    else:
        roots = [Path(r).expanduser() for r in args.roots]

    print("Scanning roots:")
    for r in roots:
        print(" ", r)

    files = find_npz_files(roots)

    if not files:
        raise SystemExit(
            "No MethaneSAT NPZ files found under the requested roots. "
            "No counts were assumed."
        )

    package_map = defaultdict(list)
    for p in files:
        package_map[package_root_for(p)].append(p)

    package_rows = []
    for root, ps in sorted(package_map.items(), key=lambda kv: -len(kv[1])):
        package_rows.append({
            "package_root": str(root),
            "npz_files": len(ps),
        })

    packages = pd.DataFrame(package_rows)
    packages.to_csv(outdir / "00_detected_packages.csv", index=False)

    print("\nDetected MethaneSAT packages:")
    print(packages.to_string(index=False))

    # Inspect all discovered MethaneSAT NPZs.
    rows = []
    file_to_root = {}
    for root, ps in package_map.items():
        for p in ps:
            file_to_root[p] = root

    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {p}")
        rows.append(read_npz_record(p, file_to_root[p]))

    inv = pd.DataFrame(rows)
    inv.to_csv(outdir / "01_methanesat_sample_inventory.csv", index=False)

    good = inv[inv.status.eq("PASS")].copy()
    good["label"] = pd.to_numeric(good["label"], errors="coerce")

    diag = label_diagnostics(good)
    diag.to_csv(outdir / "02_label_diagnostics.csv", index=False)

    scene_diag = scene_pair_diagnostics(good)
    scene_diag.to_csv(outdir / "03_scene_pair_diagnostics.csv", index=False)

    schema = array_schema_summary(good)
    schema.to_csv(outdir / "04_array_schema_summary.csv", index=False)

    n_pos = int((good.label == 1).sum())
    n_neg = int((good.label == 0).sum())
    n_unlab = int(good.label.isna().sum())

    scenes = good["scene_id"].fillna("").astype(str)
    scenes_nonempty = scenes[scenes.str.len() > 0]
    unique_scenes = int(scenes_nonempty.nunique())

    mixed = 0
    pos_scenes = set()
    neg_scenes = set()
    if unique_scenes:
        for sid, g in good[good.scene_id.fillna("").astype(str).str.len() > 0].groupby("scene_id"):
            labs = set(pd.to_numeric(g.label, errors="coerce").dropna().astype(int))
            if 1 in labs:
                pos_scenes.add(sid)
            if 0 in labs:
                neg_scenes.add(sid)
            if labs == {0, 1}:
                mixed += 1

    distance = pd.to_numeric(
        good.loc[good.label == 0, "nearest_source_distance_km"],
        errors="coerce",
    ).dropna()

    lines = []
    lines.append("# MethaneSAT confounding audit")
    lines.append("")
    lines.append("## Actual files discovered")
    lines.append(f"- Roots scanned: {len(roots)}")
    lines.append(f"- MethaneSAT NPZ files discovered: {len(files)}")
    lines.append(f"- Successful NPZ reads: {len(good)}")
    lines.append(f"- Failed NPZ reads: {int((inv.status != 'PASS').sum())}")
    lines.append(f"- Positive labels: {n_pos}")
    lines.append(f"- Negative labels: {n_neg}")
    lines.append(f"- Unresolved labels: {n_unlab}")
    lines.append("")
    lines.append("## Scene/collection structure")
    lines.append(f"- Unique resolved scenes/collections: {unique_scenes}")
    lines.append(f"- Positive scenes: {len(pos_scenes)}")
    lines.append(f"- Negative scenes: {len(neg_scenes)}")
    lines.append(f"- Mixed-label scenes containing both positive and negative: {mixed}")

    if len(distance):
        lines.append("")
        lines.append("## Negative distance metadata")
        lines.append(f"- Negative rows with distance metadata: {len(distance)}")
        lines.append(f"- Min km: {distance.min():.3f}")
        lines.append(f"- Median km: {distance.median():.3f}")
        lines.append(f"- Max km: {distance.max():.3f}")

    lines.append("")
    lines.append("## Strongest global label diagnostics")
    if len(diag):
        for _, z in diag.sort_values(
            ["orientation_free_auc", "mannwhitney_p"],
            ascending=[False, True],
        ).head(8).iterrows():
            lines.append(
                f"- {z.feature}: raw AUC={z.raw_auc_positive_high:.3f}, "
                f"orientation-free={z.orientation_free_auc:.3f}, "
                f"p={z.mannwhitney_p:.3g}"
            )

    lines.append("")
    lines.append("## Same-scene / scene-demeaned diagnostics")
    if len(scene_diag):
        for _, z in scene_diag.sort_values(
            "scene_demeaned_orientation_free_auc",
            ascending=False,
        ).iterrows():
            lines.append(
                f"- {z.feature}: mixed scenes={int(z.mixed_label_scenes)}, "
                f"scene-demeaned raw AUC={z.scene_demeaned_raw_auc:.3f}, "
                f"orientation-free={z.scene_demeaned_orientation_free_auc:.3f}, "
                f"positive>negative scenes={z.fraction_scenes_positive_gt_negative:.3f}, "
                f"Wilcoxon p={z.wilcoxon_p:.3g}"
            )
    else:
        lines.append(
            "- No reliable mixed-label scene comparison was possible from the resolved metadata."
        )

    lines.append("")
    lines.append("## How to interpret")
    lines.append(
        "- If GLOBAL background features (e.g. global_median/ring_median) separate labels "
        "but center-minus-ring / scene-demeaned features do not, background or scene "
        "confounding is likely."
    )
    lines.append(
        "- If center-localized features remain strong after scene control, that supports "
        "a localized methane signal rather than a scene/background shortcut."
    )
    lines.append(
        "- If positives and negatives do not share scenes, the next dataset step should be "
        "same-scene hard-negative construction before training."
    )

    (outdir / "SUMMARY_METHANESAT_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("\nDONE:", outdir)
    print("Upload these files:")
    for fn in [
        "SUMMARY_METHANESAT_AUDIT.md",
        "00_detected_packages.csv",
        "01_methanesat_sample_inventory.csv",
        "02_label_diagnostics.csv",
        "03_scene_pair_diagnostics.csv",
        "04_array_schema_summary.csv",
    ]:
        print(" ", outdir / fn)


if __name__ == "__main__":
    main()
