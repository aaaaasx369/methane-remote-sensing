#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except Exception as exc:
    raise SystemExit(
        "缺少 rasterio。請先執行：\n"
        "python -m pip install rasterio pandas numpy scipy matplotlib\n\n"
        f"原始錯誤：{exc}"
    )

try:
    from scipy.stats import rankdata, mannwhitneyu
except Exception as exc:
    raise SystemExit(
        "缺少 scipy。請先執行：\n"
        "python -m pip install scipy\n\n"
        f"原始錯誤：{exc}"
    )

import matplotlib.pyplot as plt

TIFF_EXTS = {".tif", ".tiff"}
CSV_EXTS = {".csv"}
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".tgz", ".7z"}

DEFAULT_SKIP = {
    ".git", ".svn", ".hg", "__pycache__", "node_modules",
    ".venv", "venv", "env", ".conda", "miniconda3", "anaconda3",
    "Library", "Applications", ".Trash",
}

STANDARD_S2_12 = [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B9", "B11", "B12",
]

PATH_ALIASES = {
    "t0": ["s2_0_path", "s2_t0_path", "t0_path", "t_0_path", "sentinel2_t0_path"],
    "t90": ["s2_90_path", "s2_t90_path", "t90_path", "t_90_path", "sentinel2_t90_path"],
    "t360": ["s2_360_path", "s2_t360_path", "t360_path", "t_360_path", "sentinel2_t360_path"],
}
LABEL_ALIASES = ["label", "true_label", "physical_release_gt", "recommended_label", "strict_label"]
SITE_ALIASES = ["site", "site_id", "site_normalized", "facility", "facility_id"]
ID_ALIASES = ["id", "sample_id", "event_id", "record_id", "scene_id", "external_eval_id"]
SENSOR_ALIASES = ["sensor", "satellite", "platform", "instrument"]
PROB_ALIASES = ["probability_positive", "prob_positive", "plume_probability", "score_positive"]

OTHER_SENSOR_TOKENS = (
    "emit", "landsat", "methanesat", "aviris", "tanager", "carbonmapper",
    "carbon_mapper", "prisma", "enmap",
)


def ncol(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def txt(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def num(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def norm_label(x):
    s = txt(x).lower()
    if s in {"1", "1.0", "true", "yes", "positive", "pos", "plume", "release"}:
        return 1.0
    if s in {"0", "0.0", "false", "no", "negative", "neg", "no_plume", "no plume", "no_release"}:
        return 0.0
    try:
        v = float(s)
        return v if v in (0.0, 1.0) else np.nan
    except Exception:
        return np.nan


def first_col(columns, aliases):
    m = {ncol(c): c for c in columns}
    for a in aliases:
        if ncol(a) in m:
            return m[ncol(a)]
    return None


def path_cols(columns):
    m = {ncol(c): c for c in columns}
    out = {}
    for slot, aliases in PATH_ALIASES.items():
        found = next((m[ncol(a)] for a in aliases if ncol(a) in m), None)
        if found is None:
            # conservative fallback for generic t0/t90/t360 path names
            pats = {
                "t0": [r"(^|_)t_?0(_|$).*path"],
                "t90": [r"(^|_)t_?90(_|$).*path"],
                "t360": [r"(^|_)t_?360(_|$).*path"],
            }[slot]
            found = next((orig for nc, orig in m.items() if any(re.search(p, nc) for p in pats)), None)
        if found is None:
            return None
        out[slot] = found
    return out


def iter_files(roots, skip_dirs, include_hidden=False):
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.exists():
            print(f"[WARN] root 不存在: {root}", file=sys.stderr)
            continue
        if root.is_file():
            yield root
            continue
        for cur, dirs, files in os.walk(root, topdown=True):
            dirs[:] = [
                d for d in dirs
                if d not in skip_dirs and (include_hidden or not d.startswith("."))
            ]
            for f in files:
                if not include_hidden and f.startswith("."):
                    continue
                yield Path(cur) / f


def build_inventory(roots, outdir, skip_dirs, include_hidden):
    rows = []
    visited = 0
    for p in iter_files(roots, skip_dirs, include_hidden):
        visited += 1
        ext = p.suffix.lower()
        if ext not in TIFF_EXTS | CSV_EXTS | ARCHIVE_EXTS | {".npz", ".npy", ".json", ".geojson"}:
            continue
        try:
            st = p.stat()
            rows.append({
                "path": str(p.resolve()),
                "name": p.name,
                "ext": ext,
                "size_bytes": st.st_size,
                "mtime_utc": pd.Timestamp(st.st_mtime, unit="s", tz="UTC").isoformat(),
            })
        except Exception:
            rows.append({"path": str(p), "name": p.name, "ext": ext, "size_bytes": np.nan, "mtime_utc": ""})
        if visited % 20000 == 0:
            print(f"[scan] visited {visited:,} files")
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "00_file_inventory.csv", index=False)
    return df


class Resolver:
    def __init__(self, tiffs, roots):
        self.roots = [Path(r).expanduser().resolve() for r in roots]
        self.by_name = defaultdict(list)
        self.by_s2 = defaultdict(list)
        self.by_s3 = defaultdict(list)
        for p in tiffs:
            p = Path(p).resolve()
            self.by_name[p.name].append(p)
            if len(p.parts) >= 2:
                self.by_s2["/".join(p.parts[-2:])].append(p)
            if len(p.parts) >= 3:
                self.by_s3["/".join(p.parts[-3:])].append(p)

    def resolve(self, raw, manifest=None):
        s = os.path.expandvars(os.path.expanduser(txt(raw)))
        if not s:
            return "", "empty"
        p = Path(s)
        if p.exists():
            return str(p.resolve()), "exact"
        if manifest is not None and not p.is_absolute():
            q = Path(manifest).parent / p
            if q.exists():
                return str(q.resolve()), "relative_manifest"
        if not p.is_absolute():
            for root in self.roots:
                q = root / p
                if q.exists():
                    return str(q.resolve()), "relative_root"
        if len(p.parts) >= 3:
            hits = self.by_s3.get("/".join(p.parts[-3:]), [])
            if len(hits) == 1:
                return str(hits[0]), "suffix3"
        if len(p.parts) >= 2:
            hits = self.by_s2.get("/".join(p.parts[-2:]), [])
            if len(hits) == 1:
                return str(hits[0]), "suffix2"
        hits = self.by_name.get(p.name, [])
        if len(hits) == 1:
            return str(hits[0]), "basename"
        if len(hits) > 1:
            return "", f"ambiguous_basename_{len(hits)}"
        return "", "not_found"


def discover_manifests(inv, resolver, outdir):
    audit = []
    samples = []
    for pstr in inv.loc[inv["ext"].eq(".csv"), "path"].tolist():
        p = Path(pstr)
        try:
            cols = list(pd.read_csv(p, nrows=0, low_memory=False).columns)
        except Exception as exc:
            audit.append({"path": str(p), "candidate": False, "reason": f"header_error:{type(exc).__name__}"})
            continue
        pc = path_cols(cols)
        audit.append({
            "path": str(p),
            "candidate": bool(pc),
            "reason": "complete_temporal_paths" if pc else "no_complete_t0_t90_t360_columns",
            "columns": json.dumps(cols[:200], ensure_ascii=False),
        })
        if not pc:
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as exc:
            continue

        idc = first_col(df.columns, ID_ALIASES)
        lc = first_col(df.columns, LABEL_ALIASES)
        sc = first_col(df.columns, SITE_ALIASES)
        sensc = first_col(df.columns, SENSOR_ALIASES)
        probc = first_col(df.columns, PROB_ALIASES)

        for i, r in df.iterrows():
            sensor_text = txt(r.get(sensc)) if sensc else ""
            path_text = " ".join(txt(r.get(pc[s])) for s in ["t0", "t90", "t360"]).lower()
            combined = (sensor_text + " " + path_text).lower()
            has_s2_evidence = any(k in combined for k in ["sentinel", "s2_", "/s2", "s2/"])
            has_other_sensor = any(k in combined for k in OTHER_SENSOR_TOKENS)
            if has_other_sensor and not has_s2_evidence:
                continue

            p0, m0 = resolver.resolve(r.get(pc["t0"]), p)
            p90, m90 = resolver.resolve(r.get(pc["t90"]), p)
            p360, m360 = resolver.resolve(r.get(pc["t360"]), p)
            sid = txt(r.get(idc)) if idc else ""
            if not sid:
                sid = f"{p.stem}:{i}"
            samples.append({
                "source_kind": "manifest",
                "source_manifest": str(p),
                "source_row": i,
                "sample_id": sid,
                "site": txt(r.get(sc)) if sc else "",
                "label": norm_label(r.get(lc)) if lc else np.nan,
                "sensor_text": sensor_text,
                "model_probability_positive": num(r.get(probc)) if probc else np.nan,
                "t0_path": p0, "t90_path": p90, "t360_path": p360,
                "t0_resolve": m0, "t90_resolve": m90, "t360_resolve": m360,
                "t0_raw": txt(r.get(pc["t0"])),
                "t90_raw": txt(r.get(pc["t90"])),
                "t360_raw": txt(r.get(pc["t360"])),
            })
    adf = pd.DataFrame(audit)
    sdf = pd.DataFrame(samples)
    adf.to_csv(outdir / "01_candidate_manifests.csv", index=False)
    sdf.to_csv(outdir / "02a_manifest_temporal_samples.csv", index=False)
    return adf, sdf


SLOT_PATTERNS = {
    "t0": [re.compile(r"(?i)(?:^|[_\-.])t0(?:[_\-.]|$)"), re.compile(r"(?i)(?:^|[_\-.])t_0(?:[_\-.]|$)")],
    "t90": [re.compile(r"(?i)(?:^|[_\-.])t90(?:[_\-.]|$)"), re.compile(r"(?i)(?:^|[_\-.])t_90(?:[_\-.]|$)"), re.compile(r"(?i)tminus90")],
    "t360": [re.compile(r"(?i)(?:^|[_\-.])t360(?:[_\-.]|$)"), re.compile(r"(?i)(?:^|[_\-.])t_360(?:[_\-.]|$)"), re.compile(r"(?i)tminus360")],
}


def slot_of(name):
    stem = Path(name).stem
    for slot, pats in SLOT_PATTERNS.items():
        if any(p.search(stem) for p in pats):
            return slot
    return None


def strip_slot(name):
    x = Path(name).stem
    for pats in SLOT_PATTERNS.values():
        for p in pats:
            x = p.sub("_SLOT_", x)
    return re.sub(r"_+", "_", x).strip("_")


def discover_filename_triplets(tiffs, outdir):
    groups = defaultdict(dict)
    for p in map(Path, tiffs):
        s = slot_of(p.name)
        if not s:
            continue
        groups[(str(p.parent.resolve()), strip_slot(p.name))][s] = p.resolve()
    rows = []
    for (parent, base), g in groups.items():
        if not {"t0", "t90", "t360"}.issubset(g):
            continue
        combined = " ".join(str(g[s]).lower() for s in ["t0", "t90", "t360"])
        has_s2 = any(k in combined for k in ["sentinel", "s2_", "/s2", "s2/"])
        has_other = any(k in combined for k in OTHER_SENSOR_TOKENS)
        if has_other and not has_s2:
            continue
        rows.append({
            "source_kind": "filename_triplet", "source_manifest": "", "source_row": np.nan,
            "sample_id": base, "site": "", "label": np.nan, "sensor_text": "",
            "model_probability_positive": np.nan,
            "t0_path": str(g["t0"]), "t90_path": str(g["t90"]), "t360_path": str(g["t360"]),
            "t0_resolve": "filename", "t90_resolve": "filename", "t360_resolve": "filename",
            "t0_raw": str(g["t0"]), "t90_raw": str(g["t90"]), "t360_raw": str(g["t360"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "02b_filename_temporal_triplets.csv", index=False)
    return df


def combine_samples(a, b, outdir):
    frames = [x for x in [a, b] if x is not None and len(x)]
    if not frames:
        out = pd.DataFrame()
        out.to_csv(outdir / "02_temporal_triplets_all.csv", index=False)
        return out
    x = pd.concat(frames, ignore_index=True, sort=False)
    x["_key"] = x.apply(lambda r: "||".join([txt(r.get("t0_path")), txt(r.get("t90_path")), txt(r.get("t360_path"))]) if all(txt(r.get(c)) for c in ["t0_path","t90_path","t360_path"]) else f"unresolved:{r.name}", axis=1)
    merged = []
    for _, g in x.groupby("_key", sort=False):
        g = g.copy()
        g["_p"] = g["source_kind"].eq("manifest").astype(int)
        g = g.sort_values("_p", ascending=False)
        d = g.iloc[0].to_dict()
        for c in ["site", "label", "sensor_text", "model_probability_positive"]:
            for v in g[c].tolist():
                if c in ["label", "model_probability_positive"]:
                    if not pd.isna(v):
                        d[c] = v; break
                elif txt(v):
                    d[c] = v; break
        d["duplicate_source_count"] = len(g)
        merged.append(d)
    out = pd.DataFrame(merged).drop(columns=["_key", "_p"], errors="ignore")
    out.to_csv(outdir / "02_temporal_triplets_all.csv", index=False)
    return out


def band_name(x):
    s = txt(x).upper().replace(" ", "").replace("BAND", "B")
    m = re.fullmatch(r"B0*(\d+)(A?)", s)
    return f"B{int(m.group(1))}{m.group(2)}" if m else s


def band_map(ds, custom_order, assume_12):
    needed = {"B4", "B8", "B11", "B12"}
    m = {}
    for i, d in enumerate(ds.descriptions or [], 1):
        b = band_name(d)
        if b:
            m[b] = i
    if needed.issubset(m):
        return m, "geotiff_descriptions"
    tags_map = {}
    for i in range(1, ds.count + 1):
        try:
            tags = ds.tags(i)
        except Exception:
            tags = {}
        for k in ["name", "NAME", "band_name", "BAND_NAME", "description", "DESCRIPTION"]:
            if k in tags:
                b = band_name(tags[k])
                if b:
                    tags_map[b] = i
    if needed.issubset(tags_map):
        return tags_map, "geotiff_tags"
    if custom_order and len(custom_order) == ds.count:
        m = {band_name(b): i+1 for i, b in enumerate(custom_order)}
        if needed.issubset(m):
            return m, "user_band_order"
    if assume_12 and ds.count == 12:
        return {b: i+1 for i, b in enumerate(STANDARD_S2_12)}, "user_opted_standard_s2_12"
    return {}, "unresolved"


def raster_info(path, custom_order, assume_12):
    if not path or not Path(path).exists():
        return {"ok": False, "reason": "missing_file"}
    try:
        with rasterio.open(path) as ds:
            bm, src = band_map(ds, custom_order, assume_12)
            return {
                "ok": True, "reason": "ok", "count": ds.count, "height": ds.height, "width": ds.width,
                "crs": str(ds.crs) if ds.crs else "", "band_map_source": src,
                "band_map_json": json.dumps(bm),
                "required_bands": all(b in bm for b in ["B4", "B8", "B11", "B12"]),
            }
    except Exception as exc:
        return {"ok": False, "reason": f"open_error:{type(exc).__name__}:{exc}"}


def validate(samples, outdir, custom_order, assume_12):
    rows = []
    for _, r in samples.iterrows():
        d = {"sample_id": r["sample_id"], "site": r.get("site", ""), "label": r.get("label", np.nan)}
        infos = {}
        for s in ["t0", "t90", "t360"]:
            infos[s] = raster_info(txt(r.get(f"{s}_path")), custom_order, assume_12)
            for k, v in infos[s].items():
                d[f"{s}_{k}"] = v
        d["triplet_openable"] = all(infos[s].get("ok", False) for s in infos)
        d["required_bands_resolved"] = all(infos[s].get("required_bands", False) for s in infos)
        d["analysis_status"] = "READY" if d["triplet_openable"] and d["required_bands_resolved"] else "EXCLUDE"
        if not d["triplet_openable"]:
            d["analysis_reason"] = "unopenable_or_missing_raster"
        elif not d["required_bands_resolved"]:
            d["analysis_reason"] = "band_mapping_unresolved"
        else:
            d["analysis_reason"] = ""
        rows.append(d)
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "03_triplet_validation.csv", index=False)
    return out


def read_band(path, bname, custom_order, assume_12):
    with rasterio.open(path) as ds:
        bm, src = band_map(ds, custom_order, assume_12)
        if bname not in bm:
            raise ValueError(f"cannot map {bname}: {path}")
        a = ds.read(bm[bname], masked=True).astype("float64")
        if np.ma.isMaskedArray(a):
            a = a.filled(np.nan)
        a = np.asarray(a, dtype=float)
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
        return a, ds.transform, ds.crs


def align(src, src_transform, src_crs, dst_shape, dst_transform, dst_crs):
    if src.shape == dst_shape and src_transform == dst_transform and str(src_crs) == str(dst_crs):
        return src
    if src_crs is None or dst_crs is None:
        if src.shape == dst_shape:
            return src
        raise ValueError("grid_mismatch_without_crs")
    dst = np.full(dst_shape, np.nan, dtype=float)
    reproject(source=src, destination=dst, src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=dst_crs, src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    return dst


def load_triplet(r, custom_order, assume_12):
    out = {}
    target = None
    for s in ["t0", "t90", "t360"]:
        out[s] = {}
        meta = None
        for b in ["B4", "B8", "B11", "B12"]:
            a, tr, crs = read_band(r[f"{s}_path"], b, custom_order, assume_12)
            out[s][b] = a
            meta = (a.shape, tr, crs)
        out[s]["meta"] = meta
        if s == "t0":
            target = meta
    shape0, tr0, crs0 = target
    for s in ["t90", "t360"]:
        _, trs, crss = out[s]["meta"]
        for b in ["B4", "B8", "B11", "B12"]:
            out[s][b] = align(out[s][b], trs, crss, shape0, tr0, crs0)
    return out


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if m.sum() < 100:
        return np.nan
    x, y = a[m], b[m]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def ratio(num_, den_):
    out = np.full(num_.shape, np.nan, dtype=float)
    m = np.isfinite(num_) & np.isfinite(den_) & (np.abs(den_) > 1e-12)
    out[m] = num_[m] / den_[m]
    return out


def ndvi(x):
    return ratio(x["B8"] - x["B4"], x["B8"] + x["B4"])


def ndbi(x):
    return ratio(x["B11"] - x["B8"], x["B11"] + x["B8"])


def swir(x):
    out = np.full(x["B11"].shape, np.nan, dtype=float)
    m = np.isfinite(x["B11"]) & np.isfinite(x["B12"]) & (x["B11"] > 0) & (x["B12"] > 0)
    out[m] = np.log((x["B11"][m] + 1e-12) / (x["B12"][m] + 1e-12))
    return out


def arr_stats(a, prefix):
    v = a[np.isfinite(a)]
    if not len(v):
        return {f"{prefix}_{k}": np.nan for k in ["mean", "median", "p90", "p95", "p99", "top5_mean", "abs_mean"]}
    q95 = np.percentile(v, 95)
    top = v[v >= q95]
    return {
        f"{prefix}_mean": float(v.mean()),
        f"{prefix}_median": float(np.median(v)),
        f"{prefix}_p90": float(np.percentile(v, 90)),
        f"{prefix}_p95": float(q95),
        f"{prefix}_p99": float(np.percentile(v, 99)),
        f"{prefix}_top5_mean": float(top.mean()) if len(top) else np.nan,
        f"{prefix}_abs_mean": float(np.abs(v).mean()),
    }


def metrics_for_ref(t0, ref):
    d = {}
    d["b4_corr"] = corr(t0["B4"], ref["B4"])
    n0, nr = ndvi(t0), ndvi(ref)
    b0, br = ndbi(t0), ndbi(ref)
    d["ndvi_t0_median"] = float(np.nanmedian(n0)) if np.isfinite(n0).any() else np.nan
    d["ndvi_ref_median"] = float(np.nanmedian(nr)) if np.isfinite(nr).any() else np.nan
    d["abs_ndvi_median_change"] = abs(d["ndvi_t0_median"] - d["ndvi_ref_median"])
    d["ndbi_t0_median"] = float(np.nanmedian(b0)) if np.isfinite(b0).any() else np.nan
    d["ndbi_ref_median"] = float(np.nanmedian(br)) if np.isfinite(br).any() else np.nan
    d["abs_ndbi_median_change"] = abs(d["ndbi_t0_median"] - d["ndbi_ref_median"])
    d.update(arr_stats(n0 - nr, "delta_ndvi"))
    d.update(arr_stats(b0 - br, "delta_ndbi"))
    d.update(arr_stats(swir(t0) - swir(ref), "swir_delta"))
    return d


def process(samples, validation, outdir, custom_order, assume_12):
    ready = set(validation.loc[validation["analysis_status"].eq("READY"), "sample_id"].astype(str))
    rows = []
    for i, r in samples.iterrows():
        sid = str(r["sample_id"])
        if sid not in ready:
            continue
        print(f"[pixel] {i+1}/{len(samples)} {sid}")
        base = {
            "sample_id": sid, "site": r.get("site", ""), "label": r.get("label", np.nan),
            "sensor_text": r.get("sensor_text", ""),
            "model_probability_positive": r.get("model_probability_positive", np.nan),
            "source_kind": r.get("source_kind", ""), "source_manifest": r.get("source_manifest", ""),
            "t0_path": r.get("t0_path", ""), "t90_path": r.get("t90_path", ""), "t360_path": r.get("t360_path", ""),
        }
        try:
            x = load_triplet(r, custom_order, assume_12)
            m90 = metrics_for_ref(x["t0"], x["t90"])
            m360 = metrics_for_ref(x["t0"], x["t360"])
            c90, c360 = m90["b4_corr"], m360["b4_corr"]
            if np.isfinite(c90) and np.isfinite(c360):
                best = "t90" if c90 >= c360 else "t360"
            elif np.isfinite(c90):
                best = "t90"
            elif np.isfinite(c360):
                best = "t360"
            else:
                raise ValueError("both B4 correlations invalid")
            mbest = m90 if best == "t90" else m360
            d = dict(base)
            d.update({
                "analysis_status": "PASS", "error": "", "best_reference": best,
                "best_b4_corr": mbest["b4_corr"],
                "best_corr_ge_0p7": bool(mbest["b4_corr"] >= 0.7),
                "best_corr_gain_vs_t90": mbest["b4_corr"] - c90 if np.isfinite(c90) else np.nan,
            })
            for strat, mm in [("t90", m90), ("t360", m360), ("best", mbest)]:
                for k, v in mm.items():
                    d[f"{strat}_{k}"] = v
            rows.append(d)
        except Exception as exc:
            d = dict(base)
            d.update({"analysis_status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
            rows.append(d)
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "04_reference_metrics_per_sample.csv", index=False)
    return out


def auc_rank(y, score):
    m = np.isfinite(y) & np.isfinite(score)
    y, score = y[m].astype(int), score[m]
    if len(np.unique(y)) < 2:
        return np.nan
    n1, n0 = np.sum(y == 1), np.sum(y == 0)
    ranks = rankdata(score, method="average")
    u = ranks[y == 1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def ols_r2(y, X):
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[m], X[m]
    if len(y) < 3 or np.std(y) == 0:
        return np.nan
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ssr = np.sum((y - pred)**2)
    sst = np.sum((y - y.mean())**2)
    return float(1 - ssr/sst) if sst > 0 else np.nan


def stats(metrics, outdir):
    if not len(metrics):
        pd.DataFrame().to_csv(outdir / "05_label_feature_summary.csv", index=False)
        pd.DataFrame().to_csv(outdir / "06_within_site_summary.csv", index=False)
        return pd.DataFrame(), pd.DataFrame()
    g = metrics[metrics["analysis_status"].eq("PASS")].copy()
    g["label_num"] = pd.to_numeric(g["label"], errors="coerce")
    features = [
        "t90_b4_corr", "t360_b4_corr", "best_b4_corr",
        "t90_abs_ndvi_median_change", "t360_abs_ndvi_median_change", "best_abs_ndvi_median_change",
        "t90_abs_ndbi_median_change", "t360_abs_ndbi_median_change", "best_abs_ndbi_median_change",
        "t90_swir_delta_p95", "t360_swir_delta_p95", "best_swir_delta_p95",
        "t90_swir_delta_top5_mean", "t360_swir_delta_top5_mean", "best_swir_delta_top5_mean",
        "t90_swir_delta_abs_mean", "t360_swir_delta_abs_mean", "best_swir_delta_abs_mean",
    ]
    rows = []
    for f in [x for x in features if x in g.columns]:
        d = g[["label_num", "site", f]].copy()
        d[f] = pd.to_numeric(d[f], errors="coerce")
        d = d.dropna(subset=["label_num", f])
        if not len(d):
            continue
        pos = d.loc[d.label_num.eq(1), f].to_numpy()
        neg = d.loc[d.label_num.eq(0), f].to_numpy()
        auc = auc_rank(d.label_num.to_numpy(), d[f].to_numpy())
        try:
            p = float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue) if len(pos) and len(neg) else np.nan
        except Exception:
            p = np.nan
        siteX = pd.get_dummies(d["site"].fillna("").astype(str), drop_first=True).to_numpy(dtype=float)
        labelX = d[["label_num"]].to_numpy(dtype=float)
        bothX = np.column_stack([siteX, labelX]) if siteX.shape[1] else labelX
        y = d[f].to_numpy(dtype=float)
        r2_label = ols_r2(y, labelX)
        r2_site = ols_r2(y, siteX) if siteX.shape[1] else np.nan
        r2_both = ols_r2(y, bothX)
        rows.append({
            "feature": f, "n": len(d), "n_positive": len(pos), "n_negative": len(neg),
            "positive_median": np.median(pos) if len(pos) else np.nan,
            "negative_median": np.median(neg) if len(neg) else np.nan,
            "raw_auc_positive_high": auc,
            "orientation_free_auc": max(auc, 1-auc) if np.isfinite(auc) else np.nan,
            "mannwhitney_p": p,
            "r2_label_only": r2_label, "r2_site_only": r2_site, "r2_site_plus_label": r2_both,
            "incremental_r2_label_after_site": r2_both - r2_site if np.isfinite(r2_both) and np.isfinite(r2_site) else np.nan,
        })
    fs = pd.DataFrame(rows)
    fs.to_csv(outdir / "05_label_feature_summary.csv", index=False)

    wrows = []
    for site, sg in g.groupby("site"):
        if not txt(site):
            continue
        for f in ["best_b4_corr", "best_abs_ndvi_median_change", "best_swir_delta_p95", "best_swir_delta_top5_mean", "best_swir_delta_abs_mean"]:
            if f not in sg.columns:
                continue
            d = sg[["label_num", f]].copy()
            d[f] = pd.to_numeric(d[f], errors="coerce")
            d = d.dropna()
            if not len(d):
                continue
            pos = d.loc[d.label_num.eq(1), f].to_numpy()
            neg = d.loc[d.label_num.eq(0), f].to_numpy()
            auc = auc_rank(d.label_num.to_numpy(), d[f].to_numpy())
            wrows.append({
                "site": site, "feature": f, "n": len(d), "n_positive": len(pos), "n_negative": len(neg),
                "positive_median": np.median(pos) if len(pos) else np.nan,
                "negative_median": np.median(neg) if len(neg) else np.nan,
                "raw_auc_positive_high": auc,
                "orientation_free_auc": max(auc, 1-auc) if np.isfinite(auc) else np.nan,
            })
    ws = pd.DataFrame(wrows)
    ws.to_csv(outdir / "06_within_site_summary.csv", index=False)
    return fs, ws


def strategy_summary(metrics, outdir):
    if not len(metrics):
        pd.DataFrame().to_csv(outdir / "07_reference_strategy_summary.csv", index=False)
        return
    g = metrics[metrics["analysis_status"].eq("PASS")]
    rows = []
    for s in ["t90", "t360", "best"]:
        for m in ["b4_corr", "abs_ndvi_median_change", "abs_ndbi_median_change", "swir_delta_p95", "swir_delta_top5_mean", "swir_delta_abs_mean"]:
            c = f"{s}_{m}"
            if c not in g.columns:
                continue
            x = pd.to_numeric(g[c], errors="coerce")
            rows.append({"strategy": s, "metric": m, "n": x.notna().sum(), "mean": x.mean(), "median": x.median(), "p25": x.quantile(.25), "p75": x.quantile(.75)})
    pd.DataFrame(rows).to_csv(outdir / "07_reference_strategy_summary.csv", index=False)


def plots(metrics, outdir):
    if not len(metrics):
        return
    g = metrics[metrics["analysis_status"].eq("PASS")].copy()
    g["label_num"] = pd.to_numeric(g["label"], errors="coerce")
    pdir = outdir / "plots"
    pdir.mkdir(exist_ok=True)
    for col, name, ylabel in [
        ("best_b4_corr", "01_best_b4_corr_by_label.png", "Best B4 correlation"),
        ("best_swir_delta_p95", "02_best_swir_p95_by_label.png", "Best-reference SWIR temporal proxy p95"),
    ]:
        if col not in g.columns:
            continue
        d = g.dropna(subset=["label_num", col])
        a = d.loc[d.label_num.eq(0), col].to_numpy()
        b = d.loc[d.label_num.eq(1), col].to_numpy()
        if len(a) and len(b):
            plt.figure(figsize=(7,5)); plt.boxplot([a,b], tick_labels=["Negative","Positive"], showfliers=True)
            plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(pdir / name, dpi=180); plt.close()
    if {"t90_b4_corr", "best_b4_corr"}.issubset(g.columns):
        d = g.dropna(subset=["t90_b4_corr", "best_b4_corr"])
        if len(d):
            plt.figure(figsize=(6,6)); plt.scatter(d.t90_b4_corr, d.best_b4_corr, s=18, alpha=.7)
            lo = min(d.t90_b4_corr.min(), d.best_b4_corr.min()); hi = max(d.t90_b4_corr.max(), d.best_b4_corr.max())
            plt.plot([lo,hi],[lo,hi], "--"); plt.xlabel("Fixed t90 B4 corr"); plt.ylabel("Best-reference B4 corr")
            plt.tight_layout(); plt.savefig(pdir / "03_t90_vs_best_corr.png", dpi=180); plt.close()


def write_summary(inv, maudit, samples, validation, metrics, fs, outdir, args):
    L = ["# Actual local-data Sentinel-2 reference audit", "", "本報告只根據本次指定 roots 實際掃到的本機檔案。", ""]
    L += ["## Inventory", f"- Relevant files: {len(inv):,}", f"- TIFFs: {int(inv.ext.isin(TIFF_EXTS).sum()) if len(inv) else 0:,}", f"- Candidate temporal CSVs: {int(maudit.candidate.sum()) if len(maudit) else 0:,}", f"- Temporal triplets discovered: {len(samples):,}", ""]
    if len(validation):
        L += ["## Validation", f"- Openable triplets: {int(validation.triplet_openable.sum()):,}", f"- Required B4/B8/B11/B12 resolved: {int(validation.required_bands_resolved.sum()):,}", f"- Pixel-ready: {int(validation.analysis_status.eq('READY').sum()):,}", ""]
        for k, n in validation.loc[validation.analysis_reason.ne(""), "analysis_reason"].value_counts().items():
            L.append(f"- Excluded {k}: {n}")
        L.append("")
    good = metrics[metrics.analysis_status.eq("PASS")] if len(metrics) else pd.DataFrame()
    if len(good):
        L += ["## Reference selection", f"- Pixel-successful samples: {len(good):,}"]
        for k, n in good.best_reference.value_counts().items():
            L.append(f"- Best reference {k}: {n}")
        L += [f"- Median best B4 corr: {pd.to_numeric(good.best_b4_corr, errors='coerce').median():.4f}", f"- Median gain over fixed t90: {pd.to_numeric(good.best_corr_gain_vs_t90, errors='coerce').median():.4f}", ""]
    if len(fs):
        L.append("## Key label/site diagnostics")
        for f in ["best_b4_corr", "best_abs_ndvi_median_change", "best_swir_delta_p95", "best_swir_delta_top5_mean", "best_swir_delta_abs_mean"]:
            z = fs[fs.feature.eq(f)]
            if len(z):
                r = z.iloc[0]
                L += [f"### {f}", f"- N={int(r.n)}, positive={int(r.n_positive)}, negative={int(r.n_negative)}", f"- raw AUROC positive-high={r.raw_auc_positive_high:.4f}", f"- orientation-free AUROC={r.orientation_free_auc:.4f}", f"- R2 label only={r.r2_label_only:.4f}", f"- R2 site only={r.r2_site_only:.4f}" if pd.notna(r.r2_site_only) else "- R2 site only=NA", f"- incremental R2(label after site)={r.incremental_r2_label_after_site:.4f}" if pd.notna(r.incremental_r2_label_after_site) else "- incremental R2(label after site)=NA", ""]
    L += ["## Interpretation rules", "1. best-reference 若提高 B4 correlation 且降低 |ΔNDVI| / |ΔNDBI|，表示固定 reference 混入 background change。", "2. background-matched 後 SWIR proxy 的 positive/negative separation 若提升，且 within-site 也提升，才比較支持 plume-related signal。", "3. 若 R2(site) 遠高於 R2(label)，且控制 site 後 label incremental R2 很小，表示 site/background confounding 仍主導。", "4. SWIR proxy = temporal change in log(B11/B12)，不是 calibrated CH4 concentration。", "", "## Scan roots"]
    for r in args.roots:
        L.append(f"- {Path(r).expanduser()}")
    L += ["", f"assume_standard_s2_order={args.assume_standard_s2_order}", f"band_order={args.band_order or '(none)'}"]
    (outdir / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True, help="實際要掃描的本機資料夾，可多個；例如 ~/methane_release_project ~/MethaneFuse ~/Downloads")
    ap.add_argument("--out", default="~/methane_reference_full_audit")
    ap.add_argument("--include-hidden", action="store_true")
    ap.add_argument("--skip-dir", action="append", default=sorted(DEFAULT_SKIP))
    ap.add_argument("--assume-standard-s2-order", action="store_true", help="只有加這個 flag 才允許 12-band TIFF 使用標準 S2 12-band order")
    ap.add_argument("--band-order", default="", help='若 TIFF 無 band metadata，可明確指定，例如 "B2,B3,B4,B8,B11,B12"')
    return ap.parse_args()


def main():
    args = parse_args()
    roots = [Path(r).expanduser().resolve() for r in args.roots]
    outdir = Path(args.out).expanduser().resolve(); outdir.mkdir(parents=True, exist_ok=True)
    custom = [band_name(x) for x in args.band_order.split(",") if x.strip()] if args.band_order.strip() else None
    print("Roots:"); [print(" -", r) for r in roots]; print("Output:", outdir)
    inv = build_inventory(roots, outdir, set(args.skip_dir), args.include_hidden)
    tiffs = inv.loc[inv.ext.isin(TIFF_EXTS), "path"].tolist() if len(inv) else []
    resolver = Resolver(tiffs, roots)
    maudit, msamples = discover_manifests(inv, resolver, outdir)
    fsamples = discover_filename_triplets(tiffs, outdir)
    samples = combine_samples(msamples, fsamples, outdir)
    print(f"Discovered triplets: {len(samples):,}")
    if not len(samples):
        write_summary(inv, maudit, samples, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), outdir, args)
        print("沒有找到完整 t0/t90/t360 triplet；請看 00/01 CSV。")
        return
    validation = validate(samples, outdir, custom, args.assume_standard_s2_order)
    print(f"Pixel-ready: {int(validation.analysis_status.eq('READY').sum()):,}/{len(validation):,}")
    if not validation.analysis_status.eq("READY").any():
        write_summary(inv, maudit, samples, validation, pd.DataFrame(), pd.DataFrame(), outdir, args)
        print("沒有可解析 B4/B8/B11/B12 的 triplet。先看 03_triplet_validation.csv。")
        print("若你的 12-band TIFF 確定為標準 S2 order，再重跑加 --assume-standard-s2-order")
        return
    m = process(samples, validation, outdir, custom, args.assume_standard_s2_order)
    fs, ws = stats(m, outdir)
    strategy_summary(m, outdir)
    plots(m, outdir)
    write_summary(inv, maudit, samples, validation, m, fs, outdir, args)
    print("\nDONE")
    for name in ["SUMMARY.md", "03_triplet_validation.csv", "04_reference_metrics_per_sample.csv", "05_label_feature_summary.csv", "06_within_site_summary.csv", "07_reference_strategy_summary.csv"]:
        print(outdir / name)


if __name__ == "__main__":
    main()
