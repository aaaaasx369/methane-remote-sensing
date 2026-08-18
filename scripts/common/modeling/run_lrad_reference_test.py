#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_lrad_reference_test.py

Next experiment after the historical-reference audit.

Uses the ACTUAL local TIFFs referenced by:
  ~/methane_reference_full_audit/04_reference_metrics_per_sample.csv

Pipeline
--------
1. Remove fake temporal aliases (t0/t90/t360 same path).
2. Deduplicate copied datasets / repeated Sentinel-2 scenes.
3. Keep all unique real temporal samples for QA.
4. Use the five-site subset (75 = 15 positive + 60 negative in the current audit)
   as the PRIMARY binary benchmark.
5. Read B3/B4/B8/B11/B12 from actual TIFFs.
6. Apply LRAD-style artifact masking:
      flare: B11 >= 1 AND B12 >= 1
      dark/smoke/shadow: B3 <= 5th percentile
      NDWI >= 0.2
      NDVI >= 0.3
      NDBI >= 0.2
      NDSI >= 0.42
7. Evaluate both:
      A) log(B11/B12) temporal difference (diagnostic proxy)
      B) MBSP-like differential reflectance factor:
            (c*B12 - B11)/B12
         with c fitted by least squares on common valid pixels
8. Compare UNMASKED vs LRAD-MASKED and fixed t90 vs best reference.
9. Report global + within-site AUROC and site-vs-label R².

Important
---------
This does NOT produce calibrated methane concentration.
It tests whether LRAD-style masking reduces artifact/site dependence and improves
positive-vs-negative spectral separation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation
from scipy.stats import rankdata, mannwhitneyu


STANDARD_S2_12 = [
    "B1","B2","B3","B4","B5","B6",
    "B7","B8","B8A","B9","B11","B12"
]
NEEDED = ["B3","B4","B8","B11","B12"]
SCENE_RE = re.compile(r"(\d{8}T\d{6}_\d{8}T\d{6}_T\d{2}[A-Z]{3})")


def safe_text(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def normalize_band_name(x):
    s = safe_text(x).upper().replace(" ", "").replace("BAND", "B")
    m = re.fullmatch(r"B0*(\d+)(A?)", s)
    if m:
        return f"B{int(m.group(1))}{m.group(2)}"
    return s


def canonical_site(x):
    s = safe_text(x)
    low = s.lower()
    if "casa_grande" in low or "casa grande" in low:
        return "Casa_Grande"
    if "ehrenberg" in low:
        return "Ehrenberg"
    return s


def dataset_group(path):
    s = safe_text(path)
    if "s2_12band_five_site_zero_shot" in s:
        return "five_site"
    if "s2_12band_exact_external" in s:
        return "exact_external"
    if "s2_12band_methaneair_p1_final16" in s:
        return "methaneair_p1_final16"
    if "s2_12band_methaneair_p1/" in s:
        return "methaneair_p1"
    if "five_site_loso/images" in s:
        return "loso_smoke"
    if "s2_12band_480m_exact3" in s:
        return "exact3_smoke"
    if "s2_12band_smoke" in s:
        return "project_smoke"
    return "other"


def scene_token(path):
    m = SCENE_RE.search(safe_text(path))
    return m.group(1) if m else ""


def clean_unique(df):
    df = df[df["analysis_status"].eq("PASS")].copy()
    df["dataset_group"] = df["t0_path"].map(dataset_group)
    df["canonical_site"] = df["site"].map(canonical_site)
    df["scene_token"] = df["t0_path"].map(scene_token)

    fake = (
        df["t0_path"].eq(df["t90_path"]) |
        df["t0_path"].eq(df["t360_path"]) |
        df["t90_path"].eq(df["t360_path"])
    )
    df = df[~fake].copy()

    priority = {
        "five_site": 0,
        "exact_external": 1,
        "methaneair_p1": 2,
        "methaneair_p1_final16": 3,
        "other": 4,
    }
    df["_priority"] = df["dataset_group"].map(priority).fillna(9)

    # Remove copied versions with identical sample_id.
    df = (
        df.sort_values(["_priority","t0_path"])
          .drop_duplicates("sample_id", keep="first")
          .copy()
    )

    # Remove same physical Sentinel acquisition appearing in multiple evaluation sets.
    has_scene = df["scene_token"].ne("")
    a = (
        df[has_scene]
        .sort_values(["_priority","t0_path"])
        .drop_duplicates(["canonical_site","scene_token"], keep="first")
    )
    b = df[~has_scene]
    out = pd.concat([a,b], ignore_index=True)
    return out.drop(columns=["_priority"], errors="ignore")


def band_map(ds, assume_standard):
    mapping = {}
    for i, desc in enumerate(ds.descriptions or [], 1):
        b = normalize_band_name(desc)
        if b:
            mapping[b] = i
    if set(NEEDED).issubset(mapping):
        return mapping, "descriptions"

    for i in range(1, ds.count+1):
        tags = ds.tags(i)
        for k in ["name","NAME","band_name","BAND_NAME","description","DESCRIPTION"]:
            if k in tags:
                b = normalize_band_name(tags[k])
                if b:
                    mapping[b] = i
    if set(NEEDED).issubset(mapping):
        return mapping, "tags"

    if assume_standard and ds.count == 12:
        return {b:i+1 for i,b in enumerate(STANDARD_S2_12)}, "standard_s2_12_opt_in"

    raise ValueError(
        f"Cannot resolve {NEEDED}; raster has {ds.count} bands. "
        "Use --assume-standard-s2-order only if you verified the 12-band order."
    )


def read_cube(path, assume_standard):
    with rasterio.open(path) as ds:
        bm, source = band_map(ds, assume_standard)
        arrays = {}
        for b in NEEDED:
            a = ds.read(bm[b], masked=True).astype("float64")
            if np.ma.isMaskedArray(a):
                a = a.filled(np.nan)
            a = np.asarray(a, dtype=float)
            if ds.nodata is not None:
                a[a == ds.nodata] = np.nan
            arrays[b] = a
        return {
            "bands": arrays,
            "transform": ds.transform,
            "crs": ds.crs,
            "shape": (ds.height, ds.width),
            "band_map_source": source,
        }


def align_cube(src, target):
    if (
        src["shape"] == target["shape"] and
        src["transform"] == target["transform"] and
        str(src["crs"]) == str(target["crs"])
    ):
        return src

    if src["crs"] is None or target["crs"] is None:
        raise ValueError("Grid mismatch and CRS missing")

    out = {
        "bands": {},
        "transform": target["transform"],
        "crs": target["crs"],
        "shape": target["shape"],
        "band_map_source": src["band_map_source"],
    }

    for b,a in src["bands"].items():
        dst = np.full(target["shape"], np.nan, dtype=float)
        reproject(
            source=a,
            destination=dst,
            src_transform=src["transform"],
            src_crs=src["crs"],
            dst_transform=target["transform"],
            dst_crs=target["crs"],
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        out["bands"][b] = dst
    return out


def infer_scale(cube, mode):
    vals = []
    for b in NEEDED:
        a = cube["bands"][b]
        v = a[np.isfinite(a) & (a > 0)]
        if len(v):
            vals.append(np.percentile(v, 99))
    p99 = np.median(vals) if vals else np.nan

    if mode != "auto":
        return float(mode), f"user:{mode}"

    # Sentinel-2 products are commonly stored either as reflectance 0..1
    # or integer-like reflectance with a 10000 scale. Record what was chosen.
    if np.isfinite(p99) and p99 > 2:
        return 10000.0, f"auto_10000_from_p99={p99:.3f}"
    return 1.0, f"auto_1_from_p99={p99:.3f}"


def scaled_bands(cube, scale):
    return {b: cube["bands"][b] / scale for b in NEEDED}


def ratio(num, den):
    out = np.full(num.shape, np.nan, dtype=float)
    m = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12)
    out[m] = num[m] / den[m]
    return out


def lrad_bad_mask(b, dilation_iterations):
    b3,b4,b8,b11,b12 = (b[x] for x in ["B3","B4","B8","B11","B12"])

    finite = (
        np.isfinite(b3) & np.isfinite(b4) & np.isfinite(b8) &
        np.isfinite(b11) & np.isfinite(b12)
    )

    valid_b3 = b3[np.isfinite(b3)]
    q5 = np.percentile(valid_b3, 5) if len(valid_b3) else np.nan

    ndwi = ratio(b3-b8, b3+b8)
    ndvi = ratio(b8-b4, b8+b4)
    ndbi = ratio(b11-b8, b11+b8)
    ndsi = ratio(b3-b11, b3+b11)

    flare = (b11 >= 1.0) & (b12 >= 1.0)
    dark = b3 <= q5
    water = ndwi >= 0.2
    vegetation = ndvi >= 0.3
    built = ndbi >= 0.2
    snow_cloud_like = ndsi >= 0.42

    bad = (
        (~finite) | flare | dark | water |
        vegetation | built | snow_cloud_like
    )

    if dilation_iterations > 0:
        bad = binary_dilation(
            bad,
            structure=np.ones((3,3), dtype=bool),
            iterations=dilation_iterations,
        )

    components = {
        "q5_b3": q5,
        "flare_fraction": float(np.mean(flare)),
        "dark_fraction": float(np.mean(dark)),
        "water_fraction": float(np.mean(water)),
        "vegetation_fraction": float(np.mean(vegetation)),
        "built_fraction": float(np.mean(built)),
        "snow_cloud_fraction": float(np.mean(snow_cloud_like)),
        "total_bad_fraction": float(np.mean(bad)),
    }
    return bad, components


def fit_c(b11, b12, mask):
    m = (
        mask &
        np.isfinite(b11) & np.isfinite(b12) &
        (b11 > 0) & (b12 > 0)
    )
    if m.sum() < 50:
        return np.nan
    x = b12[m]
    y = b11[m]
    den = np.sum(x*x)
    if den <= 0:
        return np.nan
    return float(np.sum(x*y)/den)


def mbsp_factor(b11, b12, c):
    out = np.full(b11.shape, np.nan, dtype=float)
    m = np.isfinite(b11) & np.isfinite(b12) & (b12 > 0)
    out[m] = (c*b12[m] - b11[m]) / b12[m]
    return out


def log_ratio(b11,b12):
    out = np.full(b11.shape, np.nan, dtype=float)
    m = np.isfinite(b11) & np.isfinite(b12) & (b11>0) & (b12>0)
    out[m] = np.log((b11[m]+1e-12)/(b12[m]+1e-12))
    return out


def summarize(a, mask, prefix):
    v = a[mask & np.isfinite(a)]
    if len(v) == 0:
        return {f"{prefix}_{k}":np.nan for k in
                ["mean","median","p90","p95","p99","top5_mean","abs_mean","valid_pixels"]}
    p95 = np.percentile(v,95)
    top = v[v >= p95]
    return {
        f"{prefix}_mean": float(np.mean(v)),
        f"{prefix}_median": float(np.median(v)),
        f"{prefix}_p90": float(np.percentile(v,90)),
        f"{prefix}_p95": float(p95),
        f"{prefix}_p99": float(np.percentile(v,99)),
        f"{prefix}_top5_mean": float(np.mean(top)) if len(top) else np.nan,
        f"{prefix}_abs_mean": float(np.mean(np.abs(v))),
        f"{prefix}_valid_pixels": int(len(v)),
    }


def compare_pair(t0, ref, scale_mode, dilation_iterations):
    s0, note0 = infer_scale(t0, scale_mode)
    sr, noter = infer_scale(ref, scale_mode)
    b0 = scaled_bands(t0, s0)
    br = scaled_bands(ref, sr)

    bad0, comp0 = lrad_bad_mask(b0, dilation_iterations)
    badr, compr = lrad_bad_mask(br, dilation_iterations)

    raw_common = np.ones(t0["shape"], dtype=bool)
    for b in NEEDED:
        raw_common &= np.isfinite(b0[b]) & np.isfinite(br[b])

    lrad_common = raw_common & (~bad0) & (~badr)

    # Log-ratio temporal proxy.
    lr0 = log_ratio(b0["B11"], b0["B12"])
    lrr = log_ratio(br["B11"], br["B12"])
    dlr = lr0-lrr

    # MBSP-like reflectance factor. Fit c separately per acquisition,
    # using common LRAD-valid pixels to reduce scene-brightness/background effects.
    c0 = fit_c(b0["B11"], b0["B12"], lrad_common)
    cr = fit_c(br["B11"], br["B12"], lrad_common)
    f0 = mbsp_factor(b0["B11"], b0["B12"], c0)
    fr = mbsp_factor(br["B11"], br["B12"], cr)
    dmbsp = f0-fr

    out = {
        "t0_scale": s0,
        "ref_scale": sr,
        "t0_scale_note": note0,
        "ref_scale_note": noter,
        "dilation_iterations": dilation_iterations,
        "raw_common_fraction": float(np.mean(raw_common)),
        "lrad_common_fraction": float(np.mean(lrad_common)),
        "lrad_removed_from_common_fraction": (
            float(1 - lrad_common.sum()/raw_common.sum())
            if raw_common.sum() else np.nan
        ),
        "c_t0": c0,
        "c_ref": cr,
    }

    for k,v in comp0.items():
        out[f"t0_{k}"] = v
    for k,v in compr.items():
        out[f"ref_{k}"] = v

    out.update(summarize(dlr, raw_common, "raw_logratio_delta"))
    out.update(summarize(dlr, lrad_common, "lrad_logratio_delta"))
    out.update(summarize(dmbsp, raw_common, "raw_mbspfactor_delta"))
    out.update(summarize(dmbsp, lrad_common, "lrad_mbspfactor_delta"))
    return out


def auc_positive_high(y, score):
    d = pd.DataFrame({
        "y":pd.to_numeric(y,errors="coerce"),
        "s":pd.to_numeric(score,errors="coerce"),
    }).dropna()
    if d.y.nunique() < 2:
        return np.nan
    yv=d.y.astype(int).to_numpy()
    sv=d.s.to_numpy(float)
    n1=(yv==1).sum()
    n0=(yv==0).sum()
    ranks=rankdata(sv,method="average")
    u=ranks[yv==1].sum()-n1*(n1+1)/2
    return float(u/(n1*n0))


def ols_r2(y,X):
    y=np.asarray(y,float)
    X=np.asarray(X,float)
    m=np.isfinite(y)&np.all(np.isfinite(X),axis=1)
    y=y[m]; X=X[m]
    if len(y)<3 or np.std(y)==0:
        return np.nan
    X=np.column_stack([np.ones(len(y)),X])
    beta,*_=np.linalg.lstsq(X,y,rcond=None)
    pred=X@beta
    sst=np.sum((y-y.mean())**2)
    return float(1-np.sum((y-pred)**2)/sst) if sst>0 else np.nan


def site_design(s):
    return pd.get_dummies(s.astype(str),drop_first=True).to_numpy(float)


def diagnostic_table(primary):
    features = [
        "raw_logratio_delta_p95",
        "lrad_logratio_delta_p95",
        "raw_logratio_delta_top5_mean",
        "lrad_logratio_delta_top5_mean",
        "raw_logratio_delta_abs_mean",
        "lrad_logratio_delta_abs_mean",
        "raw_mbspfactor_delta_p95",
        "lrad_mbspfactor_delta_p95",
        "raw_mbspfactor_delta_top5_mean",
        "lrad_mbspfactor_delta_top5_mean",
        "raw_mbspfactor_delta_abs_mean",
        "lrad_mbspfactor_delta_abs_mean",
    ]
    rows=[]
    for dil,g0 in primary.groupby("dilation_iterations"):
        for strategy,g in g0.groupby("reference_strategy"):
            for f in features:
                if f not in g:
                    continue
                d=g[["label","canonical_site",f]].copy()
                d["label"]=pd.to_numeric(d.label,errors="coerce")
                d[f]=pd.to_numeric(d[f],errors="coerce")
                d=d.dropna()
                if d.label.nunique()<2:
                    continue

                pos=d.loc[d.label==1,f].to_numpy()
                neg=d.loc[d.label==0,f].to_numpy()
                auc=auc_positive_high(d.label,d[f])

                Xs=site_design(d.canonical_site)
                Xl=d.label.to_numpy(float)[:,None]
                yy=d[f].to_numpy(float)
                rs=ols_r2(yy,Xs) if Xs.shape[1] else np.nan
                rl=ols_r2(yy,Xl)
                rb=ols_r2(yy,np.column_stack([Xs,Xl])) if Xs.shape[1] else rl

                try:
                    p=mannwhitneyu(pos,neg,alternative="two-sided").pvalue
                except Exception:
                    p=np.nan

                rows.append({
                    "dilation_iterations":dil,
                    "reference_strategy":strategy,
                    "feature":f,
                    "n":len(d),
                    "n_positive":len(pos),
                    "n_negative":len(neg),
                    "positive_median":np.median(pos),
                    "negative_median":np.median(neg),
                    "raw_auc_positive_high":auc,
                    "orientation_free_auc":max(auc,1-auc),
                    "mannwhitney_p":p,
                    "r2_label_only":rl,
                    "r2_site_only":rs,
                    "incremental_r2_label_after_site":rb-rs if np.isfinite(rb) and np.isfinite(rs) else np.nan,
                })
    return pd.DataFrame(rows)


def within_site_table(primary):
    rows=[]
    feats=[
        "raw_logratio_delta_p95","lrad_logratio_delta_p95",
        "raw_mbspfactor_delta_p95","lrad_mbspfactor_delta_p95",
        "raw_logratio_delta_top5_mean","lrad_logratio_delta_top5_mean",
        "raw_mbspfactor_delta_top5_mean","lrad_mbspfactor_delta_top5_mean",
    ]
    for (dil,strategy,site),g in primary.groupby(
        ["dilation_iterations","reference_strategy","canonical_site"]
    ):
        y=pd.to_numeric(g.label,errors="coerce")
        if y.nunique()<2:
            continue
        for f in feats:
            x=pd.to_numeric(g[f],errors="coerce")
            d=pd.DataFrame({"y":y,"x":x}).dropna()
            if d.y.nunique()<2:
                continue
            a=auc_positive_high(d.y,d.x)
            rows.append({
                "dilation_iterations":dil,
                "reference_strategy":strategy,
                "site":site,
                "feature":f,
                "n":len(d),
                "n_positive":int((d.y==1).sum()),
                "n_negative":int((d.y==0).sum()),
                "raw_auc_positive_high":a,
                "orientation_free_auc":max(a,1-a),
                "positive_median":d.loc[d.y==1,"x"].median(),
                "negative_median":d.loc[d.y==0,"x"].median(),
            })
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--audit-dir",default="~/methane_reference_full_audit")
    ap.add_argument("--out",default="~/methane_lrad_reference_test")
    ap.add_argument("--assume-standard-s2-order",action="store_true")
    ap.add_argument(
        "--reflectance-scale",
        default="auto",
        help="'auto', '1', or '10000'"
    )
    ap.add_argument(
        "--dilation-iterations",
        nargs="+",
        type=int,
        default=[0,1],
        help=(
            "Paper specifies morphological dilation but not a unique kernel/radius in "
            "the main text. Run sensitivity values instead of pretending one is exact."
        ),
    )
    args=ap.parse_args()

    audit=Path(args.audit_dir).expanduser().resolve()
    out=Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)

    src=pd.read_csv(audit/"04_reference_metrics_per_sample.csv")
    unique=clean_unique(src)
    unique.to_csv(out/"00_unique_real_temporal_samples.csv",index=False)

    primary=unique[unique.dataset_group.eq("five_site")].copy()
    primary.to_csv(out/"01_primary_five_site_samples.csv",index=False)

    print("Unique real temporal:",len(unique))
    print(unique.groupby("dataset_group").size().to_string())
    print("\nPrimary five-site:",len(primary))
    print(primary.label.value_counts().to_string())

    rows=[]
    for i,r in unique.iterrows():
        print(f"[{i+1}/{len(unique)}] {r.sample_id}")
        try:
            t0=read_cube(r.t0_path,args.assume_standard_s2_order)
            t90=align_cube(
                read_cube(r.t90_path,args.assume_standard_s2_order),t0
            )
            t360=align_cube(
                read_cube(r.t360_path,args.assume_standard_s2_order),t0
            )

            for strategy,ref in [
                ("fixed_t90",t90),
                ("best", t90 if r.best_reference=="t90" else t360),
            ]:
                for dil in args.dilation_iterations:
                    m=compare_pair(
                        t0,ref,args.reflectance_scale,dil
                    )
                    rec={
                        "sample_id":r.sample_id,
                        "dataset_group":r.dataset_group,
                        "canonical_site":r.canonical_site,
                        "label":r.label,
                        "best_reference":r.best_reference,
                        "reference_strategy":strategy,
                        "dilation_iterations":dil,
                        "t0_path":r.t0_path,
                        "reference_path":(
                            r.t90_path if strategy=="fixed_t90"
                            else (r.t90_path if r.best_reference=="t90" else r.t360_path)
                        ),
                        "status":"PASS",
                        "error":"",
                    }
                    rec.update(m)
                    rows.append(rec)
        except Exception as exc:
            rows.append({
                "sample_id":r.sample_id,
                "dataset_group":r.dataset_group,
                "canonical_site":r.canonical_site,
                "label":r.label,
                "status":"FAIL",
                "error":f"{type(exc).__name__}: {exc}",
            })

    metrics=pd.DataFrame(rows)
    metrics.to_csv(out/"02_lrad_metrics_per_sample.csv",index=False)

    good=metrics[metrics.status.eq("PASS")].copy()
    prim=good[good.dataset_group.eq("five_site")].copy()

    diag=diagnostic_table(prim)
    diag.to_csv(out/"03_primary_label_diagnostics.csv",index=False)

    within=within_site_table(prim)
    within.to_csv(out/"04_primary_within_site.csv",index=False)

    mask_summary=(
        good.groupby(
            ["dataset_group","reference_strategy","dilation_iterations"]
        )
        .agg(
            n=("sample_id","size"),
            median_lrad_common_fraction=("lrad_common_fraction","median"),
            median_removed_fraction=("lrad_removed_from_common_fraction","median"),
            mean_removed_fraction=("lrad_removed_from_common_fraction","mean"),
        )
        .reset_index()
    )
    mask_summary.to_csv(out/"05_mask_fraction_summary.csv",index=False)

    # Compact markdown summary.
    lines=[]
    lines.append("# LRAD + historical-reference test")
    lines.append("")
    lines.append(f"- Unique real temporal samples: {len(unique)}")
    for k,n in unique.dataset_group.value_counts().items():
        lines.append(f"- {k}: {n}")
    lines.append(f"- Primary five-site N: {len(primary)}")
    if len(primary):
        vc=primary.label.value_counts()
        lines.append(f"- Primary positive: {int(vc.get(1,0))}")
        lines.append(f"- Primary negative: {int(vc.get(0,0))}")
    lines.append("")
    lines.append("## Best diagnostic rows")
    if len(diag):
        for _,z in diag.sort_values(
            "orientation_free_auc",ascending=False
        ).head(12).iterrows():
            lines.append(
                f"- dil={int(z.dilation_iterations)} | "
                f"{z.reference_strategy} | {z.feature} | "
                f"AUROC={z.raw_auc_positive_high:.3f} | "
                f"orientation-free={z.orientation_free_auc:.3f} | "
                f"R2_site={z.r2_site_only:.3f} | "
                f"R2_label_after_site={z.incremental_r2_label_after_site:.3f}"
            )
    lines.append("")
    lines.append(
        "Interpretation: improvement is strongest when LRAD increases within-site "
        "positive/negative separation while reducing R2_site. Do not interpret the "
        "spectral proxy as calibrated methane concentration."
    )
    (out/"SUMMARY_LRAD.md").write_text("\n".join(lines),encoding="utf-8")

    print("\nDONE:",out)
    print("Upload these:")
    for fn in [
        "SUMMARY_LRAD.md",
        "02_lrad_metrics_per_sample.csv",
        "03_primary_label_diagnostics.csv",
        "04_primary_within_site.csv",
        "05_mask_fraction_summary.csv",
    ]:
        print(" ",out/fn)


if __name__=="__main__":
    main()
