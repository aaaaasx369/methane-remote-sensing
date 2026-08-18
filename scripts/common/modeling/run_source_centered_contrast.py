#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_source_centered_contrast.py

Next Sentinel-2 experiment after:
1) historical-reference matching
2) LRAD audit

Why:
- Full LRAD strongly over-masks the vegetated five-site locations.
- Whole-patch summary statistics still show strong site/background dependence.
- The five-site patches were downloaded around the source coordinate, so the source is
  at the center of each 48x48 / 480 m patch.
- This script asks a more targeted question:

    Is the methane-sensitive temporal anomaly stronger near the known source than in
    the outer background of THE SAME patch?

This is intentionally source-centered and local-background-normalized.
It does NOT estimate calibrated methane concentration.

Inputs:
    ~/methane_release_project/methane_lrad_reference_test/
        02_lrad_metrics_per_sample.csv

Primary benchmark:
    five_site only, best historical reference, one row per sample.

Outputs:
    01_source_centered_metrics.csv
    02_source_centered_diagnostics.csv
    03_within_site_diagnostics.csv
    04_desert_controlled_release_diagnostics.csv
    SUMMARY_SOURCE_CENTERED.md
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import rankdata, mannwhitneyu


STANDARD_S2_12 = [
    "B1","B2","B3","B4","B5","B6",
    "B7","B8","B8A","B9","B11","B12",
]
NEEDED = ["B3","B4","B8","B11","B12"]


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


def band_map(ds, assume_standard):
    mapping = {}
    for i, d in enumerate(ds.descriptions or [], 1):
        b = normalize_band_name(d)
        if b:
            mapping[b] = i
    if set(NEEDED).issubset(mapping):
        return mapping

    for i in range(1, ds.count + 1):
        tags = ds.tags(i)
        for k in ["name","NAME","band_name","BAND_NAME","description","DESCRIPTION"]:
            if k in tags:
                b = normalize_band_name(tags[k])
                if b:
                    mapping[b] = i
    if set(NEEDED).issubset(mapping):
        return mapping

    if assume_standard and ds.count == 12:
        return {b:i+1 for i,b in enumerate(STANDARD_S2_12)}

    raise ValueError(
        f"Cannot resolve B3/B4/B8/B11/B12 in raster with {ds.count} bands."
    )


def read_cube(path, assume_standard):
    with rasterio.open(path) as ds:
        bm = band_map(ds, assume_standard)
        bands = {}
        for b in NEEDED:
            a = ds.read(bm[b], masked=True).astype("float64")
            if np.ma.isMaskedArray(a):
                a = a.filled(np.nan)
            a = np.asarray(a, dtype=float)
            if ds.nodata is not None:
                a[a == ds.nodata] = np.nan
            bands[b] = a

        return {
            "bands": bands,
            "transform": ds.transform,
            "crs": ds.crs,
            "shape": (ds.height, ds.width),
            "res_x": abs(float(ds.transform.a)),
            "res_y": abs(float(ds.transform.e)),
        }


def align_cube(src, target):
    if (
        src["shape"] == target["shape"] and
        src["transform"] == target["transform"] and
        str(src["crs"]) == str(target["crs"])
    ):
        return src

    out = {
        "bands": {},
        "transform": target["transform"],
        "crs": target["crs"],
        "shape": target["shape"],
        "res_x": target["res_x"],
        "res_y": target["res_y"],
    }
    for b, a in src["bands"].items():
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


def infer_scale(cube):
    vals = []
    for b in NEEDED:
        a = cube["bands"][b]
        v = a[np.isfinite(a) & (a > 0)]
        if len(v):
            vals.append(np.percentile(v, 99))
    p99 = np.median(vals) if vals else np.nan
    return 10000.0 if np.isfinite(p99) and p99 > 2 else 1.0


def scaled(cube):
    s = infer_scale(cube)
    return {b:cube["bands"][b]/s for b in NEEDED}


def light_valid_mask(b0, br):
    """
    Conservative mask that preserves vegetation.

    Removes only:
      - invalid/non-positive reflectance
      - B3 lowest 5% in either image (dark/shadow/smoke-like pixels)
      - saturated SWIR-like pixels (B11>=1 AND B12>=1)

    Unlike full LRAD, it does NOT exclude NDVI>=0.3 vegetation, because the
    previous audit showed that this erased most of MA_site_043/073.
    """
    m = np.ones(b0["B3"].shape, dtype=bool)

    for b in NEEDED:
        m &= np.isfinite(b0[b]) & np.isfinite(br[b])
        m &= (b0[b] > 0) & (br[b] > 0)

    q0 = np.nanpercentile(b0["B3"][np.isfinite(b0["B3"])], 5)
    qr = np.nanpercentile(br["B3"][np.isfinite(br["B3"])], 5)

    dark = (b0["B3"] <= q0) | (br["B3"] <= qr)
    sat = ((b0["B11"] >= 1) & (b0["B12"] >= 1)) | \
          ((br["B11"] >= 1) & (br["B12"] >= 1))

    return m & (~dark) & (~sat)


def logratio(b):
    out = np.full(b["B11"].shape, np.nan, dtype=float)
    m = (
        np.isfinite(b["B11"]) & np.isfinite(b["B12"]) &
        (b["B11"] > 0) & (b["B12"] > 0)
    )
    out[m] = np.log((b["B11"][m] + 1e-12)/(b["B12"][m] + 1e-12))
    return out


def fit_c(b, mask):
    m = (
        mask &
        np.isfinite(b["B11"]) & np.isfinite(b["B12"]) &
        (b["B11"] > 0) & (b["B12"] > 0)
    )
    if m.sum() < 30:
        return np.nan
    x = b["B12"][m]
    y = b["B11"][m]
    den = np.sum(x*x)
    return float(np.sum(x*y)/den) if den > 0 else np.nan


def mbsp_factor(b, c):
    out = np.full(b["B11"].shape, np.nan, dtype=float)
    m = np.isfinite(b["B11"]) & np.isfinite(b["B12"]) & (b["B12"] > 0)
    out[m] = (c*b["B12"][m] - b["B11"][m]) / b["B12"][m]
    return out


def radial_distance_m(shape, res_x, res_y):
    h, w = shape
    # Source is at patch center because the original download region is
    # Point(source_lon, source_lat).buffer(HALF_PATCH_METERS).bounds().
    cy = (h - 1) / 2
    cx = (w - 1) / 2
    yy, xx = np.indices((h,w))
    return np.sqrt(((xx-cx)*res_x)**2 + ((yy-cy)*res_y)**2)


def robust_mad(v):
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    med = np.median(v)
    return float(np.median(np.abs(v-med)))


def region_features(delta, valid, dist_m, inner_radius, ring_inner, ring_outer, prefix):
    inner = valid & (dist_m <= inner_radius)
    ring = valid & (dist_m >= ring_inner) & (dist_m <= ring_outer)

    iv = delta[inner & np.isfinite(delta)]
    rv = delta[ring & np.isfinite(delta)]

    out = {
        f"{prefix}_inner_valid_pixels": len(iv),
        f"{prefix}_ring_valid_pixels": len(rv),
    }

    if len(iv) < 8 or len(rv) < 20:
        for k in [
            "inner_median","inner_p90","inner_p95","inner_top10_mean",
            "ring_median","ring_mad","contrast_p95","contrast_top10",
            "z_p95","z_top10","abs_contrast_p95"
        ]:
            out[f"{prefix}_{k}"] = np.nan
        return out

    p90 = np.percentile(iv,90)
    p95 = np.percentile(iv,95)
    top10 = iv[iv >= p90]

    rmed = np.median(rv)
    rmad = robust_mad(rv)
    sigma = 1.4826*rmad
    eps = 1e-8

    out.update({
        f"{prefix}_inner_median": float(np.median(iv)),
        f"{prefix}_inner_p90": float(p90),
        f"{prefix}_inner_p95": float(p95),
        f"{prefix}_inner_top10_mean": float(np.mean(top10)),
        f"{prefix}_ring_median": float(rmed),
        f"{prefix}_ring_mad": float(rmad),
        f"{prefix}_contrast_p95": float(p95-rmed),
        f"{prefix}_contrast_top10": float(np.mean(top10)-rmed),
        f"{prefix}_z_p95": float((p95-rmed)/(sigma+eps)),
        f"{prefix}_z_top10": float((np.mean(top10)-rmed)/(sigma+eps)),
        f"{prefix}_abs_contrast_p95": float(abs(p95-rmed)),
    })
    return out


def auc_positive_high(y, score):
    d = pd.DataFrame({
        "y":pd.to_numeric(y,errors="coerce"),
        "s":pd.to_numeric(score,errors="coerce")
    }).dropna()
    if d["y"].nunique() < 2:
        return np.nan

    yv = d["y"].astype(int).to_numpy()
    sv = d["s"].to_numpy(float)
    n1 = int((yv==1).sum())
    n0 = int((yv==0).sum())
    ranks = rankdata(sv, method="average")
    u = ranks[yv==1].sum() - n1*(n1+1)/2
    return float(u/(n1*n0))


def ols_r2(y, X):
    y = np.asarray(y,float)
    X = np.asarray(X,float)
    m = np.isfinite(y) & np.all(np.isfinite(X),axis=1)
    y = y[m]
    X = X[m]
    if len(y)<3 or np.std(y)==0:
        return np.nan
    X = np.column_stack([np.ones(len(y)),X])
    beta,*_ = np.linalg.lstsq(X,y,rcond=None)
    pred = X@beta
    sst = np.sum((y-y.mean())**2)
    return float(1-np.sum((y-pred)**2)/sst) if sst>0 else np.nan


def diagnostics(df, subset_name):
    feature_cols = [
        c for c in df.columns
        if any(c.endswith(suf) for suf in [
            "_contrast_p95","_contrast_top10","_z_p95","_z_top10","_abs_contrast_p95"
        ])
    ]

    rows=[]
    for f in feature_cols:
        d = df[["label","canonical_site",f]].copy()
        d["label"] = pd.to_numeric(d["label"],errors="coerce")
        d[f] = pd.to_numeric(d[f],errors="coerce")
        d = d.dropna()

        if d["label"].nunique()<2:
            continue

        pos = d.loc[d.label==1,f].to_numpy()
        neg = d.loc[d.label==0,f].to_numpy()
        a = auc_positive_high(d.label,d[f])

        try:
            p = mannwhitneyu(pos,neg,alternative="two-sided").pvalue
        except Exception:
            p = np.nan

        Xs = pd.get_dummies(d["canonical_site"].astype(str),drop_first=True).to_numpy(float)
        Xl = d["label"].to_numpy(float)[:,None]
        yy = d[f].to_numpy(float)

        rs = ols_r2(yy,Xs) if Xs.shape[1] else np.nan
        rl = ols_r2(yy,Xl)
        rb = ols_r2(yy,np.column_stack([Xs,Xl])) if Xs.shape[1] else rl

        rows.append({
            "subset":subset_name,
            "feature":f,
            "n":len(d),
            "n_positive":len(pos),
            "n_negative":len(neg),
            "positive_median":np.median(pos),
            "negative_median":np.median(neg),
            "raw_auc_positive_high":a,
            "orientation_free_auc":max(a,1-a),
            "mannwhitney_p":p,
            "r2_label_only":rl,
            "r2_site_only":rs,
            "incremental_r2_label_after_site":(
                rb-rs if np.isfinite(rb) and np.isfinite(rs) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def within_site(df):
    rows=[]
    feature_cols = [
        c for c in df.columns
        if any(c.endswith(suf) for suf in [
            "_contrast_p95","_contrast_top10","_z_p95","_z_top10","_abs_contrast_p95"
        ])
    ]

    for site,g in df.groupby("canonical_site"):
        y = pd.to_numeric(g.label,errors="coerce")
        if y.nunique()<2:
            continue

        for f in feature_cols:
            x = pd.to_numeric(g[f],errors="coerce")
            d = pd.DataFrame({"y":y,"x":x}).dropna()
            if d.y.nunique()<2:
                continue
            a = auc_positive_high(d.y,d.x)
            rows.append({
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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lrad-dir",
        default="~/methane_release_project/methane_lrad_reference_test",
    )
    ap.add_argument(
        "--out",
        default="~/methane_release_project/methane_source_centered_test",
    )
    ap.add_argument(
        "--radii-m",
        nargs="+",
        type=float,
        default=[40,60,80,100,120],
    )
    ap.add_argument("--ring-inner-m",type=float,default=160)
    ap.add_argument("--ring-outer-m",type=float,default=225)
    ap.add_argument("--assume-standard-s2-order",action="store_true")
    args = ap.parse_args()

    lrad_dir = Path(args.lrad_dir).expanduser().resolve()
    outdir = Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True,exist_ok=True)

    src = pd.read_csv(lrad_dir/"02_lrad_metrics_per_sample.csv")

    # One real row per primary sample.
    df = src[
        (src["status"]=="PASS") &
        (src["dataset_group"]=="five_site") &
        (src["reference_strategy"]=="best") &
        (src["dilation_iterations"]==0)
    ].copy()

    df = df.drop_duplicates("sample_id").reset_index(drop=True)

    rows=[]
    for i,r in df.iterrows():
        print(f"[{i+1}/{len(df)}] {r.sample_id}")
        rec = {
            "sample_id":r.sample_id,
            "canonical_site":r.canonical_site,
            "label":r.label,
            "t0_path":r.t0_path,
            "reference_path":r.reference_path,
            "status":"PASS",
            "error":"",
        }

        try:
            t0 = read_cube(r.t0_path,args.assume_standard_s2_order)
            ref = align_cube(
                read_cube(r.reference_path,args.assume_standard_s2_order),t0
            )
            b0 = scaled(t0)
            br = scaled(ref)

            valid = light_valid_mask(b0,br)
            dist = radial_distance_m(t0["shape"],t0["res_x"],t0["res_y"])

            dlog = logratio(b0)-logratio(br)

            c0 = fit_c(b0,valid)
            cr = fit_c(br,valid)
            dmbsp = mbsp_factor(b0,c0)-mbsp_factor(br,cr)

            rec.update({
                "pixel_res_x_m":t0["res_x"],
                "pixel_res_y_m":t0["res_y"],
                "light_valid_fraction":float(np.mean(valid)),
                "c_t0":c0,
                "c_ref":cr,
            })

            for radius in args.radii_m:
                tag = f"r{int(radius)}m"
                rec.update(region_features(
                    dlog,valid,dist,radius,
                    args.ring_inner_m,args.ring_outer_m,
                    f"logratio_{tag}"
                ))
                rec.update(region_features(
                    dmbsp,valid,dist,radius,
                    args.ring_inner_m,args.ring_outer_m,
                    f"mbsp_{tag}"
                ))

        except Exception as exc:
            rec["status"]="FAIL"
            rec["error"]=f"{type(exc).__name__}: {exc}"

        rows.append(rec)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir/"01_source_centered_metrics.csv",index=False)

    good = metrics[metrics.status.eq("PASS")].copy()

    diag = diagnostics(good,"five_site_all")
    diag.to_csv(outdir/"02_source_centered_diagnostics.csv",index=False)

    ws = within_site(good)
    ws.to_csv(outdir/"03_within_site_diagnostics.csv",index=False)

    desert = good[good.canonical_site.isin(["Casa_Grande","Ehrenberg"])].copy()
    dd = diagnostics(desert,"Casa_Grande_plus_Ehrenberg")
    dd.to_csv(outdir/"04_desert_controlled_release_diagnostics.csv",index=False)

    lines=[]
    lines.append("# Source-centered local-background test")
    lines.append("")
    lines.append(f"- Primary rows: {len(df)}")
    lines.append(f"- Pixel-successful: {len(good)}")
    lines.append(f"- Positive: {int((pd.to_numeric(good.label,errors='coerce')==1).sum())}")
    lines.append(f"- Negative: {int((pd.to_numeric(good.label,errors='coerce')==0).sum())}")
    lines.append(f"- Median light-valid fraction: {good.light_valid_fraction.median():.3f}")
    lines.append("")
    lines.append("## Best exploratory diagnostics — all five sites")
    if len(diag):
        for _,z in diag.sort_values(
            ["orientation_free_auc","mannwhitney_p"],
            ascending=[False,True]
        ).head(12).iterrows():
            lines.append(
                f"- {z.feature}: raw AUC={z.raw_auc_positive_high:.3f}, "
                f"orientation-free={z.orientation_free_auc:.3f}, "
                f"p={z.mannwhitney_p:.3g}, "
                f"R2_site={z.r2_site_only:.3f}, "
                f"label-after-site={z.incremental_r2_label_after_site:.3f}"
            )
    lines.append("")
    lines.append("## Best exploratory diagnostics — controlled-release desert sites")
    if len(dd):
        for _,z in dd.sort_values(
            ["orientation_free_auc","mannwhitney_p"],
            ascending=[False,True]
        ).head(12).iterrows():
            lines.append(
                f"- {z.feature}: raw AUC={z.raw_auc_positive_high:.3f}, "
                f"orientation-free={z.orientation_free_auc:.3f}, "
                f"p={z.mannwhitney_p:.3g}"
            )
    lines.append("")
    lines.append(
        "Important: multiple radii are exploratory. A single best radius must not be "
        "reported as confirmatory performance without held-out validation."
    )
    lines.append(
        "If several neighboring radii show the same direction and within-site separation "
        "improves while R2_site falls, that is stronger evidence for a source-localized signal."
    )

    (outdir/"SUMMARY_SOURCE_CENTERED.md").write_text(
        "\n".join(lines),encoding="utf-8"
    )

    print("\nDONE:",outdir)
    print("Upload:")
    for fn in [
        "SUMMARY_SOURCE_CENTERED.md",
        "01_source_centered_metrics.csv",
        "02_source_centered_diagnostics.csv",
        "03_within_site_diagnostics.csv",
        "04_desert_controlled_release_diagnostics.csv",
    ]:
        print(" ",outdir/fn)


if __name__=="__main__":
    main()
