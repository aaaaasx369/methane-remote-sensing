#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
postprocess_reference_audit_v2.py

Post-process the first local Sentinel-2 historical-reference audit.

This version explicitly removes:
1) fake temporal triplets where t0/t90/t360 are the same path;
2) duplicate copies with the same sample_id;
3) repeated Sentinel-2 acquisitions copied into multiple evaluation subsets
   (deduplicated by canonical site + Sentinel scene token).

It then produces:
- a clean unique temporal inventory;
- a primary five-site mixed-label benchmark;
- fixed-t90 vs best-reference comparison;
- within-site label separation;
- site-vs-label R² diagnostics.

No TIFF pixels are re-read; this script uses the pixel metrics already computed in
04_reference_metrics_per_sample.csv.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, mannwhitneyu


SCENE_RE = re.compile(r"(\d{8}T\d{6}_\d{8}T\d{6}_T\d{2}[A-Z]{3})")


def canonical_site(x):
    s = str(x or "").strip()
    low = s.lower()
    if "casa_grande" in low or "casa grande" in low:
        return "Casa_Grande"
    if "ehrenberg" in low:
        return "Ehrenberg"
    return s


def dataset_group(path):
    s = str(path)
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
    m = SCENE_RE.search(str(path))
    return m.group(1) if m else ""


def auc_positive_high(y, score):
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


def ols_r2(y, X):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[m]
    X = X[m]
    if len(y) < 3 or np.std(y) == 0:
        return np.nan
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def site_design(site_series):
    d = pd.get_dummies(site_series.astype(str), drop_first=True)
    return d.to_numpy(float)


def strategy_stats(df):
    rows = []
    for strategy in ["t90", "t360", "best"]:
        for metric in [
            "b4_corr",
            "abs_ndvi_median_change",
            "abs_ndbi_median_change",
            "swir_delta_p95",
            "swir_delta_top5_mean",
            "swir_delta_abs_mean",
        ]:
            col = f"{strategy}_{metric}"
            if col not in df:
                continue
            x = pd.to_numeric(df[col], errors="coerce")
            rows.append({
                "strategy": strategy,
                "metric": metric,
                "n": int(x.notna().sum()),
                "mean": x.mean(),
                "median": x.median(),
                "p25": x.quantile(.25),
                "p75": x.quantile(.75),
            })
    return pd.DataFrame(rows)


def label_stats(df):
    rows = []
    features = [
        "best_b4_corr",
        "best_abs_ndvi_median_change",
        "best_abs_ndbi_median_change",
        "best_swir_delta_p95",
        "best_swir_delta_top5_mean",
        "best_swir_delta_abs_mean",
        "t90_swir_delta_p95",
        "t90_swir_delta_top5_mean",
        "t90_swir_delta_abs_mean",
    ]

    y = pd.to_numeric(df["label"], errors="coerce")
    site_X = site_design(df["canonical_site"])
    label_X = y.to_numpy(float)[:, None]

    for f in features:
        if f not in df:
            continue
        x = pd.to_numeric(df[f], errors="coerce")
        d = pd.DataFrame({
            "label": y,
            "feature": x,
            "site": df["canonical_site"],
        }).dropna()

        if len(d) == 0:
            continue

        pos = d.loc[d.label == 1, "feature"].to_numpy()
        neg = d.loc[d.label == 0, "feature"].to_numpy()

        auc = auc_positive_high(d.label, d.feature)
        p = (
            mannwhitneyu(pos, neg, alternative="two-sided").pvalue
            if len(pos) and len(neg) else np.nan
        )

        # Rebuild design on rows with valid feature.
        Xs = site_design(d["site"])
        Xl = d["label"].to_numpy(float)[:, None]
        yy = d["feature"].to_numpy(float)

        r2_site = ols_r2(yy, Xs) if Xs.shape[1] else np.nan
        r2_label = ols_r2(yy, Xl)
        r2_both = ols_r2(yy, np.column_stack([Xs, Xl])) if Xs.shape[1] else r2_label

        rows.append({
            "feature": f,
            "n": len(d),
            "n_positive": len(pos),
            "n_negative": len(neg),
            "positive_median": np.median(pos) if len(pos) else np.nan,
            "negative_median": np.median(neg) if len(neg) else np.nan,
            "raw_auc_positive_high": auc,
            "orientation_free_auc": max(auc, 1 - auc) if np.isfinite(auc) else np.nan,
            "mannwhitney_p": p,
            "r2_label_only": r2_label,
            "r2_site_only": r2_site,
            "incremental_r2_label_after_site": (
                r2_both - r2_site
                if np.isfinite(r2_both) and np.isfinite(r2_site)
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def within_site_stats(df):
    rows = []
    for site, g in df.groupby("canonical_site"):
        y = pd.to_numeric(g.label, errors="coerce")
        if y.nunique() < 2:
            continue
        for f in [
            "best_swir_delta_p95",
            "best_swir_delta_top5_mean",
            "best_swir_delta_abs_mean",
            "t90_swir_delta_p95",
            "t90_swir_delta_top5_mean",
            "t90_swir_delta_abs_mean",
        ]:
            x = pd.to_numeric(g[f], errors="coerce")
            d = pd.DataFrame({"label": y, "feature": x}).dropna()
            if d.label.nunique() < 2:
                continue
            a = auc_positive_high(d.label, d.feature)
            rows.append({
                "site": site,
                "feature": f,
                "n": len(d),
                "n_positive": int((d.label == 1).sum()),
                "n_negative": int((d.label == 0).sum()),
                "raw_auc_positive_high": a,
                "orientation_free_auc": max(a, 1-a),
                "positive_median": d.loc[d.label == 1, "feature"].median(),
                "negative_median": d.loc[d.label == 0, "feature"].median(),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--audit-dir",
        default="~/methane_reference_full_audit",
        help="Folder containing 04_reference_metrics_per_sample.csv",
    )
    ap.add_argument(
        "--out",
        default="~/methane_reference_full_audit_v2_clean",
    )
    args = ap.parse_args()

    audit = Path(args.audit_dir).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    src = audit / "04_reference_metrics_per_sample.csv"
    df = pd.read_csv(src)
    df = df[df["analysis_status"].eq("PASS")].copy()

    # Audit flags.
    df["dataset_group"] = df["t0_path"].map(dataset_group)
    df["canonical_site"] = df["site"].map(canonical_site)
    df["scene_token"] = df["t0_path"].map(scene_token)

    df["same_t0_t90"] = df["t0_path"].eq(df["t90_path"])
    df["same_t0_t360"] = df["t0_path"].eq(df["t360_path"])
    df["same_t90_t360"] = df["t90_path"].eq(df["t360_path"])
    df["fake_temporal_alias"] = (
        df["same_t0_t90"] |
        df["same_t0_t360"] |
        df["same_t90_t360"]
    )

    df.to_csv(out / "00_all_pixel_successful_with_flags.csv", index=False)

    # Step 1: remove fake temporal aliases.
    real = df[~df["fake_temporal_alias"]].copy()

    # Step 2: remove duplicated copied datasets with identical sample_id.
    priority = {
        "five_site": 0,
        "exact_external": 1,
        "methaneair_p1": 2,
        "methaneair_p1_final16": 3,
        "other": 4,
    }
    real["_priority"] = real["dataset_group"].map(priority).fillna(9)

    # Same sample_id copied to another folder -> keep preferred copy.
    real = (
        real.sort_values(["_priority", "t0_path"])
        .drop_duplicates(subset=["sample_id"], keep="first")
        .copy()
    )

    # Step 3: same Sentinel acquisition + same canonical site appearing in multiple subsets.
    has_scene = real["scene_token"].ne("")
    scene_part = (
        real[has_scene]
        .sort_values(["_priority", "t0_path"])
        .drop_duplicates(subset=["canonical_site", "scene_token"], keep="first")
    )
    nonscene_part = real[~has_scene]

    unique = pd.concat([scene_part, nonscene_part], ignore_index=True)
    unique = unique.drop(columns=["_priority"], errors="ignore")
    unique.to_csv(out / "01_unique_real_temporal_samples.csv", index=False)

    # Primary mixed-label benchmark.
    five = unique[unique["dataset_group"].eq("five_site")].copy()
    five.to_csv(out / "02_primary_five_site_samples.csv", index=False)

    strategy_stats(five).to_csv(out / "03_five_site_reference_strategy.csv", index=False)
    label_stats(five).to_csv(out / "04_five_site_label_diagnostics.csv", index=False)
    within_site_stats(five).to_csv(out / "05_five_site_within_site.csv", index=False)

    # All unique real temporal data inventory; not automatically a fair binary benchmark.
    strategy_stats(unique).to_csv(out / "06_all_unique_reference_strategy.csv", index=False)

    # Summary.
    lines = []
    lines.append("# Corrected Sentinel-2 reference audit v2")
    lines.append("")
    lines.append(f"- Pixel-successful rows in first audit: {len(df)}")
    lines.append(f"- Fake temporal aliases removed: {int(df.fake_temporal_alias.sum())}")
    lines.append(f"- Real distinct-path temporal rows: {len(real)} after sample-id dedup")
    lines.append(f"- Unique logical temporal samples after scene/site dedup: {len(unique)}")
    lines.append("")
    lines.append("## Unique data by group")
    for k, n in unique["dataset_group"].value_counts().items():
        lines.append(f"- {k}: {n}")
    lines.append("")
    lines.append("## Primary five-site benchmark")
    lines.append(f"- N={len(five)}")
    if len(five):
        vc = five.label.value_counts()
        lines.append(f"- Positive={int(vc.get(1,0))}, Negative={int(vc.get(0,0))}")
        br = five.best_reference.value_counts()
        lines.append(f"- best=t90: {int(br.get('t90',0))}")
        lines.append(f"- best=t360: {int(br.get('t360',0))}")

        t90corr = pd.to_numeric(five.t90_b4_corr, errors="coerce")
        bestcorr = pd.to_numeric(five.best_b4_corr, errors="coerce")
        lines.append(f"- Mean B4 corr fixed t90: {t90corr.mean():.6f}")
        lines.append(f"- Mean B4 corr best: {bestcorr.mean():.6f}")
        lines.append(f"- Mean B4 corr gain: {(bestcorr-t90corr).mean():.6f}")
        lines.append(f"- Median B4 corr fixed t90: {t90corr.median():.6f}")
        lines.append(f"- Median B4 corr best: {bestcorr.median():.6f}")

        a = pd.to_numeric(five.t90_abs_ndvi_median_change, errors="coerce")
        b = pd.to_numeric(five.best_abs_ndvi_median_change, errors="coerce")
        lines.append(f"- Mean |ΔNDVI| fixed t90: {a.mean():.6f}")
        lines.append(f"- Mean |ΔNDVI| best: {b.mean():.6f}")

        a = pd.to_numeric(five.t90_abs_ndbi_median_change, errors="coerce")
        b = pd.to_numeric(five.best_abs_ndbi_median_change, errors="coerce")
        lines.append(f"- Mean |ΔNDBI| fixed t90: {a.mean():.6f}")
        lines.append(f"- Mean |ΔNDBI| best: {b.mean():.6f}")

    lines.append("")
    lines.append("## Important")
    lines.append(
        "The all-unique inventory should not automatically be used as one global binary "
        "benchmark because some added groups are positive-only. Use the five-site subset "
        "for the primary positive-vs-negative test, and report other groups separately."
    )

    (out / "SUMMARY_V2.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nDONE")
    print("Output:", out)
    print((out / "SUMMARY_V2.md").read_text())


if __name__ == "__main__":
    main()
