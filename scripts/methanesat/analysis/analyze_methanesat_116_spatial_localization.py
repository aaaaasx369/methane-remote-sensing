#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_methanesat_116_spatial_localization.py

Spatial localization audit for the frozen MethaneSAT 116-pair benchmark.

Scientific question
-------------------
The simple image statistics already separate positives from same-site,
far-time no-detection controls (best AUROC ~0.80), while center-minus-ring
is much weaker (~0.61).

This script asks WHERE the paired XCH4 difference occurs:

    D_i(x,y) = XCH4_positive_i(x,y) - XCH4_control_i(x,y)

It produces:
1) raw paired-difference mean / median maps
2) per-pixel fraction of pairs with positive > control
3) valid-pair-count map
4) background-centered paired-difference maps:
       [positive - positive outer-ring median]
       -
       [control  - control outer-ring median]
   This removes each acquisition's broad patch-level offset.
5) radial profiles in 30 m bins from 0 to 240 m
6) pair-level localization metrics
7) bootstrap 95% CIs for radial mean/median paired differences

Important
---------
The pair NPZs are standardized to 224x224 from the same physical 480 m crop.
The 224x224 grid is a normalized spatial grid for alignment/visualization;
it is NOT 224x224 independent 45 m observations and must not be described as
new spatial resolution.

Controls remain temporal no-detection controls, not externally confirmed
zero-emission observations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--benchmark-dir",
        default=(
            "/Volumes/engg-leung/dora lin/"
            "MethaneSAT_MethaneFuse/"
            "05_paired_image_benchmark_120"
        ),
    )
    p.add_argument("--crop-size-m", type=float, default=480.0)
    p.add_argument("--ring-inner-m", type=float, default=120.0)
    p.add_argument("--ring-outer-m", type=float, default=220.0)
    p.add_argument("--radial-bin-m", type=float, default=30.0)
    p.add_argument("--radial-max-m", type=float, default=240.0)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260813)
    return p.parse_args()


def percentile_ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def bootstrap_pair_stat(values, n_boot, rng, stat="mean"):
    """
    values: 1D pair-level values. Resample complete pairs.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan

    out = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        xb = x[rng.integers(0, n, size=n)]
        if stat == "mean":
            out[i] = np.mean(xb)
        elif stat == "median":
            out[i] = np.median(xb)
        else:
            raise ValueError(stat)
    return percentile_ci(out)


def exact_sign_test_two_sided(values):
    d = np.asarray(values, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return np.nan, 0, 0, 0

    k_pos = int((d > 0).sum())
    k_neg = int((d < 0).sum())
    smaller = min(k_pos, k_neg)
    prob = sum(math.comb(n, i) for i in range(smaller + 1)) / (2 ** n)
    p = min(1.0, 2.0 * prob)
    return float(p), n, k_pos, k_neg


def weighted_nanmean_stack(stack, valid):
    num = np.where(valid, stack, 0.0).sum(axis=0)
    den = valid.sum(axis=0)
    out = np.full(num.shape, np.nan, dtype=np.float64)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out, den


def nanmedian_stack(stack, valid):
    arr = np.where(valid, stack, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmedian(arr, axis=0)


def save_png(array, path, title, crop_size_m, cmap=None, vmin=None, vmax=None):
    # matplotlib is imported lazily so the numerical audit still works if
    # users only want CSV/NPY outputs and matplotlib is unavailable.
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, ax = plt.subplots(figsize=(7, 6))
    extent = [
        -crop_size_m / 2,
        crop_size_m / 2,
        -crop_size_m / 2,
        crop_size_m / 2,
    ]
    kwargs = {
        "origin": "lower",
        "extent": extent,
        "aspect": "equal",
    }
    if cmap is not None:
        kwargs["cmap"] = cmap
    if vmin is not None:
        kwargs["vmin"] = vmin
    if vmax is not None:
        kwargs["vmax"] = vmax

    im = ax.imshow(array, **kwargs)
    ax.scatter([0], [0], marker="+", s=100)
    ax.set_xlabel("East-west offset (m)")
    ax.set_ylabel("North-south offset (m)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def save_radial_plot(radial_df, path, value_col, low_col, high_col, ylabel, title):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    x = radial_df["radius_mid_m"].to_numpy(float)
    y = radial_df[value_col].to_numpy(float)
    lo = radial_df[low_col].to_numpy(float)
    hi = radial_df[high_col].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, marker="o")
    ax.fill_between(x, lo, hi, alpha=0.2)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Distance from source center (m)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main():
    args = parse_args()

    root = Path(args.benchmark_dir).expanduser()
    manifest_dir = root / "manifests"
    analysis_dir = root / "analysis" / "spatial_localization"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    pair_manifest = manifest_dir / "06_repaired_primary_pairs.csv"
    if not pair_manifest.exists():
        raise FileNotFoundError(pair_manifest)

    pairs = pd.read_csv(pair_manifest, low_memory=False)
    if len(pairs) != 116 or pairs["pair_id"].nunique() != 116:
        raise RuntimeError(
            f"Expected 116 canonical pairs; got rows={len(pairs)}, "
            f"unique={pairs['pair_id'].nunique()}"
        )

    # Resolve pair NPZs. Canonical repair manifest normally includes pair_npz
    # for repaired rows, but original valid rows may rely on standard path.
    pair_npz_dir = root / "npz" / "pairs"

    loaded = []
    expected_shape = None

    for _, r in pairs.iterrows():
        pid = str(r["pair_id"])

        candidate_path = ""
        if "pair_npz" in pairs.columns:
            candidate_path = str(r.get("pair_npz", "")).strip()

        if candidate_path and candidate_path.lower() not in {"nan", "none"}:
            p = Path(candidate_path)
        else:
            p = pair_npz_dir / f"{pid}.npz"

        if not p.exists():
            # Standard path is authoritative fallback.
            fallback = pair_npz_dir / f"{pid}.npz"
            if fallback.exists():
                p = fallback
            else:
                raise FileNotFoundError(f"Missing pair NPZ for {pid}: {p}")

        with np.load(p, allow_pickle=False) as z:
            pos = z["positive_xch4"].astype(np.float64)
            ctrl = z["temporal_control_xch4"].astype(np.float64)
            pmask = z["positive_valid_mask"].astype(bool)
            cmask = z["temporal_control_valid_mask"].astype(bool)

        if pos.shape != ctrl.shape or pos.shape != pmask.shape or pos.shape != cmask.shape:
            raise RuntimeError(f"Shape mismatch in {p}")

        if expected_shape is None:
            expected_shape = pos.shape
        if pos.shape != expected_shape:
            raise RuntimeError(
                f"Inconsistent standardized shape: {pid} {pos.shape} != {expected_shape}"
            )

        valid = pmask & cmask & np.isfinite(pos) & np.isfinite(ctrl)
        diff = np.where(valid, pos - ctrl, np.nan)

        loaded.append({
            "pair_id": pid,
            "pair_npz": str(p),
            "pos": pos,
            "ctrl": ctrl,
            "pmask": pmask,
            "cmask": cmask,
            "valid": valid,
            "diff": diff,
        })

    n_pairs = len(loaded)
    h, w = expected_shape

    # Physical coordinates on standardized 480 m square.
    yy, xx = np.mgrid[0:h, 0:w]
    xm = ((xx + 0.5) / w - 0.5) * args.crop_size_m
    ym = ((yy + 0.5) / h - 0.5) * args.crop_size_m
    rr = np.sqrt(xm * xm + ym * ym)

    bg_ring = (rr >= args.ring_inner_m) & (rr <= args.ring_outer_m)
    center60 = rr <= 60.0

    raw_stack = np.stack([d["diff"] for d in loaded], axis=0)
    raw_valid = np.stack([d["valid"] for d in loaded], axis=0)

    # Background-center each acquisition separately before paired subtraction.
    centered_stack = np.full_like(raw_stack, np.nan)
    pair_metric_rows = []

    for i, d in enumerate(loaded):
        pos = d["pos"]
        ctrl = d["ctrl"]
        pmask = d["pmask"] & np.isfinite(pos)
        cmask = d["cmask"] & np.isfinite(ctrl)
        valid = d["valid"]

        p_ring_vals = pos[pmask & bg_ring]
        c_ring_vals = ctrl[cmask & bg_ring]

        p_bg = float(np.median(p_ring_vals)) if len(p_ring_vals) else np.nan
        c_bg = float(np.median(c_ring_vals)) if len(c_ring_vals) else np.nan

        if np.isfinite(p_bg) and np.isfinite(c_bg):
            centered = np.where(
                valid,
                (pos - p_bg) - (ctrl - c_bg),
                np.nan,
            )
            centered_stack[i] = centered
        else:
            centered = np.full_like(pos, np.nan)

        raw_diff = d["diff"]

        center_vals = raw_diff[valid & center60]
        outer_vals = raw_diff[valid & bg_ring]
        all_vals = raw_diff[valid]

        centered_center_vals = centered[np.isfinite(centered) & center60]
        centered_outer_vals = centered[np.isfinite(centered) & bg_ring]

        # Distance from source center to the strongest positive paired-difference
        # location on the normalized grid. This is exploratory: interpolation
        # does not create new native spatial resolution.
        if np.isfinite(raw_diff).any():
            flat_idx = int(np.nanargmax(raw_diff))
            py, px = np.unravel_index(flat_idx, raw_diff.shape)
            max_diff = float(raw_diff[py, px])
            max_diff_radius = float(rr[py, px])
        else:
            max_diff = np.nan
            max_diff_radius = np.nan

        pair_metric_rows.append({
            "pair_id": d["pair_id"],
            "positive_ring_median": p_bg,
            "control_ring_median": c_bg,
            "ring_baseline_shift_pos_minus_ctrl": (
                p_bg - c_bg if np.isfinite(p_bg) and np.isfinite(c_bg) else np.nan
            ),
            "raw_diff_all_mean": float(np.mean(all_vals)) if len(all_vals) else np.nan,
            "raw_diff_all_median": float(np.median(all_vals)) if len(all_vals) else np.nan,
            "raw_diff_center60_mean": float(np.mean(center_vals)) if len(center_vals) else np.nan,
            "raw_diff_outer120_220_median": float(np.median(outer_vals)) if len(outer_vals) else np.nan,
            "raw_center_minus_outer": (
                float(np.mean(center_vals) - np.median(outer_vals))
                if len(center_vals) and len(outer_vals) else np.nan
            ),
            "centered_diff_center60_mean": (
                float(np.mean(centered_center_vals))
                if len(centered_center_vals) else np.nan
            ),
            "centered_diff_outer120_220_median": (
                float(np.median(centered_outer_vals))
                if len(centered_outer_vals) else np.nan
            ),
            "max_raw_paired_diff": max_diff,
            "radius_of_max_raw_paired_diff_m": max_diff_radius,
            "joint_valid_fraction": float(valid.mean()),
        })

    centered_valid = np.isfinite(centered_stack)

    raw_mean_map, raw_count_map = weighted_nanmean_stack(raw_stack, raw_valid)
    raw_median_map = nanmedian_stack(raw_stack, raw_valid)

    centered_mean_map, centered_count_map = weighted_nanmean_stack(
        centered_stack, centered_valid
    )
    centered_median_map = nanmedian_stack(centered_stack, centered_valid)

    # Per-pixel fraction of valid pairs with positive > control.
    pos_gt_count = np.where(raw_valid, raw_stack > 0, False).sum(axis=0)
    pos_gt_fraction = np.full((h, w), np.nan, dtype=np.float64)
    ok_count = raw_count_map > 0
    pos_gt_fraction[ok_count] = (
        pos_gt_count[ok_count] / raw_count_map[ok_count]
    )

    # Persist maps as NPY for exact numerical reuse.
    np.save(analysis_dir / "00_raw_mean_difference_map.npy", raw_mean_map)
    np.save(analysis_dir / "01_raw_median_difference_map.npy", raw_median_map)
    np.save(analysis_dir / "02_fraction_pairs_positive_gt_control_map.npy", pos_gt_fraction)
    np.save(analysis_dir / "03_valid_pair_count_map.npy", raw_count_map)
    np.save(analysis_dir / "04_background_centered_mean_difference_map.npy", centered_mean_map)
    np.save(analysis_dir / "05_background_centered_median_difference_map.npy", centered_median_map)

    # Radial bins.
    edges = np.arange(
        0.0,
        args.radial_max_m + args.radial_bin_m + 1e-9,
        args.radial_bin_m,
    )

    rng = np.random.default_rng(args.seed)
    radial_rows = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        ring = (rr >= lo) & (rr < hi)

        pair_raw_ring_mean = []
        pair_raw_ring_median = []
        pair_centered_ring_mean = []
        pair_centered_ring_median = []

        for i in range(n_pairs):
            rv = raw_stack[i][ring & raw_valid[i]]
            cv = centered_stack[i][ring & centered_valid[i]]

            pair_raw_ring_mean.append(
                float(np.mean(rv)) if len(rv) else np.nan
            )
            pair_raw_ring_median.append(
                float(np.median(rv)) if len(rv) else np.nan
            )
            pair_centered_ring_mean.append(
                float(np.mean(cv)) if len(cv) else np.nan
            )
            pair_centered_ring_median.append(
                float(np.median(cv)) if len(cv) else np.nan
            )

        prm = np.asarray(pair_raw_ring_mean, dtype=float)
        prmed = np.asarray(pair_raw_ring_median, dtype=float)
        pcm = np.asarray(pair_centered_ring_mean, dtype=float)
        pcmed = np.asarray(pair_centered_ring_median, dtype=float)

        raw_mean_ci = bootstrap_pair_stat(
            prm, args.bootstrap, rng, stat="mean"
        )
        raw_median_ci = bootstrap_pair_stat(
            prmed, args.bootstrap, rng, stat="median"
        )
        centered_mean_ci = bootstrap_pair_stat(
            pcm, args.bootstrap, rng, stat="mean"
        )
        centered_median_ci = bootstrap_pair_stat(
            pcmed, args.bootstrap, rng, stat="median"
        )

        sign_p_raw, sign_n_raw, sign_pos_raw, sign_neg_raw = exact_sign_test_two_sided(prm)
        sign_p_centered, sign_n_centered, sign_pos_centered, sign_neg_centered = exact_sign_test_two_sided(pcm)

        radial_rows.append({
            "radius_lo_m": lo,
            "radius_hi_m": hi,
            "radius_mid_m": (lo + hi) / 2.0,

            "n_pairs_raw": int(np.isfinite(prm).sum()),
            "raw_pair_mean_of_ring_means": float(np.nanmean(prm)),
            "raw_pair_mean_ci_low": raw_mean_ci[0],
            "raw_pair_mean_ci_high": raw_mean_ci[1],
            "raw_pair_median_of_ring_medians": float(np.nanmedian(prmed)),
            "raw_pair_median_ci_low": raw_median_ci[0],
            "raw_pair_median_ci_high": raw_median_ci[1],
            "raw_fraction_pairs_ring_mean_gt0": float(np.nanmean(prm > 0)),
            "raw_sign_test_p": sign_p_raw,

            "n_pairs_background_centered": int(np.isfinite(pcm).sum()),
            "background_centered_pair_mean_of_ring_means": float(np.nanmean(pcm)),
            "background_centered_pair_mean_ci_low": centered_mean_ci[0],
            "background_centered_pair_mean_ci_high": centered_mean_ci[1],
            "background_centered_pair_median_of_ring_medians": float(np.nanmedian(pcmed)),
            "background_centered_pair_median_ci_low": centered_median_ci[0],
            "background_centered_pair_median_ci_high": centered_median_ci[1],
            "background_centered_fraction_pairs_ring_mean_gt0": float(np.nanmean(pcm > 0)),
            "background_centered_sign_test_p": sign_p_centered,
        })

    radial_df = pd.DataFrame(radial_rows)
    pair_metrics = pd.DataFrame(pair_metric_rows)

    radial_df.to_csv(analysis_dir / "06_radial_profile.csv", index=False)
    pair_metrics.to_csv(analysis_dir / "07_pair_localization_metrics.csv", index=False)

    # Overall pair-level summary metrics.
    summary_metrics = []

    for metric in [
        "ring_baseline_shift_pos_minus_ctrl",
        "raw_diff_all_mean",
        "raw_diff_center60_mean",
        "raw_diff_outer120_220_median",
        "raw_center_minus_outer",
        "centered_diff_center60_mean",
        "radius_of_max_raw_paired_diff_m",
    ]:
        vals = pd.to_numeric(pair_metrics[metric], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]

        mean_ci = bootstrap_pair_stat(vals, args.bootstrap, rng, "mean")
        med_ci = bootstrap_pair_stat(vals, args.bootstrap, rng, "median")
        sign_p, n_nonzero, kpos, kneg = exact_sign_test_two_sided(vals)

        summary_metrics.append({
            "metric": metric,
            "n_pairs": len(vals),
            "mean": float(np.mean(vals)) if len(vals) else np.nan,
            "mean_ci_low": mean_ci[0],
            "mean_ci_high": mean_ci[1],
            "median": float(np.median(vals)) if len(vals) else np.nan,
            "median_ci_low": med_ci[0],
            "median_ci_high": med_ci[1],
            "fraction_gt0": float(np.mean(vals > 0)) if len(vals) else np.nan,
            "sign_test_p": sign_p,
        })

    summary_df = pd.DataFrame(summary_metrics)
    summary_df.to_csv(analysis_dir / "08_localization_summary_metrics.csv", index=False)

    # Useful diagnostic: how broad is the raw offset?
    # Compare outer-ring baseline shift with all-patch shift.
    valid_metric = pair_metrics[
        np.isfinite(pair_metrics["ring_baseline_shift_pos_minus_ctrl"])
        & np.isfinite(pair_metrics["raw_diff_all_mean"])
    ].copy()

    if len(valid_metric) >= 3:
        corr = float(np.corrcoef(
            valid_metric["ring_baseline_shift_pos_minus_ctrl"].to_numpy(float),
            valid_metric["raw_diff_all_mean"].to_numpy(float),
        )[0, 1])
    else:
        corr = np.nan

    # PNGs.
    # Symmetric limits make positive/negative spatial contrast easier to compare.
    finite_raw = raw_median_map[np.isfinite(raw_median_map)]
    raw_abs = float(np.percentile(np.abs(finite_raw), 98)) if len(finite_raw) else None

    finite_centered = centered_median_map[np.isfinite(centered_median_map)]
    centered_abs = (
        float(np.percentile(np.abs(finite_centered), 98))
        if len(finite_centered) else None
    )

    save_png(
        raw_mean_map,
        analysis_dir / "09_raw_mean_difference_map.png",
        "Mean paired XCH4 difference: positive - temporal control",
        args.crop_size_m,
        cmap="coolwarm",
        vmin=-raw_abs if raw_abs else None,
        vmax=raw_abs if raw_abs else None,
    )
    save_png(
        raw_median_map,
        analysis_dir / "10_raw_median_difference_map.png",
        "Median paired XCH4 difference: positive - temporal control",
        args.crop_size_m,
        cmap="coolwarm",
        vmin=-raw_abs if raw_abs else None,
        vmax=raw_abs if raw_abs else None,
    )
    save_png(
        pos_gt_fraction,
        analysis_dir / "11_fraction_pairs_positive_gt_control.png",
        "Fraction of valid pairs with positive XCH4 > control",
        args.crop_size_m,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    save_png(
        centered_median_map,
        analysis_dir / "12_background_centered_median_difference_map.png",
        "Median paired difference after outer-ring background centering",
        args.crop_size_m,
        cmap="coolwarm",
        vmin=-centered_abs if centered_abs else None,
        vmax=centered_abs if centered_abs else None,
    )
    save_png(
        raw_count_map,
        analysis_dir / "13_valid_pair_count_map.png",
        "Valid pair count per normalized pixel",
        args.crop_size_m,
        cmap="viridis",
    )

    save_radial_plot(
        radial_df,
        analysis_dir / "14_raw_radial_profile.png",
        "raw_pair_mean_of_ring_means",
        "raw_pair_mean_ci_low",
        "raw_pair_mean_ci_high",
        "Positive - control XCH4",
        "Raw paired XCH4 difference vs distance from source center",
    )
    save_radial_plot(
        radial_df,
        analysis_dir / "15_background_centered_radial_profile.png",
        "background_centered_pair_mean_of_ring_means",
        "background_centered_pair_mean_ci_low",
        "background_centered_pair_mean_ci_high",
        "Background-centered positive - control XCH4",
        "Background-centered paired difference vs distance from source",
    )

    # Console summary.
    print("=" * 110)
    print("METHANESAT 116-PAIR SPATIAL LOCALIZATION AUDIT")
    print("=" * 110)
    print("Pairs loaded:", n_pairs)
    print("Standardized grid:", f"{h}x{w}")
    print("Physical crop:", f"{args.crop_size_m:.0f} m x {args.crop_size_m:.0f} m")
    print()
    print("PAIR-LEVEL LOCALIZATION METRICS")
    print(summary_df.to_string(index=False))
    print()
    print("RADIAL PROFILE")
    show_cols = [
        "radius_lo_m",
        "radius_hi_m",
        "n_pairs_raw",
        "raw_pair_mean_of_ring_means",
        "raw_pair_mean_ci_low",
        "raw_pair_mean_ci_high",
        "background_centered_pair_mean_of_ring_means",
        "background_centered_pair_mean_ci_low",
        "background_centered_pair_mean_ci_high",
    ]
    print(radial_df[show_cols].to_string(index=False))
    print()
    print("Correlation: outer-ring baseline shift vs all-patch raw shift")
    print(f"  r = {corr:.4f}" if np.isfinite(corr) else "  r = NaN")
    print()

    # Summary markdown.
    smap = {
        row["metric"]: row
        for _, row in summary_df.iterrows()
    }

    def fnum(x, digits=3):
        try:
            x = float(x)
            if not np.isfinite(x):
                return "NA"
            return f"{x:.{digits}f}"
        except Exception:
            return "NA"

    lines = [
        "# MethaneSAT 116-pair spatial localization audit",
        "",
        "Dataset: 116 L4-detected positive observations paired with 116 same-site, far-time temporal no-detection controls.",
        "",
        "The 224x224 arrays are standardized normalized spatial grids derived from the same 480 m physical crop; they do not represent 224x224 independent native-resolution observations.",
        "",
        "## Key pair-level metrics",
        "",
    ]

    for metric in [
        "ring_baseline_shift_pos_minus_ctrl",
        "raw_diff_all_mean",
        "raw_diff_center60_mean",
        "raw_diff_outer120_220_median",
        "raw_center_minus_outer",
        "centered_diff_center60_mean",
        "radius_of_max_raw_paired_diff_m",
    ]:
        r = smap[metric]
        lines += [
            f"### {metric}",
            f"- n pairs: {int(r['n_pairs'])}",
            f"- mean: {fnum(r['mean'])} "
            f"[95% pair-bootstrap CI {fnum(r['mean_ci_low'])}, {fnum(r['mean_ci_high'])}]",
            f"- median: {fnum(r['median'])} "
            f"[95% pair-bootstrap CI {fnum(r['median_ci_low'])}, {fnum(r['median_ci_high'])}]",
            f"- fraction > 0: {fnum(r['fraction_gt0'])}",
            f"- exact two-sided sign-test p: {r['sign_test_p']:.6g}"
            if np.isfinite(r["sign_test_p"]) else "- exact two-sided sign-test p: NA",
            "",
        ]

    lines += [
        "## Broad-offset diagnostic",
        "",
        f"- Pearson correlation between outer-ring baseline shift and all-patch raw paired shift: r = {fnum(corr, 4)}",
        "",
        "Interpretation guide:",
        "- If raw paired differences remain positive across most radii and outer-ring shift tracks all-patch shift strongly, the ~0.80 global-feature separability is dominated by a broad patch-level XCH4 shift.",
        "- If background-centering removes most of that radial signal, the broad-offset interpretation is strengthened.",
        "- If a strong positive signal remains near the center after background-centering and falls with radius, that supports a localized source/plume component.",
        "",
        "Temporal controls are no-detection controls, not externally confirmed zero-emission states.",
    ]

    (analysis_dir / "SUMMARY_SPATIAL_LOCALIZATION.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("Saved:")
    for p in [
        "00_raw_mean_difference_map.npy",
        "01_raw_median_difference_map.npy",
        "02_fraction_pairs_positive_gt_control_map.npy",
        "03_valid_pair_count_map.npy",
        "04_background_centered_mean_difference_map.npy",
        "05_background_centered_median_difference_map.npy",
        "06_radial_profile.csv",
        "07_pair_localization_metrics.csv",
        "08_localization_summary_metrics.csv",
        "09_raw_mean_difference_map.png",
        "10_raw_median_difference_map.png",
        "11_fraction_pairs_positive_gt_control.png",
        "12_background_centered_median_difference_map.png",
        "13_valid_pair_count_map.png",
        "14_raw_radial_profile.png",
        "15_background_centered_radial_profile.png",
        "SUMMARY_SPATIAL_LOCALIZATION.md",
    ]:
        print(" ", analysis_dir / p)


if __name__ == "__main__":
    main()
