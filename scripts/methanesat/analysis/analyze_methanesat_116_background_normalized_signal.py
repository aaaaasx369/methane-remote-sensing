#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_SET = [
    "mean",
    "median",
    "p90",
    "p95",
    "p99",
    "max",
    "center_r60_mean",
    "ring_r120_220_median",
    "center_minus_ring",
]


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
    p.add_argument("--center-radius-m", type=float, default=60.0)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260813)
    return p.parse_args()


def auc_rank(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy()
    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def exact_sign_test_two_sided(diff):
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return np.nan, 0, 0, 0
    k = int((d > 0).sum())
    smaller = min(k, n - k)
    prob = sum(math.comb(n, i) for i in range(smaller + 1)) / (2 ** n)
    p = min(1.0, 2.0 * prob)
    return float(p), n, k, int((d < 0).sum())


def percentile_ci(values, lo=2.5, hi=97.5):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def paired_bootstrap(pos, ctrl, n_boot, rng):
    pos = np.asarray(pos, dtype=float)
    ctrl = np.asarray(ctrl, dtype=float)
    ok = np.isfinite(pos) & np.isfinite(ctrl)
    pos = pos[ok]
    ctrl = ctrl[ok]
    n = len(pos)
    if n == 0:
        return {
            "auc_ci_low": np.nan,
            "auc_ci_high": np.nan,
            "mean_diff_ci_low": np.nan,
            "mean_diff_ci_high": np.nan,
            "median_diff_ci_low": np.nan,
            "median_diff_ci_high": np.nan,
        }
    aucs = np.empty(n_boot, dtype=float)
    mean_d = np.empty(n_boot, dtype=float)
    median_d = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pb = pos[idx]
        cb = ctrl[idx]
        y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
        s = np.concatenate([pb, cb])
        aucs[b] = auc_rank(y, s)
        d = pb - cb
        mean_d[b] = np.mean(d)
        median_d[b] = np.median(d)
    auc_lo, auc_hi = percentile_ci(aucs)
    mean_lo, mean_hi = percentile_ci(mean_d)
    med_lo, med_hi = percentile_ci(median_d)
    return {
        "auc_ci_low": auc_lo,
        "auc_ci_high": auc_hi,
        "mean_diff_ci_low": mean_lo,
        "mean_diff_ci_high": mean_hi,
        "median_diff_ci_low": med_lo,
        "median_diff_ci_high": med_hi,
    }


def safe_stats(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def image_features_from_array(arr, valid, center_mask, ring_mask):
    vals = arr[valid]
    center_vals = arr[valid & center_mask]
    ring_vals = arr[valid & ring_mask]
    out = safe_stats(vals)
    out["center_r60_mean"] = float(np.mean(center_vals)) if len(center_vals) else np.nan
    out["ring_r120_220_median"] = float(np.median(ring_vals)) if len(ring_vals) else np.nan
    out["center_minus_ring"] = (
        out["center_r60_mean"] - out["ring_r120_220_median"]
        if np.isfinite(out["center_r60_mean"]) and np.isfinite(out["ring_r120_220_median"])
        else np.nan
    )
    return out


def analyze_feature_table(feat_df, feature_cols, n_boot, rng):
    pos = feat_df[feat_df["class_name"].eq("positive")].set_index("pair_id")
    ctrl = feat_df[feat_df["class_name"].eq("temporal_control")].set_index("pair_id")
    common = sorted(set(pos.index) & set(ctrl.index))
    results = []
    pair_diff_rows = []
    for feature in feature_cols:
        p = pd.to_numeric(pos.loc[common, feature], errors="coerce").to_numpy(float)
        c = pd.to_numeric(ctrl.loc[common, feature], errors="coerce").to_numpy(float)
        ok = np.isfinite(p) & np.isfinite(c)
        pp = p[ok]
        cc = c[ok]
        ids = np.asarray(common, dtype=object)[ok]
        d = pp - cc
        y = np.concatenate([np.ones(len(pp), dtype=int), np.zeros(len(cc), dtype=int)])
        score = np.concatenate([pp, cc])
        auc = auc_rank(y, score)
        sign_p, n_nonzero, n_posdiff, n_negdiff = exact_sign_test_two_sided(d)
        boot = paired_bootstrap(pp, cc, n_boot, rng)
        n_equal = int(np.sum(d == 0))
        prop_pos_gt = float(np.mean(d > 0)) if len(d) else np.nan
        results.append({
            "feature": feature,
            "n_pairs_complete": len(d),
            "positive_mean": float(np.mean(pp)) if len(pp) else np.nan,
            "control_mean": float(np.mean(cc)) if len(cc) else np.nan,
            "positive_median": float(np.median(pp)) if len(pp) else np.nan,
            "control_median": float(np.median(cc)) if len(cc) else np.nan,
            "mean_paired_diff_pos_minus_ctrl": float(np.mean(d)) if len(d) else np.nan,
            "median_paired_diff_pos_minus_ctrl": float(np.median(d)) if len(d) else np.nan,
            "pairs_positive_gt_control": n_posdiff,
            "pairs_positive_lt_control": n_negdiff,
            "pairs_equal": n_equal,
            "fraction_pairs_positive_gt_control": prop_pos_gt,
            "sign_test_p_two_sided": sign_p,
            "raw_auc_positive_high": auc,
            "direction_free_auc": max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan,
            **boot,
        })
        for pid, pv, cv, dv in zip(ids, pp, cc, d):
            pair_diff_rows.append({
                "pair_id": pid,
                "feature": feature,
                "positive_value": pv,
                "control_value": cv,
                "paired_diff_pos_minus_ctrl": dv,
            })
    res = pd.DataFrame(results)
    pairdiff = pd.DataFrame(pair_diff_rows)
    rank = res.copy()
    rank["auc_distance_from_0_5"] = (rank["raw_auc_positive_high"] - 0.5).abs()
    rank = rank.sort_values(
        ["auc_distance_from_0_5", "fraction_pairs_positive_gt_control"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return res, pairdiff, rank


def main():
    args = parse_args()
    root = Path(args.benchmark_dir).expanduser()
    manifest_dir = root / "manifests"
    analysis_dir = root / "analysis" / "background_normalized"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    pair_path = manifest_dir / "06_repaired_primary_pairs.csv"
    pairs = pd.read_csv(pair_path, low_memory=False)
    if len(pairs) != 116 or pairs["pair_id"].nunique() != 116:
        raise RuntimeError("Expected 116 canonical pairs.")

    pair_npz_dir = root / "npz" / "pairs"
    first_path = None
    for _, r in pairs.iterrows():
        pid = str(r["pair_id"])
        pth = ""
        if "pair_npz" in pairs.columns:
            pth = str(r.get("pair_npz", "")).strip()
        candidate = Path(pth) if pth and pth.lower() not in {"nan", "none"} else (pair_npz_dir / f"{pid}.npz")
        if not candidate.exists():
            candidate = pair_npz_dir / f"{pid}.npz"
        if candidate.exists():
            first_path = candidate
            break
    if first_path is None:
        raise RuntimeError("No pair NPZ found.")

    with np.load(first_path, allow_pickle=False) as z:
        h, w = z["positive_xch4"].shape

    yy, xx = np.mgrid[0:h, 0:w]
    xm = ((xx + 0.5) / w - 0.5) * args.crop_size_m
    ym = ((yy + 0.5) / h - 0.5) * args.crop_size_m
    rr = np.sqrt(xm * xm + ym * ym)
    center_mask = rr <= args.center_radius_m
    ring_mask = (rr >= args.ring_inner_m) & (rr <= args.ring_outer_m)

    rows = []
    for _, r in pairs.iterrows():
        pid = str(r["pair_id"])
        pth = ""
        if "pair_npz" in pairs.columns:
            pth = str(r.get("pair_npz", "")).strip()
        pair_npz = Path(pth) if pth and pth.lower() not in {"nan", "none"} else (pair_npz_dir / f"{pid}.npz")
        if not pair_npz.exists():
            pair_npz = pair_npz_dir / f"{pid}.npz"
        with np.load(pair_npz, allow_pickle=False) as z:
            pos = z["positive_xch4"].astype(np.float64)
            ctrl = z["temporal_control_xch4"].astype(np.float64)
            pmask = z["positive_valid_mask"].astype(bool)
            cmask = z["temporal_control_valid_mask"].astype(bool)
        for class_name, arr, valid in [
            ("positive", pos, pmask),
            ("temporal_control", ctrl, cmask),
        ]:
            ring_vals = arr[valid & ring_mask]
            bg_med = float(np.median(ring_vals)) if len(ring_vals) else np.nan
            bg_std = float(np.std(ring_vals, ddof=0)) if len(ring_vals) else np.nan
            centered = np.where(valid, arr - bg_med, np.nan)
            if np.isfinite(bg_std) and bg_std > 0:
                zarr = np.where(valid, (arr - bg_med) / bg_std, np.nan)
            else:
                zarr = np.full(arr.shape, np.nan, dtype=np.float64)

            f_centered = image_features_from_array(
                centered, valid & np.isfinite(centered), center_mask, ring_mask
            )
            f_z = image_features_from_array(
                zarr, valid & np.isfinite(zarr), center_mask, ring_mask
            )

            row = {
                "pair_id": pid,
                "class_name": class_name,
                "background_ring_median": bg_med,
                "background_ring_std": bg_std,
                "valid_fraction": float(valid.mean()),
            }
            for k, v in f_centered.items():
                row[f"centered_{k}"] = v
            for k, v in f_z.items():
                row[f"z_{k}"] = v
            rows.append(row)

    feat = pd.DataFrame(rows)
    centered_cols = [f"centered_{f}" for f in FEATURE_SET]
    z_cols = [f"z_{f}" for f in FEATURE_SET]

    centered_res, centered_pairdiff, centered_rank = analyze_feature_table(
        feat, centered_cols, args.bootstrap, np.random.default_rng(args.seed)
    )
    z_res, z_pairdiff, z_rank = analyze_feature_table(
        feat, z_cols, args.bootstrap, np.random.default_rng(args.seed + 1)
    )

    feat.to_csv(analysis_dir / "00_background_normalized_feature_rows.csv", index=False)
    centered_res.to_csv(analysis_dir / "01_centered_feature_results.csv", index=False)
    centered_pairdiff.to_csv(analysis_dir / "02_centered_pairwise_differences.csv", index=False)
    centered_rank.to_csv(analysis_dir / "03_centered_feature_ranking.csv", index=False)
    z_res.to_csv(analysis_dir / "04_z_feature_results.csv", index=False)
    z_pairdiff.to_csv(analysis_dir / "05_z_pairwise_differences.csv", index=False)
    z_rank.to_csv(analysis_dir / "06_z_feature_ranking.csv", index=False)

    show_cols = [
        "feature",
        "n_pairs_complete",
        "raw_auc_positive_high",
        "auc_ci_low",
        "auc_ci_high",
        "median_paired_diff_pos_minus_ctrl",
        "median_diff_ci_low",
        "median_diff_ci_high",
        "fraction_pairs_positive_gt_control",
        "sign_test_p_two_sided",
    ]

    print("=" * 128)
    print("METHANESAT 116-PAIR BACKGROUND-NORMALIZED IMAGE SIGNAL ANALYSIS")
    print("=" * 128)
    print()
    print("A) BACKGROUND-CENTERED FEATURES  [X - median(outer ring)]")
    print("-" * 128)
    print(centered_rank[show_cols].to_string(index=False))
    print()
    print("B) BACKGROUND-Z FEATURES  [(X - median(outer ring)) / std(outer ring)]")
    print("-" * 128)
    print(z_rank[show_cols].to_string(index=False))
    print()

    lines = [
        "# MethaneSAT 116-pair background-normalized image signal analysis",
        "",
        "Two normalized representations were tested:",
        "1. background-centered: X - median(outer ring 120–220 m)",
        "2. background-z: (X - median(outer ring)) / std(outer ring)",
        "",
        "If the original ~0.80 AUROC mostly came from broad patch-level offset, these normalized-feature AUROCs should move toward 0.5.",
        "If substantial AUROC remains, then a relative/local signal survives background normalization.",
        "",
        "## A. Background-centered feature ranking",
        "",
    ]

    for _, r in centered_rank.iterrows():
        lines += [
            f"### {r['feature']}",
            f"- complete pairs: {int(r['n_pairs_complete'])}",
            f"- raw AUROC: {r['raw_auc_positive_high']:.4f} [95% CI {r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}]",
            f"- median paired difference (positive-control): {r['median_paired_diff_pos_minus_ctrl']:.6g} [95% CI {r['median_diff_ci_low']:.6g}, {r['median_diff_ci_high']:.6g}]",
            f"- fraction of pairs with positive > control: {r['fraction_pairs_positive_gt_control']:.3f}",
            f"- exact two-sided sign-test p: {r['sign_test_p_two_sided']:.6g}",
            "",
        ]

    lines += ["## B. Background-z feature ranking", ""]
    for _, r in z_rank.iterrows():
        lines += [
            f"### {r['feature']}",
            f"- complete pairs: {int(r['n_pairs_complete'])}",
            f"- raw AUROC: {r['raw_auc_positive_high']:.4f} [95% CI {r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}]",
            f"- median paired difference (positive-control): {r['median_paired_diff_pos_minus_ctrl']:.6g} [95% CI {r['median_diff_ci_low']:.6g}, {r['median_diff_ci_high']:.6g}]",
            f"- fraction of pairs with positive > control: {r['fraction_pairs_positive_gt_control']:.3f}",
            f"- exact two-sided sign-test p: {r['sign_test_p_two_sided']:.6g}",
            "",
        ]

    (analysis_dir / "SUMMARY_BACKGROUND_NORMALIZED_SIGNAL.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Saved:")
    print(" ", analysis_dir / "00_background_normalized_feature_rows.csv")
    print(" ", analysis_dir / "01_centered_feature_results.csv")
    print(" ", analysis_dir / "02_centered_pairwise_differences.csv")
    print(" ", analysis_dir / "03_centered_feature_ranking.csv")
    print(" ", analysis_dir / "04_z_feature_results.csv")
    print(" ", analysis_dir / "05_z_pairwise_differences.csv")
    print(" ", analysis_dir / "06_z_feature_ranking.csv")
    print(" ", analysis_dir / "SUMMARY_BACKGROUND_NORMALIZED_SIGNAL.md")


if __name__ == "__main__":
    main()
