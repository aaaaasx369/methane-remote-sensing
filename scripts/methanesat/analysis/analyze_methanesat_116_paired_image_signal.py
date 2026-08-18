#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_methanesat_116_paired_image_signal.py

Primary image-only feature analysis for the frozen MethaneSAT paired benchmark:
  116 L4-detected positives
  116 same-site far-time temporal no-detection controls

Inputs
------
manifests/06_repaired_primary_pairs.csv
manifests/08_repaired_sample_image_features.csv

Outputs
-------
analysis/
  00_feature_results.csv
  01_pairwise_differences.csv
  02_feature_ranking.csv
  SUMMARY_IMAGE_SIGNAL_ANALYSIS.md

No model training.
No latitude/date/collection/flux metadata is used as a predictor.

For each image feature:
- positive vs control class medians/means
- paired positive-minus-control differences
- proportion of pairs with positive > control
- exact two-sided sign-test p-value
- raw AUROC (positive label = 1)
- pair-bootstrap 95% CI for AUROC
- pair-bootstrap 95% CI for mean paired difference
- pair-bootstrap 95% CI for median paired difference

Bootstrap resamples PAIRS, not individual images, preserving paired structure.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES_DEFAULT = [
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

    return float(
        (rank_sum_pos - n1 * (n1 + 1) / 2.0)
        / (n1 * n0)
    )


def exact_sign_test_two_sided(diff):
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]

    n = len(d)
    if n == 0:
        return np.nan, 0, 0, 0

    k = int((d > 0).sum())
    smaller = min(k, n - k)

    # Exact Binomial(n, 0.5) two-sided sign test:
    # 2 * P[X <= min(k, n-k)], capped at 1.
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

        y = np.concatenate([
            np.ones(n, dtype=int),
            np.zeros(n, dtype=int),
        ])
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


def main():
    args = parse_args()

    root = Path(args.benchmark_dir).expanduser()
    manifest_dir = root / "manifests"
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    pair_path = manifest_dir / "06_repaired_primary_pairs.csv"
    feat_path = manifest_dir / "08_repaired_sample_image_features.csv"

    if not pair_path.exists():
        raise FileNotFoundError(pair_path)
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)

    pairs = pd.read_csv(pair_path, low_memory=False)
    feat = pd.read_csv(feat_path, low_memory=False)

    if len(pairs) != 116 or pairs["pair_id"].nunique() != 116:
        raise RuntimeError(
            f"Expected 116 canonical pairs, got rows={len(pairs)}, "
            f"unique={pairs['pair_id'].nunique()}"
        )

    counts = feat["class_name"].value_counts().to_dict()
    if counts.get("positive", 0) != 116 or counts.get("temporal_control", 0) != 116:
        raise RuntimeError(
            f"Expected 116 positive + 116 temporal_control feature rows; got {counts}"
        )

    pair_class_counts = feat.groupby("pair_id")["class_name"].nunique()
    if feat["pair_id"].nunique() != 116 or not (pair_class_counts == 2).all():
        raise RuntimeError("Feature table is not a complete 116-pair table.")

    available_features = [f for f in FEATURES_DEFAULT if f in feat.columns]
    if not available_features:
        raise RuntimeError(
            f"None of the expected features were found. Columns: {list(feat.columns)}"
        )

    # One row per pair with positive/control columns.
    base_cols = ["pair_id", "class_name"] + available_features
    x = feat[base_cols].copy()

    pos = x[x["class_name"].eq("positive")].set_index("pair_id")
    ctrl = x[x["class_name"].eq("temporal_control")].set_index("pair_id")

    common = sorted(set(pos.index) & set(ctrl.index))
    if len(common) != 116:
        raise RuntimeError(f"Expected 116 paired feature IDs; found {len(common)}")

    rng = np.random.default_rng(args.seed)

    results = []
    pair_diff_rows = []

    for feature in available_features:
        p = pd.to_numeric(pos.loc[common, feature], errors="coerce").to_numpy(float)
        c = pd.to_numeric(ctrl.loc[common, feature], errors="coerce").to_numpy(float)

        ok = np.isfinite(p) & np.isfinite(c)
        pp = p[ok]
        cc = c[ok]
        ids = np.asarray(common, dtype=object)[ok]

        d = pp - cc

        y = np.concatenate([
            np.ones(len(pp), dtype=int),
            np.zeros(len(cc), dtype=int),
        ])
        score = np.concatenate([pp, cc])
        auc = auc_rank(y, score)

        sign_p, n_nonzero, n_posdiff, n_negdiff = exact_sign_test_two_sided(d)

        boot = paired_bootstrap(
            pp,
            cc,
            args.bootstrap,
            rng,
        )

        n_equal = int(np.sum(d == 0))
        prop_pos_gt = float(np.mean(d > 0)) if len(d) else np.nan

        result = {
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
        }
        results.append(result)

        for pid, pv, cv, dv in zip(ids, pp, cc, d):
            pair_diff_rows.append({
                "pair_id": pid,
                "feature": feature,
                "positive_value": pv,
                "control_value": cv,
                "paired_diff_pos_minus_ctrl": dv,
            })

    res = pd.DataFrame(results)

    # Rank primarily by raw AUC distance from 0.5, but keep raw direction visible.
    rank = res.copy()
    rank["auc_distance_from_0_5"] = (
        rank["raw_auc_positive_high"] - 0.5
    ).abs()
    rank = rank.sort_values(
        ["auc_distance_from_0_5", "fraction_pairs_positive_gt_control"],
        ascending=[False, False],
    ).reset_index(drop=True)

    pairdiff = pd.DataFrame(pair_diff_rows)

    res.to_csv(analysis_dir / "00_feature_results.csv", index=False)
    pairdiff.to_csv(analysis_dir / "01_pairwise_differences.csv", index=False)
    rank.to_csv(analysis_dir / "02_feature_ranking.csv", index=False)

    # Compact console + Markdown summary.
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

    print("=" * 120)
    print("METHANESAT 116-PAIR IMAGE-ONLY FEATURE ANALYSIS")
    print("=" * 120)
    print(rank[show_cols].to_string(index=False))
    print()

    lines = [
        "# MethaneSAT 116-pair image-only feature analysis",
        "",
        "Dataset: 116 L4-detected positive images + 116 same-site far-time temporal no-detection controls.",
        "",
        "Important: temporal controls are not externally confirmed zero-emission observations.",
        "",
        "## Feature ranking",
        "",
        "Raw AUROC uses positive=1 and the feature itself as the score.",
        "AUROC > 0.5 means larger feature values tend to occur in positives; AUROC < 0.5 means the reverse.",
        "Bootstrap confidence intervals resample complete pairs.",
        "",
    ]

    for _, r in rank.iterrows():
        lines += [
            f"### {r['feature']}",
            f"- complete pairs: {int(r['n_pairs_complete'])}",
            f"- raw AUROC: {r['raw_auc_positive_high']:.4f} "
            f"[95% CI {r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}]",
            f"- positive median: {r['positive_median']:.6g}",
            f"- control median: {r['control_median']:.6g}",
            f"- median paired difference (positive-control): "
            f"{r['median_paired_diff_pos_minus_ctrl']:.6g} "
            f"[95% CI {r['median_diff_ci_low']:.6g}, {r['median_diff_ci_high']:.6g}]",
            f"- fraction of pairs with positive > control: "
            f"{r['fraction_pairs_positive_gt_control']:.3f}",
            f"- exact two-sided sign-test p: {r['sign_test_p_two_sided']:.6g}",
            "",
        ]

    (analysis_dir / "SUMMARY_IMAGE_SIGNAL_ANALYSIS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Saved:")
    print(" ", analysis_dir / "00_feature_results.csv")
    print(" ", analysis_dir / "01_pairwise_differences.csv")
    print(" ", analysis_dir / "02_feature_ranking.csv")
    print(" ", analysis_dir / "SUMMARY_IMAGE_SIGNAL_ANALYSIS.md")


if __name__ == "__main__":
    main()
