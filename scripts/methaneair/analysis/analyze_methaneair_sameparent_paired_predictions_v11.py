#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, recall_score


def parse_args():
    p = argparse.ArgumentParser(description="Analyze MethaneFuse predictions for the same-parent paired MethaneAIR benchmark.")
    p.add_argument("--predictions", required=True)
    p.add_argument("--out-prefix", default="")
    return p.parse_args()


def safe_float(x):
    try:
        x = float(x)
        return None if not np.isfinite(x) else x
    except Exception:
        return None


def binary_metrics(df: pd.DataFrame) -> dict:
    y = pd.to_numeric(df["true_label"], errors="coerce").astype(int).to_numpy()
    pred = pd.to_numeric(df["predicted_label"], errors="coerce").astype(int).to_numpy()
    score = pd.to_numeric(df["probability_positive"], errors="coerce").to_numpy(float)
    pos = y == 1
    neg = y == 0
    tp = int(np.sum((pred == 1) & pos)); fn = int(np.sum((pred == 0) & pos))
    tn = int(np.sum((pred == 0) & neg)); fp = int(np.sum((pred == 1) & neg))
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    fpr = fp / max(1, tn + fp)
    auroc = roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan
    return {
        "n": int(len(df)), "positive_n": int(pos.sum()), "negative_n": int(neg.sum()),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "recall": float(recall), "specificity": float(specificity), "fpr": float(fpr),
        "auroc": safe_float(auroc),
        "median_p_positive_pos": safe_float(np.median(score[pos])) if pos.any() else None,
        "median_p_positive_neg": safe_float(np.median(score[neg])) if neg.any() else None,
    }


def main():
    args = parse_args()
    p = Path(args.predictions).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, low_memory=False)
    required = ["pair_id", "pair_role", "true_label", "predicted_label", "probability_positive"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Predictions missing columns: {missing}")

    pair_counts = df.groupby("pair_id")["pair_role"].agg(list)
    bad = pair_counts[pair_counts.apply(lambda x: sorted(x) != ["negative", "positive"])]
    if len(bad):
        raise RuntimeError(f"Malformed pairs: {len(bad)}; examples={bad.head().to_dict()}")

    overall = binary_metrics(df)

    pos = df[df["pair_role"].eq("positive")][["pair_id", "probability_positive"]].rename(columns={"probability_positive": "p_pos_positive"})
    neg_cols = ["pair_id", "probability_positive"]
    for c in ["site", "negative_evidence_grade", "source_positive_record_id", "positive_t0_abs_delta_hours"]:
        if c in df.columns:
            neg_cols.append(c)
    neg = df[df["pair_role"].eq("negative")][neg_cols].rename(columns={"probability_positive": "p_pos_negative"})
    pairs = pos.merge(neg, on="pair_id", how="inner", validate="one_to_one")
    pairs["score_delta_pos_minus_neg"] = pairs["p_pos_positive"] - pairs["p_pos_negative"]
    pairs["positive_ranks_above_negative"] = pairs["score_delta_pos_minus_neg"] > 0
    pairs["tie"] = pairs["score_delta_pos_minus_neg"] == 0
    pair_win_rate = (pairs["positive_ranks_above_negative"].mean() + 0.5 * pairs["tie"].mean()) if len(pairs) else np.nan

    pair_summary = {
        "pair_n": int(len(pairs)),
        "positive_score_gt_negative_fraction": safe_float(pairs["positive_ranks_above_negative"].mean()),
        "tie_fraction": safe_float(pairs["tie"].mean()),
        "paired_win_rate_ties_half": safe_float(pair_win_rate),
        "median_score_delta_pos_minus_neg": safe_float(pairs["score_delta_pos_minus_neg"].median()),
        "mean_score_delta_pos_minus_neg": safe_float(pairs["score_delta_pos_minus_neg"].mean()),
    }

    strata_rows = []
    for col in ["site", "negative_evidence_grade"]:
        if col not in df.columns:
            continue
        for value, g in df.groupby(col, dropna=False):
            if len(g) >= 2 and set(pd.to_numeric(g["true_label"], errors="coerce").dropna().astype(int).unique()) == {0,1}:
                m = binary_metrics(g)
                m.update({"stratum": col, "value": str(value)})
                strata_rows.append(m)
    strata = pd.DataFrame(strata_rows)

    result = {"predictions": str(p), "overall": overall, "paired": pair_summary}
    print(json.dumps(result, indent=2, allow_nan=False))
    print("\nPAIR SCORE DELTA")
    print(pairs["score_delta_pos_minus_neg"].describe(percentiles=[.1,.25,.5,.75,.9]).to_string())
    if len(strata):
        show = ["stratum","value","n","recall","specificity","fpr","balanced_accuracy","auroc"]
        print("\nSTRATIFIED METRICS")
        print(strata[show].to_string(index=False))

    if args.out_prefix:
        prefix = Path(args.out_prefix).expanduser()
    else:
        prefix = p.with_suffix("").with_name(p.stem + "_paired_analysis")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(prefix) + ".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pairs.to_csv(str(prefix) + "_pairs.csv", index=False)
    strata.to_csv(str(prefix) + "_strata.csv", index=False)


if __name__ == "__main__":
    main()
