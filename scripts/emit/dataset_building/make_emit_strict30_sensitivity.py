#!/usr/bin/env python3
"""
Create a strict ±30-day EMIT temporal sensitivity subset from the already-built
expanded60 final dataset.

No imagery is downloaded or reprocessed.

Input:
  ~/methane_release_project/EMIT_MethaneFuse_480m_TEMPORAL_60D/
    eval_emit_480m_abs.csv

Strict criterion, applied to BOTH POS and NEG rows of a pair:
  t90_target_error_days  <= 30
  t180_target_error_days <= 30

Only complete POS/NEG pairs are retained.

Outputs:
  eval_emit_480m_abs_strict30_real.csv
  eval_emit_480m_abs_strict30_t0repeated.csv
  strict30_pair_audit.csv
"""

from pathlib import Path
import pandas as pd

ROOT = (
    Path.home()
    / "methane_release_project"
    / "EMIT_MethaneFuse_480m_TEMPORAL_60D"
)

SRC = ROOT / "eval_emit_480m_abs.csv"
REAL_OUT = ROOT / "eval_emit_480m_abs_strict30_real.csv"
REP_OUT = ROOT / "eval_emit_480m_abs_strict30_t0repeated.csv"
AUDIT_OUT = ROOT / "strict30_pair_audit.csv"

if not SRC.exists():
    raise SystemExit(f"Missing input: {SRC}")

df = pd.read_csv(SRC)

required = [
    "id",
    "pair_id",
    "label",
    "emit_0_path",
    "emit_90_path",
    "emit_360_path",
    "t90_target_error_days",
    "t180_target_error_days",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

for c in ["t90_target_error_days", "t180_target_error_days"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["strict30_sample"] = (
    df["t90_target_error_days"].le(30.0)
    & df["t180_target_error_days"].le(30.0)
)

audit_rows = []
keep_pairs = []

for pair_id, g in df.groupby("pair_id", sort=True):
    labels = set(pd.to_numeric(g["label"], errors="coerce").dropna().astype(int))
    complete_pair = (len(g) == 2 and labels == {0, 1})
    strict_pair = complete_pair and bool(g["strict30_sample"].all())

    audit_rows.append(
        {
            "pair_id": pair_id,
            "rows": len(g),
            "has_pos_neg": int(labels == {0, 1}),
            "max_t90_error_days": float(g["t90_target_error_days"].max()),
            "max_t180_error_days": float(g["t180_target_error_days"].max()),
            "strict30_pair": int(strict_pair),
        }
    )

    if strict_pair:
        keep_pairs.append(pair_id)

audit = pd.DataFrame(audit_rows)
audit.to_csv(AUDIT_OUT, index=False)

strict = df[df["pair_id"].isin(keep_pairs)].copy()
strict = strict.sort_values(["pair_id", "label"], ascending=[True, False]).reset_index(drop=True)

# Remove helper flag from evaluator-facing CSV.
strict.drop(columns=["strict30_sample"], inplace=True)

# Real temporal version.
strict["sensitivity_subset"] = "strict30"
strict["temporal_ablation_mode"] = "real_t0_tminus90_tminus180"
strict.to_csv(REAL_OUT, index=False)

# Matched repeated-t0 version.
rep = strict.copy()
rep["original_emit_90_path"] = rep["emit_90_path"]
rep["original_emit_360_path"] = rep["emit_360_path"]
rep["emit_90_path"] = rep["emit_0_path"]
rep["emit_360_path"] = rep["emit_0_path"]
rep["temporal_ablation_mode"] = "t0_repeated_strict30_same_rows"
rep.to_csv(REP_OUT, index=False)

print("=" * 76)
print("EMIT STRICT ±30-DAY SENSITIVITY SUBSET")
print("=" * 76)
print("Source rows             :", len(df))
print("Source complete pairs   :", df["pair_id"].nunique())
print("Strict complete pairs   :", len(keep_pairs))
print("Strict rows             :", len(strict))
print("Positive                :", int((strict["label"] == 1).sum()))
print("Negative                :", int((strict["label"] == 0).sum()))

if len(strict):
    print()
    print("Temporal error range among retained rows:")
    print(
        "t-90 target error      :",
        f'{strict["t90_target_error_days"].min():.3f}',
        "to",
        f'{strict["t90_target_error_days"].max():.3f}',
        "days",
    )
    print(
        "t-180 target error     :",
        f'{strict["t180_target_error_days"].min():.3f}',
        "to",
        f'{strict["t180_target_error_days"].max():.3f}',
        "days",
    )

print()
print("Real temporal CSV       :", REAL_OUT)
print("T0-repeated CSV         :", REP_OUT)
print("Pair audit              :", AUDIT_OUT)

print()
print(
    "Repeated baseline check :",
    bool((rep["emit_90_path"] == rep["emit_0_path"]).all())
    and bool((rep["emit_360_path"] == rep["emit_0_path"]).all()),
)
