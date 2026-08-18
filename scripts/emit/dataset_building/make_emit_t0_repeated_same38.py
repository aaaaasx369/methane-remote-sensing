#!/usr/bin/env python3
"""
Create a matched t0-repeated baseline from the SAME 38-row EMIT real-temporal
evaluation set.

Input:
  EMIT_MethaneFuse_480m_TEMPORAL_60D/eval_emit_480m_abs.csv

Output:
  EMIT_MethaneFuse_480m_TEMPORAL_60D/eval_emit_480m_abs_t0repeated.csv

For every row:
  emit_0_path   = original real t0
  emit_90_path  = same real t0
  emit_360_path = same real t0

Everything else (rows, labels, pair IDs, sites, checkpoint interface) stays
identical, so the only manipulated factor is temporal context.
"""

from pathlib import Path
import pandas as pd

ROOT = Path.home() / "methane_release_project" / "EMIT_MethaneFuse_480m_TEMPORAL_60D"
SRC = ROOT / "eval_emit_480m_abs.csv"
DST = ROOT / "eval_emit_480m_abs_t0repeated.csv"

if not SRC.exists():
    raise SystemExit(f"Missing input CSV: {SRC}")

df = pd.read_csv(SRC)

required = ["id", "label", "pair_id", "emit_0_path", "emit_90_path", "emit_360_path"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

orig_90 = df["emit_90_path"].copy()
orig_180 = df["emit_360_path"].copy()

# Matched ablation: replace historical frames with t0 only.
df["emit_90_path"] = df["emit_0_path"]
df["emit_360_path"] = df["emit_0_path"]

# Explicit provenance columns.
df["ablation_mode"] = "t0_repeated_same38"
df["original_emit_90_path"] = orig_90
df["original_emit_360_path"] = orig_180

df.to_csv(DST, index=False)

print("=" * 72)
print("T0-REPEATED MATCHED BASELINE CREATED")
print("=" * 72)
print("Rows      :", len(df))
print("Positive  :", int((df["label"] == 1).sum()))
print("Negative  :", int((df["label"] == 0).sum()))
print("Pairs     :", df["pair_id"].nunique())
print()
print("Unique t0 paths    :", df["emit_0_path"].nunique())
print("Unique t90 paths   :", df["emit_90_path"].nunique())
print("Unique t180 paths  :", df["emit_360_path"].nunique())
print()
print("All t90 == t0   :", bool((df["emit_90_path"] == df["emit_0_path"]).all()))
print("All t180 == t0  :", bool((df["emit_360_path"] == df["emit_0_path"]).all()))
print()
print("Output:", DST)
