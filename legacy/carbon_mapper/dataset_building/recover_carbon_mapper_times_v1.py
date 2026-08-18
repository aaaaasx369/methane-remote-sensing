#!/usr/bin/env python3
"""
recover_carbon_mapper_times_v1.py

Recover missing Carbon Mapper acquisition timestamps from identifiers / asset
filenames WITHOUT modifying the original source CSV.

Input (default):
  ~/methane_release_project/carbon_mapper_inventory/carbon_mapper_all_CH4_plumes.csv

Outputs:
  carbon_mapper_time_recovery_v1/
    carbon_mapper_all_CH4_plumes_time_recovered.csv
    carbon_mapper_time_recovery_audit.csv
    carbon_mapper_time_recovery_summary.txt

Important:
- published_at and modified are NOT used as acquisition time.
- Full timestamps are preferred.
- Date-only recoveries are preserved separately and are NOT silently converted
  to noon/midnight timestamps.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


FULL_PATTERNS = [
    # 20240131T123456, 20240131t123456, 20240131_123456
    re.compile(r"(?<!\d)(20\d{6})[Tt_\-]?(\d{6})(?:\.\d+)?Z?(?!\d)"),
    # 2024-01-31T12:34:56 / 2024_01_31-123456 / 2024-01-31 12_34_56
    re.compile(
        r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})"
        r"[Tt _\-]+(\d{2})[:_\-]?(\d{2})[:_\-]?(\d{2})(?:\.\d+)?Z?(?!\d)"
    ),
]

DATE_PATTERNS = [
    re.compile(r"(?<!\d)(20\d{6})(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)"),
]

CANDIDATE_FIELDS = [
    "plume_id",
    "plume_tif",
    "con_tif",
    "rgb_tif",
    "plume_png",
    "rgb_png",
    "plume_name",
]


def valid_utc_timestamp(value: str):
    try:
        ts = pd.to_datetime(value, utc=True, errors="raise")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if not (pd.Timestamp("2000-01-01", tz="UTC") <= ts <= pd.Timestamp("2035-12-31 23:59:59", tz="UTC")):
        return None
    return ts


def extract_full_timestamp(text: str):
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None

    for i, pat in enumerate(FULL_PATTERNS):
        m = pat.search(text)
        if not m:
            continue

        if i == 0:
            d, t = m.group(1), m.group(2)
            value = f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}Z"
        else:
            y, mo, d, hh, mm, ss = m.groups()
            value = f"{y}-{mo}-{d}T{hh}:{mm}:{ss}Z"

        ts = valid_utc_timestamp(value)
        if ts is not None:
            return ts

    return None


def extract_date_only(text: str):
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None

    # Do not return date-only if we already have a full timestamp in this field.
    if extract_full_timestamp(text) is not None:
        return None

    for i, pat in enumerate(DATE_PATTERNS):
        m = pat.search(text)
        if not m:
            continue

        if i == 0:
            d = m.group(1)
            value = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        else:
            y, mo, d = m.groups()
            value = f"{y}-{mo}-{d}"

        try:
            dt = pd.to_datetime(value, errors="raise").date()
        except Exception:
            continue

        if 2000 <= dt.year <= 2035:
            return dt.isoformat()

    return None


def clean_original_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=str(Path.home() / "methane_release_project/carbon_mapper_inventory/carbon_mapper_all_CH4_plumes.csv"),
    )
    ap.add_argument(
        "--out",
        default=str(Path.home() / "methane_release_project/carbon_mapper_time_recovery_v1"),
    )
    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    df = pd.read_csv(inp, low_memory=False)

    if "scene_timestamp" not in df.columns:
        raise SystemExit("Expected column 'scene_timestamp' was not found.")

    original = clean_original_time(df["scene_timestamp"])

    recovered_full = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    recovered_date = pd.Series(pd.NA, index=df.index, dtype="string")
    source_field = pd.Series(pd.NA, index=df.index, dtype="string")
    precision = pd.Series(pd.NA, index=df.index, dtype="string")

    missing_idx = df.index[original.isna()]

    # First pass: recover an exact/full timestamp.
    for idx in missing_idx:
        for field in CANDIDATE_FIELDS:
            if field not in df.columns:
                continue
            text = str(df.at[idx, field])
            ts = extract_full_timestamp(text)
            if ts is not None:
                recovered_full.at[idx] = ts
                source_field.at[idx] = field
                precision.at[idx] = "FULL_TIMESTAMP"
                break

    # Second pass: retain date-only evidence but do not fabricate a clock time.
    still_missing = df.index[original.isna() & recovered_full.isna()]
    for idx in still_missing:
        for field in CANDIDATE_FIELDS:
            if field not in df.columns:
                continue
            text = str(df.at[idx, field])
            d = extract_date_only(text)
            if d is not None:
                recovered_date.at[idx] = d
                source_field.at[idx] = field
                precision.at[idx] = "DATE_ONLY"
                break

    final_full = original.copy()
    mask = final_full.isna() & recovered_full.notna()
    final_full.loc[mask] = recovered_full.loc[mask]

    df_out = df.copy()
    df_out["scene_timestamp_original_parsed"] = original
    df_out["scene_timestamp_recovered"] = recovered_full
    df_out["scene_date_recovered_only"] = recovered_date
    df_out["scene_timestamp_final"] = final_full
    df_out["scene_time_source"] = source_field
    df_out["scene_time_precision"] = precision

    # Mark original rows explicitly.
    orig_mask = original.notna()
    df_out.loc[orig_mask, "scene_time_source"] = "scene_timestamp"
    df_out.loc[orig_mask, "scene_time_precision"] = "FULL_TIMESTAMP"

    audit_cols = [
        c for c in [
            "plume_id", "scene_timestamp", "scene_timestamp_original_parsed",
            "scene_timestamp_recovered", "scene_date_recovered_only",
            "scene_timestamp_final", "scene_time_source", "scene_time_precision",
            "plume_tif", "con_tif", "rgb_tif", "plume_png", "rgb_png",
            "published_at", "modified",
        ] if c in df_out.columns
    ]

    audit = df_out.loc[
        original.isna(),
        audit_cols
    ].copy()

    recovered_csv = out_dir / "carbon_mapper_all_CH4_plumes_time_recovered.csv"
    audit_csv = out_dir / "carbon_mapper_time_recovery_audit.csv"
    summary_txt = out_dir / "carbon_mapper_time_recovery_summary.txt"

    df_out.to_csv(recovered_csv, index=False)
    audit.to_csv(audit_csv, index=False)

    n = len(df)
    n_orig = int(original.notna().sum())
    n_missing_orig = int(original.isna().sum())
    n_full_rec = int((original.isna() & recovered_full.notna()).sum())
    n_date_rec = int((original.isna() & recovered_full.isna() & recovered_date.notna()).sum())
    n_still_missing = int(
        (original.isna() & recovered_full.isna() & recovered_date.isna()).sum()
    )

    src_counts = (
        df_out.loc[original.isna() & source_field.notna(), "scene_time_source"]
        .value_counts()
        .to_dict()
    )

    summary = [
        "CARBON MAPPER TIME RECOVERY SUMMARY",
        "=" * 72,
        f"Input rows                         : {n}",
        f"Original valid scene_timestamp     : {n_orig}",
        f"Original missing scene_timestamp   : {n_missing_orig}",
        f"Recovered FULL timestamps          : {n_full_rec}",
        f"Recovered DATE-ONLY evidence       : {n_date_rec}",
        f"Still missing any acquisition date : {n_still_missing}",
        f"Rows with exact/full final time    : {int(final_full.notna().sum())}",
        "",
        "Recovery source fields:",
    ]
    for k, v in src_counts.items():
        summary.append(f"  {k:24s} {v}")

    summary += [
        "",
        "IMPORTANT",
        "- published_at and modified were NOT used as acquisition timestamps.",
        "- DATE_ONLY rows are not converted to an arbitrary clock time.",
        "- The original Carbon Mapper CSV was not modified.",
        "",
        "Saved:",
        str(recovered_csv),
        str(audit_csv),
    ]

    summary_txt.write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
