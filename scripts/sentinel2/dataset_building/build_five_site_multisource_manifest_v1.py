#!/usr/bin/env python3
"""
Build the five-site multisource Sentinel-2 manifest.

Combines:
1. Existing strict controlled-release Sentinel-2 rows from
   outputs/390_multisensor_master_manifest_v1.csv
2. Selected MethaneAIR positive scenes from
   outputs/512_selected_methaneair_positive_manifest_v2.csv
3. Downloaded MethaneAIR reference negatives from
   outputs/517_s2_negative_manifest_v1.csv

Output:
    outputs/519_five_site_multisource_manifest_v1.csv
    outputs/520_five_site_multisource_audit_v1.csv
    outputs/521_five_site_multisource_report_v1.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


TARGET_SITES = [
    "Casa_Grande_AZ_release_stacks",
    "Ehrenberg_AZ_release_stack",
    "MethaneAIR_site_073",
    "MethaneAIR_site_102",
    "MethaneAIR_site_120",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine strict and observational Sentinel-2 rows into a five-site manifest."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/Users/happydoraaa/methane_release_project"),
    )
    return parser.parse_args()


def first_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    lower = {str(column).strip().lower(): column for column in df.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def canonicalize_strict(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "sample_id": ("sample_id", "event_id", "id"),
        "site_id": ("site_id", "site", "site_name"),
        "scene_id": ("scene_id", "s2_scene_id", "image_id", "system_index"),
        "acquisition_time_utc": (
            "acquisition_time_utc", "datetime_utc", "scene_time_utc"
        ),
        "release_rate_kg_h": (
            "release_rate_kg_h", "emission_kg_hr", "emission_kg_h",
            "matched_positive_release_rate_kg_h"
        ),
        "label": ("label", "final_label"),
        "patch_path": (
            "resolved_patch_path", "patch_path", "relative_path",
            "file_path", "filename"
        ),
        "latitude": ("latitude", "lat", "source_latitude"),
        "longitude": ("longitude", "lon", "source_longitude"),
    }
    out = pd.DataFrame(index=df.index)
    for target, names in aliases.items():
        column = first_column(df, names)
        out[target] = df[column] if column is not None else np.nan

    out["source_origin"] = "strict_controlled_release"
    out["ground_truth_type"] = "metered_controlled_release"
    out["negative_confidence"] = np.where(
        pd.to_numeric(out["label"], errors="coerce").eq(0),
        "matched_controlled_release_negative",
        "",
    )
    return out


def canonicalize_positive(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "sample_id": ("event_id_canonical", "event_id", "sample_id"),
        "site_id": ("site_id",),
        "scene_id": ("scene_id", "s2_scene_id"),
        "acquisition_time_utc": (
            "s2_acquisition_time_utc", "acquisition_time_utc",
            "datetime_utc_canonical"
        ),
        "release_rate_kg_h": (
            "emission_kg_h_canonical", "emission_kg_hr"
        ),
        "patch_path": (
            "resolved_patch_path", "patch_path_canonical", "patch_path"
        ),
        "latitude": ("site_centroid_latitude", "latitude_canonical", "latitude"),
        "longitude": (
            "site_centroid_longitude", "longitude_canonical", "longitude"
        ),
    }
    out = pd.DataFrame(index=df.index)
    for target, names in aliases.items():
        column = first_column(df, names)
        out[target] = df[column] if column is not None else np.nan

    out["label"] = 1
    out["source_origin"] = "MethaneAIR"
    out["ground_truth_type"] = "observational_plume"
    out["negative_confidence"] = ""
    return out


def canonicalize_negative(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "sample_id": ("sample_id",),
        "site_id": ("site_id",),
        "scene_id": ("s2_scene_id", "scene_id"),
        "acquisition_time_utc": ("s2_time_utc", "acquisition_time_utc"),
        "release_rate_kg_h": ("release_rate_kg_h",),
        "patch_path": ("patch_path",),
        "latitude": ("latitude",),
        "longitude": ("longitude",),
        "label": ("label",),
        "source_origin": ("source_origin",),
        "ground_truth_type": ("ground_truth_type",),
        "negative_confidence": ("negative_confidence",),
    }
    out = pd.DataFrame(index=df.index)
    for target, names in aliases.items():
        column = first_column(df, names)
        out[target] = df[column] if column is not None else np.nan
    return out


def clean_manifest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    for column in (
        "sample_id", "site_id", "scene_id", "source_origin",
        "ground_truth_type", "negative_confidence", "patch_path"
    ):
        df[column] = df[column].astype("string").str.strip()

    df["acquisition_time_utc"] = pd.to_datetime(
        df["acquisition_time_utc"], errors="coerce", utc=True
    )
    df["release_rate_kg_h"] = pd.to_numeric(
        df["release_rate_kg_h"], errors="coerce"
    )
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df["patch_exists_now"] = df["patch_path"].map(
        lambda value: Path(str(value)).expanduser().exists()
        if pd.notna(value) and str(value).strip()
        else False
    )

    # One scene per site/label; keep the highest-emission positive if duplicated.
    df = df.sort_values(
        ["site_id", "scene_id", "label", "release_rate_kg_h"],
        ascending=[True, True, True, False],
        na_position="last",
    )
    df = df.drop_duplicates(
        subset=["site_id", "scene_id", "label"], keep="first"
    )

    missing_sample = df["sample_id"].isna() | df["sample_id"].eq("")
    df.loc[missing_sample, "sample_id"] = [
        f"five_site_sample_{index:05d}"
        for index in range(missing_sample.sum())
    ]

    return df.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    outputs = root / "outputs"

    strict_path = outputs / "390_multisensor_master_manifest_v1.csv"
    positive_path = outputs / "512_selected_methaneair_positive_manifest_v2.csv"
    negative_path = outputs / "517_s2_negative_manifest_v1.csv"

    for path in (strict_path, positive_path, negative_path):
        if not path.exists():
            raise SystemExit(f"Required input not found: {path}")

    strict = canonicalize_strict(pd.read_csv(strict_path))
    positive = canonicalize_positive(pd.read_csv(positive_path))
    negative_raw = pd.read_csv(negative_path)
    if "download_ok" in negative_raw.columns:
        download_ok = negative_raw["download_ok"]
        if download_ok.dtype == object:
            download_ok = download_ok.astype(str).str.lower().isin(
                ["true", "1", "yes"]
            )
        negative_raw = negative_raw[download_ok].copy()
    negative = canonicalize_negative(negative_raw)

    combined = clean_manifest(
        pd.concat([strict, positive, negative], ignore_index=True, sort=False)
    )
    combined = combined[
        combined["site_id"].astype(str).isin(TARGET_SITES)
    ].copy()

    manifest_path = outputs / "519_five_site_multisource_manifest_v1.csv"
    combined.to_csv(manifest_path, index=False)

    audit = (
        combined.groupby(
            ["site_id", "source_origin", "ground_truth_type"], dropna=False
        )
        .agg(
            rows=("sample_id", "size"),
            positive=("label", lambda values: int((values == 1).sum())),
            negative=("label", lambda values: int((values == 0).sum())),
            unique_scenes=("scene_id", "nunique"),
            readable_patches=("patch_exists_now", "sum"),
            min_emission_kg_h=("release_rate_kg_h", "min"),
            median_emission_kg_h=("release_rate_kg_h", "median"),
            max_emission_kg_h=("release_rate_kg_h", "max"),
        )
        .reset_index()
    )
    audit["has_both_classes"] = (
        audit["positive"].gt(0) & audit["negative"].gt(0)
    )
    audit_path = outputs / "520_five_site_multisource_audit_v1.csv"
    audit.to_csv(audit_path, index=False)

    report_lines = [
        "=" * 110,
        "FIVE-SITE MULTISOURCE SENTINEL-2 MANIFEST V1",
        "=" * 110,
        "",
        f"Total rows: {len(combined)}",
        f"Positive: {int((combined['label'] == 1).sum())}",
        f"Negative: {int((combined['label'] == 0).sum())}",
        f"Unique sites: {combined['site_id'].nunique()}",
        f"Unique sources: {combined['source_origin'].nunique()}",
        f"Readable patches: {int(combined['patch_exists_now'].sum())}",
        "",
        "SITE AUDIT",
        "-" * 110,
        audit.to_string(index=False),
        "",
        "GROUND-TRUTH WARNING",
        "-" * 110,
        "Casa Grande and Ehrenberg use metered controlled-release ground truth.",
        "The three MethaneAIR sites use observational plume positives and",
        "no-known-plume reference negatives. Therefore this is a five-site",
        "multisource benchmark, not a five-site controlled-release benchmark.",
        "",
        "NEXT COMMAND",
        "-" * 110,
        "python run_multisource_s2_model_v2.py \\",
        "  --project-root /Users/happydoraaa/methane_release_project \\",
        "  --input outputs/519_five_site_multisource_manifest_v1.csv",
    ]
    report_path = outputs / "521_five_site_multisource_report_v1.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\nCreated:")
    print(manifest_path)
    print(audit_path)
    print(report_path)
    print("\nAudit:")
    print(audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
