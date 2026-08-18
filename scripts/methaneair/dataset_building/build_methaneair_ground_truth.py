#!/usr/bin/env python3
"""
Build a unified ground-truth table for MethaneAIR/Sentinel-2 matching.

Confirmed labels:
- MethaneAIR L4 point-source detections: positive observational labels.
- Existing physical/controlled-release records: positive or negative labels.

Candidate labels:
- Same-site dates after MethaneAIR positives are proposed negatives only.
  They remain unconfirmed until known releases, known plumes and cloud/snow
  contamination have been excluded.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path("/project/6002520/yunjung1/MethaneFuse"),
    )
    p.add_argument(
        "--master",
        type=Path,
        default=None,
    )
    return p.parse_args()


def first_present(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    mapping = {
        re.sub(r"[^a-z0-9]+", "_", str(col).lower()).strip("_"): col
        for col in df.columns
    }
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "_", alias.lower()).strip("_")
        if key in mapping:
            return df[mapping[key]]
    return pd.Series([pd.NA] * len(df), index=df.index)


def canonical_master(master: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=master.index)

    out["record_id"] = first_present(
        master,
        ["master_id", "sample_id", "record_id", "source_row_ids"],
    )
    out["site_id"] = first_present(
        master,
        ["site_id", "site", "facility_id", "location"],
    )
    out["latitude"] = pd.to_numeric(
        first_present(
            master,
            ["latitude", "lat", "source_latitude", "center_lat"],
        ),
        errors="coerce",
    )
    out["longitude"] = pd.to_numeric(
        first_present(
            master,
            ["longitude", "lon", "lng", "source_longitude", "center_lon"],
        ),
        errors="coerce",
    )
    out["acquisition_time_utc"] = pd.to_datetime(
        first_present(
            master,
            [
                "acquisition_time_utc",
                "acquisition_time",
                "scene_timestamp",
                "timestamp",
            ],
        ),
        errors="coerce",
        utc=True,
    )
    out["label"] = pd.to_numeric(
        first_present(master, ["label", "ground_truth", "target"]),
        errors="coerce",
    ).astype("Int64")
    out["ground_truth_source"] = first_present(
        master,
        ["ground_truth_source", "source_name", "dataset_source"],
    ).astype("string")
    out["ground_truth_type"] = first_present(
        master,
        ["ground_truth_type", "label_type"],
    ).astype("string")
    out["emission_rate_kg_hr"] = pd.to_numeric(
        first_present(
            master,
            [
                "emission_rate_kg_hr",
                "release_rate_kg_hr",
                "ground_truth_rate_kg_hr",
            ],
        ),
        errors="coerce",
    )

    source_text = (
        out["ground_truth_source"].fillna("")
        + " "
        + out["ground_truth_type"].fillna("")
    ).str.lower()

    controlled = source_text.str.contains(
        r"physical_release|physical release|controlled|matched benchmark",
        regex=True,
    )

    out = out[
        controlled
        & out["label"].isin([0, 1])
    ].copy()

    out["label_status"] = "confirmed"
    out["proposed_label"] = pd.NA
    out["label_confidence"] = "high"
    out["controlled_release_verified"] = True
    out["sensor_ground_truth"] = "controlled_release"
    out["source_file_group"] = "existing_unified_master"
    out["acquisition_time_utc"] = out[
        "acquisition_time_utc"
    ].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return out


def canonical_methaneair(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame()

    out["record_id"] = df["record_id"].astype("string")
    out["site_id"] = df.get("basin", pd.Series([pd.NA] * len(df)))
    out["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    out["acquisition_time_utc"] = pd.to_datetime(
        df["time_coverage_start"],
        errors="coerce",
        utc=True,
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["label"] = 1
    out["proposed_label"] = pd.NA
    out["label_status"] = "confirmed_observational_positive"
    out["ground_truth_source"] = "MethaneAIR_L4_point_sources"
    out["ground_truth_type"] = "MethaneAIR_observational_detection"
    out["label_confidence"] = "medium"
    out["controlled_release_verified"] = False
    out["sensor_ground_truth"] = "MethaneAIR"
    out["source_file_group"] = "official_methaneair_l4"
    out["emission_rate_kg_hr"] = pd.to_numeric(
        df.get("flux_kg_hr"),
        errors="coerce",
    )
    out["flight_id"] = df.get("flight_id")
    out["plume_id"] = df.get("plume_id")
    return out


def build_negative_candidates(positives: pd.DataFrame) -> pd.DataFrame:
    offsets = [1, 3, 7, 14]
    rows = []

    for _, row in positives.iterrows():
        timestamp = pd.to_datetime(
            row["acquisition_time_utc"],
            errors="coerce",
            utc=True,
        )
        if pd.isna(timestamp):
            continue
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            continue

        for offset in offsets:
            target = timestamp + pd.Timedelta(days=offset)
            rows.append(
                {
                    "record_id": (
                        f"{row['record_id']}__candidate_negative_p{offset}d"
                    ),
                    "site_id": row.get("site_id"),
                    "latitude": row.get("latitude"),
                    "longitude": row.get("longitude"),
                    "acquisition_time_utc": target.strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "label": pd.NA,
                    "proposed_label": 0,
                    "label_status": "candidate_unconfirmed",
                    "ground_truth_source": "MethaneAIR_temporal_candidate",
                    "ground_truth_type": "same_site_temporal_candidate",
                    "label_confidence": "pending",
                    "controlled_release_verified": False,
                    "sensor_ground_truth": "constructed_candidate",
                    "source_file_group": "generated_from_methaneair_positive",
                    "emission_rate_kg_hr": pd.NA,
                    "source_positive_record_id": row["record_id"],
                    "days_after_positive": offset,
                    "known_release_excluded": False,
                    "known_plume_excluded": False,
                    "nearby_plume_excluded": False,
                    "cloud_snow_qa_pass": False,
                    "negative_validity": "pending_validation",
                }
            )

    return pd.DataFrame(rows)


def find_master(project_root: Path, supplied: Path | None) -> Path | None:
    if supplied:
        return supplied.expanduser().resolve()

    candidates = [
        project_root / "outputs/027_unified_methane_master_landsat_recovered.csv",
        project_root / "outputs/012_unified_methane_master_fixed.csv",
        project_root / "outputs/002_unified_methane_master_dedup.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_root = project_root / "data" / "methaneair_full"
    output_root.mkdir(parents=True, exist_ok=True)

    l4_path = output_root / "methaneair_l4_points.csv"
    if not l4_path.exists():
        raise FileNotFoundError(
            f"MethaneAIR L4 table not found: {l4_path}\n"
            "Run export_methaneair_gee.py first."
        )

    methaneair = canonical_methaneair(l4_path)

    master_path = find_master(project_root, args.master)
    controlled = pd.DataFrame()
    if master_path and master_path.exists():
        controlled = canonical_master(pd.read_csv(master_path))

    confirmed = pd.concat(
        [methaneair, controlled],
        ignore_index=True,
        sort=False,
    )
    confirmed = confirmed.drop_duplicates(
        subset=[
            "record_id",
            "acquisition_time_utc",
            "latitude",
            "longitude",
            "label",
        ],
        keep="first",
    )

    candidates = build_negative_candidates(methaneair)

    confirmed_path = output_root / "ground_truth_confirmed.csv"
    candidate_path = output_root / "ground_truth_negative_candidates.csv"
    all_path = output_root / "ground_truth_all.csv"
    summary_path = output_root / "ground_truth_summary.csv"

    confirmed.to_csv(confirmed_path, index=False)
    candidates.to_csv(candidate_path, index=False)
    pd.concat(
        [confirmed, candidates],
        ignore_index=True,
        sort=False,
    ).to_csv(all_path, index=False)

    summary_rows = []
    for name, df in [
        ("confirmed", confirmed),
        ("negative_candidates", candidates),
    ]:
        labels = pd.to_numeric(df.get("label"), errors="coerce")
        summary_rows.append(
            {
                "group": name,
                "rows": len(df),
                "positive": int((labels == 1).sum()),
                "negative": int((labels == 0).sum()),
                "unlabeled": int(labels.isna().sum()),
                "with_coordinates": int(
                    df["latitude"].notna().mul(
                        df["longitude"].notna()
                    ).sum()
                ),
                "with_time": int(
                    df["acquisition_time_utc"].notna().sum()
                ),
            }
        )

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("Confirmed ground truth:")
    print(confirmed["label"].value_counts(dropna=False).to_string())
    print(f"Confirmed rows: {len(confirmed)}")
    print(f"Candidate negatives: {len(candidates)}")
    print(f"Saved: {confirmed_path}")
    print(f"Saved: {candidate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
