#!/usr/bin/env python3
"""
build_enmap_download_phases_v1.py

Turn the frozen EnMAP download-priority manifest into concrete phased download
lists, prioritizing:
  1) strict temporal proximity,
  2) nominal L2A quality,
  3) cross-source support,
  4) controlled-release / high-confidence sources,
  5) smaller absolute time difference.

No imagery is downloaded.

Input:
  ~/methane_release_project/enmap_download_manifests_v1/
      12_download_priority_manifest.csv

Outputs:
  ~/methane_release_project/enmap_download_phases_v1/
      01_phase1_strict_A_nominal.csv
      02_phase2_primary_increment_B_nominal.csv
      03_phase2_primary_total_AB_nominal.csv
      04_phase3_expanded_increment_C_nominal.csv
      05_phase3_expanded_total_ABC_nominal.csv
      06_cross_source_nominal_priority.csv
      07_high_confidence_source_nominal_priority.csv
      08_phase_summary.csv
      enmap_download_phases_summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


HIGH_CONFIDENCE_DATASETS = [
    "CONTROLLED_RELEASE_VERIFIED_107",
    "STANFORD_2024_2025_746",
]

SECONDARY_PRIORITY_DATASETS = [
    "METHANEAIR_435",
    "METHANESAT_POSNEG_222",
    "AVIRIS3_SCENES_493",
    "EMIT_POSNEG_100",
]

def contains_any(series: pd.Series, names) -> pd.Series:
    s = series.fillna("").astype(str)
    out = pd.Series(False, index=s.index)
    for name in names:
        out |= s.str.contains(name, regex=False)
    return out

def add_priority_fields(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["is_cross_source"] = (
        pd.to_numeric(d["supporting_datasets_count"], errors="coerce")
        .fillna(0)
        .ge(2)
    )

    d["has_high_confidence_source"] = contains_any(
        d["supporting_datasets"], HIGH_CONFIDENCE_DATASETS
    )
    d["has_secondary_priority_source"] = contains_any(
        d["supporting_datasets"], SECONDARY_PRIORITY_DATASETS
    )

    # Smaller number = higher priority.
    d["source_priority_rank"] = 3
    d.loc[d["has_secondary_priority_source"], "source_priority_rank"] = 2
    d.loc[d["has_high_confidence_source"], "source_priority_rank"] = 1
    d.loc[d["is_cross_source"], "source_priority_rank"] = 0

    d["min_abs_delta_hours"] = pd.to_numeric(
        d["min_abs_delta_hours"], errors="coerce"
    )

    d = d.sort_values(
        [
            "source_priority_rank",
            "supporting_datasets_count",
            "min_abs_delta_hours",
            "supporting_records",
        ],
        ascending=[True, False, True, False],
    )
    return d

def save(df, path):
    df.to_csv(path, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=str(
            Path.home()
            / "methane_release_project/enmap_download_manifests_v1/"
              "12_download_priority_manifest.csv"
        ),
    )
    ap.add_argument(
        "--out",
        default=str(
            Path.home()
            / "methane_release_project/enmap_download_phases_v1"
        ),
    )
    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    df = pd.read_csv(inp, low_memory=False)

    nominal = df[
        (df["quality_action"] == "PREFERRED")
        & (df["download_recommendation"] == "DOWNLOAD")
    ].copy()

    nominal = add_priority_fields(nominal)

    phase1 = nominal[nominal["tier"] == "A_LE24H"].copy()
    phase2_inc = nominal[nominal["tier"] == "B_LE72H"].copy()
    phase2_total = nominal[nominal["tier"].isin(["A_LE24H", "B_LE72H"])].copy()
    phase3_inc = nominal[nominal["tier"] == "C_LE7D"].copy()
    phase3_total = nominal[
        nominal["tier"].isin(["A_LE24H", "B_LE72H", "C_LE7D"])
    ].copy()

    cross_source = nominal[nominal["is_cross_source"]].copy()

    high_conf = nominal[
        nominal["has_high_confidence_source"]
        | nominal["has_secondary_priority_source"]
    ].copy()

    save(phase1, out / "01_phase1_strict_A_nominal.csv")
    save(phase2_inc, out / "02_phase2_primary_increment_B_nominal.csv")
    save(phase2_total, out / "03_phase2_primary_total_AB_nominal.csv")
    save(phase3_inc, out / "04_phase3_expanded_increment_C_nominal.csv")
    save(phase3_total, out / "05_phase3_expanded_total_ABC_nominal.csv")
    save(cross_source, out / "06_cross_source_nominal_priority.csv")
    save(high_conf, out / "07_high_confidence_source_nominal_priority.csv")

    summary_rows = [
        {
            "phase": "Phase 1 strict",
            "definition": "A_LE24H + nominal",
            "unique_l2a_scenes": len(phase1),
        },
        {
            "phase": "Phase 2 increment",
            "definition": "B_LE72H + nominal only",
            "unique_l2a_scenes": len(phase2_inc),
        },
        {
            "phase": "Phase 2 primary total",
            "definition": "A+B <=72h + nominal",
            "unique_l2a_scenes": len(phase2_total),
        },
        {
            "phase": "Phase 3 increment",
            "definition": "C_LE7D + nominal only",
            "unique_l2a_scenes": len(phase3_inc),
        },
        {
            "phase": "Phase 3 expanded total",
            "definition": "A+B+C <=7d + nominal",
            "unique_l2a_scenes": len(phase3_total),
        },
        {
            "phase": "Cross-source nominal priority",
            "definition": ">=2 supporting datasets",
            "unique_l2a_scenes": len(cross_source),
        },
        {
            "phase": "High-confidence/secondary-source nominal priority",
            "definition": "controlled-release, Stanford, MethaneAIR, MethaneSAT, AVIRIS3, or EMIT support",
            "unique_l2a_scenes": len(high_conf),
        },
    ]

    summary = pd.DataFrame(summary_rows)
    save(summary, out / "08_phase_summary.csv")

    lines = [
        "ENMAP DOWNLOAD PHASES SUMMARY",
        "=" * 80,
        f"Phase 1 strict A nominal             : {len(phase1)}",
        f"Phase 2 additional B nominal         : {len(phase2_inc)}",
        f"Phase 2 primary total A+B nominal    : {len(phase2_total)}",
        f"Phase 3 additional C nominal         : {len(phase3_inc)}",
        f"Phase 3 expanded total A+B+C nominal : {len(phase3_total)}",
        f"Cross-source nominal scenes          : {len(cross_source)}",
        f"High-confidence/secondary priority   : {len(high_conf)}",
        "",
        "RECOMMENDED DOWNLOAD ORDER",
        "1. 06_cross_source_nominal_priority.csv",
        "2. 07_high_confidence_source_nominal_priority.csv",
        "3. 01_phase1_strict_A_nominal.csv",
        "4. 02_phase2_primary_increment_B_nominal.csv",
        "5. 04_phase3_expanded_increment_C_nominal.csv",
        "",
        "Notes:",
        "- Files overlap by design; do not download duplicates by file row count.",
        "- l2a_scene_id is the unique scene key.",
        "- Phase 2 total is the recommended main <=72 h nominal dataset.",
        "- Phase 3 is an expanded sensitivity set.",
        "- Reduced/low-quality scenes are intentionally not included here.",
    ]

    (out / "enmap_download_phases_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"\nSaved to: {out}")

if __name__ == "__main__":
    main()
