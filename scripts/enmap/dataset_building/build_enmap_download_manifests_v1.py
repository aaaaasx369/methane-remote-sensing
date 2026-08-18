#!/usr/bin/env python3
"""
build_enmap_download_manifests_v1.py

Build EnMAP download candidate manifests from the frozen V4 match results.

Input (default)
---------------
~/methane_release_project/enmap_full_match_v1/enmap_match_results_v4.csv

Outputs
-------
~/methane_release_project/enmap_download_manifests_v1/
    01_sample_level_all_matched.csv
    02_tier_A_within24h_l2a.csv
    03_tier_B_within72h_l2a.csv
    04_tier_C_within7d_l2a.csv
    05_tier_D_reference_gt30d_l2a.csv
    06_unique_l2a_scenes_A.csv
    07_unique_l2a_scenes_B.csv
    08_unique_l2a_scenes_C.csv
    09_unique_l2a_scenes_D_reference.csv
    10_quality_summary_by_tier_dataset.csv
    11_unique_scene_summary_by_tier_dataset.csv
    12_download_priority_manifest.csv
    enmap_download_manifest_summary.txt

Important
---------
- No imagery is downloaded.
- Methane labels are preserved exactly as provided by the source manifests.
- "Reference" >30 d rows are NOT asserted to be confirmed negatives.
- Quality policy is explicit:
    NOMINAL  -> preferred
    REDUCED  -> review
    LOW      -> exclude by default
    UNKNOWN  -> review
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return (
        s.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def quality_class(df: pd.DataFrame) -> pd.Series:
    if "l2a_overall_quality" in df.columns:
        q = df["l2a_overall_quality"].astype("string").str.strip().str.upper()
    else:
        q = pd.Series(pd.NA, index=df.index, dtype="string")

    # Fall back to numeric code if text label is absent.
    if "l2a_overall_quality_code" in df.columns:
        code = pd.to_numeric(df["l2a_overall_quality_code"], errors="coerce")
        q = q.where(q.notna() & ~q.isin(["", "<NA>", "NAN", "NONE"]))
        q = q.fillna(
            code.map({0: "NOMINAL", 1: "REDUCED", 2: "LOW"})
        )

    q = q.replace({
        "0": "NOMINAL",
        "1": "REDUCED",
        "2": "LOW",
        "NAN": pd.NA,
        "NONE": pd.NA,
        "<NA>": pd.NA,
        "": pd.NA,
    })

    return q.fillna("UNKNOWN")


def quality_action(q: pd.Series) -> pd.Series:
    return q.map({
        "NOMINAL": "PREFERRED",
        "REDUCED": "REVIEW",
        "LOW": "EXCLUDE_LOW_QUALITY",
        "UNKNOWN": "REVIEW_UNKNOWN_QUALITY",
    }).fillna("REVIEW_UNKNOWN_QUALITY")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in ["within_24h", "within_72h", "within_7d", "within_30d",
              "l2a_same_acquisition_available"]:
        if c not in df.columns:
            df[c] = False
        df[c] = as_bool(df[c])

    df["event_time_utc"] = pd.to_datetime(
        df.get("event_time_utc"), errors="coerce", utc=True
    )
    df["l0_nearest_time"] = pd.to_datetime(
        df.get("l0_nearest_time"), errors="coerce", utc=True
    )
    df["l2a_time"] = pd.to_datetime(
        df.get("l2a_time"), errors="coerce", utc=True
    )

    df["quality_class"] = quality_class(df)
    df["quality_action"] = quality_action(df["quality_class"])

    abs_h = pd.to_numeric(df.get("l0_abs_delta_hours"), errors="coerce")
    df["abs_delta_hours"] = abs_h

    # Mutually exclusive temporal band.
    df["temporal_band"] = "UNCLASSIFIED"
    df.loc[abs_h <= 24, "temporal_band"] = "LE_24H"
    df.loc[(abs_h > 24) & (abs_h <= 72), "temporal_band"] = "GT24_LE72H"
    df.loc[(abs_h > 72) & (abs_h <= 24*7), "temporal_band"] = "GT72H_LE7D"
    df.loc[(abs_h > 24*7) & (abs_h <= 24*30), "temporal_band"] = "GT7D_LE30D"
    df.loc[abs_h > 24*30, "temporal_band"] = "GT30D"

    return df


def unique_scene_manifest(df: pd.DataFrame, tier_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    # Prefer l2a_scene_id as unique product identity.
    key_cols = ["l2a_scene_id"]
    d = df[df["l2a_scene_id"].notna()].copy()

    if d.empty:
        return pd.DataFrame()

    rows = []
    for scene_id, g in d.groupby("l2a_scene_id", dropna=False):
        datasets = sorted(g["dataset"].dropna().astype(str).unique().tolist())
        record_ids = g["record_id"].dropna().astype(str).tolist()

        row = {
            "tier": tier_name,
            "l2a_scene_id": scene_id,
            "l2a_time": g["l2a_time"].dropna().min(),
            "l2a_datatake_id": (
                g["l2a_datatake_id"].dropna().astype(str).iloc[0]
                if "l2a_datatake_id" in g.columns and g["l2a_datatake_id"].notna().any()
                else pd.NA
            ),
            "quality_class": (
                g["quality_class"].dropna().astype(str).iloc[0]
                if g["quality_class"].notna().any()
                else "UNKNOWN"
            ),
            "quality_action": (
                g["quality_action"].dropna().astype(str).iloc[0]
                if g["quality_action"].notna().any()
                else "REVIEW_UNKNOWN_QUALITY"
            ),
            "l2a_cloud_cover": (
                pd.to_numeric(g["l2a_cloud_cover"], errors="coerce").median()
                if "l2a_cloud_cover" in g.columns
                else pd.NA
            ),
            "l2a_snow_cover": (
                pd.to_numeric(g["l2a_snow_cover"], errors="coerce").median()
                if "l2a_snow_cover" in g.columns
                else pd.NA
            ),
            "supporting_records": len(g),
            "supporting_datasets_count": len(datasets),
            "supporting_datasets": " | ".join(datasets),
            "supporting_record_ids": " | ".join(record_ids[:100]),
            "min_abs_delta_hours": pd.to_numeric(
                g["abs_delta_hours"], errors="coerce"
            ).min(),
            "median_abs_delta_hours": pd.to_numeric(
                g["abs_delta_hours"], errors="coerce"
            ).median(),
            "positive_records": int(
                (pd.to_numeric(g.get("label"), errors="coerce") == 1).sum()
            ),
            "negative_records": int(
                (pd.to_numeric(g.get("label"), errors="coerce") == 0).sum()
            ),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        pref_order = {
            "PREFERRED": 0,
            "REVIEW": 1,
            "REVIEW_UNKNOWN_QUALITY": 2,
            "EXCLUDE_LOW_QUALITY": 3,
        }
        out["_pref"] = out["quality_action"].map(pref_order).fillna(9)
        out = out.sort_values(
            ["_pref", "min_abs_delta_hours", "supporting_records"],
            ascending=[True, True, False],
        ).drop(columns="_pref")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=str(
            Path.home()
            / "methane_release_project/enmap_full_match_v1/enmap_match_results_v4.csv"
        ),
    )
    ap.add_argument(
        "--out",
        default=str(
            Path.home()
            / "methane_release_project/enmap_download_manifests_v1"
        ),
    )
    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    df = pd.read_csv(inp, low_memory=False)
    df = prepare(df)

    # Only rows with actual same-acquisition L2A availability can be download candidates.
    matched = df[df["l2a_same_acquisition_available"]].copy()

    tier_a = matched[matched["within_24h"]].copy()
    tier_b = matched[matched["within_72h"]].copy()
    tier_c = matched[matched["within_7d"]].copy()

    # Reference candidates: spatially covered, nearest acquisition >30d,
    # AND same-acquisition L2A exists. These are NOT negative labels.
    tier_d = matched[
        (pd.to_numeric(matched["abs_delta_hours"], errors="coerce") > 24*30)
    ].copy()

    matched.to_csv(out_dir / "01_sample_level_all_matched.csv", index=False)
    tier_a.to_csv(out_dir / "02_tier_A_within24h_l2a.csv", index=False)
    tier_b.to_csv(out_dir / "03_tier_B_within72h_l2a.csv", index=False)
    tier_c.to_csv(out_dir / "04_tier_C_within7d_l2a.csv", index=False)
    tier_d.to_csv(out_dir / "05_tier_D_reference_gt30d_l2a.csv", index=False)

    unique_a = unique_scene_manifest(tier_a, "A_LE24H")
    unique_b = unique_scene_manifest(tier_b, "B_LE72H")
    unique_c = unique_scene_manifest(tier_c, "C_LE7D")
    unique_d = unique_scene_manifest(tier_d, "D_REFERENCE_GT30D")

    unique_a.to_csv(out_dir / "06_unique_l2a_scenes_A.csv", index=False)
    unique_b.to_csv(out_dir / "07_unique_l2a_scenes_B.csv", index=False)
    unique_c.to_csv(out_dir / "08_unique_l2a_scenes_C.csv", index=False)
    unique_d.to_csv(out_dir / "09_unique_l2a_scenes_D_reference.csv", index=False)

    # Quality summary by tier and dataset.
    tier_frames = []
    for name, d in [
        ("A_LE24H", tier_a),
        ("B_LE72H", tier_b),
        ("C_LE7D", tier_c),
        ("D_REFERENCE_GT30D", tier_d),
    ]:
        if d.empty:
            continue
        x = (
            d.groupby(["dataset", "quality_class"], dropna=False)
            .size()
            .reset_index(name="sample_records")
        )
        x.insert(0, "tier", name)
        tier_frames.append(x)

    quality_summary = (
        pd.concat(tier_frames, ignore_index=True)
        if tier_frames
        else pd.DataFrame(
            columns=["tier", "dataset", "quality_class", "sample_records"]
        )
    )
    quality_summary.to_csv(
        out_dir / "10_quality_summary_by_tier_dataset.csv", index=False
    )

    # Unique-scene summary by tier and supporting dataset.
    unique_summary_rows = []
    for tier_name, d in [
        ("A_LE24H", tier_a),
        ("B_LE72H", tier_b),
        ("C_LE7D", tier_c),
        ("D_REFERENCE_GT30D", tier_d),
    ]:
        for dataset, g in d.groupby("dataset", dropna=False):
            unique_summary_rows.append({
                "tier": tier_name,
                "dataset": dataset,
                "sample_records": len(g),
                "unique_l2a_scenes": int(g["l2a_scene_id"].nunique(dropna=True)),
                "nominal_records": int((g["quality_class"] == "NOMINAL").sum()),
                "reduced_records": int((g["quality_class"] == "REDUCED").sum()),
                "low_records": int((g["quality_class"] == "LOW").sum()),
                "unknown_quality_records": int((g["quality_class"] == "UNKNOWN").sum()),
            })
    unique_summary = pd.DataFrame(unique_summary_rows)
    unique_summary.to_csv(
        out_dir / "11_unique_scene_summary_by_tier_dataset.csv", index=False
    )

    # Download priority manifest = union of unique scenes in A/B/C, with best
    # temporal tier retained and quality-action-aware priority.
    priority_rows = []
    all_unique = []
    for rank, d in [(1, unique_a), (2, unique_b), (3, unique_c)]:
        if d.empty:
            continue
        x = d.copy()
        x["tier_rank"] = rank
        all_unique.append(x)

    if all_unique:
        combined = pd.concat(all_unique, ignore_index=True)
        combined = combined.sort_values(
            ["l2a_scene_id", "tier_rank", "min_abs_delta_hours"]
        )
        best = combined.drop_duplicates("l2a_scene_id", keep="first").copy()

        action_rank = {
            "PREFERRED": 0,
            "REVIEW": 1,
            "REVIEW_UNKNOWN_QUALITY": 2,
            "EXCLUDE_LOW_QUALITY": 3,
        }
        best["quality_rank"] = best["quality_action"].map(action_rank).fillna(9)

        best["download_recommendation"] = "REVIEW"
        best.loc[
            best["quality_action"] == "PREFERRED",
            "download_recommendation"
        ] = "DOWNLOAD"
        best.loc[
            best["quality_action"] == "EXCLUDE_LOW_QUALITY",
            "download_recommendation"
        ] = "DO_NOT_DOWNLOAD_BY_DEFAULT"

        best = best.sort_values(
            ["download_recommendation", "tier_rank", "quality_rank",
             "min_abs_delta_hours", "supporting_records"],
            ascending=[True, True, True, True, False],
        )
        best.to_csv(
            out_dir / "12_download_priority_manifest.csv", index=False
        )
    else:
        best = pd.DataFrame()
        best.to_csv(
            out_dir / "12_download_priority_manifest.csv", index=False
        )

    def qcounts(d):
        return {
            "nominal": int((d["quality_class"] == "NOMINAL").sum()),
            "reduced": int((d["quality_class"] == "REDUCED").sum()),
            "low": int((d["quality_class"] == "LOW").sum()),
            "unknown": int((d["quality_class"] == "UNKNOWN").sum()),
        }

    qa, qb, qc, qd = map(qcounts, [tier_a, tier_b, tier_c, tier_d])

    lines = [
        "ENMAP DOWNLOAD MANIFEST SUMMARY",
        "=" * 80,
        f"Input V4 records                    : {len(df)}",
        f"Any same-acquisition L2A records    : {len(matched)}",
        "",
        "TIER A — <=24 h + L2A",
        f"Sample records                      : {len(tier_a)}",
        f"Unique L2A scenes                   : {len(unique_a)}",
        f"Quality: nominal/reduced/low/unknown: {qa['nominal']}/{qa['reduced']}/{qa['low']}/{qa['unknown']}",
        "",
        "TIER B — <=72 h + L2A",
        f"Sample records                      : {len(tier_b)}",
        f"Unique L2A scenes                   : {len(unique_b)}",
        f"Quality: nominal/reduced/low/unknown: {qb['nominal']}/{qb['reduced']}/{qb['low']}/{qb['unknown']}",
        "",
        "TIER C — <=7 d + L2A",
        f"Sample records                      : {len(tier_c)}",
        f"Unique L2A scenes                   : {len(unique_c)}",
        f"Quality: nominal/reduced/low/unknown: {qc['nominal']}/{qc['reduced']}/{qc['low']}/{qc['unknown']}",
        "",
        "TIER D — >30 d spatial reference candidates + L2A",
        f"Sample records                      : {len(tier_d)}",
        f"Unique L2A scenes                   : {len(unique_d)}",
        f"Quality: nominal/reduced/low/unknown: {qd['nominal']}/{qd['reduced']}/{qd['low']}/{qd['unknown']}",
        "",
        "DOWNLOAD PRIORITY",
        f"Unique A/B/C L2A scenes             : {len(best)}",
        f"Recommended DOWNLOAD (nominal)      : {int((best.get('download_recommendation') == 'DOWNLOAD').sum()) if len(best) else 0}",
        f"REVIEW                              : {int((best.get('download_recommendation') == 'REVIEW').sum()) if len(best) else 0}",
        f"Low-quality default exclude         : {int((best.get('download_recommendation') == 'DO_NOT_DOWNLOAD_BY_DEFAULT').sum()) if len(best) else 0}",
        "",
        "IMPORTANT",
        "- Tiers A/B/C are cumulative temporal windows.",
        "- Tier D is reference-candidate only; it is NOT a confirmed-negative set.",
        "- NOMINAL is marked DOWNLOAD by default.",
        "- REDUCED and UNKNOWN are REVIEW.",
        "- LOW is excluded by default but retained in the audit.",
        "- This script downloads no imagery.",
        "",
        "Primary file for the next step:",
        str(out_dir / "12_download_priority_manifest.csv"),
    ]

    (out_dir / "enmap_download_manifest_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
