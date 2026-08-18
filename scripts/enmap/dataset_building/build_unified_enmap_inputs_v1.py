#!/usr/bin/env python3
"""
build_unified_enmap_inputs_v1.py

Build a canonical, normalized methane-event table for EnMAP matching.

This script intentionally reads a small set of canonical source tables rather
than merging thousands of derived/intermediate CSVs. It preserves provenance
and DOES NOT modify any source file.

Outputs:
  unified_enmap_input_raw.csv
  unified_enmap_input_deduped.csv
  unified_enmap_source_audit.csv
  unified_enmap_build_summary.txt

Run from:
  ~/methane_release_project

Usage:
  python3 build_unified_enmap_inputs_v1.py

Optional custom project root:
  python3 build_unified_enmap_inputs_v1.py --project-root "/Users/.../methane_release_project"
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def first_present(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def clean_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_text(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def parse_time_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def combine_date_time(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    d = clean_text(df[date_col])
    t = clean_text(df[time_col])
    return pd.to_datetime(d.fillna("") + " " + t.fillna(""), errors="coerce", utc=True)


def coalesce_series(df: pd.DataFrame, columns: List[str], kind: str = "text") -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in columns:
        if c and c in df.columns:
            s = df[c]
            if kind == "num":
                s = clean_num(s)
            elif kind == "time":
                s = parse_time_series(s)
            else:
                s = clean_text(s)
            mask = pd.isna(out) & s.notna()
            out.loc[mask] = s.loc[mask]
    if kind == "num":
        return pd.to_numeric(out, errors="coerce")
    if kind == "time":
        return pd.to_datetime(out, errors="coerce", utc=True)
    return clean_text(out)


def truth_to_label(series: pd.Series) -> pd.Series:
    s = clean_text(series).str.lower()
    mapping = {
        "1": 1, "1.0": 1, "true": 1, "yes": 1, "positive": 1, "pos": 1,
        "release": 1, "plume": 1, "detected": 1,
        "0": 0, "0.0": 0, "false": 0, "no": 0, "negative": 0, "neg": 0,
        "no release": 0, "no_release": 0, "non-release": 0, "norelease": 0,
    }
    return s.map(mapping).astype("Int64")


def normalize(
    df: pd.DataFrame,
    dataset: str,
    source_path: Path,
    record_type: str,
    record_id: pd.Series,
    event_time: pd.Series,
    lat: pd.Series,
    lon: pd.Series,
    label: Optional[pd.Series] = None,
    emission: Optional[pd.Series] = None,
    sensor: Optional[pd.Series] = None,
    site: Optional[pd.Series] = None,
    qa: Optional[pd.Series] = None,
    note: Optional[pd.Series] = None,
) -> pd.DataFrame:
    n = len(df)
    result = pd.DataFrame({
        "dataset": dataset,
        "record_type": record_type,
        "source_path": str(source_path),
        "source_row": range(n),
        "record_id": clean_text(record_id),
        "event_time_utc": pd.to_datetime(event_time, errors="coerce", utc=True),
        "lat": clean_num(lat),
        "lon": clean_num(lon),
    })
    result["label"] = label.astype("Int64") if label is not None else pd.Series(pd.NA, index=df.index, dtype="Int64")
    result["emission_kg_hr"] = clean_num(emission) if emission is not None else pd.Series(float("nan"), index=df.index)
    result["sensor"] = clean_text(sensor) if sensor is not None else pd.Series(pd.NA, index=df.index, dtype="string")
    result["site"] = clean_text(site) if site is not None else pd.Series(pd.NA, index=df.index, dtype="string")
    result["qa_status"] = clean_text(qa) if qa is not None else pd.Series(pd.NA, index=df.index, dtype="string")
    result["note"] = clean_text(note) if note is not None else pd.Series(pd.NA, index=df.index, dtype="string")

    result["has_time"] = result["event_time_utc"].notna()
    result["has_coords"] = result["lat"].between(-90, 90) & result["lon"].between(-180, 180)
    launch = pd.Timestamp("2022-04-01T00:00:00Z")
    result["enmap_temporal_status"] = pd.NA
    result.loc[result["has_time"] & (result["event_time_utc"] < launch), "enmap_temporal_status"] = "PRE_ENMAP_LAUNCH"
    result.loc[result["has_time"] & (result["event_time_utc"] >= launch), "enmap_temporal_status"] = "QUERY_ENMAP_CATALOG"
    result.loc[~result["has_time"], "enmap_temporal_status"] = "TIME_MISSING"

    return result


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path.home() / "methane_release_project"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else root / "unified_enmap_input_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ("CONTROLLED_RELEASE_VERIFIED_107", root / "outputs/473_controlled_release_verified_event_master_v1.csv"),
        ("STANFORD_2024_2025_746", root / "stanford_2024_2025_controlled_release/01_stanford_746_event_master.csv"),
        ("METHANEAIR_435", root / "outputs/14_methaneair_events_for_gee.csv"),
        ("METHANESAT_POSNEG_222", root / "methanesat_l3_l4_posneg_222/manifest_model_ready_posneg.csv"),
        ("UNEP_MARS_PLUMES", root / "unep_mars/csv/unep_methanedata_detected_plumes.csv"),
        ("CARBON_MAPPER_CH4_PLUMES", root / "carbon_mapper_inventory/carbon_mapper_all_CH4_plumes.csv"),
        ("AVIRIS3_SCENES_493", root / "aviris3_methanefuse_temporal_audit.csv"),
        ("EMIT_POSNEG_100", root / "emit_v2_posneg_100/emit_v2_inventory.csv"),
        ("GHGSAT_OBSERVATIONS_209", root / "outputs/521_ghgsat_observation_manifest_v1.csv"),
    ]

    all_parts = []
    audit = []

    for dataset, path in sources:
        if not path.exists():
            audit.append({
                "dataset": dataset, "source_path": str(path), "status": "MISSING",
                "rows": 0, "with_time": 0, "with_coords": 0, "query_enmap_catalog": 0,
                "pre_enmap_launch": 0
            })
            print(f"MISSING: {path}")
            continue

        try:
            df = read_csv(path)
        except Exception as e:
            audit.append({
                "dataset": dataset, "source_path": str(path), "status": f"READ_ERROR:{type(e).__name__}:{e}",
                "rows": 0, "with_time": 0, "with_coords": 0, "query_enmap_catalog": 0,
                "pre_enmap_launch": 0
            })
            print(f"READ ERROR: {path}: {e}")
            continue

        if dataset == "CONTROLLED_RELEASE_VERIFIED_107":
            rid = coalesce_series(df, ["event_id_verified", "event_id", "_event_id"])
            t = coalesce_series(df, ["event_time_utc_verified", "datetime_utc", "_event_time"], "time")
            lat = coalesce_series(df, ["latitude_verified", "lat", "_latitude"], "num")
            lon = coalesce_series(df, ["longitude_verified", "lon", "_longitude"], "num")
            truth_col = first_present(df, ["true_release_verified", "true_release", "_true_release"])
            label = truth_to_label(df[truth_col]) if truth_col else pd.Series(pd.NA, index=df.index, dtype="Int64")
            emission = coalesce_series(df, ["selected_release_rate_kg_h", "median_matching_release_rate_kg_h"], "num")
            # fall back to tonnes/hour -> kg/hour only if selected rates are absent
            if "emission_tph_mean" in df.columns:
                fallback = clean_num(df["emission_tph_mean"]) * 1000.0
                emission = emission.where(emission.notna(), fallback)
            sensor = coalesce_series(df, ["satellite_clean", "satellite_from_paper"])
            site = coalesce_series(df, ["site_name"])
            qa = coalesce_series(df, ["verification_status", "row_status"])
            note = coalesce_series(df, ["verification_issue", "evaluation_group"])
            part = normalize(df, dataset, path, "ground_truth_event", rid, t, lat, lon, label, emission, sensor, site, qa, note)

        elif dataset == "STANFORD_2024_2025_746":
            rid = coalesce_series(df, ["release_ID"])
            t = coalesce_series(df, ["datetime_UTC"], "time")
            lat = coalesce_series(df, ["lat"], "num")
            lon = coalesce_series(df, ["lon"], "num")
            label_col = first_present(df, ["label"])
            label = pd.to_numeric(df[label_col], errors="coerce").astype("Int64") if label_col else pd.Series(pd.NA, index=df.index, dtype="Int64")
            emission = coalesce_series(df, ["ch4_kgh_mean"], "num")
            sensor = coalesce_series(df, ["SatelliteCode", "SatelliteMatchName"])
            site = coalesce_series(df, ["location"])
            qa = coalesce_series(df, ["QC_ExperimentTeam", "Acquisition status"])
            note = coalesce_series(df, ["ground_truth", "Phase"])
            part = normalize(df, dataset, path, "ground_truth_event", rid, t, lat, lon, label, emission, sensor, site, qa, note)

        elif dataset == "METHANEAIR_435":
            rid = coalesce_series(df, ["event_id", "plume_id", "flight_id"])
            t = coalesce_series(df, ["datetime_utc", "time_coverage_start"], "time")
            lat = coalesce_series(df, ["lat"], "num")
            lon = coalesce_series(df, ["lon"], "num")
            label = pd.to_numeric(df["label"], errors="coerce").astype("Int64") if "label" in df.columns else pd.Series(pd.NA, index=df.index, dtype="Int64")
            emission = coalesce_series(df, ["emission_kg_hr"], "num")
            sensor = pd.Series("MethaneAIR", index=df.index)
            site = coalesce_series(df, ["Basin"])
            qa = coalesce_series(df, ["ground_truth_type", "label_type"])
            note = coalesce_series(df, ["source_dataset"])
            part = normalize(df, dataset, path, "methane_detection", rid, t, lat, lon, label, emission, sensor, site, qa, note)

        elif dataset == "METHANESAT_POSNEG_222":
            rid = coalesce_series(df, ["id", "plume_id"])
            t = coalesce_series(df, ["time_coverage_start", "date", "l3_date"], "time")
            lat = coalesce_series(df, ["lat"], "num")
            lon = coalesce_series(df, ["lon"], "num")
            label = pd.to_numeric(df["label"], errors="coerce").astype("Int64") if "label" in df.columns else pd.Series(pd.NA, index=df.index, dtype="Int64")
            emission = coalesce_series(df, ["flux_kg_hr"], "num")
            sensor = pd.Series("MethaneSAT", index=df.index)
            site = coalesce_series(df, ["target_id", "collection_id"])
            qa = coalesce_series(df, ["qa_status", "qa_reason"])
            note = coalesce_series(df, ["sample_type", "negative_type", "parent_positive_id"])
            part = normalize(df, dataset, path, "model_sample", rid, t, lat, lon, label, emission, sensor, site, qa, note)

        elif dataset == "UNEP_MARS_PLUMES":
            rid = coalesce_series(df, ["id_plume"])
            t = coalesce_series(df, ["tile_date"], "time")
            lat = coalesce_series(df, ["lat"], "num")
            lon = coalesce_series(df, ["lon"], "num")
            label = pd.Series(1, index=df.index, dtype="Int64")
            emission = coalesce_series(df, ["ch4_fluxrate", "total_emission"], "num")
            sensor = coalesce_series(df, ["satellite"])
            site = coalesce_series(df, ["source_name", "country"])
            qa = coalesce_series(df, ["actionable", "notified"])
            note = coalesce_series(df, ["sector", "detection_institution"])
            part = normalize(df, dataset, path, "detected_plume", rid, t, lat, lon, label, emission, sensor, site, qa, note)

        elif dataset == "CARBON_MAPPER_CH4_PLUMES":
            rid = coalesce_series(df, ["plume_id"])
            t = coalesce_series(df, ["scene_timestamp"], "time")
            lat = coalesce_series(df, ["latitude"], "num")
            lon = coalesce_series(df, ["longitude"], "num")
            label = pd.Series(1, index=df.index, dtype="Int64")
            emission = coalesce_series(df, ["emission_auto_kg_hr"], "num")
            sensor = coalesce_series(df, ["instrument", "platform"])
            site = coalesce_series(df, ["plume_name"])
            qa = coalesce_series(df, ["plume_quality"])
            note = coalesce_series(df, ["mission_phase", "sector"])
            part = normalize(df, dataset, path, "detected_plume", rid, t, lat, lon, label, emission, sensor, site, qa, note)

        elif dataset == "AVIRIS3_SCENES_493":
            rid = coalesce_series(df, ["scene_key"])
            if "t0_date" in df.columns and "t0_time" in df.columns:
                t = combine_date_time(df, "t0_date", "t0_time")
            else:
                t = coalesce_series(df, ["t0_date"], "time")
            lat = coalesce_series(df, ["t0_center_lat"], "num")
            lon = coalesce_series(df, ["t0_center_lon"], "num")
            label = pd.Series(pd.NA, index=df.index, dtype="Int64")
            emission = pd.Series(float("nan"), index=df.index)
            sensor = pd.Series("AVIRIS-3", index=df.index)
            site = pd.Series(pd.NA, index=df.index, dtype="string")
            qa = coalesce_series(df, ["status", "strict30_triplet"])
            note = coalesce_series(df, ["three_distinct_dates", "repo_path_complete_triplet"])
            part = normalize(df, dataset, path, "sensor_scene", rid, t, lat, lon, label, emission, sensor, site, qa, note)

        elif dataset == "EMIT_POSNEG_100":
            rid = coalesce_series(df, ["sample_id", "scene_id"])
            t = coalesce_series(df, ["acquisition_time_utc"], "time")
            lat = coalesce_series(df, ["match_lat"], "num")
            lon = coalesce_series(df, ["match_lon"], "num")
            label = pd.to_numeric(df["label"], errors="coerce").astype("Int64") if "label" in df.columns else pd.Series(pd.NA, index=df.index, dtype="Int64")
            emission = pd.Series(float("nan"), index=df.index)
            sensor = pd.Series("EMIT", index=df.index)
            site = coalesce_series(df, ["scene_id"])
            qa = coalesce_series(df, ["label_strength"])
            note = coalesce_series(df, ["label_source", "pair_id"])
            part = normalize(df, dataset, path, "model_sample", rid, t, lat, lon, label, emission, sensor, site, qa, note)

        elif dataset == "GHGSAT_OBSERVATIONS_209":
            rid = coalesce_series(df, ["ghgsat_observation_id"])
            t = coalesce_series(df, ["best_available_acquisition_time_utc", "standardized_datetime_utc", "stanford_timestamp"], "time")
            lat = pd.Series(float("nan"), index=df.index)
            lon = pd.Series(float("nan"), index=df.index)
            truth_col = first_present(df, ["release_present_from_class"])
            label = truth_to_label(df[truth_col]) if truth_col else pd.Series(pd.NA, index=df.index, dtype="Int64")
            emission = coalesce_series(df, ["ground_truth_rate_median_kg_hr"], "num")
            sensor = pd.Series("GHGSat", index=df.index)
            site = pd.Series(pd.NA, index=df.index, dtype="string")
            qa = coalesce_series(df, ["analysis_role", "primary_evaluable", "locked_for_analysis"])
            note = coalesce_series(df, ["acquisition_time_source", "review_reasons"])
            part = normalize(df, dataset, path, "satellite_observation", rid, t, lat, lon, label, emission, sensor, site, qa, note)

        else:
            continue

        all_parts.append(part)

        audit.append({
            "dataset": dataset,
            "source_path": str(path),
            "status": "OK",
            "rows": len(part),
            "with_time": int(part["has_time"].sum()),
            "with_coords": int(part["has_coords"].sum()),
            "query_enmap_catalog": int((part["enmap_temporal_status"] == "QUERY_ENMAP_CATALOG").sum()),
            "pre_enmap_launch": int((part["enmap_temporal_status"] == "PRE_ENMAP_LAUNCH").sum()),
        })
        print(f"OK: {dataset}: {len(part)} rows")

    if not all_parts:
        raise SystemExit("No canonical source tables could be loaded.")

    raw = pd.concat(all_parts, ignore_index=True)

    # Dataset-internal dedupe only. We DO NOT collapse across different datasets,
    # because cross-dataset overlap must be audited explicitly.
    key = raw[["dataset", "record_id", "event_time_utc", "lat", "lon"]].astype("string")
    raw["within_dataset_duplicate"] = key.duplicated(keep="first")
    deduped = raw.loc[~raw["within_dataset_duplicate"]].copy()

    raw_path = out_dir / "unified_enmap_input_raw.csv"
    dedup_path = out_dir / "unified_enmap_input_deduped.csv"
    audit_path = out_dir / "unified_enmap_source_audit.csv"
    summary_path = out_dir / "unified_enmap_build_summary.txt"

    raw.to_csv(raw_path, index=False)
    deduped.to_csv(dedup_path, index=False)
    pd.DataFrame(audit).to_csv(audit_path, index=False)

    audit_df = pd.DataFrame(audit)
    summary = [
        "UNIFIED ENMAP INPUT BUILD SUMMARY",
        "=" * 72,
        f"Project root             : {root}",
        f"Canonical sources loaded : {(audit_df['status'] == 'OK').sum() if len(audit_df) else 0}",
        f"Raw normalized rows      : {len(raw)}",
        f"Within-source duplicates : {int(raw['within_dataset_duplicate'].sum())}",
        f"Deduped rows             : {len(deduped)}",
        f"Rows with time           : {int(deduped['has_time'].sum())}",
        f"Rows with coordinates    : {int(deduped['has_coords'].sum())}",
        f"Pre-EnMAP-launch rows    : {int((deduped['enmap_temporal_status'] == 'PRE_ENMAP_LAUNCH').sum())}",
        f"Rows to query in EnMAP   : {int((deduped['enmap_temporal_status'] == 'QUERY_ENMAP_CATALOG').sum())}",
        "",
        "BY DATASET",
    ]
    for _, r in audit_df.iterrows():
        summary.append(
            f"{r['dataset']}: status={r['status']}, rows={r['rows']}, "
            f"time={r['with_time']}, coords={r['with_coords']}, "
            f"query={r['query_enmap_catalog']}, prelaunch={r['pre_enmap_launch']}"
        )

    summary += [
        "",
        "IMPORTANT",
        "- This file intentionally uses canonical source tables, not thousands of derived CSVs.",
        "- Cross-dataset duplicates are preserved for audit; only within-dataset exact-key duplicates are flagged.",
        "- PRE_ENMAP_LAUNCH uses 2022-04-01, the official EnMAP launch date.",
        "- GHGSat rows may lack coordinates but can still be marked pre-launch from time.",
        "- MethaneSAT negatives remain candidate/weak negatives according to their source manifest; do not reinterpret them as verified no-emission ground truth.",
        "",
        "NEXT STEP",
        "Upload unified_enmap_input_deduped.csv and unified_enmap_source_audit.csv to ChatGPT.",
    ]
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print()
    print("\n".join(summary))
    print()
    print("Saved:")
    print(raw_path)
    print(dedup_path)
    print(audit_path)
    print(summary_path)


if __name__ == "__main__":
    main()
