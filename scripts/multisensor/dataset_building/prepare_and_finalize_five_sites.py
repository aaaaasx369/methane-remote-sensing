#!/usr/bin/env python3
"""
Prepare and finalize a five-site Sentinel-2 multisource experiment.

PREPARE mode (default)
----------------------
Uses:
  outputs/15_methaneair_s2_landsat_availability.csv
  outputs/18_methaneair_s2_dataset_table.csv

Creates three spatially distinct MethaneAIR sites from already-downloaded
positive Sentinel-2 patches and writes:
  outputs/540_methaneair_candidate_sites_v1.csv
  outputs/541_selected_methaneair_positive_manifest_v1.csv
  outputs/542_five_site_design_v1.csv
  outputs/543_five_site_prepare_report_v1.txt

FINALIZE mode (--finalize)
--------------------------
Uses:
  outputs/390_multisensor_master_manifest_v1.csv
  outputs/541_selected_methaneair_positive_manifest_v1.csv
  outputs/547_methaneair_reference_negative_manifest_v1.csv

Creates:
  outputs/548_five_site_multisource_manifest_v1.csv
  outputs/549_five_site_readiness_audit_v1.csv
  outputs/550_five_site_finalize_report_v1.txt

Final site design:
  Casa Grande                         -> 2024_AMT
  Ehrenberg                           -> 2023_Scientific_Reports
  Three selected MethaneAIR clusters -> MethaneAIR
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_KM = 6371.0088

EVENT_ID_ALIASES = ("event_id", "sample_id", "plume_id", "id")
LAT_ALIASES = ("lat", "latitude", "source_latitude", "source_lat")
LON_ALIASES = ("lon", "longitude", "lng", "source_longitude", "source_lon")
TIME_ALIASES = ("datetime_utc", "event_time_utc", "timestamp_utc", "acquisition_time_utc")
EMISSION_ALIASES = ("emission_kg_hr", "emission_kg_h", "release_rate_kg_h", "emission_rate_kg_h")
PATH_ALIASES = ("resolved_patch_path", "patch_path", "relative_path", "file_path", "filepath", "image_path", "filename")
SCENE_ALIASES = ("scene_id", "s2_scene_id", "system_index", "system:index", "image_id", "product_id")


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("/Users/happydoraaa/methane_release_project"))
    p.add_argument("--cluster-radius-km", type=float, default=1.0)
    p.add_argument("--minimum-positive-patches", type=int, default=2)
    p.add_argument("--minimum-unique-scenes", type=int, default=2)
    p.add_argument("--methaneair-site-count", type=int, default=3)
    p.add_argument("--selected-sites", nargs="*", default=None)
    p.add_argument("--finalize", action="store_true")
    return p.parse_args()


def first_col(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def norm_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def resolve_path(root: Path, value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    raw = Path(text).expanduser()
    guesses = [
        raw, root / raw, root / "outputs" / raw, root / "patches" / raw,
        root / "images" / raw, root / "data" / raw,
        root / "outputs" / raw.name, root / "patches" / raw.name,
        root / "images" / raw.name,
    ]
    for guess in guesses:
        if guess.exists() and guess.is_file():
            return str(guess.resolve())
    for folder_name in ("patches", "images", "data", "downloads", "outputs"):
        folder = root / folder_name
        if folder.exists():
            matches = list(folder.rglob(raw.name))
            if len(matches) == 1:
                return str(matches[0].resolve())
    return ""


def scene_from_path(path_value: object, fallback: str) -> str:
    text = str(path_value).strip()
    return Path(text).stem if text and text.lower() not in {"nan", "none", "<na>"} else fallback


def controlled_source(site: object, time_value: object) -> str:
    text = str(site).lower()
    if "ehrenberg" in text:
        return "2023_Scientific_Reports"
    if "casa" in text and "grande" in text:
        return "2024_AMT"
    ts = pd.to_datetime(time_value, errors="coerce", utc=True)
    if pd.notna(ts):
        if ts.year == 2021:
            return "2023_Scientific_Reports"
        if ts.year == 2022:
            return "2024_AMT"
    return "controlled_release_unresolved"


def prepare(root: Path, args: argparse.Namespace) -> int:
    out = root / "outputs"
    event_path = out / "15_methaneair_s2_landsat_availability.csv"
    patch_path = out / "18_methaneair_s2_dataset_table.csv"
    controlled_path = out / "390_multisensor_master_manifest_v1.csv"
    for path in (event_path, patch_path, controlled_path):
        if not path.exists():
            raise SystemExit(f"Required file missing: {path}")

    events = pd.read_csv(event_path)
    patches = pd.read_csv(patch_path)
    controlled = pd.read_csv(controlled_path)

    eid = first_col(events, EVENT_ID_ALIASES)
    lat = first_col(events, LAT_ALIASES)
    lon = first_col(events, LON_ALIASES)
    etime = first_col(events, TIME_ALIASES)
    emission = first_col(events, EMISSION_ALIASES)
    peid = first_col(patches, EVENT_ID_ALIASES)
    ppath = first_col(patches, PATH_ALIASES)
    pscene = first_col(patches, SCENE_ALIASES)
    ptime = first_col(patches, TIME_ALIASES)
    missing = [name for name, col in {
        "event_id": eid, "latitude": lat, "longitude": lon, "event_time": etime,
        "patch_event_id": peid, "patch_path": ppath,
    }.items() if col is None]
    if missing:
        raise SystemExit("Missing columns: " + ", ".join(missing))

    e = events.copy()
    e["event_id_canonical"] = norm_id(e[eid])
    e["latitude"] = pd.to_numeric(e[lat], errors="coerce")
    e["longitude"] = pd.to_numeric(e[lon], errors="coerce")
    e["event_time_utc"] = pd.to_datetime(e[etime], errors="coerce", utc=True)
    e["emission_kg_h"] = pd.to_numeric(e[emission], errors="coerce") if emission else np.nan
    e = e.dropna(subset=["event_id_canonical", "latitude", "longitude", "event_time_utc"])
    e = e.drop_duplicates("event_id_canonical")

    p = patches.copy()
    p["event_id_canonical"] = norm_id(p[peid])
    p["patch_path_raw"] = p[ppath]
    p["patch_path"] = p["patch_path_raw"].map(lambda v: resolve_path(root, v))
    p = p[p["patch_path"].astype(str).str.len().gt(0)].copy()
    p["scene_id"] = norm_id(p[pscene]) if pscene else ""
    bad_scene = p["scene_id"].isin(["", "nan", "None", "<NA>"])
    p.loc[bad_scene, "scene_id"] = p.loc[bad_scene].apply(
        lambda row: scene_from_path(row["patch_path"], f"event_{row['event_id_canonical']}"), axis=1
    )
    p["s2_acquisition_time_utc"] = pd.to_datetime(p[ptime], errors="coerce", utc=True) if ptime else pd.NaT

    pos = e.merge(
        p[["event_id_canonical", "scene_id", "patch_path", "s2_acquisition_time_utc"]],
        on="event_id_canonical", how="inner", validate="one_to_many"
    )
    pos = pos.drop_duplicates("scene_id").reset_index(drop=True)
    if pos.empty:
        raise SystemExit("No readable MethaneAIR Sentinel-2 positive patches were joined.")

    coords_rad = np.radians(pos[["latitude", "longitude"]].to_numpy())
    labels = DBSCAN(
        eps=args.cluster_radius_km / EARTH_RADIUS_KM,
        min_samples=1, metric="haversine", algorithm="ball_tree"
    ).fit_predict(coords_rad)
    pos["cluster_label"] = labels

    candidates = pos.groupby("cluster_label").agg(
        latitude=("latitude", "median"), longitude=("longitude", "median"),
        positive_patches=("scene_id", "size"), unique_events=("event_id_canonical", "nunique"),
        unique_scenes=("scene_id", "nunique"),
        unique_dates=("event_time_utc", lambda x: pd.to_datetime(x, utc=True).dt.date.nunique()),
        minimum_emission_kg_h=("emission_kg_h", "min"),
        median_emission_kg_h=("emission_kg_h", "median"),
        maximum_emission_kg_h=("emission_kg_h", "max"),
    ).reset_index().sort_values(["latitude", "longitude"]).reset_index(drop=True)
    candidates["candidate_site_id"] = [f"MethaneAIR_site_{i+1:03d}" for i in range(len(candidates))]
    candidates["ready"] = (
        candidates["positive_patches"].ge(args.minimum_positive_patches)
        & candidates["unique_scenes"].ge(args.minimum_unique_scenes)
    )
    candidates["selection_score"] = (
        candidates["ready"].astype(int) * 100000
        + candidates["unique_scenes"] * 100
        + candidates["unique_dates"]
    )
    candidates = candidates.sort_values(
        ["selection_score", "positive_patches", "unique_events"], ascending=False
    ).reset_index(drop=True)
    candidate_path = out / "540_methaneair_candidate_sites_v1.csv"
    candidates.to_csv(candidate_path, index=False)

    if args.selected_sites:
        selected_ids = list(args.selected_sites)
        missing_ids = set(selected_ids) - set(candidates["candidate_site_id"])
        if missing_ids:
            raise SystemExit("Selected site IDs not found: " + ", ".join(sorted(missing_ids)))
        selected = candidates[candidates["candidate_site_id"].isin(selected_ids)].copy()
    else:
        selected = candidates[candidates["ready"]].head(args.methaneair_site_count).copy()
    if len(selected) < args.methaneair_site_count:
        raise SystemExit(
            f"Only {len(selected)} MethaneAIR sites satisfy the minimum criteria. Inspect {candidate_path}."
        )
    selected = selected.head(args.methaneair_site_count)
    selected_ids = list(selected["candidate_site_id"])

    pos = pos.merge(candidates[["cluster_label", "candidate_site_id"]], on="cluster_label", how="left")
    pos = pos[pos["candidate_site_id"].isin(selected_ids)].copy()
    pos = pos.rename(columns={"candidate_site_id": "site_id"})
    pos["sample_id"] = pos["site_id"] + "_positive_" + (pos.groupby("site_id").cumcount() + 1).astype(str).str.zfill(3)
    pos["label"] = 1
    pos["source_origin"] = "MethaneAIR"
    pos["ground_truth_type"] = "observational_plume_positive"
    pos["benchmark_tier"] = "exploratory_external_source"
    positive_path = out / "541_selected_methaneair_positive_manifest_v1.csv"
    pos.to_csv(positive_path, index=False)

    csite = first_col(controlled, ("site", "site_id", "site_name"))
    clabel = first_col(controlled, ("label", "final_label"))
    ctime = first_col(controlled, TIME_ALIASES)
    cpath = first_col(controlled, PATH_ALIASES)
    if csite is None or clabel is None or cpath is None:
        raise SystemExit("390 manifest lacks site, label, or patch path.")
    cs = controlled.copy()
    cs["site_id"] = cs[csite].astype(str)
    cs["label_canonical"] = pd.to_numeric(cs[clabel], errors="coerce")
    cs["source_origin"] = [
        controlled_source(site, time_value)
        for site, time_value in zip(cs["site_id"], cs[ctime] if ctime else [None] * len(cs))
    ]
    cs["readable_patch"] = cs[cpath].map(lambda v: bool(resolve_path(root, v)))
    csummary = cs.groupby(["site_id", "source_origin"]).agg(
        rows=("label_canonical", "size"),
        positive=("label_canonical", lambda x: int((x == 1).sum())),
        negative=("label_canonical", lambda x: int((x == 0).sum())),
        readable_patches=("readable_patch", "sum"),
    ).reset_index()
    csummary["site_type"] = "controlled_release"

    msummary = selected[["candidate_site_id", "positive_patches", "unique_events", "unique_scenes", "latitude", "longitude"]].rename(
        columns={"candidate_site_id": "site_id", "positive_patches": "positive"}
    )
    msummary["source_origin"] = "MethaneAIR"
    msummary["rows"] = msummary["positive"]
    msummary["negative"] = 0
    msummary["readable_patches"] = msummary["positive"]
    msummary["site_type"] = "observational_plume"
    design = pd.concat([csummary, msummary], ignore_index=True, sort=False)
    design_path = out / "542_five_site_design_v1.csv"
    design.to_csv(design_path, index=False)

    report = [
        "=" * 110,
        "FIVE-SITE PREPARATION REPORT V1",
        "=" * 110,
        "",
        f"Resolved MethaneAIR positive scenes: {len(pos)}",
        f"Candidate MethaneAIR spatial clusters: {len(candidates)}",
        f"Selected MethaneAIR sites: {', '.join(selected_ids)}",
        "",
        "FINAL DESIGN",
        "-" * 110,
        design.to_string(index=False),
        "",
        "SOURCE CORRECTION",
        "-" * 110,
        "Ehrenberg -> 2023_Scientific_Reports",
        "Casa Grande -> 2024_AMT",
        "Three additional spatial clusters -> MethaneAIR",
        "",
        "NEXT",
        "-" * 110,
        "Run download_methaneair_reference_negatives.py, then rerun this script with --finalize.",
        "MethaneAIR label-0 rows will be no-known-plume references, not confirmed zero-emission ground truth.",
        "",
        "OUTPUTS",
        "-" * 110,
        str(candidate_path), str(positive_path), str(design_path),
    ]
    report_path = out / "543_five_site_prepare_report_v1.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print("\nCreated:")
    for path in (candidate_path, positive_path, design_path, report_path):
        print(" ", path)
    return 0


def finalize(root: Path) -> int:
    out = root / "outputs"
    controlled_path = out / "390_multisensor_master_manifest_v1.csv"
    positive_path = out / "541_selected_methaneair_positive_manifest_v1.csv"
    negative_path = out / "547_methaneair_reference_negative_manifest_v1.csv"
    for path in (controlled_path, positive_path, negative_path):
        if not path.exists():
            raise SystemExit(f"Required file missing: {path}")

    controlled = pd.read_csv(controlled_path)
    positive = pd.read_csv(positive_path)
    negative = pd.read_csv(negative_path)

    csite = first_col(controlled, ("site", "site_id", "site_name"))
    clabel = first_col(controlled, ("label", "final_label"))
    csample = first_col(controlled, ("sample_id", "event_id", "filename"))
    cscene = first_col(controlled, SCENE_ALIASES)
    cpath = first_col(controlled, PATH_ALIASES)
    ctime = first_col(controlled, TIME_ALIASES)
    cemission = first_col(controlled, EMISSION_ALIASES + ("matched_positive_release_rate_kg_h",))
    if any(v is None for v in (csite, clabel, cpath)):
        raise SystemExit("390 manifest lacks site, label, or patch path.")

    c = pd.DataFrame(index=controlled.index)
    c["site_id"] = controlled[csite].astype(str)
    c["label"] = pd.to_numeric(controlled[clabel], errors="coerce")
    c["sample_id"] = controlled[csample].astype(str) if csample else [f"controlled_{i:04d}" for i in range(len(controlled))]
    c["scene_id"] = controlled[cscene].astype(str) if cscene else c["sample_id"]
    c["patch_path"] = controlled[cpath].map(lambda v: resolve_path(root, v))
    c["acquisition_time_utc"] = pd.to_datetime(controlled[ctime], errors="coerce", utc=True) if ctime else pd.NaT
    c["release_rate_kg_h"] = pd.to_numeric(controlled[cemission], errors="coerce") if cemission else np.nan
    c["source_origin"] = [controlled_source(site, time_value) for site, time_value in zip(c["site_id"], c["acquisition_time_utc"])]
    c["ground_truth_type"] = "controlled_release_status"
    c["benchmark_tier"] = "strict_controlled_release"
    c["negative_confidence"] = np.where(c["label"].eq(0), "confirmed_no_release_at_acquisition", "")

    p = pd.DataFrame(index=positive.index)
    p["site_id"] = positive["site_id"].astype(str)
    p["label"] = 1
    p["sample_id"] = positive["sample_id"].astype(str)
    p["scene_id"] = positive["scene_id"].astype(str)
    p["patch_path"] = positive["patch_path"].map(lambda v: resolve_path(root, v))
    p["acquisition_time_utc"] = pd.to_datetime(positive["s2_acquisition_time_utc"], errors="coerce", utc=True)
    p["release_rate_kg_h"] = pd.to_numeric(positive["emission_kg_h"], errors="coerce")
    p["source_origin"] = "MethaneAIR"
    p["ground_truth_type"] = "observational_plume_positive"
    p["benchmark_tier"] = "exploratory_external_source"
    p["negative_confidence"] = ""

    if "download_ok" in negative.columns:
        negative = negative[negative["download_ok"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    n = pd.DataFrame(index=negative.index)
    n["site_id"] = negative["site_id"].astype(str)
    n["label"] = 0
    n["sample_id"] = negative["sample_id"].astype(str)
    n["scene_id"] = negative["scene_id"].astype(str) if "scene_id" in negative else negative["s2_scene_id"].astype(str)
    n["patch_path"] = negative["patch_path"].map(lambda v: resolve_path(root, v))
    n["acquisition_time_utc"] = pd.to_datetime(negative["s2_acquisition_time_utc"], errors="coerce", utc=True)
    n["release_rate_kg_h"] = np.nan
    n["source_origin"] = "MethaneAIR"
    n["ground_truth_type"] = "no_known_plume_reference"
    n["benchmark_tier"] = "exploratory_external_source"
    n["negative_confidence"] = "reference_only_not_confirmed_zero_emission"

    manifest = pd.concat([c, p, n], ignore_index=True, sort=False)
    manifest = manifest[manifest["label"].isin([0, 1]) & manifest["patch_path"].astype(str).str.len().gt(0)].copy()
    manifest["label"] = manifest["label"].astype(int)
    manifest = manifest.drop_duplicates(["site_id", "scene_id"]).reset_index(drop=True)
    manifest_path = out / "548_five_site_multisource_manifest_v1.csv"
    manifest.to_csv(manifest_path, index=False)

    audit = manifest.groupby(["site_id", "source_origin", "benchmark_tier"]).agg(
        rows=("sample_id", "size"), positive=("label", lambda x: int((x == 1).sum())),
        negative=("label", lambda x: int((x == 0).sum())), scenes=("scene_id", "nunique")
    ).reset_index()
    audit["has_both_classes"] = audit["positive"].gt(0) & audit["negative"].gt(0)
    audit_path = out / "549_five_site_readiness_audit_v1.csv"
    audit.to_csv(audit_path, index=False)

    sites = manifest["site_id"].nunique()
    sources = manifest["source_origin"].nunique()
    ready = int(audit["has_both_classes"].sum())
    report = [
        "=" * 112, "FIVE-SITE FINALIZE REPORT V1", "=" * 112, "",
        f"Unique sites: {sites}", f"Unique sources: {sources}", f"Sites with both classes: {ready}", "",
        "READINESS", "-" * 112, audit.to_string(index=False), "",
        "COMPLETION TEST", "-" * 112,
        "Required: Unique sites = 5, Unique sources >= 3, Sites with both classes = 5.",
        "The five-site result is exploratory because MethaneAIR negatives are reference-only.", "",
        "NEXT", "-" * 112,
        "python run_multisource_s2_model_v2.py --project-root <root> --input outputs/548_five_site_multisource_manifest_v1.csv",
    ]
    report_path = out / "550_five_site_finalize_report_v1.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(report_path.read_text())
    if sites < 5 or sources < 3 or ready < 5:
        raise SystemExit("Five-site readiness requirements are not yet satisfied. Inspect the audit.")
    return 0


def main() -> int:
    args = args_parser()
    root = args.project_root.expanduser().resolve()
    return finalize(root) if args.finalize else prepare(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
