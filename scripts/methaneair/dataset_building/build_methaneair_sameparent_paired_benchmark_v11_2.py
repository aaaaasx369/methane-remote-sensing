#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HOME = Path.home()
PROJECT_ROOT = HOME / "methane_release_project"
METHANEFUSE_ROOT = HOME / "MethaneFuse"

BASE_SCRIPT = PROJECT_ROOT / "freeze_build_methaneair_368_methanefuse_v10_1.py"
NEG_ROOT = PROJECT_ROOT / "MethaneAIR_Validated_S2_Controls_368_v1"
NEG_CANONICAL = NEG_ROOT / "canonical" / "00_canonical_368_controls.csv"
NEG_EVAL = METHANEFUSE_ROOT / "data" / "custom" / "methaneair_validated_368_strictqa_eval.csv"

DATASET_NAME = "MethaneAIR_S2_SameParent_Paired_Benchmark_v1"
DEFAULT_OUT = PROJECT_ROOT / DATASET_NAME
EXPECTED_NEG_CONTROLS = 368
EXPECTED_PARENT_COUNT = 293
POS_T0_MAX_DELTA_HOURS = 72.0
STRICT24_HOURS = 24.0
CANONICAL_OVERPASS_TOLERANCE_MINUTES = 60.0

LEGACY_MANIFEST_DEFAULTS = [
    METHANEFUSE_ROOT / "data" / "methaneair_full" / "sentinel2_temporal_manifest_best_qa_v2.csv",
    PROJECT_ROOT / "data" / "methaneair_full" / "sentinel2_temporal_manifest_best_qa_v2.csv",
]
LEGACY_READINESS_DEFAULTS = [
    METHANEFUSE_ROOT / "data" / "methaneair_full" / "sentinel2_v2_full_record_readiness.csv",
    PROJECT_ROOT / "data" / "methaneair_full" / "sentinel2_v2_full_record_readiness.csv",
]


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Required v10.1 builder not found:\n{BASE_SCRIPT}\n"
            "Keep freeze_build_methaneair_368_methanefuse_v10_1.py in ~/methane_release_project."
        )
    spec = importlib.util.spec_from_file_location("mf368base_v101", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base builder: {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = load_base_module()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a same-parent balanced Sentinel-2 benchmark: one MethaneAIR L4 parent positive "
            "paired with one MethaneAIR-validated temporal no-detection control per parent."
        )
    )
    p.add_argument("--master", default="", help="Professor Master V3 XLSX. Auto-discovered if omitted.")
    p.add_argument("--negative-canonical", default=str(NEG_CANONICAL))
    p.add_argument("--negative-eval", default=str(NEG_EVAL))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--project", default="methane-release-gee")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="Smoke-test first N selected parents; 0 = all.")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-scene-cloud", type=float, default=100.0)
    p.add_argument("--legacy-manifest", default="", help="Previously audited sentinel2_temporal_manifest_best_qa_v2.csv. Auto-discovered if omitted.")
    p.add_argument("--legacy-readiness", default="", help="Previously audited sentinel2_v2_full_record_readiness.csv. Auto-discovered if omitted.")
    p.add_argument(
        "--canonical-overpass-tolerance-minutes",
        type=float,
        default=CANONICAL_OVERPASS_TOLERANCE_MINUTES,
        help="Maximum GEE acquisition-time difference from the canonical legacy overpass time.",
    )
    p.add_argument("--allow-parent-count-mismatch", action="store_true")
    return p.parse_args()


def discover_master() -> Path:
    name = "Professor_Master_Site_Date_Source_Inventory_V3_MethaneSAT_AVIRIS3_EMIT.xlsx"
    candidates = [
        HOME / "Downloads" / name,
        PROJECT_ROOT / name,
        Path("/Volumes/engg-leung/dora lin") / name,
        Path("/Volumes/engg-leung-1/dora lin") / name,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(
        "Could not auto-find Professor Master V3 workbook. Pass --master explicitly.\nTried:\n"
        + "\n".join(f"  {x}" for x in candidates)
    )


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def as_bool(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = normalize_text(x).lower()
    return s in {"1", "true", "t", "yes", "y"}


def combine_date_time(date_value: Any, time_value: Any) -> pd.Timestamp:
    d = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(d):
        return pd.NaT
    d = pd.Timestamp(d).normalize()

    if pd.isna(time_value):
        return pd.Timestamp(d, tz="UTC")

    # Excel may yield datetime.time, datetime, Timestamp, or string.
    if hasattr(time_value, "hour") and hasattr(time_value, "minute"):
        hour = int(time_value.hour)
        minute = int(time_value.minute)
        second = int(getattr(time_value, "second", 0))
        micro = int(getattr(time_value, "microsecond", 0))
        return pd.Timestamp(d) + pd.Timedelta(hours=hour, minutes=minute, seconds=second, microseconds=micro)

    s = normalize_text(time_value)
    if not s:
        return pd.Timestamp(d)
    t = pd.to_datetime(s, errors="coerce")
    if pd.isna(t):
        return pd.Timestamp(d)
    return pd.Timestamp(d) + pd.Timedelta(
        hours=int(t.hour), minutes=int(t.minute), seconds=int(t.second), microseconds=int(t.microsecond)
    )


def ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def load_parent_events(master: Path) -> pd.DataFrame:
    cand = pd.read_excel(master, sheet_name="Candidate_Negatives", engine="openpyxl")
    gt = pd.read_excel(master, sheet_name="Confirmed_GroundTruth", engine="openpyxl")

    req = ["Site", "Latitude", "Longitude", "Date", "UTC Time", "Days After Positive", "Source Positive Record ID"]
    missing = [c for c in req if c not in cand.columns]
    if missing:
        raise RuntimeError(f"Candidate_Negatives missing columns: {missing}")

    c = cand.copy()
    c["_candidate_date"] = pd.to_datetime(c["Date"], errors="coerce")
    c["_offset"] = pd.to_numeric(c["Days After Positive"], errors="coerce")
    c["_positive_date"] = (c["_candidate_date"] - pd.to_timedelta(c["_offset"], unit="D")).dt.normalize()

    rows = []
    for source_id, g in c.groupby("Source Positive Record ID", sort=True, dropna=False):
        source_id = normalize_text(source_id)
        if not source_id:
            continue
        dates = sorted(pd.Timestamp(x) for x in g["_positive_date"].dropna().unique())
        if len(dates) != 1:
            continue
        r0 = g.iloc[0]
        event_date = dates[0]
        event_time = combine_date_time(event_date, r0.get("UTC Time"))
        source = "Candidate_Negatives_reconstruction"
        gt_match = pd.DataFrame()
        if "Scene/Observation ID" in gt.columns:
            gt_match = gt[gt["Scene/Observation ID"].astype(str).eq(source_id)].copy()
        if len(gt_match):
            # Prefer a positive MethaneAIR row with valid date/time if duplicates exist.
            if "Label" in gt_match.columns:
                pos = gt_match[pd.to_numeric(gt_match["Label"], errors="coerce").eq(1)]
                if len(pos):
                    gt_match = pos
            valid = gt_match[pd.to_datetime(gt_match["Date"], errors="coerce").notna()]
            if len(valid):
                gr = valid.iloc[0]
                event_date = pd.Timestamp(pd.to_datetime(gr["Date"])).normalize()
                event_time = combine_date_time(event_date, gr.get("UTC Time"))
                source = "Confirmed_GroundTruth"
                if pd.notna(gr.get("Latitude")):
                    r0 = r0.copy(); r0["Latitude"] = gr.get("Latitude")
                if pd.notna(gr.get("Longitude")):
                    r0 = r0.copy(); r0["Longitude"] = gr.get("Longitude")
                if normalize_text(gr.get("Site")):
                    r0 = r0.copy(); r0["Site"] = gr.get("Site")

        rows.append({
            "source_positive_record_id": source_id,
            "site": normalize_text(r0.get("Site")),
            "lat": float(pd.to_numeric(pd.Series([r0.get("Latitude")]), errors="coerce").iloc[0]),
            "lon": float(pd.to_numeric(pd.Series([r0.get("Longitude")]), errors="coerce").iloc[0]),
            "parent_positive_date": event_date.strftime("%Y-%m-%d"),
            "parent_positive_datetime_utc": ensure_utc(event_time).isoformat(),
            "parent_event_source": source,
        })

    out = pd.DataFrame(rows)
    if out["source_positive_record_id"].duplicated().any():
        raise RuntimeError("Parent event reconstruction produced duplicate source IDs.")
    return out


def select_one_negative_per_parent(canonical_path: Path, eval_path: Path, parents: pd.DataFrame, allow_mismatch: bool) -> pd.DataFrame:
    canon = pd.read_csv(canonical_path, low_memory=False)
    ev = pd.read_csv(eval_path, low_memory=False)

    if len(canon) != EXPECTED_NEG_CONTROLS or len(ev) != EXPECTED_NEG_CONTROLS:
        raise RuntimeError(f"Expected 368 negative controls; canonical={len(canon)}, eval={len(ev)}")

    if "control_id" not in canon.columns:
        raise RuntimeError("Negative canonical missing control_id.")
    if "id" not in ev.columns:
        raise RuntimeError("Negative eval CSV missing id.")

    # Rename evaluator-side fields first so parent metadata can later use clean site/lat/lon names.
    ev = ev.rename(columns={
        "site": "negative_site_eval",
        "lat": "negative_lat_eval",
        "lon": "negative_lon_eval",
        "scene_id": "negative_scene_id_eval",
        "acquisition_time_utc": "negative_acquisition_time_utc_eval",
        "s2_0_path": "s2_0_path_eval",
        "s2_90_path": "s2_90_path_eval",
        "s2_360_path": "s2_360_path_eval",
        # v10 strict-eval rows also carry parent metadata.  Keep those as
        # audit fields so they cannot collide with the authoritative parent
        # reconstruction below.
        "source_positive_record_id": "negative_source_positive_record_id_eval",
        "parent_positive_date": "negative_parent_positive_date_eval",
        "parent_positive_datetime_utc": "negative_parent_positive_datetime_utc_eval",
        "parent_event_source": "negative_parent_event_source_eval",
    })
    merged = canon.merge(ev, left_on="control_id", right_on="id", how="inner", suffixes=("", "_eval"), validate="one_to_one")
    if len(merged) != EXPECTED_NEG_CONTROLS:
        raise RuntimeError(f"Canonical/eval join did not preserve 368 rows: {len(merged)}")

    source_col = "Source Positive Record ID"
    if source_col not in merged.columns:
        raise RuntimeError(f"Missing {source_col} in negative canonical.")

    grade = merged["Final Evidence Grade"].astype(str)
    merged["_grade_priority"] = grade.map({
        "B1_STRONG_HIGH_RES_NO_L4_DETECTION": 0,
        "B2_HIGH_RES_NO_L4_DETECTION_BACKGROUND_WEAK": 1,
    }).fillna(9)
    merged["_delta"] = pd.to_numeric(merged.get("Minimum Absolute S2 Delta Hours"), errors="coerce").fillna(1e9)
    merged["_clear"] = pd.to_numeric(merged.get("S2 Clear Over Requested Fraction"), errors="coerce").fillna(-1)
    merged["_support"] = pd.to_numeric(merged.get("Supporting MethaneAIR Flight Count"), errors="coerce").fillna(0)

    merged = merged.sort_values(
        [source_col, "_grade_priority", "_delta", "_clear", "_support", "control_id"],
        ascending=[True, True, True, False, False, True],
        kind="mergesort",
    )
    selected = merged.groupby(source_col, sort=True, as_index=False).head(1).copy()

    if len(selected) != EXPECTED_PARENT_COUNT and not allow_mismatch:
        raise RuntimeError(
            f"Selected unique parent count {len(selected)} != expected {EXPECTED_PARENT_COUNT}. "
            "Do not proceed until the 368 source is reconciled, or use --allow-parent-count-mismatch intentionally."
        )

    selected["source_positive_record_id"] = selected[source_col].astype(str)

    # Defensive cleanup for older/newer eval manifests: parent fields from the
    # negative row must never suffix the authoritative parent reconstruction
    # into parent_positive_date_x / parent_positive_date_y.  Preserve any such
    # fields for audit under a negative_* name before joining.
    authoritative_parent_cols = [
        "site", "lat", "lon", "parent_positive_date",
        "parent_positive_datetime_utc", "parent_event_source",
    ]
    for c in authoritative_parent_cols:
        if c in selected.columns:
            audit_name = f"negative_{c}_prejoin"
            if audit_name in selected.columns:
                audit_name = f"{audit_name}_2"
            selected = selected.rename(columns={c: audit_name})

    selected = selected.merge(
        parents,
        on="source_positive_record_id",
        how="left",
        validate="one_to_one",
    )

    required_parent_cols = [
        "site", "lat", "lon", "parent_positive_date",
        "parent_positive_datetime_utc", "parent_event_source",
    ]
    missing_parent_cols = [c for c in required_parent_cols if c not in selected.columns]
    if missing_parent_cols:
        raise RuntimeError(
            "Parent metadata join did not produce required columns: "
            + ", ".join(missing_parent_cols)
            + "\nAvailable parent-like columns: "
            + ", ".join(sorted(c for c in selected.columns if "parent" in c.lower() or c in {"site", "lat", "lon"}))
        )

    if selected[["lat", "lon", "parent_positive_date", "parent_positive_datetime_utc"]].isna().any().any():
        bad = selected[
            selected[["lat", "lon", "parent_positive_date", "parent_positive_datetime_utc"]]
            .isna().any(axis=1)
        ]["source_positive_record_id"].tolist()
        raise RuntimeError(
            f"Could not reconstruct parent positive metadata for {len(bad)} IDs: {bad[:10]}"
        )

    selected = selected.sort_values(
        ["site", "parent_positive_date", "source_positive_record_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    selected.insert(0, "pair_id", [f"MAIRPAIR_{i:04d}" for i in range(1, len(selected)+1)])
    selected.insert(1, "positive_id", [f"MAIRPOS_{i:04d}" for i in range(1, len(selected)+1)])
    selected["negative_id"] = selected["control_id"].astype(str)
    return selected.drop(columns=[c for c in ["_grade_priority", "_delta", "_clear", "_support"] if c in selected.columns])



def discover_existing(explicit: str, candidates: list[Path], label: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        return path
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not auto-find {label}. Tried:\n" + "\n".join(f"  {x}" for x in candidates)
    )


def attach_legacy_positive_evidence(
    selected: pd.DataFrame,
    manifest_path: Path,
    readiness_path: Path,
) -> pd.DataFrame:
    """Attach the already-audited positive-side S2 temporal selection/readiness.

    This is deliberately keyed by the original MethaneAIR L4 record ID.  It avoids
    performing a fresh nearest-S2 search for the parent positive and therefore keeps
    the paired benchmark aligned with the previously frozen strict-readiness pipeline.
    """
    manifest = pd.read_csv(manifest_path, low_memory=False)
    readiness = pd.read_csv(readiness_path, low_memory=False)

    manifest_required = [
        "record_id", "label", "ground_truth_time_utc",
        "t0_scene_id", "t0_scene_time_utc", "t0_time_delta_hours",
        "t90_scene_id", "t90_scene_time_utc", "t90_time_delta_hours",
        "t360_scene_id", "t360_scene_time_utc", "t360_time_delta_hours",
    ]
    readiness_required = [
        "record_id", "label", "all_three_technical_pass",
        "all_three_qa_pass_recomputed", "strict_t0_aligned_72h",
        "strict_model_ready",
    ]
    mm = [c for c in manifest_required if c not in manifest.columns]
    rr = [c for c in readiness_required if c not in readiness.columns]
    if mm:
        raise RuntimeError(f"Legacy temporal manifest missing columns: {mm}")
    if rr:
        raise RuntimeError(f"Legacy readiness file missing columns: {rr}")

    manifest = manifest.copy()
    readiness = readiness.copy()
    manifest["record_id"] = manifest["record_id"].astype(str).str.strip()
    readiness["record_id"] = readiness["record_id"].astype(str).str.strip()
    if manifest["record_id"].duplicated().any():
        raise RuntimeError("Legacy temporal manifest has duplicate record_id values.")
    if readiness["record_id"].duplicated().any():
        raise RuntimeError("Legacy readiness file has duplicate record_id values.")

    # Keep only one copy of shared identifiers and prefix every legacy/readiness
    # field so it cannot collide with the negative-control or parent metadata.
    m = manifest[manifest_required + [
        c for c in [
            "t0_clear_fraction", "t0_qa_pass", "t0_status",
            "t90_clear_fraction", "t90_qa_pass", "t90_status",
            "t360_clear_fraction", "t360_qa_pass", "t360_status",
            "all_three_downloaded", "all_three_qa_pass",
        ] if c in manifest.columns
    ]].copy()
    m = m.rename(columns={c: f"legacy_{c}" for c in m.columns if c != "record_id"})

    r = readiness[readiness_required].copy()
    r = r.rename(columns={c: f"readiness_{c}" for c in r.columns if c != "record_id"})

    out = selected.merge(
        m,
        left_on="source_positive_record_id",
        right_on="record_id",
        how="left",
        validate="one_to_one",
    ).drop(columns=["record_id"])
    out = out.merge(
        r,
        left_on="source_positive_record_id",
        right_on="record_id",
        how="left",
        validate="one_to_one",
    ).drop(columns=["record_id"])

    parent_time = pd.to_datetime(out["parent_positive_datetime_utc"], utc=True, errors="coerce")
    t0_time = pd.to_datetime(out["legacy_t0_scene_time_utc"], utc=True, errors="coerce")
    out["legacy_t0_abs_delta_hours_recomputed"] = (
        (t0_time - parent_time).abs().dt.total_seconds() / 3600.0
    )

    for slot in ["t0", "t90", "t360"]:
        out[f"legacy_{slot}_scene_time_present"] = pd.to_datetime(
            out[f"legacy_{slot}_scene_time_utc"], utc=True, errors="coerce"
        ).notna()

    manifest_match = out["legacy_label"].notna()
    readiness_match = out["readiness_label"].notna()
    legacy_label_pos = pd.to_numeric(out["legacy_label"], errors="coerce").eq(1)
    readiness_label_pos = pd.to_numeric(out["readiness_label"], errors="coerce").eq(1)
    strict_ready = out["readiness_strict_model_ready"].apply(as_bool)
    all_times = out[[f"legacy_{s}_scene_time_present" for s in ["t0", "t90", "t360"]]].all(axis=1)
    t0_72 = out["legacy_t0_abs_delta_hours_recomputed"].le(POS_T0_MAX_DELTA_HOURS + 1e-9)
    t0_24 = out["legacy_t0_abs_delta_hours_recomputed"].le(STRICT24_HOURS + 1e-9)

    out["canonical_positive_eligible_72h"] = (
        manifest_match & readiness_match & legacy_label_pos & readiness_label_pos
        & strict_ready & all_times & t0_72
    )
    out["canonical_positive_eligible_24h"] = out["canonical_positive_eligible_72h"] & t0_24

    def status(row: pd.Series) -> str:
        if pd.isna(row.get("legacy_label")):
            return "NO_LEGACY_MANIFEST_ROW"
        if pd.isna(row.get("readiness_label")):
            return "NO_LEGACY_READINESS_ROW"
        if pd.to_numeric(pd.Series([row.get("legacy_label")]), errors="coerce").iloc[0] != 1:
            return "LEGACY_MANIFEST_NOT_POSITIVE"
        if pd.to_numeric(pd.Series([row.get("readiness_label")]), errors="coerce").iloc[0] != 1:
            return "LEGACY_READINESS_NOT_POSITIVE"
        if not as_bool(row.get("readiness_strict_model_ready")):
            return "LEGACY_NOT_STRICT_MODEL_READY"
        if not all(bool(row.get(f"legacy_{s}_scene_time_present")) for s in ["t0", "t90", "t360"]):
            return "LEGACY_MISSING_CANONICAL_SCENE_TIME"
        d = pd.to_numeric(pd.Series([row.get("legacy_t0_abs_delta_hours_recomputed")]), errors="coerce").iloc[0]
        if pd.isna(d) or float(d) > POS_T0_MAX_DELTA_HOURS + 1e-9:
            return "LEGACY_T0_OUTSIDE_72H"
        return "CANONICAL_POSITIVE_ELIGIBLE_72H"

    out["canonical_positive_status"] = out.apply(status, axis=1)
    return out


def legacy_mgrs_tile_hint(scene_id: Any) -> str:
    text = normalize_text(scene_id).upper()
    if not text:
        return ""
    # Typical Element84 S2 L2A IDs contain a 5-character MGRS tile such as 15SUR.
    matches = re.findall(r"(?:^|[_-])T?(\d{2}[A-Z]{3})(?=[_-]|$)", text)
    return matches[0] if matches else ""


def resolve_canonical_overpass_image(
    latitude: float,
    longitude: float,
    canonical_time: pd.Timestamp,
    max_scene_cloud: float,
    tolerance_minutes: float,
    canonical_scene_id: Any = "",
) -> tuple[Any | None, dict[str, Any]]:
    """Re-resolve the same canonical overpass in GEE, then apply corrected QA.

    The old manifest's selected scene time is authoritative.  We search only a
    narrow window around that time and never substitute a different day merely
    because it has better QA.
    """
    canonical_time = ensure_utc(canonical_time)
    tolerance = pd.Timedelta(minutes=float(tolerance_minutes))
    start = canonical_time - tolerance
    end = canonical_time + tolerance + pd.Timedelta(seconds=1)
    ee = BASE.ee
    point = ee.Geometry.Point([float(longitude), float(latitude)])
    collection = (
        ee.ImageCollection(BASE.S2_COLLECTION)
        .filterBounds(point)
        .filterDate(start.isoformat(), end.isoformat())
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", float(max_scene_cloud)))
        .sort("system:time_start")
    )
    size = int(BASE.retry_call(lambda: collection.size().getInfo()))
    tile_hint = legacy_mgrs_tile_hint(canonical_scene_id)
    info: dict[str, Any] = {
        "canonical_scene_time_utc": canonical_time,
        "canonical_scene_id": normalize_text(canonical_scene_id),
        "canonical_mgrs_tile_hint": tile_hint,
        "canonical_search_start_utc": start,
        "canonical_search_end_utc": end,
        "candidate_scene_count": size,
    }
    if size == 0:
        info["selection_reason"] = "canonical_overpass_no_GEE_candidate"
        return None, info

    raw = BASE.retry_call(
        lambda: collection.toList(size).map(
            lambda obj: ee.Dictionary({
                "asset_id": ee.Image(obj).id(),
                "system_index": ee.Image(obj).get("system:index"),
                "time_start": ee.Image(obj).get("system:time_start"),
                "cloud_pct": ee.Image(obj).get("CLOUDY_PIXEL_PERCENTAGE"),
                "mgrs_tile": ee.Image(obj).get("MGRS_TILE"),
                "product_id": ee.Image(obj).get("PRODUCT_ID"),
                "spacecraft_name": ee.Image(obj).get("SPACECRAFT_NAME"),
            })
        ).getInfo()
    )

    items: list[dict[str, Any]] = []
    for item in raw:
        if item.get("time_start") is None:
            continue
        ts = pd.to_datetime(item["time_start"], unit="ms", utc=True)
        item = dict(item)
        item["acquisition_time_utc"] = ts
        item["canonical_time_difference_seconds"] = abs((ts - canonical_time).total_seconds())
        item["asset_id"] = BASE.resolve_s2_asset_id(item)
        items.append(item)

    if not items:
        info["selection_reason"] = "canonical_overpass_candidates_missing_time"
        return None, info

    items.sort(key=lambda x: (float(x["canonical_time_difference_seconds"]), str(x.get("asset_id"))))
    closest_seconds = float(items[0]["canonical_time_difference_seconds"])
    info["closest_candidate_time_difference_seconds"] = closest_seconds
    if closest_seconds > float(tolerance_minutes) * 60.0 + 1e-6:
        info["selection_reason"] = "canonical_overpass_outside_time_tolerance"
        return None, info

    # Multiple MGRS tiles from the same overpass can intersect the source.  Keep
    # only the closest overpass (20-minute grouping), then choose the tile with
    # the best corrected local QA at the source.
    nearest_overpass = [
        x for x in items
        if abs((x["acquisition_time_utc"] - items[0]["acquisition_time_utc"]).total_seconds())
        <= BASE.OVERPASS_GROUP_MINUTES * 60
    ]
    info["candidate_overpass_tile_count"] = len(nearest_overpass)
    matching_tile = [x for x in nearest_overpass if tile_hint and normalize_text(x.get("mgrs_tile")).upper() == tile_hint]
    if matching_tile:
        nearest_overpass = matching_tile
        info["canonical_mgrs_tile_match_available"] = True
    else:
        info["canonical_mgrs_tile_match_available"] = False

    scored = []
    errors: list[str] = []
    for item in nearest_overpass:
        try:
            image = ee.Image(item["asset_id"])
            qa = BASE.retry_call(BASE.corrected_s2_qa, image, latitude, longitude)
            scored.append({"image": image, "item": item, "qa": qa})
        except Exception as exc:
            errors.append(
                f"{item.get('system_index') or item.get('asset_id')} | {type(exc).__name__}: {exc}"
            )

    info["qa_error_count"] = len(errors)
    info["qa_errors"] = " || ".join(errors[:10])
    if not scored:
        info["selection_reason"] = "all_candidate_QA_queries_failed"
        return None, info

    # Same canonical overpass only: current QA does not permit hopping to a
    # different date.  Prefer a QA-pass tile, then clear/requested and coverage.
    scored.sort(
        key=lambda x: (
            bool(x["qa"].get("qa_pass")),
            float(x["qa"].get("clear_over_requested_fraction") or -1),
            float(x["qa"].get("coverage_fraction") or -1),
            -float(x["item"]["canonical_time_difference_seconds"]),
        ),
        reverse=True,
    )
    best = scored[0]
    item = best["item"]
    qa = best["qa"]
    meta = {
        **item,
        **{f"qa_{k}": v for k, v in qa.items()},
        **info,
        "selection_reason": "canonical_overpass_reused",
        "selected_mgrs_tile": item.get("mgrs_tile"),
        "canonical_mgrs_tile_match": bool(tile_hint and normalize_text(item.get("mgrs_tile")).upper() == tile_hint),
    }
    return best["image"], meta

def json_default(obj: Any):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if math.isnan(x) else x
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj):
        return None
    return str(obj)


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = normalize_text(obj.get("source_positive_record_id"))
            if key:
                latest[key] = obj
    return latest


def append_checkpoint(path: Path, obj: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=json_default) + "\n")
        f.flush(); os.fsync(f.fileno())


def process_positive(row: pd.Series, out_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    pair_id = str(row["pair_id"])
    positive_id = str(row["positive_id"])
    source_id = str(row["source_positive_record_id"])
    lat = float(row["lat"])
    lon = float(row["lon"])
    parent_time = ensure_utc(pd.to_datetime(row["parent_positive_datetime_utc"], utc=True))
    patch_dir = out_root / "positive_patches" / positive_id

    base_summary = {
        "pair_id": pair_id,
        "positive_id": positive_id,
        "negative_id": str(row["negative_id"]),
        "source_positive_record_id": source_id,
        "site": str(row["site"]),
        "lat": lat,
        "lon": lon,
        "parent_positive_date": str(row["parent_positive_date"]),
        "parent_positive_datetime_utc": parent_time.isoformat(),
        "parent_event_source": str(row["parent_event_source"]),
        "negative_evidence_grade": str(row["Final Evidence Grade"]),
        "canonical_positive_status": str(row.get("canonical_positive_status", "")),
        "legacy_strict_model_ready": as_bool(row.get("readiness_strict_model_ready")),
        "legacy_t0_abs_delta_hours_recomputed": pd.to_numeric(
            pd.Series([row.get("legacy_t0_abs_delta_hours_recomputed")]), errors="coerce"
        ).iloc[0],
    }

    if not as_bool(row.get("canonical_positive_eligible_72h")):
        return {
            "complete": True,
            "retryable": False,
            "source_positive_record_id": source_id,
            "sample_summary": {**base_summary, "sample_result": "NOT_CANONICAL_POSITIVE_ELIGIBLE_72H"},
            "frames": [],
        }

    frames: list[dict[str, Any]] = []
    paths: dict[str, str] = {}
    all_technical = True
    all_qa = True
    retryable = False
    current_t0_delta_h = float("nan")
    current_t0_product_id = ""
    current_t0_datetime = pd.NaT

    for frame_name in ["t0", "t90", "t360"]:
        canonical_time_raw = row.get(f"legacy_{frame_name}_scene_time_utc")
        canonical_scene_id = row.get(f"legacy_{frame_name}_scene_id")
        canonical_time = pd.to_datetime(canonical_time_raw, utc=True, errors="coerce")
        output_path = patch_dir / f"{positive_id}__{frame_name}.tif"

        fr: dict[str, Any] = {
            **base_summary,
            "frame": frame_name,
            "frame_output_path": str(output_path.resolve()),
            "legacy_canonical_scene_id": canonical_scene_id,
            "legacy_canonical_scene_time_utc": canonical_time_raw,
            "legacy_time_delta_hours": row.get(f"legacy_{frame_name}_time_delta_hours"),
            "legacy_clear_fraction": row.get(f"legacy_{frame_name}_clear_fraction"),
            "legacy_qa_pass": row.get(f"legacy_{frame_name}_qa_pass"),
            "legacy_status": row.get(f"legacy_{frame_name}_status"),
        }

        if pd.isna(canonical_time):
            fr.update({
                "frame_status": "no_canonical_scene_time",
                "frame_selection_reason": "legacy_canonical_scene_time_missing",
                "download_status": "not_attempted",
                "tiff_valid": False,
                "scl_qa_pass": False,
            })
            frames.append(fr)
            all_technical = False
            all_qa = False
            continue

        try:
            image, meta = BASE.retry_call(
                resolve_canonical_overpass_image,
                latitude=lat,
                longitude=lon,
                canonical_time=canonical_time,
                max_scene_cloud=float(args.max_scene_cloud),
                tolerance_minutes=float(args.canonical_overpass_tolerance_minutes),
                canonical_scene_id=canonical_scene_id,
            )
        except Exception as exc:
            image = None
            meta = {
                "selection_reason": "canonical_overpass_query_error",
                "query_error": f"{type(exc).__name__}: {exc}",
            }
            retryable = True

        meta = dict(meta or {})
        fr.update({
            "frame_selection_reason": meta.get("selection_reason"),
            "frame_asset_id": meta.get("asset_id"),
            "frame_system_index": meta.get("system_index"),
            "frame_product_id": meta.get("product_id"),
            "frame_acquisition_time_utc": meta.get("acquisition_time_utc"),
            "frame_cloudy_pixel_percentage": meta.get("cloud_pct"),
            "frame_candidate_scene_count": meta.get("candidate_scene_count"),
            "frame_candidate_overpass_tile_count": meta.get("candidate_overpass_tile_count"),
            "canonical_mgrs_tile_hint": meta.get("canonical_mgrs_tile_hint"),
            "selected_mgrs_tile": meta.get("selected_mgrs_tile"),
            "canonical_mgrs_tile_match": meta.get("canonical_mgrs_tile_match"),
            "frame_GEE_minus_canonical_seconds": meta.get("canonical_time_difference_seconds"),
            "frame_closest_candidate_minus_canonical_seconds": meta.get("closest_candidate_time_difference_seconds"),
            "frame_qa_error_count": meta.get("qa_error_count"),
            "frame_qa_errors": meta.get("qa_errors"),
            "scl_coverage_fraction": meta.get("qa_coverage_fraction"),
            "scl_clear_among_covered_fraction": meta.get("qa_clear_among_covered_fraction"),
            "scl_clear_over_requested_fraction": meta.get("qa_clear_over_requested_fraction"),
            "scl_masked_fraction": meta.get("qa_masked_fraction"),
            "scl_qa_pass": meta.get("qa_qa_pass"),
        })

        if image is None:
            reason = str(meta.get("selection_reason") or "")
            if reason in {"canonical_overpass_query_error", "all_candidate_QA_queries_failed"}:
                retryable = True
            fr.update({"frame_status": "no_image_selected", "download_status": "not_attempted", "tiff_valid": False})
            frames.append(fr)
            all_technical = False
            all_qa = False
            continue

        actual_time = pd.to_datetime(meta.get("acquisition_time_utc"), utc=True, errors="coerce")
        if frame_name == "t0":
            if pd.notna(actual_time):
                current_t0_delta_h = abs((actual_time - parent_time).total_seconds()) / 3600.0
                current_t0_datetime = actual_time
            current_t0_product_id = str(meta.get("product_id") or "")
            fr["positive_t0_abs_delta_hours"] = current_t0_delta_h

        dl = BASE.download_geotiff(
            image=image,
            latitude=lat,
            longitude=lon,
            output_path=output_path,
            overwrite=args.overwrite,
        )
        fr.update(dl)
        if dl.get("tiff_valid"):
            paths[frame_name] = str(output_path.resolve())
        else:
            all_technical = False
            retryable = True
        if not bool(fr.get("scl_qa_pass")):
            all_qa = False
        fr["frame_status"] = "ready" if bool(dl.get("tiff_valid")) else "failed"
        frames.append(fr)

    all_technical = bool(all_technical and len(paths) == 3)
    all_qa = bool(all_qa and len(frames) == 3 and all(bool(f.get("scl_qa_pass")) for f in frames))
    t0_aligned = bool(np.isfinite(current_t0_delta_h) and current_t0_delta_h <= POS_T0_MAX_DELTA_HOURS + 1e-9)
    model_ready = bool(all_technical and all_qa and t0_aligned)

    if model_ready:
        sample_result = "STRICT_MODEL_READY"
    elif all_technical and all_qa and not t0_aligned:
        sample_result = "CURRENT_T0_ALIGNMENT_REGRESSION"
    elif all_technical and not all_qa:
        sample_result = "CURRENT_CORRECTED_QA_FAIL"
    else:
        sample_result = "TECHNICAL_INCOMPLETE"

    summary = {
        **base_summary,
        "positive_t0_product_id": current_t0_product_id,
        "positive_t0_datetime_utc": str(current_t0_datetime) if pd.notna(current_t0_datetime) else "",
        "positive_t0_abs_delta_hours": float(current_t0_delta_h) if np.isfinite(current_t0_delta_h) else np.nan,
        "all_three_technical_pass": all_technical,
        "all_three_qa_pass_corrected": all_qa,
        "strict_t0_aligned_72h_current": t0_aligned,
        "strict_model_ready": model_ready,
        "s2_0_path": paths.get("t0", ""),
        "s2_90_path": paths.get("t90", ""),
        "s2_360_path": paths.get("t360", ""),
        "sample_result": sample_result,
    }
    return {
        "complete": bool(model_ready or not retryable),
        "retryable": bool(retryable and not model_ready),
        "source_positive_record_id": source_id,
        "sample_summary": summary,
        "frames": frames,
    }

def latest_tables(latest: dict[str, dict[str, Any]], selected_ids: set[str] | None = None):
    samples, frames = [], []
    for sid, obj in latest.items():
        if selected_ids is not None and sid not in selected_ids:
            continue
        ss = obj.get("sample_summary")
        if ss:
            samples.append(ss)
        frames.extend(obj.get("frames") or [])
    return pd.DataFrame(samples), pd.DataFrame(frames)


def make_eval_rows(pair_selection: pd.DataFrame, pos_samples: pd.DataFrame, max_positive_delta_h: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not len(pos_samples) or "strict_model_ready" not in pos_samples.columns:
        ready = pd.DataFrame()
    else:
        ready = pos_samples[pos_samples["strict_model_ready"].apply(as_bool)].copy()
    if ready.empty:
        return pd.DataFrame(), pd.DataFrame()
    ready["positive_t0_abs_delta_hours"] = pd.to_numeric(ready["positive_t0_abs_delta_hours"], errors="coerce")
    ready = ready[ready["positive_t0_abs_delta_hours"].le(max_positive_delta_h + 1e-9)].copy()
    if ready.empty:
        return pd.DataFrame(), pd.DataFrame()

    pairs = pair_selection.merge(
        ready[[
            "pair_id", "positive_id", "source_positive_record_id", "positive_t0_product_id", "positive_t0_datetime_utc",
            "positive_t0_abs_delta_hours", "s2_0_path", "s2_90_path", "s2_360_path"
        ]],
        on=["pair_id", "positive_id", "source_positive_record_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_pos"),
    )

    rows = []
    for _, r in pairs.iterrows():
        common = {
            "pair_id": r["pair_id"],
            "site": r["site"],
            "source_positive_record_id": r["source_positive_record_id"],
            "parent_positive_date": r["parent_positive_date"],
            "parent_positive_datetime_utc": r["parent_positive_datetime_utc"],
            "lat": r["lat"],
            "lon": r["lon"],
            "negative_evidence_grade": r["Final Evidence Grade"],
            "positive_t0_abs_delta_hours": r["positive_t0_abs_delta_hours"],
        }
        rows.append({
            **common,
            "id": r["positive_id"],
            "sample_id": r["positive_id"],
            "pair_role": "positive",
            "label": 1,
            "label_provenance": "MethaneAIR_L4_parent_positive_S2_within_72h",
            "ground_truth_type": "MethaneAIR_L4_observed_plume_parent",
            "scene_id": r["positive_t0_product_id"],
            "acquisition_time_utc": r["positive_t0_datetime_utc"],
            "s2_0_path": r["s2_0_path"],
            "s2_90_path": r["s2_90_path"],
            "s2_360_path": r["s2_360_path"],
        })
        rows.append({
            **common,
            "id": r["negative_id"],
            "sample_id": r["negative_id"],
            "pair_role": "negative",
            "label": 0,
            "label_provenance": r["Final Evidence Grade"],
            "ground_truth_type": "high_res_no_L4_detection_temporal_control",
            "scene_id": r["S2 Product ID"],
            "acquisition_time_utc": r["S2 Datetime UTC"],
            "s2_0_path": r["s2_0_path_eval"],
            "s2_90_path": r["s2_90_path_eval"],
            "s2_360_path": r["s2_360_path_eval"],
        })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["pair_id", "label"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    pair_cols = [
        "pair_id", "positive_id", "negative_id", "source_positive_record_id", "site", "lat", "lon",
        "parent_positive_date", "parent_positive_datetime_utc", "Final Evidence Grade", "positive_t0_product_id",
        "positive_t0_datetime_utc", "positive_t0_abs_delta_hours", "S2 Product ID", "S2 Datetime UTC"
    ]
    return pairs[[c for c in pair_cols if c in pairs.columns]].copy(), out


def write_outputs(out_root: Path, pair_selection: pd.DataFrame, latest: dict[str, dict[str, Any]], selected_ids: set[str]):
    manifests = out_root / "manifests"; manifests.mkdir(parents=True, exist_ok=True)
    samples, frames = latest_tables(latest, selected_ids)
    if len(samples):
        samples = samples.sort_values("pair_id", kind="mergesort").reset_index(drop=True)
    if len(frames):
        order = pd.Categorical(frames["frame"], ["t0", "t90", "t360"], ordered=True)
        frames = frames.assign(_ord=order).sort_values(["pair_id", "_ord"]).drop(columns="_ord").reset_index(drop=True)

    p72, e72 = make_eval_rows(pair_selection, samples, 72.0)
    p24, e24 = make_eval_rows(pair_selection, samples, STRICT24_HOURS)

    pair_selection.to_csv(manifests / "00_pair_selection.csv", index=False)
    frames.to_csv(manifests / "01_positive_frame_manifest.csv", index=False)
    samples.to_csv(manifests / "02_positive_sample_audit.csv", index=False)
    p72.to_csv(manifests / "03_final_model_ready_pairs_72h.csv", index=False)
    e72.to_csv(manifests / "04_paired_eval_72h.csv", index=False)
    p24.to_csv(manifests / "05_final_model_ready_pairs_24h.csv", index=False)
    e24.to_csv(manifests / "06_paired_eval_24h.csv", index=False)

    if len(e72):
        dest = METHANEFUSE_ROOT / "data" / "custom" / "methaneair_sameparent_paired_72h_eval.csv"
        dest.parent.mkdir(parents=True, exist_ok=True); e72.to_csv(dest, index=False)
    if len(e24):
        dest = METHANEFUSE_ROOT / "data" / "custom" / "methaneair_sameparent_paired_24h_eval.csv"
        dest.parent.mkdir(parents=True, exist_ok=True); e24.to_csv(dest, index=False)

    ready_n = len(p72)
    ready24 = len(p24)
    summary = [
        f"{DATASET_NAME} BUILD SUMMARY",
        "="*88,
        f"Selected unique parents: {len(pair_selection)}",
        f"Positive samples represented: {len(samples)}",
        f"Positive strict model-ready (<=72h): {ready_n}",
        f"Balanced 72h eval rows: {len(e72)} = {ready_n} positive + {ready_n} negative",
        f"Positive strict model-ready (<=24h): {ready24}",
        f"Balanced 24h eval rows: {len(e24)} = {ready24} positive + {ready24} negative",
    ]
    if len(samples) and "sample_result" in samples:
        summary += ["", "Positive sample result:", samples["sample_result"].value_counts(dropna=False).to_string()]
    if len(p72) and "Final Evidence Grade" in p72:
        summary += ["", "Negative evidence grade among 72h pairs:", p72["Final Evidence Grade"].value_counts(dropna=False).to_string()]
    if len(p72) and "site" in p72:
        summary += ["", "72h pair count by site:", p72["site"].value_counts(dropna=False).to_string()]
    text = "\n".join(summary) + "\n"
    (manifests / "07_build_summary.txt").write_text(text, encoding="utf-8")

    try:
        with pd.ExcelWriter(manifests / "08_build_audit.xlsx", engine="openpyxl") as w:
            pair_selection.to_excel(w, sheet_name="Pair_Selection", index=False)
            frames.to_excel(w, sheet_name="Positive_Frames", index=False)
            samples.to_excel(w, sheet_name="Positive_Samples", index=False)
            p72.to_excel(w, sheet_name="Pairs_72h", index=False)
            e72.to_excel(w, sheet_name="Eval_72h", index=False)
            p24.to_excel(w, sheet_name="Pairs_24h", index=False)
            e24.to_excel(w, sheet_name="Eval_24h", index=False)
    except Exception as exc:
        print(f"[WARN] Excel audit skipped: {type(exc).__name__}: {exc}")

    return samples, frames, p72, e72, p24, e24, text


def main():
    args = parse_args()
    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    checkpoint = out_root / "checkpoint" / "positive_build_results_v11_2.jsonl"

    master = Path(args.master).expanduser() if args.master else discover_master()
    canonical = Path(args.negative_canonical).expanduser()
    neg_eval = Path(args.negative_eval).expanduser()
    legacy_manifest = discover_existing(
        args.legacy_manifest,
        LEGACY_MANIFEST_DEFAULTS,
        "canonical positive Sentinel-2 temporal manifest",
    )
    legacy_readiness = discover_existing(
        args.legacy_readiness,
        LEGACY_READINESS_DEFAULTS,
        "canonical positive Sentinel-2 readiness file",
    )
    for path in [master, canonical, neg_eval, legacy_manifest, legacy_readiness]:
        if not path.exists():
            raise FileNotFoundError(path)

    parents = load_parent_events(master)
    selected = select_one_negative_per_parent(
        canonical,
        neg_eval,
        parents,
        args.allow_parent_count_mismatch,
    )
    selected = attach_legacy_positive_evidence(
        selected,
        manifest_path=legacy_manifest,
        readiness_path=legacy_readiness,
    )

    preflight = out_root / "manifests" / "00_pair_selection.csv"
    preflight.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(preflight, index=False)

    eligible72 = selected[selected["canonical_positive_eligible_72h"].apply(as_bool)].copy()
    eligible24 = selected[selected["canonical_positive_eligible_24h"].apply(as_bool)].copy()

    print("=" * 104)
    print("METHANEAIR SAME-PARENT PAIRED BENCHMARK V11.2")
    print("=" * 104)
    print("Master:", master)
    print("Negative canonical:", canonical)
    print("Negative eval:", neg_eval)
    print("Legacy positive manifest:", legacy_manifest)
    print("Legacy positive readiness:", legacy_readiness)
    print("Unique selected parents:", len(selected))
    print("Canonical strict-positive eligible <=72h:", len(eligible72))
    print("Canonical strict-positive eligible <=24h:", len(eligible24))

    print("\nCanonical positive eligibility status:")
    print(selected["canonical_positive_status"].value_counts(dropna=False).to_string())
    print("\nSelected negative evidence grade:")
    print(selected["Final Evidence Grade"].value_counts(dropna=False).to_string())
    print("\nAll selected parents by site:")
    print(selected["site"].value_counts(dropna=False).to_string())
    if len(eligible72):
        print("\nCanonical <=72h eligible parents by site:")
        print(eligible72["site"].value_counts(dropna=False).to_string())
        print("\nCanonical t0 absolute delta hours among <=72h eligible:")
        print(
            pd.to_numeric(eligible72["legacy_t0_abs_delta_hours_recomputed"], errors="coerce")
            .describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])
            .to_string()
        )

    print("\nNegative selection rule: B1 > B2; min MethaneAIR-S2 delta; max S2 clear; max flight support; control_id tie-break.")
    print("Positive selection rule: reuse previously audited canonical t0/t90/t360 overpass times; no fresh nearest-date substitution.")
    print("Current build: re-resolve each canonical overpass in GEE, re-run corrected common-grid SCL QA80, export 12-band 48x48 TIFFs.")

    if args.preflight_only:
        print("\nPREFLIGHT ONLY — no Earth Engine queries/downloads performed.")
        print("Pair selection + canonical eligibility written to:", preflight)
        return

    if eligible72.empty:
        raise RuntimeError(
            "None of the selected parents are canonical strict-positive eligible within 72 h. "
            "Do not relax the cutoff automatically; inspect the preflight audit."
        )

    BASE.initialize_ee(args.project)

    # Smoke tests intentionally draw from already canonical-eligible parents.
    # This avoids spending a 5-row smoke on deterministic >72h exclusions.
    work = eligible72.head(args.limit).copy() if args.limit > 0 else eligible72.copy()
    selected_ids = set(work["source_positive_record_id"].astype(str))
    latest = load_checkpoint(checkpoint)

    todo = []
    for _, row in work.iterrows():
        sid = str(row["source_positive_record_id"])
        prev = latest.get(sid)
        if prev and prev.get("complete") and not prev.get("retryable"):
            continue
        todo.append(row)

    print(f"\nCanonical eligible parents in full paired universe: {len(eligible72)}")
    print(f"Rows represented in this run/output: {len(work)}")
    print(f"Already complete under v11.2 checkpoint: {len(work) - len(todo)}")
    print(f"To process: {len(todo)}")

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = {
                pool.submit(process_positive, row, out_root, args): str(row["source_positive_record_id"])
                for row in todo
            }
            done = 0
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "complete": False,
                        "retryable": True,
                        "source_positive_record_id": sid,
                        "sample_summary": {
                            "source_positive_record_id": sid,
                            "sample_result": "UNHANDLED_ERROR",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        "frames": [],
                        "traceback": traceback.format_exc(),
                    }
                append_checkpoint(checkpoint, result)
                latest[sid] = result
                done += 1
                ss = result.get("sample_summary", {})
                print(f"[{done}/{len(todo)}] {sid} -> {ss.get('sample_result', 'UNKNOWN')}")
                if done % 10 == 0:
                    subset = eligible72[eligible72["source_positive_record_id"].astype(str).isin(selected_ids)].copy()
                    write_outputs(out_root, subset, latest, selected_ids)

    latest = load_checkpoint(checkpoint)
    subset = eligible72[eligible72["source_positive_record_id"].astype(str).isin(selected_ids)].copy()
    _, _, _, _, _, _, text = write_outputs(out_root, subset, latest, selected_ids)

    print("\n" + "=" * 104)
    print("FINAL BUILD SUMMARY")
    print("=" * 104)
    print(text)
    print("Canonical eligible paired universe (before current corrected-QA revalidation):", len(eligible72))
    print("OUTPUT ROOT:", out_root)
    print("MethaneFuse 72h CSV:", METHANEFUSE_ROOT / "data/custom/methaneair_sameparent_paired_72h_eval.csv")
    print("MethaneFuse 24h CSV:", METHANEFUSE_ROOT / "data/custom/methaneair_sameparent_paired_24h_eval.csv")
    print("\nIMPORTANT: positive labels are MethaneAIR L4 parent detections. The S2 t0 is the previously audited canonical temporal match; it is not itself a simultaneous plume confirmation.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Existing positive TIFFs and checkpoint are preserved.")
        raise
