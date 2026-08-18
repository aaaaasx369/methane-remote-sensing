#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_methanesat_temporal_negatives_against_inventory.py

Purpose
-------
Second-stage validation for MethaneSAT same-location temporal-negative candidates.

This script DOES NOT call "no L4 detection" a confirmed no-emission event.

It classifies each candidate as one of:

CONFIRMED_NO_RELEASE
    A confirmed no-release / zero-release ground-truth record exists close in
    space and time in the user's master inventory.

TEMPORAL_WEAK_NEGATIVE_CLEAN
    No known positive/release record is found close in space/time in the master
    inventory, no MethaneSAT L4 point exists within 2 km, and (by default) there
    is no L4 point within 10 km. This remains a WEAK negative, not proof of zero
    emissions.

TEMPORAL_WEAK_NEGATIVE_NEARBY_L4
    No known same-site positive/release conflict, but another L4 point exists
    within 10 km. Kept for audit, excluded from the strict download manifest.

REJECT_KNOWN_POSITIVE
    A known positive/release/plume record exists close in space/time in the
    master inventory.

REJECT_L4_NEAR_SITE
    MethaneSAT L4 point exists within the 2-km site exclusion radius.

REJECT_SAME_POSITIVE_COLLECTION
    Candidate collection is actually the original positive collection after
    collection-ID normalization.

The script also fixes the old c-prefix issue:
    c016706B0 == 016706B0

Inputs
------
--selected
    03_selected_temporal_negatives.csv from Phase A.

--master
    CSV containing the user's combined All_Inventory-style table.
    If omitted, a small set of known local / SMB paths is tried.

Outputs
-------
00_validated_all_selected.csv
01_strict_temporal_negatives.csv
02_rejected_or_flagged.csv
03_validation_summary_by_positive.csv
SUMMARY_TEMPORAL_NEGATIVE_VALIDATION.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--selected",
        default="~/methane_release_project/methanesat_temporal_negative_search/03_selected_temporal_negatives.csv",
    )
    p.add_argument(
        "--master",
        default="",
        help="Combined All_Inventory CSV. If omitted, known locations are tried.",
    )
    p.add_argument(
        "--out",
        default="~/methane_release_project/methanesat_temporal_negative_validation",
    )
    p.add_argument(
        "--conflict-radius-m",
        type=float,
        default=2000.0,
        help="Spatial radius for known same-site positive/release conflicts.",
    )
    p.add_argument(
        "--time-window-hours",
        type=float,
        default=24.0,
        help="Time tolerance around the MethaneSAT candidate acquisition.",
    )
    p.add_argument(
        "--nearby-l4-radius-policy",
        choices=["reject", "flag", "ignore"],
        default="reject",
        help="How to handle the Phase-A 10-km nearby-L4 flag.",
    )
    p.add_argument(
        "--min-abs-days",
        type=float,
        default=0.0,
        help="Optional minimum temporal gap from the source positive.",
    )
    p.add_argument(
        "--max-per-positive",
        type=int,
        default=6,
        help="Maximum strict negatives retained per positive after validation.",
    )
    return p.parse_args()


def normalize_name(s):
    return str(s).strip().lower().replace("_", " ").replace("-", " ")


def col_lookup(df):
    return {normalize_name(c): c for c in df.columns}


def get_col(df, aliases, required=False):
    m = col_lookup(df)
    for a in aliases:
        k = normalize_name(a)
        if k in m:
            return m[k]
    if required:
        raise ValueError(
            f"Missing required column. Tried {aliases}. Available columns:\n{list(df.columns)}"
        )
    return None


def normalize_collection_id(v):
    s = str(v or "").strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    if s[:1].lower() == "c":
        s = s[1:]
    return s.upper()


def parse_master_time(df):
    date_col = get_col(df, ["Date"], required=True)
    time_col = get_col(df, ["UTC Time", "time", "utc_time"])

    dates = pd.to_datetime(df[date_col], errors="coerce")
    if time_col is None:
        return pd.to_datetime(dates.dt.strftime("%Y-%m-%d"), utc=True, errors="coerce")

    time_txt = df[time_col].fillna("").astype(str).str.strip()
    date_txt = dates.dt.strftime("%Y-%m-%d").fillna("")
    combined = (date_txt + " " + time_txt).str.strip()
    return pd.to_datetime(combined, utc=True, errors="coerce")


def haversine_m(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return np.nan
    r = 6371008.8
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def discover_master():
    workbook_name = "Professor_Master_Site_Date_Source_Inventory_V3_MethaneSAT_AVIRIS3_EMIT.xlsx"
    candidates = [
        # Preferred: authoritative V3 master workbook.
        Path("~/Downloads").expanduser() / workbook_name,
        Path("~/methane_release_project").expanduser() / workbook_name,
        Path("/Volumes/engg-leung/dora lin") / workbook_name,
        Path("/Volumes/engg-leung-1/dora lin") / workbook_name,

        # Backward-compatible combined CSVs.
        Path("~/Downloads/all_inventory_with_methanesat_aviris3.csv").expanduser(),
        Path("~/methane_release_project/all_inventory_with_methanesat_aviris3.csv").expanduser(),
        Path("/Volumes/engg-leung/dora lin/all_inventory_with_methanesat_aviris3.csv"),
        Path("/Volumes/engg-leung-1/dora lin/all_inventory_with_methanesat_aviris3.csv"),
        Path("/Volumes/engg-leung/dora lin/all_inventory_with_methanesat.csv"),
        Path("/Volumes/engg-leung-1/dora lin/all_inventory_with_methanesat.csv"),
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(
        "Could not auto-find the V3 master workbook or combined master CSV.\n"
        "Pass it explicitly with, for example:\n"
        "  --master '/Volumes/engg-leung/dora lin/"
        "Professor_Master_Site_Date_Source_Inventory_V3_MethaneSAT_AVIRIS3_EMIT.xlsx'\n\n"
        "Tried:\n" + "\n".join(f"  {p}" for p in candidates)
    )


def load_master_table(master_path: Path) -> pd.DataFrame:
    suffix = master_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(master_path, low_memory=False)

    if suffix in {".xlsx", ".xlsm"}:
        try:
            # Explicit engine avoids the earlier "format cannot be determined" issue.
            return pd.read_excel(
                master_path,
                sheet_name="All_Inventory",
                engine="openpyxl",
            )
        except ImportError as exc:
            raise RuntimeError(
                "Reading the V3 .xlsx master requires openpyxl in this venv.\n"
                "Install with:\n"
                "  python -m pip install openpyxl\n"
                "Then rerun the SAME command."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Could not read All_Inventory from: {master_path}\n"
                f"{type(exc).__name__}: {exc}\n\n"
                "If this is the network-mounted copy, first copy the small workbook locally:\n"
                "  cp '/Volumes/engg-leung/dora lin/"
                "Professor_Master_Site_Date_Source_Inventory_V3_MethaneSAT_AVIRIS3_EMIT.xlsx' "
                "'$HOME/Downloads/'\n"
                "then rerun with --master pointing to the Downloads copy."
            ) from exc

    raise ValueError(
        f"Unsupported master format: {master_path.suffix}. "
        "Use .csv or .xlsx."
    )


def text_blob(df):
    cols = []
    for aliases in [
        ["Label Type"],
        ["Ground Truth Modality"],
        ["Historical/Experiment"],
        ["Source Dataset"],
        ["Paper/Reference"],
        ["Notes"],
    ]:
        c = get_col(df, aliases)
        if c is not None:
            cols.append(c)
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    x = df[cols].fillna("").astype(str)
    return x.apply(lambda r: " | ".join(r.values), axis=1).str.lower()


def main():
    args = parse_args()

    selected_path = Path(args.selected).expanduser()
    if not selected_path.exists():
        raise FileNotFoundError(f"Selected candidate CSV not found: {selected_path}")

    master_path = Path(args.master).expanduser() if args.master else discover_master()
    if not master_path.exists():
        raise FileNotFoundError(f"Master inventory CSV not found: {master_path}")

    outdir = Path(args.out).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    cand = pd.read_csv(selected_path)
    master = load_master_table(master_path)

    # Required candidate columns.
    for c in [
        "positive_sample_id",
        "positive_collection_id",
        "candidate_collection_id",
        "candidate_time_start",
        "latitude",
        "longitude",
        "days_from_positive",
        "l4_point_count_near_site",
        "nearby_l4_count_10km",
        "xch4_valid_fraction",
    ]:
        if c not in cand.columns:
            raise ValueError(f"Selected CSV missing column: {c}")

    # Candidate times / normalized IDs.
    cand["candidate_timestamp"] = pd.to_datetime(
        cand["candidate_time_start"], utc=True, errors="coerce"
    )
    cand["positive_collection_norm"] = cand["positive_collection_id"].map(
        normalize_collection_id
    )
    cand["candidate_collection_norm"] = cand["candidate_collection_id"].map(
        normalize_collection_id
    )
    cand["same_positive_collection_fixed"] = (
        cand["positive_collection_norm"] == cand["candidate_collection_norm"]
    )

    # Master fields.
    lat_col = get_col(master, ["Latitude"], required=True)
    lon_col = get_col(master, ["Longitude"], required=True)
    label_col = get_col(master, ["Label"], required=True)
    id_col = get_col(master, ["Scene/Observation ID", "scene id", "observation id"])
    site_col = get_col(master, ["Site"])
    sensor_col = get_col(master, ["Sensor"])
    label_type_col = get_col(master, ["Label Type"])
    modality_col = get_col(master, ["Ground Truth Modality"])
    notes_col = get_col(master, ["Notes"])

    master["_lat"] = pd.to_numeric(master[lat_col], errors="coerce")
    master["_lon"] = pd.to_numeric(master[lon_col], errors="coerce")
    master["_label"] = pd.to_numeric(master[label_col], errors="coerce")
    master["_time"] = parse_master_time(master)
    master["_text"] = text_blob(master)

    # Positive/release/plume records: any explicit label=1 record.
    master["_known_positive"] = master["_label"].eq(1)

    # Confirmed zero-release evidence only when the record explicitly says so.
    confirmed_zero_terms = (
        "confirmed no-release",
        "confirmed no release",
        "zero-release",
        "zero release",
        "controlled no-release",
        "controlled no release",
    )
    zero_mask = pd.Series(False, index=master.index)
    for term in confirmed_zero_terms:
        zero_mask |= master["_text"].str.contains(term, na=False)
    master["_confirmed_zero"] = master["_label"].eq(0) & zero_mask

    # Only rows with actual coordinates/time can conflict spatially + temporally.
    mvalid = master[
        master["_lat"].notna()
        & master["_lon"].notna()
        & master["_time"].notna()
    ].copy()

    output_rows = []
    time_tol = pd.Timedelta(hours=args.time_window_hours)

    for _, r in cand.iterrows():
        rec = r.to_dict()
        ts = r["candidate_timestamp"]
        lat = float(r["latitude"])
        lon = float(r["longitude"])

        rec["known_positive_conflict_count"] = 0
        rec["confirmed_zero_match_count"] = 0
        rec["nearest_known_positive_m"] = np.nan
        rec["nearest_known_positive_time_hours"] = np.nan
        rec["matched_positive_ids"] = ""
        rec["matched_positive_sites"] = ""
        rec["matched_positive_sensors"] = ""
        rec["matched_zero_ids"] = ""
        rec["validation_class"] = ""
        rec["validation_reason"] = ""

        if pd.isna(ts):
            rec["validation_class"] = "REJECT_BAD_TIME"
            rec["validation_reason"] = "candidate_timestamp_unparseable"
            output_rows.append(rec)
            continue

        if bool(r["same_positive_collection_fixed"]):
            rec["validation_class"] = "REJECT_SAME_POSITIVE_COLLECTION"
            rec["validation_reason"] = "candidate_collection_equals_original_positive_after_normalization"
            output_rows.append(rec)
            continue

        if abs(float(r["days_from_positive"])) < args.min_abs_days:
            rec["validation_class"] = "REJECT_TEMPORAL_GAP"
            rec["validation_reason"] = f"abs_days_from_positive_below_{args.min_abs_days}"
            output_rows.append(rec)
            continue

        if float(r["l4_point_count_near_site"]) > 0:
            rec["validation_class"] = "REJECT_L4_NEAR_SITE"
            rec["validation_reason"] = "methanesat_l4_point_within_2km"
            output_rows.append(rec)
            continue

        # Narrow by time first.
        mt = mvalid[
            (mvalid["_time"] >= ts - time_tol)
            & (mvalid["_time"] <= ts + time_tol)
        ].copy()

        if not mt.empty:
            mt["_distance_m"] = [
                haversine_m(lat, lon, a, b)
                for a, b in zip(mt["_lat"], mt["_lon"])
            ]
            mt["_dt_hours"] = (
                (mt["_time"] - ts).abs().dt.total_seconds() / 3600.0
            )

            nearby = mt[mt["_distance_m"] <= args.conflict_radius_m].copy()
        else:
            nearby = mt

        pos_hits = nearby[nearby["_known_positive"]].copy()
        zero_hits = nearby[nearby["_confirmed_zero"]].copy()

        rec["known_positive_conflict_count"] = int(len(pos_hits))
        rec["confirmed_zero_match_count"] = int(len(zero_hits))

        if not pos_hits.empty:
            j = pos_hits["_distance_m"].idxmin()
            rec["nearest_known_positive_m"] = float(pos_hits.loc[j, "_distance_m"])
            rec["nearest_known_positive_time_hours"] = float(pos_hits.loc[j, "_dt_hours"])

            def joinvals(df, col):
                if col is None:
                    return ""
                return "|".join(sorted(set(df[col].fillna("").astype(str)))[:20])

            rec["matched_positive_ids"] = joinvals(pos_hits, id_col)
            rec["matched_positive_sites"] = joinvals(pos_hits, site_col)
            rec["matched_positive_sensors"] = joinvals(pos_hits, sensor_col)
            rec["validation_class"] = "REJECT_KNOWN_POSITIVE"
            rec["validation_reason"] = (
                f"master_inventory_positive_within_{args.conflict_radius_m:.0f}m_"
                f"and_{args.time_window_hours:.0f}h"
            )
            output_rows.append(rec)
            continue

        if not zero_hits.empty:
            if id_col is not None:
                rec["matched_zero_ids"] = "|".join(
                    sorted(set(zero_hits[id_col].fillna("").astype(str)))[:20]
                )
            rec["validation_class"] = "CONFIRMED_NO_RELEASE"
            rec["validation_reason"] = "explicit_confirmed_no_release_or_zero_release_ground_truth"
            output_rows.append(rec)
            continue

        nearby_l4 = float(r["nearby_l4_count_10km"])
        if nearby_l4 > 0:
            if args.nearby_l4_radius_policy == "reject":
                rec["validation_class"] = "TEMPORAL_WEAK_NEGATIVE_NEARBY_L4"
                rec["validation_reason"] = "no_master_positive_conflict_but_l4_point_exists_within_10km"
                output_rows.append(rec)
                continue
            elif args.nearby_l4_radius_policy == "flag":
                rec["validation_class"] = "TEMPORAL_WEAK_NEGATIVE_CLEAN"
                rec["validation_reason"] = "no_known_positive_in_master; nearby_l4_flag_retained"
                output_rows.append(rec)
                continue

        rec["validation_class"] = "TEMPORAL_WEAK_NEGATIVE_CLEAN"
        rec["validation_reason"] = (
            "no_known_positive_or_release_record_found_in_master_within_space_time_window;"
            "not_proof_of_zero_emissions"
        )
        output_rows.append(rec)

    out = pd.DataFrame(output_rows)

    out.to_csv(outdir / "00_validated_all_selected.csv", index=False)

    strict = out[
        out["validation_class"].isin(
            ["CONFIRMED_NO_RELEASE", "TEMPORAL_WEAK_NEGATIVE_CLEAN"]
        )
    ].copy()

    # Re-rank AFTER validation, so a removed candidate can be replaced by the
    # next clean candidate only if it is already present in the Phase-A selected set.
    strict["abs_days_from_positive"] = pd.to_numeric(
        strict["days_from_positive"], errors="coerce"
    ).abs()
    strict["confirmed_rank"] = np.where(
        strict["validation_class"].eq("CONFIRMED_NO_RELEASE"), 0, 1
    )
    strict = strict.sort_values(
        [
            "positive_sample_id",
            "confirmed_rank",
            "abs_days_from_positive",
            "xch4_valid_fraction",
        ],
        ascending=[True, True, True, False],
    )
    strict["post_validation_rank"] = strict.groupby("positive_sample_id").cumcount() + 1
    strict = strict[strict["post_validation_rank"] <= args.max_per_positive].copy()
    strict["final_model_label"] = 0
    strict["download_ready"] = True

    strict.to_csv(outdir / "01_strict_temporal_negatives.csv", index=False)

    flagged = out[
        ~out.index.isin(strict.index)
        | ~out["validation_class"].isin(
            ["CONFIRMED_NO_RELEASE", "TEMPORAL_WEAK_NEGATIVE_CLEAN"]
        )
    ].copy()
    flagged.to_csv(outdir / "02_rejected_or_flagged.csv", index=False)

    summary_rows = []
    for pid, g in out.groupby("positive_sample_id"):
        sg = strict[strict["positive_sample_id"].eq(pid)]
        summary_rows.append({
            "positive_sample_id": pid,
            "input_selected_candidates": len(g),
            "confirmed_no_release": int(g["validation_class"].eq("CONFIRMED_NO_RELEASE").sum()),
            "clean_weak_negative": int(g["validation_class"].eq("TEMPORAL_WEAK_NEGATIVE_CLEAN").sum()),
            "nearby_l4_flag": int(g["validation_class"].eq("TEMPORAL_WEAK_NEGATIVE_NEARBY_L4").sum()),
            "known_positive_reject": int(g["validation_class"].eq("REJECT_KNOWN_POSITIVE").sum()),
            "other_reject": int(g["validation_class"].str.startswith("REJECT_").sum())
                - int(g["validation_class"].eq("REJECT_KNOWN_POSITIVE").sum()),
            "strict_download_ready": len(sg),
        })
    pd.DataFrame(summary_rows).to_csv(
        outdir / "03_validation_summary_by_positive.csv", index=False
    )

    counts = out["validation_class"].value_counts().to_dict()

    lines = [
        "# MethaneSAT temporal-negative inventory validation",
        "",
        "## Inputs",
        f"- Phase-A selected candidates: {len(cand)}",
        f"- Master inventory: {master_path}",
        f"- Master source format: {master_path.suffix.lower()}",
        f"- Master rows: {len(master)}",
        "",
        "## Validation rules",
        f"- Known positive/release conflict radius: {args.conflict_radius_m:.0f} m",
        f"- Known positive/release time window: ±{args.time_window_hours:.0f} h",
        f"- Nearby MethaneSAT L4 policy (10 km Phase-A flag): {args.nearby_l4_radius_policy}",
        f"- Minimum absolute time gap from source positive: {args.min_abs_days:.1f} days",
        "",
        "## Results",
    ]
    for k in sorted(counts):
        lines.append(f"- {k}: {counts[k]}")
    lines += [
        f"- Strict download-ready rows: {len(strict)}",
        f"- Positive sources with >=1 strict row: {strict['positive_sample_id'].nunique() if len(strict) else 0}",
        "",
        "## Interpretation",
        "- CONFIRMED_NO_RELEASE requires explicit no-release / zero-release evidence in the master inventory.",
        "- TEMPORAL_WEAK_NEGATIVE_CLEAN means no KNOWN positive/release record was found in the inventory window.",
        "- TEMPORAL_WEAK_NEGATIVE_CLEAN is NOT proof that physical methane emissions were zero.",
        "- Only 01_strict_temporal_negatives.csv should be handed to the Phase-B downloader.",
    ]

    (outdir / "SUMMARY_TEMPORAL_NEGATIVE_VALIDATION.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("=" * 78)
    print("METHANESAT TEMPORAL NEGATIVE VALIDATION")
    print("=" * 78)
    print("Selected input:", len(cand))
    print("Master rows:", len(master))
    print()
    print("Validation classes:")
    print(out["validation_class"].value_counts().to_string())
    print()
    print("Strict download-ready:", len(strict))
    print("Positive sources represented:",
          strict["positive_sample_id"].nunique() if len(strict) else 0)
    print()
    print("Output:", outdir)
    print("Upload:")
    for fn in [
        "SUMMARY_TEMPORAL_NEGATIVE_VALIDATION.md",
        "00_validated_all_selected.csv",
        "01_strict_temporal_negatives.csv",
        "02_rejected_or_flagged.csv",
        "03_validation_summary_by_positive.csv",
    ]:
        print(" ", outdir / fn)


if __name__ == "__main__":
    main()
