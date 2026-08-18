#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Discover same-location, different-time MethaneSAT temporal negative candidates.

Positive source = existing MethaneSAT L4 point-source positive from the 222-row inventory.
Candidate negative = exact same latitude/longitude, different MethaneSAT L3 collection,
XCH4 crop QA pass, and no same-collection L4 point-source feature near the source.

IMPORTANT: absence of an L4 point feature is not automatically treated as confirmed
zero-emission ground truth because MethaneSAT does not provide all L3/L4 products for
all collection IDs. The script therefore separates STRONG_CANDIDATE and WEAK_CANDIDATE.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import ee
except ImportError as exc:
    raise SystemExit(
        "Missing earthengine-api. Install with: pip install earthengine-api\n"
        "Then authenticate if needed: earthengine authenticate"
    ) from exc

L3_ASSET = "projects/edf-methanesat-ee/assets/public-preview/L3concentration"
L4_POINT_ASSET = "projects/edf-methanesat-ee/assets/public-preview/L4point"
L4_AREA_V2_ASSET = "projects/edf-methanesat-ee/assets/public-preview/L4area_v2"


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", default="~/Downloads/MethaneSAT_222_inventory.csv")
    p.add_argument(
        "--out",
        default="/Volumes/engg-leung/dora lin/MethaneSAT_MethaneFuse/02_temporal_negative_search",
    )
    p.add_argument("--project", default="methane-release-gee")
    p.add_argument("--start", default="2024-05-22")
    p.add_argument("--end", default="2025-06-15")
    p.add_argument("--min-valid-fraction", type=float, default=0.50)
    p.add_argument("--crop-half-m", type=float, default=240.0)
    p.add_argument("--l4-exclusion-radius-m", type=float, default=2000.0)
    p.add_argument("--nearby-l4-radius-m", type=float, default=10000.0)
    p.add_argument("--max-negatives-per-positive", type=int, default=6)
    p.add_argument("--allow-other-targets", action="store_true")
    p.add_argument("--prefer-before", action="store_true")
    p.add_argument("--sleep", type=float, default=0.05)
    return p.parse_args()


def initialize_ee(project):
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Run: earthengine authenticate\n"
            f"Then retry with --project {project!r}\nOriginal error: {exc}"
        ) from exc


def note_value(notes, key):
    m = re.search(rf"\b{re.escape(key)}\s*=\s*([^;,\s]+)", str(notes), flags=re.I)
    return m.group(1).strip() if m else ""


def load_positives(path):
    df = pd.read_csv(path)
    cm = {str(c).strip().lower(): c for c in df.columns}
    need = ["latitude", "longitude", "label", "notes"]
    missing = [x for x in need if x not in cm]
    if missing:
        raise ValueError(f"Missing inventory columns: {missing}; columns={list(df.columns)}")

    out = pd.DataFrame()
    out["latitude"] = pd.to_numeric(df[cm["latitude"]], errors="coerce")
    out["longitude"] = pd.to_numeric(df[cm["longitude"]], errors="coerce")
    out["label"] = pd.to_numeric(df[cm["label"]], errors="coerce")
    out["notes"] = df[cm["notes"]].fillna("").astype(str)

    sid = cm.get("scene/observation id")
    site = cm.get("site")
    date = cm.get("date")
    ut = cm.get("utc time")
    rate = cm.get("release rate (kg/hr)")

    out["positive_sample_id"] = df[sid].fillna("").astype(str) if sid else [f"positive_{i:06d}" for i in range(len(df))]
    out["site"] = df[site].fillna("").astype(str) if site else ""
    out["positive_date"] = pd.to_datetime(df[date], errors="coerce") if date else pd.NaT
    out["positive_utc_time"] = df[ut].fillna("").astype(str) if ut else ""
    out["release_rate_kg_hr"] = pd.to_numeric(df[rate], errors="coerce") if rate else np.nan

    out["positive_collection_id"] = out.notes.map(lambda s: note_value(s, "collection_id"))
    out["positive_target_id"] = out.notes.map(lambda s: note_value(s, "target_id"))
    out["positive_plume_id"] = out.notes.map(lambda s: note_value(s, "plume_id"))
    out["positive_target_numeric"] = pd.to_numeric(
        out.positive_target_id.astype(str).str.replace(r"^[tT]", "", regex=True),
        errors="coerce",
    )

    out = out[
        out.label.eq(1)
        & out.latitude.notna()
        & out.longitude.notna()
        & out.positive_collection_id.astype(str).str.len().gt(0)
    ].copy()

    return out.drop_duplicates(
        subset=["positive_sample_id", "latitude", "longitude", "positive_collection_id"]
    ).reset_index(drop=True)


def pos_timestamp(row):
    if pd.isna(row.positive_date):
        return pd.NaT
    d = pd.Timestamp(row.positive_date).strftime("%Y-%m-%d")
    t = str(row.positive_utc_time).strip()
    ts = pd.to_datetime(f"{d} {t}" if t and t.lower() not in {"nan", "nat"} else d, utc=True, errors="coerce")
    return ts


def parse_candidate_time(info):
    s = info.get("time_coverage_start")
    if s:
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        if not pd.isna(ts):
            return ts
    ms = info.get("system:time_start")
    if ms is not None:
        try:
            return pd.to_datetime(int(ms), unit="ms", utc=True)
        except Exception:
            pass
    return pd.NaT


def images_at_point(l3, point, start, end):
    ic = l3.filterBounds(point).filterDate(start, end).sort("time_coverage_start")
    n = int(ic.size().getInfo())
    if n == 0:
        return []
    props = ["collection_id", "target_id", "time_coverage_start", "time_coverage_end", "system:index", "system:time_start"]
    return ic.toList(n).map(lambda x: ee.Image(x).toDictionary(props)).getInfo() or []


def l3_image(l3, cid):
    ic = l3.filter(ee.Filter.eq("collection_id", cid))
    if int(ic.size().getInfo()) < 1:
        raise RuntimeError(f"No L3 image for {cid}")
    return ee.Image(ic.first())


def valid_fraction(img, point, half_m):
    region = point.buffer(half_m).bounds()
    valid = img.select("XCH4").mask().unmask(0)
    d = valid.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=45,
        maxPixels=1_000_000,
    ).getInfo()
    v = d.get("XCH4")
    return float(v) if v is not None else 0.0


def l4_count_near(l4, cid, point, radius_m):
    return int(
        l4.filter(ee.Filter.eq("collection_id", cid))
        .filterBounds(point.buffer(radius_m))
        .size().getInfo()
    )


def l4_count_scene(l4, cid):
    return int(l4.filter(ee.Filter.eq("collection_id", cid)).size().getInfo())


def l4_area_count(l4area, cid):
    return int(l4area.filter(ee.Filter.eq("collection_id", cid)).size().getInfo())


def classify(same_collection, same_target, allow_other_targets, vf, min_vf, l4_near, area_count):
    if same_collection:
        return "REJECT", "same_positive_collection"
    if not allow_other_targets and not same_target:
        return "REJECT", "different_target_id"
    if vf < min_vf:
        return "REJECT", "xch4_valid_fraction_below_threshold"
    if l4_near > 0:
        return "REJECT", "l4_point_detected_near_same_location"
    if area_count > 0:
        return "STRONG_CANDIDATE", "qa_pass_l4area_present_no_l4point_near_site"
    return "WEAK_CANDIDATE", "qa_pass_no_l4point_near_site_l4_coverage_uncertain"


def select_strong(df, max_per, prefer_before):
    x = df[df.candidate_class.eq("STRONG_CANDIDATE")].copy()
    if x.empty:
        return x
    x["abs_days_from_positive"] = pd.to_numeric(x.days_from_positive, errors="coerce").abs()
    days = pd.to_numeric(x.days_from_positive, errors="coerce")
    x["before_rank"] = np.where(days < 0, 0 if prefer_before else 1, 1 if prefer_before else 0)
    x = x.sort_values(
        ["positive_sample_id", "abs_days_from_positive", "before_rank", "nearby_l4_count_10km", "xch4_valid_fraction"],
        ascending=[True, True, True, True, False],
    )
    x["selection_rank"] = x.groupby("positive_sample_id").cumcount() + 1
    x = x[x.selection_rank <= max_per].copy()
    x["selected_temporal_negative"] = True
    x["final_label_for_model"] = 0
    x["ground_truth_strength"] = "same-location temporal weak-negative; L4-area product present; no L4 point near site"
    return x


def main():
    args = args_parser()
    inv = Path(args.inventory).expanduser().resolve()
    outdir = Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    positives = load_positives(inv)
    positives.to_csv(outdir / "00_positive_sources.csv", index=False)

    print("=" * 78)
    print("METHANESAT SAME-SITE TEMPORAL NEGATIVE SEARCH")
    print("=" * 78)
    print("Positive rows:", len(positives))
    print("Unique positive collections:", positives.positive_collection_id.nunique())
    print("Unique target IDs:", positives.positive_target_id.nunique())
    print("Output:", outdir)

    initialize_ee(args.project)
    l3 = ee.ImageCollection(L3_ASSET)
    l4 = ee.FeatureCollection(L4_POINT_ASSET)
    l4area = ee.ImageCollection(L4_AREA_V2_ASSET)

    rows = []
    for i, pos in positives.iterrows():
        pid = str(pos.positive_sample_id)
        lat, lon = float(pos.latitude), float(pos.longitude)
        pos_cid = str(pos.positive_collection_id)
        pos_tid = pos.positive_target_numeric
        ptime = pos_timestamp(pos)
        point = ee.Geometry.Point([lon, lat])
        print(f"[{i+1}/{len(positives)}] {pid} target={pos.positive_target_id} collection={pos_cid}")

        try:
            infos = images_at_point(l3, point, args.start, args.end)
        except Exception as exc:
            rows.append({**pos.to_dict(), "candidate_collection_id": "", "candidate_class": "ERROR", "candidate_reason": f"L3_query_failed: {type(exc).__name__}: {exc}"})
            continue

        if not infos:
            rows.append({**pos.to_dict(), "candidate_collection_id": "", "candidate_class": "NO_L3_HISTORY", "candidate_reason": "no_l3_image_covering_exact_coordinate"})
            continue

        seen = set()
        for info in infos:
            cid = str(info.get("collection_id") or "").strip()
            if not cid or cid in seen:
                continue
            seen.add(cid)

            ctid = pd.to_numeric(info.get("target_id"), errors="coerce")
            same_target = bool(np.isfinite(ctid) and np.isfinite(pos_tid) and int(ctid) == int(pos_tid))
            ctime = parse_candidate_time(info)
            days = np.nan if pd.isna(ptime) or pd.isna(ctime) else (ctime - ptime).total_seconds() / 86400.0
            same_collection = cid == pos_cid

            rec = {
                **pos.to_dict(),
                "candidate_collection_id": cid,
                "candidate_target_id": int(ctid) if np.isfinite(ctid) else np.nan,
                "candidate_system_index": info.get("system:index", ""),
                "candidate_time_start": ctime.isoformat() if not pd.isna(ctime) else "",
                "candidate_time_end": info.get("time_coverage_end", ""),
                "days_from_positive": days,
                "same_target_id": same_target,
                "same_positive_collection": same_collection,
                "exact_same_latitude": lat,
                "exact_same_longitude": lon,
                "crop_size_m": 2 * args.crop_half_m,
                "xch4_valid_fraction": np.nan,
                "missing_fraction": np.nan,
                "l4_point_count_near_site": np.nan,
                "nearby_l4_count_10km": np.nan,
                "l4_point_count_scene": np.nan,
                "l4_area_v2_product_count": np.nan,
                "candidate_class": "",
                "candidate_reason": "",
            }

            try:
                img = l3_image(l3, cid)
                vf = valid_fraction(img, point, args.crop_half_m)
                near = l4_count_near(l4, cid, point, args.l4_exclusion_radius_m)
                nearby = l4_count_near(l4, cid, point, args.nearby_l4_radius_m)
                scene_n = l4_count_scene(l4, cid)
                area_n = l4_area_count(l4area, cid)
                cls, reason = classify(same_collection, same_target, args.allow_other_targets, vf, args.min_valid_fraction, near, area_n)
                rec.update({
                    "xch4_valid_fraction": vf,
                    "missing_fraction": 1.0 - vf,
                    "l4_point_count_near_site": near,
                    "nearby_l4_count_10km": nearby,
                    "l4_point_count_scene": scene_n,
                    "l4_area_v2_product_count": area_n,
                    "candidate_class": cls,
                    "candidate_reason": reason,
                })
            except Exception as exc:
                rec["candidate_class"] = "ERROR"
                rec["candidate_reason"] = f"{type(exc).__name__}: {exc}"
            rows.append(rec)
            time.sleep(args.sleep)

    allc = pd.DataFrame(rows)
    allc.to_csv(outdir / "01_all_same_site_l3_candidates.csv", index=False)

    screened = allc[allc.candidate_class.isin(["STRONG_CANDIDATE", "WEAK_CANDIDATE"])].copy()
    if not screened.empty:
        screened["abs_days_from_positive"] = pd.to_numeric(screened.days_from_positive, errors="coerce").abs()
        screened = screened.sort_values(["positive_sample_id", "candidate_class", "abs_days_from_positive", "xch4_valid_fraction"], ascending=[True, True, True, False])
    screened.to_csv(outdir / "02_screened_temporal_candidates.csv", index=False)

    selected = select_strong(allc, args.max_negatives_per_positive, args.prefer_before)
    selected.to_csv(outdir / "03_selected_temporal_negatives.csv", index=False)

    ps = []
    for pid, g in allc.groupby("positive_sample_id", dropna=False):
        ps.append({
            "positive_sample_id": pid,
            "positive_collection_id": g.positive_collection_id.iloc[0],
            "positive_target_id": g.positive_target_id.iloc[0],
            "latitude": g.latitude.iloc[0],
            "longitude": g.longitude.iloc[0],
            "l3_collections_covering_same_coordinate": int(g.candidate_collection_id.astype(str).str.len().gt(0).sum()),
            "strong_candidates": int(g.candidate_class.eq("STRONG_CANDIDATE").sum()),
            "weak_candidates": int(g.candidate_class.eq("WEAK_CANDIDATE").sum()),
            "rejected": int(g.candidate_class.eq("REJECT").sum()),
            "errors": int(g.candidate_class.eq("ERROR").sum()),
            "selected_negatives": int(selected.positive_sample_id.eq(pid).sum()) if not selected.empty else 0,
        })
    pd.DataFrame(ps).to_csv(outdir / "04_positive_level_summary.csv", index=False)

    cov = []
    vr = allc[allc.candidate_collection_id.astype(str).str.len().gt(0)].copy()
    for cid, g in vr.groupby("candidate_collection_id"):
        cov.append({
            "candidate_collection_id": cid,
            "l4_area_v2_product_present": bool(pd.to_numeric(g.l4_area_v2_product_count, errors="coerce").fillna(0).max() > 0),
            "max_l4_point_count_scene": int(pd.to_numeric(g.l4_point_count_scene, errors="coerce").fillna(0).max()),
            "n_positive_locations_intersected": g.positive_sample_id.nunique(),
        })
    pd.DataFrame(cov).to_csv(outdir / "05_collection_coverage_summary.csv", index=False)

    counts = allc.candidate_class.value_counts(dropna=False).to_dict()
    n_strong_pos = allc[allc.candidate_class.eq("STRONG_CANDIDATE")].positive_sample_id.nunique()
    n_selected_pos = selected.positive_sample_id.nunique() if not selected.empty else 0

    lines = [
        "# MethaneSAT same-site temporal-negative search",
        "",
        "## Definition",
        "- Exact same latitude/longitude as the positive source; different L3 acquisition.",
        f"- Fixed crop: {2*args.crop_half_m:.0f} m × {2*args.crop_half_m:.0f} m.",
        f"- XCH4 QA: valid fraction >= {args.min_valid_fraction:.2f} (missing <= {1-args.min_valid_fraction:.2f}).",
        f"- L4 point exclusion radius: {args.l4_exclusion_radius_m/1000:.1f} km.",
        f"- Nearby L4 flag radius: {args.nearby_l4_radius_m/1000:.1f} km.",
        "- Same target_id required by default." if not args.allow_other_targets else "- Other target_ids allowed if they cover the exact same coordinate.",
        "",
        "## Actual positive inventory",
        f"- Positive rows: {len(positives)}",
        f"- Unique positive collections: {positives.positive_collection_id.nunique()}",
        f"- Unique positive target IDs: {positives.positive_target_id.nunique()}",
        "",
        "## Candidate results",
    ]
    for k in sorted(counts):
        lines.append(f"- {k}: {counts[k]}")
    lines += [
        f"- Positive sources with >=1 STRONG_CANDIDATE: {n_strong_pos}",
        f"- Positive sources receiving selected negatives: {n_selected_pos}",
        f"- Selected temporal negatives: {len(selected)} (max {args.max_negatives_per_positive} per positive)",
        "",
        "## Label policy",
        "- STRONG_CANDIDATE is still a temporal weak-negative, not confirmed zero-emission ground truth.",
        "- WEAK_CANDIDATE is never promoted automatically because missing L4 features may reflect missing L4 product coverage.",
        "- REJECT includes same-positive collection, low XCH4 QA, different target_id, or an L4 point detection near the exact same source location.",
        "",
        "## Next phase",
        "- Review 03_selected_temporal_negatives.csv, then download the exact 480 m XCH4 crops and build paired positive/temporal-negative model inputs.",
    ]
    (outdir / "SUMMARY_TEMPORAL_NEGATIVES.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nDONE:", outdir)
    print("Candidate classes:")
    print(pd.Series(counts).sort_index().to_string())
    print("Positive sources with strong candidates:", n_strong_pos)
    print("Selected temporal negatives:", len(selected))
    print("\nUpload:")
    for fn in [
        "SUMMARY_TEMPORAL_NEGATIVES.md",
        "00_positive_sources.csv",
        "01_all_same_site_l3_candidates.csv",
        "02_screened_temporal_candidates.csv",
        "03_selected_temporal_negatives.csv",
        "04_positive_level_summary.csv",
        "05_collection_coverage_summary.csv",
    ]:
        print(" ", outdir / fn)


if __name__ == "__main__":
    main()
