#!/usr/bin/env python3
"""
Stanford 2025 all-public-archive availability audit:
  - Landsat 8/9 Collection 2 Level-2 via Microsoft Planetary Computer STAC
  - EMIT L2A reflectance + L2B CH4 enhancement/plume via NASA CMR
  - EnMAP HSI L2A via DLR EOC STAC

READ-ONLY: this script searches catalogues only. It does not download imagery.

Input:
  Stanford 620 QC-clean master CSV.

Outputs:
  stanford_public_archive_audit/
    01_landsat_event_summary.csv
    02_landsat_candidates.csv
    03_emit_event_summary.csv
    04_emit_candidates.csv
    05_enmap_event_summary.csv
    06_enmap_candidates.csv
    07_all_event_summary.csv
    SUMMARY.txt

Example:
  python audit_stanford_landsat_emit_enmap.py \
    --input stanford_master_tables/02_stanford_620_qc_ok.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

PC_STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
DLR_STAC_SEARCH = "https://geoservice.dlr.de/eoc/ogc/stac/v1/search"
CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"

EMIT_COLLECTIONS = [
    ("L2A_RFL", "EMITL2ARFL", "001"),
    ("L2B_CH4_ENH", "EMITL2BCH4ENH", "002"),
    ("L2B_CH4_PLM", "EMITL2BCH4PLM", "002"),
]

# Small spatial search box around the controlled-release location.
# 0.05 deg is ~5 km latitude; enough to catch a methane plume displaced downwind.
BBOX_HALF_DEG = 0.05


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="stanford_master_tables/02_stanford_620_qc_ok.csv",
    )
    ap.add_argument(
        "--outdir",
        default="stanford_public_archive_audit",
    )
    ap.add_argument(
        "--window-min",
        type=int,
        default=90,
        help="Primary catalogue time window on each side of Stanford time.",
    )
    ap.add_argument(
        "--exact-min",
        type=float,
        default=30.0,
        help="Time delta <= this is RESOLVED_EXACT.",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.15,
    )
    return ap.parse_args()


def parse_dt(row):
    s = str(row.get("datetime_UTC", "") or "").strip()
    if s:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    d = str(row.get("date", "") or "").strip()
    t = str(row.get("time_UTC", "") or "").strip()
    return datetime.fromisoformat(f"{d}T{t}+00:00").astimezone(timezone.utc)


def iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tiny_bbox(row):
    lon = float(row["lon"])
    lat = float(row["lat"])
    h = BBOX_HALF_DEG
    return [lon - h, lat - h, lon + h, lat + h]


def stac_dt(item):
    p = item.get("properties") or {}
    s = p.get("datetime") or p.get("start_datetime")
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def cmr_dt(entry):
    s = entry.get("time_start")
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def abs_delta_min(candidate_dt, expected_dt):
    if candidate_dt is None:
        return None
    return abs((candidate_dt - expected_dt).total_seconds()) / 60.0


def classify(delta_min, exact_min, window_min, found):
    if not found:
        return "NOT_FOUND"
    if delta_min is None:
        return "FOUND_TIME_UNKNOWN"
    if delta_min <= exact_min:
        return "RESOLVED_EXACT"
    if delta_min <= window_min:
        return "RESOLVED_NEARBY"
    return "DATE_ONLY_REVIEW"


def get_with_retry(session, url, *, params=None, timeout=90, tries=5):
    err = None
    for attempt in range(1, tries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                wait = min(60, 2 ** attempt)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r, None
        except Exception as e:
            err = repr(e)
            if attempt < tries:
                time.sleep(min(20, 2 ** attempt))
    return None, err


def post_with_retry(session, url, *, json_body=None, timeout=90, tries=5):
    err = None
    for attempt in range(1, tries + 1):
        try:
            r = session.post(url, json=json_body, timeout=timeout)
            if r.status_code == 429:
                wait = min(60, 2 ** attempt)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r, None
        except Exception as e:
            err = repr(e)
            if attempt < tries:
                time.sleep(min(20, 2 ** attempt))
    return None, err


def write_csv(path, rows):
    rows = list(rows)
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        if not keys:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def asset_summary(assets):
    if not isinstance(assets, dict):
        return "", ""
    names = []
    hrefs = []
    for k, v in assets.items():
        names.append(str(k))
        if isinstance(v, dict) and v.get("href"):
            hrefs.append(str(v["href"]))
    return "|".join(names), "|".join(hrefs)


def search_landsat(session, row, args):
    expected = parse_dt(row)
    bbox = tiny_bbox(row)
    platform_expected = {
        "LS8": "landsat-8",
        "LS9": "landsat-9",
    }.get(str(row.get("SatelliteCode", "")).strip(), "")

    def do_search(start, end):
        body = {
            "collections": ["landsat-c2-l2"],
            "bbox": bbox,
            "datetime": f"{iso(start)}/{iso(end)}",
            "limit": 100,
        }
        r, err = post_with_retry(session, PC_STAC_SEARCH, json_body=body)
        if r is None:
            return [], err
        return (r.json().get("features") or []), None

    start = expected - timedelta(minutes=args.window_min)
    end = expected + timedelta(minutes=args.window_min)
    feats, err = do_search(start, end)

    search_mode = "PRIMARY_WINDOW"
    if not feats and err is None:
        day0 = expected.replace(hour=0, minute=0, second=0, microsecond=0)
        feats, err = do_search(day0, day0 + timedelta(days=1))
        search_mode = "WHOLE_DAY_FALLBACK"

    candidates = []
    for item in feats:
        p = item.get("properties") or {}
        dt = stac_dt(item)
        platform = str(p.get("platform", "") or "").lower()
        names, hrefs = asset_summary(item.get("assets"))
        delta = abs_delta_min(dt, expected)
        candidates.append({
            **row,
            "archive": "PlanetaryComputer",
            "product_family": "Landsat_C2_L2",
            "candidate_id": item.get("id", ""),
            "candidate_datetime_utc": iso(dt) if dt else "",
            "abs_time_delta_min": delta if delta is not None else "",
            "candidate_platform": platform,
            "platform_expected": platform_expected,
            "platform_match": bool(platform_expected and platform == platform_expected),
            "eo_cloud_cover": p.get("eo:cloud_cover", ""),
            "landsat_scene_id": p.get("landsat:scene_id", ""),
            "landsat_product_id": p.get("landsat:product_id", ""),
            "collection_number": p.get("landsat:collection_number", ""),
            "collection_category": p.get("landsat:collection_category", ""),
            "asset_names": names,
            "asset_hrefs": hrefs,
            "search_mode": search_mode,
        })

    # Rank correct platform first, then time delta.
    def rank(c):
        pm = 0 if c["platform_match"] else 1
        try:
            d = float(c["abs_time_delta_min"])
        except Exception:
            d = 1e18
        return (pm, d, str(c["candidate_id"]))

    candidates.sort(key=rank)
    best = candidates[0] if candidates else None

    if best:
        delta = best["abs_time_delta_min"]
        try:
            delta = float(delta)
        except Exception:
            delta = None
        status = classify(delta, args.exact_min, args.window_min, True)
        if not best["platform_match"]:
            status = "PLATFORM_MISMATCH_REVIEW"
    else:
        status = "REQUEST_ERROR" if err else "NOT_FOUND"

    summary = {
        **row,
        "audit_sensor": "Landsat",
        "archive": "PlanetaryComputer",
        "availability_status": status,
        "catalog_candidate_count": len(candidates),
        "best_candidate_id": best["candidate_id"] if best else "",
        "best_candidate_datetime_utc": best["candidate_datetime_utc"] if best else "",
        "best_abs_time_delta_min": best["abs_time_delta_min"] if best else "",
        "best_platform": best["candidate_platform"] if best else "",
        "platform_match": best["platform_match"] if best else "",
        "best_cloud_cover": best["eo_cloud_cover"] if best else "",
        "best_asset_names": best["asset_names"] if best else "",
        "search_mode": search_mode,
        "request_error": err or "",
    }
    return summary, candidates


def cmr_data_links(entry):
    out = []
    for link in entry.get("links", []) or []:
        if link.get("inherited") is True:
            continue
        href = str(link.get("href", "") or "")
        rel = str(link.get("rel", "") or "").lower()
        title = str(link.get("title", "") or "")
        if not href:
            continue
        # Keep likely data/download links; omit obvious metadata browse links when possible.
        if (
            "data#" in rel
            or "s3#" in rel
            or href.lower().endswith((".nc", ".tif", ".tiff", ".h5", ".hdf5"))
            or "download" in title.lower()
        ):
            out.append(href)
    return out


def search_emit_collection(session, row, args, product_family, short_name, version):
    expected = parse_dt(row)
    bbox = tiny_bbox(row)

    def do_search(start, end):
        params = {
            "short_name": short_name,
            "version": version,
            "temporal": f"{iso(start)},{iso(end)}",
            "bounding_box": ",".join(str(x) for x in bbox),
            "page_size": 200,
        }
        r, err = get_with_retry(session, CMR_GRANULES, params=params)
        if r is None:
            return [], err
        return (((r.json().get("feed") or {}).get("entry")) or []), None

    start = expected - timedelta(minutes=args.window_min)
    end = expected + timedelta(minutes=args.window_min)
    entries, err = do_search(start, end)
    search_mode = "PRIMARY_WINDOW"

    if not entries and err is None:
        day0 = expected.replace(hour=0, minute=0, second=0, microsecond=0)
        entries, err = do_search(day0, day0 + timedelta(days=1))
        search_mode = "WHOLE_DAY_FALLBACK"

    candidates = []
    for e in entries:
        dt = cmr_dt(e)
        delta = abs_delta_min(dt, expected)
        links = cmr_data_links(e)
        candidates.append({
            **row,
            "archive": "NASA_CMR",
            "product_family": product_family,
            "short_name": short_name,
            "version": version,
            "candidate_id": e.get("id", ""),
            "producer_granule_id": e.get("producer_granule_id", ""),
            "candidate_datetime_utc": iso(dt) if dt else "",
            "candidate_time_end_utc": e.get("time_end", ""),
            "abs_time_delta_min": delta if delta is not None else "",
            "data_link_count": len(links),
            "data_links": "|".join(links),
            "search_mode": search_mode,
        })

    def rank(c):
        try:
            d = float(c["abs_time_delta_min"])
        except Exception:
            d = 1e18
        return (d, str(c["producer_granule_id"]))

    candidates.sort(key=rank)
    best = candidates[0] if candidates else None

    if best:
        try:
            delta = float(best["abs_time_delta_min"])
        except Exception:
            delta = None
        status = classify(delta, args.exact_min, args.window_min, True)
    else:
        status = "REQUEST_ERROR" if err else "NOT_FOUND"

    return {
        "status": status,
        "candidate_count": len(candidates),
        "best": best,
        "error": err or "",
        "search_mode": search_mode,
    }, candidates


def search_emit(session, row, args):
    all_candidates = []
    results = {}
    for family, short_name, version in EMIT_COLLECTIONS:
        result, candidates = search_emit_collection(
            session, row, args, family, short_name, version
        )
        results[family] = result
        all_candidates.extend(candidates)
        time.sleep(args.sleep)

    def val(family, field, default=""):
        r = results[family]
        if field in r:
            return r[field]
        b = r.get("best")
        return (b or {}).get(field, default)

    summary = {
        **row,
        "audit_sensor": "EMIT",
        "archive": "NASA_CMR",
        "availability_status": results["L2A_RFL"]["status"],
        "l2a_status": results["L2A_RFL"]["status"],
        "l2a_candidate_count": results["L2A_RFL"]["candidate_count"],
        "l2a_best_granule": val("L2A_RFL", "producer_granule_id"),
        "l2a_best_datetime_utc": val("L2A_RFL", "candidate_datetime_utc"),
        "l2a_best_abs_time_delta_min": val("L2A_RFL", "abs_time_delta_min"),
        "l2a_data_links": val("L2A_RFL", "data_links"),
        "ch4_enh_status": results["L2B_CH4_ENH"]["status"],
        "ch4_enh_candidate_count": results["L2B_CH4_ENH"]["candidate_count"],
        "ch4_enh_best_granule": val("L2B_CH4_ENH", "producer_granule_id"),
        "ch4_enh_best_datetime_utc": val("L2B_CH4_ENH", "candidate_datetime_utc"),
        "ch4_enh_best_abs_time_delta_min": val("L2B_CH4_ENH", "abs_time_delta_min"),
        "ch4_enh_data_links": val("L2B_CH4_ENH", "data_links"),
        "ch4_plm_status": results["L2B_CH4_PLM"]["status"],
        "ch4_plm_candidate_count": results["L2B_CH4_PLM"]["candidate_count"],
        "ch4_plm_best_granule": val("L2B_CH4_PLM", "producer_granule_id"),
        "ch4_plm_best_datetime_utc": val("L2B_CH4_PLM", "candidate_datetime_utc"),
        "ch4_plm_best_abs_time_delta_min": val("L2B_CH4_PLM", "abs_time_delta_min"),
        "ch4_plm_data_links": val("L2B_CH4_PLM", "data_links"),
        "request_error": " | ".join(
            x["error"] for x in results.values() if x.get("error")
        ),
    }
    return summary, all_candidates


def search_enmap(session, row, args):
    expected = parse_dt(row)
    bbox = tiny_bbox(row)

    def do_search(start, end):
        body = {
            "collections": ["ENMAP_HSI_L2A"],
            "bbox": bbox,
            "datetime": f"{iso(start)}/{iso(end)}",
            "limit": 100,
        }
        r, err = post_with_retry(session, DLR_STAC_SEARCH, json_body=body)
        if r is None:
            # Some STAC deployments may prefer GET.
            params = {
                "collections": "ENMAP_HSI_L2A",
                "bbox": ",".join(str(x) for x in bbox),
                "datetime": f"{iso(start)}/{iso(end)}",
                "limit": 100,
            }
            r, err2 = get_with_retry(session, DLR_STAC_SEARCH, params=params)
            if r is None:
                return [], (err2 or err)
        return (r.json().get("features") or []), None

    start = expected - timedelta(minutes=args.window_min)
    end = expected + timedelta(minutes=args.window_min)
    feats, err = do_search(start, end)
    search_mode = "PRIMARY_WINDOW"

    if not feats and err is None:
        day0 = expected.replace(hour=0, minute=0, second=0, microsecond=0)
        feats, err = do_search(day0, day0 + timedelta(days=1))
        search_mode = "WHOLE_DAY_FALLBACK"

    candidates = []
    for item in feats:
        p = item.get("properties") or {}
        dt = stac_dt(item)
        delta = abs_delta_min(dt, expected)
        names, hrefs = asset_summary(item.get("assets"))
        candidates.append({
            **row,
            "archive": "DLR_EOC_STAC",
            "product_family": "ENMAP_HSI_L2A",
            "candidate_id": item.get("id", ""),
            "candidate_datetime_utc": iso(dt) if dt else "",
            "abs_time_delta_min": delta if delta is not None else "",
            "enmap_overall_quality": p.get("enmap:overallQuality", ""),
            "eo_cloud_cover": p.get("eo:cloud_cover", ""),
            "asset_names": names,
            "asset_hrefs": hrefs,
            "search_mode": search_mode,
        })

    def rank(c):
        try:
            d = float(c["abs_time_delta_min"])
        except Exception:
            d = 1e18
        try:
            q = int(c["enmap_overall_quality"])
        except Exception:
            q = 99
        return (d, q, str(c["candidate_id"]))

    candidates.sort(key=rank)
    best = candidates[0] if candidates else None

    if best:
        try:
            delta = float(best["abs_time_delta_min"])
        except Exception:
            delta = None
        status = classify(delta, args.exact_min, args.window_min, True)
    else:
        status = "REQUEST_ERROR" if err else "NOT_FOUND"

    summary = {
        **row,
        "audit_sensor": "EnMAP",
        "archive": "DLR_EOC_STAC",
        "availability_status": status,
        "catalog_candidate_count": len(candidates),
        "best_candidate_id": best["candidate_id"] if best else "",
        "best_candidate_datetime_utc": best["candidate_datetime_utc"] if best else "",
        "best_abs_time_delta_min": best["abs_time_delta_min"] if best else "",
        "best_overall_quality": best["enmap_overall_quality"] if best else "",
        "best_cloud_cover": best["eo_cloud_cover"] if best else "",
        "best_asset_names": best["asset_names"] if best else "",
        "best_asset_hrefs": best["asset_hrefs"] if best else "",
        "search_mode": search_mode,
        "request_error": err or "",
    }
    return summary, candidates


def main():
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with input_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    landsat = [r for r in rows if str(r.get("SatellitePlotName", "")).strip() == "Landsat"]
    emit = [r for r in rows if str(r.get("SatellitePlotName", "")).strip() == "EMIT"]
    enmap = [r for r in rows if str(r.get("SatellitePlotName", "")).strip() == "EnMAP"]

    print("=" * 80)
    print("STANFORD PUBLIC ARCHIVE AVAILABILITY AUDIT")
    print("=" * 80)
    print("Landsat:", len(landsat))
    print("EMIT   :", len(emit))
    print("EnMAP  :", len(enmap))
    print("Total  :", len(landsat) + len(emit) + len(enmap))
    print()

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Stanford-Methane-Archive-Audit/1.0",
        "Accept": "application/json",
    })

    ls_summ, ls_cand = [], []
    for i, row in enumerate(landsat, 1):
        print(f"[Landsat {i}/{len(landsat)}] {row.get('release_ID')}")
        s, c = search_landsat(session, row, args)
        ls_summ.append(s)
        ls_cand.extend(c)
        print(" ", s["availability_status"], s["best_candidate_id"])
        time.sleep(args.sleep)

    em_summ, em_cand = [], []
    for i, row in enumerate(emit, 1):
        print(f"[EMIT {i}/{len(emit)}] {row.get('release_ID')}")
        s, c = search_emit(session, row, args)
        em_summ.append(s)
        em_cand.extend(c)
        print(
            "  L2A:", s["l2a_status"],
            "| ENH:", s["ch4_enh_status"],
            "| PLM:", s["ch4_plm_status"],
        )
        time.sleep(args.sleep)

    en_summ, en_cand = [], []
    for i, row in enumerate(enmap, 1):
        print(f"[EnMAP {i}/{len(enmap)}] {row.get('release_ID')}")
        s, c = search_enmap(session, row, args)
        en_summ.append(s)
        en_cand.extend(c)
        print(" ", s["availability_status"], s["best_candidate_id"])
        time.sleep(args.sleep)

    write_csv(outdir / "01_landsat_event_summary.csv", ls_summ)
    write_csv(outdir / "02_landsat_candidates.csv", ls_cand)
    write_csv(outdir / "03_emit_event_summary.csv", em_summ)
    write_csv(outdir / "04_emit_candidates.csv", em_cand)
    write_csv(outdir / "05_enmap_event_summary.csv", en_summ)
    write_csv(outdir / "06_enmap_candidates.csv", en_cand)

    all_summary = ls_summ + em_summ + en_summ
    write_csv(outdir / "07_all_event_summary.csv", all_summary)

    def status_counts(xs):
        return Counter(x.get("availability_status", "") for x in xs)

    with (outdir / "SUMMARY.txt").open("w", encoding="utf-8") as f:
        f.write("Stanford 2025 public-archive availability audit\n")
        f.write("=" * 72 + "\n")
        f.write(f"Landsat events: {len(landsat)}\n")
        f.write(f"EMIT events:    {len(emit)}\n")
        f.write(f"EnMAP events:   {len(enmap)}\n")
        f.write(f"Total events:   {len(all_summary)}\n\n")

        f.write("LANDSAT STATUS\n")
        for k, v in sorted(status_counts(ls_summ).items()):
            f.write(f"{k:28s}: {v}\n")

        f.write("\nEMIT L2A STATUS\n")
        for k, v in sorted(Counter(x.get("l2a_status", "") for x in em_summ).items()):
            f.write(f"{k:28s}: {v}\n")

        f.write("\nEMIT L2B CH4 ENH STATUS\n")
        for k, v in sorted(Counter(x.get("ch4_enh_status", "") for x in em_summ).items()):
            f.write(f"{k:28s}: {v}\n")

        f.write("\nEMIT L2B CH4 PLM STATUS\n")
        for k, v in sorted(Counter(x.get("ch4_plm_status", "") for x in em_summ).items()):
            f.write(f"{k:28s}: {v}\n")

        f.write("\nENMAP STATUS\n")
        for k, v in sorted(status_counts(en_summ).items()):
            f.write(f"{k:28s}: {v}\n")

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print("Output:", outdir)
    print()
    print((outdir / "SUMMARY.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
