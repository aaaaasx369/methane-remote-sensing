#!/usr/bin/env python3
"""
Resolve Stanford 2025 Sentinel-2 controlled-release events to exact
Copernicus Data Space Ecosystem Sentinel-2 L2A products.

Input:
  03_stanford_s2_174_candidates.csv

Outputs:
  stanford_s2_scene_match/
    01_all_catalog_candidates.csv
    02_selected_scene_matches.csv
    03_unresolved_or_review.csv
    SUMMARY.txt

Catalogue search itself does NOT require a CDSE access token.
Downloading full products later DOES require authentication.
"""

import argparse
import csv
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

SENSING_RE = re.compile(r"^(S2[ABC])_MSIL2A_(\d{8}T\d{6})_")
MGRS_RE = re.compile(r"_T(\d{2}[A-Z]{3})_")
ORBIT_RE = re.compile(r"_R(\d{3})_")
BASELINE_RE = re.compile(r"_N(\d{4})_")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="03_stanford_s2_174_candidates.csv",
        help="Stanford Sentinel-2 candidate CSV",
    )
    ap.add_argument(
        "--outdir",
        default="stanford_s2_scene_match",
        help="Output directory",
    )
    ap.add_argument(
        "--window-min",
        type=int,
        default=30,
        help="Catalogue search window on either side of expected overpass time",
    )
    ap.add_argument(
        "--pass-min",
        type=float,
        default=10.0,
        help="Maximum absolute time difference for PASS",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Delay between catalogue requests",
    )
    return ap.parse_args()


def parse_event_time(row):
    s = (row.get("datetime_UTC") or "").strip()
    if s:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    d = (row.get("date") or "").strip()
    t = (row.get("time_UTC") or "").strip()
    if not d or not t:
        raise ValueError("Missing date/time")
    return datetime.fromisoformat(f"{d}T{t}+00:00")


def fmt_odata_time(dt):
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def product_sensing_time(name, content_date):
    m = SENSING_RE.search(name or "")
    if m:
        return datetime.strptime(m.group(2), "%Y%m%dT%H%M%S").replace(
            tzinfo=timezone.utc
        )

    if isinstance(content_date, dict):
        s = content_date.get("Start")
        if s:
            return datetime.fromisoformat(
                s.replace("Z", "+00:00")
            ).astimezone(timezone.utc)

    return None


def get_platform_prefix(code):
    code = (code or "").strip().upper()
    if code in {"S2A", "S2B", "S2C"}:
        return code
    raise ValueError(f"Unexpected Sentinel-2 code: {code!r}")


def query_catalog(session, row, window_min):
    event_time = parse_event_time(row)
    platform = get_platform_prefix(row["SatelliteCode"])
    lat = float(row["lat"])
    lon = float(row["lon"])

    start = event_time - timedelta(minutes=window_min)
    end = event_time + timedelta(minutes=window_min)

    odata_filter = (
        "Collection/Name eq 'SENTINEL-2' "
        f"and ContentDate/Start gt {fmt_odata_time(start)} "
        f"and ContentDate/Start lt {fmt_odata_time(end)} "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})') "
        "and contains(Name,'MSIL2A') "
        f"and startswith(Name,'{platform}_')"
    )

    params = {
        "$filter": odata_filter,
        "$select": "Id,Name,S3Path,ContentDate,GeoFootprint",
        "$orderby": "ContentDate/Start asc",
        "$top": "100",
    }

    last_error = None
    for attempt in range(1, 6):
        try:
            r = session.get(CATALOG_URL, params=params, timeout=90)

            if r.status_code == 429:
                wait = min(60, 2 ** attempt)
                print(f"  HTTP 429; sleeping {wait}s")
                time.sleep(wait)
                continue

            r.raise_for_status()
            payload = r.json()
            return payload.get("value", []), event_time, None

        except Exception as e:
            last_error = repr(e)
            if attempt < 5:
                wait = min(30, 2 ** attempt)
                print(f"  request error; retry in {wait}s: {last_error}")
                time.sleep(wait)

    return [], event_time, last_error


def candidate_record(event, product, event_time):
    name = product.get("Name", "")
    sensing = product_sensing_time(name, product.get("ContentDate"))

    if sensing is None:
        delta_sec = None
        sensing_iso = ""
    else:
        delta_sec = (sensing - event_time).total_seconds()
        sensing_iso = sensing.strftime("%Y-%m-%dT%H:%M:%SZ")

    mgrs = ""
    m = MGRS_RE.search(name)
    if m:
        mgrs = m.group(1)

    orbit = ""
    m = ORBIT_RE.search(name)
    if m:
        orbit = m.group(1)

    baseline = ""
    m = BASELINE_RE.search(name)
    if m:
        baseline = m.group(1)

    out = dict(event)
    out.update(
        {
            "cdse_product_id": product.get("Id", ""),
            "cdse_product_name": name,
            "cdse_s3_path": product.get("S3Path", ""),
            "product_sensing_utc": sensing_iso,
            "time_delta_seconds": delta_sec if delta_sec is not None else "",
            "abs_time_delta_seconds": abs(delta_sec) if delta_sec is not None else "",
            "mgrs_tile": mgrs,
            "relative_orbit": orbit,
            "processing_baseline": baseline,
        }
    )
    return out


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(input_path, newline="", encoding="utf-8-sig") as f:
        events = list(csv.DictReader(f))

    print("=" * 78)
    print("STANFORD 2025 SENTINEL-2 -> CDSE L2A SCENE RESOLUTION")
    print("=" * 78)
    print("Input events :", len(events))
    print("Search window:", f"±{args.window_min} min")
    print("PASS cutoff  :", f"≤{args.pass_min:g} min")
    print()

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "Stanford-MethaneFuse-S2-Resolver/1.0"}
    )

    all_candidates = []
    selected = []

    for i, event in enumerate(events, 1):
        rid = event.get("release_ID", "")
        code = event.get("SatelliteCode", "")

        print(f"[{i}/{len(events)}] {rid} | {code}")

        try:
            products, event_time, request_error = query_catalog(
                session, event, args.window_min
            )
        except Exception as e:
            products = []
            event_time = None
            request_error = repr(e)

        candidates = []
        if event_time is not None:
            candidates = [
                candidate_record(event, p, event_time)
                for p in products
            ]
            all_candidates.extend(candidates)

        with_delta = [
            x for x in candidates
            if x["abs_time_delta_seconds"] != ""
        ]
        with_delta.sort(
            key=lambda x: (
                float(x["abs_time_delta_seconds"]),
                x.get("cdse_product_name", ""),
            )
        )

        if with_delta:
            best = dict(with_delta[0])
            delta_min = float(best["abs_time_delta_seconds"]) / 60.0

            if delta_min <= args.pass_min:
                status = "PASS"
            elif delta_min <= args.window_min:
                status = "REVIEW_TIME_DELTA"
            else:
                status = "NO_L2A_MATCH"

            best.update(
                {
                    "match_status": status,
                    "catalog_candidate_count": len(candidates),
                    "best_abs_time_delta_min": delta_min,
                    "request_error": request_error or "",
                }
            )
        else:
            best = dict(event)
            best.update(
                {
                    "cdse_product_id": "",
                    "cdse_product_name": "",
                    "cdse_s3_path": "",
                    "product_sensing_utc": "",
                    "time_delta_seconds": "",
                    "abs_time_delta_seconds": "",
                    "mgrs_tile": "",
                    "relative_orbit": "",
                    "processing_baseline": "",
                    "match_status": (
                        "REQUEST_ERROR"
                        if request_error
                        else "NO_L2A_MATCH"
                    ),
                    "catalog_candidate_count": len(candidates),
                    "best_abs_time_delta_min": "",
                    "request_error": request_error or "",
                }
            )

        selected.append(best)

        print(
            "  candidates:",
            len(candidates),
            "| status:",
            best["match_status"],
            "| delta_min:",
            best["best_abs_time_delta_min"],
        )

        time.sleep(args.sleep)

    base_fields = list(events[0].keys()) if events else []
    extra_fields = [
        "cdse_product_id",
        "cdse_product_name",
        "cdse_s3_path",
        "product_sensing_utc",
        "time_delta_seconds",
        "abs_time_delta_seconds",
        "mgrs_tile",
        "relative_orbit",
        "processing_baseline",
    ]
    selected_extra = extra_fields + [
        "match_status",
        "catalog_candidate_count",
        "best_abs_time_delta_min",
        "request_error",
    ]

    all_fields = base_fields + [
        x for x in extra_fields if x not in base_fields
    ]
    selected_fields = base_fields + [
        x for x in selected_extra if x not in base_fields
    ]

    write_csv(
        outdir / "01_all_catalog_candidates.csv",
        all_candidates,
        all_fields,
    )
    write_csv(
        outdir / "02_selected_scene_matches.csv",
        selected,
        selected_fields,
    )

    unresolved = [
        x for x in selected
        if x["match_status"] != "PASS"
    ]
    write_csv(
        outdir / "03_unresolved_or_review.csv",
        unresolved,
        selected_fields,
    )

    status_counts = Counter(x["match_status"] for x in selected)
    label_counts = Counter(
        (x["match_status"], x.get("label", ""))
        for x in selected
    )

    with open(outdir / "SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write("Stanford 2025 Sentinel-2 CDSE L2A scene resolution\n")
        f.write("=" * 70 + "\n")
        f.write(f"Input events: {len(events)}\n")
        f.write(f"Search window: +/- {args.window_min} min\n")
        f.write(f"PASS cutoff: <= {args.pass_min:g} min\n\n")
        f.write("STATUS COUNTS\n")
        for k, v in sorted(status_counts.items()):
            f.write(f"{k:24s}: {v}\n")
        f.write("\nSTATUS x LABEL\n")
        for (status, label), n in sorted(label_counts.items()):
            f.write(f"{status:24s} label={label}: {n}\n")

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)
    for k, v in sorted(status_counts.items()):
        print(f"{k:24s}: {v}")
    print()
    print("Output:", outdir)


if __name__ == "__main__":
    main()
