#!/usr/bin/env python3
"""
retry_enmap_primary259_errors_v1.py

Retry ONLY the unresolved ERROR rows from the first 762-scene EnMAP
availability audit.

The prior ERROR rows were overwhelmingly STAC HTTP 502 / read timeouts, so
they are unresolved, not unavailable.

Strategy:
1) DIRECT-ITEM FIRST:
     /collections/ENMAP_HSI_L2A/items/<original_scene_id>
   This avoids the heavier time-window search whenever possible.
2) Only if direct-item cannot resolve a usable exact product, fall back to a
   strict time-window search.
3) Strict replacement rule:
     datatake ID AND tile ID must match.
4) Probe metadata asset:
     original STAC href first,
     then repaired /<SCENE_ID>/ path if original returns 404.
5) Read only byte 0 via HTTP Range. No imagery is downloaded.
6) Slow cadence + long exponential backoff for 502/timeouts.
7) Checkpoint every 10 scenes.
8) Merge retry results back into the original 762-scene audit.

Default input:
  ~/methane_release_project/enmap_primary762_availability_v1/
      01_primary762_asset_availability.csv

Outputs:
  ~/methane_release_project/enmap_primary762_availability_retry_v1/
      01_retry259_results.csv
      02_primary762_asset_availability_updated.csv
      03_downloadable_scenes_updated.csv
      04_unavailable_or_review_updated.csv
      retry259_summary.txt
"""

from __future__ import annotations

import argparse
import getpass
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

import pandas as pd
import urllib3


STAC_ROOT = "https://geoservice.dlr.de/eoc/ogc/stac/v1/"
COLLECTION = "ENMAP_HSI_L2A"
STAC_ITEMS = f"{STAC_ROOT}collections/{COLLECTION}/items"

USERNAME = "doraaa"
USER_AGENT = "UAlberta-EnMAP-Primary259-Retry/1.0"

DEFAULT_INPUT = (
    Path.home()
    / "methane_release_project"
    / "enmap_primary762_availability_v1"
    / "01_primary762_asset_availability.csv"
)

DEFAULT_OUT = (
    Path.home()
    / "methane_release_project"
    / "enmap_primary762_availability_retry_v1"
)

SCENE_RE = re.compile(
    r"DT(?P<datatake>\d+)_"
    r"(?P<acq>\d{8}T\d{6}Z)_"
    r"(?P<tile>\d{3})_"
    r"V(?P<version>\d+)_"
    r"(?P<proc>\d{8}T\d{6}Z)"
)


def parse_scene(scene_id: str) -> Dict[str, Optional[str]]:
    m = SCENE_RE.search(str(scene_id))
    if not m:
        return {
            "datatake": None, "acq": None, "tile": None,
            "version": None, "proc": None,
        }
    return m.groupdict()


def normalize_digits(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return None
    return digits.lstrip("0") or "0"


def normalize_tile(value: Any) -> Optional[str]:
    d = normalize_digits(value)
    return str(int(d)) if d is not None else None


def feature_datatake(feat: Dict[str, Any]) -> Optional[str]:
    p = feat.get("properties", {}) or {}
    v = (
        p.get("enmap:datatakeID")
        or p.get("enmap:datatakeId")
        or p.get("enmap:datatake_id")
        or p.get("datatakeID")
    )
    if v is not None:
        return normalize_digits(v)
    return normalize_digits(
        parse_scene(str(feat.get("id") or "")).get("datatake")
    )


def feature_tile(feat: Dict[str, Any]) -> Optional[str]:
    p = feat.get("properties", {}) or {}
    v = p.get("enmap:tileID")
    if v is not None:
        return normalize_tile(v)
    return normalize_tile(
        parse_scene(str(feat.get("id") or "")).get("tile")
    )


def exact_same_acquisition(
    feat: Dict[str, Any],
    original_scene_id: str,
) -> bool:
    p = parse_scene(original_scene_id)
    wanted_dt = normalize_digits(p.get("datatake"))
    wanted_tile = normalize_tile(p.get("tile"))
    return (
        wanted_dt is not None
        and wanted_tile is not None
        and feature_datatake(feat) == wanted_dt
        and feature_tile(feat) == wanted_tile
    )


def candidate_sort_key(feat: Dict[str, Any]):
    sid = str(feat.get("id") or "")
    p = parse_scene(sid)
    props = feat.get("properties", {}) or {}
    return (
        str(p.get("proc") or ""),
        str(props.get("updated") or ""),
        str(props.get("created") or ""),
        sid,
    )


def metadata_href(feat: Dict[str, Any]) -> Optional[str]:
    assets = feat.get("assets", {}) or {}
    for key in ("metadata", "METADATA"):
        asset = assets.get(key)
        if asset and asset.get("href"):
            return str(asset["href"])
    for key, asset in assets.items():
        text = " ".join([
            str(key),
            str(asset.get("title") or ""),
            str(asset.get("href") or ""),
        ]).upper()
        if "METADATA" in text and asset.get("href"):
            return str(asset["href"])
    return None


def insert_scene_directory(url: str, scene_id: str) -> str:
    parts = urlsplit(url)
    pieces = parts.path.rstrip("/").split("/")
    if not pieces:
        return url
    filename = pieces[-1]
    if len(pieces) >= 2 and pieces[-2] == scene_id:
        return url
    new_path = "/".join(pieces[:-1] + [scene_id, filename])
    if parts.path.startswith("/") and not new_path.startswith("/"):
        new_path = "/" + new_path
    return urlunsplit(
        (parts.scheme, parts.netloc, new_path, parts.query, parts.fragment)
    )


def request_json(
    http: urllib3.PoolManager,
    url: str,
    retries: int,
    base_sleep: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[int]]:
    last_error = None
    last_status = None

    for attempt in range(1, retries + 1):
        try:
            r = http.request(
                "GET",
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/geo+json, application/json",
                },
                preload_content=True,
                redirect=True,
            )
            status = int(r.status)
            last_status = status

            if status == 200:
                try:
                    return json.loads(r.data.decode("utf-8")), None, status
                except Exception as e:
                    last_error = f"JSON decode: {type(e).__name__}: {e}"
            elif status in (404, 401, 403):
                return None, f"HTTP {status}", status
            else:
                last_error = f"HTTP {status}"

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < retries:
            wait = min(90.0, base_sleep * (2 ** (attempt - 1)))
            print(
                f"    JSON retry {attempt}/{retries}: "
                f"{last_error}; sleep {wait:.1f}s"
            )
            time.sleep(wait)

    return None, last_error, last_status


def probe_asset(
    http: urllib3.PoolManager,
    url: str,
    password: str,
    retries: int = 5,
    base_sleep: float = 2.0,
) -> Dict[str, Any]:
    headers = urllib3.make_headers(
        basic_auth=f"{USERNAME}:{password}"
    )
    headers["User-Agent"] = USER_AGENT
    headers["Range"] = "bytes=0-0"

    last_error = None
    last_status = None

    for attempt in range(1, retries + 1):
        try:
            r = http.request(
                "GET",
                url,
                headers=headers,
                preload_content=False,
                redirect=False,
            )
            status = int(r.status)
            last_status = status
            ctype = r.headers.get("Content-Type")
            crange = r.headers.get("Content-Range")
            location = r.headers.get("Location")
            _ = r.read(1024)
            r.release_conn()

            if status in (200, 206, 404, 401, 403):
                return {
                    "http_status": status,
                    "content_type": ctype,
                    "content_range": crange,
                    "location": location,
                    "error": None,
                }

            last_error = f"HTTP {status}"

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < retries:
            wait = min(60.0, base_sleep * (2 ** (attempt - 1)))
            print(
                f"    probe retry {attempt}/{retries}: "
                f"{last_error}; sleep {wait:.1f}s"
            )
            time.sleep(wait)

    return {
        "http_status": last_status,
        "content_type": None,
        "content_range": None,
        "location": None,
        "error": last_error,
    }


def resolve_direct_item_first(
    http: urllib3.PoolManager,
    original_scene_id: str,
    json_retries: int,
    base_sleep: float,
):
    url = (
        STAC_ITEMS
        + "/"
        + quote(original_scene_id, safe="")
        + "?f=json"
    )

    doc, err, status = request_json(
        http, url, retries=json_retries, base_sleep=base_sleep
    )

    if doc is not None:
        if exact_same_acquisition(doc, original_scene_id):
            return doc, "DIRECT_ITEM"
        return None, "DIRECT_ITEM_NOT_EXACT"

    return None, f"DIRECT_ITEM_FAILED:{err}"


def resolve_exact_time_search(
    http: urllib3.PoolManager,
    original_scene_id: str,
    json_retries: int,
    base_sleep: float,
):
    parsed = parse_scene(original_scene_id)
    wanted_dt = normalize_digits(parsed.get("datatake"))
    wanted_tile = normalize_tile(parsed.get("tile"))
    acq = parsed.get("acq")

    if not wanted_dt or not wanted_tile or not acq:
        return None, "UNPARSEABLE_ORIGINAL_ID"

    t = pd.to_datetime(
        acq,
        format="%Y%m%dT%H%M%SZ",
        errors="coerce",
        utc=True,
    )
    if pd.isna(t):
        return None, "BAD_ACQUISITION_TIME"

    t0 = t - pd.Timedelta(minutes=20)
    t1 = t + pd.Timedelta(minutes=20)

    params = urlencode({
        "datetime": (
            f"{t0.isoformat().replace('+00:00', 'Z')}/"
            f"{t1.isoformat().replace('+00:00', 'Z')}"
        ),
        "limit": 100,
        "f": "json",
    })

    doc, err, status = request_json(
        http,
        f"{STAC_ITEMS}?{params}",
        retries=json_retries,
        base_sleep=base_sleep,
    )

    if doc is None:
        return None, f"TIME_SEARCH_FAILED:{err}"

    feats = doc.get("features", []) or []
    exact = [
        feat
        for feat in feats
        if feature_datatake(feat) == wanted_dt
        and feature_tile(feat) == wanted_tile
    ]

    if not exact:
        return None, "NO_CURRENT_EXACT_PRODUCT"

    exact = sorted(exact, key=candidate_sort_key, reverse=True)
    return exact[0], "TIME_SEARCH_EXACT"


def resolve_current_exact(
    http: urllib3.PoolManager,
    original_scene_id: str,
    json_retries: int,
    base_sleep: float,
):
    feat, method = resolve_direct_item_first(
        http, original_scene_id, json_retries, base_sleep
    )
    if feat is not None:
        return feat, method

    feat2, method2 = resolve_exact_time_search(
        http, original_scene_id, json_retries, base_sleep
    )
    return feat2, f"{method} -> {method2}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=1.5)
    ap.add_argument("--json-retries", type=int, default=8)
    ap.add_argument("--base-backoff", type=float, default=3.0)
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    original_full = pd.read_csv(input_path, low_memory=False)
    retry_df = original_full[
        original_full["status"].astype(str).eq("ERROR")
    ].copy()

    if args.limit > 0:
        retry_df = retry_df.head(args.limit).copy()

    print("=" * 92)
    print("ENMAP PRIMARY ERROR RETRY — DIRECT ITEM FIRST")
    print("=" * 92)
    print(f"Original full audit rows : {len(original_full)}")
    print(
        f"ERROR rows in full audit : "
        f"{int((original_full['status'] == 'ERROR').sum())}"
    )
    print(f"Rows retried this run    : {len(retry_df)}")
    print(f"Username                 : {USERNAME}")
    print(f"Per-scene delay          : {args.sleep:.1f} s")
    print(f"JSON retries             : {args.json_retries}")
    print("No imagery will be downloaded.")
    print()

    if retry_df.empty:
        raise SystemExit("No ERROR rows to retry.")

    password = getpass.getpass(
        f"IPS password for {USERNAME}: "
    )
    if not password:
        raise SystemExit("Empty password; aborting.")

    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=45.0, read=120.0),
        retries=False,
        num_pools=4,
    )

    results = []

    for pos, (_, r) in enumerate(retry_df.iterrows(), start=1):
        original_scene_id = str(r["original_scene_id"]).strip()

        print()
        print(f"[{pos}/{len(retry_df)}] {original_scene_id}")

        base = r.to_dict()

        try:
            current, resolver_method = resolve_current_exact(
                http,
                original_scene_id,
                json_retries=args.json_retries,
                base_sleep=args.base_backoff,
            )

            if current is None:
                status = (
                    "NO_CURRENT_EXACT_PRODUCT"
                    if "NO_CURRENT_EXACT_PRODUCT" in resolver_method
                    else "ERROR"
                )
                print(f"  resolver: {resolver_method}")
                print(f"  STATUS  : {status}")
                results.append({
                    **base,
                    "current_scene_id": None,
                    "metadata_url_original": None,
                    "metadata_url_repaired": None,
                    "selected_metadata_url": None,
                    "path_style": None,
                    "original_http": None,
                    "repaired_http": None,
                    "status": status,
                    "error": resolver_method,
                    "retry_resolver_method": resolver_method,
                })
            else:
                current_id = str(current.get("id") or "")
                href = metadata_href(current)

                print(f"  resolver: {resolver_method}")
                print(f"  current : {current_id}")

                if not href:
                    status = "ERROR"
                    results.append({
                        **base,
                        "current_scene_id": current_id,
                        "metadata_url_original": None,
                        "metadata_url_repaired": None,
                        "selected_metadata_url": None,
                        "path_style": None,
                        "original_http": None,
                        "repaired_http": None,
                        "status": status,
                        "error": "metadata href missing",
                        "retry_resolver_method": resolver_method,
                    })
                    print("  STATUS  : ERROR — metadata href missing")
                else:
                    op = probe_asset(http, href, password)
                    original_http = op["http_status"]
                    repaired = insert_scene_directory(href, current_id)
                    repaired_http = None
                    selected = None
                    path_style = None

                    if original_http in (200, 206):
                        status = "ORIGINAL_WORKS"
                        selected = href
                        path_style = "ORIGINAL"
                        print(
                            f"  ORIGINAL HTTP {original_http} -> WORKS"
                        )
                    elif original_http == 404:
                        rp = probe_asset(http, repaired, password)
                        repaired_http = rp["http_status"]
                        if repaired_http in (200, 206):
                            status = "REPAIRED_PATH_WORKS"
                            selected = repaired
                            path_style = "SCENE_DIRECTORY_INSERTED"
                            print(
                                f"  ORIGINAL 404 -> REPAIRED "
                                f"HTTP {repaired_http} -> WORKS"
                            )
                        elif repaired_http == 404:
                            status = "STILL_404"
                            print(
                                "  ORIGINAL 404 -> REPAIRED 404"
                            )
                        elif repaired_http in (401, 403):
                            status = "AUTH_FAILURE"
                            print(
                                f"  REPAIRED HTTP {repaired_http} "
                                "-> AUTH FAILURE"
                            )
                        else:
                            status = "HTTP_OTHER"
                            print(
                                f"  REPAIRED HTTP {repaired_http}"
                            )
                    elif original_http in (401, 403):
                        status = "AUTH_FAILURE"
                        print(
                            f"  ORIGINAL HTTP {original_http} "
                            "-> AUTH FAILURE"
                        )
                    elif original_http is None:
                        status = "ERROR"
                        print(
                            f"  probe error: {op.get('error')}"
                        )
                    else:
                        status = "HTTP_OTHER"
                        print(f"  ORIGINAL HTTP {original_http}")

                    results.append({
                        **base,
                        "current_scene_id": current_id,
                        "metadata_url_original": href,
                        "metadata_url_repaired": repaired,
                        "selected_metadata_url": selected,
                        "path_style": path_style,
                        "original_http": original_http,
                        "repaired_http": repaired_http,
                        "status": status,
                        "error": op.get("error"),
                        "retry_resolver_method": resolver_method,
                    })

        except Exception as e:
            print(
                f"  STATUS: ERROR — {type(e).__name__}: {e}"
            )
            results.append({
                **base,
                "status": "ERROR",
                "error": f"{type(e).__name__}: {e}",
                "retry_resolver_method": "UNHANDLED_EXCEPTION",
            })

        if pos % 10 == 0:
            checkpoint = out_dir / "retry259_checkpoint.csv"
            pd.DataFrame(results).to_csv(checkpoint, index=False)
            print(f"  checkpoint -> {checkpoint}")

        if pos < len(retry_df):
            time.sleep(args.sleep)

    retry_out = pd.DataFrame(results)
    retry_path = out_dir / "01_retry259_results.csv"
    retry_out.to_csv(retry_path, index=False)

    updated = original_full.copy().set_index(
        "original_scene_id", drop=False
    )
    replacement = retry_out.set_index(
        "original_scene_id", drop=False
    )

    for c in replacement.columns:
        if c not in updated.columns:
            updated[c] = pd.NA

    for scene_id, new_row in replacement.iterrows():
        for c in replacement.columns:
            updated.at[scene_id, c] = new_row[c]

    updated = updated.reset_index(drop=True)

    updated_path = (
        out_dir / "02_primary762_asset_availability_updated.csv"
    )
    updated.to_csv(updated_path, index=False)

    ok_statuses = {
        "ORIGINAL_WORKS",
        "REPAIRED_PATH_WORKS",
    }

    downloadable = updated[
        updated["status"].isin(ok_statuses)
    ].copy()
    downloadable_path = (
        out_dir / "03_downloadable_scenes_updated.csv"
    )
    downloadable.to_csv(downloadable_path, index=False)

    review = updated[
        ~updated["status"].isin(ok_statuses)
    ].copy()
    review_path = (
        out_dir / "04_unavailable_or_review_updated.csv"
    )
    review.to_csv(review_path, index=False)

    retry_counts = retry_out["status"].value_counts(dropna=False)
    final_counts = updated["status"].value_counts(dropna=False)

    retry_success = int(
        retry_out["status"].isin(ok_statuses).sum()
    )
    total_downloadable = len(downloadable)

    lines = [
        "ENMAP PRIMARY 259-ERROR RETRY SUMMARY",
        "=" * 92,
        f"Original full audit rows        : {len(original_full)}",
        f"Rows retried this run           : {len(retry_out)}",
        f"Retry rows recovered/downloadable: {retry_success}",
        "",
        "RETRY STATUS COUNTS",
    ]

    for k, v in retry_counts.items():
        lines.append(f"{str(k):36s} {int(v)}")

    lines += ["", "UPDATED FULL-AUDIT STATUS COUNTS"]

    for k, v in final_counts.items():
        lines.append(f"{str(k):36s} {int(v)}")

    lines += [
        "",
        f"UPDATED confirmed downloadable : {total_downloadable}",
        f"UPDATED unresolved/review      : {len(updated) - total_downloadable}",
        "",
        "IMPORTANT",
        "- This run retries only prior ERROR rows.",
        "- Existing successful rows are untouched.",
        "- Existing STILL_404 rows are untouched.",
        "- Direct item lookup is attempted before time-window STAC search.",
        "- No EnMAP imagery was downloaded.",
        f"- Updated download manifest: {downloadable_path}",
        f"- Updated review manifest: {review_path}",
    ]

    summary_path = out_dir / "retry259_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
