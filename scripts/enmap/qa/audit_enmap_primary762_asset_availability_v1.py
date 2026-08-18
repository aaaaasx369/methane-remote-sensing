#!/usr/bin/env python3
"""
audit_enmap_primary762_asset_availability_v1.py

Full availability audit for the PRIMARY EnMAP dataset:
    <=72 h + NOMINAL L2A

For each unique scene in:
    ~/methane_release_project/enmap_download_phases_v1/
        03_phase2_primary_total_AB_nominal.csv

This script:
1) Resolves the CURRENT EnMAP L2A product using STRICT exact matching:
      same datatake ID AND same tile ID
2) Tests the metadata asset using IPS Basic Auth (username doraaa)
3) If the original STAC href returns 404, tries the repaired path:
      .../<tile>/<SCENE_ID>/<filename>
4) Reads only byte 0 (Range: bytes=0-0), never the full product.
5) Writes a complete audit and a download-ready scene manifest.

No EnMAP imagery is downloaded.

Outputs:
  ~/methane_release_project/enmap_primary762_availability_v1/
      01_primary762_asset_availability.csv
      02_downloadable_scenes.csv
      03_unavailable_or_review.csv
      04_availability_by_dataset.csv
      enmap_primary762_availability_summary.txt

Status values:
  ORIGINAL_WORKS
  REPAIRED_PATH_WORKS
  NO_CURRENT_EXACT_PRODUCT
  STILL_404
  AUTH_FAILURE
  HTTP_OTHER
  ERROR
"""

from __future__ import annotations

import argparse
import getpass
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlsplit, urlunsplit

import pandas as pd
import urllib3


STAC_ROOT = "https://geoservice.dlr.de/eoc/ogc/stac/v1/"
COLLECTION = "ENMAP_HSI_L2A"
STAC_ITEMS = f"{STAC_ROOT}collections/{COLLECTION}/items"

USERNAME = "doraaa"
USER_AGENT = "UAlberta-EnMAP-Primary762-Availability-Audit/1.0"

DEFAULT_MANIFEST = (
    Path.home()
    / "methane_release_project"
    / "enmap_download_phases_v1"
    / "03_phase2_primary_total_AB_nominal.csv"
)

DEFAULT_OUT = (
    Path.home()
    / "methane_release_project"
    / "enmap_primary762_availability_v1"
)

SCENE_RE = re.compile(
    r"DT(?P<datatake>\d+)_"
    r"(?P<acq>\d{8}T\d{6}Z)_"
    r"(?P<tile>\d{3})_"
    r"V(?P<version>\d+)_"
    r"(?P<proc>\d{8}T\d{6}Z)"
)


# =============================================================================
# Helpers
# =============================================================================

def parse_scene(scene_id: str) -> Dict[str, Optional[str]]:
    m = SCENE_RE.search(str(scene_id))
    if not m:
        return {
            "datatake": None,
            "acq": None,
            "tile": None,
            "version": None,
            "proc": None,
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


def candidate_sort_key(feat: Dict[str, Any]):
    sid = str(feat.get("id") or "")
    parsed = parse_scene(sid)
    p = feat.get("properties", {}) or {}
    return (
        str(parsed.get("proc") or ""),
        str(p.get("updated") or ""),
        str(p.get("created") or ""),
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
    path = parts.path
    pieces = path.rstrip("/").split("/")

    if len(pieces) < 1:
        return url

    filename = pieces[-1]

    if len(pieces) >= 2 and pieces[-2] == scene_id:
        return url

    new_path = "/".join(pieces[:-1] + [scene_id, filename])
    if path.startswith("/") and not new_path.startswith("/"):
        new_path = "/" + new_path

    return urlunsplit(
        (parts.scheme, parts.netloc, new_path, parts.query, parts.fragment)
    )


# =============================================================================
# STAC
# =============================================================================

def stac_get_json(
    http: urllib3.PoolManager,
    url: str,
    retries: int = 6,
) -> Dict[str, Any]:
    last = None

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

            if r.status == 200:
                return json.loads(r.data.decode("utf-8"))

            last = RuntimeError(f"HTTP {r.status}")

            if r.status in (429, 500, 502, 503, 504):
                wait = min(20, 2 ** (attempt - 1))
                print(
                    f"  STAC retry {attempt}/{retries}: "
                    f"HTTP {r.status}; sleep {wait}s"
                )
                time.sleep(wait)
                continue

            raise RuntimeError(
                f"STAC HTTP {r.status}: {url}"
            )

        except Exception as e:
            last = e
            if attempt == retries:
                break
            wait = min(20, 2 ** (attempt - 1))
            print(
                f"  STAC retry {attempt}/{retries}: "
                f"{type(e).__name__}: {e}; sleep {wait}s"
            )
            time.sleep(wait)

    raise RuntimeError(f"STAC request failed: {url}\n{last}")


def query_exact_current(
    http: urllib3.PoolManager,
    original_scene_id: str,
) -> Optional[Dict[str, Any]]:
    parsed = parse_scene(original_scene_id)

    wanted_dt = normalize_digits(parsed.get("datatake"))
    wanted_tile = normalize_tile(parsed.get("tile"))
    acq = parsed.get("acq")

    if not wanted_dt or not wanted_tile or not acq:
        return None

    t = pd.to_datetime(
        acq,
        format="%Y%m%dT%H%M%SZ",
        errors="coerce",
        utc=True,
    )
    if pd.isna(t):
        return None

    t0 = t - pd.Timedelta(minutes=20)
    t1 = t + pd.Timedelta(minutes=20)

    params = urlencode(
        {
            "datetime": (
                f"{t0.isoformat().replace('+00:00', 'Z')}/"
                f"{t1.isoformat().replace('+00:00', 'Z')}"
            ),
            "limit": 100,
            "f": "json",
        }
    )

    doc = stac_get_json(http, f"{STAC_ITEMS}?{params}")
    feats = doc.get("features", []) or []

    exact = [
        feat for feat in feats
        if feature_datatake(feat) == wanted_dt
        and feature_tile(feat) == wanted_tile
    ]

    if not exact:
        return None

    exact = sorted(
        exact,
        key=candidate_sort_key,
        reverse=True,
    )
    return exact[0]


# =============================================================================
# Download URL probing
# =============================================================================

def probe_asset(
    http: urllib3.PoolManager,
    url: str,
    password: str,
    retries: int = 4,
) -> Dict[str, Any]:
    headers = urllib3.make_headers(
        basic_auth=f"{USERNAME}:{password}"
    )
    headers["User-Agent"] = USER_AGENT
    headers["Range"] = "bytes=0-0"

    last_error = None

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
            ctype = r.headers.get("Content-Type")
            crange = r.headers.get("Content-Range")
            location = r.headers.get("Location")

            # Consume at most 1 KiB, then close.
            _ = r.read(1024)
            r.release_conn()

            if status in (429, 500, 502, 503, 504):
                if attempt < retries:
                    wait = min(15, 2 ** (attempt - 1))
                    print(
                        f"    probe retry {attempt}/{retries}: "
                        f"HTTP {status}; sleep {wait}s"
                    )
                    time.sleep(wait)
                    continue

            return {
                "http_status": status,
                "content_type": ctype,
                "content_range": crange,
                "location": location,
                "error": None,
            }

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < retries:
                wait = min(15, 2 ** (attempt - 1))
                print(
                    f"    probe retry {attempt}/{retries}: "
                    f"{last_error}; sleep {wait}s"
                )
                time.sleep(wait)

    return {
        "http_status": None,
        "content_type": None,
        "content_range": None,
        "location": None,
        "error": last_error,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all scenes",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path, low_memory=False)
    df = df.drop_duplicates("l2a_scene_id").copy()

    if args.limit > 0:
        df = df.head(args.limit).copy()

    print("=" * 92)
    print("ENMAP PRIMARY <=72H NOMINAL — FULL ASSET AVAILABILITY AUDIT")
    print("=" * 92)
    print(f"Scenes to audit : {len(df)}")
    print(f"Username        : {USERNAME}")
    print("Reads per URL   : 1 byte via HTTP Range")
    print("No imagery will be downloaded.")
    print()

    password = getpass.getpass(
        f"IPS password for {USERNAME}: "
    )
    if not password:
        raise SystemExit("Empty password; aborting.")

    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=30.0, read=60.0),
        retries=False,
        num_pools=10,
    )

    rows: List[Dict[str, Any]] = []

    for pos, (_, r) in enumerate(df.iterrows(), start=1):
        original_scene_id = str(r["l2a_scene_id"]).strip()

        print()
        print(f"[{pos}/{len(df)}] {original_scene_id}")

        base = {
            "original_scene_id": original_scene_id,
            "tier": r.get("tier"),
            "quality_class": r.get("quality_class"),
            "supporting_records": r.get("supporting_records"),
            "supporting_datasets_count": r.get(
                "supporting_datasets_count"
            ),
            "supporting_datasets": r.get("supporting_datasets"),
            "min_abs_delta_hours": r.get("min_abs_delta_hours"),
        }

        try:
            current = query_exact_current(http, original_scene_id)

            if current is None:
                print("  STATUS: NO_CURRENT_EXACT_PRODUCT")
                rows.append({
                    **base,
                    "current_scene_id": None,
                    "metadata_url_original": None,
                    "metadata_url_repaired": None,
                    "selected_metadata_url": None,
                    "path_style": None,
                    "original_http": None,
                    "repaired_http": None,
                    "status": "NO_CURRENT_EXACT_PRODUCT",
                })
                continue

            current_scene_id = str(current.get("id") or "")
            href = metadata_href(current)

            print(f"  current exact: {current_scene_id}")

            if not href:
                print("  STATUS: ERROR — metadata href missing")
                rows.append({
                    **base,
                    "current_scene_id": current_scene_id,
                    "metadata_url_original": None,
                    "metadata_url_repaired": None,
                    "selected_metadata_url": None,
                    "path_style": None,
                    "original_http": None,
                    "repaired_http": None,
                    "status": "ERROR",
                    "error": "metadata href missing",
                })
                continue

            original_probe = probe_asset(
                http, href, password
            )
            original_http = original_probe["http_status"]

            if original_http in (200, 206):
                status = "ORIGINAL_WORKS"
                selected = href
                path_style = "ORIGINAL"
                repaired = insert_scene_directory(
                    href, current_scene_id
                )
                repaired_http = None

                print(
                    f"  ORIGINAL HTTP {original_http} -> WORKS"
                )

            elif original_http == 404:
                repaired = insert_scene_directory(
                    href, current_scene_id
                )

                if repaired == href:
                    repaired_probe = original_probe
                else:
                    repaired_probe = probe_asset(
                        http, repaired, password
                    )

                repaired_http = repaired_probe["http_status"]

                if repaired_http in (200, 206):
                    status = "REPAIRED_PATH_WORKS"
                    selected = repaired
                    path_style = "SCENE_DIRECTORY_INSERTED"
                    print(
                        f"  ORIGINAL 404 -> REPAIRED "
                        f"HTTP {repaired_http} -> WORKS"
                    )
                elif repaired_http in (401, 403):
                    status = "AUTH_FAILURE"
                    selected = None
                    path_style = None
                    print(
                        f"  ORIGINAL 404 -> REPAIRED "
                        f"HTTP {repaired_http} -> AUTH FAILURE"
                    )
                elif repaired_http == 404:
                    status = "STILL_404"
                    selected = None
                    path_style = None
                    print(
                        "  ORIGINAL 404 -> REPAIRED 404"
                    )
                else:
                    status = "HTTP_OTHER"
                    selected = None
                    path_style = None
                    print(
                        f"  ORIGINAL 404 -> REPAIRED "
                        f"HTTP {repaired_http}"
                    )

            elif original_http in (401, 403):
                repaired = insert_scene_directory(
                    href, current_scene_id
                )
                repaired_http = None
                selected = None
                path_style = None
                status = "AUTH_FAILURE"
                print(
                    f"  ORIGINAL HTTP {original_http} "
                    "-> AUTH FAILURE"
                )

            elif original_http is None:
                repaired = insert_scene_directory(
                    href, current_scene_id
                )
                repaired_http = None
                selected = None
                path_style = None
                status = "ERROR"
                print(
                    "  ORIGINAL probe error: "
                    f"{original_probe.get('error')}"
                )

            else:
                repaired = insert_scene_directory(
                    href, current_scene_id
                )
                repaired_http = None
                selected = None
                path_style = None
                status = "HTTP_OTHER"
                print(
                    f"  ORIGINAL HTTP {original_http}"
                )

            rows.append({
                **base,
                "current_scene_id": current_scene_id,
                "metadata_url_original": href,
                "metadata_url_repaired": repaired,
                "selected_metadata_url": selected,
                "path_style": path_style,
                "original_http": original_http,
                "repaired_http": repaired_http,
                "status": status,
                "error": original_probe.get("error"),
            })

        except Exception as e:
            print(
                f"  STATUS: ERROR — {type(e).__name__}: {e}"
            )
            rows.append({
                **base,
                "current_scene_id": None,
                "metadata_url_original": None,
                "metadata_url_repaired": None,
                "selected_metadata_url": None,
                "path_style": None,
                "original_http": None,
                "repaired_http": None,
                "status": "ERROR",
                "error": f"{type(e).__name__}: {e}",
            })

        # Save checkpoint every 25 scenes.
        if pos % 25 == 0:
            checkpoint = out_dir / "availability_checkpoint.csv"
            pd.DataFrame(rows).to_csv(
                checkpoint, index=False
            )
            print(f"  checkpoint -> {checkpoint}")

    out = pd.DataFrame(rows)

    full_path = out_dir / "01_primary762_asset_availability.csv"
    out.to_csv(full_path, index=False)

    downloadable = out[
        out["status"].isin([
            "ORIGINAL_WORKS",
            "REPAIRED_PATH_WORKS",
        ])
    ].copy()

    downloadable_path = out_dir / "02_downloadable_scenes.csv"
    downloadable.to_csv(downloadable_path, index=False)

    review = out[
        ~out["status"].isin([
            "ORIGINAL_WORKS",
            "REPAIRED_PATH_WORKS",
        ])
    ].copy()

    review_path = out_dir / "03_unavailable_or_review.csv"
    review.to_csv(review_path, index=False)

    # Dataset support summary (scene can support multiple datasets).
    dataset_names = [
        "CONTROLLED_RELEASE_VERIFIED_107",
        "STANFORD_2024_2025_746",
        "METHANEAIR_435",
        "METHANESAT_POSNEG_222",
        "UNEP_MARS_PLUMES",
        "CARBON_MAPPER_CH4_PLUMES",
        "AVIRIS3_SCENES_493",
        "EMIT_POSNEG_100",
    ]

    ds_rows = []
    for ds in dataset_names:
        mask = (
            out["supporting_datasets"]
            .fillna("")
            .astype(str)
            .str.contains(ds, regex=False)
        )
        d = out[mask]

        ds_rows.append({
            "dataset": ds,
            "supported_unique_scenes": len(d),
            "downloadable_scenes": int(
                d["status"].isin([
                    "ORIGINAL_WORKS",
                    "REPAIRED_PATH_WORKS",
                ]).sum()
            ),
            "original_works": int(
                (d["status"] == "ORIGINAL_WORKS").sum()
            ),
            "repaired_path_works": int(
                (d["status"] == "REPAIRED_PATH_WORKS").sum()
            ),
            "still_404": int(
                (d["status"] == "STILL_404").sum()
            ),
            "no_current_exact_product": int(
                (
                    d["status"]
                    == "NO_CURRENT_EXACT_PRODUCT"
                ).sum()
            ),
        })

    ds_summary = pd.DataFrame(ds_rows)
    ds_summary.to_csv(
        out_dir / "04_availability_by_dataset.csv",
        index=False,
    )

    counts = out["status"].value_counts(dropna=False)

    original_ok = int(
        (out["status"] == "ORIGINAL_WORKS").sum()
    )
    repaired_ok = int(
        (out["status"] == "REPAIRED_PATH_WORKS").sum()
    )
    total_ok = original_ok + repaired_ok

    lines = [
        "ENMAP PRIMARY <=72H NOMINAL — ASSET AVAILABILITY SUMMARY",
        "=" * 92,
        f"Input unique scenes           : {len(out)}",
        f"Downloadable scenes           : {total_ok}",
        f"  Original STAC href works    : {original_ok}",
        f"  Repaired scene-folder works : {repaired_ok}",
        f"Unavailable / review          : {len(out) - total_ok}",
        "",
        "STATUS COUNTS",
    ]

    for k, v in counts.items():
        lines.append(
            f"{str(k):34s} {int(v)}"
        )

    lines += [
        "",
        "IMPORTANT",
        "- Exact-current products are accepted only when datatake AND tile match.",
        "- Asset probes read only byte 0; no EnMAP imagery was downloaded.",
        "- REPAIRED_PATH_WORKS means the STAC href required insertion of /<SCENE_ID>/.",
        f"- Download-ready manifest: {downloadable_path}",
        f"- Review manifest: {review_path}",
    ]

    summary_path = (
        out_dir
        / "enmap_primary762_availability_summary.txt"
    )
    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
