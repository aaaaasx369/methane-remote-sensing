#!/usr/bin/env python3
"""
Download public Carbon Mapper plume products near controlled-release sites.

Workflow
--------
1. Query Carbon Mapper's public /catalog/plumes/annotated endpoint.
2. Save raw JSON and a flattened CSV inventory.
3. Optionally download selected public assets such as:
   - plume_tif
   - con_tif
   - rgb_tif
   - plume_png
   - rgb_png

This script intentionally starts with plume-level products. It does not claim to
download non-public L1B raw hyperspectral radiance cubes.

Example
-------
python download_carbon_mapper_controlled_release.py \
    --sites carbon_mapper_controlled_release_sites.csv \
    --output data/carbon_mapper_controlled_release \
    --download \
    --assets plume_tif con_tif rgb_tif

Optional authenticated use
--------------------------
Set an access token only when needed:
    export CARBON_MAPPER_TOKEN="..."
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_BASE = "https://api.carbonmapper.org/api/v1"
PLUMES_ENDPOINT = f"{API_BASE}/catalog/plumes/annotated"

KNOWN_ASSET_KEYS = (
    "plume_tif",
    "con_tif",
    "rgb_tif",
    "plume_png",
    "plume_rgb_png",
    "rgb_png",
)

DEFAULT_ASSET_KEYS = (
    "plume_tif",
    "con_tif",
    "rgb_tif",
)


@dataclass(frozen=True)
class Site:
    site_id: str
    latitude: float
    longitude: float
    start_utc: str
    end_utc: str
    radius_km: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query and download public Carbon Mapper plume products near "
            "controlled-release sites."
        )
    )
    parser.add_argument(
        "--sites",
        type=Path,
        required=True,
        help="CSV containing site_id, latitude, longitude, start_utc, end_utc, radius_km.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/carbon_mapper_controlled_release"),
        help="Output directory.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download selected assets. Without this flag, only metadata are saved.",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=list(DEFAULT_ASSET_KEYS),
        help=(
            "Asset fields to download. Known choices: "
            + ", ".join(KNOWN_ASSET_KEYS)
            + ". Use 'all' to download every known field."
        ),
    )
    parser.add_argument(
        "--instrument",
        action="append",
        default=[],
        help=(
            "Optional instrument filter. Repeat for multiple instruments. "
            "Examples: ang, GAO, av3. By default no instrument filter is used."
        ),
    )
    parser.add_argument(
        "--quality",
        choices=("any", "good", "questionable", "bad"),
        default="any",
        help="Optional plume-quality filter.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=500,
        help="Rows requested per API page.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Delay between downloads to reduce API/server load.",
    )
    return parser.parse_args()


def make_session() -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "UAlberta-controlled-release-research/"
                "1.0 (non-commercial academic research)"
            )
        }
    )

    token = os.getenv("CARBON_MAPPER_TOKEN", "").strip()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"

    return session


def read_sites(path: Path) -> list[Site]:
    required = {
        "site_id",
        "latitude",
        "longitude",
        "start_utc",
        "end_utc",
        "radius_km",
    }

    if not path.exists():
        raise FileNotFoundError(f"Sites CSV not found: {path}")

    sites: list[Site] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Sites CSV is missing columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                site = Site(
                    site_id=str(row["site_id"]).strip(),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    start_utc=str(row["start_utc"]).strip(),
                    end_utc=str(row["end_utc"]).strip(),
                    radius_km=float(row["radius_km"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid value in {path} row {row_number}: {exc}"
                ) from exc

            if not site.site_id:
                raise ValueError(f"Blank site_id in {path} row {row_number}")
            if not (-90 <= site.latitude <= 90):
                raise ValueError(f"Invalid latitude for {site.site_id}")
            if not (-180 <= site.longitude <= 180):
                raise ValueError(f"Invalid longitude for {site.site_id}")
            if site.radius_km <= 0:
                raise ValueError(f"radius_km must be positive for {site.site_id}")

            sites.append(site)

    if not sites:
        raise ValueError(f"No sites found in {path}")

    return sites


def site_bbox(site: Site) -> tuple[float, float, float, float]:
    """
    Approximate a radius-km square bounding box around a WGS84 point.
    Sufficient for small AOIs used in this inventory query.
    """
    lat_delta = site.radius_km / 111.32
    lon_scale = max(math.cos(math.radians(site.latitude)), 1e-6)
    lon_delta = site.radius_km / (111.32 * lon_scale)

    return (
        site.longitude - lon_delta,
        site.latitude - lat_delta,
        site.longitude + lon_delta,
        site.latitude + lat_delta,
    )


def build_query_params(
    site: Site,
    offset: int,
    page_size: int,
    instruments: list[str],
    quality: str,
) -> list[tuple[str, str | int]]:
    xmin, ymin, xmax, ymax = site_bbox(site)

    # Carbon Mapper documents bbox as four repeated query parameters.
    params: list[tuple[str, str | int]] = [
        ("bbox", f"{xmin:.8f}"),
        ("bbox", f"{ymin:.8f}"),
        ("bbox", f"{xmax:.8f}"),
        ("bbox", f"{ymax:.8f}"),
        ("datetime", f"{site.start_utc}/{site.end_utc}"),
        ("plume_gas", "CH4"),
        ("sort", "asc"),
        ("limit", page_size),
        ("offset", offset),
    ]

    if quality != "any":
        params.append(("plume_quality", quality))

    for instrument in instruments:
        params.append(("instrument", instrument))

    return params


def query_site(
    session: requests.Session,
    site: Site,
    page_size: int,
    instruments: list[str],
    quality: str,
    timeout: float,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    offset = 0
    total_count: int | None = None
    bbox_count: int | None = None

    while True:
        params = build_query_params(
            site=site,
            offset=offset,
            page_size=page_size,
            instruments=instruments,
            quality=quality,
        )
        response = session.get(
            PLUMES_ENDPOINT,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()

        payload = response.json()
        page_items = payload.get("items", [])
        if not isinstance(page_items, list):
            raise RuntimeError(
                f"Unexpected API response for {site.site_id}: items is not a list"
            )

        if total_count is None:
            raw_total = payload.get("total_count")
            total_count = int(raw_total) if raw_total is not None else None
            raw_bbox = payload.get("bbox_count")
            bbox_count = int(raw_bbox) if raw_bbox is not None else None

        items.extend(page_items)

        print(
            f"[{site.site_id}] offset={offset} "
            f"received={len(page_items)} accumulated={len(items)}"
        )

        if not page_items:
            break

        offset += len(page_items)

        if total_count is not None and offset >= total_count:
            break
        if len(page_items) < page_size:
            break

    return {
        "site": {
            "site_id": site.site_id,
            "latitude": site.latitude,
            "longitude": site.longitude,
            "start_utc": site.start_utc,
            "end_utc": site.end_utc,
            "radius_km": site.radius_km,
            "bbox": list(site_bbox(site)),
        },
        "api_endpoint": PLUMES_ENDPOINT,
        "total_count": total_count,
        "bbox_count": bbox_count,
        "returned_count": len(items),
        "items": items,
    }


def safe_filename(value: str, fallback: str) -> str:
    value = value.strip()
    if not value:
        value = fallback
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value[:240]


def extract_coordinates(item: dict[str, Any]) -> tuple[Any, Any]:
    geometry = item.get("geometry_json") or item.get("geometry") or {}
    coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
        return coordinates[1], coordinates[0]
    return None, None


def flatten_item(site_id: str, item: dict[str, Any]) -> dict[str, Any]:
    latitude, longitude = extract_coordinates(item)

    row: dict[str, Any] = {
        "site_id": site_id,
        "id": item.get("id"),
        "plume_id": item.get("plume_id"),
        "gas": item.get("gas"),
        "scene_id": item.get("scene_id"),
        "scene_timestamp": item.get("scene_timestamp"),
        "instrument": item.get("instrument"),
        "platform": item.get("platform"),
        "mission_phase": item.get("mission_phase"),
        "plume_latitude": latitude,
        "plume_longitude": longitude,
        "emission_auto": item.get("emission_auto"),
        "emission_uncertainty_auto": item.get(
            "emission_uncertainty_auto"
        ),
        "wind_speed_avg_auto": item.get("wind_speed_avg_auto"),
        "wind_direction_avg_auto": item.get(
            "wind_direction_avg_auto"
        ),
        "plume_quality": item.get("plume_quality")
        or item.get("quality"),
        "collection": item.get("collection"),
        "cmf_type": item.get("cmf_type"),
        "sector": item.get("sector"),
        "status": item.get("status"),
        "published_at": item.get("published_at"),
        "modified": item.get("modified"),
    }

    for key in KNOWN_ASSET_KEYS:
        row[key] = item.get(key)

    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "site_id",
        "id",
        "plume_id",
        "gas",
        "scene_id",
        "scene_timestamp",
        "instrument",
        "platform",
        "mission_phase",
        "plume_latitude",
        "plume_longitude",
        "emission_auto",
        "emission_uncertainty_auto",
        "wind_speed_avg_auto",
        "wind_direction_avg_auto",
        "plume_quality",
        "collection",
        "cmf_type",
        "sector",
        "status",
        "published_at",
        "modified",
        *KNOWN_ASSET_KEYS,
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extension_for_asset(asset_key: str, url: str) -> str:
    parsed_suffix = Path(urlparse(url).path).suffix
    if parsed_suffix and len(parsed_suffix) <= 8:
        return parsed_suffix

    if asset_key.endswith("_tif"):
        return ".tif"
    if asset_key.endswith("_png"):
        return ".png"
    return ".bin"


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    timeout: float,
    overwrite: bool,
) -> str:
    if destination.exists() and not overwrite:
        return "exists"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    temporary.replace(destination)
    return "downloaded"


def resolve_asset_keys(values: Iterable[str]) -> tuple[str, ...]:
    normalized = [value.strip() for value in values if value.strip()]
    if not normalized:
        return DEFAULT_ASSET_KEYS

    if "all" in normalized:
        return KNOWN_ASSET_KEYS

    unknown = sorted(set(normalized).difference(KNOWN_ASSET_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown asset keys: {unknown}. "
            f"Known keys: {list(KNOWN_ASSET_KEYS)}"
        )
    return tuple(dict.fromkeys(normalized))


def download_assets(
    session: requests.Session,
    site_id: str,
    items: list[dict[str, Any]],
    output_root: Path,
    asset_keys: tuple[str, ...],
    timeout: float,
    overwrite: bool,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for item in items:
        plume_id = safe_filename(
            str(item.get("plume_id") or item.get("id") or ""),
            fallback="unknown_plume",
        )
        plume_dir = output_root / site_id / "assets" / plume_id

        for asset_key in asset_keys:
            raw_url = item.get(asset_key)
            url = str(raw_url).strip() if raw_url is not None else ""

            record = {
                "site_id": site_id,
                "plume_id": plume_id,
                "asset_key": asset_key,
                "url": url,
                "local_path": "",
                "status": "",
                "error": "",
            }

            if not url or not url.lower().startswith(("http://", "https://")):
                record["status"] = "missing_url"
                records.append(record)
                continue

            suffix = extension_for_asset(asset_key, url)
            destination = plume_dir / f"{plume_id}_{asset_key}{suffix}"
            record["local_path"] = str(destination)

            try:
                record["status"] = download_file(
                    session=session,
                    url=url,
                    destination=destination,
                    timeout=timeout,
                    overwrite=overwrite,
                )
            except requests.RequestException as exc:
                record["status"] = "failed"
                record["error"] = str(exc)
            except OSError as exc:
                record["status"] = "failed"
                record["error"] = str(exc)

            records.append(record)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return records


def write_download_manifest(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "site_id",
        "plume_id",
        "asset_key",
        "url",
        "local_path",
        "status",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    output_root: Path,
    site_payloads: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    download_rows: list[dict[str, Any]],
) -> None:
    instruments: dict[str, int] = {}
    for row in metadata_rows:
        key = str(row.get("instrument") or "missing")
        instruments[key] = instruments.get(key, 0) + 1

    statuses: dict[str, int] = {}
    for row in download_rows:
        key = str(row.get("status") or "missing")
        statuses[key] = statuses.get(key, 0) + 1

    summary = {
        "site_count": len(site_payloads),
        "plume_count": len(metadata_rows),
        "plumes_by_site": {
            payload["site"]["site_id"]: payload["returned_count"]
            for payload in site_payloads
        },
        "plumes_by_instrument": instruments,
        "download_status": statuses,
        "important_interpretation": [
            "Plume endpoint results are positive plume products, not a balanced classification dataset.",
            "No-result dates do not automatically mean a valid negative; scene coverage and quality must be checked separately.",
            "L1B raw hyperspectral radiance cubes are not assumed to be public.",
            "Keep Carbon Mapper attribution in all downstream research products.",
        ],
    }

    path = output_root / "carbon_mapper_download_summary.json"
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    if args.page_size <= 0 or args.page_size > 10000:
        raise ValueError("--page-size must be between 1 and 10000")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.sleep < 0:
        raise ValueError("--sleep cannot be negative")

    asset_keys = resolve_asset_keys(args.assets)
    sites = read_sites(args.sites)
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    session = make_session()

    all_metadata_rows: list[dict[str, Any]] = []
    all_download_rows: list[dict[str, Any]] = []
    site_payloads: list[dict[str, Any]] = []

    for site in sites:
        print("=" * 72)
        print(
            f"Querying {site.site_id}: "
            f"{site.start_utc} to {site.end_utc}, "
            f"radius={site.radius_km} km"
        )

        payload = query_site(
            session=session,
            site=site,
            page_size=args.page_size,
            instruments=args.instrument,
            quality=args.quality,
            timeout=args.timeout,
        )
        site_payloads.append(payload)

        site_dir = output_root / site.site_id
        metadata_dir = site_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        raw_path = metadata_dir / f"{site.site_id}_plumes_raw.json"
        raw_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        site_rows = [
            flatten_item(site.site_id, item)
            for item in payload["items"]
        ]
        all_metadata_rows.extend(site_rows)

        site_csv_path = metadata_dir / f"{site.site_id}_plumes.csv"
        write_csv(site_rows, site_csv_path)

        print(
            f"[{site.site_id}] plume rows saved: {len(site_rows)}"
        )

        if args.download:
            site_download_rows = download_assets(
                session=session,
                site_id=site.site_id,
                items=payload["items"],
                output_root=output_root,
                asset_keys=asset_keys,
                timeout=args.timeout,
                overwrite=args.overwrite,
                sleep_seconds=args.sleep,
            )
            all_download_rows.extend(site_download_rows)
            write_download_manifest(
                site_download_rows,
                metadata_dir / f"{site.site_id}_download_manifest.csv",
            )

    write_csv(
        all_metadata_rows,
        output_root / "carbon_mapper_all_plumes.csv",
    )

    if args.download:
        write_download_manifest(
            all_download_rows,
            output_root / "carbon_mapper_all_downloads.csv",
        )

    write_summary(
        output_root=output_root,
        site_payloads=site_payloads,
        metadata_rows=all_metadata_rows,
        download_rows=all_download_rows,
    )

    print("=" * 72)
    print(f"Output directory: {output_root}")
    print(f"Total plume rows: {len(all_metadata_rows)}")
    if args.download:
        downloaded = sum(
            row["status"] == "downloaded"
            for row in all_download_rows
        )
        existing = sum(
            row["status"] == "exists"
            for row in all_download_rows
        )
        failed = sum(
            row["status"] == "failed"
            for row in all_download_rows
        )
        print(
            f"Assets downloaded={downloaded}, "
            f"already_existing={existing}, failed={failed}"
        )
    else:
        print(
            "Metadata-only run complete. Add --download to fetch assets."
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
