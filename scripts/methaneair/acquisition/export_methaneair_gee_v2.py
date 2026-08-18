#!/usr/bin/env python3
"""
Download all MethaneAIR data that the current Earth Engine account is permitted
to access.

Official assets:
  L4 point sources:
    projects/edf-methanesat-ee/assets/mair/L4point
  L3 concentration:
    projects/edf-methanesat-ee/assets/mair/L3concentration

Outputs:
  data/methaneair_full/methaneair_l4_points.csv
  data/methaneair_full/methaneair_l4_points.geojson
  data/methaneair_full/methaneair_l3_inventory.csv
  data/methaneair_full/methaneair_l3_patches/*.tif
  data/methaneair_full/methaneair_l3_patch_manifest.csv

Notes:
- MethaneAIR L4 detections are positive observational labels.
- Absence from L4 is NOT a confirmed negative label.
- Access to the publisher datasets may require prior approval.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests


L4_ASSET = "projects/edf-methanesat-ee/assets/mair/L4point"
L3_ASSET = "projects/edf-methanesat-ee/assets/mair/L3concentration"
L3_BANDS = ["XCH4", "albedo", "surface_pressure", "terrain_height"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path("/project/6002520/yunjung1/MethaneFuse"),
    )
    p.add_argument(
        "--ee-project",
        default=os.environ.get("EE_PROJECT", "methane-release-gee"),
    )
    p.add_argument("--download-l3-patches", action="store_true")
    p.add_argument("--patch-size-km", type=float, default=4.0)
    p.add_argument("--scale-m", type=float, default=10.2)
    p.add_argument("--max-points", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--request-timeout", type=int, default=180)
    return p.parse_args()


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return text.strip("_") or "unknown"


def normalize_flight_id(value: Any) -> str:
    """Map split L4 IDs such as GV04-1/GV04-2 to L3 base ID GV04."""
    text = str(value).strip()
    return re.sub(r"-\d+$", "", text)


def initialize_ee(project: str):
    try:
        import ee
    except ImportError as exc:
        raise SystemExit(
            "earthengine-api is not installed in this Python environment.\n"
            "Use an environment that already has Earth Engine, or install "
            "`earthengine-api` before rerunning."
        ) from exc

    try:
        ee.Initialize(project=project)
    except Exception as exc:
        message = str(exc)
        raise SystemExit(
            "Earth Engine initialization failed.\n"
            f"Project: {project}\n"
            f"Error: {message}\n\n"
            "Authenticate first in an interactive terminal:\n"
            "  earthengine authenticate\n"
            "or:\n"
            "  python -c \"import ee; ee.Authenticate()\"\n"
        ) from exc
    return ee


def geometry_lon_lat(geometry: dict[str, Any] | None) -> tuple[Any, Any]:
    if not geometry:
        return pd.NA, pd.NA

    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")

    if geom_type == "Point" and isinstance(coords, list) and len(coords) >= 2:
        return coords[0], coords[1]

    return pd.NA, pd.NA


def save_l4(ee, output_root: Path) -> pd.DataFrame:
    fc = ee.FeatureCollection(L4_ASSET)

    try:
        count = int(fc.size().getInfo())
    except Exception as exc:
        output_root.mkdir(parents=True, exist_ok=True)
        access_file = output_root / "ACCESS_REQUIRED.txt"
        access_file.write_text(
            "The Earth Engine account cannot read the MethaneAIR publisher "
            "dataset.\n\n"
            f"Asset: {L4_ASSET}\n"
            f"Error: {type(exc).__name__}: {exc}\n\n"
            "Request MethaneSAT/MethaneAIR data access, then rerun this script.\n",
            encoding="utf-8",
        )
        raise SystemExit(
            "MethaneAIR access is not available for this account. "
            f"Details saved to {access_file}"
        ) from exc

    print(f"MethaneAIR L4 features available: {count}", flush=True)

    info = fc.getInfo()
    features = info.get("features", [])

    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(features):
        props = dict(feature.get("properties", {}))
        lon, lat = geometry_lon_lat(feature.get("geometry"))

        row = {
            "record_id": props.get("plume_id") or feature.get("id") or f"mair_{index:06d}",
            "plume_id": props.get("plume_id"),
            "flight_id": props.get("flight_id"),
            "basin": props.get("basin"),
            "time_coverage_start": props.get("time_coverage_start"),
            "time_coverage_end": props.get("time_coverage_end"),
            "flux_kg_hr": props.get("flux"),
            "flux_sd_kg_hr": props.get("flux_sd"),
            "longitude": lon,
            "latitude": lat,
            "label": 1,
            "proposed_label": pd.NA,
            "ground_truth_type": "MethaneAIR_observational_detection",
            "ground_truth_source": "MethaneAIR_L4_point_sources",
            "label_confidence": "medium",
            "controlled_release_verified": False,
            "image_source": "MethaneAIR_L3",
            "source_asset": L4_ASSET,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "methaneair_l4_points.csv"
    df.to_csv(csv_path, index=False)

    geojson_path = output_root / "methaneair_l4_points.geojson"
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    geojson_path.write_text(json.dumps(geojson), encoding="utf-8")

    print(f"Saved L4 CSV: {csv_path}", flush=True)
    print(f"Saved L4 GeoJSON: {geojson_path}", flush=True)
    return df


def save_l3_inventory(ee, output_root: Path) -> pd.DataFrame:
    ic = ee.ImageCollection(L3_ASSET)
    count = int(ic.size().getInfo())
    print(f"MethaneAIR L3 images available: {count}", flush=True)

    properties = [
        "system:index",
        "system:time_start",
        "flight_id",
        "target_id",
        "time_coverage_start",
        "time_coverage_end",
        "processing_id",
    ]

    rows = []
    batch_size = 100
    images = ic.toList(count)

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch = []

        for i in range(start, stop):
            image = ee.Image(images.get(i))
            dictionary = image.toDictionary(properties).getInfo()
            dictionary["image_id"] = image.id().getInfo()
            batch.append(dictionary)

        rows.extend(batch)
        print(f"L3 metadata: {stop}/{count}", flush=True)

    df = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "methaneair_l3_inventory.csv"
    df.to_csv(path, index=False)
    print(f"Saved L3 inventory: {path}", flush=True)
    return df


def write_downloaded_content(content: bytes, destination: Path) -> list[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if content[:2] == b"PK":
        created = []
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for member in zf.namelist():
                if not member.lower().endswith((".tif", ".tiff")):
                    continue
                output = destination.parent / (
                    destination.stem + "__" + Path(member).name
                )
                output.write_bytes(zf.read(member))
                created.append(output)
        return created

    destination.write_bytes(content)
    return [destination]


def download_l3_patches(
    ee,
    points: pd.DataFrame,
    output_root: Path,
    patch_size_km: float,
    scale_m: float,
    max_points: int,
    resume: bool,
    timeout: int,
) -> pd.DataFrame:
    patch_dir = output_root / "methaneair_l3_patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    work = points.copy()
    if max_points > 0:
        work = work.head(max_points)

    ic = ee.ImageCollection(L3_ASSET)
    half_size_m = patch_size_km * 1000.0 / 2.0
    session = requests.Session()

    for number, (_, row) in enumerate(work.iterrows(), start=1):
        plume_id = safe_name(row.get("plume_id") or row.get("record_id"))
        destination = patch_dir / f"{plume_id}.tif"

        manifest = {
            "record_id": row.get("record_id"),
            "plume_id": row.get("plume_id"),
            "flight_id": row.get("flight_id"),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "l3_image_id": pd.NA,
            "l3_patch_path": str(destination),
            "status": "pending",
            "error": "",
        }

        if resume and destination.exists():
            manifest["status"] = "already_exists"
            rows.append(manifest)
            continue

        try:
            lon = float(row["longitude"])
            lat = float(row["latitude"])
            point = ee.Geometry.Point([lon, lat])
            candidates = ic.filterBounds(point)

            flight_id = row.get("flight_id")
            if pd.notna(flight_id) and str(flight_id).strip():
                exact_id = str(flight_id).strip()
                base_id = normalize_flight_id(exact_id)

                matched = None
                for candidate_id in dict.fromkeys([exact_id, base_id]):
                    by_flight = candidates.filter(
                        ee.Filter.eq("flight_id", candidate_id)
                    )
                    if int(by_flight.size().getInfo()) > 0:
                        matched = by_flight
                        break

                if matched is not None:
                    candidates = matched

            if int(candidates.size().getInfo()) == 0:
                manifest["status"] = "no_l3_match"
                rows.append(manifest)
                continue

            # When a flight has multiple L3 mosaics, choose the image whose
            # system time is closest to the L4 plume observation time.
            observation_time = pd.to_datetime(
                row.get("time_coverage_start"),
                errors="coerce",
                utc=True,
            )

            if pd.notna(observation_time):
                observation_millis = int(
                    observation_time.timestamp() * 1000
                )

                def add_time_difference(candidate):
                    candidate = ee.Image(candidate)
                    image_time = ee.Number(
                        candidate.get("system:time_start")
                    )
                    return candidate.set(
                        "_l4_l3_time_difference_ms",
                        image_time.subtract(
                            observation_millis
                        ).abs(),
                    )

                candidates = candidates.map(
                    add_time_difference
                ).sort("_l4_l3_time_difference_ms")

            image = ee.Image(candidates.first())
            image_id = image.id().getInfo()
            manifest["l3_image_id"] = image_id
            manifest["matched_l3_flight_id"] = image.get(
                "flight_id"
            ).getInfo()
            manifest["l3_time_difference_seconds"] = (
                float(
                    image.get(
                        "_l4_l3_time_difference_ms"
                    ).getInfo()
                ) / 1000.0
                if pd.notna(observation_time)
                else pd.NA
            )

            region = point.buffer(half_size_m).bounds()

            # Force all L3 bands to a common data type before writing
            # a single multi-band GeoTIFF.
            export_image = (
                image
                .select(L3_BANDS)
                .toFloat()
                .clip(region)
            )

            url = export_image.getDownloadURL(
                {
                    "bands": L3_BANDS,
                    "region": region,
                    "scale": scale_m,
                    "format": "GEO_TIFF",
                }
            )

            response = session.get(url, timeout=timeout)

            if response.status_code != 200:
                content_type = response.headers.get("content-type", "")
                if "text" in content_type or "json" in content_type:
                    body = response.text[:4000]
                else:
                    body = repr(response.content[:1000])

                raise RuntimeError(
                    f"Earth Engine download HTTP {response.status_code}; "
                    f"content_type={content_type}; body={body}"
                )

            created = write_downloaded_content(response.content, destination)

            if not created:
                raise RuntimeError("No GeoTIFF was returned.")

            manifest["l3_patch_path"] = "|".join(str(path) for path in created)
            manifest["status"] = "downloaded"

        except Exception as exc:
            manifest["status"] = "error"
            manifest["error"] = f"{type(exc).__name__}: {exc}"

        rows.append(manifest)
        print(
            f"L3 patch {number}/{len(work)}: {plume_id} -> "
            f"{manifest['status']}",
            flush=True,
        )

        if number % 20 == 0:
            pd.DataFrame(rows).to_csv(
                output_root / "methaneair_l3_patch_manifest.partial.csv",
                index=False,
            )

        time.sleep(0.2)

    result = pd.DataFrame(rows)
    result.to_csv(
        output_root / "methaneair_l3_patch_manifest.csv",
        index=False,
    )
    return result


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_root = project_root / "data" / "methaneair_full"

    ee = initialize_ee(args.ee_project)
    points = save_l4(ee, output_root)
    save_l3_inventory(ee, output_root)

    if args.download_l3_patches:
        download_l3_patches(
            ee=ee,
            points=points,
            output_root=output_root,
            patch_size_km=args.patch_size_km,
            scale_m=args.scale_m,
            max_points=args.max_points,
            resume=args.resume,
            timeout=args.request_timeout,
        )

    print("\nMethaneAIR export complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
