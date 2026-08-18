#!/usr/bin/env python3
"""
Audit the ACTUAL files currently present in a MethaneFuse project.

This script does not use remembered or hard-coded dataset counts.
Run it on Fir against the live project directory.

Outputs:
  outputs/project_inventory_audit/
    001_all_data_files.csv
    002_tabular_inventory.csv
    003_status_qa_inventory.csv
    004_raster_inventory.csv
    005_known_dataset_presence.csv
    006_sentinel2_temporal_audit.csv
    007_audit_report.txt
    project_inventory_audit.zip

Optional:
  If openpyxl is installed, project_inventory_audit.xlsx is also created.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except Exception as exc:
    raise SystemExit(
        "pandas is required. Activate the carbonmapper311 environment first."
    ) from exc

try:
    import rasterio
except Exception:
    rasterio = None

try:
    import numpy as np
except Exception:
    np = None


DATA_EXTENSIONS = {
    ".csv", ".tsv", ".parquet", ".geojson", ".json",
    ".tif", ".tiff", ".npy", ".npz", ".pt", ".pth",
    ".zip", ".gz", ".nc", ".h5", ".hdf5",
}

EXCLUDED_DIRS = {
    ".git", ".idea", ".vscode", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "node_modules", "venv", ".venv",
    "env", ".env", "site-packages", "project_inventory_audit",
}

LABEL_ALIASES = (
    "label", "target", "class", "binary_label",
    "plume_label", "ground_truth_label", "y",
)
ID_ALIASES = (
    "record_id", "event_id", "plume_id", "sample_id",
    "observation_id", "acquisition_id", "dedup_key", "id",
)
SENSOR_ALIASES = (
    "sensor", "satellite", "platform", "spacecraft",
    "landsat_sensor", "source_sensor",
)
SOURCE_ALIASES = (
    "ground_truth_source", "dataset_source",
    "data_source", "source_type", "source",
)
TIME_ALIASES = (
    "event_time_utc", "datetime_utc", "datetime_UTC",
    "acquisition_time_utc", "time_coverage_start",
    "timestamp_utc", "scene_time_utc",
    "methaneair_time_utc", "datetime", "date",
)
LAT_ALIASES = (
    "latitude", "lat", "source_latitude", "source_lat", "Latitude",
)
LON_ALIASES = (
    "longitude", "lon", "lng", "source_longitude",
    "source_lon", "Longitude",
)

KNOWN_DATASETS = [
    ("Unified master fixed", "outputs/012_unified_methane_master_fixed.csv"),
    ("Controlled-release S2 patch index", "outputs/20_controlled_release_s2_patch_index.csv"),
    ("Historical MethaneAIR-S2 table", "outputs/18_methaneair_s2_dataset_table.csv"),
    ("Historical S2 features", "outputs/25_s2_patch_features.csv"),
    ("Five-site multisource manifest", "outputs/548_five_site_multisource_manifest_v1.csv"),
    ("MethaneAIR L4 points", "data/methaneair_full/methaneair_l4_points.csv"),
    ("MethaneAIR L4 GeoJSON", "data/methaneair_full/methaneair_l4_points.geojson"),
    ("MethaneAIR L3 inventory", "data/methaneair_full/methaneair_l3_inventory.csv"),
    ("MethaneAIR L3 patch manifest", "data/methaneair_full/methaneair_l3_patch_manifest.csv"),
    ("MethaneAIR/S2 temporal manifest", "data/methaneair_full/sentinel2_temporal_manifest.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit actual methane project files and dataset statistics."
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Live project root on Fir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/project_inventory_audit"),
    )
    parser.add_argument(
        "--scan-dir",
        action="append",
        default=[],
        help="Relative directory to scan; repeatable. Defaults to data and outputs.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="CSV rows per chunk.",
    )
    parser.add_argument(
        "--max-unique-ids",
        type=int,
        default=1_000_000,
        help="Stop storing unique IDs after this threshold.",
    )
    parser.add_argument(
        "--deep-raster",
        action="store_true",
        help="Read raster arrays to check zeros/NaN. Slower.",
    )
    return parser.parse_args()


def pick(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def iso_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "pass", "passed"}:
        return True
    if text in {"false", "f", "no", "n", "0", "fail", "failed"}:
        return False
    return None


def counter_text(counter: Counter, limit: int = 30) -> str:
    if not counter:
        return ""
    values = counter.most_common(limit)
    text = "; ".join(f"{key}:{count}" for key, count in values)
    if len(counter) > limit:
        text += f"; ...(+{len(counter)-limit})"
    return text


def iter_files(scan_roots: list[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for current, dirs, names in os.walk(scan_root):
            dirs[:] = [
                d for d in dirs
                if d not in EXCLUDED_DIRS and not d.startswith(".")
            ]
            for name in names:
                path = Path(current) / name
                if path.suffix.lower() not in DATA_EXTENSIONS:
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield path


def read_table_chunks(path: Path, chunksize: int):
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        yield from pd.read_csv(
            path,
            sep=sep,
            chunksize=chunksize,
            low_memory=False,
            on_bad_lines="warn",
        )
    elif suffix == ".parquet":
        yield pd.read_parquet(path)
    elif suffix == ".geojson":
        payload = json.loads(path.read_text(encoding="utf-8"))
        features = payload.get("features", [])
        rows = []
        for feature in features:
            row = dict(feature.get("properties") or {})
            geometry = feature.get("geometry") or {}
            coords = geometry.get("coordinates")
            if geometry.get("type") == "Point" and isinstance(coords, list) and len(coords) >= 2:
                row.setdefault("longitude", coords[0])
                row.setdefault("latitude", coords[1])
            rows.append(row)
        yield pd.DataFrame(rows)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            yield pd.DataFrame(payload)
        elif isinstance(payload, dict):
            yield pd.json_normalize(payload)
        else:
            yield pd.DataFrame({"value": [payload]})
    else:
        raise ValueError(f"Unsupported tabular extension: {suffix}")


def audit_table(
    path: Path,
    root: Path,
    chunksize: int,
    max_unique_ids: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = {
        "relative_path": rel(path, root),
        "file_type": path.suffix.lower().lstrip("."),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "rows": 0,
        "columns": 0,
        "column_names": "",
        "id_column": "",
        "unique_ids": None,
        "label_column": "",
        "label_counts": "",
        "sensor_column": "",
        "sensor_counts": "",
        "source_column": "",
        "source_counts": "",
        "time_column": "",
        "time_min": "",
        "time_max": "",
        "latitude_column": "",
        "longitude_column": "",
        "valid_coordinate_rows": None,
        "parse_error": "",
    }
    flag_rows: list[dict[str, Any]] = []

    columns: list[str] | None = None
    selected: dict[str, str | None] = {}
    unique_ids: set[str] = set()
    unique_id_overflow = False

    label_counts = Counter()
    sensor_counts = Counter()
    source_counts = Counter()
    valid_coordinates = 0
    time_min = None
    time_max = None

    dynamic_counters: dict[str, Counter] = defaultdict(Counter)
    dynamic_true_counts: Counter = Counter()
    dynamic_nonmissing_counts: Counter = Counter()

    try:
        for chunk in read_table_chunks(path, chunksize):
            if columns is None:
                columns = [str(c) for c in chunk.columns]
                selected = {
                    "id": pick(columns, ID_ALIASES),
                    "label": pick(columns, LABEL_ALIASES),
                    "sensor": pick(columns, SENSOR_ALIASES),
                    "source": pick(columns, SOURCE_ALIASES),
                    "time": pick(columns, TIME_ALIASES),
                    "lat": pick(columns, LAT_ALIASES),
                    "lon": pick(columns, LON_ALIASES),
                }

            summary["rows"] += len(chunk)

            id_col = selected.get("id")
            if id_col and not unique_id_overflow:
                values = chunk[id_col].dropna().astype(str)
                unique_ids.update(values.tolist())
                if len(unique_ids) > max_unique_ids:
                    unique_id_overflow = True
                    unique_ids.clear()

            for key, counter in (
                ("label", label_counts),
                ("sensor", sensor_counts),
                ("source", source_counts),
            ):
                col = selected.get(key)
                if col:
                    values = chunk[col].where(chunk[col].notna(), "<missing>")
                    counter.update(values.astype(str).tolist())

            time_col = selected.get("time")
            if time_col:
                parsed = pd.to_datetime(chunk[time_col], errors="coerce", utc=True).dropna()
                if not parsed.empty:
                    chunk_min = parsed.min()
                    chunk_max = parsed.max()
                    time_min = chunk_min if time_min is None else min(time_min, chunk_min)
                    time_max = chunk_max if time_max is None else max(time_max, chunk_max)

            lat_col = selected.get("lat")
            lon_col = selected.get("lon")
            if lat_col and lon_col:
                lat = pd.to_numeric(chunk[lat_col], errors="coerce")
                lon = pd.to_numeric(chunk[lon_col], errors="coerce")
                valid_coordinates += int(
                    (lat.between(-90, 90) & lon.between(-180, 180)).sum()
                )

            for col in chunk.columns:
                name = str(col)
                low = name.lower()
                if (
                    low == "status"
                    or low.endswith("_status")
                    or low.endswith("_qa_pass")
                    or low.endswith("_downloaded")
                    or low in {
                        "qa_pass", "model_ready", "strict_model_ready",
                        "qa_model_ready", "confirmed_records",
                        "candidate_negative_records",
                    }
                ):
                    values = chunk[name]
                    dynamic_nonmissing_counts[name] += int(values.notna().sum())
                    dynamic_counters[name].update(
                        values.where(values.notna(), "<missing>").astype(str).tolist()
                    )
                    for value in values:
                        if as_bool(value) is True:
                            dynamic_true_counts[name] += 1

        columns = columns or []
        summary["columns"] = len(columns)
        summary["column_names"] = " | ".join(columns)
        for key in ("id", "label", "sensor", "source", "time", "lat", "lon"):
            summary[f"{key if key not in {'lat','lon'} else {'lat':'latitude','lon':'longitude'}[key]}_column"] = selected.get(key) or ""

        summary["unique_ids"] = (
            f">{max_unique_ids}"
            if unique_id_overflow
            else (len(unique_ids) if selected.get("id") else None)
        )
        summary["label_counts"] = counter_text(label_counts)
        summary["sensor_counts"] = counter_text(sensor_counts)
        summary["source_counts"] = counter_text(source_counts)
        summary["time_min"] = "" if time_min is None else time_min.isoformat()
        summary["time_max"] = "" if time_max is None else time_max.isoformat()
        summary["valid_coordinate_rows"] = (
            valid_coordinates
            if selected.get("lat") and selected.get("lon")
            else None
        )

        for column in sorted(dynamic_counters):
            flag_rows.append({
                "relative_path": rel(path, root),
                "column": column,
                "nonmissing_count": dynamic_nonmissing_counts[column],
                "true_count": dynamic_true_counts[column],
                "value_counts": counter_text(dynamic_counters[column]),
            })

    except Exception as exc:
        summary["parse_error"] = f"{type(exc).__name__}: {exc}"

    return summary, flag_rows


def audit_raster(path: Path, root: Path, deep: bool) -> dict[str, Any]:
    row = {
        "relative_path": rel(path, root),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "readable": False,
        "bands": None,
        "height": None,
        "width": None,
        "crs": "",
        "res_x": None,
        "res_y": None,
        "dtype": "",
        "all_zero": None,
        "has_nan": None,
        "error": "",
    }

    if rasterio is None:
        row["error"] = "rasterio not installed; metadata not inspected"
        return row

    try:
        with rasterio.open(path) as src:
            row.update({
                "readable": True,
                "bands": src.count,
                "height": src.height,
                "width": src.width,
                "crs": str(src.crs) if src.crs else "",
                "res_x": src.res[0],
                "res_y": src.res[1],
                "dtype": "|".join(src.dtypes),
            })
            if deep:
                data = src.read(masked=True)
                values = data.compressed()
                if values.size == 0:
                    row["all_zero"] = True
                    row["has_nan"] = False
                else:
                    row["all_zero"] = bool(np.all(values == 0)) if np is not None else None
                    row["has_nan"] = (
                        bool(np.isnan(values).any())
                        if np is not None and np.issubdtype(values.dtype, np.floating)
                        else False
                    )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"

    return row


def sentinel2_temporal_detail(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame([{
            "metric": "manifest_present",
            "value": False,
            "detail": str(path),
        }])

    df = pd.read_csv(path, low_memory=False)
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: Any, detail: str = ""):
        rows.append({"metric": metric, "value": value, "detail": detail})

    add("manifest_present", True, str(path))
    add("records", len(df))

    for col in (
        "confirmed_records", "candidate_negative_records",
        "all_three_downloaded", "all_three_qa_pass",
        "t0_qa_pass", "t90_qa_pass", "t360_qa_pass",
    ):
        if col in df.columns:
            add(col, int(sum(as_bool(v) is True for v in df[col])))

    for prefix in ("t0", "t90", "t360"):
        status_col = f"{prefix}_status"
        scene_col = f"{prefix}_scene_id"
        time_col = f"{prefix}_scene_time_utc"

        if status_col in df.columns:
            counts = df[status_col].fillna("<missing>").astype(str).value_counts()
            for status, count in counts.items():
                add(f"{status_col}:{status}", int(count))

            already = df[status_col].astype(str).eq("already_exists")
            add(f"{prefix}_already_exists", int(already.sum()))

            if scene_col in df.columns:
                missing_scene = already & df[scene_col].isna()
                add(
                    f"{prefix}_already_exists_missing_scene_id",
                    int(missing_scene.sum()),
                    "Resume manifest metadata is incomplete when this is >0.",
                )

            if time_col in df.columns:
                missing_time = already & df[time_col].isna()
                add(
                    f"{prefix}_already_exists_missing_scene_time",
                    int(missing_time.sum()),
                    "Resume manifest metadata is incomplete when this is >0.",
                )

    if "record_id" in df.columns:
        add("unique_record_ids", int(df["record_id"].nunique(dropna=True)))

    return pd.DataFrame(rows)


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Project root does not exist: {root}")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.scan_dir:
        scan_roots = [
            (root / item).resolve()
            if not Path(item).is_absolute()
            else Path(item).resolve()
            for item in args.scan_dir
        ]
    else:
        candidates = [root / "data", root / "outputs", root / "results"]
        scan_roots = [p for p in candidates if p.exists()]
        if not scan_roots:
            scan_roots = [root]

    files = sorted(iter_files(scan_roots))

    all_rows = []
    table_rows = []
    status_rows = []
    raster_rows = []

    for index, path in enumerate(files, start=1):
        stat = path.stat()
        all_rows.append({
            "relative_path": rel(path, root),
            "extension": path.suffix.lower(),
            "size_mb": round(stat.st_size / 1024 / 1024, 3),
            "modified_utc": iso_time(stat.st_mtime),
        })

        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv", ".parquet", ".geojson", ".json"}:
            summary, flags = audit_table(
                path,
                root,
                args.chunksize,
                args.max_unique_ids,
            )
            table_rows.append(summary)
            status_rows.extend(flags)
        elif suffix in {".tif", ".tiff"}:
            raster_rows.append(audit_raster(path, root, args.deep_raster))

        if index % 100 == 0:
            print(f"Scanned {index}/{len(files)} files...", flush=True)

    known_rows = []
    for name, relative in KNOWN_DATASETS:
        path = root / relative
        known_rows.append({
            "dataset": name,
            "relative_path": relative,
            "exists": path.exists(),
            "size_mb": (
                round(path.stat().st_size / 1024 / 1024, 3)
                if path.exists()
                else None
            ),
        })

    temporal_path = root / "data/methaneair_full/sentinel2_temporal_manifest.csv"
    temporal_df = sentinel2_temporal_detail(temporal_path)

    write_csv(output_dir / "001_all_data_files.csv", all_rows)
    write_csv(output_dir / "002_tabular_inventory.csv", table_rows)
    write_csv(output_dir / "003_status_qa_inventory.csv", status_rows)
    write_csv(output_dir / "004_raster_inventory.csv", raster_rows)
    write_csv(output_dir / "005_known_dataset_presence.csv", known_rows)
    write_csv(output_dir / "006_sentinel2_temporal_audit.csv", temporal_df)

    extension_counts = Counter(row["extension"] for row in all_rows)
    readable_rasters = sum(row.get("readable") is True for row in raster_rows)
    raster_errors = sum(bool(row.get("error")) for row in raster_rows)
    table_errors = sum(bool(row.get("parse_error")) for row in table_rows)

    report_lines = [
        "METHANEFUSE LIVE PROJECT INVENTORY AUDIT",
        "=" * 52,
        f"Project root: {root}",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Scanned roots: {', '.join(str(p) for p in scan_roots)}",
        "",
        f"Data files found: {len(all_rows)}",
        f"Tabular files inspected: {len(table_rows)}",
        f"Raster files inspected: {len(raster_rows)}",
        f"Readable rasters: {readable_rasters}",
        f"Tabular parse errors: {table_errors}",
        f"Raster errors/warnings: {raster_errors}",
        "",
        "File extensions:",
        counter_text(extension_counts),
        "",
        "Known dataset presence:",
    ]
    for row in known_rows:
        report_lines.append(
            f"- {'FOUND' if row['exists'] else 'MISSING'}: "
            f"{row['dataset']} -> {row['relative_path']}"
        )

    report_lines.extend([
        "",
        "Sentinel-2 temporal manifest audit:",
    ])
    for _, row in temporal_df.iterrows():
        report_lines.append(
            f"- {row['metric']}: {row['value']}"
            + (f" ({row['detail']})" if str(row["detail"]) not in {"", "nan"} else "")
        )

    report_path = output_dir / "007_audit_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Optional Excel workbook
    excel_path = output_dir / "project_inventory_audit.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pd.DataFrame(all_rows).to_excel(writer, sheet_name="All Files", index=False)
            pd.DataFrame(table_rows).to_excel(writer, sheet_name="Tabular Inventory", index=False)
            pd.DataFrame(status_rows).to_excel(writer, sheet_name="Status QA", index=False)
            pd.DataFrame(raster_rows).to_excel(writer, sheet_name="Rasters", index=False)
            pd.DataFrame(known_rows).to_excel(writer, sheet_name="Known Datasets", index=False)
            temporal_df.to_excel(writer, sheet_name="S2 Temporal Audit", index=False)
        excel_created = True
    except Exception as exc:
        excel_created = False
        print(f"Excel not created: {type(exc).__name__}: {exc}", file=sys.stderr)

    zip_path = output_dir / "project_inventory_audit.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.iterdir()):
            if path == zip_path or not path.is_file():
                continue
            archive.write(path, arcname=path.name)

    print("")
    print("Audit complete.")
    print(f"Report: {report_path}")
    print(f"Bundle: {zip_path}")
    if excel_created:
        print(f"Excel: {excel_path}")
    print("")
    print("Upload project_inventory_audit.zip to ChatGPT for a verified final inventory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
