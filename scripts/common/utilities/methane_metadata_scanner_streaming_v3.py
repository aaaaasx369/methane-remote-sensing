#!/usr/bin/env python3
"""
methane_metadata_scanner.py

READ-ONLY metadata inventory scanner for methane research folders.

What it does
------------
- Recursively scans a chosen root folder.
- Looks only at metadata-like files:
  CSV, TSV, TXT, JSON, GeoJSON, Parquet, XLSX, XLS.
- Extracts file path, size, row count (when practical), columns, and likely
  methane/event metadata fields such as latitude, longitude, time, event ID,
  label, emission rate, plume/source/site/sensor.
- Does NOT modify, move, rename, or delete any source file.
- Does NOT read TIFF/NPZ/NetCDF imagery contents.

Outputs
-------
1. methane_dataset_inventory.csv   -> all scanned metadata tables
2. methane_candidate_tables.csv    -> likely useful methane/event tables only
3. methane_scan_summary.txt        -> compact summary\n4. methane_scan_checkpoint.csv     -> progress saved every 250 files

Usage
-----
python3 methane_metadata_scanner_multi.py "/path/to/local/root" "/Volumes/path/to/cloud/root"

Optional:
python3 methane_metadata_scanner_multi.py "/path/to/local/root" "/Volumes/path/to/cloud/root" --out "/path/to/output"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SUPPORTED_EXTS = {
    ".csv", ".tsv", ".txt",
    ".json", ".geojson",
    ".parquet",
    ".xlsx", ".xls",
}

# Skip obvious environments/caches/build artifacts.
SKIP_DIR_NAMES = {
    ".git", ".svn", ".hg",
    ".venv", "venv", "env",
    "__pycache__", "node_modules",
    ".cache", ".idea", ".vscode",
    "site-packages",
}

# Conservative aliases. We score matches rather than forcing one schema.
FIELD_PATTERNS = {
    "latitude": [
        r"^lat$", r"^latitude$", r"source_lat", r"plume_lat", r"event_lat",
        r"centroid_lat", r"center_lat", r"release_lat", r"facility_lat",
    ],
    "longitude": [
        r"^lon$", r"^lng$", r"^long$", r"^longitude$", r"source_lon",
        r"plume_lon", r"event_lon", r"centroid_lon", r"center_lon",
        r"release_lon", r"facility_lon",
    ],
    "time": [
        r"time", r"date", r"datetime", r"timestamp", r"acquisition",
        r"observation", r"event_time", r"release_time", r"sensing",
        r"start_time", r"end_time", r"utc",
    ],
    "event_id": [
        r"event.*id", r"record.*id", r"sample.*id", r"plume.*id",
        r"detection.*id", r"source.*id", r"release.*id", r"scene.*id",
        r"^id$", r"uuid", r"granule",
    ],
    "label": [
        r"^label$", r"class", r"target", r"positive", r"negative",
        r"detection", r"truth", r"tc_classification", r"plume_present",
    ],
    "emission": [
        r"emission", r"release.*rate", r"flow.*rate", r"kg.?h",
        r"kg.?hr", r"kg.?hour", r"rate_kg", r"ch4.*rate",
        r"methane.*rate", r"flux",
    ],
    "plume": [
        r"plume", r"enhancement", r"ch4", r"methane",
    ],
    "site": [
        r"site", r"facility", r"location", r"region",
    ],
    "sensor": [
        r"sensor", r"satellite", r"mission", r"platform", r"instrument",
    ],
    "quality": [
        r"qa", r"quality", r"cloud", r"scl", r"clear", r"usable",
    ],
}

METHANE_FILENAME_HINTS = [
    "methane", "ch4", "plume", "carbonmapper", "carbon_mapper",
    "methanesat", "methane_sat", "methaneair", "methane_air",
    "ghgsat", "enmap", "emit", "aviris", "mars", "controlled",
    "release", "sentinel", "landsat",
]


def norm_col(name: object) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-/\.]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def matched_columns(columns: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for field, patterns in FIELD_PATTERNS.items():
        hits = []
        for original in columns:
            n = norm_col(original)
            if any(re.search(p, n, flags=re.I) for p in patterns):
                hits.append(original)
        out[field] = hits
    return out


def relevance_score(path: Path, matches: Dict[str, List[str]]) -> int:
    score = 0

    # Core event matching fields get highest weight.
    score += 3 if matches["latitude"] else 0
    score += 3 if matches["longitude"] else 0
    score += 3 if matches["time"] else 0
    score += 2 if matches["event_id"] else 0
    score += 2 if matches["label"] else 0
    score += 2 if matches["emission"] else 0
    score += 1 if matches["plume"] else 0
    score += 1 if matches["site"] else 0
    score += 1 if matches["sensor"] else 0
    score += 1 if matches["quality"] else 0

    p = str(path).lower()
    if any(hint in p for hint in METHANE_FILENAME_HINTS):
        score += 2

    return score


def candidate_reason(matches: Dict[str, List[str]], score: int) -> str:
    has_coords = bool(matches["latitude"] and matches["longitude"])
    has_time = bool(matches["time"])
    if has_coords and has_time:
        return "HAS_COORDS_AND_TIME"
    if has_coords:
        return "HAS_COORDS"
    if has_time and score >= 5:
        return "HAS_TIME_AND_METHANE_FIELDS"
    if score >= 6:
        return "HIGH_RELEVANCE_SCORE"
    return "LOW_RELEVANCE"


def safe_join(values: List[str]) -> str:
    return " | ".join(str(v) for v in values)


def read_delimited_header(path: Path, ext: str) -> Tuple[List[str], Optional[int], str]:
    """
    Reads header and counts data rows streaming through the file.
    This may take a little time for very large CSVs but uses little memory.
    """
    delimiter = "\t" if ext == ".tsv" else None

    # Read a sample for delimiter sniffing.
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        sample = f.read(65536)

    if not sample.strip():
        return [], 0, "EMPTY"

    if delimiter is None:
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ","

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return [], 0, "EMPTY"

        header = [str(x).strip() for x in header]
        row_count = sum(1 for _ in reader)

    return header, row_count, f"DELIMITER={repr(delimiter)}"


def read_json_schema(path: Path) -> Tuple[List[str], Optional[int], str]:
    size_mb = path.stat().st_size / (1024 * 1024)

    # Prevent accidental huge in-memory parse.
    if size_mb > 200:
        return [], None, "SKIPPED_JSON_OVER_200MB"

    with path.open("r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        features = data.get("features") or []
        cols = set()
        for feat in features[:100]:
            if isinstance(feat, dict):
                props = feat.get("properties")
                if isinstance(props, dict):
                    cols.update(map(str, props.keys()))
                geom = feat.get("geometry")
                if isinstance(geom, dict):
                    # Geometry itself may provide spatial info even without lat/lon cols.
                    cols.add("__geojson_geometry__")
        return sorted(cols), len(features), "GEOJSON_FEATURECOLLECTION"

    if isinstance(data, list):
        cols = set()
        for item in data[:100]:
            if isinstance(item, dict):
                cols.update(map(str, item.keys()))
        return sorted(cols), len(data), "JSON_LIST"

    if isinstance(data, dict):
        return sorted(map(str, data.keys())), 1, "JSON_OBJECT"

    return [], None, f"JSON_{type(data).__name__.upper()}"


def read_with_pandas(path: Path, ext: str) -> Tuple[List[str], Optional[int], str]:
    try:
        import pandas as pd
    except Exception:
        return [], None, "PANDAS_NOT_AVAILABLE"

    if ext == ".parquet":
        try:
            df = pd.read_parquet(path)
            return [str(c) for c in df.columns], len(df), "PARQUET_OK"
        except Exception as e:
            return [], None, f"PARQUET_ERROR:{type(e).__name__}:{e}"

    if ext in {".xlsx", ".xls"}:
        try:
            # Read all sheet headers/counts, but expose a union of columns.
            xl = pd.ExcelFile(path)
            union_cols = []
            total_rows = 0
            seen = set()
            sheet_notes = []
            for sheet in xl.sheet_names:
                try:
                    df = pd.read_excel(path, sheet_name=sheet)
                    total_rows += len(df)
                    for c in df.columns:
                        s = str(c)
                        if s not in seen:
                            seen.add(s)
                            union_cols.append(s)
                    sheet_notes.append(f"{sheet}:{len(df)}")
                except Exception as e:
                    sheet_notes.append(f"{sheet}:ERR_{type(e).__name__}")
            return union_cols, total_rows, "SHEETS=" + ";".join(sheet_notes)
        except Exception as e:
            return [], None, f"EXCEL_ERROR:{type(e).__name__}:{e}"

    return [], None, "UNSUPPORTED_PANDAS_TYPE"


def scan_file(path: Path) -> Dict[str, object]:
    ext = path.suffix.lower()

    columns: List[str] = []
    row_count: Optional[int] = None
    parse_note = ""
    status = "OK"
    size_bytes: Optional[int] = None

    # SMB/network folders can briefly disappear between directory listing and access.
    # Retry transient failures a few times before recording a failure.
    stat_error = None
    for attempt in range(1, 4):
        try:
            size_bytes = path.stat().st_size
            stat_error = None
            break
        except (FileNotFoundError, OSError) as e:
            stat_error = e
            if attempt < 3:
                time.sleep(1.0 * attempt)
        except PermissionError as e:
            stat_error = e
            break

    if stat_error is not None:
        if isinstance(stat_error, PermissionError):
            status = "PERMISSION_DENIED"
            parse_note = "PermissionError while reading file metadata"
        elif isinstance(stat_error, FileNotFoundError):
            status = "FILE_NOT_FOUND_DURING_SCAN"
            parse_note = "File unavailable after 3 SMB retries"
        else:
            status = "STAT_ERROR"
            parse_note = f"{type(stat_error).__name__}: {stat_error}"

    if status == "OK":
        try:
            if ext in {".csv", ".tsv", ".txt"}:
                columns, row_count, parse_note = read_delimited_header(path, ext)
            elif ext in {".json", ".geojson"}:
                columns, row_count, parse_note = read_json_schema(path)
            elif ext in {".parquet", ".xlsx", ".xls"}:
                columns, row_count, parse_note = read_with_pandas(path, ext)
            else:
                status = "UNSUPPORTED"
        except FileNotFoundError:
            status = "FILE_NOT_FOUND_DURING_SCAN"
            parse_note = "File disappeared or SMB mount temporarily did not expose it"
        except PermissionError:
            status = "PERMISSION_DENIED"
            parse_note = "PermissionError"
        except OSError as e:
            status = "IO_ERROR"
            parse_note = f"{type(e).__name__}: {e}"
        except Exception as e:
            status = "PARSE_ERROR"
            parse_note = f"{type(e).__name__}: {e}"

    matches = matched_columns(columns)

    # GeoJSON geometry can satisfy the spatial requirement even without explicit lat/lon fields.
    geojson_geometry = "__geojson_geometry__" in columns
    if geojson_geometry:
        if not matches["latitude"]:
            matches["latitude"] = ["<GeoJSON geometry>"]
        if not matches["longitude"]:
            matches["longitude"] = ["<GeoJSON geometry>"]

    score = relevance_score(path, matches)
    reason = candidate_reason(matches, score)

    return {
        "file_path": str(path),
        "file_name": path.name,
        "extension": ext,
        "size_mb": "" if size_bytes is None else round(size_bytes / (1024 * 1024), 3),
        "row_count": "" if row_count is None else row_count,
        "column_count": len(columns),
        "columns": safe_join(columns),
        "latitude_columns": safe_join(matches["latitude"]),
        "longitude_columns": safe_join(matches["longitude"]),
        "time_columns": safe_join(matches["time"]),
        "event_id_columns": safe_join(matches["event_id"]),
        "label_columns": safe_join(matches["label"]),
        "emission_columns": safe_join(matches["emission"]),
        "plume_columns": safe_join(matches["plume"]),
        "site_columns": safe_join(matches["site"]),
        "sensor_columns": safe_join(matches["sensor"]),
        "quality_columns": safe_join(matches["quality"]),
        "relevance_score": score,
        "candidate_reason": reason,
        "status": status,
        "parse_note": parse_note,
    }


def iter_metadata_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        # Modify traversal list in place so skipped dirs are not descended into.
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIR_NAMES
            and not d.startswith(".")
        ]

        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in SUPPORTED_EXTS:
                yield p


def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        # Still create a valid empty file with standard columns.
        fieldnames = [
            "file_path", "file_name", "extension", "size_mb", "row_count",
            "column_count", "columns", "latitude_columns", "longitude_columns",
            "time_columns", "event_id_columns", "label_columns",
            "emission_columns", "plume_columns", "site_columns",
            "sensor_columns", "quality_columns", "relevance_score",
            "candidate_reason", "status", "parse_note",
        ]
    else:
        fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="SMB-safe read-only methane metadata inventory scanner."
    )
    parser.add_argument("roots", nargs="+", help="One or more top-level folders containing your research data")
    parser.add_argument(
        "--out",
        default=None,
        help="Output folder. Default: current working directory / methane_scan_output",
    )
    args = parser.parse_args()

    roots = [Path(r).expanduser().resolve() for r in args.roots]
    bad_roots = [r for r in roots if not r.exists() or not r.is_dir()]
    if bad_roots:
        for r in bad_roots:
            print(f"ERROR: folder not found or not mounted: {r}", file=sys.stderr)
        sys.exit(2)

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (Path.cwd() / "methane_scan_output").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Root folders:")
    for root in roots:
        print(f"  - {root}")
    print("Scanning immediately as files are discovered (SMB-safe, read-only)")

    rows: List[Dict[str, object]] = []
    checkpoint_path = out_dir / "methane_scan_checkpoint.csv"
    seen = set()
    i = 0

    for root in roots:
        print(f"\n=== ROOT: {root} ===")
        for path in iter_metadata_files(root):
            # Avoid resolving the network path before access; resolve() can itself
            # trigger failures on unstable SMB mounts.
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            i += 1
            print(f"[{i}] {path}")

            try:
                row = scan_file(path)
            except Exception as e:
                row = {
                    "file_path": str(path),
                    "file_name": path.name,
                    "extension": path.suffix.lower(),
                    "size_mb": "",
                    "row_count": "",
                    "column_count": 0,
                    "columns": "",
                    "latitude_columns": "",
                    "longitude_columns": "",
                    "time_columns": "",
                    "event_id_columns": "",
                    "label_columns": "",
                    "emission_columns": "",
                    "plume_columns": "",
                    "site_columns": "",
                    "sensor_columns": "",
                    "quality_columns": "",
                    "relevance_score": 0,
                    "candidate_reason": "LOW_RELEVANCE",
                    "status": "UNEXPECTED_SCAN_ERROR",
                    "parse_note": f"{type(e).__name__}: {e}",
                }

            row["scan_root"] = str(root)
            rows.append(row)

            if i % 100 == 0:
                write_csv(checkpoint_path, rows)
                print(f"  checkpoint saved -> {checkpoint_path}")

    print(f"\nMetadata-like files encountered: {i}")

    # Final unsorted checkpoint before producing normal outputs.
    write_csv(checkpoint_path, rows)

    # Most useful first.
    rows.sort(
        key=lambda r: (
            -int(r["relevance_score"]),
            str(r["file_path"]).lower(),
        )
    )

    # Candidate = has coords, or score is high enough to deserve review.
    candidates = [
        r for r in rows
        if r["candidate_reason"] != "LOW_RELEVANCE"
        and r["status"] == "OK"
    ]

    inventory_path = out_dir / "methane_dataset_inventory.csv"
    candidates_path = out_dir / "methane_candidate_tables.csv"
    summary_path = out_dir / "methane_scan_summary.txt"

    write_csv(inventory_path, rows)
    write_csv(candidates_path, candidates)

    status_counts = Counter(str(r["status"]) for r in rows)
    ext_counts = Counter(str(r["extension"]) for r in rows)
    reason_counts = Counter(str(r["candidate_reason"]) for r in candidates)

    exact_ready = [
        r for r in candidates
        if r["latitude_columns"] and r["longitude_columns"] and r["time_columns"]
    ]

    total_known_rows = sum(
        int(r["row_count"])
        for r in candidates
        if str(r["row_count"]).isdigit()
    )

    summary = [
        "METHANE METADATA SCAN SUMMARY",
        "=" * 72,
        "Root folders:",
        *[f"  - {r}" for r in roots],
        f"Metadata files scanned    : {len(rows)}",
        f"Candidate methane tables  : {len(candidates)}",
        f"Coords + time tables      : {len(exact_ready)}",
        f"Known rows in candidates  : {total_known_rows}",
        "",
        "Status counts:",
    ]
    for k, v in sorted(status_counts.items()):
        summary.append(f"  {k:24s} {v}")

    summary += ["", "Extension counts:"]
    for k, v in sorted(ext_counts.items()):
        summary.append(f"  {k:24s} {v}")

    summary += ["", "Candidate reasons:"]
    for k, v in sorted(reason_counts.items()):
        summary.append(f"  {k:24s} {v}")

    summary += [
        "",
        "NEXT STEP",
        "Upload these two small files to ChatGPT:",
        f"  1) {inventory_path.name}",
        f"  2) {candidates_path.name}",
        "",
        f"Checkpoint file           : {checkpoint_path.name}",
        "The scanner is read-only: it does not move, rename, modify, or delete source files.",
    ]

    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print()
    print("\n".join(summary))
    print()
    print(f"Saved: {inventory_path}")
    print(f"Saved: {candidates_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
