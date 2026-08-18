#!/usr/bin/env python3
"""
Read-only methane storage inventory scanner.

Scans one or more filesystem roots and writes:
  <outdir>/file_inventory.csv
  <outdir>/dataset_containers.csv
  <outdir>/summary_by_sensor.csv
  <outdir>/summary_by_extension.csv
  <outdir>/SUMMARY.txt

Examples:
  python scan_methane_storage.py \
    --root "mac=/Users/happydoraaa/methane_release_project" \
    --root "lab_smb=/Volumes/engg-leung" \
    --outdir storage_audit_mac_and_lab

  # On Fir / another server:
  python scan_methane_storage.py \
    --root "fir=/path/to/project/root" \
    --outdir storage_audit_fir
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SENSOR_PATTERNS = [
    ("Sentinel-2", [
        r"\bsentinel[_ -]?2\b", r"\bs2[abc]\b", r"\bS2[ABC]_MSI", r"\bMSIL[12][AC]\b",
        r"\bT\d{2}[A-Z]{3}\b.*(?:B0[1-9]|B1[0-2]|SCL)"
    ]),
    ("Landsat", [
        r"\blandsat\b", r"\bLC0[89]_", r"\bLO0[89]_", r"\bLT0[45]_", r"\bLE07_"
    ]),
    ("EMIT", [r"\bemit\b", r"\bEMIT_L[12][AB]_"]),
    ("EnMAP", [r"\benmap\b", r"\bENMAP\d", r"\bENMAP_"]),
    ("PRISMA", [r"\bprisma\b"]),
    ("Tanager", [r"\btanager\b", r"\bTanager-?1\b"]),
    ("Carbon Mapper", [r"\bcarbon[_ -]?mapper\b", r"\bcarbonmapper\b", r"\bCM_"]),
    ("GHGSat", [r"\bghgsat\b", r"\bGH[2-5]\b"]),
    ("GHOSt", [r"\bghost\b"]),
    ("MethaneSAT", [r"\bmethanesat\b", r"\bmethane[_ -]?sat\b"]),
    ("MethaneAIR", [r"\bmethaneair\b", r"\bmethane[_ -]?air\b"]),
    ("AVIRIS-NG", [r"\baviris\b", r"\bAV\d{2}\d{6}t\d{6}\b"]),
    ("WorldView-3", [r"\bworldview[_ -]?3\b", r"\bwv3\b"]),
    ("Ziyuan-1", [r"\bziyuan\b", r"\bzy1\b"]),
    ("Gaofen-5", [r"\bgaofen\b", r"\bgf5\b"]),
    ("Sentinel-5P", [r"\bsentinel[_ -]?5p\b", r"\bs5p\b", r"\btropomi\b"]),
]

DATA_EXTS = {
    ".tif", ".tiff", ".jp2", ".nc", ".nc4", ".h5", ".hdf", ".hdf5",
    ".he5", ".zarr", ".npz", ".npy", ".csv", ".tsv", ".parquet",
    ".json", ".geojson", ".gpkg", ".shp", ".xlsx", ".xls",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z", ".txt", ".xml",
    ".vrt", ".img", ".dat", ".bin",
}

SKIP_DIR_NAMES = {
    ".git", ".svn", ".hg", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".venv", "venv", "env", "node_modules", ".cache", ".DS_Store",
}

CONTAINER_SUFFIXES = {".SAFE", ".safe", ".zarr", ".gdb"}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", action="append", required=True,
        help='label=/absolute/path ; repeat for multiple roots'
    )
    ap.add_argument("--outdir", default="storage_audit")
    ap.add_argument(
        "--all-files", action="store_true",
        help="Include all file extensions, not only likely data/metadata files."
    )
    return ap.parse_args()

def parse_root(spec: str):
    if "=" not in spec:
        raise ValueError(f"--root must be label=/path, got: {spec}")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Bad root spec: {spec}")
    return label, Path(path).expanduser()

def sensor_hint(path_text: str) -> str:
    hits = []
    for sensor, patterns in SENSOR_PATTERNS:
        if any(re.search(p, path_text, flags=re.I) for p in patterns):
            hits.append(sensor)
    if not hits:
        return "Unknown"
    # Prefer specific instruments over generic path artifacts.
    return "|".join(dict.fromkeys(hits))

def kind_hint(ext: str, name: str) -> str:
    e = ext.lower()
    n = name.lower()
    if e in {".tif", ".tiff", ".jp2", ".img", ".vrt"}:
        return "raster"
    if e in {".nc", ".nc4", ".h5", ".hdf", ".hdf5", ".he5"}:
        return "scientific_array"
    if e in {".npz", ".npy"}:
        return "model_array"
    if e in {".csv", ".tsv", ".parquet", ".xlsx", ".xls"}:
        return "table"
    if e in {".geojson", ".gpkg", ".shp"}:
        return "vector"
    if e in {".json", ".xml", ".txt"}:
        return "metadata_or_text"
    if e in {".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z"}:
        return "archive"
    return "other"

def utc_iso(ts):
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return ""

def main():
    args = parse_args()
    roots = [parse_root(x) for x in args.root]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inv_path = outdir / "file_inventory.csv"
    cont_path = outdir / "dataset_containers.csv"

    fields = [
        "storage_label", "root_path", "relative_path", "absolute_path",
        "filename", "extension", "size_bytes", "mtime_utc",
        "sensor_hint", "kind_hint",
    ]
    cont_fields = [
        "storage_label", "root_path", "relative_path", "absolute_path",
        "container_name", "container_suffix", "sensor_hint",
    ]

    sensor_counts = Counter()
    sensor_bytes = Counter()
    ext_counts = Counter()
    ext_bytes = Counter()
    storage_counts = Counter()
    storage_bytes = Counter()
    containers = []
    errors = []
    total = 0

    with inv_path.open("w", newline="", encoding="utf-8-sig") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()

        for label, root in roots:
            print("=" * 80)
            print("STORAGE:", label)
            print("ROOT   :", root)
            print("=" * 80)

            if not root.exists():
                errors.append((label, str(root), "ROOT_NOT_FOUND"))
                print("ROOT NOT FOUND; skipped.")
                continue

            for cur, dirs, files in os.walk(root, topdown=True, followlinks=False):
                # Prune obvious code/env/cache directories.
                kept = []
                for d in dirs:
                    if d in SKIP_DIR_NAMES:
                        continue
                    p = Path(cur) / d
                    if p.suffix in CONTAINER_SUFFIXES:
                        try:
                            rel = p.relative_to(root)
                        except Exception:
                            rel = p
                        containers.append({
                            "storage_label": label,
                            "root_path": str(root),
                            "relative_path": str(rel),
                            "absolute_path": str(p),
                            "container_name": p.name,
                            "container_suffix": p.suffix,
                            "sensor_hint": sensor_hint(str(rel)),
                        })
                        # Don't descend into huge logical containers in first-pass inventory.
                        continue
                    kept.append(d)
                dirs[:] = kept

                for fn in files:
                    p = Path(cur) / fn
                    ext = p.suffix.lower()
                    if (not args.all_files) and ext not in DATA_EXTS:
                        continue

                    try:
                        st = p.stat()
                        rel = p.relative_to(root)
                        sensor = sensor_hint(str(rel))
                        row = {
                            "storage_label": label,
                            "root_path": str(root),
                            "relative_path": str(rel),
                            "absolute_path": str(p),
                            "filename": p.name,
                            "extension": ext,
                            "size_bytes": st.st_size,
                            "mtime_utc": utc_iso(st.st_mtime),
                            "sensor_hint": sensor,
                            "kind_hint": kind_hint(ext, p.name),
                        }
                        writer.writerow(row)
                        total += 1
                        storage_counts[label] += 1
                        storage_bytes[label] += st.st_size
                        sensor_counts[sensor] += 1
                        sensor_bytes[sensor] += st.st_size
                        ext_counts[ext or "[none]"] += 1
                        ext_bytes[ext or "[none]"] += st.st_size

                        if total % 10000 == 0:
                            print(f"Scanned {total:,} candidate files...")
                    except Exception as e:
                        errors.append((label, str(p), repr(e)))

    with cont_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cont_fields)
        w.writeheader()
        w.writerows(containers)

    def write_summary_csv(path, counts, bytes_map, key_name):
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=[key_name, "file_count", "size_bytes"])
            w.writeheader()
            for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                w.writerow({
                    key_name: k,
                    "file_count": n,
                    "size_bytes": bytes_map[k],
                })

    write_summary_csv(outdir / "summary_by_sensor.csv", sensor_counts, sensor_bytes, "sensor_hint")
    write_summary_csv(outdir / "summary_by_extension.csv", ext_counts, ext_bytes, "extension")
    write_summary_csv(outdir / "summary_by_storage.csv", storage_counts, storage_bytes, "storage_label")

    with (outdir / "scan_errors.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["storage_label", "path", "error"])
        w.writerows(errors)

    with (outdir / "SUMMARY.txt").open("w", encoding="utf-8") as f:
        f.write("Methane storage inventory audit\n")
        f.write("=" * 72 + "\n")
        f.write(f"Candidate files: {total:,}\n")
        f.write(f"Logical containers (.SAFE/.zarr/.gdb): {len(containers):,}\n")
        f.write(f"Errors: {len(errors):,}\n\n")
        f.write("BY STORAGE\n")
        for k, n in sorted(storage_counts.items()):
            f.write(f"{k:24s} files={n:10,d} bytes={storage_bytes[k]:15,d}\n")
        f.write("\nBY SENSOR HINT\n")
        for k, n in sensor_counts.most_common():
            f.write(f"{k:30s} files={n:10,d} bytes={sensor_bytes[k]:15,d}\n")

    print()
    print("=" * 80)
    print("SCAN COMPLETE")
    print("=" * 80)
    print("Candidate files :", f"{total:,}")
    print("Containers      :", f"{len(containers):,}")
    print("Errors          :", f"{len(errors):,}")
    print("Output          :", outdir)

if __name__ == "__main__":
    main()
