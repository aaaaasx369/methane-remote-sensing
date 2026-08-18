#!/usr/bin/env python3
"""
Merge one or more scan_methane_storage.py inventory folders into a single
cross-storage inventory. Read-only; does not move or delete source data.

Example:
  python merge_methane_inventories.py \
    --audit storage_audit_mac_and_lab \
    --audit storage_audit_fir \
    --outdir storage_audit_merged
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="append", required=True)
    ap.add_argument("--outdir", default="storage_audit_merged")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    merged = []
    for d in args.audit:
        p = Path(d) / "file_inventory.csv"
        if not p.exists():
            print("Missing:", p)
            continue
        with p.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
            merged.extend(rows)
            print(f"Loaded {len(rows):,} rows from {p}")

    if not merged:
        raise SystemExit("No inventory rows loaded.")

    fields = list(merged[0].keys())
    with (outdir / "all_storage_file_inventory.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(merged)

    # Conservative duplicate candidates:
    # same filename + same exact byte size, but do not delete anything.
    groups = {}
    for r in merged:
        key = (r.get("filename", ""), r.get("size_bytes", ""))
        groups.setdefault(key, []).append(r)

    dup_rows = []
    gid = 0
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        gid += 1
        for r in rows:
            x = dict(r)
            x["duplicate_candidate_group"] = gid
            x["duplicate_candidate_count"] = len(rows)
            dup_rows.append(x)

    dup_fields = fields + ["duplicate_candidate_group", "duplicate_candidate_count"]
    with (outdir / "duplicate_candidates_name_size.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=dup_fields)
        w.writeheader()
        w.writerows(dup_rows)

    by_storage = Counter(r.get("storage_label", "") for r in merged)
    by_sensor = Counter(r.get("sensor_hint", "") for r in merged)

    with (outdir / "SUMMARY.txt").open("w", encoding="utf-8") as f:
        f.write("Merged methane storage inventory\n")
        f.write("=" * 72 + "\n")
        f.write(f"Rows: {len(merged):,}\n")
        f.write(f"Duplicate-candidate rows: {len(dup_rows):,}\n\n")
        f.write("BY STORAGE\n")
        for k, v in sorted(by_storage.items()):
            f.write(f"{k:24s}: {v:,}\n")
        f.write("\nBY SENSOR HINT\n")
        for k, v in by_sensor.most_common():
            f.write(f"{k:30s}: {v:,}\n")

    print("Merged rows:", f"{len(merged):,}")
    print("Duplicate candidate rows:", f"{len(dup_rows):,}")
    print("Output:", outdir)

if __name__ == "__main__":
    main()
