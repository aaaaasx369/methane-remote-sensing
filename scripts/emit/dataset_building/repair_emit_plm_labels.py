#!/usr/bin/env python3
"""
Repair step 7 of the EMIT V2 positive/negative downloader.

This script:
1) reads emit_v2_posneg_100/emit_v2_pairs.csv
2) converts CH4PLMMETA names to their matching CH4PLM granule names
3) searches EMITL2BCH4PLM V002
4) downloads the missing positive plume-label granules only

It does NOT redownload the 50 positive or 50 candidate-negative CH4ENH scenes.
"""

from __future__ import annotations
import csv
import sys
from pathlib import Path

try:
    import earthaccess
except ImportError:
    print("ERROR: earthaccess is not installed.")
    print("Run: python3 -m pip install -U earthaccess")
    sys.exit(1)

ROOT = Path("emit_v2_posneg_100")
PAIRS_CSV = ROOT / "emit_v2_pairs.csv"
OUT_DIR = ROOT / "03_positive_plm_labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not PAIRS_CSV.exists():
    raise FileNotFoundError(f"Missing: {PAIRS_CSV}")

print("[1/4] Earthdata login")
try:
    earthaccess.login(strategy="netrc")
    print("Authenticated using ~/.netrc")
except Exception:
    earthaccess.login(strategy="interactive", persist=False)
    print("Authenticated interactively for this run")

print("[2/4] Reading selected positive plume IDs")
meta_names = set()

with PAIRS_CSV.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        raw = (row.get("positive_plm_granules") or "").strip()
        if not raw:
            continue
        for name in raw.split(";"):
            name = name.strip()
            if name:
                meta_names.add(name)

# The original pair table stored metadata filenames:
# EMIT_L2B_CH4PLMMETA_002_<timestamp>_<plumeid>
# The actual plume granule uses:
# EMIT_L2B_CH4PLM_002_<timestamp>_<plumeid>
plm_names = sorted(
    name.replace("EMIT_L2B_CH4PLMMETA_002_", "EMIT_L2B_CH4PLM_002_", 1)
    for name in meta_names
)

print(f"Unique selected plume complexes: {len(plm_names)}")

print("[3/4] Searching exact EMITL2BCH4PLM.002 granules")
results = []
missing = []

for i, name in enumerate(plm_names, 1):
    hits = earthaccess.search_data(
        short_name="EMITL2BCH4PLM",
        version="002",
        granule_name=name,
        count=10,
    )
    if not hits:
        # Some CMR searches are more reliable with a wildcard.
        hits = earthaccess.search_data(
            short_name="EMITL2BCH4PLM",
            version="002",
            granule_name=name + "*",
            count=10,
        )

    if hits:
        results.append(hits[0])
        print(f"[{i}/{len(plm_names)}] FOUND {name}")
    else:
        missing.append(name)
        print(f"[{i}/{len(plm_names)}] MISSING {name}")

print(f"Found: {len(results)}")
print(f"Missing: {len(missing)}")

if missing:
    miss_file = ROOT / "missing_plm_granules.txt"
    miss_file.write_text("\n".join(missing) + "\n", encoding="utf-8")
    print(f"Missing list written to: {miss_file}")

print("[4/4] Downloading missing positive CH4PLM label granules")
if results:
    earthaccess.download(
        results,
        local_path=str(OUT_DIR),
        threads=8,
    )

print("\nDONE")
print(f"Label directory: {OUT_DIR.resolve()}")
print("Run these checks next:")
print(f'  find "{OUT_DIR}" -type f | wc -l')
print(f'  find "{OUT_DIR}" -type f -name "*.tif" | wc -l')
print(f'  find "{OUT_DIR}" -type f -name "*.json" | wc -l')
