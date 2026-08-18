from pathlib import Path
import pandas as pd
import json
import os
from collections import Counter, defaultdict

ROOT = Path("raw")
OUT = Path("stanford_audit")
OUT.mkdir(exist_ok=True)

# ============================================================
# 1. ALL FILE INVENTORY
# ============================================================

all_rows = []

for p in ROOT.rglob("*"):
    if not p.is_file():
        continue

    rel = p.relative_to(ROOT)
    suffix = p.suffix.lower()

    all_rows.append({
        "relative_path": str(rel),
        "filename": p.name,
        "extension": suffix,
        "size_bytes": p.stat().st_size,
        "parent": str(rel.parent),
    })

all_df = pd.DataFrame(all_rows)

if len(all_df):
    all_df = all_df.sort_values("relative_path")

all_df.to_csv(
    OUT / "01_all_file_inventory.csv",
    index=False
)

# ============================================================
# 2. BASIC FILE TYPE COUNTS
# ============================================================

ext_counts = Counter(
    x["extension"] if x["extension"] else "[no extension]"
    for x in all_rows
)

name_categories = Counter()

for r in all_rows:
    name = r["filename"].lower()

    if "releasedata" in name:
        category = "releasedata"
    elif "summary" in name and name.endswith(".csv"):
        category = "summary_csv"
    elif "summary_report" in name:
        category = "summary_report"
    elif "meteor" in name or "weather" in name or "wind" in name:
        category = "meteorology_or_wind"
    elif name.endswith(".csv"):
        category = "other_csv"
    else:
        category = "other"

    name_categories[category] += 1

# ============================================================
# 3. CSV SCHEMA + SAMPLE ROWS
# ============================================================

csv_files = sorted(ROOT.rglob("*.csv"))

schema_rows = []
sample_records = []
errors = []

# column-set grouping
column_sets = defaultdict(list)

for i, p in enumerate(csv_files, 1):

    rel = str(p.relative_to(ROOT))

    try:
        # Only read a few rows.
        # sep=None allows delimiter inference.
        df = pd.read_csv(
            p,
            nrows=5,
            sep=None,
            engine="python",
            encoding_errors="replace"
        )

        cols = [str(c) for c in df.columns]

        lower_name = p.name.lower()

        if "releasedata" in lower_name:
            file_type = "releasedata"
        elif "summary" in lower_name:
            file_type = "summary"
        elif "meteor" in lower_name or "weather" in lower_name or "wind" in lower_name:
            file_type = "meteorology_or_wind"
        else:
            file_type = "other_csv"

        schema_key = " | ".join(cols)
        column_sets[schema_key].append(rel)

        schema_rows.append({
            "relative_path": rel,
            "filename": p.name,
            "file_type": file_type,
            "size_bytes": p.stat().st_size,
            "n_columns": len(cols),
            "columns": json.dumps(cols, ensure_ascii=False),
        })

        # Save first three rows only.
        first_rows = df.head(3).copy()

        first_rows = first_rows.where(
            pd.notnull(first_rows),
            None
        )

        sample_records.append({
            "relative_path": rel,
            "filename": p.name,
            "file_type": file_type,
            "columns": cols,
            "sample_rows": first_rows.to_dict(orient="records")
        })

    except Exception as e:
        errors.append({
            "relative_path": rel,
            "error": repr(e)
        })

    if i % 250 == 0:
        print(f"Scanned {i}/{len(csv_files)} CSVs")

schema_df = pd.DataFrame(schema_rows)

if len(schema_df):
    schema_df.to_csv(
        OUT / "02_csv_schema_inventory.csv",
        index=False
    )

with open(
    OUT / "03_csv_sample_rows.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        sample_records,
        f,
        ensure_ascii=False,
        indent=2,
        default=str
    )

with open(
    OUT / "04_csv_read_errors.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        errors,
        f,
        ensure_ascii=False,
        indent=2
    )

# ============================================================
# 4. UNIQUE COLUMN SCHEMAS
# ============================================================

schema_summary = []

for schema_id, (schema, paths) in enumerate(
    sorted(
        column_sets.items(),
        key=lambda x: len(x[1]),
        reverse=True
    ),
    1
):

    schema_summary.append({
        "schema_id": schema_id,
        "file_count": len(paths),
        "columns": schema,
        "example_file_1": paths[0] if len(paths) >= 1 else "",
        "example_file_2": paths[1] if len(paths) >= 2 else "",
        "example_file_3": paths[2] if len(paths) >= 3 else "",
    })

pd.DataFrame(schema_summary).to_csv(
    OUT / "05_unique_csv_schemas.csv",
    index=False
)

# ============================================================
# 5. DIRECTORY TREE / STRUCTURE
# ============================================================

top_counts = Counter()
second_counts = Counter()

for r in all_rows:
    parts = Path(r["relative_path"]).parts

    if len(parts) >= 1:
        top_counts[parts[0]] += 1

    if len(parts) >= 2:
        second_counts[" / ".join(parts[:2])] += 1

with open(
    OUT / "06_directory_summary.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("TOP LEVEL\n")
    f.write("=" * 80 + "\n")

    for k, v in sorted(top_counts.items()):
        f.write(f"{v:8d}  {k}\n")

    f.write("\nSECOND LEVEL\n")
    f.write("=" * 80 + "\n")

    for k, v in sorted(second_counts.items()):
        f.write(f"{v:8d}  {k}\n")

# ============================================================
# 6. IMPORTANT-FILE SUBSET
# ============================================================

important = []

keywords = [
    "releasedata",
    "summary",
    "release",
    "blank",
    "control",
    "wind",
    "meteor",
    "weather",
    "satellite",
    "overpass",
]

for r in all_rows:
    text = r["relative_path"].lower()

    if any(k in text for k in keywords):
        important.append(r)

pd.DataFrame(important).to_csv(
    OUT / "07_important_files.csv",
    index=False
)

# ============================================================
# 7. HUMAN-READABLE SUMMARY
# ============================================================

releasedata_count = sum(
    1 for p in csv_files
    if "releasedata" in p.name.lower()
)

summary_csv_count = sum(
    1 for p in csv_files
    if "summary" in p.name.lower()
)

total_size = sum(r["size_bytes"] for r in all_rows)

with open(
    OUT / "SUMMARY.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("STANFORD 2024-2025 CONTROLLED RELEASE AUDIT\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Total files        : {len(all_rows)}\n")
    f.write(f"Total size bytes   : {total_size}\n")
    f.write(f"CSV files          : {len(csv_files)}\n")
    f.write(f"releasedata CSVs   : {releasedata_count}\n")
    f.write(f"summary CSVs       : {summary_csv_count}\n")
    f.write(f"CSV read errors    : {len(errors)}\n")
    f.write(f"Unique CSV schemas : {len(column_sets)}\n")

    f.write("\nFILE EXTENSIONS\n")
    f.write("-" * 80 + "\n")

    for k, v in ext_counts.most_common():
        f.write(f"{k:15s}: {v}\n")

    f.write("\nFILE CATEGORIES\n")
    f.write("-" * 80 + "\n")

    for k, v in name_categories.most_common():
        f.write(f"{k:25s}: {v}\n")

print()
print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)
print("Total files       :", len(all_rows))
print("CSV files         :", len(csv_files))
print("releasedata CSVs  :", releasedata_count)
print("summary CSVs      :", summary_csv_count)
print("Unique CSV schemas:", len(column_sets))
print("CSV read errors   :", len(errors))
print()
print("OUTPUT:", OUT)
