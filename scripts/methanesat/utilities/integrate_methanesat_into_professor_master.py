from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import csv
import re
import shutil


# ============================================================
# SETTINGS
# ============================================================

HOME = Path.home()

PROJECT = HOME / "methane_release_project"

SEARCH_ROOTS = [
    PROJECT,
    HOME / "Downloads",
]

OUT = PROJECT / "professor_master_inventory_with_methanesat"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def clean(x):
    if x is None:
        return ""

    x = str(x).strip()

    if x.lower() in {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
    }:
        return ""

    return x


def norm(x):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        clean(x).lower()
    ).strip("_")


def normalized_row(row):
    return {
        norm(k): clean(v)
        for k, v in row.items()
        if k is not None
    }


def first(row, names):
    for name in names:
        value = row.get(norm(name), "")
        if clean(value):
            return clean(value)
    return ""


def find_files(filename):
    matches = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for p in root.rglob(filename):

            text = str(p)

            if "/.venv/" in text:
                continue

            if "/.git/" in text:
                continue

            if "professor_master_inventory_with_methanesat" in text:
                continue

            matches.append(p)

    return sorted(set(matches))


def read_csv(path):
    with path.open(
        newline="",
        encoding="utf-8-sig",
        errors="replace",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        w.writeheader()
        w.writerows(rows)


# ============================================================
# Time extraction
# ============================================================

def extract_datetime_from_text(text):
    """
    Handles e.g.
      2025-03-15T19:31:31Z
      20250315T193131
      2025-03-15
      20250315
    """

    s = clean(text)

    if not s:
        return "", ""

    # ISO datetime
    m = re.search(
        r"(20\d{2})[-_/](\d{2})[-_/](\d{2})"
        r"[T _-](\d{2}):?(\d{2}):?(\d{2})",
        s
    )

    if m:
        y, mo, d, hh, mm, ss = m.groups()

        return (
            f"{y}-{mo}-{d}",
            f"{hh}:{mm}:{ss}"
        )

    # compact YYYYMMDDTHHMMSS
    m = re.search(
        r"(20\d{2})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})",
        s
    )

    if m:
        y, mo, d, hh, mm, ss = m.groups()

        return (
            f"{y}-{mo}-{d}",
            f"{hh}:{mm}:{ss}"
        )

    # date only YYYY-MM-DD
    m = re.search(
        r"(20\d{2})[-_/](\d{2})[-_/](\d{2})",
        s
    )

    if m:
        y, mo, d = m.groups()

        return (
            f"{y}-{mo}-{d}",
            ""
        )

    # compact YYYYMMDD
    m = re.search(
        r"(20\d{2})(\d{2})(\d{2})",
        s
    )

    if m:
        y, mo, d = m.groups()

        try:
            datetime(
                int(y),
                int(mo),
                int(d)
            )

            return (
                f"{y}-{mo}-{d}",
                ""
            )

        except ValueError:
            pass

    return "", ""


def extract_time(row):
    """
    Prefer proper metadata columns.
    Only then try filenames / IDs.
    """

    direct_candidates = [
        "acquisition_time_utc",
        "acquisition_datetime_utc",
        "observation_time_utc",
        "datetime_utc",
        "collection_time_utc",
        "time_coverage_start",
        "start_time",
    ]

    for c in direct_candidates:
        value = row.get(norm(c), "")

        if value:
            date, utc = extract_datetime_from_text(value)

            if date:
                return date, utc

    filename_candidates = [
        "l3_filename",
        "l3_file",
        "l3_path",
        "filename",
        "collection_id",
        "processing_id",
        "granule_id",
    ]

    for c in filename_candidates:
        value = row.get(norm(c), "")

        if value:
            date, utc = extract_datetime_from_text(value)

            if date:
                return date, utc

    return "", ""


# ============================================================
# Locate MethaneSAT manifest
# ============================================================

manifest_candidates = find_files(
    "manifest_model_ready_posneg.csv"
)

if not manifest_candidates:
    raise SystemExit(
        "\nERROR: manifest_model_ready_posneg.csv not found.\n"
        "Put the MethaneSAT dataset inside "
        "~/methane_release_project or ~/Downloads."
    )


print("\nMethaneSAT manifest candidates:")

for p in manifest_candidates:
    print(" ", p)


# Choose the file that looks like the expected 222-row dataset
MANIFEST = None
manifest_rows = None

for p in manifest_candidates:
    try:
        rows = read_csv(p)

        labels = Counter(
            clean(r.get("label"))
            for r in rows
        )

        if (
            len(rows) == 222
            and labels.get("1", 0) == 111
            and labels.get("0", 0) == 111
        ):
            MANIFEST = p
            manifest_rows = rows
            break

    except Exception:
        pass


if MANIFEST is None:

    print(
        "\nWARNING: No exact 222-row / 111+111 manifest found."
    )

    MANIFEST = manifest_candidates[0]
    manifest_rows = read_csv(MANIFEST)


print("\nSelected MethaneSAT manifest:")
print(MANIFEST)

print("Rows:", len(manifest_rows))

label_counts = Counter(
    clean(r.get("label"))
    for r in manifest_rows
)

print("Labels:", dict(label_counts))


# ============================================================
# Locate existing professor master
# ============================================================

master_candidates = find_files(
    "master_site_date_source_inventory.csv"
)

if not master_candidates:
    raise SystemExit(
        "\nERROR: master_site_date_source_inventory.csv not found.\n"
        "Put the professor master CSV somewhere under "
        "~/methane_release_project or ~/Downloads."
    )


# Prefer the project-level professor file,
# but avoid old generated V2 output.
MASTER_OLD = master_candidates[0]

print("\nExisting professor master:")
print(MASTER_OLD)


old_master_raw = read_csv(MASTER_OLD)


# ============================================================
# New unified schema
# ============================================================

FIELDS = [
    "Inventory Level",
    "Site",
    "Location Level",
    "Latitude",
    "Longitude",
    "Date",
    "UTC Time",
    "Label",
    "Label Type",
    "Ground Truth Modality",
    "Sensor",
    "Scene/Observation ID",
    "Release Rate (kg/hr)",
    "Historical/Experiment",
    "Source Dataset",
    "Paper/Reference",
    "Notes",
]


def infer_old_inventory_level(r):
    sensor = clean(
        r.get("Sensor")
        or r.get("sensor")
    )

    if sensor in {
        "Sentinel-2",
        "Landsat-8/9",
        "Carbon Mapper Tanager",
    }:
        return "Sensor Observation"

    return "Ground Truth"


def infer_location_level(site):
    site = clean(site)

    if site in {
        "Casa_Grande",
        "Ehrenberg",
        "MA_site_038",
        "MA_site_043",
        "MA_site_073",
    }:
        return "Exact / named site"

    if site == "MethaneAIR_P1":
        return "Dataset group / needs review"

    if site.startswith("CarbonMapper_"):
        return "Regional search window"

    if site in {
        "Permian (Delaware)",
        "SW Marcellus",
        "NE Marcellus",
        "Haynesville",
        "Eagle Ford",
        "Baltimore Metro",
        "TCCON OK",
    }:
        return "Regional / campaign area"

    if site in {
        "Baltimore",
        "NYC",
        "Test Flight Denver Metro CO",
    }:
        return "Named location / test area"

    return "Named site / region"


def infer_old_modality(r):
    sensor = clean(r.get("Sensor"))

    label_type = clean(
        r.get("Label Type")
    ).lower()

    source = clean(
        r.get("Source Dataset")
    ).lower()

    if sensor == "MethaneAIR":
        return "MethaneAIR observational plume"

    if sensor == "controlled_release":
        return "Controlled release"

    if sensor in {
        "Sentinel-2",
        "Landsat-8/9",
    }:

        if "physical_release" in source:
            return "Controlled-release matched imagery"

        if "plume_reference" in source:
            return "Historical no-known-plume reference"

        return "Sensor observation"

    if sensor == "Carbon Mapper Tanager":
        return "Published plume observation"

    if "controlled" in label_type:
        return "Controlled release"

    return "Ground-truth observation"


# ============================================================
# Convert existing professor master
# ============================================================

combined = []


for raw in old_master_raw:

    r = {
        clean(k): clean(v)
        for k, v in raw.items()
    }

    site = clean(r.get("Site"))

    release_rate = (
        clean(r.get("Release Rate (kg/hr)"))
        or clean(r.get("Release Rate"))
    )

    sensor = clean(r.get("Sensor"))

    # Do not keep "controlled_release" as a sensor
    if sensor == "controlled_release":
        sensor_clean = ""
    else:
        sensor_clean = sensor

    combined.append({

        "Inventory Level":
            infer_old_inventory_level(r),

        "Site":
            site,

        "Location Level":
            infer_location_level(site),

        "Latitude":
            clean(r.get("Latitude")),

        "Longitude":
            clean(r.get("Longitude")),

        "Date":
            clean(r.get("Date")),

        "UTC Time":
            clean(r.get("UTC Time")),

        "Label":
            clean(r.get("Label")),

        "Label Type":
            clean(r.get("Label Type")),

        "Ground Truth Modality":
            infer_old_modality(r),

        "Sensor":
            sensor_clean,

        "Scene/Observation ID":
            clean(r.get("Scene/Observation ID")),

        "Release Rate (kg/hr)":
            release_rate,

        "Historical/Experiment":
            clean(r.get("Historical/Experiment")),

        "Source Dataset":
            clean(r.get("Source Dataset")),

        "Paper/Reference":
            clean(r.get("Paper/Reference")),

        "Notes":
            clean(r.get("Notes")),
    })


# ============================================================
# Convert MethaneSAT manifest
# ============================================================

methanesat_rows = []


for raw in manifest_rows:

    r = normalized_row(raw)

    sample_id = first(
        r,
        [
            "id",
            "sample_id",
            "record_id",
        ]
    )

    label = first(
        r,
        ["label"]
    )

    collection_id = first(
        r,
        [
            "collection_id",
            "collection",
            "collectionid",
        ]
    )

    target_id = first(
        r,
        [
            "target_id",
            "target",
            "site_id",
        ]
    )

    lat = first(
        r,
        [
            "lat",
            "latitude",
            "sample_lat",
            "center_lat",
        ]
    )

    lon = first(
        r,
        [
            "lon",
            "longitude",
            "lng",
            "sample_lon",
            "center_lon",
        ]
    )

    plume_id = first(
        r,
        [
            "plume_id",
            "source_id",
            "l4_source_id",
        ]
    )

    flux = first(
        r,
        [
            "flux_kg_hr",
            "flux_kgh",
            "emission_rate_kg_hr",
            "release_rate_kg_hr",
        ]
    )

    flux_sd = first(
        r,
        [
            "flux_sd",
            "flux_sd_kg_hr",
        ]
    )

    sample_type = first(
        r,
        [
            "sample_type",
            "type",
        ]
    )

    distance = first(
        r,
        [
            "distance_to_nearest_l4_m",
            "nearest_l4_distance_m",
            "distance_l4_m",
        ]
    )

    npz_path = first(
        r,
        [
            "s5p_0_path",
            "npz_path",
            "path",
        ]
    )

    l3_file = first(
        r,
        [
            "l3_filename",
            "l3_file",
            "l3_path",
            "source_l3_file",
        ]
    )

    processing_id = first(
        r,
        [
            "processing_id",
            "processingid",
        ]
    )

    date, utc = extract_time(r)


    # -------------------------
    # Site / target
    # -------------------------

    if target_id:
        site = f"MethaneSAT_{target_id}"
    elif collection_id:
        site = f"MethaneSAT_collection_{collection_id}"
    else:
        site = "MethaneSAT_unassigned_target"


    # -------------------------
    # Physical meaning
    # -------------------------

    if label == "1":

        label_type = (
            "MethaneSAT L4-detected point-source positive"
        )

        modality = (
            "MethaneSAT L4 point-source detection"
        )

        hist_exp = "Observational"

    elif label == "0":

        label_type = (
            "MethaneSAT spatial weak negative"
        )

        modality = (
            "Spatial exclusion from known MethaneSAT "
            "L4 detections (>=10 km)"
        )

        hist_exp = "Spatial reference"

        # Weak negative has no measured source flux
        flux = ""

    else:

        label_type = "MethaneSAT sample"
        modality = "MethaneSAT"
        hist_exp = "Observational"


    notes = [

        "MethaneSAT external binary classification sample",

        "L3 Band 1 = XCH4",

        "480 m x 480 m crop",

        "resized to 224 x 224",

    ]


    if sample_type:
        notes.append(
            f"sample_type={sample_type}"
        )

    if collection_id:
        notes.append(
            f"collection_id={collection_id}"
        )

    if target_id:
        notes.append(
            f"target_id={target_id}"
        )

    if plume_id:
        notes.append(
            f"plume_id={plume_id}"
        )

    if flux_sd:
        notes.append(
            f"flux_sd={flux_sd}"
        )

    if distance:
        notes.append(
            f"distance_to_nearest_L4_m={distance}"
        )

    if l3_file:
        notes.append(
            f"L3={l3_file}"
        )

    if processing_id:
        notes.append(
            f"processing_id={processing_id}"
        )

    if npz_path:
        notes.append(
            f"model_input={npz_path}"
        )

    if not date:
        notes.append(
            "acquisition date not recovered from current manifest"
        )


    row = {

        "Inventory Level":
            "Sensor Observation",

        "Site":
            site,

        "Location Level":
            "MethaneSAT target / sample location",

        "Latitude":
            lat,

        "Longitude":
            lon,

        "Date":
            date,

        "UTC Time":
            utc,

        "Label":
            label,

        "Label Type":
            label_type,

        "Ground Truth Modality":
            modality,

        "Sensor":
            "MethaneSAT L3 XCH4",

        "Scene/Observation ID":
            sample_id,

        "Release Rate (kg/hr)":
            flux,

        "Historical/Experiment":
            hist_exp,

        "Source Dataset":
            (
                "MethaneSAT L3 XCH4 + "
                "L4 point-source detections"
            ),

        "Paper/Reference":
            (
                "MethaneSAT L3 XCH4 and "
                "L4 point-source products"
            ),

        "Notes":
            "; ".join(notes),
    }


    methanesat_rows.append(row)


# ============================================================
# MethaneSAT QA
# ============================================================

ms_labels = Counter(
    r["Label"]
    for r in methanesat_rows
)

print("\n" + "=" * 100)
print("METHANESAT QA")
print("=" * 100)

print(
    "Samples:",
    len(methanesat_rows)
)

print(
    "Positive:",
    ms_labels.get("1", 0)
)

print(
    "Negative:",
    ms_labels.get("0", 0)
)


# Check 10 km rule if distance is available
violations = []

for raw in manifest_rows:

    r = normalized_row(raw)

    label = first(
        r,
        ["label"]
    )

    if label != "0":
        continue

    d = first(
        r,
        [
            "distance_to_nearest_l4_m",
            "nearest_l4_distance_m",
        ]
    )

    if not d:
        continue

    try:
        if float(d) < 10000:
            violations.append(
                first(r, ["id", "sample_id"])
            )

    except ValueError:
        pass


print(
    "Negative <10 km violations:",
    len(violations)
)


if violations:
    print(
        "WARNING:",
        violations[:20]
    )


# ============================================================
# Deduplicate combined inventory
# ============================================================

def dedup_key(r):

    return (
        r["Inventory Level"],
        r["Sensor"],
        r["Scene/Observation ID"],
        r["Latitude"],
        r["Longitude"],
        r["Label"],
    )


seen = set()
combined_dedup = []


for r in combined + methanesat_rows:

    k = dedup_key(r)

    if k in seen:
        continue

    seen.add(k)
    combined_dedup.append(r)


combined_dedup.sort(
    key=lambda r: (
        r["Site"],
        r["Date"],
        r["UTC Time"],
        r["Sensor"],
        r["Scene/Observation ID"],
    )
)


# ============================================================
# Write MethaneSAT-only table
# ============================================================

MS_OUT = (
    OUT
    / "methanesat_sample_inventory.csv"
)

write_csv(
    MS_OUT,
    methanesat_rows,
    FIELDS,
)


# ============================================================
# Write final combined master
# ============================================================

COMBINED_OUT = (
    OUT
    / "all_inventory_with_methanesat.csv"
)

write_csv(
    COMBINED_OUT,
    combined_dedup,
    FIELDS,
)


# ============================================================
# Build Site / Date summary
# ============================================================

summary = defaultdict(
    lambda: {
        "records": 0,
        "dates": set(),
        "positive": 0,
        "negative": 0,
        "sensors": set(),
        "levels": set(),
    }
)


for r in combined_dedup:

    site = r["Site"]

    s = summary[site]

    s["records"] += 1

    if r["Date"]:
        s["dates"].add(
            r["Date"]
        )

    if r["Label"] == "1":
        s["positive"] += 1

    elif r["Label"] == "0":
        s["negative"] += 1

    if r["Sensor"]:
        s["sensors"].add(
            r["Sensor"]
        )

    if r["Inventory Level"]:
        s["levels"].add(
            r["Inventory Level"]
        )


SUMMARY_FIELDS = [
    "Site",
    "Records",
    "Unique Dates",
    "First Date",
    "Last Date",
    "Positive Rows",
    "Negative Rows",
    "Sensors",
    "Inventory Levels",
]


summary_rows = []


for site in sorted(summary):

    s = summary[site]

    dates = sorted(
        s["dates"]
    )

    summary_rows.append({

        "Site":
            site,

        "Records":
            s["records"],

        "Unique Dates":
            len(dates),

        "First Date":
            dates[0] if dates else "",

        "Last Date":
            dates[-1] if dates else "",

        "Positive Rows":
            s["positive"],

        "Negative Rows":
            s["negative"],

        "Sensors":
            " | ".join(
                sorted(s["sensors"])
            ),

        "Inventory Levels":
            " | ".join(
                sorted(s["levels"])
            ),
    })


SUMMARY_OUT = (
    OUT
    / "site_summary_with_methanesat.csv"
)

write_csv(
    SUMMARY_OUT,
    summary_rows,
    SUMMARY_FIELDS,
)


# ============================================================
# Source coverage
# ============================================================

source_candidates = find_files(
    "source_coverage_summary.csv"
)

coverage_rows = []


if source_candidates:

    old_cov = read_csv(
        source_candidates[0]
    )

    for r in old_cov:

        if clean(
            r.get("Source")
        ).lower().startswith(
            "methanesat"
        ):
            continue

        coverage_rows.append(r)


unique_collections = set()
unique_targets = set()


for raw in manifest_rows:

    r = normalized_row(raw)

    c = first(
        r,
        ["collection_id"]
    )

    t = first(
        r,
        ["target_id"]
    )

    if c:
        unique_collections.add(c)

    if t:
        unique_targets.add(t)


coverage_rows.append({

    "Source":
        "MethaneSAT",

    "Status":
        "Compiled",

    "Local data":
        "Yes",

    "Included in master":
        "Yes",

    "Notes":
        (
            f"{len(methanesat_rows)} classification samples: "
            f"{ms_labels.get('1',0)} L4-centered positives + "
            f"{ms_labels.get('0',0)} same-scene spatial weak negatives; "
            f"{len(unique_collections)} unique collections; "
            f"{len(unique_targets)} unique target IDs. "
            "Weak negatives are >=10 km from known L4 detections "
            "and are not confirmed no-methane ground truth."
        ),
})


COVERAGE_FIELDS = [
    "Source",
    "Status",
    "Local data",
    "Included in master",
    "Notes",
]


COVERAGE_OUT = (
    OUT
    / "source_coverage_with_methanesat.csv"
)

write_csv(
    COVERAGE_OUT,
    coverage_rows,
    COVERAGE_FIELDS,
)


# ============================================================
# Preserve original temporal candidate negatives
# ============================================================

candidate_candidates = find_files(
    "candidate_negative_inventory.csv"
)

if candidate_candidates:

    shutil.copy2(
        candidate_candidates[0],
        OUT / "candidate_negative_inventory.csv"
    )


# ============================================================
# Audit
# ============================================================

inventory_levels = Counter(
    r["Inventory Level"]
    for r in combined_dedup
)

sensors = Counter(
    r["Sensor"]
    for r in combined_dedup
)

audit_lines = [

    "PROFESSOR MASTER + METHANESAT INTEGRATION",

    "=" * 80,

    f"Old master rows          : {len(old_master_raw)}",

    f"MethaneSAT rows added    : {len(methanesat_rows)}",

    f"Final combined rows      : {len(combined_dedup)}",

    "",

    f"MethaneSAT positive      : {ms_labels.get('1',0)}",

    f"MethaneSAT weak negative : {ms_labels.get('0',0)}",

    f"<10 km violations        : {len(violations)}",

    f"Unique collections       : {len(unique_collections)}",

    f"Unique target IDs        : {len(unique_targets)}",

    "",

    "Inventory levels:",
]


for k, v in inventory_levels.items():
    audit_lines.append(
        f"  {k}: {v}"
    )


audit_lines.append("")
audit_lines.append("Sensors:")


for k, v in sensors.most_common():

    audit_lines.append(
        f"  {k or '[N/A]'}: {v}"
    )


AUDIT_OUT = (
    OUT
    / "integration_audit.txt"
)


AUDIT_OUT.write_text(
    "\n".join(audit_lines),
    encoding="utf-8",
)


print("\n" + "=" * 100)
print("INTEGRATION COMPLETE")
print("=" * 100)

for line in audit_lines:
    print(line)

print("\nOUTPUT DIRECTORY:")
print(OUT)

print("\nFILES:")

for p in [
    MS_OUT,
    COMBINED_OUT,
    SUMMARY_OUT,
    COVERAGE_OUT,
    AUDIT_OUT,
]:
    print(" ", p)
