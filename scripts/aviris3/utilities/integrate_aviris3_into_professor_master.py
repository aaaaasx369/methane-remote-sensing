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

OLD_MASTER_DIR = (
    PROJECT
    / "professor_master_inventory_with_methanesat"
)

OLD_MASTER = (
    OLD_MASTER_DIR
    / "all_inventory_with_methanesat.csv"
)

OLD_COVERAGE = (
    OLD_MASTER_DIR
    / "source_coverage_with_methanesat.csv"
)

OLD_CANDIDATES = (
    OLD_MASTER_DIR
    / "candidate_negative_inventory.csv"
)

OUT = (
    PROJECT
    / "professor_master_inventory_with_methanesat_aviris3"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


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


def normalize_row(row):

    return {
        norm(k): clean(v)
        for k, v in row.items()
        if k is not None
    }


def first(row, names):

    for name in names:

        value = row.get(
            norm(name),
            ""
        )

        if clean(value):
            return clean(value)

    return ""


def read_csv(path):

    with path.open(
        newline="",
        encoding="utf-8-sig",
        errors="replace",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def write_csv(
    path,
    rows,
    fields
):

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Locate AVIRIS dataset directory
# ============================================================

dataset_dirs = []


for root in SEARCH_ROOTS:

    if not root.exists():
        continue

    for p in root.rglob(
        "aviris3_methanefuse_final20_exact"
    ):

        if p.is_dir():
            dataset_dirs.append(p)


dataset_dirs = sorted(
    set(dataset_dirs)
)


if not dataset_dirs:

    raise SystemExit(
        "\nERROR: Could not find "
        "aviris3_methanefuse_final20_exact\n"
        "under ~/methane_release_project or ~/Downloads."
    )


print("\nAVIRIS-3 dataset candidates:")

for p in dataset_dirs:
    print(" ", p)


DATASET = dataset_dirs[0]


print("\nSelected dataset:")
print(DATASET)


# ============================================================
# Find all CSV files
# ============================================================

csv_files = sorted(
    DATASET.rglob("*.csv")
)


print("\nCSV files:")

for p in csv_files:

    try:
        rows = read_csv(p)

        print(
            f"  {p.relative_to(DATASET)}"
            f" | rows={len(rows)}"
        )

    except Exception as e:

        print(
            "  ERROR:",
            p,
            e
        )


# ============================================================
# Detect the 20-row classification manifest
# ============================================================

def detect_label_column(rows):

    if not rows:
        return None

    normalized_keys = {
        norm(k)
        for k in rows[0].keys()
        if k
    }

    for c in [
        "label",
        "true_label",
        "class",
        "target_label",
    ]:

        if c in normalized_keys:
            return c

    return None


manifest_candidates = []


for p in csv_files:

    try:
        raw = read_csv(p)

    except Exception:
        continue

    if len(raw) != 20:
        continue

    rows = [
        normalize_row(r)
        for r in raw
    ]

    label_col = detect_label_column(
        raw
    )

    if label_col:

        labels = Counter(
            first(
                r,
                [
                    "label",
                    "true_label",
                    "class",
                    "target_label",
                ]
            )
            for r in rows
        )

        manifest_candidates.append(
            (
                p,
                raw,
                labels
            )
        )


print("\n20-row manifest candidates:")

for p, rows, labels in manifest_candidates:

    print(
        " ",
        p.relative_to(DATASET),
        dict(labels)
    )


MANIFEST = None
manifest_raw = None


for p, rows, labels in manifest_candidates:

    if (
        labels.get("1", 0) == 10
        and labels.get("0", 0) == 10
    ):

        MANIFEST = p
        manifest_raw = rows
        break


if MANIFEST is None:

    raise SystemExit(
        "\nERROR: Could not automatically find "
        "a 20-row manifest containing "
        "10 positive + 10 negative samples."
    )


print("\nSelected classification manifest:")
print(MANIFEST)


manifest_rows = [
    normalize_row(r)
    for r in manifest_raw
]


# ============================================================
# Detect provenance_manifest.csv
# ============================================================

PROVENANCE = None


for p in csv_files:

    if (
        "provenance"
        in p.name.lower()
    ):

        PROVENANCE = p
        break


provenance_rows = []


if PROVENANCE:

    provenance_rows = [
        normalize_row(r)
        for r in read_csv(PROVENANCE)
    ]

    print("\nProvenance manifest:")
    print(PROVENANCE)

    print(
        "Rows:",
        len(provenance_rows)
    )

else:

    print(
        "\nWARNING: "
        "No provenance_manifest.csv detected."
    )


# ============================================================
# Merge manifest + provenance
# ============================================================

ID_CANDIDATES = [
    "id",
    "query_id",
    "sample_id",
    "record_id",
    "master_id",
]


def get_id(row):

    return first(
        row,
        ID_CANDIDATES
    )


merge_method = (
    "manifest_only"
)


merged = []


if provenance_rows:

    manifest_ids = {
        get_id(r)
        for r in manifest_rows
        if get_id(r)
    }

    provenance_ids = {
        get_id(r)
        for r in provenance_rows
        if get_id(r)
    }


    common_ids = (
        manifest_ids
        & provenance_ids
    )


    if common_ids:

        prov_map = {
            get_id(r): r
            for r in provenance_rows
            if get_id(r)
        }

        for r in manifest_rows:

            combined = {}

            pid = get_id(r)

            if pid in prov_map:
                combined.update(
                    prov_map[pid]
                )

            combined.update(r)

            merged.append(
                combined
            )

        merge_method = (
            "ID_based"
        )


    elif (
        len(provenance_rows)
        == len(manifest_rows)
        == 20
    ):

        # Safe fallback only because
        # both tables have exactly 20 rows.
        for a, b in zip(
            manifest_rows,
            provenance_rows
        ):

            combined = {}

            combined.update(b)
            combined.update(a)

            merged.append(
                combined
            )

        merge_method = (
            "row_order_fallback"
        )


    else:

        merged = manifest_rows

else:

    merged = manifest_rows


print(
    "\nMerge method:",
    merge_method
)


# ============================================================
# Time parsing
# ============================================================

def parse_iso_datetime(text):

    s = clean(text)

    if not s:
        return "", ""

    s = (
        s.replace(
            " ",
            "T"
        )
        .replace(
            "+00:00",
            "Z"
        )
    )

    m = re.search(
        r"(20\d{2})-(\d{2})-(\d{2})"
        r"T(\d{2}):(\d{2}):(\d{2})",
        s
    )

    if m:

        y, mo, d, hh, mm, ss = (
            m.groups()
        )

        return (
            f"{y}-{mo}-{d}",
            f"{hh}:{mm}:{ss}"
        )

    return "", ""


def parse_aviris_filename(text):

    """
    Example:
    AV320240905t203326_010_L2A...
       YYYYMMDD t HHMMSS
    """

    s = clean(text)

    m = re.search(
        r"AV3(20\d{6})t(\d{6})",
        s,
        re.IGNORECASE
    )

    if not m:
        return "", ""

    date_string = m.group(1)
    time_string = m.group(2)

    return (
        (
            f"{date_string[0:4]}-"
            f"{date_string[4:6]}-"
            f"{date_string[6:8]}"
        ),
        (
            f"{time_string[0:2]}:"
            f"{time_string[2:4]}:"
            f"{time_string[4:6]}"
        ),
    )


def extract_aviris_time(row):

    # -------------------------------------------
    # Explicit AVIRIS acquisition fields first
    # -------------------------------------------

    for c in [
        "aviris3_acquisition_time_utc",
        "av3_acquisition_time_utc",
        "t0_acquisition_time_utc",
        "acquisition_time_utc",
        "aviris_datetime",
        "av3_datetime",
    ]:

        value = row.get(
            norm(c),
            ""
        )

        if value:

            d, t = parse_iso_datetime(
                value
            )

            if d:
                return d, t


    # -------------------------------------------
    # Search all metadata values for AV3 filename
    # -------------------------------------------

    for value in row.values():

        d, t = parse_aviris_filename(
            value
        )

        if d:
            return d, t


    return "", ""


# ============================================================
# Main professor schema
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


# ============================================================
# Convert AVIRIS records
# ============================================================

aviris_rows = []


for i, r in enumerate(
    merged,
    start=1
):

    sample_id = first(
        r,
        [
            "id",
            "sample_id",
            "record_id",
            "master_id",
        ]
    )

    if not sample_id:

        sample_id = (
            f"AV3_SAMPLE_{i:04d}"
        )


    label = first(
        r,
        [
            "label",
            "true_label",
            "class",
            "target_label",
        ]
    )


    lat = first(
        r,
        [
            "latitude",
            "lat",
            "center_lat",
            "sample_lat",
            "source_lat",
            "query_lat",
            "cm_latitude",
            "cm_lat",
        ]
    )


    lon = first(
        r,
        [
            "longitude",
            "lon",
            "lng",
            "center_lon",
            "sample_lon",
            "source_lon",
            "query_lon",
            "cm_longitude",
            "cm_lon",
        ]
    )


    target_id = first(
        r,
        [
            "target_id",
            "cm_target_id",
            "site_id",
            "facility_id",
        ]
    )


    cm_record_id = first(
        r,
        [
            "cm_record_id",
            "carbon_mapper_record_id",
            "plume_id",
        ]
    )


    cm_datetime = first(
        r,
        [
            "cm_datetime",
            "carbon_mapper_datetime",
        ]
    )


    cm_emission = first(
        r,
        [
            "cm_emission",
            "cm_emission_kghr",
            "cm_emission_kg_hr",
            "emission_rate_kg_hr",
            "flux_kg_hr",
        ]
    )


    t0_path = first(
        r,
        [
            "emit_0_path",
            "t0_path",
            "image_0_path",
        ]
    )


    t90_path = first(
        r,
        [
            "emit_90_path",
            "t90_path",
            "image_90_path",
        ]
    )


    t360_path = first(
        r,
        [
            "emit_360_path",
            "t360_path",
            "image_360_path",
        ]
    )


    av3_l2a = first(
        r,
        [
            "aviris3_l2a_file",
            "av3_l2a_file",
            "av3_l2a_rfl_url",
            "aviris3_rfl_file",
            "av3_rfl_file",
            "t0_source_file",
        ]
    )


    av3_l2b_ch4 = first(
        r,
        [
            "aviris3_ch4_file",
            "av3_ch4_file",
            "ch4_ort_url",
            "ch4_ort",
            "ch4_path",
        ]
    )


    av3_l2b_unc = first(
        r,
        [
            "aviris3_unc_file",
            "av3_unc_file",
            "ch4_unc_url",
            "ch4_unc_path",
        ]
    )


    av3_l2b_sns = first(
        r,
        [
            "aviris3_sns_file",
            "av3_sns_file",
            "ch4_sns_url",
            "ch4_sns_path",
        ]
    )


    emit90_source = first(
        r,
        [
            "emit_90_source",
            "emit90_source",
            "emit_t90_rfl_url",
            "emit_t90_filename",
            "emit_90_nc",
        ]
    )


    emit180_source = first(
        r,
        [
            "emit_360_source",
            "emit360_source",
            "emit_180_source",
            "emit_t180_rfl_url",
            "emit_t180_filename",
            "emit_180_nc",
        ]
    )


    date, utc = (
        extract_aviris_time(r)
    )


    # -----------------------------------------
    # Site
    # -----------------------------------------

    if target_id:

        site = (
            f"AVIRIS3_{target_id}"
        )

    elif lat and lon:

        try:

            site = (
                "AVIRIS3_location_"
                f"{float(lat):.4f}_"
                f"{float(lon):.4f}"
            )

        except Exception:

            site = (
                "AVIRIS3_external_sample"
            )

    else:

        site = (
            "AVIRIS3_external_sample"
        )


    # -----------------------------------------
    # Positive / negative physical meaning
    # -----------------------------------------

    if label == "1":

        label_type = (
            "Carbon Mapper plume-positive "
            "AVIRIS-3 sample"
        )

        modality = (
            "Carbon Mapper published plume "
            "matched to AVIRIS-3"
        )

        hist_exp = (
            "Observational external test"
        )

        release_rate = (
            cm_emission
        )


    elif label == "0":

        label_type = (
            "AVIRIS-3 matched weak negative"
        )

        modality = (
            "AVIRIS-3 L2B-screened "
            "background weak negative"
        )

        hist_exp = (
            "Matched weak-negative reference"
        )

        release_rate = ""


    else:

        label_type = (
            "AVIRIS-3 external sample"
        )

        modality = (
            "AVIRIS-3 / EMIT"
        )

        hist_exp = (
            "External test"
        )

        release_rate = ""


    # -----------------------------------------
    # Notes
    # -----------------------------------------

    notes = [

        (
            "AVIRIS-3 / EMIT "
            "MethaneFuse external sample"
        ),

        (
            "t0 = AVIRIS-3 L2A reflectance"
        ),

        (
            "historical frame 1 = "
            "EMIT approximately t-90"
        ),

        (
            "historical frame 2 = "
            "EMIT approximately t-180"
        ),

        (
            "loader column emit_360 stores "
            "the approximately t-180 frame; "
            "it is not true t-360"
        ),

        (
            "hyperspectral reflectance "
            "harmonized to 16 WV3-like bands"
        ),
    ]


    if label == "0":

        notes.append(
            (
                "weak negative selected using "
                "AVIRIS-3 L2B CH4/UNC/SNS QA; "
                "not confirmed no-methane ground truth"
            )
        )


    if cm_record_id:

        notes.append(
            f"CarbonMapper_record_id="
            f"{cm_record_id}"
        )


    if cm_datetime:

        notes.append(
            f"CarbonMapper_datetime="
            f"{cm_datetime}"
        )


    if av3_l2a:

        notes.append(
            f"AVIRIS3_L2A={av3_l2a}"
        )


    if av3_l2b_ch4:

        notes.append(
            f"AVIRIS3_L2B_CH4="
            f"{av3_l2b_ch4}"
        )


    if av3_l2b_unc:

        notes.append(
            f"AVIRIS3_L2B_UNC="
            f"{av3_l2b_unc}"
        )


    if av3_l2b_sns:

        notes.append(
            f"AVIRIS3_L2B_SNS="
            f"{av3_l2b_sns}"
        )


    if emit90_source:

        notes.append(
            f"EMIT_t90_source="
            f"{emit90_source}"
        )


    if emit180_source:

        notes.append(
            f"EMIT_t180_source="
            f"{emit180_source}"
        )


    if t0_path:

        notes.append(
            f"model_t0={t0_path}"
        )


    if t90_path:

        notes.append(
            f"model_t90={t90_path}"
        )


    if t360_path:

        notes.append(
            f"model_t180_as_emit360="
            f"{t360_path}"
        )


    if not date:

        notes.append(
            "AVIRIS-3 acquisition date "
            "not automatically recovered"
        )


    aviris_rows.append({

        "Inventory Level":
            "Sensor Observation",

        "Site":
            site,

        "Location Level":
            (
                "AVIRIS-3 sample location / "
                "Carbon Mapper plume target"
            ),

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
            (
                "AVIRIS-3 + EMIT "
                "(WV3-like 16-band temporal)"
            ),

        "Scene/Observation ID":
            sample_id,

        "Release Rate (kg/hr)":
            release_rate,

        "Historical/Experiment":
            hist_exp,

        "Source Dataset":
            (
                "NASA AVIRIS-3 L2A/L2B + "
                "NASA EMIT L2A + "
                "Carbon Mapper plume records"
            ),

        "Paper/Reference":
            (
                "AV3_L2A_RFL_2357; "
                "AV3_L2B_GHG_2358; "
                "EMITL2ARFL; "
                "Carbon Mapper Data Platform API; "
                "MethaneFuse/MethaneUnion WV3 SRF"
            ),

        "Notes":
            "; ".join(notes),
    })


# ============================================================
# Validate AVIRIS dataset
# ============================================================

labels = Counter(
    r["Label"]
    for r in aviris_rows
)


print(
    "\n" + "=" * 100
)

print(
    "AVIRIS-3 FINAL20 QA"
)

print(
    "=" * 100
)

print(
    "Rows:",
    len(aviris_rows)
)

print(
    "Positive:",
    labels.get("1", 0)
)

print(
    "Negative:",
    labels.get("0", 0)
)


if len(aviris_rows) != 20:

    raise SystemExit(
        "ERROR: Expected exactly 20 AVIRIS rows."
    )


if (
    labels.get("1", 0) != 10
    or labels.get("0", 0) != 10
):

    raise SystemExit(
        "ERROR: Expected 10 positive + 10 negative."
    )


# ============================================================
# Read old professor master
# ============================================================

if not OLD_MASTER.exists():

    raise SystemExit(
        f"\nERROR: Existing professor master not found:\n"
        f"{OLD_MASTER}"
    )


old_rows = read_csv(
    OLD_MASTER
)


print(
    "\nOld professor master rows:",
    len(old_rows)
)


# ============================================================
# Prevent duplicate AVIRIS rows if script reruns
# ============================================================

base_rows = []


for r in old_rows:

    sensor = clean(
        r.get("Sensor")
    )

    source = clean(
        r.get("Source Dataset")
    )

    if (
        "AVIRIS-3"
        in sensor
        or "AVIRIS-3"
        in source
    ):
        continue

    base_rows.append(r)


# ============================================================
# Add AVIRIS rows
# ============================================================

combined = (
    base_rows
    + aviris_rows
)


def dedup_key(r):

    return (
        clean(
            r.get(
                "Inventory Level"
            )
        ),
        clean(
            r.get("Sensor")
        ),
        clean(
            r.get(
                "Scene/Observation ID"
            )
        ),
        clean(
            r.get("Latitude")
        ),
        clean(
            r.get("Longitude")
        ),
        clean(
            r.get("Label")
        ),
    )


seen = set()
final_rows = []


for r in combined:

    key = dedup_key(r)

    if key in seen:
        continue

    seen.add(key)

    final_rows.append(r)


final_rows.sort(

    key=lambda r: (

        clean(r.get("Site")),

        clean(r.get("Date")),

        clean(r.get("UTC Time")),

        clean(r.get("Sensor")),

        clean(
            r.get(
                "Scene/Observation ID"
            )
        ),
    )
)


# ============================================================
# Write AVIRIS-only inventory
# ============================================================

AVIRIS_OUT = (
    OUT
    / "aviris3_emit_final20_inventory.csv"
)


write_csv(
    AVIRIS_OUT,
    aviris_rows,
    FIELDS,
)


# ============================================================
# Write final combined inventory
# ============================================================

ALL_OUT = (
    OUT
    / "all_inventory_with_methanesat_aviris3.csv"
)


write_csv(
    ALL_OUT,
    final_rows,
    FIELDS,
)


# ============================================================
# Site summary
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


for r in final_rows:

    site = clean(
        r.get("Site")
    )

    s = summary[site]

    s["records"] += 1

    date = clean(
        r.get("Date")
    )

    if date:
        s["dates"].add(date)

    label = clean(
        r.get("Label")
    )

    if label == "1":
        s["positive"] += 1

    elif label == "0":
        s["negative"] += 1

    sensor = clean(
        r.get("Sensor")
    )

    if sensor:
        s["sensors"].add(
            sensor
        )

    level = clean(
        r.get(
            "Inventory Level"
        )
    )

    if level:
        s["levels"].add(
            level
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
            dates[0]
            if dates
            else "",

        "Last Date":
            dates[-1]
            if dates
            else "",

        "Positive Rows":
            s["positive"],

        "Negative Rows":
            s["negative"],

        "Sensors":
            " | ".join(
                sorted(
                    s["sensors"]
                )
            ),

        "Inventory Levels":
            " | ".join(
                sorted(
                    s["levels"]
                )
            ),
    })


SUMMARY_OUT = (
    OUT
    / "site_summary_with_methanesat_aviris3.csv"
)


write_csv(
    SUMMARY_OUT,
    summary_rows,
    SUMMARY_FIELDS,
)


# ============================================================
# Source coverage
# ============================================================

coverage = []


if OLD_COVERAGE.exists():

    coverage = read_csv(
        OLD_COVERAGE
    )


# Remove old AVIRIS / EMIT entries,
# because EMIT status changes now.
new_coverage = []


for r in coverage:

    source = clean(
        r.get("Source")
    ).lower()

    if (
        "aviris" in source
        or source == "emit"
    ):
        continue

    new_coverage.append(r)


new_coverage.append({

    "Source":
        "AVIRIS-3",

    "Status":
        "Compiled",

    "Local data":
        "Yes",

    "Included in master":
        "Yes",

    "Notes":
        (
            "20 MethaneFuse-compatible external samples: "
            "10 Carbon Mapper plume-positive samples and "
            "10 AVIRIS-3 L2B-screened matched weak negatives. "
            "AVIRIS-3 L2A provides t0 reflectance."
        ),
})


new_coverage.append({

    "Source":
        "EMIT",

    "Status":
        "Compiled as supporting historical imagery",

    "Local data":
        "Yes",

    "Included in master":
        "Yes, within AVIRIS-3 temporal samples",

    "Notes":
        (
            "EMIT L2A reflectance provides the historical "
            "approximately t-90 and t-180 frames for the "
            "20 AVIRIS-3 external samples. "
            "The loader's emit_360 field stores t-180."
        ),
})


coverage_fields = [
    "Source",
    "Status",
    "Local data",
    "Included in master",
    "Notes",
]


COVERAGE_OUT = (
    OUT
    / "source_coverage_with_methanesat_aviris3.csv"
)


write_csv(
    COVERAGE_OUT,
    new_coverage,
    coverage_fields,
)


# ============================================================
# Preserve 2672 temporal candidate negatives
# ============================================================

if OLD_CANDIDATES.exists():

    shutil.copy2(
        OLD_CANDIDATES,
        OUT
        / "candidate_negative_inventory.csv"
    )


# ============================================================
# Audit
# ============================================================

levels = Counter(
    clean(
        r.get(
            "Inventory Level"
        )
    )
    for r in final_rows
)


sensors = Counter(
    clean(
        r.get("Sensor")
    )
    for r in final_rows
)


missing_date = sum(
    not clean(
        r.get("Date")
    )
    for r in aviris_rows
)


missing_coord = sum(
    (
        not clean(
            r.get("Latitude")
        )
        or not clean(
            r.get("Longitude")
        )
    )
    for r in aviris_rows
)


# Count approximate source files/scenes
av3_scene_strings = set()

emit_scene_strings = set()


for r in merged:

    for value in r.values():

        value = clean(value)

        if not value:
            continue

        for m in re.findall(
            r"AV3\d{8}t\d{6}[^/\\,\s]*",
            value,
            flags=re.I
        ):

            av3_scene_strings.add(m)

        for m in re.findall(
            r"EMIT_L2A_RFL_[^/\\,\s]*",
            value,
            flags=re.I
        ):

            emit_scene_strings.add(m)


audit = [

    "PROFESSOR MASTER + METHANESAT + AVIRIS3 INTEGRATION",

    "=" * 90,

    f"Old master rows             : {len(old_rows)}",

    f"AVIRIS-3 rows added         : {len(aviris_rows)}",

    f"Final combined rows         : {len(final_rows)}",

    "",

    f"AVIRIS-3 positive           : {labels.get('1',0)}",

    f"AVIRIS-3 weak negative      : {labels.get('0',0)}",

    f"AVIRIS rows missing date    : {missing_date}",

    f"AVIRIS rows missing coords  : {missing_coord}",

    f"Manifest/provenance merge   : {merge_method}",

    f"Detected AVIRIS scene IDs   : {len(av3_scene_strings)}",

    f"Detected EMIT scene IDs     : {len(emit_scene_strings)}",

    "",

    "Inventory levels:",
]


for k, v in levels.items():

    audit.append(
        f"  {k}: {v}"
    )


audit.append("")
audit.append("Sensors:")


for k, v in sensors.most_common():

    audit.append(
        f"  {k or '[N/A]'}: {v}"
    )


AUDIT_OUT = (
    OUT
    / "integration_audit.txt"
)


AUDIT_OUT.write_text(
    "\n".join(audit),
    encoding="utf-8"
)


# ============================================================
# Final output
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "INTEGRATION COMPLETE"
)

print(
    "=" * 100
)


for line in audit:
    print(line)


print(
    "\nOUTPUT DIRECTORY:"
)

print(
    OUT
)


print(
    "\nFILES:"
)

for p in [

    AVIRIS_OUT,

    ALL_OUT,

    SUMMARY_OUT,

    COVERAGE_OUT,

    AUDIT_OUT,

]:

    print(
        " ",
        p
    )
