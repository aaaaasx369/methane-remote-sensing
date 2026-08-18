from pathlib import Path
import csv
from datetime import datetime
from collections import Counter

ROOT = Path("/project/6002520/yunjung1/MethaneFuse")

GT_FILE = ROOT / "data/methaneair_full/ground_truth_confirmed.csv"
MULTI_FILE = ROOT / "outputs/027_unified_methane_master_landsat_recovered.csv"
ALL_FILE = ROOT / "data/methaneair_full/ground_truth_all.csv"

OUTDIR = ROOT / "results/professor_master_inventory"
OUTDIR.mkdir(parents=True, exist_ok=True)

MASTER_OUT = OUTDIR / "master_site_date_source_inventory.csv"
CANDIDATE_OUT = OUTDIR / "candidate_negative_inventory.csv"
SUMMARY_OUT = OUTDIR / "site_summary.csv"
SOURCE_OUT = OUTDIR / "source_coverage_summary.csv"


FIELDS = [
    "Site",
    "Latitude",
    "Longitude",
    "Date",
    "UTC Time",
    "Label",
    "Label Type",
    "Sensor",
    "Scene/Observation ID",
    "Release Rate",
    "Historical/Experiment",
    "Source Dataset",
    "Paper/Reference",
    "Notes",
]


# ============================================================
# Helpers
# ============================================================

def clean(x):
    if x is None:
        return ""
    x = str(x).strip()
    if x.lower() in {"nan", "none", "null"}:
        return ""
    return x


def split_time(x):
    """
    Input:
      2022-11-08T18:15:51Z
      2021-10-19 18:13:59+00:00
    Output:
      date, UTC time
    """
    s = clean(x)

    if not s:
        return "", ""

    s = s.replace(" ", "T")
    s = s.replace("+00:00", "Z")

    if "T" in s:
        d, t = s.split("T", 1)
        return d, t.replace("Z", "")

    return s[:10], ""


def normalize_site(site):
    s = clean(site)

    aliases = {
        "Casa_Grande_AZ_release_stacks": "Casa_Grande",
        "Casa_Grande_AZ_release_stack": "Casa_Grande",
        "Ehrenberg_AZ_release_stack": "Ehrenberg",
        "Ehrenberg_AZ_release_stacks": "Ehrenberg",
        "MethaneAIR_site_038": "MA_site_038",
        "MethaneAIR_site_043": "MA_site_043",
        "MethaneAIR_site_073": "MA_site_073",
    }

    return aliases.get(s, s)


def formal_campaign_status(site, date):
    """
    Distinguish formal controlled-release campaign
    from historical/reference acquisitions.
    """

    if site == "Ehrenberg":
        if "2021-10-16" <= date <= "2021-11-03":
            return "Experiment"
        return "Historical/reference"

    if site == "Casa_Grande":
        if "2022-10-10" <= date <= "2022-11-30":
            return "Experiment"
        return "Historical/reference"

    return "Observational"


def label_type_gt(r):
    y = clean(r.get("label"))
    src = clean(r.get("ground_truth_source"))
    gt_type = clean(r.get("ground_truth_type"))
    status = clean(r.get("label_status"))

    if src == "physical_release":
        if y == "1":
            return "Confirmed controlled release"
        elif y == "0":
            return "Confirmed no-release / zero-release"

    if src == "MethaneAIR_L4_point_sources":
        return "Observed methane plume positive"

    if status == "confirmed":
        return "Confirmed ground truth"

    return gt_type or status


def gt_reference(r):
    site = normalize_site(r.get("site_id"))
    src = clean(r.get("ground_truth_source"))

    if site == "Ehrenberg" and src == "physical_release":
        return (
            "Scientific Reports (2023), "
            "Stanford controlled-release experiment; "
            "DOI: 10.1038/s41598-023-30761-2"
        )

    if site == "Casa_Grande" and src == "physical_release":
        return (
            "Atmospheric Measurement Techniques (2024), "
            "controlled-release experiment; "
            "DOI: 10.5194/amt-17-765-2024"
        )

    if src == "MethaneAIR_L4_point_sources":
        return "MethaneAIR L4 point-source dataset"

    return src


def multi_reference(r):
    sensor = clean(r.get("sensor"))
    site = normalize_site(r.get("site_id"))
    gt_src = clean(r.get("ground_truth_source"))

    if sensor == "Carbon Mapper Tanager":
        return "Carbon Mapper published plume / Tanager"

    if site == "Ehrenberg" and gt_src == "physical_release":
        return (
            "Scientific Reports (2023); "
            "DOI: 10.1038/s41598-023-30761-2"
        )

    if site == "Casa_Grande" and gt_src == "physical_release":
        return (
            "Atmospheric Measurement Techniques (2024); "
            "DOI: 10.5194/amt-17-765-2024"
        )

    if gt_src == "MethaneAIR_observational_detection":
        return "MethaneAIR observational detection"

    if gt_src == "plume_reference":
        return "Historical no-known-plume reference"

    return gt_src


def source_dataset_gt(r):
    src = clean(r.get("ground_truth_source"))

    if src == "physical_release":
        return "Stanford controlled-release ground truth"

    if src == "MethaneAIR_L4_point_sources":
        return "MethaneAIR L4 point sources"

    return src


# ============================================================
# 1. Ground-truth confirmed rows
# ============================================================

master_rows = []

with GT_FILE.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for r in reader:

        site = normalize_site(r.get("site_id"))
        date, utc = split_time(r.get("acquisition_time_utc"))

        obs_id = (
            clean(r.get("record_id"))
            or clean(r.get("plume_id"))
            or clean(r.get("flight_id"))
        )

        source = clean(r.get("ground_truth_source"))

        note_parts = [
            "Confirmed ground-truth record"
        ]

        if clean(r.get("controlled_release_verified")):
            note_parts.append(
                "controlled_release_verified="
                + clean(r.get("controlled_release_verified"))
            )

        if site == "Ehrenberg" and date == "2021-08-03":
            note_parts.append(
                "Outside formal 2021-10-16 to 2021-11-03 "
                "campaign window; provenance review recommended"
            )

        master_rows.append({
            "Site": site,
            "Latitude": clean(r.get("latitude")),
            "Longitude": clean(r.get("longitude")),
            "Date": date,
            "UTC Time": utc,
            "Label": clean(r.get("label")),
            "Label Type": label_type_gt(r),
            "Sensor": clean(r.get("sensor_ground_truth")),
            "Scene/Observation ID": obs_id,
            "Release Rate": clean(r.get("emission_rate_kg_hr")),
            "Historical/Experiment": formal_campaign_status(site, date),
            "Source Dataset": source_dataset_gt(r),
            "Paper/Reference": gt_reference(r),
            "Notes": "; ".join(note_parts),
        })


# ============================================================
# 2. Multisensor observations
#    Sentinel-2 / Landsat / Carbon Mapper
# ============================================================

with MULTI_FILE.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for r in reader:

        site = normalize_site(r.get("site_id"))
        date, utc = split_time(r.get("acquisition_time_utc"))

        sensor = clean(r.get("sensor"))
        gt_src = clean(r.get("ground_truth_source"))

        if gt_src == "physical_release":
            if clean(r.get("label")) == "1":
                label_type = "Controlled-release matched sensor observation"
            else:
                label_type = "Controlled no-release/reference sensor observation"

        elif gt_src == "plume_reference":
            label_type = "No-known-plume reference"

        elif gt_src == "MethaneAIR_observational_detection":
            label_type = "MethaneAIR observational positive"

        elif gt_src == "Carbon Mapper published plume":
            label_type = "Published plume positive"

        else:
            label_type = gt_src

        obs_id = (
            clean(r.get("scene_id"))
            or clean(r.get("source_row_ids"))
            or clean(r.get("master_id"))
        )

        notes = [
            "Sensor-level compiled observation"
        ]

        if clean(r.get("qa_pass")):
            notes.append("qa_pass=" + clean(r.get("qa_pass")))

        if clean(r.get("model_ready")):
            notes.append("model_ready=" + clean(r.get("model_ready")))

        if clean(r.get("source_datasets")):
            notes.append(
                "derived_from=" + clean(r.get("source_datasets"))
            )

        master_rows.append({
            "Site": site,
            "Latitude": clean(r.get("latitude")),
            "Longitude": clean(r.get("longitude")),
            "Date": date,
            "UTC Time": utc,
            "Label": clean(r.get("label")),
            "Label Type": label_type,
            "Sensor": sensor,
            "Scene/Observation ID": obs_id,
            "Release Rate": clean(r.get("emission_rate_kg_hr")),
            "Historical/Experiment": formal_campaign_status(site, date),
            "Source Dataset": clean(r.get("source_datasets")),
            "Paper/Reference": multi_reference(r),
            "Notes": "; ".join(notes),
        })


# ============================================================
# 3. Deduplicate exact identical master rows
# ============================================================

seen = set()
dedup = []

for r in master_rows:

    key = tuple(r.get(c, "") for c in FIELDS)

    if key not in seen:
        seen.add(key)
        dedup.append(r)

master_rows = sorted(
    dedup,
    key=lambda r: (
        r["Site"],
        r["Date"],
        r["UTC Time"],
        r["Sensor"],
        r["Scene/Observation ID"],
    )
)


# ============================================================
# 4. Write professor master
# ============================================================

with MASTER_OUT.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(master_rows)


# ============================================================
# 5. Candidate negatives – separate file
# ============================================================

candidate_fields = [
    "Site",
    "Latitude",
    "Longitude",
    "Date",
    "UTC Time",
    "Proposed Label",
    "Label Status",
    "Days After Positive",
    "Source Positive Record ID",
    "Known Release Excluded",
    "Known Plume Excluded",
    "Nearby Plume Excluded",
    "Cloud/Snow QA Pass",
    "Negative Validity",
    "Source Dataset",
]

candidate_rows = []

with ALL_FILE.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for r in reader:

        if clean(r.get("label_status")) != "candidate_unconfirmed":
            continue

        site = normalize_site(r.get("site_id"))
        date, utc = split_time(r.get("acquisition_time_utc"))

        candidate_rows.append({
            "Site": site,
            "Latitude": clean(r.get("latitude")),
            "Longitude": clean(r.get("longitude")),
            "Date": date,
            "UTC Time": utc,
            "Proposed Label": clean(r.get("proposed_label")),
            "Label Status": clean(r.get("label_status")),
            "Days After Positive": clean(r.get("days_after_positive")),
            "Source Positive Record ID": clean(r.get("source_positive_record_id")),
            "Known Release Excluded": clean(r.get("known_release_excluded")),
            "Known Plume Excluded": clean(r.get("known_plume_excluded")),
            "Nearby Plume Excluded": clean(r.get("nearby_plume_excluded")),
            "Cloud/Snow QA Pass": clean(r.get("cloud_snow_qa_pass")),
            "Negative Validity": clean(r.get("negative_validity")),
            "Source Dataset": "MethaneAIR temporal negative candidates",
        })

with CANDIDATE_OUT.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=candidate_fields
    )

    w.writeheader()
    w.writerows(candidate_rows)


# ============================================================
# 6. Site summary
# ============================================================

summary = {}

for r in master_rows:

    site = r["Site"]

    s = summary.setdefault(site, {
        "records": 0,
        "dates": set(),
        "positive": 0,
        "negative": 0,
        "sensors": set(),
        "sources": set(),
    })

    s["records"] += 1

    if r["Date"]:
        s["dates"].add(r["Date"])

    if r["Label"] == "1":
        s["positive"] += 1
    elif r["Label"] == "0":
        s["negative"] += 1

    if r["Sensor"]:
        s["sensors"].add(r["Sensor"])

    if r["Source Dataset"]:
        s["sources"].add(r["Source Dataset"])


summary_fields = [
    "Site",
    "Records",
    "Unique Dates",
    "First Date",
    "Last Date",
    "Positive",
    "Negative",
    "Sensors",
    "Sources",
]

with SUMMARY_OUT.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    w = csv.DictWriter(f, fieldnames=summary_fields)
    w.writeheader()

    for site in sorted(summary):

        s = summary[site]
        dates = sorted(s["dates"])

        w.writerow({
            "Site": site,
            "Records": s["records"],
            "Unique Dates": len(dates),
            "First Date": dates[0] if dates else "",
            "Last Date": dates[-1] if dates else "",
            "Positive": s["positive"],
            "Negative": s["negative"],
            "Sensors": " | ".join(sorted(s["sensors"])),
            "Sources": " | ".join(sorted(s["sources"])),
        })


# ============================================================
# 7. Source coverage summary
# ============================================================

coverage = [
    {
        "Source": "MethaneAIR",
        "Status": "Compiled",
        "Local data": "Yes",
        "Included in master": "Yes",
        "Notes": "668 observational positives in confirmed ground-truth universe",
    },
    {
        "Source": "Controlled release",
        "Status": "Compiled",
        "Local data": "Yes",
        "Included in master": "Yes",
        "Notes": "Casa Grande and Ehrenberg physical-release ground truth",
    },
    {
        "Source": "Sentinel-2",
        "Status": "Compiled",
        "Local data": "Yes",
        "Included in master": "Yes",
        "Notes": "Sensor-level observations represented in unified multisensor table",
    },
    {
        "Source": "Landsat 8/9",
        "Status": "Compiled",
        "Local data": "Yes",
        "Included in master": "Yes",
        "Notes": "16 deduplicated observations in unified multisensor table",
    },
    {
        "Source": "Carbon Mapper / Tanager",
        "Status": "Compiled",
        "Local data": "Yes",
        "Included in master": "Yes",
        "Notes": "5 published positive plume observations",
    },
    {
        "Source": "GHGSat",
        "Status": "Referenced but not locally acquired",
        "Local data": "No",
        "Included in master": "No",
        "Notes": (
            "Upstream GHGSat provenance is referenced in existing records; "
            "standalone GHGSat dataset has not yet been downloaded"
        ),
    },
    {
        "Source": "Sentinel-5P / TROPOMI",
        "Status": "Investigated",
        "Local data": "Not incorporated",
        "Included in master": "No",
        "Notes": "Separate retrieval work exists but not yet integrated into Fir master",
    },
    {
        "Source": "EMIT",
        "Status": "Availability investigated",
        "Local data": "No matched observation table",
        "Included in master": "No",
        "Notes": "No eligible matched observations currently included",
    },
]

with SOURCE_OUT.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    fields = [
        "Source",
        "Status",
        "Local data",
        "Included in master",
        "Notes",
    ]

    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(coverage)


# ============================================================
# 8. Print audit
# ============================================================

print("=" * 100)
print("PROFESSOR MASTER INVENTORY COMPLETE")
print("=" * 100)

print("Main master rows:", len(master_rows))
print("Candidate negatives:", len(candidate_rows))
print("Sites:", len(summary))

print("\nSensors:")
for k, v in Counter(
    r["Sensor"] for r in master_rows
).most_common():
    print(f"  {k or '[blank]'}: {v}")

print("\nLabel types:")
for k, v in Counter(
    r["Label Type"] for r in master_rows
).most_common():
    print(f"  {k or '[blank]'}: {v}")

print("\nFiles:")
print(" ", MASTER_OUT)
print(" ", CANDIDATE_OUT)
print(" ", SUMMARY_OUT)
print(" ", SOURCE_OUT)
