from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv
import re


ROOT = Path(".").resolve()

OUT = ROOT / "results" / "master_inventory"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Files / directories that should not be scanned
# ============================================================

SKIP_PARTS = {
    ".git",
    "venv",
    "venvs",
    ".venv",
    "logs",
    "checkpoints",
    "weights",
    "__pycache__",
}

SKIP_NAMES = {
    "master_inventory.csv",
    "site_date_inventory.csv",
    "site_summary.csv",
    "file_audit.csv",
}


# ============================================================
# 2. Column aliases accumulated across our project
# ============================================================

ALIASES = {

    "id": [
        "id",
        "event_id",
        "sample_id",
        "observation_id",
        "plume_id",
        "source_sample_id",
        "external_eval_id",
        "prediction_sample_id",
    ],

    "site": [
        "site",
        "site_id",
        "site_name",
        "site_normalized",
        "facility",
        "facility_name",
        "location",
    ],

    "lat": [
        "latitude",
        "lat",
        "source_latitude",
        "source_lat",
    ],

    "lon": [
        "longitude",
        "lon",
        "lng",
        "source_longitude",
        "source_lon",
    ],

    "label": [
        "label",
        "true_label",
        "physical_release_gt",
        "ground_truth_label",
        "release_label",
    ],

    "event_time": [
        "ground_truth_time_utc",
        "event_time_utc",
        "event_time",
        "observation_time_utc",
        "release_time_utc",
        "release_time",
        "date",
    ],

    "acquisition_time": [
        "acquisition_time_utc",
        "acquisition_time",
        "datetime_utc",
        "datetime",
        "overpass_time_utc",
        "overpass_time",
        "operator_timestamp",
    ],

    "release_start": [
        "release_start_utc",
        "release_start",
        "start_time_utc",
    ],

    "release_end": [
        "release_end_utc",
        "release_end",
        "end_time_utc",
    ],

    "sensor": [
        "sensor",
        "instrument",
        "platform",
    ],

    "satellite": [
        "satellite",
        "spacecraft",
    ],

    "scene_id": [
        "scene_id",
        "image_id",
        "asset_id",
        "acquisition_id",
        "granule_id",
        "product_id",
    ],

    "t0_scene_id": [
        "t0_scene_id",
        "s2_0_scene_id",
    ],

    "campaign": [
        "campaign_id",
        "campaign",
        "study",
    ],

    "ground_truth_source": [
        "ground_truth_source",
        "source_dataset",
        "dataset_source",
        "paper",
    ],

    "ground_truth_type": [
        "ground_truth_type",
        "label_provenance",
        "label_type",
    ],

    "emission": [
        "emission_rate_kg_hr",
        "emission_rate_kg_h",
        "emission_rate_kg_hr_raw",
        "release_rate_kg_hr",
        "release_rate_kg_h",
        "release_rate_kg_hr_raw",
        "consensus_release_rate_kg_h",
        "ground_truth_rate_kg_hr",
        "ground_truth_emission_rate_kg_hr",
    ],
}


def normalize_column(s):

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(s).strip().lower()
    ).strip("_")


ALIASES = {
    key: [normalize_column(x) for x in values]
    for key, values in ALIASES.items()
}


def get_value(row, key):

    for column in ALIASES[key]:

        value = str(
            row.get(column, "")
        ).strip()

        if (
            value
            and value.lower()
            not in {
                "nan",
                "none",
                "null",
                "n/a",
                "na",
            }
        ):
            return value

    return ""


# ============================================================
# 3. Normalize labels
# ============================================================

def normalize_label(value):

    x = str(value).strip().lower()

    if x in {
        "1",
        "1.0",
        "true",
        "positive",
        "pos",
        "plume",
        "release",
    }:
        return "1"

    if x in {
        "0",
        "0.0",
        "false",
        "negative",
        "neg",
        "no_plume",
        "no plume",
        "no_release",
        "no release",
    }:
        return "0"

    return str(value).strip()


# ============================================================
# 4. Time parsing
# ============================================================

def normalize_time(value):

    s = str(value).strip()

    if not s:
        return ""

    # Unix milliseconds
    if re.fullmatch(r"\d{13}", s):

        dt = datetime.fromtimestamp(
            int(s) / 1000,
            tz=timezone.utc,
        )

        return (
            dt.isoformat()
            .replace("+00:00", "Z")
        )

    # Unix seconds
    if re.fullmatch(r"\d{10}", s):

        dt = datetime.fromtimestamp(
            int(s),
            tz=timezone.utc,
        )

        return (
            dt.isoformat()
            .replace("+00:00", "Z")
        )

    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}",
        s,
    ):
        return s

    return (
        s.replace(" ", "T")
        .replace("+00:00", "Z")
    )


def scene_time(scene_id):

    """
    Extract e.g.
    20211103T181531
    from Sentinel/Landsat-like scene IDs.
    """

    match = re.search(
        r"(20\d{6})T(\d{6})",
        str(scene_id),
    )

    if not match:
        return ""

    date_string = match.group(1)
    time_string = match.group(2)

    return (
        f"{date_string[0:4]}-"
        f"{date_string[4:6]}-"
        f"{date_string[6:8]}T"
        f"{time_string[0:2]}:"
        f"{time_string[2:4]}:"
        f"{time_string[4:6]}Z"
    )


def date_only(value):

    match = re.search(
        r"20\d{2}-\d{2}-\d{2}",
        normalize_time(value),
    )

    if match:
        return match.group(0)

    return ""


# ============================================================
# 5. Site normalization
# ============================================================

def infer_site_from_path(path):

    text = str(path).lower()

    rules = [

        (
            [
                "casa_grande",
                "casagrande",
            ],
            "Casa_Grande",
        ),

        (
            ["ehrenberg"],
            "Ehrenberg",
        ),

        (
            ["site_038"],
            "MA_site_038",
        ),

        (
            ["site_043"],
            "MA_site_043",
        ),

        (
            ["site_073"],
            "MA_site_073",
        ),

        (
            ["haynesville"],
            "Haynesville",
        ),

        (
            ["ne_marcellus"],
            "NE Marcellus",
        ),

        (
            ["sw_marcellus"],
            "SW Marcellus",
        ),

        (
            ["permian"],
            "Permian (Delaware)",
        ),

        (
            ["baltimore"],
            "Baltimore",
        ),

        (
            ["denver"],
            "Test Flight Denver Metro CO",
        ),

        (
            ["nyc"],
            "NYC",
        ),
    ]

    for keywords, site in rules:

        if any(
            keyword in text
            for keyword in keywords
        ):
            return site

    return ""


def normalize_site(site):

    x = str(site).strip()

    key = normalize_column(x)

    mapping = {

        "casa_grande":
            "Casa_Grande",

        "casagrande":
            "Casa_Grande",

        "ehrenberg":
            "Ehrenberg",

        "ma_site_038":
            "MA_site_038",

        "methaneair_site_038":
            "MA_site_038",

        "ma_site_043":
            "MA_site_043",

        "methaneair_site_043":
            "MA_site_043",

        "ma_site_073":
            "MA_site_073",

        "methaneair_site_073":
            "MA_site_073",

        "ne_marcellus":
            "NE Marcellus",

        "sw_marcellus":
            "SW Marcellus",

        "permian_delaware":
            "Permian (Delaware)",

        "test_flight_denver_metro_co":
            "Test Flight Denver Metro CO",
    }

    return mapping.get(
        key,
        x,
    )


# ============================================================
# 6. Sensor inference
# ============================================================

def infer_sensor(
    explicit_sensor,
    satellite,
    scene,
    path,
):

    text = " ".join([
        explicit_sensor,
        satellite,
        scene,
        str(path),
    ]).lower()

    if (
        "sentinel-5" in text
        or "s5p" in text
        or "tropomi" in text
    ):
        return "Sentinel-5P"

    if (
        "sentinel-2" in text
        or "copernicus/s2" in text
        or "s2_sr" in text
    ):
        return "Sentinel-2"

    if (
        "landsat" in text
        or "lc08" in text
        or "lc09" in text
    ):
        return "Landsat 8/9"

    if "ghgsat" in text:
        return "GHGSat"

    if "tanager" in text:
        return "Tanager"

    if (
        "carbonmapper" in text
        or "carbon mapper" in text
    ):
        return "Carbon Mapper"

    if "emit" in text:
        return "EMIT"

    if explicit_sensor:
        return explicit_sensor

    if satellite:
        return satellite

    return ""


# ============================================================
# 7. Ground-truth / dataset provenance
# ============================================================

def infer_source(
    ground_truth_source,
    campaign,
    path,
    record_id,
):

    text = " ".join([
        ground_truth_source,
        campaign,
        str(path),
        record_id,
    ]).lower()

    if (
        "2023_scientific" in text
        or "scientific_reports" in text
        or "s41598-023-30761-2" in text
    ):
        return (
            "2023 Scientific Reports "
            "controlled release"
        )

    if (
        "2024_amt" in text
        or "amt-17-765-2024" in text
    ):
        return (
            "2024 AMT "
            "controlled release"
        )

    if "methaneair" in text:
        return "MethaneAIR"

    if (
        "carbonmapper" in text
        or "carbon mapper" in text
        or "tanager" in text
    ):
        return "Carbon Mapper"

    if "ghgsat" in text:
        return "GHGSat"

    if (
        "s5p" in text
        or "tropomi" in text
    ):
        return "Sentinel-5P"

    if "emit" in text:
        return "EMIT"

    if "landsat" in text:
        return "Landsat"

    if (
        "sentinel" in text
        or "s2_" in text
    ):
        return "Sentinel-2"

    return (
        ground_truth_source
        or campaign
        or "Unclassified"
    )


REFERENCES = {

    "2023 Scientific Reports controlled release":
        (
            "Sherwin et al., Scientific Reports (2023), "
            "DOI 10.1038/s41598-023-30761-2"
        ),

    "2024 AMT controlled release":
        (
            "Sherwin et al., Atmospheric Measurement "
            "Techniques 17, 765-782 (2024), "
            "DOI 10.5194/amt-17-765-2024"
        ),

    "MethaneAIR":
        (
            "MethaneAIR project dataset / "
            "compiled plume-reference records"
        ),

    "Carbon Mapper":
        (
            "Carbon Mapper Data Portal / "
            "compiled Carbon Mapper records"
        ),

    "GHGSat":
        (
            "GHGSat / compiled "
            "controlled-release records"
        ),

    "Sentinel-5P":
        (
            "Copernicus Sentinel-5P "
            "methane product"
        ),

    "EMIT":
        "NASA EMIT",

    "Landsat":
        "USGS Landsat 8/9",

    "Sentinel-2":
        "Copernicus Sentinel-2",
}


# ============================================================
# 8. Formal experiment windows
# ============================================================

def campaign_window_status(
    site,
    date,
):

    if site == "Ehrenberg":

        if (
            "2021-10-16"
            <= date
            <= "2021-11-03"
        ):
            return "formal_campaign"

        return (
            "historical_or_review"
        )

    if site == "Casa_Grande":

        if (
            "2022-10-10"
            <= date
            <= "2022-11-30"
        ):
            return "formal_campaign"

        return (
            "historical_or_review"
        )

    return ""


# ============================================================
# 9. Strength of negative labels
# ============================================================

def negative_status(
    y,
    ground_truth_type,
    source,
):

    if y != "0":
        return ""

    text = (
        ground_truth_type
        + " "
        + source
    ).lower()

    if (
        "controlled" in text
        or "physical_release" in text
        or "scientific reports" in text
        or "amt" in text
    ):
        return (
            "confirmed_no_release_or_zero_release"
        )

    if (
        "plume_reference" in text
        or "no_known_plume" in text
        or "methaneair" in text
    ):
        return (
            "no_known_plume_reference_not_confirmed"
        )

    return "needs_review"


# ============================================================
# 10. Determine which CSVs are observation-like
# ============================================================

def should_skip(path):

    relative = path.relative_to(ROOT)

    if any(
        part.lower() in SKIP_PARTS
        for part in relative.parts
    ):
        return True

    if path.name in SKIP_NAMES:
        return True

    if OUT in path.parents:
        return True

    return False


def relevant_table(
    header,
):

    h = {
        normalize_column(x)
        for x in header
    }

    has_site = (
        bool(
            h
            & set(ALIASES["site"])
        )
        or (
            bool(
                h
                & set(ALIASES["lat"])
            )
            and bool(
                h
                & set(ALIASES["lon"])
            )
        )
    )

    has_time = bool(
        h
        & (
            set(ALIASES["event_time"])
            | set(ALIASES["acquisition_time"])
            | set(ALIASES["scene_id"])
            | set(ALIASES["t0_scene_id"])
        )
    )

    has_observation_identity = bool(
        h
        & (
            set(ALIASES["label"])
            | set(ALIASES["sensor"])
            | set(ALIASES["satellite"])
            | set(ALIASES["ground_truth_source"])
            | set(ALIASES["emission"])
        )
    )

    return (
        has_site
        and has_time
        and has_observation_identity
    )


# ============================================================
# 11. Master schema
# ============================================================

FIELDS = [

    "record_id",

    "site",

    "latitude",
    "longitude",

    "label",

    "negative_status",

    "ground_truth_type",

    "event_time_utc",

    "acquisition_time_utc",

    "inventory_date",

    "sensor",

    "satellite",

    "scene_id",

    "t0_scene_id",

    "emission_rate_kg_hr",

    "campaign_id",

    "source_family",

    "reference",

    "campaign_window_status",

    "source_files",

    "source_mentions",
]


# ============================================================
# 12. Scan every CSV in the project
# ============================================================

canonical = {}

audit = []

csv_files = sorted(
    path
    for path in ROOT.rglob("*.csv")
    if not should_skip(path)
)

csv.field_size_limit(
    100_000_000
)


for path in csv_files:

    relative = str(
        path.relative_to(ROOT)
    )

    rows_scanned = 0
    rows_accepted = 0

    try:

        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file
            )

            header = (
                reader.fieldnames
                or []
            )

            if not relevant_table(
                header
            ):

                audit.append([
                    relative,
                    "skipped",
                    0,
                    0,
                    "not observation-like",
                ])

                continue


            for source_row, raw_row in enumerate(
                reader,
                start=2,
            ):

                rows_scanned += 1

                row = {

                    normalize_column(k):
                        str(v).strip()

                    for k, v
                    in raw_row.items()

                    if k
                }


                record_id = get_value(
                    row,
                    "id",
                )


                site = normalize_site(

                    get_value(
                        row,
                        "site",
                    )

                    or infer_site_from_path(
                        path
                    )
                )


                if not site:
                    continue


                event_time = normalize_time(
                    get_value(
                        row,
                        "event_time",
                    )
                )


                scene_id = get_value(
                    row,
                    "scene_id",
                )


                t0_scene_id = get_value(
                    row,
                    "t0_scene_id",
                )


                acquisition_time = (

                    normalize_time(
                        get_value(
                            row,
                            "acquisition_time",
                        )
                    )

                    or scene_time(
                        t0_scene_id
                    )

                    or scene_time(
                        scene_id
                    )
                )


                inventory_date = (

                    date_only(
                        acquisition_time
                    )

                    or date_only(
                        event_time
                    )
                )


                if not inventory_date:
                    continue


                y = normalize_label(
                    get_value(
                        row,
                        "label",
                    )
                )


                ground_truth_source = (
                    get_value(
                        row,
                        "ground_truth_source",
                    )
                )


                campaign = get_value(
                    row,
                    "campaign",
                )


                ground_truth_type = (
                    get_value(
                        row,
                        "ground_truth_type",
                    )
                )


                satellite = get_value(
                    row,
                    "satellite",
                )


                sensor = infer_sensor(

                    get_value(
                        row,
                        "sensor",
                    ),

                    satellite,

                    scene_id
                    + " "
                    + t0_scene_id,

                    path,
                )


                source_family = (
                    infer_source(

                        ground_truth_source,

                        campaign,

                        path,

                        record_id,
                    )
                )


                record = {

                    "record_id":
                        record_id,

                    "site":
                        site,

                    "latitude":
                        get_value(
                            row,
                            "lat",
                        ),

                    "longitude":
                        get_value(
                            row,
                            "lon",
                        ),

                    "label":
                        y,

                    "negative_status":
                        negative_status(
                            y,
                            ground_truth_type,
                            source_family,
                        ),

                    "ground_truth_type":
                        ground_truth_type,

                    "event_time_utc":
                        event_time,

                    "acquisition_time_utc":
                        acquisition_time,

                    "inventory_date":
                        inventory_date,

                    "sensor":
                        sensor,

                    "satellite":
                        satellite,

                    "scene_id":
                        scene_id,

                    "t0_scene_id":
                        t0_scene_id,

                    "emission_rate_kg_hr":
                        get_value(
                            row,
                            "emission",
                        ),

                    "campaign_id":
                        campaign,

                    "source_family":
                        source_family,

                    "reference":
                        REFERENCES.get(
                            source_family,
                            source_family,
                        ),

                    "campaign_window_status":
                        campaign_window_status(
                            site,
                            inventory_date,
                        ),
                }


                # --------------------------------------------
                # Deduplication
                # --------------------------------------------

                if record_id:

                    key = (
                        source_family
                        + "|ID|"
                        + record_id
                    )

                else:

                    key = "|".join([

                        source_family,

                        site,

                        event_time
                        or acquisition_time,

                        sensor,

                        scene_id
                        or t0_scene_id,

                        y,

                        record["latitude"],

                        record["longitude"],
                    ])


                if key not in canonical:

                    canonical[key] = (
                        record
                        | {
                            "_files":
                                {relative},

                            "_mentions":
                                1,
                        }
                    )

                else:

                    current = canonical[
                        key
                    ]

                    current[
                        "_files"
                    ].add(
                        relative
                    )

                    current[
                        "_mentions"
                    ] += 1


                    # Fill information that may
                    # be missing in another version
                    for k, value in record.items():

                        if (
                            not current.get(k)
                            and value
                        ):
                            current[k] = value


                rows_accepted += 1


        audit.append([

            relative,

            "accepted",

            rows_scanned,

            rows_accepted,

            "",
        ])


    except Exception as error:

        audit.append([

            relative,

            "error",

            rows_scanned,

            rows_accepted,

            (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        ])


# ============================================================
# 13. Write canonical master inventory
# ============================================================

rows = []


for current in canonical.values():

    row = {

        key:
            current.get(
                key,
                "",
            )

        for key in FIELDS

        if key not in {
            "source_files",
            "source_mentions",
        }
    }


    row[
        "source_files"
    ] = " | ".join(

        sorted(
            current["_files"]
        )
    )


    row[
        "source_mentions"
    ] = current[
        "_mentions"
    ]


    rows.append(
        row
    )


rows.sort(

    key=lambda r: (

        r["site"],

        r["inventory_date"],

        r["sensor"],

        r["label"],

        r["record_id"],
    )
)


MASTER = (
    OUT
    / "master_inventory.csv"
)


with MASTER.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=FIELDS,
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# ============================================================
# 14. Site × date × sensor × label table
# ============================================================

groups = {}


for row in rows:

    key = (

        row["site"],

        row["inventory_date"],

        row["label"],

        row["sensor"],

        row["source_family"],
    )


    group = groups.setdefault(

        key,

        {
            "count": 0,
            "references": set(),
            "files": set(),
        },
    )


    group[
        "count"
    ] += 1


    group[
        "references"
    ].add(
        row["reference"]
    )


    group[
        "files"
    ].update(

        x

        for x
        in row[
            "source_files"
        ].split(" | ")

        if x
    )


SITE_DATE = (
    OUT
    / "site_date_inventory.csv"
)


SITE_DATE_FIELDS = [

    "site",

    "date",

    "label",

    "sensor",

    "source_family",

    "observation_count",

    "reference",

    "source_files",
]


with SITE_DATE.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=SITE_DATE_FIELDS,
    )

    writer.writeheader()


    for key, group in sorted(
        groups.items()
    ):

        site, date, y, sensor, source = key

        writer.writerow({

            "site":
                site,

            "date":
                date,

            "label":
                y,

            "sensor":
                sensor,

            "source_family":
                source,

            "observation_count":
                group["count"],

            "reference":
                " | ".join(
                    sorted(
                        group[
                            "references"
                        ]
                    )
                ),

            "source_files":
                " | ".join(
                    sorted(
                        group[
                            "files"
                        ]
                    )
                ),
        })


# ============================================================
# 15. Site summary
# ============================================================

summary = defaultdict(
    lambda: {
        "dates": set(),
        "labels": Counter(),
        "sensors": set(),
        "sources": set(),
        "count": 0,
    }
)


for row in rows:

    item = summary[
        row["site"]
    ]

    item[
        "count"
    ] += 1

    item[
        "dates"
    ].add(
        row["inventory_date"]
    )

    if row["label"]:

        item[
            "labels"
        ][
            row["label"]
        ] += 1

    if row["sensor"]:

        item[
            "sensors"
        ].add(
            row["sensor"]
        )

    item[
        "sources"
    ].add(
        row[
            "source_family"
        ]
    )


SUMMARY = (
    OUT
    / "site_summary.csv"
)


SUMMARY_FIELDS = [

    "site",

    "observations",

    "unique_dates",

    "first_date",

    "last_date",

    "positive",

    "negative",

    "sensors",

    "sources",
]


with SUMMARY.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=SUMMARY_FIELDS,
    )

    writer.writeheader()


    for site in sorted(
        summary
    ):

        item = summary[
            site
        ]

        dates = sorted(
            item["dates"]
        )


        writer.writerow({

            "site":
                site,

            "observations":
                item["count"],

            "unique_dates":
                len(dates),

            "first_date":
                dates[0]
                if dates
                else "",

            "last_date":
                dates[-1]
                if dates
                else "",

            "positive":
                item[
                    "labels"
                ].get(
                    "1",
                    0,
                ),

            "negative":
                item[
                    "labels"
                ].get(
                    "0",
                    0,
                ),

            "sensors":
                " | ".join(
                    sorted(
                        item[
                            "sensors"
                        ]
                    )
                ),

            "sources":
                " | ".join(
                    sorted(
                        item[
                            "sources"
                        ]
                    )
                ),
        })


# ============================================================
# 16. File audit
# ============================================================

AUDIT = (
    OUT
    / "file_audit.csv"
)


with AUDIT.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow([

        "file",

        "status",

        "rows_scanned",

        "rows_accepted",

        "note",
    ])

    writer.writerows(
        audit
    )


# ============================================================
# 17. Non-CSV files that still need review
# ============================================================

other_files = []


for extension in [

    "*.xlsx",

    "*.xls",

    "*.parquet",

    "*.json",

    "*.geojson",
]:

    for path in ROOT.rglob(
        extension
    ):

        if should_skip(
            path
        ):
            continue

        other_files.append(

            str(
                path.relative_to(
                    ROOT
                )
            )
        )


NON_CSV = (
    OUT
    / "non_csv_files_to_review.txt"
)


NON_CSV.write_text(

    "\n".join(
        sorted(
            set(
                other_files
            )
        )
    ),

    encoding="utf-8",
)


# ============================================================
# 18. Print final summary
# ============================================================

print(
    "=" * 100
)

print(
    "MASTER INVENTORY COMPLETE"
)

print(
    "=" * 100
)

print(
    "CSV files found:",
    len(csv_files),
)

print(
    "Accepted observation tables:",
    sum(
        row[1] == "accepted"
        for row in audit
    ),
)

print(
    "Errors:",
    sum(
        row[1] == "error"
        for row in audit
    ),
)

print(
    "Canonical observations:",
    len(rows),
)

print(
    "Sites:",
    len(summary),
)

print()


for site in sorted(
    summary
):

    item = summary[
        site
    ]

    dates = sorted(
        item["dates"]
    )

    print(

        f"{site:35s} "

        f"obs={item['count']:5d} "

        f"dates={len(dates):3d} "

        f"pos={item['labels'].get('1',0):4d} "

        f"neg={item['labels'].get('0',0):4d} "

        f"{dates[0] if dates else '?'} "

        f"-> "

        f"{dates[-1] if dates else '?'}"
    )


print()

print(
    "OUTPUT DIRECTORY:"
)

print(
    OUT
)

print()

print(
    "1.",
    MASTER,
)

print(
    "2.",
    SITE_DATE,
)

print(
    "3.",
    SUMMARY,
)

print(
    "4.",
    AUDIT,
)

print(
    "5.",
    NON_CSV,
)
