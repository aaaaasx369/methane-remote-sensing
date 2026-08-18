import os
import sys
import json
import csv
import time
import urllib.parse
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from datetime import datetime

# ============================================================
# SETTINGS
# ============================================================

BASE_URL = "https://api.carbonmapper.org/api/v1/catalog/plumes/annotated"

TOKEN = os.environ.get("CARBON_MAPPER_TOKEN")

PAGE_SIZE = 5000
SLEEP_SECONDS = 0.15

OUT_JSON = "carbon_mapper_all_CH4_plumes.json"
OUT_CSV = "carbon_mapper_all_CH4_plumes.csv"
OUT_SENSOR_CSV = "carbon_mapper_CH4_sensor_summary.csv"


# ============================================================
# CHECK TOKEN
# ============================================================

if not TOKEN:
    print("ERROR: CARBON_MAPPER_TOKEN is not set.")
    print("")
    print("Run:")
    print("export CARBON_MAPPER_TOKEN='YOUR_TOKEN'")
    sys.exit(1)

print("Carbon Mapper token found.")
print("Token length:", len(TOKEN))


# ============================================================
# HTTP REQUEST
# ============================================================

def api_get(params):

    query = urllib.parse.urlencode(params)

    url = BASE_URL + "?" + query

    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json",
            "User-Agent": "methane-research-inventory/1.0",
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as e:
        print("")
        print("HTTP ERROR:", e.code)
        print("URL:", url)

        try:
            print(e.read().decode("utf-8"))
        except Exception:
            pass

        raise

    except Exception as e:
        print("")
        print("REQUEST FAILED:", repr(e))
        print("URL:", url)
        raise


# ============================================================
# TEST REQUEST
# ============================================================

print("")
print("=" * 60)
print("TESTING CARBON MAPPER API")
print("=" * 60)

test = api_get({
    "plume_gas": "CH4",
    "limit": 3,
    "offset": 0,
})

print("API connection successful.")

total_count = test.get("total_count")

print("API-reported CH4 plume count:", total_count)

test_items = test.get("items", [])

print("Test records returned:", len(test_items))

if test_items:
    print("")
    print("Fields in first record:")
    for key in sorted(test_items[0].keys()):
        print("  ", key)

else:
    print("ERROR: API returned zero CH4 records.")
    sys.exit(1)


# ============================================================
# DOWNLOAD ALL CH4 METADATA
# ============================================================

print("")
print("=" * 60)
print("DOWNLOADING ALL CH4 PLUME METADATA")
print("=" * 60)

all_items = []

offset = 0

while True:

    data = api_get({
        "plume_gas": "CH4",
        "limit": PAGE_SIZE,
        "offset": offset,
    })

    items = data.get("items", [])

    if not items:
        break

    all_items.extend(items)

    api_total = data.get("total_count", total_count)

    print(
        f"Fetched {len(all_items):,} / "
        f"{api_total:,} CH4 plume records"
    )

    offset += len(items)

    if api_total is not None and offset >= api_total:
        break

    time.sleep(SLEEP_SECONDS)


print("")
print("Finished fetching catalogue.")
print("Total records downloaded:", len(all_items))


# ============================================================
# SAVE RAW JSON
# ============================================================

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        all_items,
        f,
        indent=2,
        ensure_ascii=False
    )

print("")
print("Saved raw JSON:")
print(OUT_JSON)


# ============================================================
# HELPERS
# ============================================================

def first_nonempty(d, *keys):
    for k in keys:
        value = d.get(k)

        if value is not None and value != "":
            return value

    return None


def extract_lon_lat(p):

    # Preferred geometry object
    geom = p.get("geometry_json")

    if isinstance(geom, dict):

        coords = geom.get("coordinates")

        if (
            isinstance(coords, list)
            and len(coords) >= 2
            and isinstance(coords[0], (int, float))
            and isinstance(coords[1], (int, float))
        ):
            return coords[0], coords[1]

    # Alternate public fields
    lon = first_nonempty(
        p,
        "plume_longitude",
        "longitude",
        "lon"
    )

    lat = first_nonempty(
        p,
        "plume_latitude",
        "latitude",
        "lat"
    )

    return lon, lat


def has_value(v):
    return v is not None and v != "" and v != [] and v != {}


# ============================================================
# BUILD FLATTENED ROWS
# ============================================================

rows = []

for p in all_items:

    lon, lat = extract_lon_lat(p)

    row = {

        "plume_id":
            first_nonempty(
                p,
                "plume_id",
                "id"
            ),

        "plume_name":
            first_nonempty(
                p,
                "plume_name",
                "name"
            ),

        "gas":
            first_nonempty(
                p,
                "gas",
                "plume_gas"
            ),

        "scene_timestamp":
            first_nonempty(
                p,
                "scene_timestamp",
                "datetime",
                "timestamp"
            ),

        "longitude":
            lon,

        "latitude":
            lat,

        "instrument":
            p.get("instrument"),

        "platform":
            p.get("platform"),

        "mission_phase":
            p.get("mission_phase"),

        "sector":
            first_nonempty(
                p,
                "sector",
                "ipcc_sector"
            ),

        "emission_auto_kg_hr":
            p.get("emission_auto"),

        "emission_uncertainty_auto":
            first_nonempty(
                p,
                "emission_uncertainty_auto",
                "emission_auto_uncertainty"
            ),

        "wind_speed_avg_auto":
            p.get("wind_speed_avg_auto"),

        "wind_direction_avg_auto":
            p.get("wind_direction_avg_auto"),

        "plume_quality":
            p.get("plume_quality"),

        "gsd_m":
            p.get("gsd"),

        "off_nadir":
            p.get("off_nadir"),

        "published_at":
            p.get("published_at"),

        "modified":
            p.get("modified"),

        "plume_tif":
            p.get("plume_tif"),

        "con_tif":
            p.get("con_tif"),

        "rgb_tif":
            p.get("rgb_tif"),

        "plume_png":
            p.get("plume_png"),

        "rgb_png":
            p.get("rgb_png"),

        "plume_bounds":
            json.dumps(
                p.get("plume_bounds"),
                ensure_ascii=False
            ) if p.get("plume_bounds") is not None else None,
    }

    rows.append(row)


# ============================================================
# SAVE MAIN CSV
# ============================================================

if rows:

    fieldnames = list(rows[0].keys())

    with open(
        OUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


print("")
print("Saved flattened CSV:")
print(OUT_CSV)


# ============================================================
# SENSOR COUNTS
# ============================================================

instrument_counts = Counter(
    str(r["instrument"] or "UNKNOWN")
    for r in rows
)

platform_counts = Counter(
    str(r["platform"] or "UNKNOWN")
    for r in rows
)

instrument_platform_counts = Counter(
    (
        str(r["instrument"] or "UNKNOWN"),
        str(r["platform"] or "UNKNOWN")
    )
    for r in rows
)


# ============================================================
# FILE AVAILABILITY BY SENSOR
# ============================================================

sensor_stats = defaultdict(
    lambda: {
        "count": 0,
        "with_plume_tif": 0,
        "with_con_tif": 0,
        "with_rgb_tif": 0,
        "with_plume_png": 0,
        "with_emission_rate": 0,
    }
)

for r in rows:

    inst = str(
        r["instrument"] or "UNKNOWN"
    )

    s = sensor_stats[inst]

    s["count"] += 1

    if has_value(r["plume_tif"]):
        s["with_plume_tif"] += 1

    if has_value(r["con_tif"]):
        s["with_con_tif"] += 1

    if has_value(r["rgb_tif"]):
        s["with_rgb_tif"] += 1

    if has_value(r["plume_png"]):
        s["with_plume_png"] += 1

    if has_value(r["emission_auto_kg_hr"]):
        s["with_emission_rate"] += 1


# ============================================================
# SAVE SENSOR SUMMARY CSV
# ============================================================

with open(
    OUT_SENSOR_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fields = [
        "instrument",
        "count",
        "with_plume_tif",
        "with_con_tif",
        "with_rgb_tif",
        "with_plume_png",
        "with_emission_rate",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    for inst, stats in sorted(
        sensor_stats.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    ):

        writer.writerow({
            "instrument": inst,
            **stats
        })


# ============================================================
# GLOBAL AVAILABILITY
# ============================================================

total = len(rows)

with_plume_tif = sum(
    has_value(r["plume_tif"])
    for r in rows
)

with_con_tif = sum(
    has_value(r["con_tif"])
    for r in rows
)

with_rgb_tif = sum(
    has_value(r["rgb_tif"])
    for r in rows
)

with_plume_png = sum(
    has_value(r["plume_png"])
    for r in rows
)

with_emission = sum(
    has_value(r["emission_auto_kg_hr"])
    for r in rows
)


# ============================================================
# DATE RANGE
# ============================================================

timestamps = [
    str(r["scene_timestamp"])
    for r in rows
    if r["scene_timestamp"]
]

if timestamps:
    earliest = min(timestamps)
    latest = max(timestamps)
else:
    earliest = None
    latest = None


# ============================================================
# SECTOR COUNTS
# ============================================================

sector_counts = Counter(
    str(r["sector"] or "UNKNOWN")
    for r in rows
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("")
print("=" * 60)
print("CARBON MAPPER CH4 INVENTORY SUMMARY")
print("=" * 60)

print("")
print("TOTAL CH4 PLUMES:")
print(total)

print("")
print("DATE RANGE:")
print("Earliest:", earliest)
print("Latest:  ", latest)


print("")
print("=" * 60)
print("BY INSTRUMENT")
print("=" * 60)

for instrument, count in instrument_counts.most_common():

    s = sensor_stats[instrument]

    print("")
    print(instrument)
    print("  total:", count)
    print(
        "  plume_tif:",
        s["with_plume_tif"]
    )
    print(
        "  con_tif:",
        s["with_con_tif"]
    )
    print(
        "  rgb_tif:",
        s["with_rgb_tif"]
    )
    print(
        "  plume_png:",
        s["with_plume_png"]
    )
    print(
        "  emission rate:",
        s["with_emission_rate"]
    )


print("")
print("=" * 60)
print("BY PLATFORM")
print("=" * 60)

for platform, count in platform_counts.most_common():
    print(
        f"{platform}: {count}"
    )


print("")
print("=" * 60)
print("INSTRUMENT + PLATFORM")
print("=" * 60)

for (inst, platform), count in \
        instrument_platform_counts.most_common():

    print(
        f"{inst} | {platform} | {count}"
    )


print("")
print("=" * 60)
print("GLOBAL PRODUCT AVAILABILITY")
print("=" * 60)

print(
    "with plume_tif:",
    with_plume_tif,
    "/",
    total
)

print(
    "with con_tif:",
    with_con_tif,
    "/",
    total
)

print(
    "with rgb_tif:",
    with_rgb_tif,
    "/",
    total
)

print(
    "with plume_png:",
    with_plume_png,
    "/",
    total
)

print(
    "with emission rate:",
    with_emission,
    "/",
    total
)


print("")
print("=" * 60)
print("TOP SECTORS")
print("=" * 60)

for sector, count in sector_counts.most_common(20):
    print(
        f"{sector}: {count}"
    )


print("")
print("=" * 60)
print("FILES CREATED")
print("=" * 60)

print(OUT_JSON)
print(OUT_CSV)
print(OUT_SENSOR_CSV)

print("")
print("DONE.")
print("No GeoTIFFs were downloaded yet.")
