from pathlib import Path
from datetime import timedelta
import json
import math
import os

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

ROOT = (
    Path.home()
    / "methane_release_project"
    / "candidate_negative_validation"
)

INPUT = ROOT / "pilot_10_positive_40_candidates_s2qa.csv"

OUTDIR = ROOT / "haynesville_P04_D03_validation"
OUTDIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_ID = "PILOT_P04_D03"

# Search radius is for DISCOVERY only.
# It is NOT automatically used as the final exclusion threshold.
SEARCH_RADIUS_KM = 5.0

CM_API = "https://api.carbonmapper.org/api/v1"

EMIT_CH4ENH_COLLECTION = "C3242680113-LPCLOUD"

TIMEOUT = 90


# ============================================================
# HELPERS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        +
        math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


def make_bbox(lat, lon, radius_km):
    dlat = radius_km / 111.32

    dlon = (
        radius_km
        /
        (
            111.32
            *
            max(
                math.cos(math.radians(lat)),
                0.01,
            )
        )
    )

    west = lon - dlon
    south = lat - dlat
    east = lon + dlon
    north = lat + dlat

    return [
        west,
        south,
        east,
        north,
    ]


def save_json(path, obj):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def get_items(obj):
    if not isinstance(obj, dict):
        return []

    if isinstance(obj.get("items"), list):
        return obj["items"]

    if isinstance(obj.get("features"), list):
        return obj["features"]

    return []


def get_point_from_item(item):
    """
    Try common Carbon Mapper geometry layouts.
    Return (lat, lon) or (None, None).
    """

    candidates = []

    if isinstance(item, dict):

        if isinstance(
            item.get("geometry_json"),
            dict,
        ):
            candidates.append(
                item["geometry_json"]
            )

        if isinstance(
            item.get("geometry"),
            dict,
        ):
            candidates.append(
                item["geometry"]
            )

    for geom in candidates:

        if geom.get("type") == "Point":

            coords = geom.get("coordinates")

            if (
                isinstance(coords, list)
                and len(coords) >= 2
            ):
                return (
                    float(coords[1]),
                    float(coords[0]),
                )

    # explicit lat/lon fallbacks
    lat_keys = [
        "latitude",
        "plume_latitude",
        "lat",
    ]

    lon_keys = [
        "longitude",
        "plume_longitude",
        "lon",
        "lng",
    ]

    lat = None
    lon = None

    for k in lat_keys:
        if item.get(k) is not None:
            try:
                lat = float(item[k])
                break
            except Exception:
                pass

    for k in lon_keys:
        if item.get(k) is not None:
            try:
                lon = float(item[k])
                break
            except Exception:
                pass

    return lat, lon


def flatten_dict(d, prefix=""):
    out = {}

    if not isinstance(d, dict):
        return out

    for k, v in d.items():

        name = (
            f"{prefix}.{k}"
            if prefix
            else str(k)
        )

        if isinstance(v, dict):
            out.update(
                flatten_dict(v, name)
            )

        else:
            out[name] = v

    return out


def find_cloud_fields(item):
    flat = flatten_dict(item)

    return {
        k: v
        for k, v in flat.items()
        if "cloud" in k.lower()
    }


# ============================================================
# LOAD CANDIDATE
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_csv(
    INPUT,
    low_memory=False,
)

hit = df[
    df["Pilot Candidate ID"]
    ==
    CANDIDATE_ID
].copy()

if len(hit) != 1:
    raise RuntimeError(
        f"Expected exactly one {CANDIDATE_ID}, "
        f"found {len(hit)}"
    )

row = hit.iloc[0]

lat = float(row["Latitude"])
lon = float(row["Longitude"])

date = pd.Timestamp(
    row["Date"]
).normalize()

actual_s2 = pd.to_datetime(
    row[
        "Actual Model Acquisition Datetime UTC"
    ],
    utc=True,
)

offset_days = int(
    row["Resolved Offset Days"]
)

inferred_parent_date = (
    date
    -
    pd.Timedelta(days=offset_days)
)

bbox = make_bbox(
    lat,
    lon,
    SEARCH_RADIUS_KM,
)

start = date.strftime(
    "%Y-%m-%dT00:00:00Z"
)

end = date.strftime(
    "%Y-%m-%dT23:59:59Z"
)

print("=" * 100)
print("HAYNESVILLE TEMPORAL NEGATIVE VALIDATION")
print("=" * 100)

print("\nCandidate:")
print(" ID              :", CANDIDATE_ID)
print(" Site            :", row["Site"])
print(" Lat/Lon         :", lat, lon)
print(" Candidate date  :", date.date())
print(" Target offset   :", f"+{offset_days} days")
print(" Parent date*    :", inferred_parent_date.date())
print(" Actual S2       :", actual_s2)
print(" S2 clear        :", row["S2 Clear Fraction"])
print(" S2 QA           :", row["Cloud/Snow QA Pass"])

print(
    "\n* Parent date is inferred from "
    "candidate date - Days After Positive."
)


# ============================================================
# SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent":
        "MethaneFuse-candidate-negative-validation/1.0",
    "Accept": "application/json",
})

# Optional Carbon Mapper token
CM_TOKEN = os.environ.get(
    "CARBON_MAPPER_TOKEN"
)

if CM_TOKEN:
    session.headers.update({
        "Authorization": f"Bearer {CM_TOKEN}"
    })

    print(
        "\nCarbon Mapper token: FOUND in "
        "CARBON_MAPPER_TOKEN"
    )

else:
    print(
        "\nCarbon Mapper token: NOT SET"
    )

    print(
        "Will try public endpoints first."
    )


# ============================================================
# CARBON MAPPER OPENAPI DISCOVERY
# ============================================================

print("\n" + "=" * 100)
print("1. CARBON MAPPER API DISCOVERY")
print("=" * 100)

openapi_url = (
    CM_API
    +
    "/openapi.json"
)

openapi = None

try:

    r = session.get(
        openapi_url,
        timeout=TIMEOUT,
    )

    print(
        "OpenAPI HTTP:",
        r.status_code
    )

    r.raise_for_status()

    openapi = r.json()

    save_json(
        OUTDIR / "carbonmapper_openapi.json",
        openapi,
    )

except Exception as e:

    print(
        "OpenAPI discovery failed:",
        repr(e)
    )


scene_path = None
plume_path = None

if isinstance(openapi, dict):

    paths = openapi.get(
        "paths",
        {}
    )

    for path, methods in paths.items():

        text = (
            path
            +
            " "
            +
            json.dumps(methods)
        ).lower()

        if (
            "scenes annotated" in text
            or (
                "scene" in path.lower()
                and "annotated" in path.lower()
            )
        ):
            scene_path = path

        if (
            "plumes annotated" in text
            or (
                "plume" in path.lower()
                and "annotated" in path.lower()
            )
        ):
            plume_path = path


# Known API path fallback
if plume_path is None:
    plume_path = (
        "/api/v1/catalog/plumes/annotated"
    )

if scene_path is None:
    scene_path = (
        "/api/v1/catalog/scenes/annotated"
    )


print(
    "Plume endpoint:",
    plume_path
)

print(
    "Scene endpoint:",
    scene_path
)


# ============================================================
# CARBON MAPPER QUERY PARAMETER BUILDER
# ============================================================

def endpoint_params_from_openapi(path):

    names = set()

    if not isinstance(openapi, dict):
        return names

    op = (
        openapi
        .get("paths", {})
        .get(path, {})
        .get("get", {})
    )

    for p in op.get(
        "parameters",
        []
    ):

        if isinstance(p, dict):
            name = p.get("name")

            if name:
                names.add(name)

    return names


def make_cm_params(path):

    allowed = (
        endpoint_params_from_openapi(
            path
        )
    )

    print(
        "\nSupported params for",
        path,
        ":",
        sorted(allowed)
        if allowed
        else "OpenAPI unavailable; using defaults"
    )

    p = {}

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    if not allowed or "bbox" in allowed:
        p["bbox"] = bbox

    # --------------------------------------------------------
    # Temporal query
    # --------------------------------------------------------

    # STAC/RFC3339-style interval
    interval = f"{start}/{end}"

    if "datetime" in allowed:
        p["datetime"] = interval

    if "datetime_min" in allowed:
        p["datetime_min"] = start

    if "datetime_max" in allowed:
        p["datetime_max"] = end

    if "start_datetime" in allowed:
        p["start_datetime"] = start

    if "end_datetime" in allowed:
        p["end_datetime"] = end

    if "gas" in allowed:
        p["gas"] = "CH4"

    if not allowed or "limit" in allowed:
        p["limit"] = 1000

    if not allowed or "offset" in allowed:
        p["offset"] = 0

    if "sort" in allowed:
        p["sort"] = "desc"

    return p


def cm_get(path):

    if path.startswith("/api/"):
        url = (
            "https://api.carbonmapper.org"
            +
            path
        )

    else:
        url = (
            CM_API
            +
            "/"
            +
            path.lstrip("/")
        )

    params = make_cm_params(
        path
    )

    print("\nGET:")
    print(url)

    r = session.get(
        url,
        params=params,
        timeout=TIMEOUT,
    )

    print(
        "HTTP:",
        r.status_code
    )

    print(
        "Final URL:",
        r.url
    )

    if r.status_code >= 400:

        print(
            "Response:"
        )

        print(
            r.text[:2000]
        )

        return {
            "_http_status": r.status_code,
            "_error": r.text[:5000],
        }

    return r.json()


# ============================================================
# 2. CARBON MAPPER PLUME SEARCH
# ============================================================

print("\n" + "=" * 100)
print("2. CARBON MAPPER PLUME SEARCH")
print("=" * 100)

cm_plume_json = cm_get(
    plume_path
)

save_json(
    OUTDIR
    /
    "carbonmapper_plume_query.json",
    cm_plume_json,
)

cm_plumes = get_items(
    cm_plume_json
)

print(
    "\nReturned plume records:",
    len(cm_plumes)
)

plume_rows = []

for p in cm_plumes:

    plat, plon = (
        get_point_from_item(p)
    )

    distance_km = None

    if (
        plat is not None
        and plon is not None
    ):

        distance_km = (
            haversine_km(
                lat,
                lon,
                plat,
                plon,
            )
        )

    timestamp = (
        p.get("scene_timestamp")
        or p.get("datetime")
        or p.get("timestamp")
    )

    plume_rows.append({
        "plume_id":
            p.get("plume_id")
            or p.get("id"),

        "scene_id":
            p.get("scene_id"),

        "timestamp":
            timestamp,

        "instrument":
            p.get("instrument"),

        "platform":
            p.get("platform"),

        "gas":
            p.get("gas"),

        "plume_quality":
            p.get("plume_quality"),

        "emission_kg_hr":
            p.get("emission_auto"),

        "latitude":
            plat,

        "longitude":
            plon,

        "distance_km":
            distance_km,
    })


plume_df = pd.DataFrame(
    plume_rows
)

if len(plume_df):

    plume_df = plume_df.sort_values(
        "distance_km",
        na_position="last",
    )

    plume_df.to_csv(
        OUTDIR
        /
        "carbonmapper_nearby_plumes.csv",
        index=False,
    )

    print(
        "\nNearest returned plumes:"
    )

    print(
        plume_df
        .head(20)
        .to_string(index=False)
    )

else:

    print(
        "\nNo Carbon Mapper plume records "
        "returned by the query."
    )


# ============================================================
# 3. CARBON MAPPER SCENE / COVERAGE SEARCH
# ============================================================

print("\n" + "=" * 100)
print("3. CARBON MAPPER SCENE COVERAGE SEARCH")
print("=" * 100)

cm_scene_json = cm_get(
    scene_path
)

save_json(
    OUTDIR
    /
    "carbonmapper_scene_query.json",
    cm_scene_json,
)

cm_scenes = get_items(
    cm_scene_json
)

print(
    "\nReturned scene records:",
    len(cm_scenes)
)

scene_rows = []

for s in cm_scenes:

    cloud_fields = (
        find_cloud_fields(s)
    )

    scene_rows.append({
        "scene_id":
            s.get("scene_id")
            or s.get("id"),

        "timestamp":
            s.get("scene_timestamp")
            or s.get("datetime")
            or s.get("timestamp"),

        "instrument":
            s.get("instrument"),

        "platform":
            s.get("platform"),

        "status":
            s.get("status"),

        "cloud_fields":
            json.dumps(
                cloud_fields,
                ensure_ascii=False,
            ),
    })


scene_df = pd.DataFrame(
    scene_rows
)

if len(scene_df):

    scene_df.to_csv(
        OUTDIR
        /
        "carbonmapper_scene_coverage.csv",
        index=False,
    )

    print(
        "\nCarbon Mapper scenes:"
    )

    print(
        scene_df
        .head(50)
        .to_string(index=False)
    )

else:

    print(
        "\nNo Carbon Mapper scene records "
        "returned by the query."
    )


# ============================================================
# 4. NASA EMIT CH4ENH V2 EXACT-DATE COVERAGE
# ============================================================

print("\n" + "=" * 100)
print("4. NASA EMIT CH4ENH V2 EXACT-DATE COVERAGE")
print("=" * 100)

cmr_url = (
    "https://cmr.earthdata.nasa.gov/"
    "search/granules.json"
)

cmr_params = {
    "collection_concept_id":
        EMIT_CH4ENH_COLLECTION,

    # NASA CMR point syntax = lon,lat
    "point":
        f"{lon},{lat}",

    "temporal":
        f"{start},{end}",

    "page_size":
        100,
}

r = session.get(
    cmr_url,
    params=cmr_params,
    timeout=TIMEOUT,
)

print(
    "CMR HTTP:",
    r.status_code
)

print(
    "Final URL:",
    r.url
)

r.raise_for_status()

emit_json = r.json()

save_json(
    OUTDIR
    /
    "emit_ch4enh_v2_exact_date.json",
    emit_json,
)

emit_entries = (
    emit_json
    .get("feed", {})
    .get("entry", [])
)

print(
    "\nExact-date EMIT CH4ENH granules:",
    len(emit_entries)
)

emit_rows = []

for e in emit_entries:

    emit_rows.append({
        "granule_id":
            e.get("id"),

        "title":
            e.get("title"),

        "time_start":
            e.get("time_start"),

        "time_end":
            e.get("time_end"),

        "updated":
            e.get("updated"),
    })


emit_df = pd.DataFrame(
    emit_rows
)

if len(emit_df):

    emit_df.to_csv(
        OUTDIR
        /
        "emit_exact_date_coverage.csv",
        index=False,
    )

    print(
        "\nEMIT coverage:"
    )

    print(
        emit_df.to_string(
            index=False
        )
    )

else:

    print(
        "\nNo exact-date EMIT CH4ENH V2 "
        "coverage at candidate point."
    )


# ============================================================
# 5. EMIT CONTEXT SEARCH ±3 DAYS
#
# This is CONTEXT ONLY.
# It is NOT used to call the S2 candidate negative.
# ============================================================

print("\n" + "=" * 100)
print("5. EMIT ±3-DAY CONTEXT SEARCH")
print("=" * 100)

context_start = (
    date
    -
    pd.Timedelta(days=3)
).strftime(
    "%Y-%m-%dT00:00:00Z"
)

context_end = (
    date
    +
    pd.Timedelta(days=3)
).strftime(
    "%Y-%m-%dT23:59:59Z"
)

context_params = {
    "collection_concept_id":
        EMIT_CH4ENH_COLLECTION,

    "point":
        f"{lon},{lat}",

    "temporal":
        f"{context_start},{context_end}",

    "page_size":
        100,
}

r2 = session.get(
    cmr_url,
    params=context_params,
    timeout=TIMEOUT,
)

print(
    "CMR HTTP:",
    r2.status_code
)

r2.raise_for_status()

emit_context_json = (
    r2.json()
)

save_json(
    OUTDIR
    /
    "emit_ch4enh_v2_plusminus3d.json",
    emit_context_json,
)

context_entries = (
    emit_context_json
    .get("feed", {})
    .get("entry", [])
)

print(
    "EMIT granules within ±3 days:",
    len(context_entries)
)

for e in context_entries:

    print(
        " ",
        e.get("title"),
        e.get("time_start")
    )


# ============================================================
# 6. PROVISIONAL METADATA SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("6. PROVISIONAL METADATA SUMMARY")
print("=" * 100)

cm_plume_http = (
    cm_plume_json.get(
        "_http_status"
    )
    if isinstance(cm_plume_json, dict)
    else None
)

cm_scene_http = (
    cm_scene_json.get(
        "_http_status"
    )
    if isinstance(cm_scene_json, dict)
    else None
)

nearest_plume_km = None
nearest_plume_id = None

if (
    len(plume_df)
    and plume_df[
        "distance_km"
    ].notna().any()
):

    q = (
        plume_df
        .dropna(
            subset=["distance_km"]
        )
        .sort_values(
            "distance_km"
        )
        .iloc[0]
    )

    nearest_plume_km = (
        float(
            q["distance_km"]
        )
    )

    nearest_plume_id = (
        q["plume_id"]
    )


summary = {
    "Pilot Candidate ID":
        CANDIDATE_ID,

    "Site":
        row["Site"],

    "Latitude":
        lat,

    "Longitude":
        lon,

    "Candidate Date":
        date.strftime(
            "%Y-%m-%d"
        ),

    "Actual S2 Datetime UTC":
        str(actual_s2),

    "S2 Clear Fraction":
        row["S2 Clear Fraction"],

    "S2 QA":
        row["Cloud/Snow QA Pass"],

    "CM plume records returned":
        len(cm_plumes),

    "CM nearest plume ID":
        nearest_plume_id,

    "CM nearest plume distance km":
        nearest_plume_km,

    "CM scene records returned":
        len(cm_scenes),

    "CM plume query HTTP":
        cm_plume_http,

    "CM scene query HTTP":
        cm_scene_http,

    "EMIT exact-date CH4ENH count":
        len(emit_entries),

    "EMIT +/-3d CH4ENH count":
        len(context_entries),

    "Final Validation Class":
        "U_UNKNOWN",

    "Evidence Grade":
        "U",

    "Validation Notes":
        (
            "Metadata audit only. "
            "Do not call negative until methane "
            "coverage/no-detection evidence is evaluated."
        ),
}


summary_df = pd.DataFrame(
    [summary]
)

summary_csv = (
    OUTDIR
    /
    "haynesville_P04_D03_metadata_summary.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False,
    encoding="utf-8-sig",
)

print(
    summary_df.T.to_string(
        header=False
    )
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 100)
print("METADATA VALIDATION COMPLETE")
print("=" * 100)

print(
    "\nOutput directory:"
)

print(
    OUTDIR
)

print(
    "\nSummary:"
)

print(
    summary_csv
)

print(
    "\nIMPORTANT:"
)

print(
    "U_UNKNOWN is intentional at this stage."
)

print(
    "A missing plume record alone is NOT "
    "treated as a negative."
)

