import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import earthaccess
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import CRS, Transformer


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("AVIRIS3_MethaneFuse_build")
AV3_DIR = ROOT / "L2A_AVIRIS3_t0"

QA = (
    ROOT
    / "AVIRIS3_MethaneFuse_test_58"
    / "qa_report.csv"
)

MANIFEST = Path(
    "aviris3_methanefuse_queries_58.csv"
)

OUT = Path(
    "aviris3_l2b_spatial_rematch_candidates.csv"
)

L2B_CONCEPT_ID = (
    "C3236537512-ORNL_CLOUD"
)

TIME_WINDOW_MIN = 30


# ============================================================
# HELPERS
# ============================================================

def scene_key_from_name(name):

    m = re.search(
        r"(AV3\d{8}t\d{6}_\d{3})",
        str(name)
    )

    return m.group(1) if m else None


def parse_scene_time(scene):

    m = re.search(
        r"AV3(\d{8})t(\d{6})",
        scene
    )

    if not m:
        raise ValueError(scene)

    return datetime.strptime(
        m.group(1) + m.group(2),
        "%Y%m%d%H%M%S"
    ).replace(
        tzinfo=timezone.utc
    )


def read_av3_crs(ds):

    v = ds.variables[
        "transverse_mercator"
    ]

    attrs = {
        k: getattr(v, k)
        for k in v.ncattrs()
    }

    for key in [
        "spatial_ref",
        "crs_wkt"
    ]:

        if key in attrs:
            try:
                return CRS.from_wkt(
                    str(attrs[key])
                )
            except Exception:
                pass

    return CRS.from_cf(
        attrs
    )


def find_av3(scene):

    hits = [
        p for p
        in AV3_DIR.glob("*.nc")
        if scene_key_from_name(
            p.name
        ) == scene
    ]

    if len(hits) != 1:
        raise RuntimeError(
            f"{scene}: AV3 hits={len(hits)}"
        )

    return hits[0]


def av3_bbox_wgs84(path):

    with Dataset(
        str(path),
        "r"
    ) as ds:

        east = np.asarray(
            ds.variables[
                "easting"
            ][:],
            dtype=float
        )

        north = np.asarray(
            ds.variables[
                "northing"
            ][:],
            dtype=float
        )

        crs = read_av3_crs(ds)

    to_ll = Transformer.from_crs(
        crs,
        "EPSG:4326",
        always_xy=True
    )

    corners = [
        (east.min(), north.min()),
        (east.min(), north.max()),
        (east.max(), north.min()),
        (east.max(), north.max()),
    ]

    ll = [
        to_ll.transform(x, y)
        for x, y in corners
    ]

    lons = [
        x[0] for x in ll
    ]

    lats = [
        x[1] for x in ll
    ]

    return (
        float(min(lons)),
        float(min(lats)),
        float(max(lons)),
        float(max(lats)),
    )


def get_links(granule):

    try:
        links = granule.data_links()
    except Exception:
        links = []

    return [
        str(x)
        for x in links
    ]


# ============================================================
# LOGIN
# ============================================================

auth = earthaccess.login(
    strategy="netrc"
)

print(
    "Authenticated:",
    auth.authenticated
)


# ============================================================
# GET THE 10 PATCH-VALID POSITIVES
# ============================================================

qa = pd.read_csv(QA)
manifest = pd.read_csv(MANIFEST)

pass_positive_ids = set(
    qa.loc[
        (
            qa["status"]
            .astype(str)
            .str.upper()
            == "PASS"
        )
        &
        (
            pd.to_numeric(
                qa["label"],
                errors="coerce"
            )
            == 1
        ),
        "id"
    ].astype(str)
)

pos = manifest[
    manifest[
        "query_id"
    ].astype(str).isin(
        pass_positive_ids
    )
].copy()

scenes = sorted(
    pos["scene_key"]
    .drop_duplicates()
)

print("\nPatch-valid scenes:")
for s in scenes:
    print(" ", s)


# ============================================================
# SEARCH L2B BY ACTUAL L2A FOOTPRINT
# ============================================================

records = []

for scene in scenes:

    print("\n")
    print("=" * 78)
    print(scene)
    print("=" * 78)

    av3 = find_av3(scene)

    bbox = av3_bbox_wgs84(
        av3
    )

    t0 = parse_scene_time(
        scene
    )

    start = (
        t0
        - timedelta(
            minutes=TIME_WINDOW_MIN
        )
    )

    end = (
        t0
        + timedelta(
            minutes=TIME_WINDOW_MIN
        )
    )

    print("L2A:", av3.name)

    print(
        "BBox:",
        tuple(
            round(x, 6)
            for x in bbox
        )
    )

    print(
        "Search time:",
        start.isoformat(),
        "to",
        end.isoformat()
    )

    results = earthaccess.search_data(
        concept_id=L2B_CONCEPT_ID,

        bounding_box=bbox,

        temporal=(
            start,
            end
        ),

        count=100
    )

    print(
        "CMR granules intersecting:",
        len(results)
    )

    scene_candidates = []

    for gi, granule in enumerate(
        results
    ):

        links = get_links(
            granule
        )

        ch4 = [
            u for u in links
            if (
                "_CH4_ORT.tif"
                in u
                and
                "_QL"
                not in u
            )
        ]

        unc = [
            u for u in links
            if "_CH4_UNC_ORT.tif"
            in u
        ]

        sns = [
            u for u in links
            if "_CH4_SNS_ORT.tif"
            in u
        ]

        if not ch4:
            continue

        for ch4_url in ch4:

            filename = Path(
                urlparse(
                    ch4_url
                ).path
            ).name

            candidate_scene = (
                scene_key_from_name(
                    filename
                )
            )

            rec = {
                "l2a_scene_key":
                    scene,

                "l2a_filename":
                    av3.name,

                "l2a_min_lon":
                    bbox[0],

                "l2a_min_lat":
                    bbox[1],

                "l2a_max_lon":
                    bbox[2],

                "l2a_max_lat":
                    bbox[3],

                "l2a_t0":
                    t0.isoformat(),

                "l2b_scene_key":
                    candidate_scene,

                "ch4_filename":
                    filename,

                "ch4_url":
                    ch4_url,

                "unc_url":
                    unc[0]
                    if unc
                    else "",

                "sns_url":
                    sns[0]
                    if sns
                    else "",
            }

            records.append(
                rec
            )

            scene_candidates.append(
                rec
            )

    print(
        "CH4 candidates:",
        len(scene_candidates)
    )

    for r in scene_candidates:

        same_suffix = (
            r[
                "l2b_scene_key"
            ]
            == scene
        )

        print(
            " ",
            r["l2b_scene_key"],
            "| same key =",
            same_suffix,
            "|",
            r["ch4_filename"]
        )


# ============================================================
# SAVE
# ============================================================

out = pd.DataFrame(
    records
)

out.to_csv(
    OUT,
    index=False
)

print("\n")
print("=" * 78)
print("SUMMARY")
print("=" * 78)

print(
    "Rows:",
    len(out)
)

if len(out):

    print(
        "\nCandidates per L2A scene:"
    )

    print(
        out[
            "l2a_scene_key"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nL2A -> L2B scene keys"
    )

    for scene in scenes:

        x = out[
            out[
                "l2a_scene_key"
            ] == scene
        ]

        print(
            "\n",
            scene
        )

        print(
            x[
                [
                    "l2b_scene_key",
                    "ch4_filename"
                ]
            ].to_string(
                index=False
            )
        )

print("\nSaved:")
print(OUT.resolve())
