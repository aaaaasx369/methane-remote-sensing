import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from netCDF4 import Dataset
from pyproj import CRS, Transformer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "AVIRIS3_MethaneFuse_build"
)

AV3_DIR = (
    ROOT
    / "L2A_AVIRIS3_t0"
)

CH4_DIR = (
    ROOT
    / "L2B_REMATCH_CH4"
)

REMATCH = Path(
    "aviris3_l2b_spatial_rematch_candidates.csv"
)

QA = (
    ROOT
    / "AVIRIS3_MethaneFuse_test_58"
    / "qa_report.csv"
)

MANIFEST = Path(
    "aviris3_methanefuse_queries_58.csv"
)

OUT = Path(
    "aviris3_l2a_l2b_true_overlap_audit.csv"
)


# ============================================================
# HELPERS
# ============================================================

def scene_key(name):

    m = re.search(
        r"(AV3\d{8}t\d{6}_\d{3})",
        str(name)
    )

    return (
        m.group(1)
        if m
        else None
    )


def find_av3(scene):

    hits = [
        p
        for p in AV3_DIR.glob("*.nc")
        if scene_key(p.name) == scene
    ]

    if len(hits) != 1:

        raise RuntimeError(
            f"{scene}: "
            f"L2A files={len(hits)}"
        )

    return hits[0]


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


def read_l2a_geometry(path):

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

    # pixel-center coordinates
    dx = float(
        np.median(
            np.abs(
                np.diff(east)
            )
        )
    )

    dy = float(
        np.median(
            np.abs(
                np.diff(north)
            )
        )
    )

    return {
        "crs": crs,

        "left":
            float(east.min() - dx/2),

        "right":
            float(east.max() + dx/2),

        "bottom":
            float(north.min() - dy/2),

        "top":
            float(north.max() + dy/2),

        "dx": dx,
        "dy": dy,
    }


def transform_bbox(
    bbox,
    src_crs,
    dst_crs,
):

    left = bbox["left"]
    right = bbox["right"]
    bottom = bbox["bottom"]
    top = bbox["top"]

    # use corners + edge midpoints
    pts = [
        (left, bottom),
        (left, top),
        (right, bottom),
        (right, top),

        (
            (left+right)/2,
            bottom
        ),

        (
            (left+right)/2,
            top
        ),

        (
            left,
            (bottom+top)/2
        ),

        (
            right,
            (bottom+top)/2
        ),
    ]

    if CRS.from_user_input(
        src_crs
    ) == CRS.from_user_input(
        dst_crs
    ):

        xy = pts

    else:

        tr = (
            Transformer
            .from_crs(
                src_crs,
                dst_crs,
                always_xy=True
            )
        )

        xy = [
            tr.transform(x, y)
            for x, y in pts
        ]

    xs = [
        p[0] for p in xy
    ]

    ys = [
        p[1] for p in xy
    ]

    return {
        "left": min(xs),
        "right": max(xs),
        "bottom": min(ys),
        "top": max(ys),
    }


def intersection(
    a,
    b,
):

    left = max(
        a["left"],
        b["left"]
    )

    right = min(
        a["right"],
        b["right"]
    )

    bottom = max(
        a["bottom"],
        b["bottom"]
    )

    top = min(
        a["top"],
        b["top"]
    )

    width = max(
        0.0,
        right-left
    )

    height = max(
        0.0,
        top-bottom
    )

    return (
        left,
        bottom,
        right,
        top,
        width,
        height,
        width*height,
    )


# ============================================================
# PATCH-VALID POSITIVES
# ============================================================

qa = pd.read_csv(QA)
manifest = pd.read_csv(MANIFEST)

pass_ids = set(
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
        pass_ids
    )
].copy()

print(
    "Patch-valid positives:",
    len(pos)
)

print(
    "Positive scenes:",
    pos["scene_key"].nunique()
)


# ============================================================
# CANDIDATE TABLE
# ============================================================

cand = pd.read_csv(
    REMATCH
)

records = []


# ============================================================
# AUDIT EACH L2B TILE
# ============================================================

for _, row in cand.iterrows():

    l2a_scene = str(
        row[
            "l2a_scene_key"
        ]
    )

    l2b_scene = str(
        row[
            "l2b_scene_key"
        ]
    )

    ch4_name = str(
        row[
            "ch4_filename"
        ]
    )

    av3_path = find_av3(
        l2a_scene
    )

    ch4_path = (
        CH4_DIR
        / ch4_name
    )

    if not ch4_path.exists():

        print(
            "MISSING:",
            ch4_path.name
        )

        continue

    # --------------------------------------------------------
    # L2A geometry
    # --------------------------------------------------------

    a = read_l2a_geometry(
        av3_path
    )

    # --------------------------------------------------------
    # L2B geometry
    # --------------------------------------------------------

    with rasterio.open(
        ch4_path
    ) as ds:

        l2b_crs = ds.crs

        b = {
            "left":
                ds.bounds.left,

            "right":
                ds.bounds.right,

            "bottom":
                ds.bounds.bottom,

            "top":
                ds.bounds.top,
        }

        width_px = ds.width
        height_px = ds.height

        res_x = abs(
            ds.transform.a
        )

        res_y = abs(
            ds.transform.e
        )

    if l2b_crs is None:

        raise RuntimeError(
            f"{ch4_name}: "
            "missing CRS"
        )

    # L2A extent transformed to L2B CRS
    a2 = transform_bbox(
        a,
        a["crs"],
        l2b_crs
    )

    (
        ileft,
        ibottom,
        iright,
        itop,
        iw,
        ih,
        iarea,
    ) = intersection(
        a2,
        b
    )

    l2b_area = (
        max(
            0,
            b["right"]
            - b["left"]
        )
        *
        max(
            0,
            b["top"]
            - b["bottom"]
        )
    )

    overlap_fraction_l2b = (
        iarea / l2b_area
        if l2b_area > 0
        else 0
    )

    # --------------------------------------------------------
    # Which known positive points lie in this L2B tile?
    # --------------------------------------------------------

    scene_pos = pos[
        pos[
            "scene_key"
        ] == l2a_scene
    ]

    to_l2b = (
        Transformer
        .from_crs(
            "EPSG:4326",
            l2b_crs,
            always_xy=True
        )
    )

    contained_ids = []

    for _, p in scene_pos.iterrows():

        x, y = (
            to_l2b.transform(
                float(
                    p["query_lon"]
                ),
                float(
                    p["query_lat"]
                )
            )
        )

        # Require full 480m patch
        # geometrically inside L2B tile:
        margin = 240.0

        full_patch_inside = (
            x >= b["left"] + margin
            and
            x <= b["right"] - margin
            and
            y >= b["bottom"] + margin
            and
            y <= b["top"] - margin
        )

        if full_patch_inside:

            contained_ids.append(
                str(
                    p["query_id"]
                )
            )

    rec = {
        "l2a_scene_key":
            l2a_scene,

        "l2b_scene_key":
            l2b_scene,

        "ch4_filename":
            ch4_name,

        "l2b_crs":
            str(
                l2b_crs
            ),

        "l2b_width_px":
            width_px,

        "l2b_height_px":
            height_px,

        "l2b_res_x_m":
            res_x,

        "l2b_res_y_m":
            res_y,

        "overlap_width_m":
            iw,

        "overlap_height_m":
            ih,

        "overlap_area_km2":
            iarea / 1e6,

        "overlap_fraction_l2b":
            overlap_fraction_l2b,

        "can_fit_480m_patch":
            (
                iw >= 480
                and
                ih >= 480
            ),

        "n_patchvalid_positive_inside":
            len(
                contained_ids
            ),

        "patchvalid_positive_ids":
            ";".join(
                contained_ids
            ),
    }

    records.append(
        rec
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


# ============================================================
# REPORT
# ============================================================

print("\n")
print("=" * 78)
print("TRUE L2A ↔ L2B OVERLAP")
print("=" * 78)

print(
    "Rows:",
    len(out)
)

print(
    "Tiles with overlap:",
    int(
        (
            out[
                "overlap_area_km2"
            ] > 0
        ).sum()
    )
)

print(
    "Tiles able to fit 480m patch:",
    int(
        out[
            "can_fit_480m_patch"
        ].sum()
    )
)

print(
    "Tiles containing >=1 "
    "known positive full patch:",
    int(
        (
            out[
                "n_patchvalid_positive_inside"
            ] > 0
        ).sum()
    )
)


for scene in sorted(
    out[
        "l2a_scene_key"
    ].unique()
):

    x = out[
        out[
            "l2a_scene_key"
        ] == scene
    ].copy()

    x = x.sort_values(
        [
            "n_patchvalid_positive_inside",
            "overlap_area_km2",
        ],
        ascending=[
            False,
            False,
        ]
    )

    print("\n")
    print("=" * 78)
    print(scene)
    print("=" * 78)

    print(
        x[
            [
                "l2b_scene_key",
                "overlap_area_km2",
                "overlap_width_m",
                "overlap_height_m",
                "can_fit_480m_patch",
                "n_patchvalid_positive_inside",
                "patchvalid_positive_ids",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# POSITIVE COVERAGE
# ============================================================

print("\n")
print("=" * 78)
print("POSITIVE -> L2B TILE COVERAGE")
print("=" * 78)

all_covered = set()

for _, r in out.iterrows():

    ids = str(
        r[
            "patchvalid_positive_ids"
        ]
    )

    if (
        ids
        and
        ids.lower()
        != "nan"
    ):

        for qid in ids.split(";"):

            if qid:
                all_covered.add(
                    qid
                )

for _, p in pos.iterrows():

    qid = str(
        p["query_id"]
    )

    print(
        qid,
        "|",
        p["scene_key"],
        "| L2B full-patch coverage =",
        qid in all_covered
    )

print(
    "\nKnown positives with "
    "L2B full-patch coverage:",
    len(all_covered),
    "/",
    len(pos)
)

print("\nSaved:")
print(OUT.resolve())
