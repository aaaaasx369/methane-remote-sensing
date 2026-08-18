import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import rasterio

from netCDF4 import Dataset
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("AVIRIS3_MethaneFuse_build")

ORIGINAL_MANIFEST = Path(
    "aviris3_methanefuse_queries_58.csv"
)

QA_REPORT = (
    ROOT
    / "AVIRIS3_MethaneFuse_test_58"
    / "qa_report.csv"
)

OFFICIAL_POS = Path(
    "aviris3_carbonmapper_positive_queries.csv"
)

AV3_DIR = ROOT / "L2A_AVIRIS3_t0"
EMIT90_DIR = ROOT / "L2A_EMIT_t90"
EMIT180_DIR = ROOT / "L2A_EMIT_t180"

ORT_DIR = ROOT / "L2B_MATCHED_CH4"
UNC_DIR = ROOT / "L2B_MATCHED_UNC"
SNS_DIR = ROOT / "L2B_MATCHED_SNS"


OUT_POS = Path(
    "aviris3_patchvalid_positives_10.csv"
)

OUT_NEG = Path(
    "aviris3_patchvalid_negatives_10.csv"
)

OUT_FINAL = Path(
    "aviris3_methanefuse_queries_patchvalid_20.csv"
)

OUT_AUDIT = Path(
    "aviris3_patchvalid_negative_candidate_audit.csv"
)


# ------------------------------------------------------------
# Same geometry as tested converter
# ------------------------------------------------------------

CHIP_SIZE = 512
SCALE_M = 60.0
QUERY_PX = 8

MISSING_THRESH = 0.25

QUERY_SIZE_M = 480.0

MIN_PLUME_DISTANCE_M = 800.0
MIN_QA_FRACTION = 0.85

SNS_MIN = 0.50
SNS_MAX = 1.50


# ============================================================
# FILE HELPERS
# ============================================================

def filename_from_url(url):

    return Path(
        urlparse(str(url)).path
    ).name


def locate_file(url, folder):

    name = filename_from_url(url)

    p = folder / name

    if p.exists():
        return p

    hits = list(
        folder.glob(f"*{name}*")
    )

    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"Could not locate {name} "
        f"in {folder}"
    )


def scene_key_from_name(name):

    m = re.search(
        r"(AV3\d{8}t\d{6}_\d{3})",
        str(name)
    )

    return (
        m.group(1)
        if m
        else None
    )


def build_scene_map(folder):

    out = {}

    for p in folder.glob("*"):

        k = scene_key_from_name(
            p.name
        )

        if k:
            out[k] = p

    return out


# ============================================================
# COORDINATE HELPERS
# ============================================================

def get_utm_crs(lat, lon):

    zone = (
        int(
            (lon + 180) / 6
        )
        + 1
    )

    epsg = (
        32600 + zone
        if lat >= 0
        else 32700 + zone
    )

    return CRS.from_epsg(
        epsg
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
        "crs_wkt",
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


def nearest_indices(
    coords,
    targets,
):

    coords = np.asarray(
        coords,
        dtype=np.float64
    )

    targets = np.asarray(
        targets,
        dtype=np.float64
    )

    ascending = (
        coords[-1]
        >= coords[0]
    )

    if not ascending:

        rev = nearest_indices(
            coords[::-1],
            targets
        )

        return (
            len(coords)
            - 1
            - rev
        )

    pos = np.searchsorted(
        coords,
        targets
    )

    pos = np.clip(
        pos,
        1,
        len(coords) - 1
    )

    left = coords[
        pos - 1
    ]

    right = coords[
        pos
    ]

    choose_right = (
        np.abs(
            targets - right
        )
        <
        np.abs(
            targets - left
        )
    )

    return np.where(
        choose_right,
        pos,
        pos - 1
    )


def central_8_xy(cx, cy):

    half_size = (
        CHIP_SIZE
        * SCALE_M
        / 2.0
    )

    xx = np.linspace(
        cx - half_size,
        cx + half_size,
        CHIP_SIZE
    )

    yy = np.linspace(
        cy + half_size,
        cy - half_size,
        CHIP_SIZE
    )

    center = (
        CHIP_SIZE // 2
    )

    half = (
        QUERY_PX // 2
    )

    sl = slice(
        center - half,
        center - half + QUERY_PX
    )

    return (
        xx[sl],
        yy[sl]
    )


# ============================================================
# AVIRIS-3 LIGHTWEIGHT COVERAGE
# ============================================================

class AV3Coverage:

    def __init__(self, path):

        self.path = Path(path)

        with Dataset(
            str(path),
            "r"
        ) as ds:

            self.east = np.asarray(
                ds.variables[
                    "easting"
                ][:],
                dtype=np.float64
            )

            self.north = np.asarray(
                ds.variables[
                    "northing"
                ][:],
                dtype=np.float64
            )

            self.crs = read_av3_crs(
                ds
            )

            g = ds.groups[
                "reflectance"
            ]

            wave = np.asarray(
                g.variables[
                    "wavelength"
                ][:],
                dtype=np.float64
            )

            # Representative visible band,
            # only for detecting fill/nodata.
            bi = int(
                np.argmin(
                    np.abs(
                        wave - 450.0
                    )
                )
            )

            arr = g.variables[
                "reflectance"
            ][
                bi,
                :,
                :
            ]

            if np.ma.isMaskedArray(
                arr
            ):
                arr = arr.filled(
                    np.nan
                )

            self.rep = np.asarray(
                arr,
                dtype=np.float32
            )

        self.to_src = (
            Transformer
            .from_crs(
                "EPSG:4326",
                self.crs,
                always_xy=True,
            )
        )

        self.dx = float(
            np.median(
                np.abs(
                    np.diff(
                        self.east
                    )
                )
            )
        )

        self.dy = float(
            np.median(
                np.abs(
                    np.diff(
                        self.north
                    )
                )
            )
        )

        self.emin = float(
            self.east.min()
        )

        self.emax = float(
            self.east.max()
        )

        self.nmin = float(
            self.north.min()
        )

        self.nmax = float(
            self.north.max()
        )

    def missing_ratio(
        self,
        lat,
        lon,
    ):

        cx, cy = (
            self.to_src.transform(
                lon,
                lat
            )
        )

        xx, yy = central_8_xy(
            cx,
            cy
        )

        outside = (
            (xx[None, :]
             < self.emin
             - self.dx / 2)
            |
            (xx[None, :]
             > self.emax
             + self.dx / 2)
            |
            (yy[:, None]
             < self.nmin
             - self.dy / 2)
            |
            (yy[:, None]
             > self.nmax
             + self.dy / 2)
        )

        xi = nearest_indices(
            self.east,
            xx
        )

        yi = nearest_indices(
            self.north,
            yy
        )

        vals = self.rep[
            yi[:, None],
            xi[None, :]
        ]

        bad_value = (
            ~np.isfinite(vals)
            |
            (vals <= -9990)
        )

        bad = (
            outside
            |
            bad_value
        )

        return float(
            bad.mean()
        )


# ============================================================
# EMIT LIGHTWEIGHT COVERAGE
# ============================================================

class EMITCoverage:

    def __init__(
        self,
        path,
        bbox,
    ):

        self.path = Path(path)

        min_lat, max_lat, \
        min_lon, max_lon = bbox

        # only need nearby points.
        margin = 0.03

        with Dataset(
            str(path),
            "r"
        ) as ds:

            loc = ds.groups[
                "location"
            ]

            lat = np.asarray(
                loc.variables[
                    "lat"
                ][:],
                dtype=np.float64
            )

            lon = np.asarray(
                loc.variables[
                    "lon"
                ][:],
                dtype=np.float64
            )

            wave = np.asarray(
                ds.groups[
                    "sensor_band_parameters"
                ].variables[
                    "wavelengths"
                ][:],
                dtype=np.float64
            )

            bi = int(
                np.argmin(
                    np.abs(
                        wave - 450.0
                    )
                )
            )

            rfl = ds.variables[
                "reflectance"
            ][
                :,
                :,
                bi
            ]

            if np.ma.isMaskedArray(
                rfl
            ):
                rfl = rfl.filled(
                    np.nan
                )

            rfl = np.asarray(
                rfl,
                dtype=np.float32
            )

        valid_geo = (
            np.isfinite(lat)
            &
            np.isfinite(lon)
            &
            (lat >= min_lat - margin)
            &
            (lat <= max_lat + margin)
            &
            (lon >= min_lon - margin)
            &
            (lon <= max_lon + margin)
        )

        if not np.any(
            valid_geo
        ):
            raise RuntimeError(
                f"No nearby EMIT "
                f"geolocation for "
                f"{path.name}"
            )

        self.raw_flat = (
            np.flatnonzero(
                valid_geo.ravel()
            )
        )

        coords = np.stack(
            [
                lat[
                    valid_geo
                ],
                lon[
                    valid_geo
                ],
            ],
            axis=1
        )

        self.tree = cKDTree(
            coords
        )

        self.rep_flat = (
            rfl.ravel()
        )

    def missing_ratio(
        self,
        lat,
        lon,
    ):

        utm = get_utm_crs(
            lat,
            lon
        )

        to_utm = (
            Transformer
            .from_crs(
                "EPSG:4326",
                utm,
                always_xy=True,
            )
        )

        to_wgs = (
            Transformer
            .from_crs(
                utm,
                "EPSG:4326",
                always_xy=True,
            )
        )

        cx, cy = (
            to_utm.transform(
                lon,
                lat
            )
        )

        xx, yy = central_8_xy(
            cx,
            cy
        )

        mx, my = np.meshgrid(
            xx,
            yy
        )

        qlon, qlat = (
            to_wgs.transform(
                mx,
                my
            )
        )

        q = np.stack(
            [
                qlat.ravel(),
                qlon.ravel(),
            ],
            axis=1
        )

        dist, idx = (
            self.tree.query(
                q,
                distance_upper_bound=0.001,
            )
        )

        missing = (
            ~np.isfinite(
                dist
            )
        )

        safe = idx.copy()

        safe[
            missing
        ] = 0

        raw_idx = (
            self.raw_flat[
                safe
            ]
        )

        vals = (
            self.rep_flat[
                raw_idx
            ]
        )

        bad_value = (
            ~np.isfinite(vals)
            |
            (vals <= -9990)
        )

        bad = (
            missing
            |
            bad_value
        )

        return float(
            bad.mean()
        )


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

manifest = pd.read_csv(
    ORIGINAL_MANIFEST
)

qa = pd.read_csv(
    QA_REPORT
)

official = pd.read_csv(
    OFFICIAL_POS
)


# ============================================================
# LOCK THE 10 TRUE PATCH-VALID POSITIVES
# ============================================================

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

positive = manifest[
    manifest[
        "query_id"
    ].astype(str).isin(
        pass_positive_ids
    )
].copy()

positive = positive.sort_values(
    [
        "scene_key",
        "query_id",
    ]
).reset_index(
    drop=True
)

print("\n================================")
print("PATCH-VALID POSITIVES")
print("================================")

print(
    "Rows:",
    len(positive)
)

print(
    "Scenes:",
    positive[
        "scene_key"
    ].nunique()
)

print("\nPositive count per scene:")

need_by_scene = (
    positive[
        "scene_key"
    ]
    .value_counts()
    .sort_index()
)

print(
    need_by_scene
    .to_string()
)

if len(positive) != 10:
    raise RuntimeError(
        f"Expected 10 patch-valid "
        f"positives; got {len(positive)}"
    )

positive.to_csv(
    OUT_POS,
    index=False
)


# ============================================================
# OFFICIAL PLUME COORDS
# ============================================================

official[
    "query_lat"
] = pd.to_numeric(
    official[
        "query_lat"
    ],
    errors="coerce"
)

official[
    "query_lon"
] = pd.to_numeric(
    official[
        "query_lon"
    ],
    errors="coerce"
)

official = official.dropna(
    subset=[
        "query_lat",
        "query_lon",
    ]
)

official_ll = (
    official[
        [
            "query_lon",
            "query_lat",
        ]
    ].to_numpy(
        dtype=np.float64
    )
)

print(
    "\nOfficial plume points "
    "used for exclusion:",
    len(official_ll)
)


# ============================================================
# L2B FILE MAPS
# ============================================================

ort_map = build_scene_map(
    ORT_DIR
)

unc_map = build_scene_map(
    UNC_DIR
)

sns_map = build_scene_map(
    SNS_DIR
)


# ============================================================
# GENERATE PATCH-VALID NEGATIVES
# ============================================================

negative_records = []
candidate_audit = []

neg_counter = 1


for scene, n_needed in (
    need_by_scene.items()
):

    print("\n")
    print("=" * 75)
    print(
        "SCENE:",
        scene
    )
    print(
        "Need negatives:",
        n_needed
    )
    print("=" * 75)

    template = (
        positive[
            positive[
                "scene_key"
            ] == scene
        ]
        .iloc[0]
        .copy()
    )

    # --------------------------------------------------------
    # Resolve temporal source files
    # --------------------------------------------------------

    av3_nc = locate_file(
        template[
            "av3_l2a_rfl_url"
        ],
        AV3_DIR
    )

    emit90_nc = locate_file(
        template[
            "emit_t90_rfl_url"
        ],
        EMIT90_DIR
    )

    emit180_nc = locate_file(
        template[
            "emit_t180_rfl_url"
        ],
        EMIT180_DIR
    )

    if scene not in ort_map:
        raise RuntimeError(
            f"{scene}: missing ORT"
        )

    if scene not in unc_map:
        raise RuntimeError(
            f"{scene}: missing UNC"
        )

    if scene not in sns_map:
        raise RuntimeError(
            f"{scene}: missing SNS"
        )

    # --------------------------------------------------------
    # Read AVIRIS-3 methane QA rasters
    # --------------------------------------------------------

    with rasterio.open(
        ort_map[scene]
    ) as ds:

        ch4 = ds.read(
            1
        ).astype(
            np.float64
        )

        transform = ds.transform
        crs = ds.crs

        nodata_ch4 = ds.nodata

        height = ds.height
        width = ds.width

        dx = abs(
            ds.transform.a
        )

        dy = abs(
            ds.transform.e
        )

    with rasterio.open(
        unc_map[scene]
    ) as ds:

        unc = ds.read(
            1
        ).astype(
            np.float64
        )

        nodata_unc = ds.nodata

    with rasterio.open(
        sns_map[scene]
    ) as ds:

        sns = ds.read(
            1
        ).astype(
            np.float64
        )

        nodata_sns = ds.nodata

    if not (
        ch4.shape
        ==
        unc.shape
        ==
        sns.shape
    ):
        raise RuntimeError(
            f"{scene}: "
            "L2B shape mismatch"
        )

    if crs is None:
        raise RuntimeError(
            f"{scene}: "
            "L2B has no CRS"
        )

    valid = (
        np.isfinite(ch4)
        &
        np.isfinite(unc)
        &
        np.isfinite(sns)
    )

    if nodata_ch4 is not None:
        valid &= (
            ch4
            != nodata_ch4
        )

    if nodata_unc is not None:
        valid &= (
            unc
            != nodata_unc
        )

    if nodata_sns is not None:
        valid &= (
            sns
            != nodata_sns
        )

    valid &= (
        ch4 != -9999
    )

    valid &= (
        unc != -9999
    )

    valid &= (
        sns != -9999
    )

    valid &= (
        unc != -1
    )

    valid &= (
        sns != -1
    )

    valid &= (
        unc > 0
    )

    valid &= (
        sns > 0
    )

    # --------------------------------------------------------
    # Exact 480m L2B QA window
    # --------------------------------------------------------

    win_w = max(
        1,
        int(
            round(
                QUERY_SIZE_M
                / dx
            )
        )
    )

    win_h = max(
        1,
        int(
            round(
                QUERY_SIZE_M
                / dy
            )
        )
    )

    half_w = (
        win_w // 2
    )

    half_h = (
        win_h // 2
    )

    stride_x = max(
        1,
        win_w // 2
    )

    stride_y = max(
        1,
        win_h // 2
    )

    print(
        "L2B pixel:",
        round(dx, 3),
        "x",
        round(dy, 3),
        "m"
    )

    print(
        "L2B 480m window:",
        win_h,
        "x",
        win_w,
        "pixels"
    )

    # --------------------------------------------------------
    # All official plume origins -> L2B CRS
    # --------------------------------------------------------

    to_xy = (
        Transformer
        .from_crs(
            "EPSG:4326",
            crs,
            always_xy=True,
        )
    )

    to_ll = (
        Transformer
        .from_crs(
            crs,
            "EPSG:4326",
            always_xy=True,
        )
    )

    plume_xy = []

    for lon, lat in official_ll:

        x, y = (
            to_xy.transform(
                float(lon),
                float(lat)
            )
        )

        plume_xy.append(
            (
                float(x),
                float(y)
            )
        )

    # --------------------------------------------------------
    # Candidate background patches
    # --------------------------------------------------------

    candidates = []

    r_start = half_h
    r_stop = (
        height
        - (
            win_h
            - half_h
        )
        + 1
    )

    c_start = half_w
    c_stop = (
        width
        - (
            win_w
            - half_w
        )
        + 1
    )

    for r in range(
        r_start,
        max(
            r_start,
            r_stop
        ),
        stride_y,
    ):

        for c in range(
            c_start,
            max(
                c_start,
                c_stop
            ),
            stride_x,
        ):

            y0 = (
                r - half_h
            )

            x0 = (
                c - half_w
            )

            y1 = (
                y0 + win_h
            )

            x1 = (
                x0 + win_w
            )

            if (
                y0 < 0
                or x0 < 0
                or y1 > height
                or x1 > width
            ):
                continue

            vm = valid[
                y0:y1,
                x0:x1
            ]

            qa_fraction = float(
                vm.mean()
            )

            if (
                qa_fraction
                < MIN_QA_FRACTION
            ):
                continue

            ch = ch4[
                y0:y1,
                x0:x1
            ][vm]

            un = unc[
                y0:y1,
                x0:x1
            ][vm]

            sn = sns[
                y0:y1,
                x0:x1
            ][vm]

            if ch.size == 0:
                continue

            sns_p50 = float(
                np.percentile(
                    sn,
                    50
                )
            )

            if not (
                SNS_MIN
                <= sns_p50
                <= SNS_MAX
            ):
                continue

            snr = (
                np.maximum(
                    ch,
                    0
                )
                /
                np.maximum(
                    un,
                    1e-6
                )
            )

            snr_p99 = float(
                np.percentile(
                    snr,
                    99
                )
            )

            ch4_p99 = float(
                np.percentile(
                    ch,
                    99
                )
            )

            x, y = (
                rasterio
                .transform
                .xy(
                    transform,
                    r,
                    c,
                    offset="center"
                )
            )

            x = float(x)
            y = float(y)

            nearest_plume = min(
                np.hypot(
                    x - px,
                    y - py
                )
                for px, py
                in plume_xy
            )

            if (
                nearest_plume
                <
                MIN_PLUME_DISTANCE_M
            ):
                continue

            lon, lat = (
                to_ll.transform(
                    x,
                    y
                )
            )

            candidates.append({
                "scene_key":
                    scene,

                "row_px":
                    r,

                "col_px":
                    c,

                "x":
                    x,

                "y":
                    y,

                "query_lon":
                    float(lon),

                "query_lat":
                    float(lat),

                "qa_valid_fraction":
                    qa_fraction,

                "sns_p50":
                    sns_p50,

                "negative_snr_p99":
                    snr_p99,

                "negative_ch4_p99":
                    ch4_p99,

                "nearest_official_plume_m":
                    float(
                        nearest_plume
                    ),
            })

    print(
        "Raw CH4/QA candidates:",
        len(candidates)
    )

    if not candidates:
        raise RuntimeError(
            f"{scene}: "
            "no CH4/QA candidates"
        )

    cdf = pd.DataFrame(
        candidates
    )

    # --------------------------------------------------------
    # Build lightweight temporal coverage objects
    # only around candidate geographic extent
    # --------------------------------------------------------

    bbox = (
        float(
            cdf[
                "query_lat"
            ].min()
        ),

        float(
            cdf[
                "query_lat"
            ].max()
        ),

        float(
            cdf[
                "query_lon"
            ].min()
        ),

        float(
            cdf[
                "query_lon"
            ].max()
        ),
    )

    print(
        "Building t0 coverage..."
    )

    av3_cov = AV3Coverage(
        av3_nc
    )

    print(
        "Building t90 coverage..."
    )

    emit90_cov = EMITCoverage(
        emit90_nc,
        bbox
    )

    print(
        "Building t180 coverage..."
    )

    emit180_cov = EMITCoverage(
        emit180_nc,
        bbox
    )

    # --------------------------------------------------------
    # Lowest signal first
    # --------------------------------------------------------

    cdf = cdf.sort_values(
        [
            "negative_snr_p99",
            "qa_valid_fraction",
            "nearest_official_plume_m",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    patchvalid = []

    reject_t0 = 0
    reject_t90 = 0
    reject_t180 = 0

    for _, cand in cdf.iterrows():

        lat = float(
            cand[
                "query_lat"
            ]
        )

        lon = float(
            cand[
                "query_lon"
            ]
        )

        m0 = (
            av3_cov
            .missing_ratio(
                lat,
                lon
            )
        )

        rec = cand.to_dict()

        rec[
            "t0_missing_ratio_precheck"
        ] = m0

        if (
            m0
            > MISSING_THRESH
        ):

            rec[
                "coverage_status"
            ] = "FAIL_T0"

            reject_t0 += 1

            candidate_audit.append(
                rec
            )

            continue

        m90 = (
            emit90_cov
            .missing_ratio(
                lat,
                lon
            )
        )

        rec[
            "t90_missing_ratio_precheck"
        ] = m90

        if (
            m90
            > MISSING_THRESH
        ):

            rec[
                "coverage_status"
            ] = "FAIL_T90"

            reject_t90 += 1

            candidate_audit.append(
                rec
            )

            continue

        m180 = (
            emit180_cov
            .missing_ratio(
                lat,
                lon
            )
        )

        rec[
            "t180_missing_ratio_precheck"
        ] = m180

        if (
            m180
            > MISSING_THRESH
        ):

            rec[
                "coverage_status"
            ] = "FAIL_T180"

            reject_t180 += 1

            candidate_audit.append(
                rec
            )

            continue

        rec[
            "coverage_status"
        ] = "PASS"

        patchvalid.append(
            rec
        )

        candidate_audit.append(
            rec
        )

    pv = pd.DataFrame(
        patchvalid
    )

    print(
        "Patch-valid candidates:",
        len(pv)
    )

    print(
        "Rejected:"
        f" t0={reject_t0},"
        f" t90={reject_t90},"
        f" t180={reject_t180}"
    )

    if len(pv) < n_needed:

        raise RuntimeError(
            f"{scene}: need "
            f"{n_needed} negatives "
            f"but only "
            f"{len(pv)} "
            "patch-valid candidates"
        )

    # --------------------------------------------------------
    # Greedy spatial separation
    # --------------------------------------------------------

    chosen = []

    used_sep = None

    for min_sep in [
        600.0,
        480.0,
        300.0,
        0.0,
    ]:

        chosen = []

        for _, cand in (
            pv.iterrows()
        ):

            if (
                len(chosen)
                >= n_needed
            ):
                break

            ok = True

            for old in chosen:

                d = np.hypot(
                    float(
                        cand["x"]
                    )
                    -
                    float(
                        old["x"]
                    ),

                    float(
                        cand["y"]
                    )
                    -
                    float(
                        old["y"]
                    )
                )

                if d < min_sep:
                    ok = False
                    break

            if ok:
                chosen.append(
                    cand.to_dict()
                )

        if (
            len(chosen)
            >= n_needed
        ):
            used_sep = min_sep
            break

    if len(chosen) < n_needed:

        raise RuntimeError(
            f"{scene}: spatial "
            "selection failed"
        )

    print(
        "Selected:",
        len(chosen),
        "| minimum center "
        "separation:",
        used_sep,
        "m"
    )

    # --------------------------------------------------------
    # Create negative manifest records
    # --------------------------------------------------------

    for cand in (
        chosen[
            :n_needed
        ]
    ):

        rec = template.copy()

        rec[
            "query_id"
        ] = (
            f"AV3_NEG_PV_"
            f"{neg_counter:04d}"
        )

        rec[
            "label"
        ] = 0

        rec[
            "query_lat"
        ] = cand[
            "query_lat"
        ]

        rec[
            "query_lon"
        ] = cand[
            "query_lon"
        ]

        rec[
            "ground_truth_source"
        ] = (
            "matched_background_"
            "weak_negative"
        )

        # Remove positive-specific
        # Carbon Mapper quantities.
        for col in rec.index:

            if str(col).startswith(
                "cm_"
            ):
                rec[col] = np.nan

        rec[
            "negative_snr_p99"
        ] = cand[
            "negative_snr_p99"
        ]

        rec[
            "negative_ch4_p99"
        ] = cand[
            "negative_ch4_p99"
        ]

        rec[
            "negative_qa_valid_fraction"
        ] = cand[
            "qa_valid_fraction"
        ]

        rec[
            "negative_sns_p50"
        ] = cand[
            "sns_p50"
        ]

        rec[
            "nearest_official_plume_m"
        ] = cand[
            "nearest_official_plume_m"
        ]

        rec[
            "precheck_t0_missing_ratio"
        ] = cand[
            "t0_missing_ratio_precheck"
        ]

        rec[
            "precheck_t90_missing_ratio"
        ] = cand[
            "t90_missing_ratio_precheck"
        ]

        rec[
            "precheck_t180_missing_ratio"
        ] = cand[
            "t180_missing_ratio_precheck"
        ]

        negative_records.append(
            rec
        )

        neg_counter += 1


# ============================================================
# SAVE CANDIDATE AUDIT
# ============================================================

pd.DataFrame(
    candidate_audit
).to_csv(
    OUT_AUDIT,
    index=False
)


# ============================================================
# FINAL BALANCED SET
# ============================================================

negative = pd.DataFrame(
    negative_records
)

negative.to_csv(
    OUT_NEG,
    index=False
)

print("\n")
print("=" * 75)
print("NEW NEGATIVES")
print("=" * 75)

print(
    "Rows:",
    len(negative)
)

print(
    "\nNegative count per scene:"
)

print(
    negative[
        "scene_key"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

if len(negative) != 10:

    raise RuntimeError(
        f"Expected 10 negatives, "
        f"got {len(negative)}"
    )


# ------------------------------------------------------------
# Align columns
# ------------------------------------------------------------

all_cols = sorted(
    set(
        positive.columns
    )
    |
    set(
        negative.columns
    )
)

for c in all_cols:

    if c not in positive:
        positive[c] = np.nan

    if c not in negative:
        negative[c] = np.nan


final = pd.concat(
    [
        positive[
            all_cols
        ],
        negative[
            all_cols
        ],
    ],
    ignore_index=True
)

final = final.sort_values(
    [
        "scene_key",
        "label",
        "query_id",
    ],
    ascending=[
        True,
        False,
        True,
    ]
).reset_index(
    drop=True
)

final.to_csv(
    OUT_FINAL,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 75)
print("PATCH-VALID BALANCED MANIFEST")
print("=" * 75)

print(
    "Rows:",
    len(final)
)

print("\nLabels:")

print(
    final[
        "label"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print(
    "\nScenes:",
    final[
        "scene_key"
    ].nunique()
)

print(
    "\nLABELS PER SCENE"
)

print(
    pd.crosstab(
        final[
            "scene_key"
        ],
        final[
            "label"
        ]
    ).to_string()
)

print(
    "\nNEGATIVE QA"
)

print(
    "Median SNR p99:",
    round(
        negative[
            "negative_snr_p99"
        ].median(),
        3
    )
)

print(
    "Max SNR p99:",
    round(
        negative[
            "negative_snr_p99"
        ].max(),
        3
    )
)

print(
    "Median QA valid:",
    round(
        negative[
            "negative_qa_valid_fraction"
        ].median(),
        3
    )
)

print(
    "Minimum plume distance:",
    round(
        negative[
            "nearest_official_plume_m"
        ].min(),
        1
    ),
    "m"
)

print(
    "\nPRECHECK MISSING MAX"
)

for c in [
    "precheck_t0_missing_ratio",
    "precheck_t90_missing_ratio",
    "precheck_t180_missing_ratio",
]:

    print(
        c,
        "=",
        round(
            pd.to_numeric(
                negative[c],
                errors="coerce"
            ).max(),
            4
        )
    )

print("\nFILES")

print(
    "Positive:",
    OUT_POS.resolve()
)

print(
    "Negative:",
    OUT_NEG.resolve()
)

print(
    "Final:",
    OUT_FINAL.resolve()
)

print(
    "Candidate audit:",
    OUT_AUDIT.resolve()
)


# ------------------------------------------------------------
# Strict requirements
# ------------------------------------------------------------

counts = (
    final[
        "label"
    ]
    .value_counts()
    .to_dict()
)

if (
    len(final) != 20
    or counts.get(0) != 10
    or counts.get(1) != 10
):

    raise RuntimeError(
        "Final set is not 10/10"
    )


ct = pd.crosstab(
    final[
        "scene_key"
    ],
    final[
        "label"
    ]
)

for scene in ct.index:

    n0 = int(
        ct.loc[
            scene
        ].get(
            0,
            0
        )
    )

    n1 = int(
        ct.loc[
            scene
        ].get(
            1,
            0
        )
    )

    if n0 != n1:

        raise RuntimeError(
            f"{scene}: "
            f"not balanced "
            f"({n0}/{n1})"
        )

print(
    "\nPATCH-VALID 20-ROW "
    "MANIFEST READY"
)
