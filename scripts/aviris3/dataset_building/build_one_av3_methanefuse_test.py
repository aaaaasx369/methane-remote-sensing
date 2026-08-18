import numpy as np
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse

from netCDF4 import Dataset
from scipy.spatial import KDTree
from pyproj import CRS, Transformer
import rasterio


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("AVIRIS3_MethaneFuse_build")

MANIFEST = Path(
    "aviris3_methanefuse_queries_58.csv"
)

AV3_DIR = ROOT / "L2A_AVIRIS3_t0"
EMIT90_DIR = ROOT / "L2A_EMIT_t90"
EMIT180_DIR = ROOT / "L2A_EMIT_t180"

SRF_CSV = (
    ROOT
    / "reference"
    / "WV3_VNIR_SWIR_response.csv"
)

OUT_DIR = (
    ROOT
    / "TEST_ONE_METHANEFUSE_SAMPLE_V2"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHIP_SIZE = 512
SCALE_M = 60.0

QUERY_SIZE_M = 480.0
QUERY_PX = 8

TARGET_SIZE = 518

ROW_BLOCK = 32


WV3_BANDS = [
    "Coastal (MS7)",
    "Blue (MS4)",
    "Green (MS3)",
    "Yellow (MS6)",
    "Red (MS2)",
    "Red Edge (MS5)",
    "NIR1 (MS1)",
    "NIR2 (MS8)",
    "SWIR1",
    "SWIR2",
    "SWIR3",
    "SWIR4",
    "SWIR5",
    "SWIR6",
    "SWIR7",
    "SWIR8",
]


# ============================================================
# GENERAL HELPERS
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
        f"Cannot find {name}"
    )


def get_utm_crs(lat, lon):

    zone = int(
        (lon + 180) / 6
    ) + 1

    epsg = (
        32600 + zone
        if lat >= 0
        else 32700 + zone
    )

    return CRS.from_epsg(epsg)


def build_srf_matrix(
    wavelengths,
    srf_df,
):

    wavelengths = np.asarray(
        wavelengths,
        dtype=np.float64
    )

    waves = srf_df[
        "nm/Band"
    ].to_numpy(
        dtype=np.float64
    )

    M = np.zeros(
        (
            len(wavelengths),
            16
        ),
        dtype=np.float32
    )

    for i, band in enumerate(
        WV3_BANDS
    ):

        response = np.interp(
            wavelengths,
            waves,
            srf_df[
                band
            ].to_numpy(
                dtype=np.float64
            ),
            left=0,
            right=0,
        )

        total = response.sum()

        if total <= 0:

            raise RuntimeError(
                f"No SRF response for {band}"
            )

        M[:, i] = (
            response / total
        ).astype(
            np.float32
        )

    return M


def author_stretch(simulated):

    """
    Input:
        H x W x 16 float

    Output:
        16 x H x W uint16

    Matches MethaneUnion EMIT_wv3.py:
      valid > 0
      percentile 1 / 99
      stretch * 6000 + 8000
      clip 0..65535
    """

    h, w, _ = (
        simulated.shape
    )

    out = np.zeros(
        (16, h, w),
        dtype=np.uint16
    )

    for b in range(16):

        band = simulated[
            :, :, b
        ]

        valid = (
            np.isfinite(band)
            &
            (band > 0)
        )

        if not np.any(valid):
            continue

        p1 = np.percentile(
            band[valid],
            1
        )

        p99 = np.percentile(
            band[valid],
            99
        )

        stretched = (
            band - p1
        ) / (
            p99 - p1 + 1e-6
        )

        final = (
            stretched * 6000.0
            + 8000.0
        )

        final = np.where(
            valid,
            final,
            0
        )

        out[b] = np.clip(
            final,
            0,
            65535
        ).astype(
            np.uint16
        )

    return out


def resize_chw_linear(
    img,
    out_size,
):

    c, h, w = img.shape

    ys = np.linspace(
        0,
        h - 1,
        out_size
    )

    xs = np.linspace(
        0,
        w - 1,
        out_size
    )

    y0 = np.floor(
        ys
    ).astype(int)

    x0 = np.floor(
        xs
    ).astype(int)

    y1 = np.clip(
        y0 + 1,
        0,
        h - 1
    )

    x1 = np.clip(
        x0 + 1,
        0,
        w - 1
    )

    wy = (
        ys - y0
    )[:, None]

    wx = (
        xs - x0
    )[None, :]

    out = np.empty(
        (
            c,
            out_size,
            out_size
        ),
        dtype=np.float32
    )

    for i in range(c):

        b = img[i].astype(
            np.float32
        )

        a = b[
            y0[:, None],
            x0[None, :]
        ]

        bb = b[
            y0[:, None],
            x1[None, :]
        ]

        cc = b[
            y1[:, None],
            x0[None, :]
        ]

        d = b[
            y1[:, None],
            x1[None, :]
        ]

        out[i] = (
            a
            * (1 - wx)
            * (1 - wy)
            +
            bb
            * wx
            * (1 - wy)
            +
            cc
            * (1 - wx)
            * wy
            +
            d
            * wx
            * wy
        )

    return out


def center_query(raw512):

    center = (
        CHIP_SIZE // 2
    )

    half = (
        QUERY_PX // 2
    )

    small = raw512[
        :,
        center-half:
        center-half+QUERY_PX,
        center-half:
        center-half+QUERY_PX,
    ]

    missing = float(
        (
            np.isnan(small[0])
            |
            (small[0] == 0)
        ).mean()
    )

    if missing > 0.25:

        raise RuntimeError(
            "480m crop fails "
            f"missing QA: {missing:.3f}"
        )

    final = resize_chw_linear(
        small.astype(
            np.float32
        ),
        TARGET_SIZE
    )

    return final, missing


def write_tif(
    path,
    arr,
):

    arr = np.asarray(
        arr,
        dtype=np.float32
    )

    c, h, w = arr.shape

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": c,
        "dtype": "float32",
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(
        path,
        "w",
        **profile
    ) as dst:
        dst.write(arr)

    with rasterio.open(
        path
    ) as ds:

        check = ds.read()

    if check.shape != (
        16,
        TARGET_SIZE,
        TARGET_SIZE
    ):
        raise RuntimeError(
            f"Bad TIFF shape: "
            f"{check.shape}"
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

    p = np.searchsorted(
        coords,
        targets
    )

    p = np.clip(
        p,
        1,
        len(coords) - 1
    )

    left = coords[
        p - 1
    ]

    right = coords[
        p
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
        p,
        p - 1
    )


# ============================================================
# AVIRIS-3 CRS
# ============================================================

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

    try:

        return CRS.from_cf(
            attrs
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not parse AVIRIS-3 "
            "transverse_mercator CRS: "
            f"{exc}"
        )


# ============================================================
# AVIRIS-3 t0 CONVERTER
# ============================================================

def make_av3_512(
    path,
    lat,
    lon,
    srf_df,
):

    print("\n================================")
    print("AVIRIS-3 t0")
    print("================================")

    print(path.name)

    with Dataset(
        str(path),
        "r"
    ) as ds:

        east = np.asarray(
            ds.variables[
                "easting"
            ][:],
            dtype=np.float64
        )

        north = np.asarray(
            ds.variables[
                "northing"
            ][:],
            dtype=np.float64
        )

        crs = read_av3_crs(
            ds
        )

        g = ds.groups[
            "reflectance"
        ]

        wavelengths = np.asarray(
            g.variables[
                "wavelength"
            ][:],
            dtype=np.float64
        )

        rfl = g.variables[
            "reflectance"
        ]

        print(
            "reflectance:",
            rfl.shape
        )

        print(
            "wavelengths:",
            len(wavelengths),
            round(
                wavelengths.min(),
                2
            ),
            "to",
            round(
                wavelengths.max(),
                2
            )
        )

        print(
            "CRS:",
            crs.to_string()
        )

        dx_native = float(
            np.median(
                np.abs(
                    np.diff(east)
                )
            )
        )

        dy_native = float(
            np.median(
                np.abs(
                    np.diff(north)
                )
            )
        )

        print(
            "native GSD:",
            round(
                dx_native,
                3
            ),
            "x",
            round(
                dy_native,
                3
            ),
            "m"
        )

        to_src = (
            Transformer
            .from_crs(
                "EPSG:4326",
                crs,
                always_xy=True,
            )
        )

        cx, cy = to_src.transform(
            lon,
            lat
        )

        # ----------------------------------------------
        # Context region:
        # reproduce author's ±0.15 degree
        # neighborhood before percentile scaling.
        # ----------------------------------------------

        corner_lon = [
            lon - 0.15,
            lon + 0.15,
            lon - 0.15,
            lon + 0.15,
        ]

        corner_lat = [
            lat - 0.15,
            lat - 0.15,
            lat + 0.15,
            lat + 0.15,
        ]

        xs, ys = to_src.transform(
            corner_lon,
            corner_lat
        )

        xlo = min(xs)
        xhi = max(xs)

        ylo = min(ys)
        yhi = max(ys)

        xmask = (
            (east >= xlo)
            &
            (east <= xhi)
        )

        ymask = (
            (north >= ylo)
            &
            (north <= yhi)
        )

        xi = np.where(
            xmask
        )[0]

        yi = np.where(
            ymask
        )[0]

        if len(xi) == 0:
            raise RuntimeError(
                "AV3 no easting overlap"
            )

        if len(yi) == 0:
            raise RuntimeError(
                "AV3 no northing overlap"
            )

        x0 = int(xi.min())
        x1 = int(xi.max())

        y0 = int(yi.min())
        y1 = int(yi.max())

        sub_e = east[
            x0:x1+1
        ]

        sub_n = north[
            y0:y1+1
        ]

        h = (
            y1 - y0 + 1
        )

        w = (
            x1 - x0 + 1
        )

        print(
            "context native crop:",
            h,
            "x",
            w
        )

        srf = build_srf_matrix(
            wavelengths,
            srf_df
        )

        simulated = np.empty(
            (
                h,
                w,
                16
            ),
            dtype=np.float32
        )

        for ys0 in range(
            0,
            h,
            ROW_BLOCK
        ):

            ys1 = min(
                h,
                ys0 + ROW_BLOCK
            )

            block = rfl[
                :,
                y0+ys0:
                y0+ys1,
                x0:x1+1
            ]

            if np.ma.isMaskedArray(
                block
            ):
                block = block.filled(
                    np.nan
                )

            block = np.asarray(
                block,
                dtype=np.float32
            )

            block = np.nan_to_num(
                block,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # wavelength,y,x
            # -> y,x,wavelength
            block = np.moveaxis(
                block,
                0,
                -1
            )

            simulated[
                ys0:ys1
            ] = np.matmul(
                block,
                srf
            )

        scaled = author_stretch(
            simulated
        )

    # --------------------------------------------------------
    # Build same 512 target geometry as author:
    # 512 grid points over ~30.72 km.
    # --------------------------------------------------------

    hs = (
        CHIP_SIZE
        * SCALE_M
        / 2
    )

    tx = np.linspace(
        cx - hs,
        cx + hs,
        CHIP_SIZE
    )

    ty = np.linspace(
        cy + hs,
        cy - hs,
        CHIP_SIZE
    )

    xidx = nearest_indices(
        sub_e,
        tx
    )

    yidx = nearest_indices(
        sub_n,
        ty
    )

    res = scaled[
        :,
        yidx[:, None],
        xidx[None, :]
    ]

    # Do not allow nearest-neighbor extrapolation
    # outside the actual AV3/context bounds.
    x_min = sub_e.min()
    x_max = sub_e.max()

    y_min = sub_n.min()
    y_max = sub_n.max()

    outside = (
        (tx[None, :] < x_min - dx_native/2)
        |
        (tx[None, :] > x_max + dx_native/2)
        |
        (ty[:, None] < y_min - dy_native/2)
        |
        (ty[:, None] > y_max + dy_native/2)
    )

    res[
        :,
        outside
    ] = 0

    print(
        "512 shape:",
        res.shape
    )

    print(
        "512 missing:",
        round(
            float(
                (
                    res[0] == 0
                ).mean()
            ),
            4
        )
    )

    return res


# ============================================================
# EMIT t-90 / t-180 CONVERTER
# ============================================================

def make_emit_512(
    path,
    lat,
    lon,
    srf_df,
):

    print("\n================================")
    print("EMIT")
    print("================================")

    print(path.name)

    with Dataset(
        str(path),
        "r"
    ) as ds:

        loc = ds.groups[
            "location"
        ]

        lats = np.asarray(
            loc.variables[
                "lat"
            ][:]
        )

        lons = np.asarray(
            loc.variables[
                "lon"
            ][:]
        )

    spatial = (
        np.isfinite(lats)
        &
        np.isfinite(lons)
        &
        (lats > lat - 0.15)
        &
        (lats < lat + 0.15)
        &
        (lons > lon - 0.15)
        &
        (lons < lon + 0.15)
    )

    yy, xx = np.where(
        spatial
    )

    if len(yy) == 0:
        raise RuntimeError(
            "EMIT coordinate "
            "not inside granule"
        )

    y0 = int(yy.min())
    y1 = int(yy.max())

    x0 = int(xx.min())
    x1 = int(xx.max())

    lat_crop = lats[
        y0:y1,
        x0:x1
    ]

    lon_crop = lons[
        y0:y1,
        x0:x1
    ]

    h, w = lat_crop.shape

    print(
        "raw context:",
        h,
        "x",
        w
    )

    with Dataset(
        str(path),
        "r"
    ) as ds:

        wavelengths = np.asarray(
            ds.groups[
                "sensor_band_parameters"
            ].variables[
                "wavelengths"
            ][:],
            dtype=np.float64
        )

        rfl = ds.variables[
            "reflectance"
        ]

        print(
            "reflectance:",
            rfl.shape
        )

        print(
            "wavelengths:",
            len(wavelengths),
            round(
                wavelengths.min(),
                2
            ),
            "to",
            round(
                wavelengths.max(),
                2
            )
        )

        srf = build_srf_matrix(
            wavelengths,
            srf_df
        )

        simulated = np.empty(
            (
                h,
                w,
                16
            ),
            dtype=np.float32
        )

        for ys0 in range(
            0,
            h,
            ROW_BLOCK
        ):

            ys1 = min(
                h,
                ys0 + ROW_BLOCK
            )

            block = rfl[
                y0+ys0:
                y0+ys1,
                x0:x1,
                :
            ]

            if np.ma.isMaskedArray(
                block
            ):
                block = block.filled(
                    np.nan
                )

            block = np.asarray(
                block,
                dtype=np.float32
            )

            block = np.nan_to_num(
                block,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            simulated[
                ys0:ys1
            ] = np.matmul(
                block,
                srf
            )

    scaled = author_stretch(
        simulated
    )

    # ----------------------------------------------
    # Author-style KDTree resampling to
    # query-centered 512 @ 60m.
    # ----------------------------------------------

    valid_geo = (
        np.isfinite(
            lat_crop
        )
        &
        np.isfinite(
            lon_crop
        )
        &
        (lat_crop >= -90)
        &
        (lat_crop <= 90)
        &
        (lon_crop >= -180)
        &
        (lon_crop <= 180)
    )

    if not np.any(valid_geo):

        raise RuntimeError(
            "EMIT no valid geolocation"
        )

    coords = np.stack(
        [
            lat_crop[
                valid_geo
            ],
            lon_crop[
                valid_geo
            ],
        ],
        axis=1
    )

    flat_valid = np.flatnonzero(
        valid_geo.ravel()
    )

    tree = KDTree(
        coords
    )

    utm = get_utm_crs(
        lat,
        lon
    )

    to_utm = Transformer.from_crs(
        "EPSG:4326",
        utm,
        always_xy=True,
    )

    to_wgs = Transformer.from_crs(
        utm,
        "EPSG:4326",
        always_xy=True,
    )

    cx, cy = to_utm.transform(
        lon,
        lat
    )

    hs = (
        CHIP_SIZE
        * SCALE_M
        / 2
    )

    tx = np.linspace(
        cx - hs,
        cx + hs,
        CHIP_SIZE
    )

    ty = np.linspace(
        cy + hs,
        cy - hs,
        CHIP_SIZE
    )

    mx, my = np.meshgrid(
        tx,
        ty
    )

    qlon, qlat = (
        to_wgs.transform(
            mx,
            my
        )
    )

    query_points = np.stack(
        [
            qlat.ravel(),
            qlon.ravel(),
        ],
        axis=1
    )

    dist, idx = tree.query(
        query_points,
        distance_upper_bound=0.001,
    )

    missing = ~np.isfinite(
        dist
    )

    safe_idx = idx.copy()

    safe_idx[
        missing
    ] = 0

    raw_idx = flat_valid[
        safe_idx
    ]

    flat_scaled = scaled.reshape(
        16,
        -1
    )

    res = flat_scaled[
        :,
        raw_idx
    ].reshape(
        16,
        CHIP_SIZE,
        CHIP_SIZE
    )

    miss2d = missing.reshape(
        CHIP_SIZE,
        CHIP_SIZE
    )

    res[
        :,
        miss2d
    ] = 0

    print(
        "512 shape:",
        res.shape
    )

    print(
        "512 missing:",
        round(
            float(
                (
                    res[0] == 0
                ).mean()
            ),
            4
        )
    )

    return res


# ============================================================
# LOAD WV3 SRF
# ============================================================

srf_df = pd.read_csv(
    SRF_CSV
)

missing_cols = [
    c for c in (
        ["nm/Band"]
        + WV3_BANDS
    )
    if c not in srf_df.columns
]

if missing_cols:
    raise RuntimeError(
        f"Bad SRF CSV: {missing_cols}"
    )


# ============================================================
# PICK FIRST POSITIVE
# ============================================================

df = pd.read_csv(
    MANIFEST
)

row = (
    df[
        df["label"] == 1
    ]
    .iloc[0]
)

qid = str(
    row["query_id"]
)

lat = float(
    row["query_lat"]
)

lon = float(
    row["query_lon"]
)

print("\n================================")
print("TEST SAMPLE")
print("================================")

print("ID:", qid)
print("lat/lon:", lat, lon)
print("scene:", row["scene_key"])


# ============================================================
# LOCAL SOURCE FILES
# ============================================================

t0_nc = locate_file(
    row["av3_l2a_rfl_url"],
    AV3_DIR
)

t90_nc = locate_file(
    row["emit_t90_rfl_url"],
    EMIT90_DIR
)

t180_nc = locate_file(
    row["emit_t180_rfl_url"],
    EMIT180_DIR
)


# ============================================================
# MAKE 512 WV3-LIKE FRAMES
# ============================================================

t0_512 = make_av3_512(
    t0_nc,
    lat,
    lon,
    srf_df,
)

t90_512 = make_emit_512(
    t90_nc,
    lat,
    lon,
    srf_df,
)

t180_512 = make_emit_512(
    t180_nc,
    lat,
    lon,
    srf_df,
)


# ============================================================
# 480m CROP → 518
# ============================================================

finals = {}

for name, arr in [
    ("t0", t0_512),
    ("t90", t90_512),
    ("t180", t180_512),
]:

    final, missing = (
        center_query(arr)
    )

    out = (
        OUT_DIR
        / f"{qid}__{name}.tif"
    )

    write_tif(
        out,
        final
    )

    finals[name] = {
        "path":
            str(
                out.resolve()
            ),

        "shape":
            final.shape,

        "dtype":
            str(final.dtype),

        "missing":
            missing,

        "min":
            float(
                np.nanmin(final)
            ),

        "max":
            float(
                np.nanmax(final)
            ),
    }


# ============================================================
# METHANEFUSE CSV
# ============================================================

csv_out = (
    OUT_DIR
    / "test_one_methanefuse.csv"
)

test = pd.DataFrame(
    [
        {
            "id": qid,
            "label": 1,

            "emit_0_path":
                finals[
                    "t0"
                ]["path"],

            "emit_90_path":
                finals[
                    "t90"
                ]["path"],

            # Actual historical frame is t-180.
            # MethaneFuse wide-table field name
            # is nevertheless emit_360_path.
            "emit_360_path":
                finals[
                    "t180"
                ]["path"],
        }
    ]
)

test.to_csv(
    csv_out,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n================================")
print("METHANEFUSE SAMPLE READY")
print("================================")

for name in [
    "t0",
    "t90",
    "t180",
]:

    x = finals[name]

    print(
        name,
        "|",
        x["shape"],
        "|",
        x["dtype"],
        "| missing=",
        round(
            x["missing"],
            4
        ),
        "| min/max=",
        round(
            x["min"],
            2
        ),
        "/",
        round(
            x["max"],
            2
        )
    )

print("\nCSV:")
print(csv_out.resolve())

print("\nCSV CONTENT:")
print(
    test.to_string(
        index=False
    )
)
