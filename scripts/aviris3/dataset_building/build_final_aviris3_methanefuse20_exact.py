import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


# ============================================================
# PATHS
# ============================================================

ROOT = Path("AVIRIS3_MethaneFuse_build")

MANIFEST = Path(
    "aviris3_methanefuse_queries_exact20.csv"
)

CONVERTER = Path(
    "build_one_av3_methanefuse_test.py"
)

OLD_BUILD = (
    ROOT
    / "AVIRIS3_MethaneFuse_test_58"
)

OLD_SAMPLES = (
    OLD_BUILD
    / "samples"
)

OLD_QA = (
    OLD_BUILD
    / "qa_report.csv"
)

FINAL_ROOT = (
    ROOT
    / "aviris3_methanefuse_final20_exact"
)

SAMPLES = (
    FINAL_ROOT
    / "samples"
)

FINAL_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

SAMPLES.mkdir(
    parents=True,
    exist_ok=True
)

LOCAL_CSV = (
    FINAL_ROOT
    / "aviris3_methanefuse_test_local.csv"
)

PORTABLE_CSV = (
    FINAL_ROOT
    / "aviris3_methanefuse_test.csv"
)

QA_OUT = (
    FINAL_ROOT
    / "qa_report.csv"
)

PROVENANCE_OUT = (
    FINAL_ROOT
    / "provenance_manifest.csv"
)


# ============================================================
# COPY EXACT WV3 SRF INTO PACKAGE
# ============================================================

SRF_SRC = (
    ROOT
    / "reference"
    / "WV3_VNIR_SWIR_response.csv"
)

SRF_DST = (
    FINAL_ROOT
    / "WV3_VNIR_SWIR_response.csv"
)

if not SRF_SRC.exists():
    raise FileNotFoundError(
        SRF_SRC.resolve()
    )

shutil.copy2(
    SRF_SRC,
    SRF_DST
)


# ============================================================
# LOAD TESTED CONVERTER FUNCTIONS
#
# Execute only the imports/config/functions section,
# not its single-sample main routine.
# ============================================================

if not CONVERTER.exists():
    raise FileNotFoundError(
        CONVERTER.resolve()
    )

source = CONVERTER.read_text()

marker = """
# ============================================================
# PICK FIRST POSITIVE
# ============================================================
"""

if marker not in source:
    raise RuntimeError(
        "Could not locate converter split marker."
    )

prefix = source.split(marker)[0]

ns = {}

exec(
    compile(
        prefix,
        str(CONVERTER),
        "exec"
    ),
    ns
)

AV3_DIR = ns["AV3_DIR"]
EMIT90_DIR = ns["EMIT90_DIR"]
EMIT180_DIR = ns["EMIT180_DIR"]

srf_df = ns["srf_df"]

locate_file = ns["locate_file"]
make_av3_512 = ns["make_av3_512"]
make_emit_512 = ns["make_emit_512"]
center_query = ns["center_query"]
write_tif = ns["write_tif"]


# ============================================================
# VALIDATE TIFF EXACTLY LIKE REPO-FACING FORMAT
# ============================================================

def validate_tif(path):

    path = Path(path)

    if not path.exists():
        raise RuntimeError(
            f"Missing TIFF: {path}"
        )

    arr = tifffile.imread(
        str(path)
    )

    if arr.ndim != 3:
        raise RuntimeError(
            f"{path.name}: "
            f"ndim={arr.ndim}"
        )

    # MethaneFuse supports either CHW or HWC.
    valid_shape = (
        arr.shape == (16, 518, 518)
        or
        arr.shape == (518, 518, 16)
    )

    if not valid_shape:
        raise RuntimeError(
            f"{path.name}: "
            f"bad tifffile shape "
            f"{arr.shape}"
        )

    if not np.isfinite(arr).all():
        raise RuntimeError(
            f"{path.name}: "
            "contains NaN/Inf"
        )

    if np.all(arr == 0):
        raise RuntimeError(
            f"{path.name}: all zero"
        )

    return {
        "tifffile_shape":
            str(arr.shape),

        "dtype":
            str(arr.dtype),

        "min":
            float(
                np.min(arr)
            ),

        "max":
            float(
                np.max(arr)
            ),

        "zero_fraction":
            float(
                (arr == 0).mean()
            ),
    }


# ============================================================
# LOAD FINAL 20 MANIFEST
# ============================================================

df = pd.read_csv(
    MANIFEST
)

print("=" * 75)
print("INPUT FINAL MANIFEST")
print("=" * 75)

print("Rows:", len(df))

print("\nLabels:")
print(
    df["label"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nScenes:", df["scene_key"].nunique())

print("\nLabels per scene:")
print(
    pd.crosstab(
        df["scene_key"],
        df["label"]
    ).to_string()
)

if len(df) != 20:
    raise RuntimeError(
        f"Expected 20 rows, got {len(df)}"
    )

counts = (
    df["label"]
    .value_counts()
    .to_dict()
)

if (
    counts.get(0) != 10
    or
    counts.get(1) != 10
):
    raise RuntimeError(
        f"Expected 10/10 labels: {counts}"
    )


# ============================================================
# OLD QA:
# identify the 10 positives already physically built
# ============================================================

old_qa = pd.read_csv(
    OLD_QA
)

pass_positive_ids = set(
    old_qa.loc[
        (
            old_qa["status"]
            .astype(str)
            .str.upper()
            == "PASS"
        )
        &
        (
            pd.to_numeric(
                old_qa["label"],
                errors="coerce"
            )
            == 1
        ),
        "id"
    ].astype(str)
)

print(
    "\nPreviously built PASS positives:",
    len(pass_positive_ids)
)


# ============================================================
# BUILD FINAL 20
# ============================================================

local_rows = []
portable_rows = []
qa_rows = []

for i, row in df.iterrows():

    qid = str(
        row["query_id"]
    )

    label = int(
        row["label"]
    )

    lat = float(
        row["query_lat"]
    )

    lon = float(
        row["query_lon"]
    )

    print("\n")
    print("=" * 75)
    print(
        f"[{i+1}/20] "
        f"{qid} | label={label}"
    )
    print("=" * 75)

    out_dir = (
        SAMPLES
        / qid
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    out0 = (
        out_dir
        / "emit_0.tif"
    )

    out90 = (
        out_dir
        / "emit_90.tif"
    )

    out360 = (
        out_dir
        / "emit_360.tif"
    )

    qa = {
        "id": qid,
        "label": label,
        "scene_key":
            row["scene_key"],
        "status": "FAIL",
        "source_method": "",
        "t0_missing_ratio":
            np.nan,
        "t90_missing_ratio":
            np.nan,
        "t180_missing_ratio":
            np.nan,
    }

    # --------------------------------------------------------
    # POSITIVE:
    # copy already-tested PASS TIFFs
    # --------------------------------------------------------

    if label == 1:

        if qid not in pass_positive_ids:
            raise RuntimeError(
                f"{qid}: positive was not "
                "a previous PASS sample"
            )

        src_dir = (
            OLD_SAMPLES
            / qid
        )

        src0 = (
            src_dir
            / "emit_0.tif"
        )

        src90 = (
            src_dir
            / "emit_90.tif"
        )

        src360 = (
            src_dir
            / "emit_360.tif"
        )

        for src in [
            src0,
            src90,
            src360,
        ]:

            if not src.exists():
                raise FileNotFoundError(
                    src.resolve()
                )

        shutil.copy2(
            src0,
            out0
        )

        shutil.copy2(
            src90,
            out90
        )

        shutil.copy2(
            src360,
            out360
        )

        old_row = old_qa[
            old_qa["id"]
            .astype(str)
            == qid
        ].iloc[0]

        qa[
            "t0_missing_ratio"
        ] = old_row.get(
            "t0_missing_ratio",
            np.nan
        )

        qa[
            "t90_missing_ratio"
        ] = old_row.get(
            "t90_missing_ratio",
            np.nan
        )

        qa[
            "t180_missing_ratio"
        ] = old_row.get(
            "t180_missing_ratio",
            np.nan
        )

        qa[
            "source_method"
        ] = "reused_previous_PASS_positive"

        print(
            "Reused existing "
            "PASS positive TIFFs"
        )

    # --------------------------------------------------------
    # NEGATIVE:
    # generate with the exact tested converter
    # --------------------------------------------------------

    else:

        t0_nc = locate_file(
            row[
                "av3_l2a_rfl_url"
            ],
            AV3_DIR
        )

        t90_nc = locate_file(
            row[
                "emit_t90_rfl_url"
            ],
            EMIT90_DIR
        )

        t180_nc = locate_file(
            row[
                "emit_t180_rfl_url"
            ],
            EMIT180_DIR
        )

        # t0
        print(">>> AVIRIS-3 t0")

        x0 = make_av3_512(
            t0_nc,
            lat,
            lon,
            srf_df
        )

        f0, m0 = center_query(
            x0
        )

        write_tif(
            out0,
            f0
        )

        del x0, f0

        # t90
        print(">>> EMIT t-90")

        x90 = make_emit_512(
            t90_nc,
            lat,
            lon,
            srf_df
        )

        f90, m90 = center_query(
            x90
        )

        write_tif(
            out90,
            f90
        )

        del x90, f90

        # t180
        print(">>> EMIT t-180")

        x180 = make_emit_512(
            t180_nc,
            lat,
            lon,
            srf_df
        )

        f180, m180 = center_query(
            x180
        )

        write_tif(
            out360,
            f180
        )

        del x180, f180

        qa[
            "t0_missing_ratio"
        ] = m0

        qa[
            "t90_missing_ratio"
        ] = m90

        qa[
            "t180_missing_ratio"
        ] = m180

        qa[
            "source_method"
        ] = "new_patchvalid_negative"

        print(
            "Actual missing:",
            round(m0, 4),
            round(m90, 4),
            round(m180, 4)
        )

    # --------------------------------------------------------
    # TIFF QA using tifffile
    # --------------------------------------------------------

    info0 = validate_tif(
        out0
    )

    info90 = validate_tif(
        out90
    )

    info360 = validate_tif(
        out360
    )

    qa.update({
        "t0_shape":
            info0[
                "tifffile_shape"
            ],

        "t90_shape":
            info90[
                "tifffile_shape"
            ],

        "t180_shape":
            info360[
                "tifffile_shape"
            ],

        "t0_dtype":
            info0["dtype"],

        "t90_dtype":
            info90["dtype"],

        "t180_dtype":
            info360["dtype"],

        "t0_min":
            info0["min"],

        "t0_max":
            info0["max"],

        "t90_min":
            info90["min"],

        "t90_max":
            info90["max"],

        "t180_min":
            info360["min"],

        "t180_max":
            info360["max"],
    })

    for c in [
        "t0_missing_ratio",
        "t90_missing_ratio",
        "t180_missing_ratio",
    ]:

        v = pd.to_numeric(
            pd.Series(
                [qa[c]]
            ),
            errors="coerce"
        ).iloc[0]

        if (
            np.isfinite(v)
            and v > 0.25
        ):
            raise RuntimeError(
                f"{qid}: {c}={v} > 0.25"
            )

    qa["status"] = "PASS"

    qa_rows.append(
        qa
    )

    # --------------------------------------------------------
    # LOCAL CSV
    # --------------------------------------------------------

    local_rows.append({
        "id": qid,
        "label": label,

        "emit_0_path":
            str(
                out0.resolve()
            ),

        "emit_90_path":
            str(
                out90.resolve()
            ),

        "emit_360_path":
            str(
                out360.resolve()
            ),
    })

    # --------------------------------------------------------
    # PORTABLE CSV
    #
    # Assumption:
    # folder is copied into:
    #
    # MethaneFuse/
    # data/custom/
    # aviris3_methanefuse_final20/
    #
    # and evaluation is launched from
    # the MethaneFuse repo root.
    # --------------------------------------------------------

    repo_base = (
        Path("data")
        / "custom"
        / "aviris3_methanefuse_final20_exact"
        / "samples"
        / qid
    )

    portable_rows.append({
        "id": qid,
        "label": label,

        "emit_0_path":
            str(
                repo_base
                / "emit_0.tif"
            ),

        "emit_90_path":
            str(
                repo_base
                / "emit_90.tif"
            ),

        "emit_360_path":
            str(
                repo_base
                / "emit_360.tif"
            ),
    })

    print(
        "PASS:",
        qid
    )


# ============================================================
# SAVE CSVs
# ============================================================

local_df = pd.DataFrame(
    local_rows
)

portable_df = pd.DataFrame(
    portable_rows
)

qa_df = pd.DataFrame(
    qa_rows
)

local_df.to_csv(
    LOCAL_CSV,
    index=False
)

portable_df.to_csv(
    PORTABLE_CSV,
    index=False
)

qa_df.to_csv(
    QA_OUT,
    index=False
)


# ============================================================
# PROVENANCE
# ============================================================

prov = df.copy()

path_map = portable_df.set_index(
    "id"
)

prov["final_emit_0_path"] = (
    prov["query_id"]
    .astype(str)
    .map(
        path_map[
            "emit_0_path"
        ]
    )
)

prov["final_emit_90_path"] = (
    prov["query_id"]
    .astype(str)
    .map(
        path_map[
            "emit_90_path"
        ]
    )
)

prov["final_emit_360_path"] = (
    prov["query_id"]
    .astype(str)
    .map(
        path_map[
            "emit_360_path"
        ]
    )
)

prov[
    "actual_third_frame"
] = "t-180"

prov[
    "dataset_interpretation"
] = (
    "hybrid_hyperspectral_external_test:"
    "AVIRIS3_t0+EMIT_historical"
)

prov.to_csv(
    PROVENANCE_OUT,
    index=False
)


# ============================================================
# FINAL PHYSICAL QA
# ============================================================

print("\n")
print("=" * 75)
print("FINAL DATASET SUMMARY")
print("=" * 75)

print(
    "Rows:",
    len(portable_df)
)

print("\nLabels:")

print(
    portable_df["label"]
    .value_counts()
    .sort_index()
    .to_string()
)

print(
    "\nPASS QA:",
    int(
        (
            qa_df["status"]
            == "PASS"
        ).sum()
    ),
    "/",
    len(qa_df)
)

paths = []

for c in [
    "emit_0_path",
    "emit_90_path",
    "emit_360_path",
]:

    paths.extend(
        local_df[c]
        .tolist()
    )

print(
    "\nExpected TIFFs:",
    len(local_df) * 3
)

print(
    "Physical TIFFs:",
    sum(
        Path(p).exists()
        for p in paths
    )
)


# ============================================================
# EXACT TIFFFILE SHAPE SUMMARY
# ============================================================

print("\nTIFFFILE SHAPES")

print(
    qa_df[
        [
            "t0_shape",
            "t90_shape",
            "t180_shape",
        ]
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# MISSING SUMMARY
# ============================================================

print("\nMISSING RATIO SUMMARY")

for c in [
    "t0_missing_ratio",
    "t90_missing_ratio",
    "t180_missing_ratio",
]:

    x = pd.to_numeric(
        qa_df[c],
        errors="coerce"
    )

    print(
        c,
        "| median=",
        round(
            x.median(),
            4
        ),
        "| max=",
        round(
            x.max(),
            4
        )
    )


# ============================================================
# BALANCE BY SCENE
# ============================================================

check = portable_df.merge(
    df[
        [
            "query_id",
            "scene_key"
        ]
    ].rename(
        columns={
            "query_id": "id"
        }
    ),
    on="id",
    how="left"
)

print("\nLABELS PER SCENE")

print(
    pd.crosstab(
        check[
            "scene_key"
        ],
        check[
            "label"
        ]
    ).to_string()
)


# ============================================================
# HARD ASSERTIONS
# ============================================================

if len(portable_df) != 20:
    raise RuntimeError(
        "Final CSV is not 20 rows"
    )

counts = (
    portable_df[
        "label"
    ]
    .value_counts()
    .to_dict()
)

if (
    counts.get(0) != 10
    or
    counts.get(1) != 10
):
    raise RuntimeError(
        f"Final labels not 10/10: "
        f"{counts}"
    )

if len(paths) != 60:
    raise RuntimeError(
        "Expected 60 TIFF paths"
    )

if not all(
    Path(p).exists()
    for p in paths
):
    raise RuntimeError(
        "Some final TIFFs missing"
    )

if not (
    qa_df["status"]
    == "PASS"
).all():
    raise RuntimeError(
        "Some final samples failed QA"
    )


print("\n================================")
print("FINAL 20 DATASET BUILD COMPLETE")
print("================================")

print("\nPackage:")
print(
    FINAL_ROOT.resolve()
)

print("\nPortable MethaneFuse CSV:")
print(
    PORTABLE_CSV.resolve()
)

print("\nLocal validation CSV:")
print(
    LOCAL_CSV.resolve()
)

print("\nWV3 SRF:")
print(
    SRF_DST.resolve()
)
