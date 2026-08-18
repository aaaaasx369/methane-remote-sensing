import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# REUSE THE TESTED CONVERTER FUNCTIONS
# ============================================================

SOURCE_SCRIPT = Path(
    "build_one_av3_methanefuse_test.py"
)

if not SOURCE_SCRIPT.exists():
    raise FileNotFoundError(
        SOURCE_SCRIPT.resolve()
    )

text = SOURCE_SCRIPT.read_text()

marker = """
# ============================================================
# PICK FIRST POSITIVE
# ============================================================
"""

if marker not in text:
    raise RuntimeError(
        "Could not find PICK FIRST POSITIVE marker "
        "in build_one_av3_methanefuse_test.py"
    )

# Execute only imports/config/functions/SRF loading.
# Do NOT execute the one-sample test section.
prefix = text.split(marker)[0]

ns = {}
exec(
    compile(
        prefix,
        str(SOURCE_SCRIPT),
        "exec"
    ),
    ns
)


# Pull tested functions/config from namespace
ROOT = ns["ROOT"]
MANIFEST = ns["MANIFEST"]

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
# OUTPUT
# ============================================================

FINAL_ROOT = (
    ROOT
    / "AVIRIS3_MethaneFuse_test_58"
)

SAMPLE_ROOT = (
    FINAL_ROOT
    / "samples"
)

FINAL_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

SAMPLE_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_CSV = (
    FINAL_ROOT
    / "aviris3_methanefuse_test.csv"
)

QA_CSV = (
    FINAL_ROOT
    / "qa_report.csv"
)

PROVENANCE_CSV = (
    FINAL_ROOT
    / "provenance_manifest.csv"
)


# ============================================================
# TIFF VALIDATION
# ============================================================

def validate_final_tif(path):

    path = Path(path)

    if not path.exists():
        return False, "missing_file", {}

    try:

        with rasterio.open(path) as ds:

            arr = ds.read()

            info = {
                "shape":
                    tuple(arr.shape),

                "dtype":
                    str(arr.dtype),

                "finite_fraction":
                    float(
                        np.isfinite(arr).mean()
                    ),

                "zero_fraction":
                    float(
                        (arr == 0).mean()
                    ),

                "min":
                    float(
                        np.nanmin(arr)
                    ),

                "max":
                    float(
                        np.nanmax(arr)
                    ),
            }

        if arr.shape != (
            16,
            518,
            518
        ):
            return (
                False,
                f"bad_shape:{arr.shape}",
                info,
            )

        if arr.dtype != np.float32:
            return (
                False,
                f"bad_dtype:{arr.dtype}",
                info,
            )

        if not np.isfinite(arr).all():
            return (
                False,
                "nonfinite_values",
                info,
            )

        if np.all(arr == 0):
            return (
                False,
                "all_zero",
                info,
            )

        return True, "PASS", info

    except Exception as exc:

        return (
            False,
            f"read_error:{exc}",
            {}
        )


# ============================================================
# LOAD MANIFEST
# ============================================================

df = pd.read_csv(
    MANIFEST
)

print("=" * 70)
print("INPUT")
print("=" * 70)

print("Rows:", len(df))

print("\nLabels:")
print(
    df["label"]
    .value_counts()
    .sort_index()
    .to_string()
)

print(
    "\nUnique AV3 scenes:",
    df["scene_key"].nunique()
)

if len(df) != 58:
    print(
        "WARNING: expected 58 rows"
    )


# ============================================================
# PROCESS ALL 58
# ============================================================

final_rows = []
qa_rows = []
provenance_rows = []

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
    print("=" * 70)

    print(
        f"[{i+1}/{len(df)}] "
        f"{qid} | label={label}"
    )

    print(
        "scene:",
        row["scene_key"]
    )

    print(
        "lat/lon:",
        lat,
        lon
    )

    print("=" * 70)

    sample_dir = (
        SAMPLE_ROOT
        / qid
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # MethaneFuse-facing names
    out0 = (
        sample_dir
        / "emit_0.tif"
    )

    out90 = (
        sample_dir
        / "emit_90.tif"
    )

    # Field is called 360 by MethaneFuse;
    # physical source is our t-180 EMIT frame.
    out360 = (
        sample_dir
        / "emit_360.tif"
    )

    qa = {
        "id": qid,
        "label": label,
        "scene_key":
            row["scene_key"],
        "status": "FAIL",
        "reason": "",
    }

    try:

        # ----------------------------------------------------
        # Resolve 3 source NetCDFs
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # t0 — AVIRIS-3
        # ----------------------------------------------------

        print("\n>>> t0 AVIRIS-3")

        arr0_512 = make_av3_512(
            t0_nc,
            lat,
            lon,
            srf_df,
        )

        final0, miss0 = (
            center_query(
                arr0_512
            )
        )

        write_tif(
            out0,
            final0
        )

        del arr0_512
        del final0

        # ----------------------------------------------------
        # t-90 — EMIT
        # ----------------------------------------------------

        print("\n>>> t-90 EMIT")

        arr90_512 = make_emit_512(
            t90_nc,
            lat,
            lon,
            srf_df,
        )

        final90, miss90 = (
            center_query(
                arr90_512
            )
        )

        write_tif(
            out90,
            final90
        )

        del arr90_512
        del final90

        # ----------------------------------------------------
        # t-180 — EMIT
        # ----------------------------------------------------

        print("\n>>> t-180 EMIT")

        arr180_512 = make_emit_512(
            t180_nc,
            lat,
            lon,
            srf_df,
        )

        final180, miss180 = (
            center_query(
                arr180_512
            )
        )

        write_tif(
            out360,
            final180
        )

        del arr180_512
        del final180

        # ----------------------------------------------------
        # Physical TIFF validation
        # ----------------------------------------------------

        ok0, reason0, info0 = (
            validate_final_tif(
                out0
            )
        )

        ok90, reason90, info90 = (
            validate_final_tif(
                out90
            )
        )

        ok360, reason360, info360 = (
            validate_final_tif(
                out360
            )
        )

        qa.update({
            "t0_missing_ratio":
                miss0,

            "t90_missing_ratio":
                miss90,

            "t180_missing_ratio":
                miss180,

            "t0_tif_status":
                reason0,

            "t90_tif_status":
                reason90,

            "t180_tif_status":
                reason360,

            "t0_min":
                info0.get(
                    "min",
                    np.nan
                ),

            "t0_max":
                info0.get(
                    "max",
                    np.nan
                ),

            "t90_min":
                info90.get(
                    "min",
                    np.nan
                ),

            "t90_max":
                info90.get(
                    "max",
                    np.nan
                ),

            "t180_min":
                info360.get(
                    "min",
                    np.nan
                ),

            "t180_max":
                info360.get(
                    "max",
                    np.nan
                ),

            "t0_zero_fraction":
                info0.get(
                    "zero_fraction",
                    np.nan
                ),

            "t90_zero_fraction":
                info90.get(
                    "zero_fraction",
                    np.nan
                ),

            "t180_zero_fraction":
                info360.get(
                    "zero_fraction",
                    np.nan
                ),
        })

        if not (
            ok0
            and ok90
            and ok360
        ):

            raise RuntimeError(
                "TIFF validation failed: "
                f"t0={reason0}; "
                f"t90={reason90}; "
                f"t180={reason360}"
            )

        # center_query already applies <=0.25,
        # but verify explicitly here.
        if (
            miss0 > 0.25
            or miss90 > 0.25
            or miss180 > 0.25
        ):

            raise RuntimeError(
                "missing_ratio > 0.25"
            )

        # ----------------------------------------------------
        # MethaneFuse wide-table record
        # ----------------------------------------------------

        final_rows.append({
            "id":
                qid,

            "label":
                label,

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

        # ----------------------------------------------------
        # Detailed provenance
        # ----------------------------------------------------

        rec = {
            "id":
                qid,

            "label":
                label,

            "scene_key":
                row[
                    "scene_key"
                ],

            "query_lat":
                lat,

            "query_lon":
                lon,

            "ground_truth_source":
                row.get(
                    "ground_truth_source",
                    ""
                ),

            "aviris3_t0_source":
                str(
                    t0_nc.resolve()
                ),

            "emit_t90_source":
                str(
                    t90_nc.resolve()
                ),

            "emit_t180_source":
                str(
                    t180_nc.resolve()
                ),

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

            "actual_third_frame":
                "t-180",

            "t0_date":
                row.get(
                    "t0_date",
                    ""
                ),

            "emit_t90_date":
                row.get(
                    "emit_t90_date",
                    ""
                ),

            "emit_t180_date":
                row.get(
                    "emit_t180_date",
                    ""
                ),

            "emit_t90_age_days":
                row.get(
                    "emit_t90_age_days",
                    ""
                ),

            "emit_t180_age_days":
                row.get(
                    "emit_t180_age_days",
                    ""
                ),

            "emit_t90_error_days":
                row.get(
                    "emit_t90_error_days",
                    ""
                ),

            "emit_t180_error_days":
                row.get(
                    "emit_t180_error_days",
                    ""
                ),

            "cm_emission_kghr":
                row.get(
                    "cm_emission_kghr",
                    ""
                ),

            "nearest_official_plume_m":
                row.get(
                    "nearest_official_plume_m",
                    ""
                ),
        }

        provenance_rows.append(
            rec
        )

        qa[
            "status"
        ] = "PASS"

        print(
            "\nPASS",
            qid,
            "| missing:",
            round(miss0, 4),
            round(miss90, 4),
            round(miss180, 4),
        )

    except Exception as exc:

        qa[
            "reason"
        ] = str(exc)

        print(
            "\nFAIL:",
            qid
        )

        print(
            str(exc)
        )

        traceback.print_exc()

    qa_rows.append(
        qa
    )

    # Save checkpoint after every row.
    pd.DataFrame(
        qa_rows
    ).to_csv(
        QA_CSV,
        index=False
    )

    pd.DataFrame(
        final_rows
    ).to_csv(
        FINAL_CSV,
        index=False
    )

    pd.DataFrame(
        provenance_rows
    ).to_csv(
        PROVENANCE_CSV,
        index=False
    )


# ============================================================
# FINAL QA
# ============================================================

qa_df = pd.DataFrame(
    qa_rows
)

final_df = pd.DataFrame(
    final_rows
)

print("\n")
print("=" * 70)
print("FINAL BUILD SUMMARY")
print("=" * 70)

print(
    "Input rows:",
    len(df)
)

print(
    "PASS:",
    int(
        (
            qa_df["status"]
            == "PASS"
        ).sum()
    )
)

print(
    "FAIL:",
    int(
        (
            qa_df["status"]
            == "FAIL"
        ).sum()
    )
)

print(
    "\nFinal CSV rows:",
    len(final_df)
)

if len(final_df):

    print("\nFinal labels:")

    print(
        final_df[
            "label"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    all_paths = []

    for c in [
        "emit_0_path",
        "emit_90_path",
        "emit_360_path",
    ]:

        all_paths.extend(
            final_df[c].tolist()
        )

    print(
        "\nExpected TIFFs:",
        len(final_df) * 3
    )

    print(
        "TIFFs physically present:",
        sum(
            Path(p).exists()
            for p in all_paths
        )
    )


# ============================================================
# PER-TEMPORAL MISSING QA
# ============================================================

if len(qa_df):

    passed = qa_df[
        qa_df["status"]
        == "PASS"
    ]

    if len(passed):

        print(
            "\nMISSING RATIO SUMMARY"
        )

        for c in [
            "t0_missing_ratio",
            "t90_missing_ratio",
            "t180_missing_ratio",
        ]:

            x = pd.to_numeric(
                passed[c],
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

if len(final_df) == 58:

    tmp = df[
        [
            "query_id",
            "scene_key",
        ]
    ].rename(
        columns={
            "query_id": "id"
        }
    )

    merged = final_df.merge(
        tmp,
        on="id",
        how="left"
    )

    print(
        "\nLABELS PER SCENE"
    )

    print(
        pd.crosstab(
            merged[
                "scene_key"
            ],
            merged[
                "label"
            ]
        ).to_string()
    )


# ============================================================
# REQUIRE PERFECT BUILD
# ============================================================

fail_n = int(
    (
        qa_df["status"]
        == "FAIL"
    ).sum()
)

print("\nFILES")
print("Final CSV :", FINAL_CSV.resolve())
print("QA report :", QA_CSV.resolve())
print(
    "Provenance:",
    PROVENANCE_CSV.resolve()
)

if fail_n > 0:

    print(
        "\nBUILD INCOMPLETE — "
        "do not send dataset yet."
    )

    sys.exit(2)

if len(final_df) != 58:

    raise RuntimeError(
        f"Expected 58 final rows, "
        f"got {len(final_df)}"
    )

if (
    final_df["label"]
    .value_counts()
    .to_dict()
    != {0: 29, 1: 29}
):

    raise RuntimeError(
        "Final label balance "
        "is not 29/29"
    )

print("\nALL 58 SAMPLES PASS")
print("DATASET BUILD COMPLETE")
