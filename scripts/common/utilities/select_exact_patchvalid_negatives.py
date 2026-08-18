import gc
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path("AVIRIS3_MethaneFuse_build")

CONVERTER = Path(
    "build_one_av3_methanefuse_test.py"
)

POSITIVE_CSV = Path(
    "aviris3_patchvalid_positives_10.csv"
)

CANDIDATE_CSV = Path(
    "aviris3_patchvalid_negative_candidate_audit.csv"
)

OUT_NEG = Path(
    "aviris3_patchvalid_negatives_10_exact.csv"
)

OUT_FINAL = Path(
    "aviris3_methanefuse_queries_exact20.csv"
)

OUT_AUDIT = Path(
    "aviris3_exact_negative_conversion_audit.csv"
)

MISSING_THRESHOLD = 0.25


# ============================================================
# LOAD EXACT TESTED CONVERTER
# ============================================================

text = CONVERTER.read_text()

marker = """
# ============================================================
# PICK FIRST POSITIVE
# ============================================================
"""

if marker not in text:
    raise RuntimeError(
        "Cannot find converter marker"
    )

prefix = text.split(marker)[0]

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


# ============================================================
# EXACT CENTRAL 480m MISSING
#
# This is the SAME quantity used by center_query().
# No resize needed for selection.
# ============================================================

def exact_center_missing(arr):

    if arr.shape != (
        16,
        512,
        512
    ):
        raise RuntimeError(
            f"Unexpected 512 array "
            f"shape: {arr.shape}"
        )

    center = 512 // 2
    half = 8 // 2

    band0 = arr[
        0,
        center-half:
        center-half+8,
        center-half:
        center-half+8,
    ]

    bad = (
        ~np.isfinite(band0)
        |
        (band0 == 0)
    )

    return float(
        bad.mean()
    )


# ============================================================
# LOAD POSITIVES / CANDIDATES
# ============================================================

positive = pd.read_csv(
    POSITIVE_CSV
)

cand = pd.read_csv(
    CANDIDATE_CSV
)

print("=" * 75)
print("INPUT")
print("=" * 75)

print(
    "Patch-valid positives:",
    len(positive)
)

need_by_scene = (
    positive[
        "scene_key"
    ]
    .value_counts()
    .sort_index()
)

print("\nNeed negatives per scene:")
print(
    need_by_scene.to_string()
)


# ============================================================
# ONLY CANDIDATES THAT PASSED THE LIGHTWEIGHT PRECHECK
# ============================================================

if "coverage_status" in cand.columns:

    cand = cand[
        cand[
            "coverage_status"
        ]
        .astype(str)
        .str.upper()
        == "PASS"
    ].copy()

print(
    "\nLightweight-PASS candidates:",
    len(cand)
)

print("\nCandidates per scene:")

print(
    cand[
        "scene_key"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# SORT:
# lowest methane signal first
# ============================================================

sort_cols = []
ascending = []

for c, asc in [
    ("negative_snr_p99", True),
    ("qa_valid_fraction", False),
    ("nearest_official_plume_m", False),
]:

    if c in cand.columns:
        sort_cols.append(c)
        ascending.append(asc)

if sort_cols:

    cand = cand.sort_values(
        sort_cols,
        ascending=ascending
    )

cand = cand.reset_index(
    drop=True
)


# ============================================================
# EXACT AUDIT CACHE
# ============================================================

cache = {}
audit_rows = []


def exact_test_candidate(
    idx,
    row,
    template,
):

    if idx in cache:
        return cache[idx]

    lat = float(
        row[
            "query_lat"
        ]
    )

    lon = float(
        row[
            "query_lon"
        ]
    )

    scene = str(
        row[
            "scene_key"
        ]
    )

    print("\n")
    print("-" * 75)
    print(
        f"EXACT TEST candidate={idx}"
    )
    print(
        scene,
        "|",
        round(lat, 6),
        round(lon, 6)
    )
    print("-" * 75)

    rec = {
        "candidate_index":
            idx,

        "scene_key":
            scene,

        "query_lat":
            lat,

        "query_lon":
            lon,

        "status":
            "FAIL",

        "fail_stage":
            "",

        "exact_t0_missing_ratio":
            np.nan,

        "exact_t90_missing_ratio":
            np.nan,

        "exact_t180_missing_ratio":
            np.nan,
    }

    try:

        # ----------------------------------------------------
        # SOURCE FILES
        # ----------------------------------------------------

        t0_nc = locate_file(
            template[
                "av3_l2a_rfl_url"
            ],
            AV3_DIR
        )

        t90_nc = locate_file(
            template[
                "emit_t90_rfl_url"
            ],
            EMIT90_DIR
        )

        t180_nc = locate_file(
            template[
                "emit_t180_rfl_url"
            ],
            EMIT180_DIR
        )

        # ----------------------------------------------------
        # AVIRIS-3
        # ----------------------------------------------------

        print(">>> exact t0")

        a0 = make_av3_512(
            t0_nc,
            lat,
            lon,
            srf_df
        )

        m0 = exact_center_missing(
            a0
        )

        rec[
            "exact_t0_missing_ratio"
        ] = m0

        del a0
        gc.collect()

        print(
            "exact t0 central missing:",
            round(m0, 4)
        )

        if m0 > MISSING_THRESHOLD:

            rec["fail_stage"] = (
                "FAIL_T0"
            )

            cache[idx] = rec
            audit_rows.append(rec)

            print("REJECT: t0")

            return rec

        # ----------------------------------------------------
        # EMIT t90
        # ----------------------------------------------------

        print(">>> exact t90")

        a90 = make_emit_512(
            t90_nc,
            lat,
            lon,
            srf_df
        )

        m90 = exact_center_missing(
            a90
        )

        rec[
            "exact_t90_missing_ratio"
        ] = m90

        del a90
        gc.collect()

        print(
            "exact t90 central missing:",
            round(m90, 4)
        )

        if m90 > MISSING_THRESHOLD:

            rec["fail_stage"] = (
                "FAIL_T90"
            )

            cache[idx] = rec
            audit_rows.append(rec)

            print("REJECT: t90")

            return rec

        # ----------------------------------------------------
        # EMIT t180
        # ----------------------------------------------------

        print(">>> exact t180")

        a180 = make_emit_512(
            t180_nc,
            lat,
            lon,
            srf_df
        )

        m180 = exact_center_missing(
            a180
        )

        rec[
            "exact_t180_missing_ratio"
        ] = m180

        del a180
        gc.collect()

        print(
            "exact t180 central missing:",
            round(m180, 4)
        )

        if m180 > MISSING_THRESHOLD:

            rec["fail_stage"] = (
                "FAIL_T180"
            )

            cache[idx] = rec
            audit_rows.append(rec)

            print("REJECT: t180")

            return rec

        rec["status"] = "PASS"
        rec["fail_stage"] = ""

        cache[idx] = rec
        audit_rows.append(rec)

        print(
            "EXACT PASS:",
            round(m0, 4),
            round(m90, 4),
            round(m180, 4)
        )

        return rec

    except Exception as exc:

        rec[
            "fail_stage"
        ] = "ERROR"

        rec[
            "error"
        ] = str(exc)

        cache[idx] = rec
        audit_rows.append(rec)

        print(
            "ERROR:",
            str(exc)
        )

        return rec


# ============================================================
# SELECT NEGATIVES
# ============================================================

selected = []

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
        "Need exact negatives:",
        n_needed
    )
    print("=" * 75)

    scene_candidates = cand[
        cand[
            "scene_key"
        ].astype(str)
        == str(scene)
    ].copy()

    if len(
        scene_candidates
    ) == 0:

        raise RuntimeError(
            f"{scene}: "
            "no lightweight PASS candidates"
        )

    template = (
        positive[
            positive[
                "scene_key"
            ].astype(str)
            == str(scene)
        ]
        .iloc[0]
    )

    final_scene = None

    # --------------------------------------------------------
    # Keep spatial separation if possible.
    # Exact conversion result is cached, so lowering
    # separation does not redo already-tested candidates.
    # --------------------------------------------------------

    for min_sep in [
        600.0,
        480.0,
        300.0,
        0.0,
    ]:

        chosen = []

        print(
            "\nTrying separation:",
            min_sep,
            "m"
        )

        for idx, row in (
            scene_candidates
            .iterrows()
        ):

            # ----------------------------------------------
            # spatial separation from already-selected
            # ----------------------------------------------

            x = float(
                row["x"]
            )

            y = float(
                row["y"]
            )

            too_close = False

            for old in chosen:

                d = np.hypot(
                    x
                    -
                    float(
                        old["x"]
                    ),

                    y
                    -
                    float(
                        old["y"]
                    )
                )

                if d < min_sep:

                    too_close = True
                    break

            if too_close:
                continue

            # ----------------------------------------------
            # exact converter audit
            # ----------------------------------------------

            exact = (
                exact_test_candidate(
                    idx,
                    row,
                    template
                )
            )

            if (
                exact["status"]
                != "PASS"
            ):
                continue

            c = row.to_dict()

            c.update(
                exact
            )

            chosen.append(
                c
            )

            if (
                len(chosen)
                >= n_needed
            ):
                break

        if (
            len(chosen)
            >= n_needed
        ):

            final_scene = (
                chosen[
                    :n_needed
                ]
            )

            print(
                "\nSelected",
                len(final_scene),
                "with separation",
                min_sep,
                "m"
            )

            break

    if final_scene is None:

        # Save diagnostics before failing
        pd.DataFrame(
            audit_rows
        ).to_csv(
            OUT_AUDIT,
            index=False
        )

        pass_in_scene = sum(
            1
            for r in cache.values()
            if (
                r[
                    "scene_key"
                ] == scene
                and
                r[
                    "status"
                ] == "PASS"
            )
        )

        raise RuntimeError(
            f"{scene}: "
            f"need {n_needed} exact negatives "
            f"but only found "
            f"{pass_in_scene}"
        )

    # --------------------------------------------------------
    # Turn selected coordinates into manifest rows
    # --------------------------------------------------------

    for c in final_scene:

        rec = template.copy()

        rec[
            "query_id"
        ] = (
            f"AV3_NEG_EX_"
            f"{neg_counter:04d}"
        )

        rec[
            "label"
        ] = 0

        rec[
            "query_lat"
        ] = c[
            "query_lat"
        ]

        rec[
            "query_lon"
        ] = c[
            "query_lon"
        ]

        rec[
            "ground_truth_source"
        ] = (
            "matched_background_"
            "weak_negative_exact"
        )

        # positive-only Carbon Mapper fields
        for col in rec.index:

            if str(
                col
            ).startswith(
                "cm_"
            ):

                rec[col] = np.nan

        # CH4 / QA metadata
        for col in [
            "negative_snr_p99",
            "negative_ch4_p99",
            "qa_valid_fraction",
            "sns_p50",
            "nearest_official_plume_m",
        ]:

            if col in c:

                target = (
                    "negative_qa_valid_fraction"
                    if col
                    == "qa_valid_fraction"
                    else
                    "negative_sns_p50"
                    if col
                    == "sns_p50"
                    else
                    col
                )

                rec[
                    target
                ] = c[col]

        # lightweight prechecks
        for src, dst in [
            (
                "t0_missing_ratio_precheck",
                "precheck_t0_missing_ratio"
            ),
            (
                "t90_missing_ratio_precheck",
                "precheck_t90_missing_ratio"
            ),
            (
                "t180_missing_ratio_precheck",
                "precheck_t180_missing_ratio"
            ),
        ]:

            if src in c:

                rec[
                    dst
                ] = c[src]

        # exact converter checks
        rec[
            "exact_t0_missing_ratio"
        ] = c[
            "exact_t0_missing_ratio"
        ]

        rec[
            "exact_t90_missing_ratio"
        ] = c[
            "exact_t90_missing_ratio"
        ]

        rec[
            "exact_t180_missing_ratio"
        ] = c[
            "exact_t180_missing_ratio"
        ]

        selected.append(
            rec
        )

        neg_counter += 1


# ============================================================
# SAVE EXACT AUDIT
# ============================================================

audit = pd.DataFrame(
    audit_rows
)

audit.to_csv(
    OUT_AUDIT,
    index=False
)


# ============================================================
# EXACT NEGATIVES
# ============================================================

neg = pd.DataFrame(
    selected
)

neg.to_csv(
    OUT_NEG,
    index=False
)

print("\n")
print("=" * 75)
print("EXACT NEGATIVES")
print("=" * 75)

print(
    "Rows:",
    len(neg)
)

print(
    "\nCount per scene:"
)

print(
    neg[
        "scene_key"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

if len(neg) != 10:
    raise RuntimeError(
        f"Expected 10 exact negatives, "
        f"got {len(neg)}"
    )


# ============================================================
# FINAL 20
# ============================================================

all_cols = sorted(
    set(
        positive.columns
    )
    |
    set(
        neg.columns
    )
)

for c in all_cols:

    if c not in positive:
        positive[c] = np.nan

    if c not in neg:
        neg[c] = np.nan

final = pd.concat(
    [
        positive[
            all_cols
        ],

        neg[
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
# REPORT
# ============================================================

print("\n")
print("=" * 75)
print("EXACT PATCH-VALID FINAL MANIFEST")
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

print("\nLABELS PER SCENE")

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

print("\nEXACT NEGATIVE MISSING MAX")

for c in [
    "exact_t0_missing_ratio",
    "exact_t90_missing_ratio",
    "exact_t180_missing_ratio",
]:

    x = pd.to_numeric(
        neg[c],
        errors="coerce"
    )

    print(
        c,
        "=",
        round(
            x.max(),
            4
        )
    )


print("\nEXACT AUDIT STATUS")

print(
    audit[
        "status"
    ]
    .value_counts()
    .to_string()
)

if (
    "fail_stage"
    in audit.columns
):

    print(
        "\nEXACT REJECT STAGES"
    )

    print(
        audit[
            "fail_stage"
        ]
        .fillna("")
        .replace(
            "",
            "PASS"
        )
        .value_counts()
        .to_string()
    )


print("\nFILES")

print(
    "Exact negatives:",
    OUT_NEG.resolve()
)

print(
    "Exact final 20:",
    OUT_FINAL.resolve()
)

print(
    "Exact audit:",
    OUT_AUDIT.resolve()
)


# ============================================================
# HARD ASSERTIONS
# ============================================================

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
        "Final exact dataset "
        "is not 10/10"
    )

for c in [
    "exact_t0_missing_ratio",
    "exact_t90_missing_ratio",
    "exact_t180_missing_ratio",
]:

    if (
        pd.to_numeric(
            neg[c],
            errors="coerce"
        ).max()
        > MISSING_THRESHOLD
    ):

        raise RuntimeError(
            f"{c} failed"
        )


print("\n================================")
print("EXACT 10 NEGATIVES READY")
print("================================")
