import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import earthaccess
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "AVIRIS3_MethaneFuse_build"
)

OVERLAP = Path(
    "aviris3_l2a_l2b_true_overlap_audit.csv"
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

CH4_SOURCE = (
    ROOT
    / "L2B_REMATCH_CH4"
)

UNC_SOURCE = (
    ROOT
    / "L2B_REMATCH_UNC"
)

SNS_SOURCE = (
    ROOT
    / "L2B_REMATCH_SNS"
)

MATCH_CH4 = (
    ROOT
    / "L2B_MATCHED_CH4"
)

MATCH_UNC = (
    ROOT
    / "L2B_MATCHED_UNC"
)

MATCH_SNS = (
    ROOT
    / "L2B_MATCHED_SNS"
)

SELECTED_OUT = Path(
    "aviris3_selected_l2b_master_tiles.csv"
)

OLD_SCRIPT = Path(
    "build_patchvalid_balanced20.py"
)

NEW_SCRIPT = Path(
    "build_patchvalid_balanced20_rematched.py"
)


for d in [
    UNC_SOURCE,
    SNS_SOURCE,
    MATCH_CH4,
    MATCH_UNC,
    MATCH_SNS,
]:
    d.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 1. COUNT TRUE PATCH-VALID POSITIVES PER L2A SCENE
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
    manifest["query_id"]
    .astype(str)
    .isin(pass_ids)
].copy()

need = (
    pos["scene_key"]
    .value_counts()
    .sort_index()
)

print("\n================================")
print("PATCH-VALID POSITIVES")
print("================================")

print(need.to_string())

if len(pos) != 10:
    raise RuntimeError(
        f"Expected 10 positives, got {len(pos)}"
    )


# ============================================================
# 2. CHOOSE ONE MASTER L2B TILE PER L2A SCENE
#
# Requirements:
# - can fit 480m
# - contains ALL patch-valid positives from that L2A scene
# - among those, choose largest L2A/L2B overlap
# ============================================================

ov = pd.read_csv(OVERLAP)

selected = []

for scene, npos in need.items():

    x = ov[
        ov["l2a_scene_key"]
        .astype(str)
        == str(scene)
    ].copy()

    x["can_fit_480m_patch"] = (
        x["can_fit_480m_patch"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    x["n_patchvalid_positive_inside"] = (
        pd.to_numeric(
            x[
                "n_patchvalid_positive_inside"
            ],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    valid = x[
        x["can_fit_480m_patch"]
        &
        (
            x[
                "n_patchvalid_positive_inside"
            ]
            >= int(npos)
        )
    ].copy()

    if len(valid) == 0:
        raise RuntimeError(
            f"{scene}: no L2B tile "
            f"covers all {npos} positives"
        )

    valid = valid.sort_values(
        "overlap_area_km2",
        ascending=False
    )

    best = valid.iloc[0].copy()

    selected.append({
        "l2a_scene_key":
            scene,

        "positive_count":
            int(npos),

        "l2b_scene_key":
            best[
                "l2b_scene_key"
            ],

        "overlap_area_km2":
            best[
                "overlap_area_km2"
            ],

        "positive_ids":
            best[
                "patchvalid_positive_ids"
            ],
    })


sel = pd.DataFrame(selected)

print("\n================================")
print("SELECTED MASTER L2B TILES")
print("================================")

print(
    sel.to_string(
        index=False
    )
)


# ============================================================
# 3. JOIN DOWNLOAD URLS
# ============================================================

rematch = pd.read_csv(REMATCH)

sel = sel.merge(
    rematch[
        [
            "l2a_scene_key",
            "l2b_scene_key",
            "ch4_filename",
            "ch4_url",
            "unc_url",
            "sns_url",
        ]
    ],
    on=[
        "l2a_scene_key",
        "l2b_scene_key",
    ],
    how="left",
    validate="one_to_one"
)

if sel[
    [
        "ch4_url",
        "unc_url",
        "sns_url",
    ]
].isna().any().any():

    raise RuntimeError(
        "Missing CH4/UNC/SNS URLs "
        "after rematch join"
    )

sel.to_csv(
    SELECTED_OUT,
    index=False
)


# ============================================================
# 4. DOWNLOAD ONLY 4 UNC + 4 SNS
# ============================================================

auth = earthaccess.login(
    strategy="netrc"
)

print(
    "\nAuthenticated:",
    auth.authenticated
)

unc_urls = sorted(
    sel["unc_url"]
    .astype(str)
    .unique()
)

sns_urls = sorted(
    sel["sns_url"]
    .astype(str)
    .unique()
)

print("\n================================")
print("DOWNLOAD")
print("================================")

print(
    "UNC files:",
    len(unc_urls)
)

print(
    "SNS files:",
    len(sns_urls)
)

unc_paths = earthaccess.download(
    unc_urls,
    local_path=UNC_SOURCE,
    threads=4
)

sns_paths = earthaccess.download(
    sns_urls,
    local_path=SNS_SOURCE,
    threads=4
)

print(
    "UNC downloaded/available:",
    len(unc_paths)
)

print(
    "SNS downloaded/available:",
    len(sns_paths)
)


# ============================================================
# 5. LOCAL FILE LOOKUP
# ============================================================

def basename(url):

    return Path(
        urlparse(str(url)).path
    ).name


def locate(folder, filename):

    p = folder / filename

    if p.exists():
        return p

    hits = list(
        folder.glob(
            f"*{filename}*"
        )
    )

    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"{filename} not found "
        f"in {folder}"
    )


# ============================================================
# 6. CREATE MATCHED ALIASES
#
# build_patchvalid_balanced20.py originally expects
# L2B filenames to key on the L2A scene.
#
# We preserve provenance in the alias filename:
#
# L2A_SCENE__SRC__REAL_L2B_FILENAME
#
# build_scene_map() sees the FIRST AV3 key,
# therefore correctly maps to the L2A scene.
# ============================================================

for d in [
    MATCH_CH4,
    MATCH_UNC,
    MATCH_SNS,
]:

    # Remove previous aliases only.
    for p in d.iterdir():

        if p.is_symlink():
            p.unlink()


for _, row in sel.iterrows():

    l2a = str(
        row[
            "l2a_scene_key"
        ]
    )

    l2b = str(
        row[
            "l2b_scene_key"
        ]
    )

    ch4_name = str(
        row[
            "ch4_filename"
        ]
    )

    unc_name = basename(
        row[
            "unc_url"
        ]
    )

    sns_name = basename(
        row[
            "sns_url"
        ]
    )

    ch4_src = locate(
        CH4_SOURCE,
        ch4_name
    )

    unc_src = locate(
        UNC_SOURCE,
        unc_name
    )

    sns_src = locate(
        SNS_SOURCE,
        sns_name
    )

    aliases = [
        (
            ch4_src,
            MATCH_CH4
            / (
                f"{l2a}"
                f"__SRC__"
                f"{ch4_src.name}"
            )
        ),
        (
            unc_src,
            MATCH_UNC
            / (
                f"{l2a}"
                f"__SRC__"
                f"{unc_src.name}"
            )
        ),
        (
            sns_src,
            MATCH_SNS
            / (
                f"{l2a}"
                f"__SRC__"
                f"{sns_src.name}"
            )
        ),
    ]

    for src, dst in aliases:

        if dst.exists() or dst.is_symlink():
            dst.unlink()

        os.symlink(
            src.resolve(),
            dst
        )

    print(
        l2a,
        "->",
        l2b
    )


# ============================================================
# 7. CHECK ALIASES
# ============================================================

print("\n================================")
print("MATCHED LOCAL FILES")
print("================================")

print(
    "CH4:",
    len(
        list(
            MATCH_CH4.glob("*")
        )
    )
)

print(
    "UNC:",
    len(
        list(
            MATCH_UNC.glob("*")
        )
    )
)

print(
    "SNS:",
    len(
        list(
            MATCH_SNS.glob("*")
        )
    )
)


# ============================================================
# 8. PATCH THE NEGATIVE GENERATOR
# ============================================================

if not OLD_SCRIPT.exists():

    raise FileNotFoundError(
        f"Missing:\n"
        f"{OLD_SCRIPT.resolve()}"
    )

text = OLD_SCRIPT.read_text()

replacements = {
    'ORT_DIR = ROOT / "L2B_CH4_ORT"':
        'ORT_DIR = ROOT / "L2B_MATCHED_CH4"',

    'UNC_DIR = ROOT / "L2B_CH4_UNC"':
        'UNC_DIR = ROOT / "L2B_MATCHED_UNC"',

    'SNS_DIR = ROOT / "L2B_CH4_SNS"':
        'SNS_DIR = ROOT / "L2B_MATCHED_SNS"',
}

for old, new in replacements.items():

    if old not in text:

        raise RuntimeError(
            f"Could not find line "
            f"to patch:\n{old}"
        )

    text = text.replace(
        old,
        new
    )

NEW_SCRIPT.write_text(
    text
)

print("\nCreated:")
print(
    NEW_SCRIPT.resolve()
)


# ============================================================
# 9. RUN PATCH-VALID NEGATIVE GENERATOR
# ============================================================

print("\n================================")
print("RUN REMATCHED NEGATIVE BUILD")
print("================================")

result = subprocess.run(
    [
        "python3",
        str(NEW_SCRIPT),
    ]
)

print(
    "\nReturn code:",
    result.returncode
)

if result.returncode != 0:

    raise SystemExit(
        result.returncode
    )

print(
    "\nREMATCHED NEGATIVE "
    "GENERATION COMPLETE"
)
