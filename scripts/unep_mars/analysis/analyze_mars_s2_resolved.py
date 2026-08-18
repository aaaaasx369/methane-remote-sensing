from pathlib import Path
import re
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CANDIDATES = Path(
    "methanefuse_candidates/mars_s2_candidates.csv"
)

RESOLVER = Path(
    "methanefuse_candidates/mars_s2_exact_t0_resolver.csv"
)

OUTDIR = Path(
    "methanefuse_candidates/mars_s2_export"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

cand = pd.read_csv(CANDIDATES)
res = pd.read_csv(RESOLVER)

print("=" * 80)
print("MARS SENTINEL-2 POST-RESOLUTION ANALYSIS")
print("=" * 80)

print("\nCandidate rows :", len(cand))
print("Resolver rows  :", len(res))


# ============================================================
# NORMALIZE
# ============================================================

res["status"] = (
    res["status"]
    .fillna("")
    .astype(str)
)

res["resolved_product_id"] = (
    res["resolved_product_id"]
    .fillna("")
    .astype(str)
)

res["lat"] = pd.to_numeric(
    res["lat"],
    errors="coerce"
)

res["lon"] = pd.to_numeric(
    res["lon"],
    errors="coerce"
)


# ============================================================
# SPLIT RESOLVED / UNRESOLVED
# ============================================================

resolved = res[
    res["status"].isin([
        "RESOLVED",
        "SUBMITTED"
    ])
    &
    res["resolved_product_id"].ne("")
].copy()

unresolved = res[
    ~res.index.isin(resolved.index)
].copy()


print("\n" + "=" * 80)
print("RESOLUTION")
print("=" * 80)

print("Resolved   :", len(resolved))
print("Unresolved :", len(unresolved))

if len(res):
    print(
        "Resolution rate:",
        f"{100 * len(resolved) / len(res):.4f}%"
    )


# ============================================================
# UNRESOLVED ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("UNRESOLVED 18")
print("=" * 80)


def parse_tile(tile):

    if pd.isna(tile):
        return {}

    tile = str(tile)

    m = re.match(
        r"^(S2[ABC])_MSIL1C_"
        r"(\d{8}T\d{6})_"
        r"N(\d{4})_"
        r"R(\d{3})_"
        r"T([0-9A-Z]{5})_"
        r"(\d{8}T\d{6})$",
        tile
    )

    if not m:
        return {}

    return {
        "parsed_mission": m.group(1),
        "parsed_sensing": m.group(2),
        "parsed_baseline": m.group(3),
        "parsed_orbit": int(m.group(4)),
        "parsed_mgrs": m.group(5),
        "parsed_generation": m.group(6),
    }


parsed_rows = []

for _, row in unresolved.iterrows():

    x = row.to_dict()

    x.update(
        parse_tile(
            row.get("mars_tile")
        )
    )

    parsed_rows.append(x)


unresolved_detail = pd.DataFrame(
    parsed_rows
)

unresolved_path = (
    OUTDIR /
    "01_unresolved_18.csv"
)

unresolved_detail.to_csv(
    unresolved_path,
    index=False
)


if len(unresolved_detail):

    show_cols = [
        "id_plume",
        "source_name",
        "event_time",
        "lat",
        "lon",
        "mars_tile",
        "parsed_mission",
        "parsed_sensing",
        "parsed_orbit",
        "parsed_mgrs",
        "status",
        "error",
    ]

    show_cols = [
        c for c in show_cols
        if c in unresolved_detail.columns
    ]

    print(
        unresolved_detail[
            show_cols
        ].to_string(
            index=False
        )
    )


# ============================================================
# UNIQUE SCENE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("SCENE ANALYSIS")
print("=" * 80)

n_scene = (
    resolved[
        "resolved_product_id"
    ]
    .nunique()
)

print(
    "Resolved plume rows :",
    len(resolved)
)

print(
    "Unique L2A scenes   :",
    n_scene
)

print(
    "Repeated scene refs :",
    len(resolved) - n_scene
)


scene_counts = (
    resolved
    .groupby(
        "resolved_product_id",
        dropna=False
    )
    .agg(
        plume_rows=(
            "id_plume",
            "size"
        ),
        unique_sources=(
            "source_name",
            "nunique"
        ),
        minimum_lat=(
            "lat",
            "min"
        ),
        maximum_lat=(
            "lat",
            "max"
        ),
        minimum_lon=(
            "lon",
            "min"
        ),
        maximum_lon=(
            "lon",
            "max"
        ),
    )
    .reset_index()
    .sort_values(
        "plume_rows",
        ascending=False
    )
)

scene_counts.to_csv(
    OUTDIR /
    "02_scene_counts.csv",
    index=False
)


print("\nTop scenes by plume count:")

print(
    scene_counts
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# IMPORTANT:
# SAME SCENE != SAME EXPORT
#
# Crop is centered on lat/lon.
#
# Therefore deduplication should use:
#
# resolved scene
# +
# source location
#
# rather than scene alone.
# ============================================================

# Coordinates in MARS are already roughly source-level coordinates.
# Round only to 5 decimal places (~1 m scale latitude)
# to avoid trivial floating point differences.

resolved["lat_key"] = (
    resolved["lat"]
    .round(5)
)

resolved["lon_key"] = (
    resolved["lon"]
    .round(5)
)


resolved["scene_location_key"] = (
    resolved["resolved_product_id"]
    .astype(str)
    +
    "__"
    +
    resolved["lat_key"]
    .astype(str)
    +
    "__"
    +
    resolved["lon_key"]
    .astype(str)
)


n_scene_location = (
    resolved[
        "scene_location_key"
    ]
    .nunique()
)


print("\n" + "=" * 80)
print("SCENE + LOCATION DEDUPLICATION")
print("=" * 80)

print(
    "Resolved plume rows            :",
    len(resolved)
)

print(
    "Unique scenes                  :",
    n_scene
)

print(
    "Unique scene + exact locations :",
    n_scene_location
)

print(
    "Exact crop duplicates removable:",
    len(resolved) - n_scene_location
)


# ============================================================
# ALSO CHECK scene + source_name
# ============================================================

resolved["scene_source_key"] = (
    resolved["resolved_product_id"]
    .astype(str)
    +
    "__"
    +
    resolved[
        "source_name"
    ]
    .fillna("")
    .astype(str)
)

n_scene_source = (
    resolved[
        "scene_source_key"
    ]
    .nunique()
)

print(
    "Unique scene + source_name     :",
    n_scene_source
)

print(
    "Duplicates by scene+source     :",
    len(resolved) - n_scene_source
)


# ============================================================
# CHECK WHETHER SAME SOURCE HAS DIFFERENT COORDINATES
# ============================================================

source_coord = (
    resolved
    .groupby(
        "source_name",
        dropna=False
    )
    .agg(
        rows=(
            "id_plume",
            "size"
        ),
        coordinate_pairs=(
            "scene_location_key",
            lambda x: len(
                set(
                    s.split("__", 1)[1]
                    for s in x
                )
            )
        ),
        min_lat=(
            "lat",
            "min"
        ),
        max_lat=(
            "lat",
            "max"
        ),
        min_lon=(
            "lon",
            "min"
        ),
        max_lon=(
            "lon",
            "max"
        ),
    )
    .reset_index()
)

source_coord.to_csv(
    OUTDIR /
    "03_source_coordinate_variation.csv",
    index=False
)


# ============================================================
# CREATE EXACT EXPORT MANIFEST
#
# One GeoTIFF per:
#   resolved_product_id + lat/lon
#
# Keeps a representative id_plume but also stores every plume
# ID sharing the exact same crop.
# ============================================================

groups = []

for key, g in resolved.groupby(
    "scene_location_key",
    sort=False
):

    first = g.iloc[0]

    groups.append({
        "export_id":
            f"MARS_S2_{len(groups)+1:06d}",

        "resolved_product_id":
            first[
                "resolved_product_id"
            ],

        "lat":
            first["lat_key"],

        "lon":
            first["lon_key"],

        "representative_id_plume":
            first["id_plume"],

        "source_name":
            first.get(
                "source_name",
                ""
            ),

        "n_plume_rows":
            len(g),

        "all_id_plumes":
            "|".join(
                g[
                    "id_plume"
                ]
                .astype(str)
                .tolist()
            ),

        "all_source_names":
            "|".join(
                sorted(
                    set(
                        g[
                            "source_name"
                        ]
                        .dropna()
                        .astype(str)
                        .tolist()
                    )
                )
            ),

        "match_method":
            first.get(
                "match_method",
                ""
            ),

        "gee_system_time":
            first.get(
                "gee_system_time",
                ""
            ),

        "mars_sensing_time":
            first.get(
                "mars_sensing_time",
                ""
            ),

        "mars_mgrs":
            first.get(
                "mars_mgrs",
                ""
            ),

        "mars_orbit":
            first.get(
                "mars_orbit",
                ""
            ),

        "mission":
            first.get(
                "mission",
                ""
            ),
    })


manifest = pd.DataFrame(
    groups
)


# ============================================================
# EXPORT FILE NAME
# ============================================================

manifest[
    "output_filename"
] = (
    manifest[
        "export_id"
    ]
    +
    "__t0.tif"
)


manifest_path = (
    OUTDIR /
    "04_exact_t0_export_manifest.csv"
)

manifest.to_csv(
    manifest_path,
    index=False
)


# ============================================================
# PLUME -> EXPORT LOOKUP
# ============================================================

lookup = resolved[
    [
        "id_plume",
        "source_name",
        "resolved_product_id",
        "lat",
        "lon",
        "scene_location_key",
    ]
].copy()

key_to_export = (
    manifest
    .set_index(
        manifest[
            "resolved_product_id"
        ].astype(str)
        +
        "__"
        +
        manifest["lat"]
        .astype(str)
        +
        "__"
        +
        manifest["lon"]
        .astype(str)
    )[
        "export_id"
    ]
    .to_dict()
)


def make_key(row):
    return (
        str(
            row[
                "resolved_product_id"
            ]
        )
        +
        "__"
        +
        str(
            round(
                float(row["lat"]),
                5
            )
        )
        +
        "__"
        +
        str(
            round(
                float(row["lon"]),
                5
            )
        )
    )


lookup["export_key"] = (
    lookup.apply(
        make_key,
        axis=1
    )
)

lookup["export_id"] = (
    lookup[
        "export_key"
    ]
    .map(
        key_to_export
    )
)

lookup.to_csv(
    OUTDIR /
    "05_plume_to_export_lookup.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL EXPORT MANIFEST")
print("=" * 80)

print(
    "Original MARS S2 rows         :",
    len(res)
)

print(
    "Exact L2A resolved rows       :",
    len(resolved)
)

print(
    "Unresolved rows               :",
    len(unresolved)
)

print(
    "Unique L2A scenes             :",
    n_scene
)

print(
    "Unique scene-location exports :",
    len(manifest)
)

print(
    "Crop exports saved by dedupe  :",
    len(resolved) - len(manifest)
)

print()
print("Files:")

for f in sorted(
    OUTDIR.glob("*.csv")
):
    print(" ", f)

