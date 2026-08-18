from pathlib import Path
import pandas as pd


ROOT = Path("/project/6002520/yunjung1/MethaneFuse")
DATA = ROOT / "data/methaneair_full"

READINESS_PATH = (
    DATA / "sentinel2_v2_full_record_readiness.csv"
)

METADATA_CANDIDATES = [
    DATA / "ground_truth_confirmed_search_ready_v2.csv",
    DATA / "sentinel2_temporal_manifest_best_qa_v2.csv",
    DATA / "ground_truth_confirmed.csv",
]

WEAK_PATH = (
    DATA / "negative_pilot50_stage2_qa_pass.csv"
)

CONFIRMED_OUTPUT = (
    DATA / "sentinel2_v2_full_record_readiness_grouped.csv"
)

WEAK_OUTPUT = (
    DATA / "negative_pilot50_stage2_qa_pass_grouped.csv"
)


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
        })
    )


# ============================================================
# Read strict-readiness table
# ============================================================

readiness = pd.read_csv(
    READINESS_PATH,
    low_memory=False,
)

if "record_id" not in readiness.columns:
    raise SystemExit(
        "Readiness table 缺少 record_id。"
    )

if readiness["record_id"].duplicated().any():
    raise SystemExit(
        "Readiness table 的 record_id 不是唯一值。"
    )

strict = readiness[
    as_bool(readiness["strict_model_ready"])
].copy()

print("Readiness rows:", len(readiness))
print("Strict model-ready:", len(strict))


# ============================================================
# Collect metadata from available source tables
# ============================================================

metadata_fields = [
    "site_id",
    "facility_id",
    "latitude",
    "longitude",
    "source_positive_record_id",
    "ground_truth_source",
]

metadata = pd.DataFrame({
    "record_id": readiness["record_id"].astype(str)
})

for field in metadata_fields:
    metadata[field] = pd.NA

used_sources = []

for path in METADATA_CANDIDATES:
    if not path.exists():
        print("Missing metadata source:", path)
        continue

    source = pd.read_csv(
        path,
        low_memory=False,
    )

    if "record_id" not in source.columns:
        print("No record_id:", path)
        continue

    available = [
        column
        for column in metadata_fields
        if column in source.columns
    ]

    if not available:
        print("No useful metadata:", path)
        continue

    temp = source[
        ["record_id"] + available
    ].copy()

    temp["record_id"] = temp["record_id"].astype(str)

    # Every source should contribute at most one row per record.
    temp = temp.drop_duplicates(
        subset=["record_id"],
        keep="first",
    )

    renamed = {
        field: f"{field}__new"
        for field in available
    }

    temp = temp.rename(columns=renamed)

    metadata = metadata.merge(
        temp,
        on="record_id",
        how="left",
        validate="one_to_one",
    )

    for field in available:
        new_field = f"{field}__new"

        metadata[field] = (
            metadata[field]
            .combine_first(metadata[new_field])
        )

        metadata = metadata.drop(
            columns=[new_field]
        )

    used_sources.append(str(path))


# ============================================================
# Construct leakage-control group_id
#
# Prefer facility/site plus coordinates rounded to 3 decimals.
# 0.001 degree is roughly 100 m in latitude.
# ============================================================

site = normalize_text(metadata["site_id"])
facility = normalize_text(metadata["facility_id"])

base_name = facility.combine_first(site)

latitude = pd.to_numeric(
    metadata["latitude"],
    errors="coerce",
)

longitude = pd.to_numeric(
    metadata["longitude"],
    errors="coerce",
)

valid_coordinates = (
    latitude.between(-90, 90)
    & longitude.between(-180, 180)
)

coordinate_key = pd.Series(
    pd.NA,
    index=metadata.index,
    dtype="string",
)

coordinate_key.loc[valid_coordinates] = (
    "geo_"
    + latitude.loc[valid_coordinates]
        .round(3)
        .map(lambda value: f"{value:.3f}")
    + "_"
    + longitude.loc[valid_coordinates]
        .round(3)
        .map(lambda value: f"{value:.3f}")
)

group_id = pd.Series(
    pd.NA,
    index=metadata.index,
    dtype="string",
)

both_available = (
    base_name.notna()
    & coordinate_key.notna()
)

group_id.loc[both_available] = (
    base_name.loc[both_available]
    .str.replace(r"\s+", "_", regex=True)
    + "__"
    + coordinate_key.loc[both_available]
)

only_coordinates = (
    base_name.isna()
    & coordinate_key.notna()
)

group_id.loc[only_coordinates] = (
    coordinate_key.loc[only_coordinates]
)

only_name = (
    base_name.notna()
    & coordinate_key.isna()
)

group_id.loc[only_name] = (
    base_name.loc[only_name]
    .str.replace(r"\s+", "_", regex=True)
)

metadata["group_id"] = group_id


# ============================================================
# Merge metadata into readiness
# ============================================================

confirmed_grouped = readiness.merge(
    metadata,
    on="record_id",
    how="left",
    validate="one_to_one",
    suffixes=("", "__metadata"),
)

for field in metadata_fields + ["group_id"]:
    metadata_field = f"{field}__metadata"

    if metadata_field in confirmed_grouped.columns:
        if field in confirmed_grouped.columns:
            confirmed_grouped[field] = (
                confirmed_grouped[field]
                .combine_first(
                    confirmed_grouped[metadata_field]
                )
            )
        else:
            confirmed_grouped[field] = (
                confirmed_grouped[metadata_field]
            )

        confirmed_grouped = confirmed_grouped.drop(
            columns=[metadata_field]
        )


strict_grouped = confirmed_grouped[
    as_bool(confirmed_grouped["strict_model_ready"])
].copy()

missing_group = strict_grouped[
    "group_id"
].isna()

print("\nMetadata sources used:")
for source in used_sources:
    print("-", source)

print("\nStrict grouped audit:")
print("Strict rows:", len(strict_grouped))
print(
    "group_id available:",
    int((~missing_group).sum()),
)
print(
    "group_id missing:",
    int(missing_group.sum()),
)
print(
    "Unique groups:",
    strict_grouped["group_id"].nunique(
        dropna=True
    ),
)

if "label" in strict_grouped.columns:
    label_numeric = pd.to_numeric(
        strict_grouped["label"],
        errors="coerce",
    )

    group_label = strict_grouped.assign(
        _label=label_numeric
    ).groupby(
        "group_id",
        dropna=True,
    )["_label"].agg(
        records="size",
        positives=lambda values: int(
            (values == 1).sum()
        ),
        negatives=lambda values: int(
            (values == 0).sum()
        ),
    )

    print(
        "Groups containing negative:",
        int((group_label["negatives"] > 0).sum()),
    )

if missing_group.any():
    columns = [
        column
        for column in [
            "record_id",
            "label",
            "site_id",
            "facility_id",
            "latitude",
            "longitude",
            "ground_truth_source",
        ]
        if column in strict_grouped.columns
    ]

    print("\nStrict rows missing group_id:")
    print(
        strict_grouped.loc[
            missing_group,
            columns,
        ].head(20).to_string(index=False)
    )

    raise SystemExit(
        "\n仍有 strict model-ready records 缺少 group_id，"
        "暫不建立 split。"
    )

confirmed_grouped.to_csv(
    CONFIRMED_OUTPUT,
    index=False,
)


# ============================================================
# Give weak negatives the SAME group as their source positive
# ============================================================

weak = pd.read_csv(
    WEAK_PATH,
    low_memory=False,
)

required_weak = {
    "record_id",
    "source_positive_record_id",
}

missing_weak = required_weak - set(weak.columns)

if missing_weak:
    raise SystemExit(
        f"Weak table 缺少欄位：{sorted(missing_weak)}"
    )

parent_map = (
    confirmed_grouped[
        ["record_id", "group_id"]
    ]
    .dropna(subset=["group_id"])
    .drop_duplicates("record_id")
    .set_index("record_id")["group_id"]
)

weak["source_positive_record_id"] = (
    weak["source_positive_record_id"]
    .astype(str)
)

weak["group_id"] = (
    weak["source_positive_record_id"]
    .map(parent_map)
)

print("\nWeak-negative grouped audit:")
print("Weak rows:", len(weak))
print(
    "Mapped to parent group:",
    int(weak["group_id"].notna().sum()),
)
print(
    "Missing parent group:",
    int(weak["group_id"].isna().sum()),
)
print(
    "Weak unique groups:",
    weak["group_id"].nunique(
        dropna=True
    ),
)

if weak["group_id"].isna().any():
    print("\nWeak rows without parent group:")
    print(
        weak.loc[
            weak["group_id"].isna(),
            [
                "record_id",
                "source_positive_record_id",
                "site_id",
            ],
        ].to_string(index=False)
    )

    raise SystemExit(
        "\n有 weak negatives 無法對應其 source positive，"
        "暫不建立 augmented split。"
    )

weak.to_csv(
    WEAK_OUTPUT,
    index=False,
)

print("\nSaved:")
print(CONFIRMED_OUTPUT)
print(WEAK_OUTPUT)
