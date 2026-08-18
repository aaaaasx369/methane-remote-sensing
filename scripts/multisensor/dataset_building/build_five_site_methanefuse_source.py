from pathlib import Path
import re
import pandas as pd


PROJECT_ROOT = Path(
    "/Users/happydoraaa/methane_release_project"
)

OUTPUTS = PROJECT_ROOT / "outputs"

OUTPUT_PATH = (
    OUTPUTS
    / "601_five_site_ground_truth_for_methanefuse.csv"
)

TARGET_SITES = {
    "Casa_Grande",
    "Ehrenberg",
    "MA_site_038",
    "MA_site_043",
    "MA_site_073",
}


COLUMN_ALIASES = {
    "site": [
        "site",
        "site_id",
        "held_out_site",
        "master_site",
        "canonical_site",
        "site_name",
    ],
    "label": [
        "label",
        "true_label",
        "final_label",
        "target",
        "physical_release_gt",
        "plume_detection_label",
        "ground_truth_label",
    ],
    "scene_id": [
        "scene_id",
        "s2_scene_id",
        "sentinel2_scene_id",
        "sentinel_2_scene_id",
        "system:index",
        "system_index",
        "s2_system_index",
        "s2_product_id",
        "product_id",
    ],
    "acquisition_time_utc": [
        "acquisition_time_utc",
        "s2_acquisition_time_utc",
        "scene_time_utc",
        "scene_datetime_utc",
        "datetime_utc",
        "acquisition_time",
        "acquisition_datetime",
        "scene_time",
        "timestamp",
    ],
    "lat": [
        "lat",
        "latitude",
        "source_latitude",
        "site_latitude",
        "site_centroid_latitude",
        "center_lat",
        "centre_lat",
    ],
    "lon": [
        "lon",
        "lng",
        "longitude",
        "source_longitude",
        "site_longitude",
        "site_centroid_longitude",
        "center_lon",
        "centre_lon",
    ],
    "sensor": [
        "sensor",
        "satellite",
        "platform",
        "s2_sensor",
    ],
    "release_rate": [
        "release_rate_kg_hr",
        "metered_release_rate_kg_hr",
        "ground_truth_rate_kg_hr",
        "emission_rate_kg_hr",
        "release_rate",
    ],
}


def find_column(columns, aliases):
    lookup = {
        str(column).strip().lower(): column
        for column in columns
    }

    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]

    return None


def normalize_site(value):
    raw = str(value).strip()

    key = re.sub(
        r"[^a-z0-9]+",
        "_",
        raw.lower(),
    ).strip("_")

    aliases = {
        "casa_grande": "Casa_Grande",
        "casagrande": "Casa_Grande",
        "casa_grande_az_release_stack": "Casa_Grande",
        "casa_grande_az_release_stacks": "Casa_Grande",

        "ehrenberg": "Ehrenberg",
        "ehrenberg_az_release_stack": "Ehrenberg",
        "ehrenberg_az_release_stacks": "Ehrenberg",

        "site_038": "MA_site_038",
        "ma_site_038": "MA_site_038",
        "methaneair_site_038": "MA_site_038",

        "site_043": "MA_site_043",
        "ma_site_043": "MA_site_043",
        "methaneair_site_043": "MA_site_043",

        "site_073": "MA_site_073",
        "ma_site_073": "MA_site_073",
        "methaneair_site_073": "MA_site_073",
    }

    return aliases.get(key, raw)


def parse_label(series):
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    positive = text.isin([
        "true",
        "yes",
        "positive",
        "plume",
        "release",
        "on",
        "tp",
        "1",
    ])

    negative = text.isin([
        "false",
        "no",
        "negative",
        "no_plume",
        "no plume",
        "reference",
        "off",
        "tn",
        "0",
    ])

    result = numeric.copy()

    result.loc[numeric.isna() & positive] = 1
    result.loc[numeric.isna() & negative] = 0

    return result


candidates = []
near_matches = []

for path in sorted(OUTPUTS.rglob("*.csv")):
    if path == OUTPUT_PATH:
        continue

    try:
        df = pd.read_csv(
            path,
            low_memory=False,
        )
    except Exception:
        continue

    columns = {
        target: find_column(
            df.columns,
            aliases,
        )
        for target, aliases in COLUMN_ALIASES.items()
    }

    if columns["site"] is None:
        continue

    normalized_sites = (
        df[columns["site"]]
        .dropna()
        .map(normalize_site)
    )

    site_set = set(normalized_sites.unique())
    overlap = len(site_set & TARGET_SITES)

    if overlap < 3:
        continue

    required_targets = [
        "site",
        "label",
        "scene_id",
        "acquisition_time_utc",
        "lat",
        "lon",
    ]

    missing_targets = [
        target
        for target in required_targets
        if columns[target] is None
    ]

    labels = None
    positive = None
    negative = None

    if columns["label"] is not None:
        labels = parse_label(
            df[columns["label"]]
        )

        positive = int((labels == 1).sum())
        negative = int((labels == 0).sum())

    near_matches.append({
        "path": path,
        "rows": len(df),
        "overlap": overlap,
        "sites": sorted(site_set),
        "columns": columns,
        "missing": missing_targets,
        "positive": positive,
        "negative": negative,
    })

    if overlap < 5 or missing_targets:
        continue

    score = 0

    score += overlap * 100

    # 優先選擇目前鎖定的 75-row dataset。
    score -= abs(len(df) - 75)

    if len(df) == 75:
        score += 1000

    if positive == 15:
        score += 300

    if negative == 60:
        score += 300

    filename = path.name.lower()

    if "multisite" in filename:
        score += 100

    if "manifest" in filename:
        score += 50

    if "master" in filename:
        score += 50

    if "exact" in filename:
        score -= 100

    candidates.append({
        "score": score,
        "path": path,
        "df": df,
        "columns": columns,
        "positive": positive,
        "negative": negative,
        "sites": sorted(site_set),
    })


if not candidates:
    print(
        "\n找不到同時具有五站與以下欄位的 CSV："
    )
    print(
        "site, label, scene_id, acquisition_time, lat, lon"
    )

    print("\n最接近的候選：")

    near_matches.sort(
        key=lambda item: (
            -item["overlap"],
            len(item["missing"]),
            abs(item["rows"] - 75),
        )
    )

    for item in near_matches[:15]:
        print("=" * 100)
        print("Path:", item["path"])
        print("Rows:", item["rows"])
        print("Five-site overlap:", item["overlap"])
        print("Positive:", item["positive"])
        print("Negative:", item["negative"])
        print("Missing:", item["missing"])
        print("Mapped columns:", item["columns"])
        print("Sites:", item["sites"])

    raise RuntimeError(
        "沒有可直接建立 MethaneFuse 五站來源表的 CSV。"
    )


candidates.sort(
    key=lambda item: item["score"],
    reverse=True,
)

print("\n可用候選：")

for item in candidates[:10]:
    print("=" * 100)
    print("Score:", item["score"])
    print("Path:", item["path"])
    print("Rows:", len(item["df"]))
    print("Positive:", item["positive"])
    print("Negative:", item["negative"])
    print("Sites:", item["sites"])
    print("Mapped columns:", item["columns"])


selected = candidates[0]

source_path = selected["path"]
source_df = selected["df"].copy()
columns = selected["columns"]

print("\n" + "#" * 100)
print("Selected source:")
print(source_path)
print("#" * 100)


standard = source_df.copy()

standard["site"] = (
    source_df[columns["site"]]
    .map(normalize_site)
)

standard["physical_release_gt"] = (
    parse_label(
        source_df[columns["label"]]
    )
)

standard["scene_id"] = (
    source_df[columns["scene_id"]]
    .astype(str)
    .str.strip()
)

standard["acquisition_time_utc"] = pd.to_datetime(
    source_df[columns["acquisition_time_utc"]],
    errors="coerce",
    utc=True,
)

standard["lat"] = pd.to_numeric(
    source_df[columns["lat"]],
    errors="coerce",
)

standard["lon"] = pd.to_numeric(
    source_df[columns["lon"]],
    errors="coerce",
)


if columns["sensor"] is not None:
    standard["sensor"] = (
        source_df[columns["sensor"]]
        .astype(str)
        .str.strip()
    )
else:
    standard["sensor"] = "Sentinel-2"


# 只留下五個目標 site。
standard = standard[
    standard["site"].isin(TARGET_SITES)
].copy()


# 若表中同時包含其他衛星，只留下 Sentinel-2。
sensor_text = (
    standard["sensor"]
    .astype(str)
    .str.lower()
)

s2_mask = (
    sensor_text.str.contains("sentinel")
    | sensor_text.str.contains(r"\bs2\b", regex=True)
)

if s2_mask.any():
    standard = standard[s2_mask].copy()

standard["sensor"] = "Sentinel-2"


# Ground-truth provenance 不可混為一談。
standard["label_provenance"] = standard["site"].map({
    "Casa_Grande": "physical_release",
    "Ehrenberg": "physical_release",
    "MA_site_038": "plume_reference",
    "MA_site_043": "plume_reference",
    "MA_site_073": "plume_reference",
})


if columns["release_rate"] is not None:
    standard["release_rate_kg_hr"] = pd.to_numeric(
        source_df.loc[
            standard.index,
            columns["release_rate"],
        ],
        errors="coerce",
    )
elif "release_rate_kg_hr" not in standard.columns:
    standard["release_rate_kg_hr"] = pd.NA


# 為 exact script 可能使用的可用性欄位提供預設值。
for column in [
    "bin_usable",
    "s2_usable",
    "selected_for_external_eval",
]:
    if column not in standard.columns:
        standard[column] = True


required_output = [
    "sensor",
    "scene_id",
    "site",
    "acquisition_time_utc",
    "physical_release_gt",
    "lat",
    "lon",
    "label_provenance",
]

before = len(standard)

standard = standard.dropna(
    subset=[
        "scene_id",
        "site",
        "acquisition_time_utc",
        "physical_release_gt",
        "lat",
        "lon",
    ]
).copy()

standard["physical_release_gt"] = (
    standard["physical_release_gt"]
    .astype(int)
)

standard = standard[
    standard["physical_release_gt"].isin([0, 1])
].copy()

standard["acquisition_time_utc"] = (
    standard["acquisition_time_utc"]
    .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
)

standard = standard.drop_duplicates(
    subset=[
        "site",
        "scene_id",
    ],
    keep="first",
).reset_index(drop=True)


# 將標準欄位放前面，原始欄位保留在後方供 audit。
remaining_columns = [
    column
    for column in standard.columns
    if column not in required_output
]

standard = standard[
    required_output + remaining_columns
]

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

standard.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\nCreated:")
print(OUTPUT_PATH)

print("\nSource rows:", before)
print("Output rows:", len(standard))

print("\nSite × label:")
print(
    pd.crosstab(
        standard["site"],
        standard["physical_release_gt"],
        margins=True,
    ).to_string()
)

print("\nLabel provenance:")
print(
    standard["label_provenance"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nAcquisition time range:")
print(
    standard.groupby("site")["acquisition_time_utc"]
    .agg(["min", "max"])
    .to_string()
)

print("\nRequired-column missing counts:")
print(
    standard[required_output]
    .isna()
    .sum()
    .to_string()
)
