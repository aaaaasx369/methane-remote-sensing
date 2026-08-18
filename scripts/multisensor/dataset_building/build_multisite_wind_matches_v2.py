from pathlib import Path
import math
import re

import numpy as np
import pandas as pd


ROOT = Path("/Users/happydoraaa/methane_release_project")
OUTPUTS = ROOT / "outputs"

MASTER_PATH = OUTPUTS / "36_multisite_s2_master_table.csv"

WIND_TABLES = [
    # 優先使用較乾淨的表；若有完全相同紀錄，保留前面的版本
    OUTPUTS / "06_strict_event_table_for_gee.csv",
    OUTPUTS / "05_clean_event_table_for_gee.csv",
    OUTPUTS / "03_event_table_draft.csv",
]

MAX_SPATIAL_DISTANCE_KM = 25.0
PRIMARY_MAX_TIME_HOURS = 12.0

REQUIRED_COLUMNS = [
    "datetime_utc",
    "site_name",
    "lat",
    "lon",
    "wind_speed",
    "wind_direction",
]

EARTH_RADIUS_KM = 6371.0088


def normalize_site(value):
    text = str(value).strip().lower()

    if text in {"", "nan", "none", "<na>"}:
        return ""

    text = re.sub(r"[^a-z0-9]+", " ", text)

    removable = {
        "release",
        "releases",
        "stack",
        "stacks",
        "site",
        "facility",
        "controlled",
        "methane",
        "source",
        "test",
        "station",
    }

    tokens = [
        token
        for token in text.split()
        if token not in removable
    ]

    return " ".join(tokens)


def haversine_vectorized(lat1, lon1, lat2, lon2):
    lat1 = np.radians(float(lat1))
    lon1 = np.radians(float(lon1))

    lat2 = np.radians(
        pd.to_numeric(lat2, errors="coerce").to_numpy(dtype=float)
    )
    lon2 = np.radians(
        pd.to_numeric(lon2, errors="coerce").to_numpy(dtype=float)
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(a))
    )


def time_tier(hours):
    if pd.isna(hours):
        return "no_match"

    if hours <= 1:
        return "le_1h"

    if hours <= 3:
        return "gt_1h_le_3h"

    if hours <= 12:
        return "gt_3h_le_12h"

    if hours <= 24:
        return "gt_12h_le_24h"

    return "gt_24h"


if not MASTER_PATH.exists():
    raise SystemExit(f"找不到 master table：{MASTER_PATH}")


# ============================================================
# 1. 讀取並清理風場資料
# ============================================================

wind_parts = []
table_reports = []

for priority, path in enumerate(WIND_TABLES):
    if not path.exists():
        print("Missing:", path)
        continue

    header = pd.read_csv(path, nrows=0)
    available = [
        column
        for column in REQUIRED_COLUMNS
        if column in header.columns
    ]

    missing = sorted(
        set(REQUIRED_COLUMNS) - set(available)
    )

    if missing:
        print(
            f"Skip {path.name}: missing columns {missing}"
        )
        continue

    df = pd.read_csv(
        path,
        usecols=REQUIRED_COLUMNS,
        low_memory=False,
    )

    df["wind_time_utc"] = pd.to_datetime(
        df["datetime_utc"],
        errors="coerce",
        utc=True,
    )

    df["latitude"] = pd.to_numeric(
        df["lat"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["lon"],
        errors="coerce",
    )

    df["wind_speed_m_s_as_stored"] = pd.to_numeric(
        df["wind_speed"],
        errors="coerce",
    )

    df["wind_direction_from_deg"] = pd.to_numeric(
        df["wind_direction"],
        errors="coerce",
    )

    valid = (
        df["wind_time_utc"].notna()
        & df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
        & df["wind_speed_m_s_as_stored"].ge(0)
        & df["wind_direction_from_deg"].ge(0)
        & df["wind_direction_from_deg"].lt(360)
    )

    clean = df.loc[valid].copy()

    clean["site_id_wind"] = (
        clean["site_name"].astype(str)
    )

    clean["site_normalized"] = (
        clean["site_id_wind"].map(normalize_site)
    )

    clean["wind_source_file"] = str(path)
    clean["wind_source_row"] = clean.index
    clean["source_priority"] = priority

    wind_parts.append(
        clean[
            [
                "wind_time_utc",
                "site_id_wind",
                "site_normalized",
                "latitude",
                "longitude",
                "wind_speed_m_s_as_stored",
                "wind_direction_from_deg",
                "wind_source_file",
                "wind_source_row",
                "source_priority",
            ]
        ]
    )

    speed = clean["wind_speed_m_s_as_stored"]

    table_reports.append(
        {
            "source_file": str(path),
            "input_rows": len(df),
            "valid_time_speed_direction_position_rows": len(clean),
            "minimum_wind_speed_as_stored": (
                speed.min() if len(speed) else np.nan
            ),
            "median_wind_speed_as_stored": (
                speed.median() if len(speed) else np.nan
            ),
            "maximum_wind_speed_as_stored": (
                speed.max() if len(speed) else np.nan
            ),
            "unique_sites": clean["site_normalized"].nunique(),
        }
    )


if not wind_parts:
    raise SystemExit(
        "03、05、06 都沒有可用的數值風場紀錄。"
    )


wind = pd.concat(
    wind_parts,
    ignore_index=True,
    sort=False,
)

wind["latitude_round"] = wind["latitude"].round(5)
wind["longitude_round"] = wind["longitude"].round(5)
wind["speed_round"] = (
    wind["wind_speed_m_s_as_stored"].round(4)
)
wind["direction_round"] = (
    wind["wind_direction_from_deg"].round(3)
)

wind = (
    wind.sort_values("source_priority")
    .drop_duplicates(
        subset=[
            "wind_time_utc",
            "site_normalized",
            "latitude_round",
            "longitude_round",
            "speed_round",
            "direction_round",
        ],
        keep="first",
    )
    .reset_index(drop=True)
)

wind.drop(
    columns=[
        "latitude_round",
        "longitude_round",
        "speed_round",
        "direction_round",
    ],
    inplace=True,
)

table_report = pd.DataFrame(table_reports)

table_report.to_csv(
    OUTPUTS / "48_wind_source_table_quality.csv",
    index=False,
)

wind.to_csv(
    OUTPUTS / "48_canonical_multisource_wind_inventory.csv",
    index=False,
)


# ============================================================
# 2. 讀取 Sentinel-2 master table
# ============================================================

master = pd.read_csv(
    MASTER_PATH,
    low_memory=False,
)

master["acquisition_time_utc"] = pd.to_datetime(
    master["acquisition_time_utc"],
    errors="coerce",
    utc=True,
)

master["site_normalized"] = (
    master["site_id"].map(normalize_site)
)

for column in [
    "source_latitude",
    "source_longitude",
]:
    master[column] = pd.to_numeric(
        master[column],
        errors="coerce",
    )


# ============================================================
# 3. 每一筆 S2 尋找最近且有空間關聯的 wind record
# ============================================================

matches = []

for _, sample in master.iterrows():
    result = {
        "sample_id": sample.get("sample_id", ""),
        "site_id": sample.get("site_id", ""),
        "scene_id": sample.get("scene_id", ""),
        "label": sample.get("label"),
        "acquisition_time_utc": (
            sample.get("acquisition_time_utc")
        ),
        "wind_match_found": False,
        "wind_match_found_24h": False,
        "wind_match_method": "",
        "wind_match_tier": "no_match",
        "wind_time_utc": pd.NaT,
        "wind_time_difference_hours": np.nan,
        "wind_distance_km": np.nan,
        "wind_speed_m_s": np.nan,
        "wind_speed_unit_status": "as_stored_unverified",
        "wind_direction_from_deg": np.nan,
        "wind_derivation": "direct_speed_direction",
        "wind_source_file": "",
        "wind_source_row": np.nan,
    }

    acquisition_time = sample["acquisition_time_utc"]

    if pd.isna(acquisition_time):
        matches.append(result)
        continue

    site_candidates = wind[
        wind["site_normalized"].ne("")
        & wind["site_normalized"].eq(
            sample["site_normalized"]
        )
    ].copy()

    if not site_candidates.empty:
        candidates = site_candidates
        method = "normalized_site"

        candidates["distance_km"] = haversine_vectorized(
            sample["source_latitude"],
            sample["source_longitude"],
            candidates["latitude"],
            candidates["longitude"],
        )

    elif (
        pd.notna(sample["source_latitude"])
        and pd.notna(sample["source_longitude"])
    ):
        candidates = wind.copy()

        candidates["distance_km"] = haversine_vectorized(
            sample["source_latitude"],
            sample["source_longitude"],
            candidates["latitude"],
            candidates["longitude"],
        )

        candidates = candidates[
            candidates["distance_km"]
            <= MAX_SPATIAL_DISTANCE_KM
        ].copy()

        method = "coordinate_le_25km"

    else:
        candidates = wind.iloc[0:0].copy()
        method = ""

    if candidates.empty:
        matches.append(result)
        continue

    candidates["time_difference_hours"] = (
        candidates["wind_time_utc"]
        - acquisition_time
    ).abs().dt.total_seconds() / 3600.0

    candidates = candidates.sort_values(
        [
            "time_difference_hours",
            "distance_km",
            "source_priority",
        ],
        ascending=True,
    )

    selected = candidates.iloc[0]
    hours = float(selected["time_difference_hours"])

    result.update(
        {
            "wind_match_found": (
                hours <= PRIMARY_MAX_TIME_HOURS
            ),
            "wind_match_found_24h": hours <= 24,
            "wind_match_method": method,
            "wind_match_tier": time_tier(hours),
            "wind_time_utc": selected["wind_time_utc"],
            "wind_time_difference_hours": hours,
            "wind_distance_km": selected["distance_km"],
            "wind_speed_m_s": selected[
                "wind_speed_m_s_as_stored"
            ],
            "wind_direction_from_deg": selected[
                "wind_direction_from_deg"
            ],
            "wind_source_file": selected[
                "wind_source_file"
            ],
            "wind_source_row": selected[
                "wind_source_row"
            ],
        }
    )

    matches.append(result)


matched = pd.DataFrame(matches)

matched.to_csv(
    OUTPUTS / "49_multisite_wind_matches_v2.csv",
    index=False,
)

# 同時寫到 wind-feature 程式預期的位置
matched.to_csv(
    OUTPUTS / "46_multisite_wind_matches.csv",
    index=False,
)


# ============================================================
# 4. 摘要
# ============================================================

summary = (
    matched.groupby("site_id")
    .agg(
        rows=("sample_id", "size"),
        wind_matches_le_12h=(
            "wind_match_found",
            "sum",
        ),
        wind_matches_le_24h=(
            "wind_match_found_24h",
            "sum",
        ),
        median_time_difference_hours=(
            "wind_time_difference_hours",
            "median",
        ),
        median_distance_km=(
            "wind_distance_km",
            "median",
        ),
        median_wind_speed_as_stored=(
            "wind_speed_m_s",
            "median",
        ),
    )
    .reset_index()
)

summary.to_csv(
    OUTPUTS / "49_multisite_wind_match_summary_v2.csv",
    index=False,
)

tier_summary = (
    matched.groupby(
        ["site_id", "wind_match_tier"],
        dropna=False,
    )
    .size()
    .reset_index(name="rows")
)

tier_summary.to_csv(
    OUTPUTS / "49_multisite_wind_match_tiers_v2.csv",
    index=False,
)


print("=" * 100)
print("SOURCE TABLE QUALITY")
print("=" * 100)
print(table_report.to_string(index=False))

print("\n" + "=" * 100)
print("CANONICAL WIND INVENTORY")
print("=" * 100)
print("Rows:", len(wind))
print(
    "Sites:",
    wind["site_normalized"].nunique(),
)

print("\n" + "=" * 100)
print("WIND MATCH SUMMARY")
print("=" * 100)
print(summary.to_string(index=False))

print("\nMATCH TIERS")
print(tier_summary.to_string(index=False))

print("\nPrimary matches <=12 h:")
print(
    int(matched["wind_match_found"].sum()),
    "/",
    len(matched),
)

print("\nSensitivity matches <=24 h:")
print(
    int(matched["wind_match_found_24h"].sum()),
    "/",
    len(matched),
)

print("\nCreated:")
print(
    OUTPUTS / "48_wind_source_table_quality.csv"
)
print(
    OUTPUTS / "48_canonical_multisource_wind_inventory.csv"
)
print(
    OUTPUTS / "49_multisite_wind_matches_v2.csv"
)
print(
    OUTPUTS / "49_multisite_wind_match_summary_v2.csv"
)
print(
    OUTPUTS / "49_multisite_wind_match_tiers_v2.csv"
)
