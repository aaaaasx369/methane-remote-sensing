from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/happydoraaa/methane_release_project")
OUT = ROOT / "outputs"

MASTER_PATH = OUT / "36_multisite_s2_master_table.csv"
AVAILABILITY_PATH = OUT / "37_multisite_s2_availability.csv"
WIND_MATCH_PATH = OUT / "49_multisite_wind_matches_v2.csv"
WIND_FEATURE_PATH = OUT / "47_multisite_s2_features_with_wind.csv"

for path in [
    MASTER_PATH,
    WIND_MATCH_PATH,
    WIND_FEATURE_PATH,
]:
    if not path.exists():
        raise SystemExit(f"找不到：{path}")


def to_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


master = pd.read_csv(
    MASTER_PATH,
    low_memory=False,
)

wind = pd.read_csv(
    WIND_MATCH_PATH,
    low_memory=False,
)

features = pd.read_csv(
    WIND_FEATURE_PATH,
    low_memory=False,
)

# ------------------------------------------------------------
# 1. Boolean flags
# ------------------------------------------------------------

wind["wind_match_found"] = to_bool(
    wind["wind_match_found"]
)

if "wind_match_found_24h" in wind.columns:
    wind["wind_match_found_24h"] = to_bool(
        wind["wind_match_found_24h"]
    )

if "wind_feature_success" in features.columns:
    features["wind_feature_success"] = to_bool(
        features["wind_feature_success"]
    )
else:
    features["wind_feature_success"] = False

# ------------------------------------------------------------
# 2. 決定 merge keys
# ------------------------------------------------------------

preferred_keys = [
    "sample_id",
    "site_id",
    "scene_id",
]

join_keys = [
    column
    for column in preferred_keys
    if (
        column in master.columns
        and column in wind.columns
        and column in features.columns
    )
]

if not join_keys:
    raise SystemExit(
        "Master、wind、feature tables 沒有共同識別欄位。"
    )

print("Join keys:", join_keys)

# ------------------------------------------------------------
# 3. 從 master 取得 metadata
# ------------------------------------------------------------

master_metadata = [
    column
    for column in [
        *join_keys,
        "label",
        "ground_truth_source",
        "acquisition_time_utc",
        "source_latitude",
        "source_longitude",
        "acquisition_inside_release_interval",
        "inside_release_interval",
        "release_start_utc",
        "release_end_utc",
        "emission_rate_kg_hr",
        "image_path",
        "patch_path",
        "resolved_patch_path",
    ]
    if column in master.columns
]

master_small = (
    master[master_metadata]
    .drop_duplicates(
        subset=join_keys,
        keep="first",
    )
)

# ------------------------------------------------------------
# 4. 取得 wind features
# ------------------------------------------------------------

important_wind_feature_columns = [
    column
    for column in [
        *join_keys,
        "wind_feature_success",
        "wind_feature_error",
        "wind_b11_downwind_minus_upwind",
        "wind_b12_downwind_minus_upwind",
        "wind_ndvi_downwind_minus_upwind",
        "wind_swir_ratio_downwind_minus_upwind",
        "wind_swir_nd_downwind_minus_upwind",
    ]
    if column in features.columns
]

feature_small = (
    features[important_wind_feature_columns]
    .drop_duplicates(
        subset=join_keys,
        keep="first",
    )
)

# ------------------------------------------------------------
# 5. 合併
# ------------------------------------------------------------

combined = wind.merge(
    master_small,
    on=join_keys,
    how="left",
    suffixes=("", "_master"),
    validate="one_to_one",
)

combined = combined.merge(
    feature_small,
    on=join_keys,
    how="left",
    validate="one_to_one",
)

if "wind_feature_success" not in combined.columns:
    combined["wind_feature_success"] = False

combined["wind_feature_success"] = to_bool(
    combined["wind_feature_success"]
)

combined["valid_wind_row"] = (
    combined["wind_match_found"]
    & combined["wind_feature_success"]
)

valid = combined[
    combined["valid_wind_row"]
].copy()

# ------------------------------------------------------------
# 6. Release interval flag
# ------------------------------------------------------------

interval_column = next(
    (
        column
        for column in [
            "acquisition_inside_release_interval",
            "inside_release_interval",
        ]
        if column in valid.columns
    ),
    None,
)

if interval_column is not None:
    valid["inside_release_interval_verified"] = to_bool(
        valid[interval_column]
    )
else:
    valid["inside_release_interval_verified"] = False

# ------------------------------------------------------------
# 7. Correct matched-only summary
# ------------------------------------------------------------

coverage = (
    combined.groupby("site_id")
    .agg(
        total_rows=("site_id", "size"),
        wind_matches_le_12h=(
            "wind_match_found",
            "sum",
        ),
        wind_feature_success=(
            "wind_feature_success",
            "sum",
        ),
    )
    .reset_index()
)

matched_time = (
    valid.groupby("site_id")
    .agg(
        matched_rows=("site_id", "size"),
        median_matched_time_difference_hours=(
            "wind_time_difference_hours",
            "median",
        ),
        maximum_matched_time_difference_hours=(
            "wind_time_difference_hours",
            "max",
        ),
        median_wind_speed_as_stored=(
            "wind_speed_m_s",
            "median",
        ),
        minimum_wind_speed_as_stored=(
            "wind_speed_m_s",
            "min",
        ),
        maximum_wind_speed_as_stored=(
            "wind_speed_m_s",
            "max",
        ),
    )
    .reset_index()
)

coverage = coverage.merge(
    matched_time,
    on="site_id",
    how="left",
)

# ------------------------------------------------------------
# 8. Label balance
# ------------------------------------------------------------

if "label" in valid.columns:
    valid["label"] = pd.to_numeric(
        valid["label"],
        errors="coerce",
    )

    label_balance = (
        valid.groupby(
            ["site_id", "label"],
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
    )
else:
    label_balance = pd.DataFrame()

# ------------------------------------------------------------
# 9. Time tiers
# ------------------------------------------------------------

def matched_tier(hours):
    if pd.isna(hours):
        return "missing"

    if hours <= 1:
        return "le_1h"

    if hours <= 3:
        return "gt_1h_le_3h"

    if hours <= 12:
        return "gt_3h_le_12h"

    return "invalid_gt_12h"


valid["matched_time_tier"] = (
    valid["wind_time_difference_hours"]
    .map(matched_tier)
)

tier_summary = (
    valid.groupby(
        ["site_id", "matched_time_tier"]
    )
    .size()
    .reset_index(name="rows")
)

# ------------------------------------------------------------
# 10. 輸出
# ------------------------------------------------------------

valid.to_csv(
    OUT / "50_valid_wind_feature_subset.csv",
    index=False,
)

coverage.to_csv(
    OUT / "50_wind_coverage_corrected.csv",
    index=False,
)

label_balance.to_csv(
    OUT / "50_wind_label_balance.csv",
    index=False,
)

tier_summary.to_csv(
    OUT / "50_wind_matched_time_tiers.csv",
    index=False,
)

display_columns = [
    column
    for column in [
        "sample_id",
        "site_id",
        "scene_id",
        "label",
        "acquisition_time_utc",
        "wind_time_utc",
        "wind_time_difference_hours",
        "matched_time_tier",
        "wind_speed_m_s",
        "wind_direction_from_deg",
        "inside_release_interval_verified",
        "wind_b11_downwind_minus_upwind",
        "wind_b12_downwind_minus_upwind",
        "wind_swir_ratio_downwind_minus_upwind",
    ]
    if column in valid.columns
]

report_lines = [
    "CORRECTED WIND COVERAGE",
    "=" * 100,
    f"All multisite rows: {len(combined)}",
    f"Valid wind-feature rows: {len(valid)}",
    "",
    "MATCHED-ONLY COVERAGE",
    coverage.to_string(index=False),
    "",
    "MATCHED TIME TIERS",
    tier_summary.to_string(index=False),
    "",
    "LABEL BALANCE",
    (
        label_balance.to_string(index=False)
        if not label_balance.empty
        else "Label column unavailable."
    ),
    "",
    "VALID WIND ROWS",
    valid[display_columns].to_string(index=False),
    "",
    "Important:",
    "The matched-only median excludes nearest wind records that were more than 12 hours away.",
    "Wind speed units remain as stored and should be verified from the source documentation.",
]

report = "\n".join(report_lines)

(
    OUT / "50_valid_wind_feature_report.txt"
).write_text(
    report,
    encoding="utf-8",
)

print(report)

print("\nCreated:")
print(OUT / "50_valid_wind_feature_subset.csv")
print(OUT / "50_wind_coverage_corrected.csv")
print(OUT / "50_wind_label_balance.csv")
print(OUT / "50_wind_matched_time_tiers.csv")
print(OUT / "50_valid_wind_feature_report.txt")
