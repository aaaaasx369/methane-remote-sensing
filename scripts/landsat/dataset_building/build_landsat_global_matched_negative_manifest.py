from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


POSITIVE_INPUT = Path(
    "outputs/396_landsat_final_confirmed_features_site_repaired_v1.csv"
)

NEGATIVE_POOL_INPUT = Path(
    "outputs/411_landsat_combined_clean_24h_candidates_v1.csv"
)

PAIR_OUTPUT = Path(
    "outputs/412_landsat_global_matched_negative_pairs_v1.csv"
)

BENCHMARK_OUTPUT = Path(
    "outputs/413_landsat_matched_benchmark_manifest_v1.csv"
)

RESERVE_OUTPUT = Path(
    "outputs/414_landsat_unselected_clean24h_reserve_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/415_landsat_global_matching_report_v1.txt"
)


INVALID_COST = 1_000_000_000.0

# 同平台是偏好，不是硬性條件。
SENSOR_MISMATCH_PENALTY = 7.0

# 非主要 WRS path/row 的候選會被降低優先度。
NON_REFERENCE_WRS_PENALTY = 5.0

# Scene-level 雲量只作為較小的排序因素。
CLOUD_COST_WEIGHT = 0.02


def find_column(frame, candidates, table_name, required=True):
    for column in candidates:
        if column in frame.columns:
            return column

    if required:
        raise KeyError(
            f"{table_name} 找不到欄位："
            + ", ".join(candidates)
        )

    return None


def normalize_site(value):
    text = str(value).strip().lower()

    if "casa" in text:
        return "casa_grande"

    if "ehrenberg" in text:
        return "ehrenberg"

    if text in {"", "nan", "none", "<na>"}:
        return pd.NA

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    ).strip("_")


def parse_bool(value):
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def load_positives():
    frame = pd.read_csv(
        POSITIVE_INPUT,
        low_memory=False,
    )

    frame["label"] = pd.to_numeric(
        frame["label"],
        errors="raise",
    ).astype(int)

    frame = frame[
        frame["label"].eq(1)
    ].copy()

    id_column = find_column(
        frame,
        [
            "raster_group_id",
            "sample_id",
            "pixel_hash",
            "LANDSAT_PRODUCT_ID",
            "scene_id",
        ],
        "Positive table",
    )

    time_column = find_column(
        frame,
        [
            "landsat_image_time",
            "acquisition_time_utc",
            "image_time_utc",
            "scene_time_utc",
        ],
        "Positive table",
    )

    site_column = find_column(
        frame,
        [
            "site",
            "site_name",
            "site_name_normalized",
            "site_key",
        ],
        "Positive table",
    )

    sensor_column = find_column(
        frame,
        [
            "landsat_sensor",
            "sensor",
            "platform",
        ],
        "Positive table",
    )

    scene_column = find_column(
        frame,
        [
            "LANDSAT_PRODUCT_ID",
            "landsat_product_id",
            "scene_id",
            "landsat_scene_id",
            "system:index",
        ],
        "Positive table",
        required=False,
    )

    rate_column = find_column(
        frame,
        [
            "release_rate_kg_h",
            "final_release_rate_kg_h",
            "preferred_release_rate_kg_h",
            "matched_release_rate_kg_h",
            "cr_kgh_CH4_mean300",
        ],
        "Positive table",
        required=False,
    )

    result = pd.DataFrame({
        "positive_id":
            frame[id_column].astype(str),

        "positive_time_utc":
            pd.to_datetime(
                frame[time_column],
                errors="coerce",
                utc=True,
            ),

        "positive_site":
            frame[site_column].astype(str),

        "positive_site_alias":
            frame[site_column].map(normalize_site),

        "positive_sensor":
            frame[sensor_column].astype(str),
    })

    if scene_column is None:
        result["positive_scene_id"] = result["positive_id"]
    else:
        result["positive_scene_id"] = (
            frame[scene_column].astype(str)
        )

    if rate_column is None:
        result["positive_release_rate_kg_h"] = np.nan
    else:
        result["positive_release_rate_kg_h"] = pd.to_numeric(
            frame[rate_column],
            errors="coerce",
        )

    if result["positive_time_utc"].isna().any():
        raise RuntimeError(
            "部分 positive acquisition time 無法解析。"
        )

    if result["positive_site_alias"].isna().any():
        raise RuntimeError(
            "部分 positive site 無法解析。"
        )

    if len(result) != 7:
        raise RuntimeError(
            f"預期 7 張 positives，實際為 {len(result)}。"
        )

    if result["positive_id"].duplicated().any():
        raise RuntimeError(
            "Positive ID 有重複。"
        )

    return result.reset_index(drop=True)


def load_negative_pool():
    frame = pd.read_csv(
        NEGATIVE_POOL_INPUT,
        low_memory=False,
    )

    time_column = find_column(
        frame,
        [
            "candidate_acquisition_time_utc",
            "candidate_time_parsed_utc",
            "candidate_time_utc",
            "acquisition_time_utc",
        ],
        "Negative pool",
    )

    site_column = find_column(
        frame,
        [
            "candidate_site_alias",
            "site_alias",
            "site_key",
            "site_name_normalized",
            "site",
        ],
        "Negative pool",
    )

    scene_column = find_column(
        frame,
        [
            "LANDSAT_PRODUCT_ID",
            "candidate_scene_id_standard",
            "candidate_scene_id",
            "LANDSAT_SCENE_ID",
            "system:index",
        ],
        "Negative pool",
    )

    sensor_column = find_column(
        frame,
        [
            "landsat_sensor",
            "candidate_sensor",
            "SPACECRAFT_ID",
            "platform",
        ],
        "Negative pool",
    )

    overpass_column = find_column(
        frame,
        [
            "independent_overpass_key",
            "independent_candidate_id",
            "expanded_candidate_id",
        ],
        "Negative pool",
        required=False,
    )

    cloud_column = find_column(
        frame,
        [
            "cloud_cover_numeric",
            "CLOUD_COVER",
            "candidate_cloud",
            "CLOUD_COVER_LAND",
        ],
        "Negative pool",
        required=False,
    )

    wrs_column = find_column(
        frame,
        [
            "same_reference_wrs_bool",
            "same_reference_wrs",
        ],
        "Negative pool",
        required=False,
    )

    result = frame.copy()

    result["negative_time_utc"] = pd.to_datetime(
        result[time_column],
        errors="coerce",
        utc=True,
    )

    result["negative_site_alias"] = (
        result[site_column].map(normalize_site)
    )

    result["negative_scene_id"] = (
        result[scene_column].astype(str)
    )

    result["negative_sensor"] = (
        result[sensor_column].astype(str)
    )

    if overpass_column is None:
        result["negative_overpass_key"] = (
            result["negative_site_alias"].astype(str)
            + "|"
            + result["negative_sensor"].astype(str)
            + "|"
            + result["negative_time_utc"].dt.strftime("%Y-%m-%d")
        )
    else:
        result["negative_overpass_key"] = (
            result[overpass_column].astype(str)
        )

    if cloud_column is None:
        result["negative_cloud_cover"] = np.nan
    else:
        result["negative_cloud_cover"] = pd.to_numeric(
            result[cloud_column],
            errors="coerce",
        )

    if wrs_column is None:
        result["same_reference_wrs_standard"] = True
    else:
        result["same_reference_wrs_standard"] = (
            result[wrs_column].map(parse_bool)
        )

    result = result.dropna(
        subset=[
            "negative_time_utc",
            "negative_site_alias",
            "negative_scene_id",
            "negative_overpass_key",
        ]
    ).copy()

    result = result.drop_duplicates(
        subset=["negative_overpass_key"],
        keep="first",
    ).reset_index(drop=True)

    if len(result) != 39:
        print(
            "Warning: expected 39 clean-24h candidates, "
            f"found {len(result)}."
        )

    return result


def make_slots(positives):
    rows = []

    for _, positive in positives.iterrows():
        for temporal_side in ["before", "after"]:
            for side_slot in [1, 2]:
                rows.append({
                    **positive.to_dict(),

                    "temporal_side":
                        temporal_side,

                    "side_slot":
                        side_slot,

                    "pair_slot":
                        (
                            f"{temporal_side}_"
                            f"{side_slot}"
                        ),
                })

    return pd.DataFrame(rows)


def candidate_cost(slot, candidate, candidate_index):
    if (
        slot["positive_site_alias"]
        != candidate["negative_site_alias"]
    ):
        return INVALID_COST

    time_difference_days = (
        candidate["negative_time_utc"]
        - slot["positive_time_utc"]
    ).total_seconds() / 86400.0

    if (
        slot["temporal_side"] == "before"
        and time_difference_days >= 0
    ):
        return INVALID_COST

    if (
        slot["temporal_side"] == "after"
        and time_difference_days <= 0
    ):
        return INVALID_COST

    absolute_days = abs(time_difference_days)

    sensor_penalty = (
        0.0
        if (
            str(slot["positive_sensor"])
            == str(candidate["negative_sensor"])
        )
        else SENSOR_MISMATCH_PENALTY
    )

    wrs_penalty = (
        0.0
        if candidate["same_reference_wrs_standard"]
        else NON_REFERENCE_WRS_PENALTY
    )

    cloud = candidate["negative_cloud_cover"]

    cloud_cost = (
        0.0
        if pd.isna(cloud)
        else float(cloud) * CLOUD_COST_WEIGHT
    )

    # 最後的小數項只用來讓結果固定、避免完全同分。
    tie_breaker = candidate_index * 0.000001

    return (
        absolute_days
        + sensor_penalty
        + wrs_penalty
        + cloud_cost
        + tie_breaker
    )


def main():
    positives = load_positives()
    negatives = load_negative_pool()
    slots = make_slots(positives)

    cost_matrix = np.full(
        shape=(len(slots), len(negatives)),
        fill_value=INVALID_COST,
        dtype=float,
    )

    for slot_index, slot in slots.iterrows():
        for candidate_index, candidate in negatives.iterrows():
            cost_matrix[
                slot_index,
                candidate_index,
            ] = candidate_cost(
                slot,
                candidate,
                candidate_index,
            )

    row_indices, column_indices = (
        linear_sum_assignment(cost_matrix)
    )

    if len(row_indices) != len(slots):
        raise RuntimeError(
            "無法為所有 28 個 slots 完成配對。"
        )

    assigned_costs = cost_matrix[
        row_indices,
        column_indices,
    ]

    invalid_assignments = (
        assigned_costs >= INVALID_COST / 2
    )

    if invalid_assignments.any():
        bad_slots = slots.iloc[
            row_indices[invalid_assignments]
        ]

        raise RuntimeError(
            "全域唯一 2-before/2-after 配對不可行：\n"
            + bad_slots.to_string(index=False)
        )

    selected_rows = []

    for slot_index, candidate_index, cost in zip(
        row_indices,
        column_indices,
        assigned_costs,
    ):
        slot = slots.iloc[slot_index]
        candidate = negatives.iloc[candidate_index]

        days_from_positive = (
            candidate["negative_time_utc"]
            - slot["positive_time_utc"]
        ).total_seconds() / 86400.0

        selected_rows.append({
            "matched_positive_id":
                slot["positive_id"],

            "matched_positive_scene_id":
                slot["positive_scene_id"],

            "positive_site":
                slot["positive_site"],

            "positive_site_alias":
                slot["positive_site_alias"],

            "positive_time_utc":
                slot["positive_time_utc"],

            "positive_sensor":
                slot["positive_sensor"],

            "positive_release_rate_kg_h":
                slot["positive_release_rate_kg_h"],

            "pair_slot":
                slot["pair_slot"],

            "temporal_side":
                slot["temporal_side"],

            "negative_scene_id":
                candidate["negative_scene_id"],

            "negative_overpass_key":
                candidate["negative_overpass_key"],

            "negative_time_utc":
                candidate["negative_time_utc"],

            "negative_sensor":
                candidate["negative_sensor"],

            "negative_cloud_cover":
                candidate["negative_cloud_cover"],

            "same_reference_wrs":
                candidate["same_reference_wrs_standard"],

            "same_sensor_as_positive":
                (
                    str(slot["positive_sensor"])
                    == str(candidate["negative_sensor"])
                ),

            "days_from_positive":
                days_from_positive,

            "absolute_days_from_positive":
                abs(days_from_positive),

            "matching_cost":
                float(cost),

            "release_negative_rule":
                "no_exact_overlap_and_more_than_24h",
        })

    pairs = pd.DataFrame(selected_rows).sort_values(
        [
            "positive_site_alias",
            "positive_time_utc",
            "temporal_side",
            "negative_time_utc",
        ]
    ).reset_index(drop=True)

    if len(pairs) != 28:
        raise RuntimeError(
            f"預期 28 pairs，實際為 {len(pairs)}。"
        )

    if pairs["negative_overpass_key"].duplicated().any():
        raise RuntimeError(
            "同一個 negative overpass 被重複使用。"
        )

    side_check = pd.crosstab(
        pairs["matched_positive_id"],
        pairs["temporal_side"],
    )

    for side in ["before", "after"]:
        if side not in side_check.columns:
            raise RuntimeError(
                f"缺少 temporal side：{side}"
            )

        if not side_check[side].eq(2).all():
            raise RuntimeError(
                f"並非每張 positive 都有 2 個 {side}。"
            )

    pairs.to_csv(
        PAIR_OUTPUT,
        index=False,
    )

    positive_manifest_rows = []

    for _, positive in positives.iterrows():
        positive_manifest_rows.append({
            "sample_id":
                positive["positive_id"],

            "label":
                1,

            "sample_role":
                "confirmed_positive",

            "site":
                positive["positive_site"],

            "site_alias":
                positive["positive_site_alias"],

            "acquisition_time_utc":
                positive["positive_time_utc"],

            "landsat_sensor":
                positive["positive_sensor"],

            "scene_id":
                positive["positive_scene_id"],

            "matched_positive_id":
                positive["positive_id"],

            "pair_slot":
                "positive",

            "temporal_side":
                "positive",

            "days_from_positive":
                0.0,

            "release_rate_kg_h":
                positive["positive_release_rate_kg_h"],

            "negative_release_rule":
                pd.NA,
        })

    negative_manifest_rows = []

    for number, row in pairs.iterrows():
        negative_manifest_rows.append({
            "sample_id":
                f"LANDSAT_MATCHED_NEG_{number + 1:03d}",

            "label":
                0,

            "sample_role":
                "matched_negative_clean_24h",

            "site":
                row["positive_site"],

            "site_alias":
                row["positive_site_alias"],

            "acquisition_time_utc":
                row["negative_time_utc"],

            "landsat_sensor":
                row["negative_sensor"],

            "scene_id":
                row["negative_scene_id"],

            "matched_positive_id":
                row["matched_positive_id"],

            "pair_slot":
                row["pair_slot"],

            "temporal_side":
                row["temporal_side"],

            "days_from_positive":
                row["days_from_positive"],

            "release_rate_kg_h":
                0.0,

            "negative_release_rule":
                row["release_negative_rule"],
        })

    benchmark = pd.DataFrame(
        positive_manifest_rows
        + negative_manifest_rows
    )

    benchmark[
        "benchmark_name"
    ] = "landsat_matched_controlled_release_v1"

    benchmark[
        "benchmark_design"
    ] = "2_before_2_after_global_unique_clean_24h"

    benchmark.to_csv(
        BENCHMARK_OUTPUT,
        index=False,
    )

    used_keys = set(
        pairs["negative_overpass_key"]
    )

    reserve = negatives[
        ~negatives[
            "negative_overpass_key"
        ].isin(used_keys)
    ].copy()

    reserve.to_csv(
        RESERVE_OUTPUT,
        index=False,
    )

    site_summary = pd.crosstab(
        benchmark["site_alias"],
        benchmark["label"],
        margins=True,
    )

    sensor_summary = pd.crosstab(
        benchmark["landsat_sensor"],
        benchmark["label"],
        margins=True,
    )

    side_summary = pd.crosstab(
        pairs["matched_positive_id"],
        pairs["temporal_side"],
        margins=True,
    )

    report_lines = [
        "=" * 110,
        "LANDSAT GLOBAL MATCHED-NEGATIVE REPORT V1",
        "=" * 110,
        "",
        f"Confirmed positives: {len(positives)}",
        f"Clean-24h candidate pool: {len(negatives)}",
        f"Selected unique negatives: {len(pairs)}",
        f"Unselected clean-24h reserve: {len(reserve)}",
        f"Final benchmark rows: {len(benchmark)}",
        "",
        "Benchmark label counts:",
        benchmark["label"].value_counts().sort_index().to_string(),
        "",
        "Label by site:",
        site_summary.to_string(),
        "",
        "Label by Landsat sensor:",
        sensor_summary.to_string(),
        "",
        "Before/after allocation:",
        side_summary.to_string(),
        "",
        (
            "Same-sensor negative pairs: "
            f"{int(pairs['same_sensor_as_positive'].sum())} / {len(pairs)}"
        ),
        (
            "Median absolute temporal distance: "
            f"{pairs['absolute_days_from_positive'].median():.2f} days"
        ),
        (
            "Maximum absolute temporal distance: "
            f"{pairs['absolute_days_from_positive'].max():.2f} days"
        ),
        "",
        "Validation:",
        (
            "All negatives are globally unique: "
            f"{not pairs['negative_overpass_key'].duplicated().any()}"
        ),
        (
            "Every positive has exactly 2 before and 2 after: "
            f"{bool(side_check[['before', 'after']].eq(2).all().all())}"
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 110)
    print("LANDSAT GLOBAL MATCHING COMPLETE")
    print("=" * 110)

    print("\nConfirmed positives:", len(positives))
    print("Clean-24h candidate pool:", len(negatives))
    print("Selected unique negatives:", len(pairs))
    print("Reserve candidates:", len(reserve))
    print("Final benchmark rows:", len(benchmark))

    print("\nLabel counts:")
    print(
        benchmark["label"]
        .value_counts()
        .sort_index()
    )

    print("\nBefore/after allocation:")
    print(side_check)

    print(
        "\nDuplicated selected negative overpasses:",
        int(
            pairs[
                "negative_overpass_key"
            ].duplicated().sum()
        ),
    )

    print(
        "Same-sensor pairs:",
        int(
            pairs[
                "same_sensor_as_positive"
            ].sum()
        ),
        "/",
        len(pairs),
    )

    print(
        "Median absolute date gap:",
        round(
            pairs[
                "absolute_days_from_positive"
            ].median(),
            2,
        ),
        "days",
    )

    print("\nSaved:")
    print(PAIR_OUTPUT)
    print(BENCHMARK_OUTPUT)
    print(RESERVE_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
