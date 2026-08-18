from __future__ import annotations

import os
from pathlib import Path

import ee
import numpy as np
import pandas as pd


PROJECT = os.environ.get(
    "EE_PROJECT",
    "methane-release-gee",
)

POSITIVE_INDEX = Path(
    "outputs/141_evanston_confirmed_positive_patch_index.csv"
)

ALL_RELEASES = Path(
    "outputs/125_stanford_all_release_summaries.csv"
)

CANDIDATE_OUTPUT = Path(
    "outputs/142_evanston_negative_scene_candidates.csv"
)

SPLIT_OUTPUT = Path(
    "outputs/143_evanston_negative_split_manifest.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/144_evanston_negative_selection_summary.csv"
)


COLLECTIONS = {
    "Landsat-8": "LANDSAT/LC08/C02/T1_L2",
    "Landsat-9": "LANDSAT/LC09/C02/T1_L2",
}

CLOUD_LIMIT = 35.0
SEARCH_MARGIN_DAYS = 60
MAX_DAYS_TO_POSITIVE = 45
MIN_SELECTED_DATE_GAP_DAYS = 5

# 先多下載一些，經過排放源附近 QA 後，
# 最終至少保留 5 calibration + 5 test。
TARGET_TOTAL = 14


def parse_time_milliseconds(value):
    if value is None:
        return pd.NaT

    return pd.to_datetime(
        value,
        unit="ms",
        errors="coerce",
        utc=True,
    )


def nearest_date_distance_days(
    candidate_date,
    positive_dates,
):
    distances = [
        abs(
            (
                candidate_date
                - positive_date
            ).days
        )
        for positive_date in positive_dates
    ]

    return min(distances)


def search_collection(
    collection_id,
    expected_sensor,
    point,
    start_date,
    end_date,
):
    collection = (
        ee.ImageCollection(
            collection_id
        )
        .filterBounds(point)
        .filterDate(
            start_date,
            end_date,
        )
        .filter(
            ee.Filter.lte(
                "CLOUD_COVER",
                CLOUD_LIMIT,
            )
        )
        .sort(
            "system:time_start"
        )
    )

    info = collection.getInfo()

    records = []

    for feature in info.get(
        "features",
        [],
    ):
        properties = feature.get(
            "properties",
            {},
        )

        acquisition_time = (
            parse_time_milliseconds(
                properties.get(
                    "system:time_start"
                )
            )
        )

        records.append({
            "collection_id":
                collection_id,
            "expected_sensor":
                expected_sensor,
            "system_index":
                properties.get(
                    "system:index",
                    feature.get("id"),
                ),
            "landsat_product_id":
                properties.get(
                    "LANDSAT_PRODUCT_ID"
                ),
            "spacecraft_id":
                properties.get(
                    "SPACECRAFT_ID"
                ),
            "acquisition_time_utc":
                acquisition_time,
            "acquisition_date":
                (
                    acquisition_time.date()
                    if pd.notna(
                        acquisition_time
                    )
                    else pd.NaT
                ),
            "cloud_cover":
                properties.get(
                    "CLOUD_COVER"
                ),
            "wrs_path":
                properties.get(
                    "WRS_PATH"
                ),
            "wrs_row":
                properties.get(
                    "WRS_ROW"
                ),
            "collection_category":
                properties.get(
                    "COLLECTION_CATEGORY"
                ),
        })

    return records


def select_diverse_candidates(
    candidates,
    target_total,
):
    if candidates.empty:
        return candidates.copy()

    ordered = candidates.sort_values(
        [
            "cloud_cover",
            "days_to_nearest_positive",
            "acquisition_time_utc",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    )

    selected_indices = []
    selected_dates = []

    for index, row in ordered.iterrows():
        candidate_date = pd.Timestamp(
            row["acquisition_date"]
        )

        sufficiently_separated = all(
            abs(
                (
                    candidate_date
                    - existing_date
                ).days
            )
            >= MIN_SELECTED_DATE_GAP_DAYS
            for existing_date
            in selected_dates
        )

        if not sufficiently_separated:
            continue

        selected_indices.append(index)
        selected_dates.append(
            candidate_date
        )

        if len(selected_indices) >= target_total:
            break

    # 若日期間隔限制導致數量不足，
    # 再從其餘低雲 scene 補足。
    if len(selected_indices) < target_total:
        for index in ordered.index:
            if index in selected_indices:
                continue

            selected_indices.append(index)

            if len(selected_indices) >= target_total:
                break

    return (
        candidates.loc[
            selected_indices
        ]
        .copy()
        .sort_values(
            "acquisition_time_utc"
        )
        .reset_index(drop=True)
    )


def main():
    if not POSITIVE_INDEX.exists():
        raise FileNotFoundError(
            POSITIVE_INDEX
        )

    if not ALL_RELEASES.exists():
        raise FileNotFoundError(
            ALL_RELEASES
        )

    ee.Initialize(
        project=PROJECT
    )

    print(
        f"[OK] Earth Engine initialized: "
        f"{PROJECT}"
    )

    positives = pd.read_csv(
        POSITIVE_INDEX,
        low_memory=False,
    )

    positives = positives[
        positives["download_status"]
        == "success"
    ].copy()

    if positives.empty:
        raise RuntimeError(
            "No successful positive patches "
            "were found."
        )

    positives[
        "acquisition_time_parsed"
    ] = pd.to_datetime(
        positives[
            "acquisition_time_utc"
        ],
        errors="coerce",
        utc=True,
    )

    positives = positives.dropna(
        subset=[
            "acquisition_time_parsed",
            "latitude",
            "longitude",
            "wrs_path",
            "wrs_row",
        ]
    )

    positive_dates = sorted({
        value.normalize()
        for value
        in positives[
            "acquisition_time_parsed"
        ]
    })

    positive_product_ids = set(
        positives[
            "landsat_product_id"
        ].dropna().astype(str)
    )

    wrs_pairs = {
        (
            int(row["wrs_path"]),
            int(row["wrs_row"]),
        )
        for _, row
        in positives.iterrows()
    }

    latitude = float(
        positives["latitude"].median()
    )

    longitude = float(
        positives["longitude"].median()
    )

    search_start = (
        min(positive_dates)
        - pd.Timedelta(
            days=SEARCH_MARGIN_DAYS
        )
    )

    search_end = (
        max(positive_dates)
        + pd.Timedelta(
            days=SEARCH_MARGIN_DAYS
        )
        + pd.Timedelta(days=1)
    )

    releases = pd.read_csv(
        ALL_RELEASES,
        low_memory=False,
    )

    releases = releases[
        releases["site_key"]
        .astype(str)
        .str.lower()
        .eq("evanston")
    ].copy()

    releases[
        "release_datetime"
    ] = pd.to_datetime(
        releases["datetime_utc"],
        errors="coerce",
        utc=True,
    )

    # 任何存在 Evanston controlled release 的日期，
    # 都不允許當 negative。
    release_dates = {
        timestamp.date()
        for timestamp
        in releases[
            "release_datetime"
        ].dropna()
    }

    point = ee.Geometry.Point([
        longitude,
        latitude,
    ])

    all_records = []

    for sensor, collection_id in (
        COLLECTIONS.items()
    ):
        print(
            f"Searching {sensor}: "
            f"{search_start.date()} "
            f"to {search_end.date()}"
        )

        records = search_collection(
            collection_id=
                collection_id,
            expected_sensor=
                sensor,
            point=
                point,
            start_date=
                search_start.strftime(
                    "%Y-%m-%d"
                ),
            end_date=
                search_end.strftime(
                    "%Y-%m-%d"
                ),
        )

        all_records.extend(records)

    candidates = pd.DataFrame(
        all_records
    )

    if candidates.empty:
        raise RuntimeError(
            "No Landsat candidates found."
        )

    candidates["cloud_cover"] = (
        pd.to_numeric(
            candidates["cloud_cover"],
            errors="coerce",
        )
    )

    candidates["wrs_path"] = (
        pd.to_numeric(
            candidates["wrs_path"],
            errors="coerce",
        )
    )

    candidates["wrs_row"] = (
        pd.to_numeric(
            candidates["wrs_row"],
            errors="coerce",
        )
    )

    candidates[
        "same_positive_wrs"
    ] = candidates.apply(
        lambda row: (
            (
                int(row["wrs_path"]),
                int(row["wrs_row"]),
            )
            in wrs_pairs
            if (
                pd.notna(
                    row["wrs_path"]
                )
                and pd.notna(
                    row["wrs_row"]
                )
            )
            else False
        ),
        axis=1,
    )

    candidates[
        "is_release_date"
    ] = candidates[
        "acquisition_date"
    ].isin(
        release_dates
    )

    candidates[
        "is_positive_product"
    ] = candidates[
        "landsat_product_id"
    ].astype(str).isin(
        positive_product_ids
    )

    candidates[
        "days_to_nearest_positive"
    ] = candidates[
        "acquisition_time_utc"
    ].apply(
        lambda value:
            nearest_date_distance_days(
                pd.Timestamp(
                    value
                ).normalize(),
                positive_dates,
            )
    )

    candidates[
        "eligible_negative"
    ] = (
        candidates[
            "same_positive_wrs"
        ]
        & ~candidates[
            "is_release_date"
        ]
        & ~candidates[
            "is_positive_product"
        ]
        & candidates[
            "landsat_product_id"
        ].notna()
        & (
            candidates[
                "days_to_nearest_positive"
            ]
            <= MAX_DAYS_TO_POSITIVE
        )
    )

    candidates[
        "latitude"
    ] = latitude

    candidates[
        "longitude"
    ] = longitude

    candidates = (
        candidates.sort_values(
            [
                "eligible_negative",
                "cloud_cover",
                "days_to_nearest_positive",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .drop_duplicates(
            subset=[
                "landsat_product_id"
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    candidates.to_csv(
        CANDIDATE_OUTPUT,
        index=False,
    )

    eligible = candidates[
        candidates[
            "eligible_negative"
        ]
    ].copy()

    selected = select_diverse_candidates(
        candidates=eligible,
        target_total=TARGET_TOTAL,
    )

    selected[
        "negative_id"
    ] = [
        f"EV_NEG_{number:03d}"
        for number in range(
            1,
            len(selected) + 1,
        )
    ]

    # 在查看影像與模型結果之前凍結分組。
    # 依時間交錯，讓兩組的季節分布接近。
    selected[
        "negative_role"
    ] = [
        (
            "calibration_negative"
            if index % 2 == 0
            else "test_negative"
        )
        for index in range(
            len(selected)
        )
    ]

    selected["label"] = 0
    selected["site_key"] = "evanston"
    selected[
        "ground_truth_type"
    ] = (
        "schedule_supported_no_release"
    )

    selected.to_csv(
        SPLIT_OUTPUT,
        index=False,
    )

    summary = (
        selected.groupby(
            [
                "negative_role",
                "expected_sensor",
            ],
            dropna=False,
        )
        .agg(
            scene_count=(
                "landsat_product_id",
                "size",
            ),
            first_date=(
                "acquisition_time_utc",
                "min",
            ),
            last_date=(
                "acquisition_time_utc",
                "max",
            ),
            median_cloud_cover=(
                "cloud_cover",
                "median",
            ),
            maximum_cloud_cover=(
                "cloud_cover",
                "max",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 105)
    print("EVANSTON NEGATIVE SELECTION")
    print("=" * 105)

    print(
        "Candidate scenes found:",
        len(candidates),
    )

    print(
        "Eligible no-release scenes:",
        len(eligible),
    )

    print(
        "Selected for download:",
        len(selected),
    )

    print("\nRole counts:")
    print(
        selected[
            "negative_role"
        ].value_counts()
    )

    print("\nRole by sensor:")
    print(
        pd.crosstab(
            selected[
                "negative_role"
            ],
            selected[
                "expected_sensor"
            ],
            margins=True,
        )
    )

    display_columns = [
        "negative_id",
        "negative_role",
        "expected_sensor",
        "landsat_product_id",
        "acquisition_time_utc",
        "cloud_cover",
        "wrs_path",
        "wrs_row",
        "days_to_nearest_positive",
    ]

    print("\nSelected scenes:")
    print(
        selected[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.2f}",
        )
    )

    print("\nSaved:")
    print(CANDIDATE_OUTPUT)
    print(SPLIT_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
