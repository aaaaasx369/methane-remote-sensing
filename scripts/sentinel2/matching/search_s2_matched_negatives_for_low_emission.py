from pathlib import Path
import os
import re

import ee
import numpy as np
import pandas as pd


POSITIVE_INPUT = Path(
    "outputs/318_s2_low_emission_primary_scenes_v1.csv"
)

ALL_SCENE_INPUT = Path(
    "outputs/317_s2_low_emission_scene_manifest_v1.csv"
)

RELEASE_INPUT = Path(
    "outputs/309_all_exact_release_intervals_for_s2.csv"
)

CANDIDATE_OUTPUT = Path(
    "outputs/321_s2_low_emission_negative_candidates.csv"
)

SELECTED_OUTPUT = Path(
    "outputs/322_s2_low_emission_matched_negative_manifest_v1.csv"
)


COLLECTION_ID = (
    "COPERNICUS/S2_SR_HARMONIZED"
)

SEARCH_WINDOW_DAYS = 60
RELEASE_EXCLUSION_HOURS = 24
NEGATIVES_PER_POSITIVE = 4
NEGATIVES_BEFORE = 2
NEGATIVES_AFTER = 2

# Scene-level 雲量只是初步篩選。
# 後面仍會檢查釋放點附近 local cloud。
MAX_SCENE_CLOUD_PERCENT = 40.0


def initialize_earth_engine():
    project = os.environ.get(
        "EE_PROJECT"
    )

    if not project:
        raise RuntimeError(
            "EE_PROJECT 尚未設定。"
        )

    ee.Initialize(
        project=project
    )

    print(
        "Earth Engine project:",
        project,
    )


def parse_bool(value):
    return (
        str(value)
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )


def extract_mgrs_tile(
    scene_id,
):
    match = re.search(
        r"_T(\d{2}[A-Z]{3})$",
        str(scene_id),
    )

    if not match:
        return None

    return match.group(1)


def load_release_intervals():
    releases = pd.read_csv(
        RELEASE_INPUT,
        low_memory=False,
    )

    releases[
        "release_start_utc"
    ] = pd.to_datetime(
        releases[
            "release_start_utc"
        ],
        errors="coerce",
        utc=True,
    )

    releases[
        "release_end_utc"
    ] = pd.to_datetime(
        releases[
            "release_end_utc"
        ],
        errors="coerce",
        utc=True,
    )

    releases[
        "release_rate_kg_h"
    ] = pd.to_numeric(
        releases[
            "release_rate_kg_h"
        ],
        errors="coerce",
    )

    releases = releases.dropna(
        subset=[
            "site",
            "release_start_utc",
            "release_end_utc",
            "release_rate_kg_h",
        ]
    ).copy()

    # Negative 只能避開真正的非零排放。
    releases = releases[
        releases[
            "release_rate_kg_h"
        ].gt(0)
    ].copy()

    return releases


def safely_away_from_releases(
    acquisition_time,
    site_releases,
):
    if site_releases.empty:
        return True, np.nan

    buffer_delta = pd.Timedelta(
        hours=RELEASE_EXCLUSION_HOURS
    )

    unsafe_start = (
        site_releases[
            "release_start_utc"
        ]
        - buffer_delta
    )

    unsafe_end = (
        site_releases[
            "release_end_utc"
        ]
        + buffer_delta
    )

    unsafe = (
        unsafe_start.le(acquisition_time)
        & unsafe_end.ge(acquisition_time)
    )

    if unsafe.any():
        return False, 0.0

    distances = []

    for _, release in (
        site_releases.iterrows()
    ):
        if (
            acquisition_time
            < release[
                "release_start_utc"
            ]
        ):
            distance = (
                release[
                    "release_start_utc"
                ]
                - acquisition_time
            )

        elif (
            acquisition_time
            > release[
                "release_end_utc"
            ]
        ):
            distance = (
                acquisition_time
                - release[
                    "release_end_utc"
                ]
            )

        else:
            distance = pd.Timedelta(0)

        distances.append(
            distance.total_seconds()
            / 3600.0
        )

    return True, min(distances)


def fetch_candidates(
    positive_row,
    site_releases,
    excluded_scene_ids,
):
    positive_time = positive_row[
        "acquisition_time_utc"
    ]

    latitude = float(
        positive_row["lat"]
    )

    longitude = float(
        positive_row["lon"]
    )

    positive_scene_id = str(
        positive_row["scene_id"]
    )

    mgrs_tile = extract_mgrs_tile(
        positive_scene_id
    )

    if not mgrs_tile:
        raise RuntimeError(
            "無法從 scene ID 取得 MGRS tile："
            + positive_scene_id
        )

    search_start = (
        positive_time
        - pd.Timedelta(
            days=SEARCH_WINDOW_DAYS
        )
    )

    search_end = (
        positive_time
        + pd.Timedelta(
            days=SEARCH_WINDOW_DAYS
        )
        + pd.Timedelta(seconds=1)
    )

    point = ee.Geometry.Point([
        longitude,
        latitude,
    ])

    collection = (
        ee.ImageCollection(
            COLLECTION_ID
        )
        .filterBounds(point)
        .filterDate(
            search_start.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            search_end.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
        )
        .filter(
            ee.Filter.eq(
                "MGRS_TILE",
                mgrs_tile,
            )
        )
        .filter(
            ee.Filter.lte(
                "CLOUDY_PIXEL_PERCENTAGE",
                MAX_SCENE_CLOUD_PERCENT,
            )
        )
        .sort(
            "system:time_start"
        )
    )

    info = (
        collection
        .select([])
        .getInfo()
    )

    records = []

    for feature in info.get(
        "features",
        [],
    ):
        properties = feature.get(
            "properties",
            {},
        )

        scene_id = feature.get(
            "id"
        )

        if not scene_id:
            continue

        if scene_id in excluded_scene_ids:
            continue

        timestamp_ms = properties.get(
            "system:time_start"
        )

        if timestamp_ms is None:
            continue

        acquisition_time = pd.to_datetime(
            timestamp_ms,
            unit="ms",
            utc=True,
            errors="coerce",
        )

        if pd.isna(acquisition_time):
            continue

        safe, nearest_release_hours = (
            safely_away_from_releases(
                acquisition_time,
                site_releases,
            )
        )

        days_from_positive = (
            acquisition_time
            - positive_time
        ).total_seconds() / 86400.0

        records.append({
            "matched_positive_scene_id":
                positive_scene_id,

            "matched_positive_time_utc":
                positive_time,

            "matched_positive_rate_kg_h":
                positive_row[
                    "final_release_rate_kg_h"
                ],

            "site":
                positive_row["site"],

            "lat":
                latitude,

            "lon":
                longitude,

            "scene_id":
                scene_id,

            "system_index":
                properties.get(
                    "system:index"
                ),

            "product_id":
                properties.get(
                    "PRODUCT_ID"
                ),

            "spacecraft_name":
                properties.get(
                    "SPACECRAFT_NAME"
                ),

            "mgrs_tile":
                properties.get(
                    "MGRS_TILE"
                ),

            "acquisition_time_utc":
                acquisition_time,

            "days_from_positive":
                days_from_positive,

            "absolute_days_from_positive":
                abs(
                    days_from_positive
                ),

            "temporal_side":
                (
                    "before"
                    if days_from_positive < 0
                    else "after"
                ),

            "scene_cloud_percentage":
                pd.to_numeric(
                    properties.get(
                        "CLOUDY_PIXEL_PERCENTAGE"
                    ),
                    errors="coerce",
                ),

            "safe_from_release":
                safe,

            "nearest_nonzero_release_hours":
                nearest_release_hours,

            "release_exclusion_hours":
                RELEASE_EXCLUSION_HOURS,

            "candidate_status":
                (
                    "eligible"
                    if safe
                    else "excluded_near_release"
                ),
        })

    return pd.DataFrame(
        records
    )


def select_side(
    candidates,
    side,
    count,
    globally_used_scene_ids,
):
    subset = candidates[
        candidates[
            "temporal_side"
        ].eq(side)
        & candidates[
            "candidate_status"
        ].eq("eligible")
        & ~candidates[
            "scene_id"
        ].isin(
            globally_used_scene_ids
        )
    ].copy()

    subset = subset.sort_values(
        [
            "absolute_days_from_positive",
            "scene_cloud_percentage",
            "acquisition_time_utc",
        ]
    )

    selected = []

    used_dates = set()

    for _, row in subset.iterrows():
        date = row[
            "acquisition_time_utc"
        ].date()

        if date in used_dates:
            continue

        selected.append(
            row.to_dict()
        )

        used_dates.add(date)

        if len(selected) >= count:
            break

    return selected


def select_negatives(
    candidates,
    globally_used_scene_ids,
):
    selected = []

    selected.extend(
        select_side(
            candidates,
            side="before",
            count=NEGATIVES_BEFORE,
            globally_used_scene_ids=
                globally_used_scene_ids,
        )
    )

    provisional_used = (
        globally_used_scene_ids
        | {
            row["scene_id"]
            for row in selected
        }
    )

    selected.extend(
        select_side(
            candidates,
            side="after",
            count=NEGATIVES_AFTER,
            globally_used_scene_ids=
                provisional_used,
        )
    )

    # 若某一側不足，就從所有剩餘候選補滿。
    remaining_needed = (
        NEGATIVES_PER_POSITIVE
        - len(selected)
    )

    if remaining_needed > 0:
        used = (
            globally_used_scene_ids
            | {
                row["scene_id"]
                for row in selected
            }
        )

        remaining = candidates[
            candidates[
                "candidate_status"
            ].eq("eligible")
            & ~candidates[
                "scene_id"
            ].isin(used)
        ].copy()

        remaining = remaining.sort_values(
            [
                "absolute_days_from_positive",
                "scene_cloud_percentage",
                "acquisition_time_utc",
            ]
        )

        used_dates = {
            pd.to_datetime(
                row[
                    "acquisition_time_utc"
                ],
                utc=True,
            ).date()
            for row in selected
        }

        for _, row in (
            remaining.iterrows()
        ):
            date = row[
                "acquisition_time_utc"
            ].date()

            if date in used_dates:
                continue

            selected.append(
                row.to_dict()
            )

            used_dates.add(date)

            if (
                len(selected)
                >= NEGATIVES_PER_POSITIVE
            ):
                break

    return pd.DataFrame(
        selected
    )


def main():
    initialize_earth_engine()

    positives = pd.read_csv(
        POSITIVE_INPUT,
        low_memory=False,
    )

    all_manifest = pd.read_csv(
        ALL_SCENE_INPUT,
        low_memory=False,
    )

    releases = (
        load_release_intervals()
    )

    positives[
        "acquisition_time_utc"
    ] = pd.to_datetime(
        positives[
            "acquisition_time_utc"
        ],
        errors="coerce",
        utc=True,
    )

    positives[
        "final_release_rate_kg_h"
    ] = pd.to_numeric(
        positives[
            "final_release_rate_kg_h"
        ],
        errors="coerce",
    )

    positives = positives.dropna(
        subset=[
            "scene_id",
            "site",
            "lat",
            "lon",
            "acquisition_time_utc",
            "final_release_rate_kg_h",
        ]
    ).copy()

    excluded_scene_ids = set(
        all_manifest[
            "scene_id"
        ]
        .dropna()
        .astype(str)
    )

    candidate_frames = []
    selected_frames = []
    globally_used_scene_ids = set()

    print("=" * 110)
    print(
        "SENTINEL-2 MATCHED NEGATIVE SEARCH"
    )
    print("=" * 110)

    print(
        "\nPrimary positives:",
        len(positives),
    )

    for number, positive in (
        positives.sort_values(
            "acquisition_time_utc"
        )
        .reset_index(drop=True)
        .iterrows()
    ):
        site_releases = releases[
            releases["site"].eq(
                positive["site"]
            )
        ].copy()

        print(
            f"\n[{number + 1}/{len(positives)}] "
            f"{positive['site']} | "
            f"{positive['acquisition_time_utc']} | "
            f"{positive['final_release_rate_kg_h']:.3f} kg/h"
        )

        candidates = fetch_candidates(
            positive_row=positive,
            site_releases=site_releases,
            excluded_scene_ids=
                excluded_scene_ids,
        )

        candidate_frames.append(
            candidates
        )

        eligible_count = int(
            candidates[
                "candidate_status"
            ].eq("eligible").sum()
        )

        excluded_count = int(
            candidates[
                "candidate_status"
            ].eq(
                "excluded_near_release"
            ).sum()
        )

        print(
            "  Total candidate scenes:",
            len(candidates),
        )

        print(
            "  Eligible negatives:",
            eligible_count,
        )

        print(
            "  Excluded near release:",
            excluded_count,
        )

        selected = select_negatives(
            candidates,
            globally_used_scene_ids,
        )

        if selected.empty:
            print(
                "  Selected negatives: 0"
            )
            continue

        selected[
            "label"
        ] = 0

        selected[
            "dataset_role"
        ] = "matched_negative"

        selected[
            "local_qa_status"
        ] = "pending"

        selected[
            "selection_version"
        ] = (
            "s2_low_emission_negative_v1"
        )

        selected[
            "negative_id"
        ] = [
            (
                f"S2_NEG_"
                f"{number + 1:02d}_"
                f"{index + 1:02d}"
            )
            for index in range(
                len(selected)
            )
        ]

        selected_frames.append(
            selected
        )

        globally_used_scene_ids.update(
            selected[
                "scene_id"
            ].astype(str)
        )

        print(
            "  Selected negatives:",
            len(selected),
        )

        print(
            selected[
                [
                    "negative_id",
                    "acquisition_time_utc",
                    "days_from_positive",
                    "scene_cloud_percentage",
                    "nearest_nonzero_release_hours",
                ]
            ].to_string(
                index=False,
            )
        )

    all_candidates = (
        pd.concat(
            candidate_frames,
            ignore_index=True,
            sort=False,
        )
        if candidate_frames
        else pd.DataFrame()
    )

    selected_manifest = (
        pd.concat(
            selected_frames,
            ignore_index=True,
            sort=False,
        )
        if selected_frames
        else pd.DataFrame()
    )

    if not selected_manifest.empty:
        selected_manifest = (
            selected_manifest.sort_values(
                [
                    "matched_positive_time_utc",
                    "acquisition_time_utc",
                ]
            )
            .drop_duplicates(
                subset=["scene_id"],
                keep="first",
            )
            .reset_index(drop=True)
        )

    all_candidates.to_csv(
        CANDIDATE_OUTPUT,
        index=False,
    )

    selected_manifest.to_csv(
        SELECTED_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 110)
    print(
        "MATCHED NEGATIVE SEARCH SUMMARY"
    )
    print("=" * 110)

    print(
        "\nAll candidate rows:",
        len(all_candidates),
    )

    print(
        "Unique selected negatives:",
        len(selected_manifest),
    )

    if not selected_manifest.empty:
        print("\nSelected negatives per positive:")
        print(
            selected_manifest.groupby(
                [
                    "matched_positive_time_utc",
                    "matched_positive_rate_kg_h",
                ]
            )["scene_id"].nunique()
        )

        print("\nTemporal side:")
        print(
            selected_manifest[
                "temporal_side"
            ].value_counts(
                dropna=False
            )
        )

        print("\nCloud statistics:")
        print(
            selected_manifest[
                "scene_cloud_percentage"
            ].describe()
        )

        print(
            "\nMinimum distance from "
            "nonzero release (hours):"
        )

        print(
            selected_manifest[
                "nearest_nonzero_release_hours"
            ].min()
        )

        print("\nFinal selected scenes:")
        print(
            selected_manifest[
                [
                    "negative_id",
                    "matched_positive_time_utc",
                    "matched_positive_rate_kg_h",
                    "acquisition_time_utc",
                    "days_from_positive",
                    "temporal_side",
                    "scene_cloud_percentage",
                    "nearest_nonzero_release_hours",
                    "scene_id",
                ]
            ].to_string(
                index=False,
            )
        )

    print("\nSaved:")
    print(CANDIDATE_OUTPUT)
    print(SELECTED_OUTPUT)


if __name__ == "__main__":
    main()
