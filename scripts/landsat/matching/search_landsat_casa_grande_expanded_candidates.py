from pathlib import Path
import os
import re

import ee
import numpy as np
import pandas as pd


INPUT_CSV = Path(
    "outputs/57_landsat_final_confirmed_features.csv"
)

SITE_OUTPUT_CSV = Path(
    "outputs/402_casa_grande_expanded_site_reference_v1.csv"
)

CANDIDATE_OUTPUT_CSV = Path(
    "outputs/403_casa_grande_expanded_candidates_v1.csv"
)

EXISTING_LOOKUP_CSV = Path(
    "outputs/404_casa_grande_expanded_existing_lookup_v1.csv"
)


COLLECTIONS = {
    "Landsat-8": "LANDSAT/LC08/C02/T1_L2",
    "Landsat-9": "LANDSAT/LC09/C02/T1_L2",
}


KNOWN_SITES = [
    {
        "site_key": "casa_grande",
        "site_name_normalized":
            "Casa_Grande_AZ_release_stacks",
        "reference_lat": 32.821821,
        "reference_lon": -111.785773,
    },
    {
        "site_key": "ehrenberg",
        "site_name_normalized":
            "Ehrenberg_AZ_release_stack",
        "reference_lat": 33.630645,
        "reference_lon": -114.489150,
    },
]


# 候選搜尋期間：
# 每個場址最早 confirmed scene 前 45 天，
# 到最晚 confirmed scene 後 45 天。
SEARCH_BUFFER_DAYS = 90

# 這裡先使用較寬鬆的 scene-level 雲量門檻。
# 因為你的正樣本中有一張 CLOUD_COVER = 45.18，
# 之後還會用 QA_PIXEL 檢查場址附近的局部雲量。
MAX_SCENE_CLOUD_COVER = 50.0

# Earth Engine 與本地時間可能有零星秒數差異。
EXISTING_TIME_MATCH_TOLERANCE_SECONDS = 180


GEE_PROPERTIES = [
    "system:index",
    "system:time_start",
    "LANDSAT_PRODUCT_ID",
    "LANDSAT_SCENE_ID",
    "SPACECRAFT_ID",
    "SENSOR_ID",
    "WRS_PATH",
    "WRS_ROW",
    "CLOUD_COVER",
    "CLOUD_COVER_LAND",
    "COLLECTION_CATEGORY",
    "PROCESSING_LEVEL",
    "IMAGE_QUALITY_OLI",
    "GEOMETRIC_RMSE_MODEL",
]


def detect_gee_project():
    """
    依序嘗試：
    1. 環境變數 EE_PROJECT
    2. 舊 Landsat 下載程式中的 ee.Initialize(project=...)
    3. 不指定 project，使用目前 Earth Engine 登入設定
    """
    environment_project = os.environ.get(
        "EE_PROJECT"
    )

    if environment_project:
        return environment_project

    previous_scripts = [
        Path(
            "auto_download_controlled_release_"
            "landsat_patches.py"
        ),
        Path(
            "auto_download_controlled_release_"
            "s2_patches.py"
        ),
    ]

    pattern = re.compile(
        r"""ee\.Initialize\(
            \s*project\s*=\s*
            ["']([^"']+)["']
        """,
        flags=re.VERBOSE,
    )

    for script_path in previous_scripts:
        if not script_path.exists():
            continue

        text = script_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        match = pattern.search(text)

        if match:
            return match.group(1)

    return None


def initialize_earth_engine():
    project = detect_gee_project()

    try:
        if project:
            ee.Initialize(project=project)
            print(
                f"Earth Engine initialized "
                f"with project: {project}"
            )
        else:
            ee.Initialize()
            print(
                "Earth Engine initialized using "
                "the current default project."
            )

    except Exception as error:
        print("\nEarth Engine initialization failed.")
        print(f"Original error: {error}")
        print(
            "\nIf authentication expired, run:"
        )
        print("earthengine authenticate")
        print(
            "\nIf a project is required, run:"
        )
        print(
            'export EE_PROJECT="your-google-cloud-project-id"'
        )

        raise


def assign_site(latitude, longitude):
    """
    將每一列依照最近的已知座標分配到場址。
    """
    latitude = float(latitude)
    longitude = float(longitude)

    best_site = None
    best_distance = np.inf

    for site in KNOWN_SITES:
        distance = np.sqrt(
            (
                latitude
                - site["reference_lat"]
            ) ** 2
            + (
                longitude
                - site["reference_lon"]
            ) ** 2
        )

        if distance < best_distance:
            best_distance = distance
            best_site = site

    # 約 0.01 度大致相當於 1 km 左右；
    # 目前資料座標幾乎完全一致，因此應遠小於此值。
    if best_distance > 0.01:
        raise ValueError(
            "A coordinate could not be assigned "
            f"to a known site: lat={latitude}, "
            f"lon={longitude}, "
            f"distance={best_distance}"
        )

    return (
        best_site["site_key"],
        best_site["site_name_normalized"],
        best_distance,
    )


def retrieve_collection_table(
    collection,
    properties,
):
    """
    使用 aggregate_array 取得指定 collection 的屬性表。
    不下載任何影像像素。
    """
    collection = collection.sort(
        "system:time_start"
    )

    count = int(
        collection.size().getInfo()
    )

    if count == 0:
        return pd.DataFrame(
            columns=properties
        )

    property_values = {}

    for property_name in properties:
        values = (
            collection
            .aggregate_array(property_name)
            .getInfo()
        )

        # 少數影像可能缺少某個非必要 metadata。
        if len(values) < count:
            values = list(values) + [
                None
            ] * (count - len(values))

        property_values[property_name] = (
            values[:count]
        )

    return pd.DataFrame(
        property_values
    )


def nearest_time_difference_days(
    candidate_time,
    reference_times,
):
    reference_times = pd.Series(
        reference_times
    ).dropna()

    if len(reference_times) == 0:
        return np.nan

    differences = (
        reference_times - candidate_time
    ).abs()

    return (
        differences.min().total_seconds()
        / 86400
    )


def identify_existing_scene(
    candidate_row,
    confirmed,
):
    same_site_sensor = confirmed[
        (
            confirmed["site_key"]
            == candidate_row["site_key"]
        )
        & (
            confirmed["landsat_sensor"]
            == candidate_row["landsat_sensor"]
        )
    ].copy()

    if len(same_site_sensor) == 0:
        return pd.Series({
            "existing_raster_group_id": np.nan,
            "existing_label": np.nan,
            "existing_time_difference_seconds":
                np.nan,
            "candidate_role":
                "new_candidate_needs_release_check",
        })

    same_site_sensor[
        "time_difference_seconds"
    ] = (
        same_site_sensor[
            "landsat_time_utc"
        ]
        - candidate_row[
            "candidate_time_utc"
        ]
    ).abs().dt.total_seconds()

    nearest = same_site_sensor.sort_values(
        "time_difference_seconds"
    ).iloc[0]

    if (
        nearest["time_difference_seconds"]
        <= EXISTING_TIME_MATCH_TOLERANCE_SECONDS
    ):
        label = int(nearest["label"])

        role = (
            "existing_confirmed_positive"
            if label == 1
            else "existing_confirmed_negative"
        )

        return pd.Series({
            "existing_raster_group_id":
                nearest["raster_group_id"],
            "existing_label":
                label,
            "existing_time_difference_seconds":
                nearest["time_difference_seconds"],
            "candidate_role":
                role,
        })

    return pd.Series({
        "existing_raster_group_id": np.nan,
        "existing_label": np.nan,
        "existing_time_difference_seconds":
            np.nan,
        "candidate_role":
            "new_candidate_needs_release_check",
    })


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    confirmed = pd.read_csv(
        INPUT_CSV
    )

    required_columns = [
        "raster_group_id",
        "label",
        "lat",
        "lon",
        "landsat_sensor",
        "landsat_image_time",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in confirmed.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required input columns: "
            + ", ".join(missing_columns)
        )

    confirmed["lat"] = pd.to_numeric(
        confirmed["lat"],
        errors="coerce",
    )

    confirmed["lon"] = pd.to_numeric(
        confirmed["lon"],
        errors="coerce",
    )

    confirmed["landsat_time_utc"] = (
        pd.to_datetime(
            confirmed["landsat_image_time"],
            errors="coerce",
            utc=True,
        )
    )

    invalid_rows = confirmed[
        confirmed["lat"].isna()
        | confirmed["lon"].isna()
        | confirmed["landsat_time_utc"].isna()
    ]

    if len(invalid_rows) > 0:
        raise ValueError(
            "Some confirmed scenes have invalid "
            "coordinates or acquisition times:\n"
            + invalid_rows[
                [
                    "raster_group_id",
                    "lat",
                    "lon",
                    "landsat_image_time",
                ]
            ].to_string(index=False)
        )

    site_assignments = confirmed.apply(
        lambda row: assign_site(
            row["lat"],
            row["lon"],
        ),
        axis=1,
    )

    confirmed["site_key"] = [
        item[0]
        for item in site_assignments
    ]

    confirmed[
        "site_name_normalized"
    ] = [
        item[1]
        for item in site_assignments
    ]

    confirmed[
        "coordinate_match_distance_degrees"
    ] = [
        item[2]
        for item in site_assignments
    ]

    site_rows = []

    for site_key, group in confirmed.groupby(
        "site_key"
    ):
        site_definition = next(
            site
            for site in KNOWN_SITES
            if site["site_key"] == site_key
        )

        minimum_time = (
            group["landsat_time_utc"].min()
        )

        maximum_time = (
            group["landsat_time_utc"].max()
        )

        search_start = (
            minimum_time
            - pd.Timedelta(
                days=SEARCH_BUFFER_DAYS
            )
        )

        search_end_inclusive = (
            maximum_time
            + pd.Timedelta(
                days=SEARCH_BUFFER_DAYS
            )
        )

        positive_sensors = sorted(
            group.loc[
                group["label"] == 1,
                "landsat_sensor",
            ]
            .dropna()
            .unique()
            .tolist()
        )

        site_rows.append({
            "site_key": site_key,
            "site_name_normalized":
                site_definition[
                    "site_name_normalized"
                ],
            "lat":
                site_definition[
                    "reference_lat"
                ],
            "lon":
                site_definition[
                    "reference_lon"
                ],
            "confirmed_scene_count":
                len(group),
            "confirmed_positive_count":
                int((group["label"] == 1).sum()),
            "confirmed_negative_count":
                int((group["label"] == 0).sum()),
            "positive_sensors":
                " | ".join(positive_sensors),
            "first_confirmed_scene":
                minimum_time,
            "last_confirmed_scene":
                maximum_time,
            "search_start":
                search_start,
            "search_end_inclusive":
                search_end_inclusive,
            # Earth Engine filterDate 的 end 是 exclusive。
            "gee_search_end_exclusive":
                search_end_inclusive
                + pd.Timedelta(days=1),
        })

    sites = pd.DataFrame(
        site_rows
    ).sort_values(
        "site_key"
    ).reset_index(drop=True)

    # Expanded search only for Casa Grande.
    sites = sites[
        sites["site_key"].eq("casa_grande")
    ].copy()

    if len(sites) != 1:
        raise RuntimeError(
            f"Expected one Casa Grande site, found {len(sites)}"
        )

    SITE_OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sites.to_csv(
        SITE_OUTPUT_CSV,
        index=False,
    )

    print("=" * 100)
    print("SITE REFERENCE TABLE")
    print("=" * 100)

    print(
        sites.to_string(index=False)
    )

    initialize_earth_engine()

    candidate_tables = []

    for _, site in sites.iterrows():
        point = ee.Geometry.Point([
            float(site["lon"]),
            float(site["lat"]),
        ])

        start_date = pd.Timestamp(
            site["search_start"]
        ).strftime("%Y-%m-%d")

        end_date = pd.Timestamp(
            site[
                "gee_search_end_exclusive"
            ]
        ).strftime("%Y-%m-%d")

        print("\n" + "-" * 100)
        print(
            f"Searching site: "
            f"{site['site_name_normalized']}"
        )
        print(
            f"Date range: {start_date} "
            f"to {end_date} (end exclusive)"
        )

        for sensor, collection_id in (
            COLLECTIONS.items()
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
                        MAX_SCENE_CLOUD_COVER,
                    )
                )
            )

            table = retrieve_collection_table(
                collection,
                GEE_PROPERTIES,
            )

            print(
                f"{sensor}: "
                f"{len(table)} candidate scenes"
            )

            if len(table) == 0:
                continue

            table["site_key"] = (
                site["site_key"]
            )

            table[
                "site_name_normalized"
            ] = site[
                "site_name_normalized"
            ]

            table["site_lat"] = site["lat"]
            table["site_lon"] = site["lon"]
            table["landsat_sensor"] = sensor
            table[
                "gee_collection_id"
            ] = collection_id
            table[
                "search_start"
            ] = start_date
            table[
                "search_end_exclusive"
            ] = end_date

            candidate_tables.append(
                table
            )

    if len(candidate_tables) == 0:
        raise RuntimeError(
            "No Landsat candidate scenes were found."
        )

    candidates = pd.concat(
        candidate_tables,
        ignore_index=True,
        sort=False,
    )

    candidates[
        "candidate_time_utc"
    ] = pd.to_datetime(
        candidates["system:time_start"],
        unit="ms",
        errors="coerce",
        utc=True,
    )

    numeric_columns = [
        "WRS_PATH",
        "WRS_ROW",
        "CLOUD_COVER",
        "CLOUD_COVER_LAND",
        "IMAGE_QUALITY_OLI",
        "GEOMETRIC_RMSE_MODEL",
    ]

    for column in numeric_columns:
        if column in candidates.columns:
            candidates[column] = pd.to_numeric(
                candidates[column],
                errors="coerce",
            )

    existing_information = (
        candidates.apply(
            lambda row: identify_existing_scene(
                row,
                confirmed,
            ),
            axis=1,
        )
    )

    candidates = pd.concat(
        [
            candidates.reset_index(drop=True),
            existing_information.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    # 找出各場址在現有 confirmed scenes 中
    # 對應的 WRS path/row。
    reference_wrs = (
        candidates[
            candidates[
                "candidate_role"
            ].isin([
                "existing_confirmed_positive",
                "existing_confirmed_negative",
            ])
        ][
            [
                "site_key",
                "WRS_PATH",
                "WRS_ROW",
            ]
        ]
        .dropna()
        .drop_duplicates()
    )

    reference_wrs_pairs = {}

    for site_key, group in reference_wrs.groupby(
        "site_key"
    ):
        reference_wrs_pairs[
            site_key
        ] = set(
            zip(
                group["WRS_PATH"].astype(int),
                group["WRS_ROW"].astype(int),
            )
        )

    def same_reference_wrs(row):
        if (
            pd.isna(row["WRS_PATH"])
            or pd.isna(row["WRS_ROW"])
        ):
            return False

        pair = (
            int(row["WRS_PATH"]),
            int(row["WRS_ROW"]),
        )

        return pair in reference_wrs_pairs.get(
            row["site_key"],
            set(),
        )

    candidates[
        "same_reference_wrs"
    ] = candidates.apply(
        same_reference_wrs,
        axis=1,
    )

    site_positive_sensors = (
        confirmed[
            confirmed["label"] == 1
        ]
        .groupby("site_key")[
            "landsat_sensor"
        ]
        .apply(set)
        .to_dict()
    )

    candidates[
        "sensor_matches_positive_at_site"
    ] = candidates.apply(
        lambda row: (
            row["landsat_sensor"]
            in site_positive_sensors.get(
                row["site_key"],
                set(),
            )
        ),
        axis=1,
    )

    positive_times_by_site = {
        site_key: group[
            "landsat_time_utc"
        ].tolist()
        for site_key, group
        in confirmed[
            confirmed["label"] == 1
        ].groupby("site_key")
    }

    negative_times_by_site = {
        site_key: group[
            "landsat_time_utc"
        ].tolist()
        for site_key, group
        in confirmed[
            confirmed["label"] == 0
        ].groupby("site_key")
    }

    candidates[
        "days_to_nearest_confirmed_positive"
    ] = candidates.apply(
        lambda row: nearest_time_difference_days(
            row["candidate_time_utc"],
            positive_times_by_site.get(
                row["site_key"],
                [],
            ),
        ),
        axis=1,
    )

    candidates[
        "days_to_nearest_confirmed_negative"
    ] = candidates.apply(
        lambda row: nearest_time_difference_days(
            row["candidate_time_utc"],
            negative_times_by_site.get(
                row["site_key"],
                [],
            ),
        ),
        axis=1,
    )

    candidates[
        "release_check_required"
    ] = (
        candidates["candidate_role"]
        == "new_candidate_needs_release_check"
    )

    # 排名分數越低越優先：
    # 1. 必須是新 scene
    # 2. 優先同 WRS path/row
    # 3. 優先同 sensor
    # 4. 優先低 scene cloud cover
    # 5. 優先接近 positive acquisition dates
    candidates[
        "candidate_priority_score"
    ] = (
        candidates[
            "CLOUD_COVER"
        ].fillna(100)
        + candidates[
            "days_to_nearest_confirmed_positive"
        ].fillna(999) * 0.10
        + np.where(
            candidates["same_reference_wrs"],
            0,
            100,
        )
        + np.where(
            candidates[
                "sensor_matches_positive_at_site"
            ],
            0,
            30,
        )
        + np.where(
            candidates[
                "candidate_role"
            ]
            == "new_candidate_needs_release_check",
            0,
            1000,
        )
    )

    candidates = candidates.sort_values(
        by=[
            "site_key",
            "candidate_priority_score",
            "candidate_time_utc",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    candidates[
        "candidate_rank_within_site"
    ] = (
        candidates.groupby(
            "site_key"
        ).cumcount()
        + 1
    )

    candidates.to_csv(
        CANDIDATE_OUTPUT_CSV,
        index=False,
    )

    existing_lookup = candidates[
        candidates[
            "candidate_role"
        ].isin([
            "existing_confirmed_positive",
            "existing_confirmed_negative",
        ])
    ].copy()

    existing_lookup.to_csv(
        EXISTING_LOOKUP_CSV,
        index=False,
    )

    print("\n" + "=" * 100)
    print("CANDIDATE SEARCH SUMMARY")
    print("=" * 100)

    print(
        f"\nTotal Earth Engine scenes: "
        f"{len(candidates)}"
    )

    print("\nCandidate role counts:")
    print(
        candidates[
            "candidate_role"
        ].value_counts()
    )

    new_candidates = candidates[
        candidates[
            "candidate_role"
        ]
        == "new_candidate_needs_release_check"
    ].copy()

    print("\nNew candidates by site and sensor:")
    if len(new_candidates) == 0:
        print("None")
    else:
        print(
            pd.crosstab(
                new_candidates[
                    "site_name_normalized"
                ],
                new_candidates[
                    "landsat_sensor"
                ],
                margins=True,
            )
        )

    print("\nExisting scene matches:")
    existing_display_columns = [
        "site_name_normalized",
        "candidate_time_utc",
        "landsat_sensor",
        "LANDSAT_PRODUCT_ID",
        "WRS_PATH",
        "WRS_ROW",
        "CLOUD_COVER",
        "existing_raster_group_id",
        "existing_label",
        "candidate_role",
    ]

    print(
        existing_lookup[
            existing_display_columns
        ].to_string(index=False)
    )

    print("\nTop new candidates:")
    new_display_columns = [
        "candidate_rank_within_site",
        "site_name_normalized",
        "candidate_time_utc",
        "landsat_sensor",
        "LANDSAT_PRODUCT_ID",
        "WRS_PATH",
        "WRS_ROW",
        "CLOUD_COVER",
        "CLOUD_COVER_LAND",
        "same_reference_wrs",
        "sensor_matches_positive_at_site",
        "days_to_nearest_confirmed_positive",
        "candidate_priority_score",
    ]

    print(
        new_candidates[
            new_display_columns
        ].head(30).to_string(index=False)
    )

    print("\nSaved:")
    print(SITE_OUTPUT_CSV)
    print(CANDIDATE_OUTPUT_CSV)
    print(EXISTING_LOOKUP_CSV)


if __name__ == "__main__":
    main()
