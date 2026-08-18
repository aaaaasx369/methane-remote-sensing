from pathlib import Path

import pandas as pd


PAIR_INPUT = Path(
    "outputs/412_landsat_global_matched_negative_pairs_v1.csv"
)

POOL_INPUT = Path(
    "outputs/411_landsat_combined_clean_24h_candidates_v1.csv"
)

OUTPUT = Path(
    "outputs/416_landsat_selected_negative_download_manifest_v1.csv"
)

PATCH_DIR = Path(
    "sample_patches/landsat_matched_negatives_v1"
)


def find_column(frame, candidates, table_name):
    for column in candidates:
        if column in frame.columns:
            return column

    raise KeyError(
        f"{table_name} 找不到欄位："
        + ", ".join(candidates)
    )


def main():
    pairs = pd.read_csv(
        PAIR_INPUT,
        low_memory=False,
    )

    pool = pd.read_csv(
        POOL_INPUT,
        low_memory=False,
    )

    if len(pairs) != 28:
        raise RuntimeError(
            f"預期 28 個 selected negatives，實際為 {len(pairs)}。"
        )

    pair_key = find_column(
        pairs,
        [
            "negative_overpass_key",
        ],
        "Pair table",
    )

    pool_key = find_column(
        pool,
        [
            "independent_overpass_key",
            "negative_overpass_key",
        ],
        "Candidate pool",
    )

    if pairs[pair_key].duplicated().any():
        raise RuntimeError(
            "Pair table 的 negative overpass 有重複。"
        )

    if pool[pool_key].duplicated().any():
        duplicated = pool.loc[
            pool[pool_key].duplicated(
                keep=False
            ),
            pool_key,
        ]

        raise RuntimeError(
            "Candidate pool 的 overpass key 有重複：\n"
            + duplicated.to_string(index=False)
        )

    selected = pairs.merge(
        pool,
        left_on=pair_key,
        right_on=pool_key,
        how="left",
        validate="one_to_one",
        suffixes=("_pair", ""),
        indicator=True,
    )

    unmatched = selected[
        selected["_merge"].ne("both")
    ]

    if not unmatched.empty:
        raise RuntimeError(
            "部分 selected negatives 無法在候選池找到：\n"
            + unmatched[
                [
                    "negative_scene_id",
                    pair_key,
                ]
            ].to_string(index=False)
        )

    selected = selected.drop(
        columns=["_merge"]
    )

    product_column = find_column(
        selected,
        [
            "LANDSAT_PRODUCT_ID",
            "negative_scene_id",
            "candidate_scene_id_standard",
        ],
        "Merged table",
    )

    system_index_column = find_column(
        selected,
        [
            "system:index",
            "LANDSAT_SCENE_ID",
            "LANDSAT_PRODUCT_ID",
        ],
        "Merged table",
    )

    collection_column = find_column(
        selected,
        [
            "gee_collection_id",
        ],
        "Merged table",
    )

    time_column = find_column(
        selected,
        [
            "candidate_acquisition_time_utc",
            "candidate_time_parsed_utc",
            "candidate_time_utc",
            "negative_time_utc",
        ],
        "Merged table",
    )

    latitude_column = find_column(
        selected,
        [
            "site_lat",
            "lat",
            "latitude",
        ],
        "Merged table",
    )

    longitude_column = find_column(
        selected,
        [
            "site_lon",
            "lon",
            "longitude",
        ],
        "Merged table",
    )

    site_column = find_column(
        selected,
        [
            "candidate_site_alias",
            "site_alias",
            "site_key",
            "positive_site_alias",
        ],
        "Merged table",
    )

    sensor_column = find_column(
        selected,
        [
            "landsat_sensor",
            "negative_sensor",
        ],
        "Merged table",
    )

    selected["acquisition_time_utc"] = (
        pd.to_datetime(
            selected[time_column],
            errors="coerce",
            utc=True,
        )
    )

    selected["site_lat_standard"] = (
        pd.to_numeric(
            selected[latitude_column],
            errors="coerce",
        )
    )

    selected["site_lon_standard"] = (
        pd.to_numeric(
            selected[longitude_column],
            errors="coerce",
        )
    )

    selected = selected.sort_values(
        [
            "positive_site_alias",
            "positive_time_utc",
            "temporal_side",
            "acquisition_time_utc",
        ]
    ).reset_index(drop=True)

    selected["sample_id"] = [
        f"LANDSAT_MATCHED_NEG_{number:03d}"
        for number in range(
            1,
            len(selected) + 1,
        )
    ]

    selected["label"] = 0

    selected["site_alias_standard"] = (
        selected[site_column].astype(str)
    )

    selected["landsat_sensor_standard"] = (
        selected[sensor_column].astype(str)
    )

    selected["landsat_product_id_standard"] = (
        selected[product_column].astype(str)
    )

    selected["gee_system_index_standard"] = (
        selected[system_index_column].astype(str)
    )

    selected["gee_collection_id_standard"] = (
        selected[collection_column].astype(str)
    )

    selected["patch_filename"] = (
        selected["sample_id"]
        + "_label_0.tif"
    )

    selected["patch_path"] = (
        selected["patch_filename"].map(
            lambda filename:
                str(PATCH_DIR / filename)
        )
    )

    selected[
        "download_status"
    ] = "pending"

    selected[
        "negative_definition"
    ] = (
        "no_exact_release_overlap_"
        "and_more_than_24h"
    )

    required_download_columns = [
        "sample_id",
        "landsat_product_id_standard",
        "gee_system_index_standard",
        "gee_collection_id_standard",
        "acquisition_time_utc",
        "site_lat_standard",
        "site_lon_standard",
        "patch_path",
    ]

    missing_counts = (
        selected[
            required_download_columns
        ]
        .isna()
        .sum()
    )

    missing_counts = missing_counts[
        missing_counts.gt(0)
    ]

    if not missing_counts.empty:
        raise RuntimeError(
            "下載必要欄位有缺值：\n"
            + missing_counts.to_string()
        )

    if selected[
        "landsat_product_id_standard"
    ].duplicated().any():
        raise RuntimeError(
            "Selected LANDSAT_PRODUCT_ID 有重複。"
        )

    if selected[
        "patch_path"
    ].duplicated().any():
        raise RuntimeError(
            "Patch output path 有重複。"
        )

    site_counts = (
        selected[
            "site_alias_standard"
        ].value_counts()
    )

    if int(
        site_counts.get(
            "casa_grande",
            0,
        )
    ) != 20:
        raise RuntimeError(
            "Casa Grande 應有 20 張 selected negatives。"
        )

    if int(
        site_counts.get(
            "ehrenberg",
            0,
        )
    ) != 8:
        raise RuntimeError(
            "Ehrenberg 應有 8 張 selected negatives。"
        )

    PATCH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 100)
    print(
        "LANDSAT SELECTED NEGATIVE DOWNLOAD MANIFEST"
    )
    print("=" * 100)

    print("\nRows:", len(selected))
    print(
        "Unique product IDs:",
        selected[
            "landsat_product_id_standard"
        ].nunique(),
    )

    print(
        "Unique overpasses:",
        selected[
            "negative_overpass_key"
        ].nunique(),
    )

    print("\nBy site:")
    print(site_counts)

    print("\nBy sensor:")
    print(
        selected[
            "landsat_sensor_standard"
        ].value_counts()
    )

    print("\nBy temporal side:")
    print(
        selected[
            "temporal_side"
        ].value_counts()
    )

    print(
        "\nMissing GEE collection IDs:",
        int(
            selected[
                "gee_collection_id_standard"
            ].isna().sum()
        ),
    )

    print(
        "Missing coordinates:",
        int(
            selected[
                [
                    "site_lat_standard",
                    "site_lon_standard",
                ]
            ].isna().any(axis=1).sum()
        ),
    )

    print("\nPatch directory:")
    print(PATCH_DIR)

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
