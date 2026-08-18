from pathlib import Path

import pandas as pd


INPUT_CSV = Path(
    "outputs/61_landsat_matched_negative_candidates.csv"
)

RESOLVED_OUTPUT_CSV = Path(
    "outputs/65_landsat_candidates_resolved.csv"
)

OVERLAP_OUTPUT_CSV = Path(
    "outputs/66_landsat_adjacent_overlap_exclusions.csv"
)


EXISTING_ROLES = {
    "existing_confirmed_positive",
    "existing_confirmed_negative",
}


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    df["candidate_time_utc"] = pd.to_datetime(
        df["candidate_time_utc"],
        errors="coerce",
        utc=True,
    )

    df["existing_time_difference_seconds"] = (
        pd.to_numeric(
            df["existing_time_difference_seconds"],
            errors="coerce",
        )
    )

    df["scene_match_resolution"] = ""

    existing_mask = df["candidate_role"].isin(
        EXISTING_ROLES
    )

    existing = df[
        existing_mask
    ].copy()

    # 找出同一個本地 raster 被多個 GEE product 配對的情形。
    duplicated_groups = (
        existing[
            "existing_raster_group_id"
        ]
        .value_counts()
    )

    duplicated_groups = (
        duplicated_groups[
            duplicated_groups > 1
        ]
        .index
        .tolist()
    )

    excluded_indices = []

    for raster_group_id in duplicated_groups:
        group = existing[
            existing["existing_raster_group_id"]
            == raster_group_id
        ].copy()

        # 與本地 landsat_image_time 最接近者，
        # 才是真正下載過的 Earth Engine product。
        group = group.sort_values(
            by=[
                "existing_time_difference_seconds",
                "candidate_time_utc",
            ]
        )

        keep_index = group.index[0]

        df.loc[
            keep_index,
            "scene_match_resolution",
        ] = "nearest_time_exact_existing_product"

        extra_indices = group.index[1:].tolist()

        for index in extra_indices:
            original_role = df.loc[
                index,
                "candidate_role",
            ]

            df.loc[
                index,
                "candidate_role",
            ] = (
                "adjacent_wrs_overlap_"
                "same_overpass_exclude"
            )

            df.loc[
                index,
                "release_check_required",
            ] = False

            df.loc[
                index,
                "scene_match_resolution",
            ] = (
                "Same Landsat overpass and date as an "
                "existing raster, but an adjacent overlapping "
                "WRS row. Excluded as a non-independent scene. "
                f"Original role: {original_role}."
            )

            excluded_indices.append(index)

    # 唯一配對的 existing scenes 也記錄原因。
    remaining_existing_mask = (
        df["candidate_role"].isin(
            EXISTING_ROLES
        )
        & (
            df["scene_match_resolution"]
            == ""
        )
    )

    df.loc[
        remaining_existing_mask,
        "scene_match_resolution",
    ] = "unique_existing_product_match"

    overlap_exclusions = df.loc[
        excluded_indices
    ].copy()

    # 排序後輸出。
    df = df.sort_values(
        by=[
            "site_key",
            "candidate_role",
            "candidate_time_utc",
        ]
    ).reset_index(drop=True)

    RESOLVED_OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        RESOLVED_OUTPUT_CSV,
        index=False,
    )

    overlap_exclusions.to_csv(
        OVERLAP_OUTPUT_CSV,
        index=False,
    )

    print("=" * 100)
    print("RESOLVED LANDSAT EXISTING-SCENE MATCHES")
    print("=" * 100)

    print(f"\nInput rows: {len(df)}")

    print("\nResolved candidate-role counts:")
    print(
        df["candidate_role"]
        .value_counts()
    )

    print("\nUnique existing raster groups by label:")

    resolved_existing = df[
        df["candidate_role"].isin(
            EXISTING_ROLES
        )
    ].copy()

    print(
        resolved_existing.groupby(
            "existing_label"
        )["existing_raster_group_id"]
        .nunique()
    )

    print("\nAdjacent WRS overlap exclusions:")

    if len(overlap_exclusions) == 0:
        print("None")
    else:
        display_columns = [
            column
            for column in [
                "existing_raster_group_id",
                "existing_label",
                "site_name_normalized",
                "candidate_time_utc",
                "LANDSAT_PRODUCT_ID",
                "WRS_PATH",
                "WRS_ROW",
                "existing_time_difference_seconds",
                "candidate_role",
                "scene_match_resolution",
            ]
            if column in overlap_exclusions.columns
        ]

        print(
            overlap_exclusions[
                display_columns
            ].to_string(index=False)
        )

    print("\nResolved existing matches:")

    display_columns = [
        column
        for column in [
            "existing_raster_group_id",
            "existing_label",
            "site_name_normalized",
            "candidate_time_utc",
            "LANDSAT_PRODUCT_ID",
            "WRS_PATH",
            "WRS_ROW",
            "existing_time_difference_seconds",
            "candidate_role",
        ]
        if column in resolved_existing.columns
    ]

    print(
        resolved_existing[
            display_columns
        ].sort_values(
            [
                "existing_label",
                "existing_raster_group_id",
            ]
        ).to_string(index=False)
    )

    print("\nSaved:")
    print(RESOLVED_OUTPUT_CSV)
    print(OVERLAP_OUTPUT_CSV)


if __name__ == "__main__":
    main()
