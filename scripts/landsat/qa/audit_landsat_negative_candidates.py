from pathlib import Path
import pandas as pd


CANDIDATE_CSV = Path(
    "outputs/61_landsat_matched_negative_candidates.csv"
)

AUDIT_OUTPUT_CSV = Path(
    "outputs/63_landsat_candidate_duplicate_audit.csv"
)

UNIQUE_CANDIDATE_OUTPUT_CSV = Path(
    "outputs/64_landsat_unique_new_candidates.csv"
)


def print_duplicate_rows(df, column, title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if column not in df.columns:
        print(f"Column not found: {column}")
        return pd.DataFrame()

    duplicate_mask = (
        df[column].notna()
        & df[column].duplicated(keep=False)
    )

    duplicated = df[
        duplicate_mask
    ].copy()

    if len(duplicated) == 0:
        print("No duplicates.")
        return duplicated

    display_columns = [
        candidate
        for candidate in [
            "site_key",
            "site_name_normalized",
            "landsat_sensor",
            "candidate_time_utc",
            "system:index",
            "LANDSAT_PRODUCT_ID",
            "LANDSAT_SCENE_ID",
            "WRS_PATH",
            "WRS_ROW",
            "CLOUD_COVER",
            "candidate_role",
            "existing_raster_group_id",
            "existing_label",
            column,
        ]
        if candidate in duplicated.columns
    ]

    display_columns = list(
        dict.fromkeys(display_columns)
    )

    print(
        duplicated[
            display_columns
        ].sort_values(
            [column, "candidate_time_utc"]
        ).to_string(index=False)
    )

    return duplicated


def main():
    if not CANDIDATE_CSV.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {CANDIDATE_CSV}"
        )

    df = pd.read_csv(
        CANDIDATE_CSV,
        low_memory=False,
    )

    if "candidate_time_utc" in df.columns:
        df["candidate_time_utc"] = pd.to_datetime(
            df["candidate_time_utc"],
            errors="coerce",
            utc=True,
        )

    print("=" * 100)
    print("LANDSAT NEGATIVE-CANDIDATE AUDIT")
    print("=" * 100)

    print(f"\nInput rows: {len(df)}")

    print("\nCandidate-role counts:")
    print(
        df["candidate_role"]
        .value_counts(dropna=False)
    )

    print("\nUnique counts:")
    for column in [
        "system:index",
        "LANDSAT_PRODUCT_ID",
        "LANDSAT_SCENE_ID",
    ]:
        if column in df.columns:
            print(
                f"{column}: "
                f"{df[column].nunique(dropna=True)}"
            )

    duplicate_tables = []

    for column, title in [
        (
            "system:index",
            "DUPLICATE system:index ROWS",
        ),
        (
            "LANDSAT_PRODUCT_ID",
            "DUPLICATE LANDSAT_PRODUCT_ID ROWS",
        ),
        (
            "LANDSAT_SCENE_ID",
            "DUPLICATE LANDSAT_SCENE_ID ROWS",
        ),
    ]:
        duplicated = print_duplicate_rows(
            df,
            column,
            title,
        )

        if len(duplicated) > 0:
            duplicated = duplicated.copy()
            duplicated["duplicate_key_type"] = column
            duplicate_tables.append(duplicated)

    print("\n" + "=" * 100)
    print("EXISTING RASTER-GROUP MATCH COUNTS")
    print("=" * 100)

    existing = df[
        df["candidate_role"].isin([
            "existing_confirmed_positive",
            "existing_confirmed_negative",
        ])
    ].copy()

    match_counts = (
        existing[
            "existing_raster_group_id"
        ]
        .value_counts(dropna=False)
    )

    print(match_counts)

    repeated_existing_groups = (
        match_counts[
            match_counts > 1
        ]
    )

    print("\nExisting raster groups matched more than once:")

    if len(repeated_existing_groups) == 0:
        print("None")
    else:
        repeated_rows = existing[
            existing[
                "existing_raster_group_id"
            ].isin(
                repeated_existing_groups.index
            )
        ].copy()

        display_columns = [
            column
            for column in [
                "existing_raster_group_id",
                "existing_label",
                "site_name_normalized",
                "landsat_sensor",
                "candidate_time_utc",
                "system:index",
                "LANDSAT_PRODUCT_ID",
                "LANDSAT_SCENE_ID",
                "existing_time_difference_seconds",
                "candidate_role",
            ]
            if column in repeated_rows.columns
        ]

        print(
            repeated_rows[
                display_columns
            ].sort_values(
                [
                    "existing_raster_group_id",
                    "candidate_time_utc",
                ]
            ).to_string(index=False)
        )

    print("\n" + "=" * 100)
    print("EXISTING MATCH SUMMARY")
    print("=" * 100)

    existing_summary_columns = [
        column
        for column in [
            "existing_raster_group_id",
            "existing_label",
            "site_name_normalized",
            "landsat_sensor",
            "candidate_time_utc",
            "system:index",
            "LANDSAT_PRODUCT_ID",
            "LANDSAT_SCENE_ID",
            "WRS_PATH",
            "WRS_ROW",
            "CLOUD_COVER",
            "existing_time_difference_seconds",
            "candidate_role",
        ]
        if column in existing.columns
    ]

    print(
        existing[
            existing_summary_columns
        ].sort_values(
            [
                "existing_label",
                "existing_raster_group_id",
                "candidate_time_utc",
            ]
        ).to_string(index=False)
    )

    # 建立新候選的唯一 scene 表。
    new_candidates = df[
        df["candidate_role"]
        == "new_candidate_needs_release_check"
    ].copy()

    scene_key = None

    for candidate_key in [
        "LANDSAT_PRODUCT_ID",
        "system:index",
        "LANDSAT_SCENE_ID",
    ]:
        if (
            candidate_key in new_candidates.columns
            and new_candidates[
                candidate_key
            ].notna().any()
        ):
            scene_key = candidate_key
            break

    if scene_key is None:
        raise ValueError(
            "No usable Landsat scene identifier exists."
        )

    new_candidates = new_candidates.sort_values(
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
    )

    unique_new_candidates = (
        new_candidates.drop_duplicates(
            subset=[
                "site_key",
                scene_key,
            ],
            keep="first",
        )
        .copy()
        .reset_index(drop=True)
    )

    unique_new_candidates[
        "unique_candidate_rank_within_site"
    ] = (
        unique_new_candidates.groupby(
            "site_key"
        ).cumcount()
        + 1
    )

    unique_new_candidates.to_csv(
        UNIQUE_CANDIDATE_OUTPUT_CSV,
        index=False,
    )

    if len(duplicate_tables) > 0:
        duplicate_audit = pd.concat(
            duplicate_tables,
            ignore_index=True,
            sort=False,
        )
    else:
        duplicate_audit = pd.DataFrame(
            columns=[
                "duplicate_key_type",
            ]
        )

    duplicate_audit.to_csv(
        AUDIT_OUTPUT_CSV,
        index=False,
    )

    print("\n" + "=" * 100)
    print("UNIQUE NEW-CANDIDATE SUMMARY")
    print("=" * 100)

    print(f"\nScene identifier used: {scene_key}")
    print(f"Original new-candidate rows: {len(new_candidates)}")
    print(
        "Unique new candidate scenes:",
        len(unique_new_candidates),
    )
    print(
        "Removed duplicate candidate rows:",
        len(new_candidates)
        - len(unique_new_candidates),
    )

    print("\nUnique new candidates by site and sensor:")
    print(
        pd.crosstab(
            unique_new_candidates[
                "site_name_normalized"
            ],
            unique_new_candidates[
                "landsat_sensor"
            ],
            margins=True,
        )
    )

    print("\nTop 20 unique new candidates:")
    display_columns = [
        column
        for column in [
            "unique_candidate_rank_within_site",
            "site_name_normalized",
            "candidate_time_utc",
            "landsat_sensor",
            "LANDSAT_PRODUCT_ID",
            "system:index",
            "WRS_PATH",
            "WRS_ROW",
            "CLOUD_COVER",
            "CLOUD_COVER_LAND",
            "same_reference_wrs",
            "sensor_matches_positive_at_site",
            "days_to_nearest_confirmed_positive",
            "candidate_priority_score",
        ]
        if column in unique_new_candidates.columns
    ]

    print(
        unique_new_candidates[
            display_columns
        ].head(20).to_string(index=False)
    )

    print("\nSaved:")
    print(AUDIT_OUTPUT_CSV)
    print(UNIQUE_CANDIDATE_OUTPUT_CSV)


if __name__ == "__main__":
    main()
