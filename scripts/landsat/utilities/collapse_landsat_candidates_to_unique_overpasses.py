from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = Path(
    "outputs/65_landsat_candidates_resolved.csv"
)

UNIQUE_OUTPUT_CSV = Path(
    "outputs/67_landsat_unique_candidate_overpasses.csv"
)

EXCLUDED_OUTPUT_CSV = Path(
    "outputs/68_landsat_candidate_adjacent_wrs_exclusions.csv"
)


# 相同場址、相同衛星，而且拍攝時間相差不超過 3 分鐘，
# 視為同一次 Landsat overpass。
OVERPASS_TIME_TOLERANCE_SECONDS = 180


def parse_boolean(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
        .fillna(False)
    )


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    new_candidates = df[
        df["candidate_role"]
        == "new_candidate_needs_release_check"
    ].copy()

    new_candidates["candidate_time_utc"] = pd.to_datetime(
        new_candidates["candidate_time_utc"],
        errors="coerce",
        utc=True,
    )

    if new_candidates["candidate_time_utc"].isna().any():
        raise ValueError(
            "Some new candidates have invalid acquisition times."
        )

    for column in [
        "same_reference_wrs",
        "sensor_matches_positive_at_site",
    ]:
        if column in new_candidates.columns:
            new_candidates[column] = parse_boolean(
                new_candidates[column]
            )
        else:
            new_candidates[column] = False

    for column in [
        "CLOUD_COVER",
        "candidate_priority_score",
        "WRS_PATH",
        "WRS_ROW",
    ]:
        if column in new_candidates.columns:
            new_candidates[column] = pd.to_numeric(
                new_candidates[column],
                errors="coerce",
            )

    new_candidates = new_candidates.sort_values(
        by=[
            "site_key",
            "landsat_sensor",
            "candidate_time_utc",
        ]
    ).reset_index(drop=True)

    # 建立 overpass group。
    overpass_group_numbers = []
    global_group_number = 0

    for _, group in new_candidates.groupby(
        ["site_key", "landsat_sensor"],
        sort=False,
    ):
        previous_time = None

        for index, row in group.iterrows():
            current_time = row["candidate_time_utc"]

            if (
                previous_time is None
                or (
                    current_time - previous_time
                ).total_seconds()
                > OVERPASS_TIME_TOLERANCE_SECONDS
            ):
                global_group_number += 1

            overpass_group_numbers.append(
                (
                    index,
                    global_group_number,
                )
            )

            previous_time = current_time

    group_lookup = dict(overpass_group_numbers)

    new_candidates["overpass_group_number"] = (
        new_candidates.index.map(group_lookup)
    )

    new_candidates["overpass_id"] = (
        "OP_"
        + new_candidates[
            "overpass_group_number"
        ]
        .astype(int)
        .astype(str)
        .str.zfill(3)
    )

    representative_rows = []
    excluded_rows = []

    for overpass_id, group in new_candidates.groupby(
        "overpass_id",
        sort=False,
    ):
        group = group.copy()

        group["selection_same_wrs_rank"] = np.where(
            group["same_reference_wrs"],
            0,
            1,
        )

        group["selection_sensor_rank"] = np.where(
            group["sensor_matches_positive_at_site"],
            0,
            1,
        )

        group["selection_cloud_rank"] = (
            group["CLOUD_COVER"]
            .fillna(999)
        )

        group["selection_priority_rank"] = (
            group["candidate_priority_score"]
            .fillna(9999)
        )

        group = group.sort_values(
            by=[
                "selection_same_wrs_rank",
                "selection_sensor_rank",
                "selection_cloud_rank",
                "selection_priority_rank",
                "candidate_time_utc",
            ],
            ascending=True,
        )

        representative = group.iloc[0].copy()

        representative["overpass_group_size"] = len(group)
        representative["overpass_representative"] = True

        if len(group) == 1:
            representative["representative_reason"] = (
                "Only product in this overpass group."
            )
        else:
            representative["representative_reason"] = (
                "Selected from adjacent WRS products by preferring "
                "the reference WRS footprint, matching sensor, "
                "lower cloud cover, and lower priority score."
            )

        representative_rows.append(representative)

        if len(group) > 1:
            excluded = group.iloc[1:].copy()

            excluded["overpass_group_size"] = len(group)
            excluded["overpass_representative"] = False
            excluded["excluded_reason"] = (
                "Adjacent WRS product from the same independent "
                "Landsat overpass."
            )

            excluded["kept_landsat_product_id"] = (
                representative.get("LANDSAT_PRODUCT_ID")
            )

            excluded["kept_candidate_time_utc"] = (
                representative.get("candidate_time_utc")
            )

            excluded_rows.append(excluded)

    unique_overpasses = pd.DataFrame(
        representative_rows
    )

    if excluded_rows:
        excluded_overlaps = pd.concat(
            excluded_rows,
            ignore_index=True,
            sort=False,
        )
    else:
        excluded_overlaps = pd.DataFrame()

    unique_overpasses = unique_overpasses.sort_values(
        by=[
            "site_key",
            "candidate_time_utc",
            "landsat_sensor",
        ]
    ).reset_index(drop=True)

    unique_overpasses[
        "unique_overpass_rank_within_site"
    ] = (
        unique_overpasses.groupby(
            "site_key"
        ).cumcount()
        + 1
    )

    UNIQUE_OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    unique_overpasses.to_csv(
        UNIQUE_OUTPUT_CSV,
        index=False,
    )

    excluded_overlaps.to_csv(
        EXCLUDED_OUTPUT_CSV,
        index=False,
    )

    print("=" * 100)
    print("LANDSAT UNIQUE CANDIDATE OVERPASSES")
    print("=" * 100)

    print(
        f"\nOriginal new Landsat products: "
        f"{len(new_candidates)}"
    )

    print(
        f"Independent candidate overpasses: "
        f"{len(unique_overpasses)}"
    )

    print(
        f"Adjacent WRS products excluded: "
        f"{len(excluded_overlaps)}"
    )

    print("\nOverpass group-size counts:")
    print(
        unique_overpasses[
            "overpass_group_size"
        ].value_counts().sort_index()
    )

    print("\nUnique overpasses by site and sensor:")
    print(
        pd.crosstab(
            unique_overpasses[
                "site_name_normalized"
            ],
            unique_overpasses[
                "landsat_sensor"
            ],
            margins=True,
        )
    )

    multi_product = unique_overpasses[
        unique_overpasses[
            "overpass_group_size"
        ] > 1
    ].copy()

    print("\nOverpasses containing multiple WRS products:")

    if len(multi_product) == 0:
        print("None")
    else:
        display_columns = [
            column
            for column in [
                "overpass_id",
                "site_name_normalized",
                "candidate_time_utc",
                "landsat_sensor",
                "LANDSAT_PRODUCT_ID",
                "WRS_PATH",
                "WRS_ROW",
                "CLOUD_COVER",
                "overpass_group_size",
            ]
            if column in multi_product.columns
        ]

        print(
            multi_product[
                display_columns
            ].to_string(index=False)
        )

    print("\nUnique candidate overpass list:")

    display_columns = [
        column
        for column in [
            "unique_overpass_rank_within_site",
            "overpass_id",
            "site_name_normalized",
            "candidate_time_utc",
            "landsat_sensor",
            "LANDSAT_PRODUCT_ID",
            "WRS_PATH",
            "WRS_ROW",
            "CLOUD_COVER",
            "same_reference_wrs",
            "overpass_group_size",
        ]
        if column in unique_overpasses.columns
    ]

    print(
        unique_overpasses[
            display_columns
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(UNIQUE_OUTPUT_CSV)
    print(EXCLUDED_OUTPUT_CSV)


if __name__ == "__main__":
    main()
