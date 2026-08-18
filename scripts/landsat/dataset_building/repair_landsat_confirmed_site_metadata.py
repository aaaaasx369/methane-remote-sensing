from pathlib import Path

import pandas as pd


CONFIRMED_INPUT = Path(
    "outputs/57_landsat_final_confirmed_features.csv"
)

CANDIDATE_INPUT = Path(
    "outputs/61_landsat_matched_negative_candidates.csv"
)

REPAIRED_OUTPUT = Path(
    "outputs/396_landsat_final_confirmed_features_site_repaired_v1.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/397_landsat_site_metadata_repair_audit_v1.csv"
)


def clean_text(series):
    result = series.astype("string").str.strip()

    return result.mask(
        result.str.lower().isin(
            ["", "nan", "none", "<na>"]
        )
    )


def main():
    confirmed = pd.read_csv(
        CONFIRMED_INPUT,
        low_memory=False,
    )

    candidates = pd.read_csv(
        CANDIDATE_INPUT,
        low_memory=False,
    )

    required_confirmed = [
        "raster_group_id",
        "label",
    ]

    required_candidates = [
        "existing_raster_group_id",
        "site_key",
        "site_name_normalized",
        "site_lat",
        "site_lon",
    ]

    missing_confirmed = [
        column
        for column in required_confirmed
        if column not in confirmed.columns
    ]

    missing_candidates = [
        column
        for column in required_candidates
        if column not in candidates.columns
    ]

    if missing_confirmed:
        raise KeyError(
            "Confirmed table 缺少："
            + ", ".join(missing_confirmed)
        )

    if missing_candidates:
        raise KeyError(
            "Candidate table 缺少："
            + ", ".join(missing_candidates)
        )

    candidates[
        "existing_raster_group_id"
    ] = clean_text(
        candidates[
            "existing_raster_group_id"
        ]
    )

    existing = candidates.dropna(
        subset=[
            "existing_raster_group_id",
        ]
    ).copy()

    # 檢查同一 raster group 是否被指派到不同場址。
    site_conflicts = (
        existing.groupby(
            "existing_raster_group_id"
        )["site_name_normalized"]
        .nunique(
            dropna=True
        )
    )

    site_conflicts = site_conflicts[
        site_conflicts.gt(1)
    ]

    if not site_conflicts.empty:
        raise RuntimeError(
            "同一 raster_group_id 對應多個場址：\n"
            + site_conflicts.to_string()
        )

    lookup = (
        existing.sort_values(
            [
                "existing_raster_group_id",
                "existing_time_difference_seconds",
            ],
            na_position="last",
        )
        .drop_duplicates(
            subset=[
                "existing_raster_group_id",
            ],
            keep="first",
        )
        [
            [
                "existing_raster_group_id",
                "site_key",
                "site_name_normalized",
                "site_lat",
                "site_lon",
                "landsat_sensor",
                "LANDSAT_PRODUCT_ID",
                "candidate_time_utc",
            ]
        ]
        .rename(
            columns={
                "existing_raster_group_id":
                    "raster_group_id",

                "site_key":
                    "lookup_site_key",

                "site_name_normalized":
                    "lookup_site_name",

                "site_lat":
                    "lookup_site_lat",

                "site_lon":
                    "lookup_site_lon",

                "landsat_sensor":
                    "lookup_landsat_sensor",

                "LANDSAT_PRODUCT_ID":
                    "lookup_landsat_product_id",

                "candidate_time_utc":
                    "lookup_candidate_time_utc",
            }
        )
    )

    repaired = confirmed.merge(
        lookup,
        on="raster_group_id",
        how="left",
        validate="one_to_one",
    )

    original_site_column = None

    for candidate in [
        "site",
        "site_name",
        "site_name_normalized",
        "release_site",
    ]:
        if candidate in repaired.columns:
            original_site_column = candidate
            break

    if original_site_column is None:
        repaired[
            "original_site_value"
        ] = pd.NA
    else:
        repaired[
            "original_site_value"
        ] = clean_text(
            repaired[
                original_site_column
            ]
        )

    repaired[
        "site"
    ] = repaired[
        "original_site_value"
    ].fillna(
        clean_text(
            repaired[
                "lookup_site_name"
            ]
        )
    )

    repaired[
        "site_key"
    ] = clean_text(
        repaired.get(
            "site_key",
            pd.Series(
                pd.NA,
                index=repaired.index,
            ),
        )
    ).fillna(
        clean_text(
            repaired[
                "lookup_site_key"
            ]
        )
    )

    # 補場址座標，但不覆蓋原本已有的有效值。
    lat_column = next(
        (
            column
            for column in [
                "lat",
                "latitude",
                "site_lat",
                "source_lat",
                "release_lat",
            ]
            if column in repaired.columns
        ),
        None,
    )

    lon_column = next(
        (
            column
            for column in [
                "lon",
                "longitude",
                "site_lon",
                "source_lon",
                "release_lon",
            ]
            if column in repaired.columns
        ),
        None,
    )

    if lat_column is None:
        repaired["site_lat"] = pd.to_numeric(
            repaired[
                "lookup_site_lat"
            ],
            errors="coerce",
        )
    else:
        repaired[lat_column] = pd.to_numeric(
            repaired[lat_column],
            errors="coerce",
        ).fillna(
            pd.to_numeric(
                repaired[
                    "lookup_site_lat"
                ],
                errors="coerce",
            )
        )

    if lon_column is None:
        repaired["site_lon"] = pd.to_numeric(
            repaired[
                "lookup_site_lon"
            ],
            errors="coerce",
        )
    else:
        repaired[lon_column] = pd.to_numeric(
            repaired[lon_column],
            errors="coerce",
        ).fillna(
            pd.to_numeric(
                repaired[
                    "lookup_site_lon"
                ],
                errors="coerce",
            )
        )

    repaired[
        "site_resolution_source"
    ] = "original_confirmed_table"

    repaired.loc[
        repaired[
            "original_site_value"
        ].isna()
        & repaired[
            "site"
        ].notna(),
        "site_resolution_source",
    ] = (
        "candidate_existing_raster_group_id_lookup"
    )

    repaired.loc[
        repaired["site"].isna(),
        "site_resolution_source",
    ] = "unresolved"

    repaired["label"] = pd.to_numeric(
        repaired["label"],
        errors="raise",
    ).astype(int)

    unresolved = repaired[
        repaired["site"].isna()
    ].copy()

    if not unresolved.empty:
        raise RuntimeError(
            "仍有場址無法補齊：\n"
            + unresolved[
                [
                    "raster_group_id",
                    "label",
                ]
            ].to_string(index=False)
        )

    if len(repaired) != 9:
        raise RuntimeError(
            "修復後應有 9 筆，"
            f"實際為 {len(repaired)}。"
        )

    positive = repaired[
        repaired["label"].eq(1)
    ].copy()

    audit_columns = [
        "raster_group_id",
        "label",
        "original_site_value",
        "lookup_site_name",
        "site",
        "site_key",
        "site_resolution_source",
        "lookup_landsat_product_id",
        "lookup_candidate_time_utc",
    ]

    audit = repaired[
        audit_columns
    ].copy()

    repaired.to_csv(
        REPAIRED_OUTPUT,
        index=False,
    )

    audit.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    print("=" * 105)
    print("LANDSAT SITE METADATA REPAIR")
    print("=" * 105)

    print(
        "\nInput rows:",
        len(confirmed),
    )

    print(
        "Repaired rows:",
        len(repaired),
    )

    print(
        "Unresolved sites:",
        int(
            repaired["site"].isna().sum()
        ),
    )

    print(
        "\nSite resolution source:"
    )

    print(
        repaired[
            "site_resolution_source"
        ].value_counts()
    )

    print(
        "\nAll rows by site and label:"
    )

    print(
        pd.crosstab(
            repaired["site"],
            repaired["label"],
            margins=True,
        )
    )

    print(
        "\nPositive scenes by site:"
    )

    print(
        positive[
            "site"
        ].value_counts()
    )

    print(
        "\nPositive scene mapping:"
    )

    print(
        positive[
            [
                "raster_group_id",
                "site",
                "site_resolution_source",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(REPAIRED_OUTPUT)
    print(AUDIT_OUTPUT)


if __name__ == "__main__":
    main()
