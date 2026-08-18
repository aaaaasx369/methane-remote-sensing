from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "raw_data/MARS-S2L/validated_images_all.csv"
)

SITE_SUMMARY_OUTPUT = Path(
    "outputs/220_marss2l_unseen_site_summary.csv"
)

ELIGIBLE_SITE_OUTPUT = Path(
    "outputs/221_marss2l_external_eligible_sites.csv"
)

CANDIDATE_IMAGE_OUTPUT = Path(
    "outputs/222_marss2l_external_candidate_images.csv"
)


HIGH_EMISSION_THRESHOLD_KG_H = 1000.0

MIN_CALIBRATION_NEGATIVES = 5
MIN_TEST_NEGATIVES = 3
MIN_TOTAL_NEGATIVES = (
    MIN_CALIBRATION_NEGATIVES
    + MIN_TEST_NEGATIVES
)


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
    )


def first_existing_column(
    dataframe,
    candidates,
):
    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required = [
        "satellite",
        "isplume",
        "observability",
        "split_name",
        "id_location",
        "ch4_fluxrate",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}"
        )

    df["satellite_normalized"] = (
        df["satellite"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["observability_normalized"] = (
        df["observability"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["split_normalized"] = (
        df["split_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["site_key"] = (
        df["id_location"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["isplume_parsed"] = (
        parse_boolean(
            df["isplume"]
        )
    )

    df["ch4_fluxrate"] = pd.to_numeric(
        df["ch4_fluxrate"],
        errors="coerce",
    )

    date_column = first_existing_column(
        df,
        [
            "tile_date",
            "datetime",
            "acquisition_datetime",
            "date",
        ],
    )

    if date_column is not None:
        df["acquisition_datetime_utc"] = (
            pd.to_datetime(
                df[date_column],
                errors="coerce",
                utc=True,
            )
        )
    else:
        df["acquisition_datetime_utc"] = (
            pd.NaT
        )

    tile_column = first_existing_column(
        df,
        [
            "tile",
            "product_id",
            "landsat_product_id",
            "scene_id",
        ],
    )

    if tile_column is not None:
        df["landsat_tile"] = (
            df[tile_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        df["landsat_tile"] = ""

    # 建立影像鍵值，避免同一張影像因多個 plume polygon 重複。
    df["image_key"] = (
        df["site_key"]
        + "|"
        + df["satellite_normalized"]
        + "|"
        + df["landsat_tile"]
        + "|"
        + df[
            "acquisition_datetime_utc"
        ].astype(str)
    )

    landsat_clear = df[
        df["satellite_normalized"].isin([
            "LC08",
            "LC09",
        ])
        & df[
            "observability_normalized"
        ].eq("clear")
        & df["site_key"].ne("")
        & df[
            "isplume_parsed"
        ].notna()
    ].copy()

    # 只要場址曾出現在 train 或 validation，
    # 就不能當作完全未見的新場址。
    development_sites = set(
        landsat_clear.loc[
            landsat_clear[
                "split_normalized"
            ].isin([
                "train_2023",
                "val_2023",
            ]),
            "site_key",
        ]
    )

    # 主要外部測試只使用官方 test split。
    external = landsat_clear[
        landsat_clear[
            "split_normalized"
        ].eq("test_2023")
        & ~landsat_clear[
            "site_key"
        ].isin(
            development_sites
        )
    ].copy()

    external[
        "benchmark_role"
    ] = np.select(
        [
            external[
                "isplume_parsed"
            ].eq(True)
            & external[
                "ch4_fluxrate"
            ].ge(
                HIGH_EMISSION_THRESHOLD_KG_H
            ),

            external[
                "isplume_parsed"
            ].eq(False),
        ],
        [
            "high_emission_positive",
            "no_plume_negative",
        ],
        default="excluded",
    )

    benchmark = external[
        external[
            "benchmark_role"
        ].isin([
            "high_emission_positive",
            "no_plume_negative",
        ])
    ].copy()

    # 每個 site/影像/角色只留一筆。
    benchmark = (
        benchmark.sort_values(
            [
                "site_key",
                "benchmark_role",
                "ch4_fluxrate",
            ],
            ascending=[
                True,
                True,
                False,
            ],
            na_position="last",
        )
        .drop_duplicates(
            subset=[
                "image_key",
                "benchmark_role",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    positives = benchmark[
        benchmark[
            "benchmark_role"
        ].eq(
            "high_emission_positive"
        )
    ].copy()

    negatives = benchmark[
        benchmark[
            "benchmark_role"
        ].eq(
            "no_plume_negative"
        )
    ].copy()

    positive_summary = (
        positives.groupby(
            "site_key"
        )
        .agg(
            high_emission_positive_count=(
                "image_key",
                "nunique",
            ),
            positive_landsat8_count=(
                "satellite_normalized",
                lambda values:
                    int(
                        (
                            values == "LC08"
                        ).sum()
                    ),
            ),
            positive_landsat9_count=(
                "satellite_normalized",
                lambda values:
                    int(
                        (
                            values == "LC09"
                        ).sum()
                    ),
            ),
            minimum_positive_flux_kg_h=(
                "ch4_fluxrate",
                "min",
            ),
            median_positive_flux_kg_h=(
                "ch4_fluxrate",
                "median",
            ),
            maximum_positive_flux_kg_h=(
                "ch4_fluxrate",
                "max",
            ),
            first_positive_time=(
                "acquisition_datetime_utc",
                "min",
            ),
            last_positive_time=(
                "acquisition_datetime_utc",
                "max",
            ),
        )
        .reset_index()
    )

    negative_summary = (
        negatives.groupby(
            "site_key"
        )
        .agg(
            clear_negative_count=(
                "image_key",
                "nunique",
            ),
            negative_landsat8_count=(
                "satellite_normalized",
                lambda values:
                    int(
                        (
                            values == "LC08"
                        ).sum()
                    ),
            ),
            negative_landsat9_count=(
                "satellite_normalized",
                lambda values:
                    int(
                        (
                            values == "LC09"
                        ).sum()
                    ),
            ),
            first_negative_time=(
                "acquisition_datetime_utc",
                "min",
            ),
            last_negative_time=(
                "acquisition_datetime_utc",
                "max",
            ),
        )
        .reset_index()
    )

    site_summary = (
        positive_summary.merge(
            negative_summary,
            on="site_key",
            how="left",
            validate="one_to_one",
        )
    )

    count_columns = [
        "clear_negative_count",
        "negative_landsat8_count",
        "negative_landsat9_count",
    ]

    for column in count_columns:
        site_summary[column] = (
            site_summary[column]
            .fillna(0)
            .astype(int)
        )

    # 加入國家名稱。
    if "country" in benchmark.columns:
        countries = (
            benchmark.groupby(
                "site_key"
            )["country"]
            .first()
            .reset_index()
        )

        site_summary = (
            site_summary.merge(
                countries,
                on="site_key",
                how="left",
                validate="one_to_one",
            )
        )

    site_summary[
        "minimum_external_eligible"
    ] = (
        site_summary[
            "high_emission_positive_count"
        ].ge(1)
        & site_summary[
            "clear_negative_count"
        ].ge(
            MIN_TOTAL_NEGATIVES
        )
    )

    site_summary[
        "strong_external_eligible"
    ] = (
        site_summary[
            "high_emission_positive_count"
        ].ge(2)
        & site_summary[
            "clear_negative_count"
        ].ge(10)
    )

    site_summary[
        "eligibility_class"
    ] = np.select(
        [
            site_summary[
                "strong_external_eligible"
            ],
            site_summary[
                "minimum_external_eligible"
            ],
        ],
        [
            "strong",
            "minimum",
        ],
        default="insufficient_negatives",
    )

    site_summary = site_summary.sort_values(
        [
            "strong_external_eligible",
            "minimum_external_eligible",
            "high_emission_positive_count",
            "clear_negative_count",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    eligible_sites = site_summary[
        site_summary[
            "minimum_external_eligible"
        ]
    ].copy()

    eligible_keys = set(
        eligible_sites["site_key"]
    )

    candidate_images = benchmark[
        benchmark[
            "site_key"
        ].isin(
            eligible_keys
        )
    ].copy()

    candidate_images[
        "frozen_high_emission_threshold_kg_h"
    ] = HIGH_EMISSION_THRESHOLD_KG_H

    candidate_images[
        "external_dataset"
    ] = "MARS-S2L"

    candidate_images[
        "ground_truth_type"
    ] = np.where(
        candidate_images[
            "benchmark_role"
        ].eq(
            "high_emission_positive"
        ),
        "human_confirmed_landsat_plume_"
        "satellite_estimated_flux",
        "human_reviewed_clear_no_plume",
    )

    SITE_SUMMARY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    site_summary.to_csv(
        SITE_SUMMARY_OUTPUT,
        index=False,
    )

    eligible_sites.to_csv(
        ELIGIBLE_SITE_OUTPUT,
        index=False,
    )

    candidate_images.to_csv(
        CANDIDATE_IMAGE_OUTPUT,
        index=False,
    )

    print("=" * 108)
    print("MARS-S2L UNSEEN-SITE SCREENING")
    print("=" * 108)

    print(
        "\nDevelopment-site count:",
        len(development_sites),
    )

    print(
        "Unseen test high-emission "
        "positive images:",
        len(positives),
    )

    print(
        "Unseen test positive sites:",
        positives[
            "site_key"
        ].nunique(),
    )

    print(
        "Unseen clear no-plume images:",
        len(negatives),
    )

    print(
        "\nMinimum eligible sites "
        f"(>=1 positive and >="
        f"{MIN_TOTAL_NEGATIVES} negatives):",
        len(eligible_sites),
    )

    print(
        "Strong eligible sites "
        "(>=2 positives and >=10 negatives):",
        int(
            site_summary[
                "strong_external_eligible"
            ].sum()
        ),
    )

    print("\nEligibility classes:")
    print(
        site_summary[
            "eligibility_class"
        ].value_counts()
    )

    if not eligible_sites.empty:
        print("\nEligible site totals:")
        print(
            "High-emission positives:",
            int(
                eligible_sites[
                    "high_emission_positive_count"
                ].sum()
            ),
        )

        print(
            "Available clear negatives:",
            int(
                eligible_sites[
                    "clear_negative_count"
                ].sum()
            ),
        )

        display_columns = [
            "site_key",
            "country",
            "eligibility_class",
            "high_emission_positive_count",
            "clear_negative_count",
            "positive_landsat8_count",
            "positive_landsat9_count",
            "minimum_positive_flux_kg_h",
            "median_positive_flux_kg_h",
            "maximum_positive_flux_kg_h",
        ]

        available = [
            column
            for column in display_columns
            if column
            in eligible_sites.columns
        ]

        print("\nTop eligible sites:")
        print(
            eligible_sites[
                available
            ].head(30).to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.2f}",
            )
        )

    print("\nSaved:")
    print(SITE_SUMMARY_OUTPUT)
    print(ELIGIBLE_SITE_OUTPUT)
    print(CANDIDATE_IMAGE_OUTPUT)


if __name__ == "__main__":
    main()
