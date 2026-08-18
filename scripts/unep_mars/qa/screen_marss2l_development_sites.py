from pathlib import Path

import pandas as pd


INPUT = Path(
    "raw_data/MARS-S2L/validated_images_all.csv"
)

SITE_OUTPUT = Path(
    "outputs/250_marss2l_development_site_summary.csv"
)

IMAGE_OUTPUT = Path(
    "outputs/251_marss2l_development_candidate_images.csv"
)

ELIGIBLE_OUTPUT = Path(
    "outputs/252_marss2l_development_eligible_sites.csv"
)

HIGH_EMISSION_THRESHOLD = 1000.0
MIN_POSITIVES = 1
MIN_NEGATIVES = 10


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


def main():
    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    df["satellite_normalized"] = (
        df["satellite"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["split_normalized"] = (
        df["split_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["observability_normalized"] = (
        df["observability"]
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

    df["isplume_parsed"] = parse_boolean(
        df["isplume"]
    )

    df["ch4_fluxrate"] = pd.to_numeric(
        df["ch4_fluxrate"],
        errors="coerce",
    )

    df["acquisition_datetime_utc"] = (
        pd.to_datetime(
            df["tile_date"],
            errors="coerce",
            utc=True,
        )
    )

    development = df[
        df["satellite_normalized"].isin([
            "LC08",
            "LC09",
        ])
        & df["split_normalized"].isin([
            "train_2023",
            "val_2023",
        ])
        & df[
            "observability_normalized"
        ].eq("clear")
        & df["site_key"].ne("")
        & df["isplume_parsed"].notna()
    ].copy()

    development["benchmark_role"] = (
        "excluded"
    )

    development.loc[
        development["isplume_parsed"].eq(True)
        & development["ch4_fluxrate"].ge(
            HIGH_EMISSION_THRESHOLD
        ),
        "benchmark_role",
    ] = "high_emission_positive"

    development.loc[
        development["isplume_parsed"].eq(False),
        "benchmark_role",
    ] = "no_plume_negative"

    development = development[
        development["benchmark_role"].isin([
            "high_emission_positive",
            "no_plume_negative",
        ])
    ].copy()

    development["image_key"] = (
        development["site_key"]
        + "|"
        + development["tile"]
        .fillna("")
        .astype(str)
        + "|"
        + development[
            "acquisition_datetime_utc"
        ].astype(str)
        + "|"
        + development["benchmark_role"]
    )

    development = (
        development.sort_values(
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
            subset=["image_key"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    summary = (
        development.groupby(
            [
                "split_normalized",
                "site_key",
            ]
        )
        .agg(
            positive_count=(
                "benchmark_role",
                lambda values:
                    int(
                        (
                            values
                            == "high_emission_positive"
                        ).sum()
                    ),
            ),
            negative_count=(
                "benchmark_role",
                lambda values:
                    int(
                        (
                            values
                            == "no_plume_negative"
                        ).sum()
                    ),
            ),
            landsat8_count=(
                "satellite_normalized",
                lambda values:
                    int(
                        (
                            values == "LC08"
                        ).sum()
                    ),
            ),
            landsat9_count=(
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
                lambda values:
                    values[
                        values
                        >= HIGH_EMISSION_THRESHOLD
                    ].min(),
            ),
            median_positive_flux_kg_h=(
                "ch4_fluxrate",
                lambda values:
                    values[
                        values
                        >= HIGH_EMISSION_THRESHOLD
                    ].median(),
            ),
            maximum_positive_flux_kg_h=(
                "ch4_fluxrate",
                lambda values:
                    values[
                        values
                        >= HIGH_EMISSION_THRESHOLD
                    ].max(),
            ),
        )
        .reset_index()
    )

    summary["eligible"] = (
        summary["positive_count"].ge(
            MIN_POSITIVES
        )
        & summary["negative_count"].ge(
            MIN_NEGATIVES
        )
    )

    eligible = summary[
        summary["eligible"]
    ].copy()

    eligible_keys = set(
        eligible[
            [
                "split_normalized",
                "site_key",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    development["eligible_site"] = [
        (
            split_name,
            site_key,
        ) in eligible_keys
        for split_name, site_key
        in zip(
            development["split_normalized"],
            development["site_key"],
        )
    ]

    train_sites = set(
        summary.loc[
            summary["split_normalized"]
            .eq("train_2023"),
            "site_key",
        ]
    )

    validation_sites = set(
        summary.loc[
            summary["split_normalized"]
            .eq("val_2023"),
            "site_key",
        ]
    )

    overlap = sorted(
        train_sites
        & validation_sites
    )

    SITE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SITE_OUTPUT,
        index=False,
    )

    development.to_csv(
        IMAGE_OUTPUT,
        index=False,
    )

    eligible.to_csv(
        ELIGIBLE_OUTPUT,
        index=False,
    )

    print("=" * 105)
    print("MARS-S2L DEVELOPMENT SITE SCREEN")
    print("=" * 105)

    print("\nEligible-site rule:")
    print(
        f">={MIN_POSITIVES} positive and "
        f">={MIN_NEGATIVES} negatives"
    )

    print("\nAll development images:")
    print(
        development[
            "benchmark_role"
        ].value_counts()
    )

    print("\nAll sites by split:")
    print(
        summary[
            "split_normalized"
        ].value_counts()
    )

    print("\nEligible sites by split:")
    print(
        eligible[
            "split_normalized"
        ].value_counts()
    )

    print("\nEligible image totals:")
    eligible_images = development[
        development["eligible_site"]
    ]

    print(
        eligible_images[
            "benchmark_role"
        ].value_counts()
    )

    print(
        "\nTrain/validation site overlap:",
        len(overlap),
    )

    if overlap:
        print(
            "First overlapping sites:",
            overlap[:20],
        )

    print("\nTop eligible sites:")
    print(
        eligible.sort_values(
            [
                "positive_count",
                "negative_count",
            ],
            ascending=[
                False,
                False,
            ],
        ).head(30).to_string(
            index=False,
            float_format=lambda value:
                f"{value:.2f}",
        )
    )

    print("\nSaved:")
    print(SITE_OUTPUT)
    print(IMAGE_OUTPUT)
    print(ELIGIBLE_OUTPUT)


if __name__ == "__main__":
    main()
