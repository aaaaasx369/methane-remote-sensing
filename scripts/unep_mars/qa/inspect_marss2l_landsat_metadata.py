from pathlib import Path

import pandas as pd


INPUT = Path(
    "raw_data/MARS-S2L/"
    "validated_images_all.csv"
)

OUTPUT = Path(
    "outputs/219_marss2l_landsat_metadata_summary.csv"
)


def parse_bool(series):
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

    df["isplume_parsed"] = parse_bool(
        df["isplume"]
    )

    df["ch4_fluxrate"] = pd.to_numeric(
        df["ch4_fluxrate"],
        errors="coerce",
    )

    df["satellite"] = (
        df["satellite"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["observability"] = (
        df["observability"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    landsat = df[
        df["satellite"].isin([
            "LC08",
            "LC09",
        ])
    ].copy()

    clear = landsat[
        landsat["observability"]
        == "clear"
    ].copy()

    positives = clear[
        clear["isplume_parsed"]
        == True
    ].copy()

    negatives = clear[
        clear["isplume_parsed"]
        == False
    ].copy()

    high_emission = positives[
        positives["ch4_fluxrate"]
        >= 1000
    ].copy()

    development_sites = set(
        df.loc[
            df["split_name"]
            .astype(str)
            .isin([
                "train_2023",
                "val_2023",
            ]),
            "id_location",
        ]
        .dropna()
        .astype(str)
    )

    unseen_high_emission = high_emission[
        ~high_emission["id_location"]
        .astype(str)
        .isin(development_sites)
    ].copy()

    summary = pd.DataFrame([
        {
            "all_dataset_images":
                len(df),
            "landsat_images":
                len(landsat),
            "clear_landsat_images":
                len(clear),
            "clear_landsat_positives":
                len(positives),
            "clear_landsat_negatives":
                len(negatives),
            "clear_landsat_positive_sites":
                positives[
                    "id_location"
                ].nunique(),
            "landsat_positive_ge1000":
                len(high_emission),
            "landsat_positive_ge1000_sites":
                high_emission[
                    "id_location"
                ].nunique(),
            "unseen_site_ge1000_positives":
                len(unseen_high_emission),
            "unseen_site_ge1000_sites":
                unseen_high_emission[
                    "id_location"
                ].nunique(),
        }
    ])

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 100)
    print("MARS-S2L LANDSAT SUMMARY")
    print("=" * 100)

    print(
        summary.to_string(
            index=False
        )
    )

    print("\nLandsat by satellite:")
    print(
        clear["satellite"]
        .value_counts()
    )

    print("\nLandsat by split:")
    print(
        pd.crosstab(
            clear["split_name"],
            clear["isplume_parsed"],
            margins=True,
        )
    )

    print("\nHigh-emission positives by country:")
    print(
        high_emission["country"]
        .value_counts()
        .head(20)
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
