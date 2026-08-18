from pathlib import Path
import pandas as pd


INPUT_CSV = Path(
    "outputs/57_landsat_final_confirmed_features.csv"
)

OUTPUT_CSV = Path(
    "outputs/59_landsat_negative_search_inputs.csv"
)


LATITUDE_CANDIDATES = [
    "latitude",
    "lat",
    "site_latitude",
    "site_lat",
    "release_latitude",
    "release_lat",
    "source_latitude",
    "source_lat",
]

LONGITUDE_CANDIDATES = [
    "longitude",
    "lon",
    "lng",
    "site_longitude",
    "site_lon",
    "release_longitude",
    "release_lon",
    "source_longitude",
    "source_lon",
]


def first_existing_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    print("=" * 90)
    print("LANDSAT MATCHED-NEGATIVE SEARCH INPUT AUDIT")
    print("=" * 90)

    print(f"\nInput file: {INPUT_CSV}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nAll columns:")
    for number, column in enumerate(
        df.columns,
        start=1,
    ):
        print(f"{number:03d}. {column}")

    latitude_column = first_existing_column(
        df.columns,
        LATITUDE_CANDIDATES,
    )

    longitude_column = first_existing_column(
        df.columns,
        LONGITUDE_CANDIDATES,
    )

    print("\nDetected coordinate columns:")
    print(f"Latitude:  {latitude_column}")
    print(f"Longitude: {longitude_column}")

    if latitude_column is not None:
        df[latitude_column] = pd.to_numeric(
            df[latitude_column],
            errors="coerce",
        )

    if longitude_column is not None:
        df[longitude_column] = pd.to_numeric(
            df[longitude_column],
            errors="coerce",
        )

    if "landsat_image_time" in df.columns:
        df["landsat_time_parsed"] = pd.to_datetime(
            df["landsat_image_time"],
            errors="coerce",
            utc=True,
        )
    else:
        df["landsat_time_parsed"] = pd.NaT

    preferred_columns = [
        "raster_group_id",
        "label",
        "site_name",
        "event_id",
        "landsat_sensor",
        "landsat_image_time",
        "landsat_time_parsed",
        "landsat_cloud_cover",
        latitude_column,
        longitude_column,
    ]

    preferred_columns = [
        column
        for column in preferred_columns
        if column is not None
        and column in df.columns
    ]

    print("\nConfirmed scene list:")
    print(
        df[preferred_columns]
        .to_string(index=False)
    )

    print("\nLabel counts:")
    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    if "site_name" in df.columns:
        print("\nSite counts:")
        print(
            df["site_name"]
            .fillna("[missing]")
            .value_counts(dropna=False)
        )

        print("\nLabel by site:")
        print(
            pd.crosstab(
                df["site_name"].fillna("[missing]"),
                df["label"],
                margins=True,
            )
        )

    if (
        latitude_column is not None
        and longitude_column is not None
    ):
        valid_coordinates = df[
            df[latitude_column].notna()
            & df[longitude_column].notna()
        ].copy()

        print(
            "\nRows with valid coordinates:",
            len(valid_coordinates),
        )

        print("\nUnique coordinate pairs:")
        coordinate_columns = [
            latitude_column,
            longitude_column,
        ]

        if "site_name" in valid_coordinates.columns:
            coordinate_columns.insert(
                0,
                "site_name",
            )

        print(
            valid_coordinates[
                coordinate_columns
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

    positive_df = df[
        df["label"] == 1
    ].copy()

    negative_df = df[
        df["label"] == 0
    ].copy()

    print("\nConfirmed positive scenes:")
    print(
        positive_df[
            preferred_columns
        ].to_string(index=False)
    )

    print("\nExisting confirmed negative scenes:")
    print(
        negative_df[
            preferred_columns
        ].to_string(index=False)
    )

    # 建立後續搜尋用的一列一張正樣本表。
    output_columns = [
        column
        for column in [
            "raster_group_id",
            "site_name",
            latitude_column,
            longitude_column,
            "landsat_sensor",
            "landsat_image_time",
            "landsat_cloud_cover",
            "pixel_hash",
        ]
        if column is not None
        and column in positive_df.columns
    ]

    search_df = positive_df[
        output_columns
    ].copy()

    search_df["reference_positive_year"] = (
        positive_df["landsat_time_parsed"].dt.year.values
    )

    search_df["reference_positive_month"] = (
        positive_df["landsat_time_parsed"].dt.month.values
    )

    search_df["reference_positive_day"] = (
        positive_df["landsat_time_parsed"].dt.day.values
    )

    search_df["recommended_search_start"] = (
        positive_df["landsat_time_parsed"]
        - pd.Timedelta(days=45)
    ).dt.strftime("%Y-%m-%d").values

    search_df["recommended_search_end"] = (
        positive_df["landsat_time_parsed"]
        + pd.Timedelta(days=45)
    ).dt.strftime("%Y-%m-%d").values

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    search_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\n" + "=" * 90)
    print("SEARCH-INPUT SUMMARY")
    print("=" * 90)

    print(
        f"\nPositive reference scenes: "
        f"{len(search_df)}"
    )

    print(
        f"Rows with usable latitude: "
        f"{search_df[latitude_column].notna().sum()}"
        if latitude_column in search_df.columns
        else "Rows with usable latitude: 0"
    )

    print(
        f"Rows with usable longitude: "
        f"{search_df[longitude_column].notna().sum()}"
        if longitude_column in search_df.columns
        else "Rows with usable longitude: 0"
    )

    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
