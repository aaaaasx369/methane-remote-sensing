from pathlib import Path
import pandas as pd


FILES = [
    Path("outputs/12_dataset_candidate_events_with_latlon.csv"),
    Path("outputs/final_controlled_release_satellite_availability.csv"),
    Path("outputs/15_methaneair_s2_landsat_availability.csv"),
    Path("outputs/07_unique_overpass_events.csv"),
    Path("outputs/09_final_unique_overpass_events.csv"),
    Path("outputs/11_availability_summary.csv"),
]

SITE_KEYWORDS = [
    "site",
    "location",
    "facility",
    "campaign",
    "city",
    "field",
    "source_name",
]

LAT_KEYWORDS = [
    "latitude",
    "lat",
]

LON_KEYWORDS = [
    "longitude",
    "lon",
    "lng",
]

LANDSAT_KEYWORDS = [
    "landsat",
    "l8",
    "l9",
]

LABEL_KEYWORDS = [
    "label",
    "target",
    "release",
    "positive",
    "emission",
]


def find_columns(columns, keywords):
    matches = []

    for column in columns:
        lower = column.lower()

        if any(
            keyword == lower
            or keyword in lower
            for keyword in keywords
        ):
            matches.append(column)

    return matches


def first_numeric_column(df, candidates):
    for column in candidates:
        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if converted.notna().sum() > 0:
            return column

    return None


for path in FILES:
    print("\n" + "=" * 110)
    print(path)
    print("=" * 110)

    if not path.exists():
        print("[MISSING]")
        continue

    try:
        df = pd.read_csv(
            path,
            low_memory=False,
        )
    except Exception as error:
        print("[READ ERROR]", error)
        continue

    print("Shape:", df.shape)

    site_columns = find_columns(
        df.columns,
        SITE_KEYWORDS,
    )

    latitude_columns = find_columns(
        df.columns,
        LAT_KEYWORDS,
    )

    longitude_columns = find_columns(
        df.columns,
        LON_KEYWORDS,
    )

    landsat_columns = find_columns(
        df.columns,
        LANDSAT_KEYWORDS,
    )

    label_columns = find_columns(
        df.columns,
        LABEL_KEYWORDS,
    )

    print("Possible site columns:")
    print(site_columns)

    print("Possible latitude columns:")
    print(latitude_columns)

    print("Possible longitude columns:")
    print(longitude_columns)

    print("Possible Landsat columns:")
    print(landsat_columns)

    print("Possible label/release columns:")
    print(label_columns)

    found_named_site = False

    for column in site_columns:
        values = (
            df[column]
            .astype(str)
            .str.strip()
        )

        useful = values[
            ~values.str.lower().isin([
                "nan",
                "none",
                "",
            ])
        ]

        excluded = useful.str.lower().str.contains(
            "casa|ehrenberg",
            na=False,
        )

        third_site_values = useful[
            ~excluded
        ]

        unique_count = (
            third_site_values.nunique()
        )

        if unique_count == 0:
            continue

        found_named_site = True

        print(
            f"\nThird-site candidates from `{column}`:"
        )

        print(
            third_site_values
            .value_counts()
            .head(50)
            .to_string()
        )

    # Some master tables may not contain a clean site name.
    # In that case, group approximately by latitude and longitude.
    latitude_column = first_numeric_column(
        df,
        latitude_columns,
    )

    longitude_column = first_numeric_column(
        df,
        longitude_columns,
    )

    if (
        not found_named_site
        and latitude_column is not None
        and longitude_column is not None
    ):
        coordinates = pd.DataFrame({
            "latitude": pd.to_numeric(
                df[latitude_column],
                errors="coerce",
            ),
            "longitude": pd.to_numeric(
                df[longitude_column],
                errors="coerce",
            ),
        }).dropna()

        coordinates[
            "latitude_round"
        ] = coordinates[
            "latitude"
        ].round(3)

        coordinates[
            "longitude_round"
        ] = coordinates[
            "longitude"
        ].round(3)

        grouped = (
            coordinates.groupby([
                "latitude_round",
                "longitude_round",
            ])
            .size()
            .reset_index(
                name="event_rows"
            )
            .sort_values(
                "event_rows",
                ascending=False,
            )
        )

        print(
            "\nCandidate coordinate groups "
            "(rounded to 0.001 degree):"
        )

        print(
            grouped.head(50).to_string(
                index=False
            )
        )

    print("\nFirst 2 rows:")
    print(
        df.head(2).to_string(
            index=False
        )
    )
