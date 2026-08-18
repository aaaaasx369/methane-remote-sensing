from pathlib import Path
import pandas as pd


FILES = [
    Path("outputs/12_dataset_candidate_events_with_latlon.csv"),
    Path("outputs/final_controlled_release_satellite_availability.csv"),
    Path("outputs/15_methaneair_s2_landsat_availability.csv"),
    Path("outputs/11_availability_summary.csv"),
    Path("outputs/67_landsat_unique_candidate_overpasses.csv"),
    Path("outputs/111_landsat_high_emission_manifest_summary.csv"),
]

SITE_KEYWORDS = [
    "site",
    "location",
    "facility",
    "campaign",
    "city",
    "source_name",
]

LABEL_KEYWORDS = [
    "label",
    "target",
    "release",
    "positive",
    "high_emission",
]

LANDSAT_KEYWORDS = [
    "landsat",
    "l8",
    "l9",
    "image",
    "scene",
    "product",
    "available",
]


def matching_columns(columns, keywords):
    matches = []

    for column in columns:
        lower = column.lower()

        if any(
            keyword in lower
            for keyword in keywords
        ):
            matches.append(column)

    return matches


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

    print("\nAll columns:")
    print(df.columns.tolist())

    site_columns = matching_columns(
        df.columns,
        SITE_KEYWORDS,
    )

    label_columns = matching_columns(
        df.columns,
        LABEL_KEYWORDS,
    )

    landsat_columns = matching_columns(
        df.columns,
        LANDSAT_KEYWORDS,
    )

    print("\nPossible site columns:")
    print(site_columns)

    print("\nPossible label/release columns:")
    print(label_columns)

    print("\nPossible Landsat columns:")
    print(landsat_columns)

    for column in site_columns:
        print(
            f"\nTop values for `{column}`:"
        )

        values = (
            df[column]
            .astype(str)
            .replace("nan", pd.NA)
            .dropna()
            .value_counts()
            .head(30)
        )

        print(values.to_string())

    print("\nFirst 3 rows:")
    print(
        df.head(3).to_string(
            index=False
        )
    )
