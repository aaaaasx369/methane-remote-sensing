from pathlib import Path
import re

import numpy as np
import pandas as pd


INPUT_ROOT = Path(
    "raw_data/stanford_large_scale_release/metadata"
)

ALL_RELEASES_OUTPUT = Path(
    "outputs/125_stanford_all_release_summaries.csv"
)

LOCATION_SUMMARY_OUTPUT = Path(
    "outputs/126_stanford_location_summary.csv"
)

THIRD_SITE_OUTPUT = Path(
    "outputs/127_stanford_third_site_candidates.csv"
)

READ_ERRORS_OUTPUT = Path(
    "outputs/128_stanford_summary_read_errors.csv"
)


# 你原本已經使用過的兩個場址。
KNOWN_SITES = {
    "casa_grande": {
        "latitude": 32.821749,
        "longitude": -111.785795,
    },
    "ehrenberg": {
        "latitude": 33.630645,
        "longitude": -114.489150,
    },
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    return text.strip("_")


def haversine_distance_km(
    latitude_1,
    longitude_1,
    latitude_2,
    longitude_2,
):
    if any(
        pd.isna(value)
        for value in [
            latitude_1,
            longitude_1,
            latitude_2,
            longitude_2,
        ]
    ):
        return np.nan

    earth_radius_km = 6371.0088

    lat1 = np.radians(
        float(latitude_1)
    )

    lon1 = np.radians(
        float(longitude_1)
    )

    lat2 = np.radians(
        float(latitude_2)
    )

    lon2 = np.radians(
        float(longitude_2)
    )

    delta_latitude = lat2 - lat1
    delta_longitude = lon2 - lon1

    a = (
        np.sin(
            delta_latitude / 2
        ) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(
            delta_longitude / 2
        ) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    return float(
        earth_radius_km * c
    )


def extract_phase(path):
    path_text = str(path)

    match = re.search(
        r"(Phase\s*\d+[^/]*)",
        path_text,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return ""


def identify_known_site(
    latitude,
    longitude,
    location_normalized,
):
    distances = {}

    for site_key, coordinates in (
        KNOWN_SITES.items()
    ):
        distances[site_key] = (
            haversine_distance_km(
                latitude,
                longitude,
                coordinates["latitude"],
                coordinates["longitude"],
            )
        )

    valid_distances = {
        key: value
        for key, value
        in distances.items()
        if np.isfinite(value)
    }

    if valid_distances:
        nearest_site = min(
            valid_distances,
            key=valid_distances.get,
        )

        nearest_distance = (
            valid_distances[
                nearest_site
            ]
        )
    else:
        nearest_site = ""
        nearest_distance = np.nan

    name_match = ""

    if "casa" in location_normalized:
        name_match = "casa_grande"

    elif "ehrenberg" in location_normalized:
        name_match = "ehrenberg"

    # 10 km 內視為同一既有場址。
    coordinate_match = (
        nearest_site
        if (
            np.isfinite(
                nearest_distance
            )
            and nearest_distance <= 10
        )
        else ""
    )

    matched_site = (
        name_match
        or coordinate_match
    )

    return (
        matched_site,
        nearest_site,
        nearest_distance,
    )


def load_summary_file(path):
    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    dataframe[
        "source_summary_file"
    ] = str(path)

    dataframe[
        "source_phase"
    ] = extract_phase(path)

    dataframe[
        "source_folder"
    ] = path.parent.name

    return dataframe


def main():
    files = sorted(
        INPUT_ROOT.rglob(
            "*summary.csv"
        )
    )

    print("=" * 105)
    print("STANFORD RELEASE SUMMARY MERGE")
    print("=" * 105)

    print(
        "Summary files found:",
        len(files),
    )

    if not files:
        raise FileNotFoundError(
            f"No summary CSV files found "
            f"under {INPUT_ROOT}"
        )

    frames = []
    errors = []

    for index, path in enumerate(
        files,
        start=1,
    ):
        try:
            dataframe = (
                load_summary_file(path)
            )

            frames.append(dataframe)

        except Exception as error:
            errors.append({
                "path": str(path),
                "error": str(error),
            })

        if (
            index % 100 == 0
            or index == len(files)
        ):
            print(
                f"Processed "
                f"{index}/{len(files)}"
            )

    if not frames:
        raise RuntimeError(
            "None of the summary files "
            "could be read."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    required_columns = [
        "release_ID",
        "date",
        "time_UTC",
        "location",
        "lat",
        "lon",
        "ch4_kgh_mean",
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in combined.columns
    ]

    if missing_required:
        raise KeyError(
            "Missing required columns: "
            f"{missing_required}"
        )

    combined["lat"] = pd.to_numeric(
        combined["lat"],
        errors="coerce",
    )

    combined["lon"] = pd.to_numeric(
        combined["lon"],
        errors="coerce",
    )

    flow_columns = [
        "ch4_kgh_mean",
        "ch4_kgh_sigma",
        "ci95_lower",
        "ci95_upper",
        "PredInt95_lower",
        "PredInt95_upper",
    ]

    for column in flow_columns:
        if column in combined.columns:
            combined[column] = (
                pd.to_numeric(
                    combined[column],
                    errors="coerce",
                )
            )

    combined[
        "datetime_utc"
    ] = pd.to_datetime(
        combined["date"].astype(str)
        + " "
        + combined["time_UTC"].astype(str),
        errors="coerce",
        utc=True,
    )

    combined[
        "location_normalized"
    ] = combined["location"].apply(
        normalize_text
    )

    combined[
        "instrument_code"
    ] = (
        combined["release_ID"]
        .astype(str)
        .str.extract(
            r"_([^_]+)$",
            expand=False,
        )
        .fillna("")
    )

    known_site_results = combined.apply(
        lambda row: identify_known_site(
            row["lat"],
            row["lon"],
            row[
                "location_normalized"
            ],
        ),
        axis=1,
        result_type="expand",
    )

    known_site_results.columns = [
        "matched_existing_site",
        "nearest_existing_site",
        "nearest_existing_site_distance_km",
    ]

    combined = pd.concat(
        [
            combined,
            known_site_results,
        ],
        axis=1,
    )

    combined[
        "is_existing_site"
    ] = (
        combined[
            "matched_existing_site"
        ].ne("")
    )

    # 若 location 缺失，使用四位小數座標建立場址 key。
    coordinate_key = (
        "coord_"
        + combined["lat"]
        .round(4)
        .astype(str)
        + "_"
        + combined["lon"]
        .round(4)
        .astype(str)
    )

    combined["site_key"] = (
        combined[
            "location_normalized"
        ].where(
            combined[
                "location_normalized"
            ].ne(""),
            coordinate_key,
        )
    )

    combined[
        "is_positive_release"
    ] = (
        combined["ch4_kgh_mean"]
        > 0
    )

    combined[
        "is_zero_or_blank_release"
    ] = (
        combined["ch4_kgh_mean"]
        <= 0
    )

    duplicate_columns = [
        "release_ID",
        "datetime_utc",
        "lat",
        "lon",
        "ch4_kgh_mean",
    ]

    combined[
        "is_duplicate_release"
    ] = combined.duplicated(
        subset=duplicate_columns,
        keep=False,
    )

    combined = combined.sort_values(
        [
            "site_key",
            "datetime_utc",
            "release_ID",
        ]
    ).reset_index(drop=True)

    instrument_summary = (
        combined.groupby(
            "site_key"
        )["instrument_code"]
        .apply(
            lambda values:
                "|".join(
                    sorted({
                        str(value)
                        for value
                        in values.dropna()
                        if str(value)
                    })
                )
        )
        .rename(
            "instrument_codes"
        )
    )

    location_names = (
        combined.groupby(
            "site_key"
        )["location"]
        .apply(
            lambda values:
                "|".join(
                    sorted({
                        str(value)
                        for value
                        in values.dropna()
                        if str(value).strip()
                    })
                )
        )
        .rename(
            "location_names"
        )
    )

    location_summary = (
        combined.groupby(
            "site_key"
        )
        .agg(
            rows=(
                "release_ID",
                "size",
            ),
            unique_release_ids=(
                "release_ID",
                "nunique",
            ),
            first_datetime_utc=(
                "datetime_utc",
                "min",
            ),
            last_datetime_utc=(
                "datetime_utc",
                "max",
            ),
            latitude_median=(
                "lat",
                "median",
            ),
            longitude_median=(
                "lon",
                "median",
            ),
            latitude_min=(
                "lat",
                "min",
            ),
            latitude_max=(
                "lat",
                "max",
            ),
            longitude_min=(
                "lon",
                "min",
            ),
            longitude_max=(
                "lon",
                "max",
            ),
            flow_mean_kg_h=(
                "ch4_kgh_mean",
                "mean",
            ),
            flow_median_kg_h=(
                "ch4_kgh_mean",
                "median",
            ),
            flow_min_kg_h=(
                "ch4_kgh_mean",
                "min",
            ),
            flow_max_kg_h=(
                "ch4_kgh_mean",
                "max",
            ),
            positive_release_rows=(
                "is_positive_release",
                "sum",
            ),
            zero_or_blank_rows=(
                "is_zero_or_blank_release",
                "sum",
            ),
            duplicate_rows=(
                "is_duplicate_release",
                "sum",
            ),
            existing_site=(
                "is_existing_site",
                "max",
            ),
            matched_existing_site=(
                "matched_existing_site",
                lambda values:
                    "|".join(
                        sorted({
                            str(value)
                            for value
                            in values
                            if str(value)
                        })
                    ),
            ),
            nearest_existing_site_distance_km=(
                "nearest_existing_site_distance_km",
                "median",
            ),
        )
        .join(location_names)
        .join(instrument_summary)
        .reset_index()
    )

    location_summary = (
        location_summary.sort_values(
            [
                "existing_site",
                "positive_release_rows",
                "rows",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
    )

    third_site_summary = (
        location_summary[
            ~location_summary[
                "existing_site"
            ]
        ].copy()
    )

    candidate_site_keys = set(
        third_site_summary[
            "site_key"
        ]
    )

    third_site_releases = (
        combined[
            combined["site_key"].isin(
                candidate_site_keys
            )
        ].copy()
    )

    ALL_RELEASES_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        ALL_RELEASES_OUTPUT,
        index=False,
    )

    location_summary.to_csv(
        LOCATION_SUMMARY_OUTPUT,
        index=False,
    )

    third_site_releases.to_csv(
        THIRD_SITE_OUTPUT,
        index=False,
    )

    pd.DataFrame(
        errors
    ).to_csv(
        READ_ERRORS_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 105)
    print("MERGE SUMMARY")
    print("=" * 105)

    print(
        "Rows read:",
        len(combined),
    )

    print(
        "Unique release IDs:",
        combined[
            "release_ID"
        ].nunique(),
    )

    print(
        "Unique site keys:",
        combined[
            "site_key"
        ].nunique(),
    )

    print(
        "Read errors:",
        len(errors),
    )

    print(
        "Duplicate release rows:",
        int(
            combined[
                "is_duplicate_release"
            ].sum()
        ),
    )

    print("\n" + "=" * 105)
    print("LOCATION SUMMARY")
    print("=" * 105)

    display_columns = [
        "site_key",
        "location_names",
        "rows",
        "positive_release_rows",
        "zero_or_blank_rows",
        "latitude_median",
        "longitude_median",
        "flow_median_kg_h",
        "flow_max_kg_h",
        "existing_site",
        "matched_existing_site",
        "nearest_existing_site_distance_km",
        "instrument_codes",
    ]

    print(
        location_summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.3f}",
        )
    )

    print("\n" + "=" * 105)
    print("THIRD-SITE CANDIDATES")
    print("=" * 105)

    if third_site_summary.empty:
        print(
            "No new locations were found "
            "outside Casa Grande and "
            "Ehrenberg."
        )
    else:
        print(
            third_site_summary[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.3f}",
            )
        )

    print("\nSaved:")
    print(ALL_RELEASES_OUTPUT)
    print(LOCATION_SUMMARY_OUTPUT)
    print(THIRD_SITE_OUTPUT)
    print(READ_ERRORS_OUTPUT)


if __name__ == "__main__":
    main()
