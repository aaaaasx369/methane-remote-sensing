from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_CSV = Path(
    "outputs/67_landsat_unique_candidate_overpasses.csv"
)

EXCLUDED_CSV = Path(
    "outputs/58_landsat_excluded_ambiguous_scenes.csv"
)

ALL_OUTPUT_CSV = Path(
    "outputs/72_landsat_candidate_overpass_provenance.csv"
)

MATCH_OUTPUT_CSV = Path(
    "outputs/73_existing_excluded_overpass_matches.csv"
)

TRULY_NEW_OUTPUT_CSV = Path(
    "outputs/74_truly_new_landsat_candidate_overpasses.csv"
)


TIME_TOLERANCE_SECONDS = 180


KNOWN_SITES = {
    "casa_grande": {
        "lat": 32.821821,
        "lon": -111.785773,
    },
    "ehrenberg": {
        "lat": 33.630645,
        "lon": -114.489150,
    },
}


def assign_site(latitude, longitude):
    latitude = pd.to_numeric(
        latitude,
        errors="coerce",
    )

    longitude = pd.to_numeric(
        longitude,
        errors="coerce",
    )

    if pd.isna(latitude) or pd.isna(longitude):
        return np.nan

    best_site = None
    best_distance = np.inf

    for site_key, site in KNOWN_SITES.items():
        distance = np.sqrt(
            (latitude - site["lat"]) ** 2
            + (longitude - site["lon"]) ** 2
        )

        if distance < best_distance:
            best_distance = distance
            best_site = site_key

    if best_distance > 0.01:
        return np.nan

    return best_site


def first_existing_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


def main():
    if not CANDIDATE_CSV.exists():
        raise FileNotFoundError(
            f"Missing candidate file: {CANDIDATE_CSV}"
        )

    if not EXCLUDED_CSV.exists():
        raise FileNotFoundError(
            f"Missing excluded-scene file: {EXCLUDED_CSV}"
        )

    candidates = pd.read_csv(
        CANDIDATE_CSV,
        low_memory=False,
    )

    excluded = pd.read_csv(
        EXCLUDED_CSV,
        low_memory=False,
    )

    candidates["candidate_time_utc"] = pd.to_datetime(
        candidates["candidate_time_utc"],
        errors="coerce",
        utc=True,
    )

    excluded_time_column = first_existing_column(
        excluded.columns,
        [
            "landsat_image_time",
            "landsat_image_time_previous",
            "landsat_time_utc",
        ],
    )

    excluded_sensor_column = first_existing_column(
        excluded.columns,
        [
            "landsat_sensor",
            "landsat_sensor_previous",
        ],
    )

    excluded_lat_column = first_existing_column(
        excluded.columns,
        [
            "lat",
            "latitude",
            "lat_previous",
        ],
    )

    excluded_lon_column = first_existing_column(
        excluded.columns,
        [
            "lon",
            "longitude",
            "lon_previous",
        ],
    )

    required_detected = {
        "time": excluded_time_column,
        "sensor": excluded_sensor_column,
        "lat": excluded_lat_column,
        "lon": excluded_lon_column,
    }

    missing = [
        name
        for name, column in required_detected.items()
        if column is None
    ]

    if missing:
        raise ValueError(
            "Could not detect excluded-scene columns: "
            + ", ".join(missing)
        )

    excluded["excluded_time_utc"] = pd.to_datetime(
        excluded[excluded_time_column],
        errors="coerce",
        utc=True,
    )

    excluded["excluded_sensor"] = (
        excluded[excluded_sensor_column]
        .astype(str)
        .str.strip()
    )

    excluded["site_key_detected"] = excluded.apply(
        lambda row: assign_site(
            row[excluded_lat_column],
            row[excluded_lon_column],
        ),
        axis=1,
    )

    candidates["existing_excluded_ambiguous"] = False
    candidates["excluded_raster_group_id"] = np.nan
    candidates["excluded_time_utc"] = pd.NaT
    candidates["excluded_time_difference_seconds"] = np.nan
    candidates["excluded_original_label"] = np.nan
    candidates["excluded_review_status"] = np.nan
    candidates["candidate_provenance_role"] = (
        "truly_new_candidate_overpass"
    )

    for candidate_index, candidate in candidates.iterrows():
        possible = excluded[
            (
                excluded["site_key_detected"]
                == candidate["site_key"]
            )
            & (
                excluded["excluded_sensor"]
                == candidate["landsat_sensor"]
            )
            & excluded["excluded_time_utc"].notna()
        ].copy()

        if len(possible) == 0:
            continue

        possible["time_difference_seconds"] = (
            possible["excluded_time_utc"]
            - candidate["candidate_time_utc"]
        ).abs().dt.total_seconds()

        nearest = possible.sort_values(
            "time_difference_seconds"
        ).iloc[0]

        if (
            nearest["time_difference_seconds"]
            > TIME_TOLERANCE_SECONDS
        ):
            continue

        candidates.loc[
            candidate_index,
            "existing_excluded_ambiguous",
        ] = True

        candidates.loc[
            candidate_index,
            "excluded_raster_group_id",
        ] = nearest.get("raster_group_id")

        candidates.loc[
            candidate_index,
            "excluded_time_utc",
        ] = nearest["excluded_time_utc"]

        candidates.loc[
            candidate_index,
            "excluded_time_difference_seconds",
        ] = nearest["time_difference_seconds"]

        candidates.loc[
            candidate_index,
            "excluded_original_label",
        ] = nearest.get("label")

        candidates.loc[
            candidate_index,
            "excluded_review_status",
        ] = nearest.get("review_status")

        candidates.loc[
            candidate_index,
            "candidate_provenance_role",
        ] = "existing_excluded_ambiguous_scene"

    matched = candidates[
        candidates["existing_excluded_ambiguous"]
        == True
    ].copy()

    truly_new = candidates[
        candidates["existing_excluded_ambiguous"]
        == False
    ].copy()

    candidates = candidates.sort_values(
        [
            "site_key",
            "candidate_time_utc",
        ]
    ).reset_index(drop=True)

    matched = matched.sort_values(
        [
            "site_key",
            "candidate_time_utc",
        ]
    ).reset_index(drop=True)

    truly_new = truly_new.sort_values(
        [
            "site_key",
            "candidate_time_utc",
        ]
    ).reset_index(drop=True)

    ALL_OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        ALL_OUTPUT_CSV,
        index=False,
    )

    matched.to_csv(
        MATCH_OUTPUT_CSV,
        index=False,
    )

    truly_new.to_csv(
        TRULY_NEW_OUTPUT_CSV,
        index=False,
    )

    print("=" * 100)
    print("CANDIDATE–EXCLUDED SCENE MATCH AUDIT")
    print("=" * 100)

    print(f"\nInput candidate overpasses: {len(candidates)}")
    print(
        "Existing excluded ambiguous scenes matched:",
        len(matched),
    )
    print(
        "Truly new candidate overpasses:",
        len(truly_new),
    )

    print("\nCandidate provenance counts:")
    print(
        candidates[
            "candidate_provenance_role"
        ].value_counts()
    )

    print("\nMatched excluded scenes:")

    if len(matched) == 0:
        print("None")
    else:
        columns = [
            "overpass_id",
            "site_name_normalized",
            "candidate_time_utc",
            "landsat_sensor",
            "LANDSAT_PRODUCT_ID",
            "excluded_raster_group_id",
            "excluded_time_utc",
            "excluded_time_difference_seconds",
            "excluded_original_label",
            "excluded_review_status",
        ]

        print(
            matched[columns]
            .to_string(index=False)
        )

    print("\nTruly new candidates by site and sensor:")
    print(
        pd.crosstab(
            truly_new["site_name_normalized"],
            truly_new["landsat_sensor"],
            margins=True,
        )
    )

    print("\nSaved:")
    print(ALL_OUTPUT_CSV)
    print(MATCH_OUTPUT_CSV)
    print(TRULY_NEW_OUTPUT_CSV)


if __name__ == "__main__":
    main()
