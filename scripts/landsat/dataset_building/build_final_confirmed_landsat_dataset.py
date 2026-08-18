from pathlib import Path
import pandas as pd


FEATURES_CSV = Path(
    "outputs/43_landsat_unique_clean_features.csv"
)

REVIEW_2022_CSV = Path(
    "outputs/52_2022_landsat_scene_label_review.csv"
)

REVIEW_2021_CSV = Path(
    "outputs/56_2021_landsat_scene_label_review.csv"
)

FINAL_CSV = Path(
    "outputs/57_landsat_final_confirmed_features.csv"
)

EXCLUDED_CSV = Path(
    "outputs/58_landsat_excluded_ambiguous_scenes.csv"
)


def main():
    features = pd.read_csv(FEATURES_CSV)
    review_2022 = pd.read_csv(REVIEW_2022_CSV)
    review_2021 = pd.read_csv(REVIEW_2021_CSV)

    reviews = pd.concat(
        [review_2021, review_2022],
        ignore_index=True,
        sort=False,
    )

    reviews["recommended_label"] = pd.to_numeric(
        reviews["recommended_label"],
        errors="coerce",
    )

    if reviews["raster_group_id"].duplicated().any():
        duplicated = reviews.loc[
            reviews["raster_group_id"].duplicated(
                keep=False
            ),
            "raster_group_id",
        ].tolist()

        raise ValueError(
            f"Duplicate review groups found: {duplicated}"
        )

    merged = features.merge(
        reviews,
        on="raster_group_id",
        how="left",
        suffixes=("", "_review"),
        validate="one_to_one",
    )

    if merged["review_status"].isna().any():
        missing = merged.loc[
            merged["review_status"].isna(),
            "raster_group_id",
        ].tolist()

        raise ValueError(
            f"Scenes without review decisions: {missing}"
        )

    confirmed_statuses = {
        "confirmed_positive",
        "confirmed_negative",
    }

    confirmed = merged[
        merged["review_status"].isin(
            confirmed_statuses
        )
        & merged["recommended_label"].notna()
    ].copy()

    excluded = merged[
        ~merged.index.isin(confirmed.index)
    ].copy()

    confirmed["label_before_scene_review"] = (
        confirmed["label"]
    )

    confirmed["label"] = (
        confirmed["recommended_label"]
        .astype(int)
    )

    confirmed["final_scene_label"] = (
        confirmed["label"]
    )

    confirmed["final_label_source"] = (
        "release_interval_review"
    )

    confirmed = confirmed.sort_values(
        ["label", "landsat_image_time"]
    ).reset_index(drop=True)

    excluded = excluded.sort_values(
        ["review_status", "landsat_image_time"]
    ).reset_index(drop=True)

    # Integrity checks
    if confirmed["pixel_hash"].duplicated().any():
        raise ValueError(
            "Duplicate pixel hashes remain."
        )

    if confirmed["raster_group_id"].duplicated().any():
        raise ValueError(
            "Duplicate raster groups remain."
        )

    if not confirmed["label"].isin([0, 1]).all():
        raise ValueError(
            "Invalid final labels remain."
        )

    FINAL_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    confirmed.to_csv(
        FINAL_CSV,
        index=False,
    )

    excluded.to_csv(
        EXCLUDED_CSV,
        index=False,
    )

    print("=" * 80)
    print("FINAL CONFIRMED LANDSAT DATASET")
    print("=" * 80)

    print(f"\nInput unique scenes: {len(features)}")
    print(f"Confirmed scenes: {len(confirmed)}")
    print(f"Excluded ambiguous scenes: {len(excluded)}")

    print("\nFinal label counts:")
    print(
        confirmed["label"]
        .value_counts()
        .sort_index()
    )

    print("\nFinal sensor counts:")
    print(
        confirmed["landsat_sensor"]
        .value_counts()
    )

    print("\nFinal label by sensor:")
    print(
        pd.crosstab(
            confirmed["landsat_sensor"],
            confirmed["label"],
            margins=True,
        )
    )

    print("\nLabel changes after interval review:")
    print(
        pd.crosstab(
            confirmed["label_before_scene_review"],
            confirmed["label"],
            margins=True,
        )
    )

    print("\nConfirmed scene list:")
    display_columns = [
        column
        for column in [
            "raster_group_id",
            "landsat_image_time",
            "landsat_sensor",
            "label_before_scene_review",
            "label",
            "review_status",
            "label_confidence",
            "ch4_kgh_mean",
        ]
        if column in confirmed.columns
    ]

    print(
        confirmed[display_columns]
        .to_string(index=False)
    )

    print("\nSaved:")
    print(FINAL_CSV)
    print(EXCLUDED_CSV)


if __name__ == "__main__":
    main()
