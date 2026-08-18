from pathlib import Path

import pandas as pd


UNIQUE_FEATURES_CSV = Path(
    "outputs/43_landsat_unique_clean_features.csv"
)

ORIGINAL_CONFIRMED_CSV = Path(
    "outputs/57_landsat_final_confirmed_features.csv"
)

ADJUDICATION_CSV = Path(
    "outputs/78_casa_grande_schedule_adjudication.csv"
)

CORE_OUTPUT_CSV = Path(
    "outputs/79_landsat_core_schedule_confirmed_features.csv"
)

EXTENDED_OUTPUT_CSV = Path(
    "outputs/80_landsat_extended_schedule_confirmed_features.csv"
)


SCHEDULE_DECISIONS = [
    {
        "raster_group_id": "RG_60d2e632a2c6",
        "final_label": 0,
        "schedule_status":
            "confirmed_negative_no_november_weekend_release",
        "label_confidence": "high",
        "include_in_core": True,
        "evidence_source":
            "published_campaign_schedule",
        "decision_reason":
            "The acquisition occurred on 2022-11-19. "
            "The published study states that there were no "
            "weekend releases in November.",
    },
    {
        "raster_group_id": "RG_135869ef6162",
        "final_label": 0,
        "schedule_status":
            "confirmed_negative_no_november_weekend_release",
        "label_confidence": "high",
        "include_in_core": True,
        "evidence_source":
            "published_campaign_schedule",
        "decision_reason":
            "The acquisition occurred on 2022-11-26. "
            "The published study states that there were no "
            "weekend releases in November.",
    },
    {
        "raster_group_id": "RG_9f73cd68c47a",
        "final_label": 0,
        "schedule_status":
            "confirmed_negative_no_november_weekend_release",
        "label_confidence": "high",
        "include_in_core": True,
        "evidence_source":
            "published_campaign_schedule",
        "decision_reason":
            "The acquisition occurred on 2022-11-27. "
            "The published study states that there were no "
            "weekend releases in November.",
    },
    {
        "raster_group_id": "RG_eb8b0e23b1c2",
        "final_label": 0,
        "schedule_status":
            "provisional_negative_before_official_test_period",
        "label_confidence": "medium_high",
        "include_in_core": False,
        "evidence_source":
            "published_test_period_and_release_log",
        "decision_reason":
            "The acquisition occurred on 2022-10-09, one day "
            "before the published official test period began "
            "on 2022-10-10. No release-log interval exists on "
            "the acquisition date. Kept outside the core dataset "
            "because preliminary releases occurred before the "
            "official test period.",
    },
]


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
        .fillna(False)
    )


def check_dataset(df, name):
    if df["raster_group_id"].duplicated().any():
        duplicated = df.loc[
            df["raster_group_id"].duplicated(
                keep=False
            ),
            "raster_group_id",
        ].tolist()

        raise ValueError(
            f"{name} has duplicate raster groups: "
            f"{duplicated}"
        )

    if (
        "pixel_hash" in df.columns
        and df["pixel_hash"].duplicated().any()
    ):
        duplicated = df.loc[
            df["pixel_hash"].duplicated(
                keep=False
            ),
            [
                "raster_group_id",
                "pixel_hash",
            ],
        ]

        raise ValueError(
            f"{name} has duplicate pixel hashes:\n"
            + duplicated.to_string(index=False)
        )

    if not df["label"].isin([0, 1]).all():
        raise ValueError(
            f"{name} contains invalid labels."
        )


def prepare_schedule_rows(
    unique_features,
    adjudication,
):
    rows = unique_features[
        unique_features["raster_group_id"].isin(
            adjudication["raster_group_id"]
        )
    ].copy()

    missing_groups = sorted(
        set(adjudication["raster_group_id"])
        - set(rows["raster_group_id"])
    )

    if missing_groups:
        raise ValueError(
            "Schedule-adjudicated raster groups are "
            f"missing from the feature table: {missing_groups}"
        )

    rows = rows.merge(
        adjudication,
        on="raster_group_id",
        how="left",
        validate="one_to_one",
    )

    rows["label_before_scene_review"] = (
        pd.to_numeric(
            rows["label"],
            errors="coerce",
        )
    )

    rows["label"] = (
        pd.to_numeric(
            rows["final_label"],
            errors="raise",
        )
        .astype(int)
    )

    rows["recommended_label"] = rows["label"]
    rows["final_scene_label"] = rows["label"]

    rows["review_status"] = (
        rows["schedule_status"]
    )

    rows["review_reason"] = (
        rows["decision_reason"]
    )

    rows["final_label_source"] = (
        rows["evidence_source"]
    )

    return rows


def main():
    for path in [
        UNIQUE_FEATURES_CSV,
        ORIGINAL_CONFIRMED_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input file: {path}"
            )

    unique_features = pd.read_csv(
        UNIQUE_FEATURES_CSV,
        low_memory=False,
    )

    original_confirmed = pd.read_csv(
        ORIGINAL_CONFIRMED_CSV,
        low_memory=False,
    )

    adjudication = pd.DataFrame(
        SCHEDULE_DECISIONS
    )

    adjudication["include_in_core"] = (
        parse_boolean(
            adjudication["include_in_core"]
        )
    )

    ADJUDICATION_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    adjudication.to_csv(
        ADJUDICATION_CSV,
        index=False,
    )

    already_confirmed = set(
        original_confirmed["raster_group_id"]
    )

    overlap = sorted(
        already_confirmed
        & set(adjudication["raster_group_id"])
    )

    if overlap:
        raise ValueError(
            "Schedule-adjudicated rows are already in "
            f"the original confirmed dataset: {overlap}"
        )

    schedule_rows = prepare_schedule_rows(
        unique_features,
        adjudication,
    )

    core_additions = schedule_rows[
        schedule_rows["include_in_core"] == True
    ].copy()

    extended_additions = schedule_rows.copy()

    core = pd.concat(
        [
            original_confirmed,
            core_additions,
        ],
        ignore_index=True,
        sort=False,
    )

    extended = pd.concat(
        [
            original_confirmed,
            extended_additions,
        ],
        ignore_index=True,
        sort=False,
    )

    core["dataset_tier"] = "core"
    extended["dataset_tier"] = "extended"

    core = core.sort_values(
        by=[
            "label",
            "landsat_image_time",
            "raster_group_id",
        ],
    ).reset_index(drop=True)

    extended = extended.sort_values(
        by=[
            "label",
            "landsat_image_time",
            "raster_group_id",
        ],
    ).reset_index(drop=True)

    check_dataset(
        core,
        "Core dataset",
    )

    check_dataset(
        extended,
        "Extended dataset",
    )

    expected_core_groups = (
        len(original_confirmed)
        + int(
            adjudication[
                "include_in_core"
            ].sum()
        )
    )

    expected_extended_groups = (
        len(original_confirmed)
        + len(adjudication)
    )

    if len(core) != expected_core_groups:
        raise ValueError(
            "Unexpected core dataset size: "
            f"{len(core)} instead of "
            f"{expected_core_groups}"
        )

    if len(extended) != expected_extended_groups:
        raise ValueError(
            "Unexpected extended dataset size: "
            f"{len(extended)} instead of "
            f"{expected_extended_groups}"
        )

    core.to_csv(
        CORE_OUTPUT_CSV,
        index=False,
    )

    extended.to_csv(
        EXTENDED_OUTPUT_CSV,
        index=False,
    )

    print("=" * 90)
    print("LANDSAT DATASETS WITH CAMPAIGN-SCHEDULE REVIEW")
    print("=" * 90)

    print(
        f"\nOriginal confirmed scenes: "
        f"{len(original_confirmed)}"
    )

    print(
        f"Schedule-adjudicated scenes: "
        f"{len(schedule_rows)}"
    )

    print("\nSchedule decisions:")
    print(
        adjudication[
            [
                "raster_group_id",
                "final_label",
                "schedule_status",
                "label_confidence",
                "include_in_core",
            ]
        ].to_string(index=False)
    )

    print("\n" + "-" * 90)
    print("CORE DATASET")
    print("-" * 90)

    print(f"Scenes: {len(core)}")

    print("\nLabel counts:")
    print(
        core["label"]
        .value_counts()
        .sort_index()
    )

    print("\nSensor counts:")
    print(
        core["landsat_sensor"]
        .value_counts()
    )

    print("\nLabel by sensor:")
    print(
        pd.crosstab(
            core["landsat_sensor"],
            core["label"],
            margins=True,
        )
    )

    print("\n" + "-" * 90)
    print("EXTENDED DATASET")
    print("-" * 90)

    print(f"Scenes: {len(extended)}")

    print("\nLabel counts:")
    print(
        extended["label"]
        .value_counts()
        .sort_index()
    )

    print("\nSensor counts:")
    print(
        extended["landsat_sensor"]
        .value_counts()
    )

    print("\nLabel by sensor:")
    print(
        pd.crosstab(
            extended["landsat_sensor"],
            extended["label"],
            margins=True,
        )
    )

    print("\nNewly restored negative scenes:")
    print(
        schedule_rows[
            [
                "raster_group_id",
                "landsat_image_time",
                "landsat_sensor",
                "label_before_scene_review",
                "label",
                "schedule_status",
                "label_confidence",
                "include_in_core",
            ]
        ].sort_values(
            "landsat_image_time"
        ).to_string(index=False)
    )

    print("\nSaved:")
    print(ADJUDICATION_CSV)
    print(CORE_OUTPUT_CSV)
    print(EXTENDED_OUTPUT_CSV)


if __name__ == "__main__":
    main()
