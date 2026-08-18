from pathlib import Path

import pandas as pd


SCENE_CSV = Path(
    "outputs/43_landsat_unique_clean_features.csv"
)

POSSIBLE_RELEASE_FILES = [
    Path(
        "raw_data/2024_SU_Controlled_Releases/"
        "sahar-elabbadi-SU-Controlled-Releases-2022-0a604d7/"
        "Satellite_overpasses_with_release_rates_20230404.csv"
    ),
    Path(
        "raw_data/2024_SU_Controlled_Releases/"
        "sahar-elabbadi-SU-Controlled-Releases-2022-0a604d7/"
        "00_raw_reports/"
        "Satellite_overpasses_with_release_rates_20230404.csv"
    ),
]

OUTPUT_CSV = Path(
    "outputs/51_2022_landsat_release_interval_candidates.csv"
)


def locate_release_file():
    for path in POSSIBLE_RELEASE_FILES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the 2022 release interval file.\n"
        + "\n".join(str(path) for path in POSSIBLE_RELEASE_FILES)
    )


def parse_utc(series):
    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )


def main():
    if not SCENE_CSV.exists():
        raise FileNotFoundError(
            f"Scene dataset not found: {SCENE_CSV}"
        )

    release_path = locate_release_file()

    scenes = pd.read_csv(SCENE_CSV)
    releases = pd.read_csv(release_path)

    print("=" * 90)
    print("2022 LANDSAT–RELEASE INTERVAL INSPECTION")
    print("=" * 90)

    print(f"\nScene file: {SCENE_CSV}")
    print(f"Release file: {release_path}")

    print(f"\nRelease rows: {len(releases)}")
    print(f"Release columns: {len(releases.columns)}")

    print("\nRelease-table columns:")
    for index, column in enumerate(
        releases.columns,
        start=1,
    ):
        print(f"{index:02d}. {column}")

    required_release_columns = [
        "start_release",
        "end_release",
    ]

    missing_release_columns = [
        column
        for column in required_release_columns
        if column not in releases.columns
    ]

    if missing_release_columns:
        raise ValueError(
            "Missing release columns: "
            + ", ".join(missing_release_columns)
        )

    if "landsat_image_time" not in scenes.columns:
        raise ValueError(
            "landsat_image_time is missing from the scene dataset."
        )

    scenes["landsat_time_utc"] = parse_utc(
        scenes["landsat_image_time"]
    )

    releases["release_start_utc"] = parse_utc(
        releases["start_release"]
    )

    releases["release_end_utc"] = parse_utc(
        releases["end_release"]
    )

    valid_releases = releases[
        releases["release_start_utc"].notna()
        & releases["release_end_utc"].notna()
    ].copy()

    scenes_2022 = scenes[
        scenes["landsat_time_utc"].dt.year == 2022
    ].copy()

    print(
        f"\nValid release intervals: "
        f"{len(valid_releases)}"
    )

    print(
        f"Unique Landsat scenes from 2022: "
        f"{len(scenes_2022)}"
    )

    if len(valid_releases) > 0:
        print("\nRelease interval date range:")
        print(
            valid_releases["release_start_utc"].min(),
            "to",
            valid_releases["release_end_utc"].max(),
        )

    output_rows = []

    useful_release_columns = [
        column
        for column in [
            "Date",
            "Timestamp (UTC)",
            "DateTime (UTC)",
            "start_release",
            "end_release",
            "ch4_fraction_km",
            "ch4_fraction_km_sigma",
            "ch4_kgh_mean",
            "ch4_kgh_sigma",
        ]
        if column in valid_releases.columns
    ]

    for _, scene in scenes_2022.iterrows():
        scene_time = scene["landsat_time_utc"]

        if pd.isna(scene_time):
            continue

        candidates = valid_releases.copy()

        candidates["overlaps_landsat"] = (
            (candidates["release_start_utc"] <= scene_time)
            & (candidates["release_end_utc"] >= scene_time)
        )

        candidates["seconds_to_interval"] = 0.0

        before_mask = (
            scene_time < candidates["release_start_utc"]
        )

        after_mask = (
            scene_time > candidates["release_end_utc"]
        )

        candidates.loc[
            before_mask,
            "seconds_to_interval",
        ] = (
            candidates.loc[
                before_mask,
                "release_start_utc",
            ]
            - scene_time
        ).dt.total_seconds()

        candidates.loc[
            after_mask,
            "seconds_to_interval",
        ] = (
            scene_time
            - candidates.loc[
                after_mask,
                "release_end_utc",
            ]
        ).dt.total_seconds()

        # 只保留距離 Landsat 拍攝時間最近的 5 個 interval
        candidates = candidates.sort_values(
            by=[
                "overlaps_landsat",
                "seconds_to_interval",
            ],
            ascending=[
                False,
                True,
            ],
        ).head(5)

        print("\n" + "-" * 90)

        print(
            f"Raster: {scene.get('raster_group_id', '')}"
        )
        print(
            f"Landsat time: {scene_time}"
        )
        print(
            f"Current label: {scene.get('label', '')}"
        )

        if len(candidates) == 0:
            print("No release interval candidates found.")
            continue

        display_columns = [
            "release_start_utc",
            "release_end_utc",
            "overlaps_landsat",
            "seconds_to_interval",
        ] + useful_release_columns

        display_columns = list(
            dict.fromkeys(display_columns)
        )

        print(
            candidates[
                display_columns
            ].to_string(index=False)
        )

        for rank, (_, candidate) in enumerate(
            candidates.iterrows(),
            start=1,
        ):
            output_row = {
                "raster_group_id":
                    scene.get("raster_group_id", ""),
                "landsat_image_time":
                    scene.get("landsat_image_time", ""),
                "landsat_time_utc":
                    scene_time,
                "current_label":
                    scene.get("label", ""),
                "candidate_rank":
                    rank,
                "release_start_utc":
                    candidate["release_start_utc"],
                "release_end_utc":
                    candidate["release_end_utc"],
                "overlaps_landsat":
                    candidate["overlaps_landsat"],
                "seconds_to_interval":
                    candidate["seconds_to_interval"],
                "hours_to_interval":
                    candidate["seconds_to_interval"]
                    / 3600,
            }

            for column in useful_release_columns:
                output_row[column] = candidate.get(
                    column
                )

            output_rows.append(output_row)

    output_df = pd.DataFrame(output_rows)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    if len(output_df) == 0:
        print("\nNo matches were produced.")
    else:
        nearest = (
            output_df[
                output_df["candidate_rank"] == 1
            ]
            .copy()
        )

        print("\nNearest interval for each Landsat scene:")
        print(
            nearest[
                [
                    "raster_group_id",
                    "landsat_time_utc",
                    "current_label",
                    "release_start_utc",
                    "release_end_utc",
                    "overlaps_landsat",
                    "hours_to_interval",
                    *[
                        column
                        for column in [
                            "ch4_kgh_mean",
                            "ch4_kgh_sigma",
                        ]
                        if column in nearest.columns
                    ],
                ]
            ].to_string(index=False)
        )

        print(
            "\nScenes overlapping a listed "
            "release interval:",
            int(nearest["overlaps_landsat"].sum()),
        )

        print(
            "Scenes not overlapping a listed "
            "release interval:",
            int((~nearest["overlaps_landsat"]).sum()),
        )

    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
