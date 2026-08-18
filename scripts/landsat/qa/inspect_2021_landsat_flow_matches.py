from pathlib import Path

import numpy as np
import pandas as pd


SCENE_FILE = Path(
    "outputs/53_2021_landsat_scenes_for_review.csv"
)

OUTPUT_FILE = Path(
    "outputs/55_2021_landsat_flow_matches.csv"
)


DIRECT_20211021_CANDIDATES = [
    Path(
        "raw_data/2023_Controlled_Release_2021/"
        "GHGSatTestData/211021_release_dat.csv"
    ),
    Path(
        "raw_data/2023_SatelliteTesting/OLD/"
        "Controlled_Release_2021_main/"
        "GHGSatTestData/211021_release_dat.csv"
    ),
]


MATCHED_TABLE_CANDIDATES = [
    Path(
        "raw_data/2023_Controlled_Release_2021/"
        "matchedDF_Satellites_230118c.csv"
    ),
    Path(
        "raw_data/2023_Controlled_Release_2021/"
        "Dataframes for Stanford analysis/"
        "matchedDF_Satellites_230118c.csv"
    ),
    Path(
        "raw_data/2023_Controlled_Release_2021/"
        "matchedDF_Satellites_220601.csv"
    ),
    Path(
        "raw_data/2023_Controlled_Release_2021/"
        "Dataframes for Stanford analysis/"
        "matchedDF_Satellites_220601.csv"
    ),
]


SHUTOFF_CANDIDATES = [
    Path(
        "raw_data/2023_Controlled_Release_2021/"
        "GHGSatTestData/shut_off_stamps.csv"
    ),
    Path(
        "raw_data/2023_SatelliteTesting/OLD/"
        "Controlled_Release_2021_main/"
        "GHGSatTestData/shut_off_stamps.csv"
    ),
]


TRANSITION_CANDIDATES = [
    Path(
        "raw_data/2023_Controlled_Release_2021/"
        "GHGSatTestData/transition_stamps_v2.csv"
    ),
    Path(
        "raw_data/2023_Controlled_Release_2021/"
        "GHGSatTestData/transition_stamps.csv"
    ),
    Path(
        "raw_data/2023_SatelliteTesting/OLD/"
        "Controlled_Release_2021_main/"
        "GHGSatTestData/transition_stamps_v2.csv"
    ),
]


FLOW_COLUMNS = [
    "cr_kgh_CH4_mean90",
    "cr_kgh_CH4_mean60",
    "cr_kgh_CH4_mean30",
    "cr_kgh_CH4_mean300",
    "cr_kgh_CH4_mean600",
    "cr_kgh_CH4_mean900",
    "cr_allmeters_scfh",
    "cr_quad_scfh",
]


def locate_first(paths, description):
    for path in paths:
        if path.exists():
            print(f"[FOUND] {description}: {path}")
            return path

    print(f"[NOT FOUND] {description}")

    for path in paths:
        print(f"  checked: {path}")

    return None


def parse_utc(series, dayfirst=False):
    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
        dayfirst=dayfirst,
    )


def inspect_direct_20211021(scene_time):
    path = locate_first(
        DIRECT_20211021_CANDIDATES,
        "2021-10-21 direct release file",
    )

    if path is None:
        return

    df = pd.read_csv(path)

    print("\n" + "=" * 100)
    print("DIRECT 2021-10-21 RELEASE DATA")
    print("=" * 100)

    print(f"\nFile: {path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if "timestamp" not in df.columns:
        print("timestamp column was not found.")
        print(df.to_string(index=False))
        return

    df["parsed_time_utc"] = parse_utc(
        df["timestamp"],
        dayfirst=True,
    )

    if "SCFH" in df.columns:
        df["SCFH"] = pd.to_numeric(
            df["SCFH"],
            errors="coerce",
        )

    df["seconds_from_landsat"] = (
        df["parsed_time_utc"] - scene_time
    ).dt.total_seconds()

    print(f"\nLandsat acquisition: {scene_time}")

    print("\nAll direct release records:")
    print(
        df.to_string(index=False)
    )

    before = df[
        df["parsed_time_utc"] <= scene_time
    ].sort_values(
        "parsed_time_utc"
    )

    after = df[
        df["parsed_time_utc"] > scene_time
    ].sort_values(
        "parsed_time_utc"
    )

    print("\nLast release record at or before Landsat:")

    if len(before) == 0:
        print("None")
    else:
        print(
            before.tail(1).to_string(
                index=False
            )
        )

    print("\nFirst release record after Landsat:")

    if len(after) == 0:
        print("None")
    else:
        print(
            after.head(1).to_string(
                index=False
            )
        )


def inspect_stamp_file(paths, description):
    path = locate_first(
        paths,
        description,
    )

    if path is None:
        return

    try:
        df = pd.read_csv(
            path,
            header=None,
        )

        print("\n" + "-" * 100)
        print(description.upper())
        print("-" * 100)

        print(f"File: {path}")
        print(f"Shape: {df.shape}")
        print(df.to_string(index=False, header=False))

    except Exception as error:
        print(
            f"[ERROR] Could not read {path}: {error}"
        )


def find_timestamp_column(df):
    preferred = [
        "Operator_Timestamp",
        "Stanford_timestamp",
        "SurveyTime",
    ]

    for column in preferred:
        if column in df.columns:
            parsed = parse_utc(df[column])

            if parsed.notna().sum() > 0:
                return column, parsed

    raise ValueError(
        "No usable timestamp column found in matched table."
    )


def inspect_matched_table(scenes):
    path = locate_first(
        MATCHED_TABLE_CANDIDATES,
        "processed matched satellite table",
    )

    if path is None:
        return []

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    timestamp_column, parsed_times = (
        find_timestamp_column(df)
    )

    df["_matched_time_utc"] = parsed_times

    available_flow_columns = [
        column
        for column in FLOW_COLUMNS
        if column in df.columns
    ]

    for column in available_flow_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    print("\n" + "=" * 100)
    print("PROCESSED MATCHED SATELLITE TABLE")
    print("=" * 100)

    print(f"\nFile: {path}")
    print(f"Rows: {len(df)}")
    print(f"Timestamp column: {timestamp_column}")
    print(
        f"Flow columns available: "
        f"{available_flow_columns}"
    )

    output_rows = []

    for _, scene in scenes.iterrows():
        scene_time = scene["landsat_time_utc"]

        temporary = df[
            df["_matched_time_utc"].notna()
        ].copy()

        temporary["seconds_from_landsat"] = (
            temporary["_matched_time_utc"]
            - scene_time
        ).dt.total_seconds()

        temporary[
            "absolute_seconds_from_landsat"
        ] = temporary[
            "seconds_from_landsat"
        ].abs()

        # Show records within ±36 hours, nearest first.
        nearby = temporary[
            temporary[
                "absolute_seconds_from_landsat"
            ] <= 36 * 3600
        ].sort_values(
            "absolute_seconds_from_landsat"
        ).head(15)

        print("\n" + "-" * 100)

        print(
            f"Raster group: "
            f"{scene['raster_group_id']}"
        )

        print(
            f"Landsat time: {scene_time}"
        )

        print(
            f"Current label: "
            f"{scene.get('label', '')}"
        )

        display_columns = [
            column
            for column in [
                timestamp_column,
                "_matched_time_utc",
                "seconds_from_landsat",
                "DateOfSurvey",
                "StartTime",
                "EndTime",
                "SurveyTime",
                "cr_start",
                "cr_end",
                "cr_idx",
                "tc_Classification",
                "Detection",
                *available_flow_columns,
            ]
            if column in nearby.columns
        ]

        if len(nearby) == 0:
            print(
                "No records found within ±36 hours."
            )
            continue

        print(
            nearby[display_columns]
            .to_string(index=False)
        )

        for rank, (_, row) in enumerate(
            nearby.iterrows(),
            start=1,
        ):
            output_row = {
                "raster_group_id":
                    scene["raster_group_id"],
                "landsat_time_utc":
                    scene_time,
                "current_label":
                    scene.get("label"),
                "matched_source_file":
                    str(path),
                "matched_timestamp_column":
                    timestamp_column,
                "candidate_rank":
                    rank,
                "matched_time_utc":
                    row["_matched_time_utc"],
                "seconds_from_landsat":
                    row["seconds_from_landsat"],
                "absolute_seconds_from_landsat":
                    row[
                        "absolute_seconds_from_landsat"
                    ],
            }

            for column in display_columns:
                if column.startswith("_"):
                    continue

                output_row[column] = row.get(column)

            output_rows.append(output_row)

    return output_rows


def main():
    if not SCENE_FILE.exists():
        raise FileNotFoundError(
            f"Scene file not found: {SCENE_FILE}"
        )

    scenes = pd.read_csv(
        SCENE_FILE
    )

    if "landsat_time_utc" in scenes.columns:
        scenes["landsat_time_utc"] = (
            parse_utc(
                scenes["landsat_time_utc"]
            )
        )
    elif "landsat_image_time" in scenes.columns:
        scenes["landsat_time_utc"] = (
            parse_utc(
                scenes["landsat_image_time"]
            )
        )
    else:
        raise ValueError(
            "No Landsat time column was found."
        )

    scenes = scenes[
        scenes["landsat_time_utc"].notna()
    ].copy()

    print("=" * 100)
    print("2021 LANDSAT FLOW MATCH INSPECTION")
    print("=" * 100)

    print(f"\nScenes: {len(scenes)}")

    print(
        scenes[
            [
                column
                for column in [
                    "raster_group_id",
                    "landsat_time_utc",
                    "label",
                    "event_id",
                ]
                if column in scenes.columns
            ]
        ].to_string(index=False)
    )

    scene_20211021 = scenes[
        scenes[
            "landsat_time_utc"
        ].dt.strftime("%Y-%m-%d")
        == "2021-10-21"
    ]

    if len(scene_20211021) > 0:
        inspect_direct_20211021(
            scene_20211021[
                "landsat_time_utc"
            ].iloc[0]
        )

        inspect_stamp_file(
            SHUTOFF_CANDIDATES,
            "GHGSat shut-off stamps",
        )

        inspect_stamp_file(
            TRANSITION_CANDIDATES,
            "GHGSat transition stamps",
        )

    output_rows = inspect_matched_table(
        scenes
    )

    output_df = pd.DataFrame(
        output_rows
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 100)
    print("SAVED")
    print("=" * 100)

    print(f"\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
