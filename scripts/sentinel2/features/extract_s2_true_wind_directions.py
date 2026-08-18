from pathlib import Path
import numpy as np
import pandas as pd


MANIFEST_INPUT = Path(
    "outputs/317_s2_low_emission_scene_manifest_v1.csv"
)

CSV_OUTPUT = Path(
    "outputs/339_s2_true_wind_direction_evidence.csv"
)

REPORT_OUTPUT = Path(
    "outputs/340_s2_true_wind_direction_report.txt"
)


SEARCH_MINUTES = 30

DIRECTION_COLUMNS = [
    "WindDirection",
    "Wind_dir_mean30",
    "Wind_dir_mean60",
    "Wind_dir_mean90",
    "Wind_dir_mean300",
    "Wind_dir_mean600",
    "Wind_dir_mean900",
]

SPEED_COLUMNS = [
    "WindSpeed",
    "Wind_MPS_mean30",
    "Wind_MPS_mean60",
    "Wind_MPS_mean90",
    "Wind_MPS_mean300",
    "Wind_MPS_mean600",
    "Wind_MPS_mean900",
]

TIME_COLUMNS = [
    "Operator_Timestamp",
    "Stanford_timestamp",
    "datetime_UTC",
    "DateOfSurvey",
]

PLUME_DIRECTION_DEGREES = 315.0


def parse_bool(value):
    return (
        str(value).strip().lower()
        in {"true", "1", "yes"}
    )


def angular_difference(angle_a, angle_b):
    return abs(
        (
            float(angle_a)
            - float(angle_b)
            + 180
        ) % 360
        - 180
    )


def parse_times(series, acquisition_time):
    text = (
        series.astype("string")
        .str.strip()
    )

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns, UTC]",
    )

    full_datetime = text.str.contains(
        r"(?:19|20)\d{2}",
        regex=True,
        na=False,
    )

    result.loc[full_datetime] = (
        pd.to_datetime(
            text.loc[full_datetime],
            errors="coerce",
            utc=True,
        )
    )

    time_only = text.str.match(
        r"^\s*(?:[01]?\d|2[0-3]):"
        r"[0-5]\d"
        r"(?::[0-5]\d(?:\.\d+)?)?"
        r"\s*$",
        na=False,
    )

    use_time_only = (
        time_only
        & result.isna()
    )

    combined = (
        acquisition_time.strftime(
            "%Y-%m-%d"
        )
        + " "
        + text.loc[use_time_only]
    )

    result.loc[use_time_only] = (
        pd.to_datetime(
            combined,
            errors="coerce",
            utc=True,
        )
    )

    return result


def get_context_value(row, columns):
    values = []

    for column in columns:
        if column not in row.index:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        text = str(value).strip()

        if (
            text
            and text.lower()
            not in {"nan", "none", "null", "-"}
        ):
            values.append(
                f"{column}={text}"
            )

    return " | ".join(values)


def main():
    manifest = pd.read_csv(
        MANIFEST_INPUT,
        low_memory=False,
    )

    manifest = manifest[
        manifest[
            "primary_include"
        ].map(parse_bool)
    ].copy()

    manifest[
        "acquisition_time_utc"
    ] = pd.to_datetime(
        manifest[
            "acquisition_time_utc"
        ],
        errors="coerce",
        utc=True,
    )

    evidence_rows = []

    for _, scene in manifest.iterrows():
        acquisition = scene[
            "acquisition_time_utc"
        ]

        source_path = Path(
            str(scene["source_file"])
        )

        if not source_path.exists():
            print(
                "Missing source:",
                source_path,
            )
            continue

        raw = pd.read_csv(
            source_path,
            low_memory=False,
        )

        available_direction_columns = [
            column
            for column in DIRECTION_COLUMNS
            if column in raw.columns
        ]

        available_time_columns = [
            column
            for column in TIME_COLUMNS
            if column in raw.columns
        ]

        for time_column in (
            available_time_columns
        ):
            parsed_times = parse_times(
                raw[time_column],
                acquisition,
            )

            time_difference = (
                parsed_times
                - acquisition
            ).abs().dt.total_seconds() / 60

            nearby_rows = time_difference[
                parsed_times.notna()
                & time_difference.le(
                    SEARCH_MINUTES
                )
            ].sort_values().index

            for raw_index in nearby_rows:
                raw_row = raw.loc[
                    raw_index
                ]

                for direction_column in (
                    available_direction_columns
                ):
                    direction = pd.to_numeric(
                        pd.Series([
                            raw_row[
                                direction_column
                            ]
                        ]),
                        errors="coerce",
                    ).iloc[0]

                    if (
                        pd.isna(direction)
                        or direction < 0
                        or direction > 360
                    ):
                        continue

                    # 假設欄位採氣象慣例：
                    # 風向表示風「從哪裡來」。
                    implied_plume_direction = (
                        direction + 180
                    ) % 360

                    difference_if_wind_from = (
                        angular_difference(
                            implied_plume_direction,
                            PLUME_DIRECTION_DEGREES,
                        )
                    )

                    # 也保留另一種解讀：
                    # 若欄位其實直接表示吹往哪裡。
                    difference_if_wind_to = (
                        angular_difference(
                            direction,
                            PLUME_DIRECTION_DEGREES,
                        )
                    )

                    evidence_rows.append({
                        "site":
                            scene["site"],

                        "release_rate_kg_h":
                            scene[
                                "final_release_rate_kg_h"
                            ],

                        "acquisition_time_utc":
                            acquisition,

                        "source_file":
                            str(source_path),

                        "raw_row_index":
                            int(raw_index),

                        "time_column":
                            time_column,

                        "parsed_time_utc":
                            parsed_times.loc[
                                raw_index
                            ],

                        "time_difference_minutes":
                            float(
                                time_difference.loc[
                                    raw_index
                                ]
                            ),

                        "wind_type":
                            raw_row.get(
                                "WindType"
                            ),

                        "direction_column":
                            direction_column,

                        "wind_direction_degrees":
                            float(direction),

                        "implied_plume_direction_if_wind_from":
                            float(
                                implied_plume_direction
                            ),

                        "difference_from_NW_if_wind_from_degrees":
                            float(
                                difference_if_wind_from
                            ),

                        "difference_from_NW_if_wind_to_degrees":
                            float(
                                difference_if_wind_to
                            ),

                        "wind_speed_context":
                            get_context_value(
                                raw_row,
                                SPEED_COLUMNS,
                            ),
                    })

    evidence = pd.DataFrame(
        evidence_rows
    )

    if not evidence.empty:
        evidence = (
            evidence.sort_values(
                [
                    "acquisition_time_utc",
                    "time_difference_minutes",
                    "wind_type",
                    "direction_column",
                ]
            )
            .drop_duplicates(
                subset=[
                    "site",
                    "acquisition_time_utc",
                    "raw_row_index",
                    "time_column",
                    "wind_type",
                    "direction_column",
                    "wind_direction_degrees",
                ]
            )
            .reset_index(drop=True)
        )

    evidence.to_csv(
        CSV_OUTPUT,
        index=False,
    )

    report_lines = [
        "=" * 115,
        "TRUE SENTINEL-2 WIND-DIRECTION EVIDENCE",
        "=" * 115,
    ]

    for (
        site,
        acquisition,
        release_rate,
    ), group in evidence.groupby(
        [
            "site",
            "acquisition_time_utc",
            "release_rate_kg_h",
        ],
        dropna=False,
    ):
        report_lines.extend([
            "",
            "#" * 115,
            f"Site: {site}",
            f"Acquisition: {acquisition}",
            f"Release rate: {release_rate} kg/h",
            "Observed image anomaly direction: NW = 315°",
            "",
        ])

        nearest = (
            group.sort_values(
                [
                    "time_difference_minutes",
                    "direction_column",
                ]
            )
            .head(40)
        )

        for _, row in nearest.iterrows():
            report_lines.extend([
                (
                    f"Raw row {row['raw_row_index']} | "
                    f"time difference "
                    f"{row['time_difference_minutes']:.3f} min"
                ),
                (
                    f"  Wind type: "
                    f"{row['wind_type']}"
                ),
                (
                    f"  Direction field: "
                    f"{row['direction_column']}"
                ),
                (
                    f"  Recorded wind direction: "
                    f"{row['wind_direction_degrees']:.2f}°"
                ),
                (
                    "  Implied plume direction "
                    "if meteorological wind-from: "
                    f"{row['implied_plume_direction_if_wind_from']:.2f}°"
                ),
                (
                    "  Difference from observed NW plume "
                    "under wind-from interpretation: "
                    f"{row['difference_from_NW_if_wind_from_degrees']:.2f}°"
                ),
                (
                    "  Difference from observed NW plume "
                    "under wind-to interpretation: "
                    f"{row['difference_from_NW_if_wind_to_degrees']:.2f}°"
                ),
                (
                    f"  Speed context: "
                    f"{row['wind_speed_context']}"
                ),
                "",
            ])

    if evidence.empty:
        report_lines.append(
            "\nNo valid directional evidence found."
        )

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 105)
    print("TRUE WIND-DIRECTION EXTRACTION")
    print("=" * 105)

    print(
        "\nDirection evidence rows:",
        len(evidence),
    )

    if not evidence.empty:
        nearest = (
            evidence.sort_values(
                "time_difference_minutes"
            )
            .groupby(
                [
                    "acquisition_time_utc",
                    "wind_type",
                    "direction_column",
                ],
                dropna=False,
            )
            .head(1)
        )

        print("\nNearest direction evidence:")
        print(
            nearest[
                [
                    "release_rate_kg_h",
                    "time_difference_minutes",
                    "wind_type",
                    "direction_column",
                    "wind_direction_degrees",
                    "implied_plume_direction_if_wind_from",
                    "difference_from_NW_if_wind_from_degrees",
                    "difference_from_NW_if_wind_to_degrees",
                ]
            ].to_string(
                index=False,
            )
        )

    print("\nSaved:")
    print(CSV_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
