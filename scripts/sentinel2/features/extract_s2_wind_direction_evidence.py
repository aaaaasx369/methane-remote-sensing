from pathlib import Path
import re

import numpy as np
import pandas as pd


MANIFEST_INPUT = Path(
    "outputs/317_s2_low_emission_scene_manifest_v1.csv"
)

CSV_OUTPUT = Path(
    "outputs/337_s2_low_emission_wind_evidence.csv"
)

REPORT_OUTPUT = Path(
    "outputs/338_s2_low_emission_wind_evidence_report.txt"
)

SEARCH_MINUTES = 30

WIND_PATTERN = re.compile(
    r"wind|direction|bearing|azimuth|"
    r"meteorolog|speed|velocity|"
    r"\bwd\b|\bws\b",
    flags=re.IGNORECASE,
)

TIME_PATTERN = re.compile(
    r"time|date|timestamp|utc",
    flags=re.IGNORECASE,
)


def parse_bool(value):
    return (
        str(value).strip().lower()
        in {"true", "1", "yes"}
    )


def parse_time_series(
    series,
    acquisition_time,
):
    text = (
        series.astype("string")
        .str.strip()
    )

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns, UTC]",
    )

    # 完整日期時間。
    has_year = text.str.contains(
        r"(?:19|20)\d{2}",
        regex=True,
        na=False,
    )

    result.loc[has_year] = pd.to_datetime(
        text.loc[has_year],
        errors="coerce",
        utc=True,
    )

    # 只有時分秒，例如 18:34:54。
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

    result.loc[use_time_only] = pd.to_datetime(
        combined,
        errors="coerce",
        utc=True,
    )

    return result


def angular_difference(
    angle_a,
    angle_b,
):
    return abs(
        (
            float(angle_a)
            - float(angle_b)
            + 180
        ) % 360
        - 180
    )


def main():
    manifest = pd.read_csv(
        MANIFEST_INPUT,
        low_memory=False,
    )

    manifest = manifest[
        manifest["primary_include"].map(
            parse_bool
        )
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
    report_lines = []

    report_lines.append(
        "=" * 115
    )

    report_lines.append(
        "SENTINEL-2 LOW-EMISSION WIND EVIDENCE"
    )

    report_lines.append(
        "=" * 115
    )

    for _, scene in manifest.iterrows():
        acquisition = scene[
            "acquisition_time_utc"
        ]

        source_path = Path(
            str(scene["source_file"])
        )

        report_lines.append("")
        report_lines.append(
            "#" * 115
        )

        report_lines.append(
            f"Site: {scene['site']}"
        )

        report_lines.append(
            f"Acquisition: {acquisition}"
        )

        report_lines.append(
            "Release rate: "
            f"{scene['final_release_rate_kg_h']} kg/h"
        )

        report_lines.append(
            f"Source file: {source_path}"
        )

        report_lines.append(
            "Observed anomaly direction: "
            "NW, approximately 315 degrees"
        )

        report_lines.append(
            "Expected meteorological wind-from "
            "direction for an NW plume: "
            "SE, approximately 135 degrees"
        )

        if not source_path.exists():
            report_lines.append(
                "ERROR: source file not found"
            )
            continue

        raw = pd.read_csv(
            source_path,
            low_memory=False,
        )

        wind_columns = [
            column
            for column in raw.columns
            if WIND_PATTERN.search(
                str(column)
            )
        ]

        time_columns = [
            column
            for column in raw.columns
            if TIME_PATTERN.search(
                str(column)
            )
        ]

        report_lines.append(
            "\nWind-related columns:"
        )

        if wind_columns:
            for column in wind_columns:
                report_lines.append(
                    f"  {column}"
                )
        else:
            report_lines.append(
                "  [none found]"
            )

        report_lines.append(
            "\nTime-related columns:"
        )

        for column in time_columns:
            report_lines.append(
                f"  {column}"
            )

        scene_records = []

        for time_column in time_columns:
            parsed = parse_time_series(
                raw[time_column],
                acquisition,
            )

            difference_minutes = (
                parsed - acquisition
            ).abs().dt.total_seconds() / 60

            near_mask = (
                parsed.notna()
                & difference_minutes.le(
                    SEARCH_MINUTES
                )
            )

            near_indices = (
                difference_minutes[
                    near_mask
                ]
                .sort_values()
                .head(10)
                .index
            )

            for raw_index in near_indices:
                for wind_column in wind_columns:
                    value = raw.at[
                        raw_index,
                        wind_column,
                    ]

                    if pd.isna(value):
                        continue

                    text = str(value).strip()

                    if (
                        not text
                        or text.lower()
                        in {"nan", "none", "null", "-"}
                    ):
                        continue

                    numeric_value = pd.to_numeric(
                        pd.Series([value]),
                        errors="coerce",
                    ).iloc[0]

                    if (
                        pd.notna(numeric_value)
                        and 0 <= numeric_value <= 360
                    ):
                        difference_to_nw = (
                            angular_difference(
                                numeric_value,
                                315,
                            )
                        )

                        difference_to_se = (
                            angular_difference(
                                numeric_value,
                                135,
                            )
                        )
                    else:
                        difference_to_nw = np.nan
                        difference_to_se = np.nan

                    scene_records.append({
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
                            parsed.loc[
                                raw_index
                            ],

                        "time_difference_minutes":
                            float(
                                difference_minutes.loc[
                                    raw_index
                                ]
                            ),

                        "wind_column":
                            wind_column,

                        "wind_value":
                            value,

                        "numeric_wind_value":
                            numeric_value,

                        "difference_to_315_NW_degrees":
                            difference_to_nw,

                        "difference_to_135_SE_degrees":
                            difference_to_se,
                    })

        scene_evidence = pd.DataFrame(
            scene_records
        )

        if scene_evidence.empty:
            report_lines.append(
                "\nNo non-empty wind evidence found "
                f"within ±{SEARCH_MINUTES} minutes."
            )

        else:
            scene_evidence = (
                scene_evidence.sort_values(
                    [
                        "time_difference_minutes",
                        "wind_column",
                    ]
                )
                .drop_duplicates(
                    subset=[
                        "raw_row_index",
                        "time_column",
                        "wind_column",
                        "wind_value",
                    ]
                )
            )

            evidence_rows.extend(
                scene_evidence.to_dict(
                    "records"
                )
            )

            report_lines.append(
                "\nNearest wind evidence:"
            )

            for _, row in (
                scene_evidence.head(40).iterrows()
            ):
                report_lines.append(
                    "\n"
                    f"  Raw row: "
                    f"{row['raw_row_index']}"
                )

                report_lines.append(
                    "  Parsed time: "
                    f"{row['parsed_time_utc']}"
                )

                report_lines.append(
                    "  Time difference: "
                    f"{row['time_difference_minutes']:.3f} min"
                )

                report_lines.append(
                    "  Field: "
                    f"{row['wind_column']}"
                )

                report_lines.append(
                    "  Value: "
                    f"{row['wind_value']}"
                )

                if pd.notna(
                    row[
                        "difference_to_315_NW_degrees"
                    ]
                ):
                    report_lines.append(
                        "  Difference from NW/315°: "
                        f"{row['difference_to_315_NW_degrees']:.2f}°"
                    )

                    report_lines.append(
                        "  Difference from SE/135°: "
                        f"{row['difference_to_135_SE_degrees']:.2f}°"
                    )

    evidence = pd.DataFrame(
        evidence_rows
    )

    evidence.to_csv(
        CSV_OUTPUT,
        index=False,
    )

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 100)
    print("WIND EVIDENCE EXTRACTION")
    print("=" * 100)

    print(
        "\nPrimary scenes:",
        len(manifest),
    )

    print(
        "Wind evidence rows:",
        len(evidence),
    )

    if not evidence.empty:
        print("\nWind fields:")
        print(
            evidence[
                "wind_column"
            ].value_counts()
        )

        print("\nNearest evidence:")
        print(
            evidence.sort_values(
                "time_difference_minutes"
            )[
                [
                    "site",
                    "release_rate_kg_h",
                    "time_difference_minutes",
                    "wind_column",
                    "wind_value",
                    "difference_to_315_NW_degrees",
                    "difference_to_135_SE_degrees",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    print("\nSaved:")
    print(CSV_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
