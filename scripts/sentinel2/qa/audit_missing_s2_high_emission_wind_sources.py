from pathlib import Path
import re

import pandas as pd

import extract_and_lock_s2_high_emission_wind_plan as base


POSITIVE_INPUT = Path(
    "outputs/355_s2_high_emission_positive_manifest_clean_v1.csv"
)

WIND_PLAN_INPUT = Path(
    "outputs/377_s2_high_emission_wind_plan_v1.csv"
)

COLUMN_OUTPUT = Path(
    "outputs/380_s2_high_emission_missing_wind_column_audit_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/381_s2_high_emission_missing_wind_source_report_v1.txt"
)


WIND_PATTERN = re.compile(
    r"wind|direction|dir_|dirmean|speed|velocity|"
    r"weather|meteor|met_|u10|v10|gust|bearing",
    flags=re.IGNORECASE,
)


def main():
    positives = pd.read_csv(
        POSITIVE_INPUT,
        low_memory=False,
    )

    plan = pd.read_csv(
        WIND_PLAN_INPUT,
        low_memory=False,
    )

    missing_plan = plan[
        plan[
            "wind_plan_status"
        ].astype(str).eq(
            "missing_short_term_direction"
        )
    ][
        [
            "positive_id",
            "wind_plan_status",
        ]
    ]

    missing = positives.merge(
        missing_plan,
        on="positive_id",
        how="inner",
        validate="one_to_one",
    )

    if missing.empty:
        print(
            "沒有 missing wind scenes。"
        )
        return

    missing[
        "acquisition_time_utc"
    ] = pd.to_datetime(
        missing[
            "acquisition_time_utc"
        ],
        errors="raise",
        utc=True,
    )

    audit_rows = []
    report_lines = [
        "=" * 110,
        (
            "SENTINEL-2 HIGH-EMISSION "
            "MISSING WIND SOURCE AUDIT"
        ),
        "=" * 110,
        "",
        f"Missing scenes: {len(missing)}",
    ]

    table_cache = {}

    for _, row in missing.iterrows():
        positive_id = str(
            row["positive_id"]
        )

        source_path = (
            base.resolve_source_path(
                row["raw_source_file"]
            )
        )

        sheet_name = (
            base.normalize_sheet_name(
                row.get(
                    "raw_source_sheet",
                    None,
                )
            )
        )

        report_lines.extend([
            "",
            "-" * 110,
            (
                f"{positive_id} | "
                f"{row['release_rate_kg_h']:.6f} kg/h"
            ),
            (
                f"Acquisition: "
                f"{row['acquisition_time_utc']}"
            ),
            f"Source: {source_path}",
            f"Sheet: {sheet_name}",
        ])

        print(
            "\n" + "-" * 110
        )

        print(
            positive_id,
            "|",
            row[
                "release_rate_kg_h"
            ],
            "kg/h",
        )

        print(
            "Source:",
            source_path,
        )

        if not source_path.exists():
            report_lines.append(
                "STATUS: source file missing"
            )

            audit_rows.append({
                "positive_id":
                    positive_id,

                "source_file":
                    str(source_path),

                "source_sheet":
                    sheet_name,

                "audit_status":
                    "source_file_missing",

                "column_name":
                    pd.NA,

                "selected_row_value":
                    pd.NA,
            })

            continue

        cache_key = (
            str(source_path),
            sheet_name,
        )

        try:
            if cache_key not in table_cache:
                table_cache[
                    cache_key
                ] = base.read_source_table(
                    source_path,
                    sheet_name,
                )

            frame = table_cache[
                cache_key
            ]

            (
                selected_row,
                selected_index,
                selection_method,
                time_difference,
            ) = base.choose_source_row(
                frame,
                row,
                row[
                    "acquisition_time_utc"
                ],
            )

        except Exception as error:
            report_lines.append(
                "STATUS: source row selection failed"
            )

            report_lines.append(
                f"ERROR: {error}"
            )

            audit_rows.append({
                "positive_id":
                    positive_id,

                "source_file":
                    str(source_path),

                "source_sheet":
                    sheet_name,

                "audit_status":
                    "row_selection_failed",

                "column_name":
                    pd.NA,

                "selected_row_value":
                    str(error),
            })

            continue

        candidate_columns = [
            column
            for column in frame.columns
            if WIND_PATTERN.search(
                str(column)
            )
        ]

        report_lines.extend([
            (
                f"Selected source row: "
                f"{selected_index}"
            ),
            (
                f"Selection method: "
                f"{selection_method}"
            ),
            (
                "Time difference: "
                f"{time_difference} minutes"
            ),
            (
                "Candidate wind/weather "
                f"columns: {len(candidate_columns)}"
            ),
        ])

        print(
            "Selected source row:",
            selected_index,
        )

        print(
            "Candidate wind/weather columns:",
            len(candidate_columns),
        )

        if not candidate_columns:
            report_lines.append(
                (
                    "RESULT: no wind-like columns "
                    "were found in this source table."
                )
            )

            audit_rows.append({
                "positive_id":
                    positive_id,

                "event_id":
                    row["event_id"],

                "scene_id":
                    row["scene_id"],

                "release_rate_kg_h":
                    row[
                        "release_rate_kg_h"
                    ],

                "acquisition_time_utc":
                    row[
                        "acquisition_time_utc"
                    ],

                "source_file":
                    str(source_path),

                "source_sheet":
                    sheet_name,

                "selected_source_row":
                    selected_index,

                "source_row_selection_method":
                    selection_method,

                "source_row_time_difference_minutes":
                    time_difference,

                "audit_status":
                    "no_wind_like_columns",

                "column_name":
                    pd.NA,

                "normalized_column_name":
                    pd.NA,

                "selected_row_value":
                    pd.NA,
            })

            print(
                "  No wind-like columns found."
            )

            continue

        for column in candidate_columns:
            value = selected_row.get(
                column,
                pd.NA,
            )

            normalized = (
                base.normalize_column_name(
                    column
                )
            )

            audit_rows.append({
                "positive_id":
                    positive_id,

                "event_id":
                    row["event_id"],

                "scene_id":
                    row["scene_id"],

                "release_rate_kg_h":
                    row[
                        "release_rate_kg_h"
                    ],

                "acquisition_time_utc":
                    row[
                        "acquisition_time_utc"
                    ],

                "source_file":
                    str(source_path),

                "source_sheet":
                    sheet_name,

                "selected_source_row":
                    selected_index,

                "source_row_selection_method":
                    selection_method,

                "source_row_time_difference_minutes":
                    time_difference,

                "audit_status":
                    "candidate_column_found",

                "column_name":
                    column,

                "normalized_column_name":
                    normalized,

                "selected_row_value":
                    value,
            })

            line = (
                f"  {column!r} = {value!r}"
            )

            report_lines.append(line)
            print(line)

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        COLUMN_OUTPUT,
        index=False,
    )

    scene_summary = (
        audit.groupby(
            "positive_id"
        )["audit_status"]
        .agg(
            rows="size",
            statuses=lambda values:
                "|".join(
                    sorted(
                        set(
                            values.astype(str)
                        )
                    )
                ),
        )
    )

    report_lines.extend([
        "",
        "=" * 110,
        "SUMMARY",
        "=" * 110,
        scene_summary.to_string(),
        "",
        f"Saved: {COLUMN_OUTPUT}",
    ])

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print("MISSING WIND AUDIT SUMMARY")
    print("=" * 110)

    print(scene_summary)

    print("\nSaved:")
    print(COLUMN_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
