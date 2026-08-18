from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    "/Users/happydoraaa/methane_release_project"
)

POSITIVE_INPUT = Path(
    "outputs/355_s2_high_emission_positive_manifest_clean_v1.csv"
)

WIND_EVIDENCE_OUTPUT = Path(
    "outputs/376_s2_high_emission_true_wind_evidence_v1.csv"
)

WIND_PLAN_OUTPUT = Path(
    "outputs/377_s2_high_emission_wind_plan_v1.csv"
)

MISSING_OUTPUT = Path(
    "outputs/378_s2_high_emission_missing_wind_audit_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/379_s2_high_emission_wind_plan_report_v1.txt"
)


SHORT_TERM_WINDOWS = [
    30,
    60,
    90,
    300,
]

CONTEXT_WINDOWS = [
    600,
    900,
]

ALL_WINDOWS = (
    SHORT_TERM_WINDOWS
    + CONTEXT_WINDOWS
)

MAX_ACCEPTABLE_DIRECTION_DEVIATION = 45.0
MAX_ROW_TIME_DIFFERENCE_MINUTES = 20.0


def normalize_column_name(value):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


def normalize_sheet_name(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if (
        not value
        or value.lower()
        in {
            "nan",
            "none",
            "default",
        }
    ):
        return None

    return value


def resolve_source_path(value):
    path = Path(str(value))

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def read_source_table(
    path,
    sheet_name,
):
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(
            path,
            low_memory=False,
        )

    if suffix in {
        ".xlsx",
        ".xls",
        ".xlsm",
    }:
        kwargs = {
            "sheet_name":
                sheet_name
                if sheet_name
                is not None
                else 0,
        }

        return pd.read_excel(
            path,
            **kwargs,
        )

    if suffix == ".parquet":
        return pd.read_parquet(
            path
        )

    raise ValueError(
        f"不支援的來源格式：{path}"
    )


def candidate_datetime_columns(frame):
    keywords = [
        "time",
        "date",
        "timestamp",
        "operator",
        "stanford",
        "survey",
        "overpass",
        "acquisition",
    ]

    return [
        column
        for column in frame.columns
        if any(
            keyword
            in str(column).lower()
            for keyword in keywords
        )
    ]


def row_time_difference_minutes(
    row,
    datetime_columns,
    target_time,
):
    differences = []

    for column in datetime_columns:
        try:
            value = pd.to_datetime(
                row[column],
                errors="coerce",
                utc=True,
            )
        except Exception:
            continue

        if pd.isna(value):
            continue

        difference = abs(
            (
                value
                - target_time
            ).total_seconds()
        ) / 60.0

        differences.append(
            difference
        )

    if not differences:
        return np.nan

    return float(
        min(differences)
    )


def search_best_row_by_time(
    frame,
    target_time,
):
    datetime_columns = (
        candidate_datetime_columns(
            frame
        )
    )

    best = None

    for column in datetime_columns:
        try:
            values = pd.to_datetime(
                frame[column],
                errors="coerce",
                utc=True,
            )
        except Exception:
            continue

        differences = (
            values - target_time
        ).abs().dt.total_seconds() / 60.0

        if differences.notna().sum() == 0:
            continue

        index = differences.idxmin()
        difference = float(
            differences.loc[index]
        )

        if (
            best is None
            or difference
            < best[
                "difference_minutes"
            ]
        ):
            best = {
                "index":
                    index,

                "column":
                    column,

                "difference_minutes":
                    difference,
            }

    return best


def choose_source_row(
    frame,
    manifest_row,
    target_time,
):
    datetime_columns = (
        candidate_datetime_columns(
            frame
        )
    )

    raw_index = manifest_row.get(
        "raw_row_index",
        pd.NA,
    )

    if pd.notna(raw_index):
        try:
            raw_index_int = int(
                float(raw_index)
            )

            if raw_index_int in frame.index:
                candidate = frame.loc[
                    raw_index_int
                ]

                difference = (
                    row_time_difference_minutes(
                        candidate,
                        datetime_columns,
                        target_time,
                    )
                )

                if (
                    pd.isna(difference)
                    or difference
                    <= MAX_ROW_TIME_DIFFERENCE_MINUTES
                ):
                    return (
                        candidate,
                        raw_index_int,
                        "raw_row_index_loc",
                        difference,
                    )

            if (
                0
                <= raw_index_int
                < len(frame)
            ):
                candidate = frame.iloc[
                    raw_index_int
                ]

                difference = (
                    row_time_difference_minutes(
                        candidate,
                        datetime_columns,
                        target_time,
                    )
                )

                if (
                    pd.isna(difference)
                    or difference
                    <= MAX_ROW_TIME_DIFFERENCE_MINUTES
                ):
                    return (
                        candidate,
                        candidate.name,
                        "raw_row_index_iloc",
                        difference,
                    )

        except Exception:
            pass

    best = search_best_row_by_time(
        frame,
        target_time,
    )

    if best is None:
        raise RuntimeError(
            "無法根據 raw_row_index 或時間找到來源列。"
        )

    return (
        frame.loc[
            best["index"]
        ],
        best["index"],
        (
            "datetime_match:"
            + str(
                best["column"]
            )
        ),
        best[
            "difference_minutes"
        ],
    )


def identify_wind_columns(columns):
    direction_candidates = {}
    speed_candidates = {}

    for column in columns:
        normalized = (
            normalize_column_name(
                column
            )
        )

        excluded_direction_terms = [
            "sigma",
            "error",
            "uncertainty",
            "lower",
            "upper",
            "std",
            "speed",
        ]

        for window in ALL_WINDOWS:
            window_text = str(window)

            direction_patterns = [
                f"winddirmean{window_text}",
                f"winddirectionmean{window_text}",
                f"winddirection{window_text}",
                f"winddir{window_text}",
            ]

            if (
                any(
                    normalized.endswith(
                        pattern
                    )
                    for pattern
                    in direction_patterns
                )
                and not any(
                    term in normalized
                    for term
                    in excluded_direction_terms
                )
            ):
                direction_candidates.setdefault(
                    window,
                    [],
                ).append(column)

            speed_patterns = [
                f"windspeedmean{window_text}",
                f"windspeed{window_text}",
                f"windvelmean{window_text}",
                f"windvelocitymean{window_text}",
            ]

            if any(
                normalized.endswith(
                    pattern
                )
                for pattern
                in speed_patterns
            ):
                speed_candidates.setdefault(
                    window,
                    [],
                ).append(column)

    return (
        direction_candidates,
        speed_candidates,
    )


def choose_best_column(
    candidates,
    window,
    kind,
):
    if not candidates:
        return None

    def priority(column):
        normalized = (
            normalize_column_name(
                column
            )
        )

        if kind == "direction":
            preferred = [
                f"winddirmean{window}",
                f"winddirectionmean{window}",
                f"winddirection{window}",
                f"winddir{window}",
            ]
        else:
            preferred = [
                f"windspeedmean{window}",
                f"windspeed{window}",
                f"windvelmean{window}",
                f"windvelocitymean{window}",
            ]

        try:
            return preferred.index(
                normalized
            )
        except ValueError:
            return len(preferred)

    return sorted(
        candidates,
        key=priority,
    )[0]


def numeric_value(value):
    result = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(result):
        return np.nan

    return float(result)


def circular_mean_degrees(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return np.nan, np.nan

    radians = np.deg2rad(
        values
    )

    x = np.mean(
        np.cos(radians)
    )

    y = np.mean(
        np.sin(radians)
    )

    mean_direction = (
        np.degrees(
            np.arctan2(
                y,
                x,
            )
        )
        + 360.0
    ) % 360.0

    concentration = float(
        np.sqrt(
            x ** 2
            + y ** 2
        )
    )

    return (
        float(mean_direction),
        concentration,
    )


def circular_difference(
    angle_a,
    angle_b,
):
    return abs(
        (
            angle_a
            - angle_b
            + 180.0
        ) % 360.0
        - 180.0
    )


def main():
    positives = pd.read_csv(
        POSITIVE_INPUT,
        low_memory=False,
    )

    required_columns = [
        "positive_id",
        "event_id",
        "site",
        "scene_id",
        "acquisition_time_utc",
        "release_rate_kg_h",
        "raw_source_file",
    ]

    missing = [
        column
        for column in required_columns
        if column
        not in positives.columns
    ]

    if missing:
        raise KeyError(
            "Positive manifest 缺少欄位："
            + ", ".join(missing)
        )

    positives[
        "acquisition_time_utc"
    ] = pd.to_datetime(
        positives[
            "acquisition_time_utc"
        ],
        errors="raise",
        utc=True,
    )

    positives[
        "release_rate_kg_h"
    ] = pd.to_numeric(
        positives[
            "release_rate_kg_h"
        ],
        errors="raise",
    )

    table_cache = {}

    evidence_rows = []
    plan_rows = []
    missing_rows = []

    print("=" * 115)
    print(
        "EXTRACT AND LOCK SENTINEL-2 "
        "HIGH-EMISSION WIND PLAN"
    )
    print("=" * 115)

    for number, row in (
        positives.iterrows()
    ):
        positive_id = str(
            row["positive_id"]
        )

        target_time = row[
            "acquisition_time_utc"
        ]

        source_path = (
            resolve_source_path(
                row[
                    "raw_source_file"
                ]
            )
        )

        sheet_name = (
            normalize_sheet_name(
                row.get(
                    "raw_source_sheet",
                    None,
                )
            )
        )

        print(
            f"\n[{number + 1}/{len(positives)}] "
            f"{positive_id} | "
            f"{row['release_rate_kg_h']:.3f} kg/h",
            flush=True,
        )

        print(
            "  Source:",
            source_path,
        )

        if not source_path.exists():
            print(
                "  Missing source file"
            )

            missing_rows.append({
                "positive_id":
                    positive_id,

                "scene_id":
                    row["scene_id"],

                "release_rate_kg_h":
                    row[
                        "release_rate_kg_h"
                    ],

                "problem":
                    "source_file_not_found",

                "detail":
                    str(source_path),
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
                ] = read_source_table(
                    source_path,
                    sheet_name,
                )

            source_table = (
                table_cache[
                    cache_key
                ]
            )

            (
                source_row,
                selected_row_index,
                selection_method,
                row_time_difference,
            ) = choose_source_row(
                source_table,
                row,
                target_time,
            )

        except Exception as error:
            print(
                "  Row selection failed:",
                error,
            )

            missing_rows.append({
                "positive_id":
                    positive_id,

                "scene_id":
                    row["scene_id"],

                "release_rate_kg_h":
                    row[
                        "release_rate_kg_h"
                    ],

                "problem":
                    "source_row_selection_failed",

                "detail":
                    str(error),
            })

            continue

        (
            direction_candidates,
            speed_candidates,
        ) = identify_wind_columns(
            source_table.columns
        )

        scene_evidence = []

        for window in ALL_WINDOWS:
            direction_column = (
                choose_best_column(
                    direction_candidates.get(
                        window,
                        [],
                    ),
                    window,
                    "direction",
                )
            )

            speed_column = (
                choose_best_column(
                    speed_candidates.get(
                        window,
                        [],
                    ),
                    window,
                    "speed",
                )
            )

            wind_from = (
                numeric_value(
                    source_row[
                        direction_column
                    ]
                )
                if direction_column
                is not None
                else np.nan
            )

            wind_speed = (
                numeric_value(
                    source_row[
                        speed_column
                    ]
                )
                if speed_column
                is not None
                else np.nan
            )

            if np.isfinite(wind_from):
                wind_from = (
                    wind_from
                    % 360.0
                )

                downwind = (
                    wind_from
                    + 180.0
                ) % 360.0
            else:
                downwind = np.nan

            evidence_record = {
                "positive_id":
                    positive_id,

                "event_id":
                    row["event_id"],

                "scene_id":
                    row["scene_id"],

                "site":
                    row["site"],

                "acquisition_time_utc":
                    target_time,

                "release_rate_kg_h":
                    row[
                        "release_rate_kg_h"
                    ],

                "window_seconds":
                    window,

                "window_role":
                    (
                        "short_term_plan"
                        if window
                        in SHORT_TERM_WINDOWS
                        else "context_only"
                    ),

                "wind_direction_column":
                    direction_column,

                "wind_from_degrees":
                    wind_from,

                "downwind_degrees":
                    downwind,

                "wind_speed_column":
                    speed_column,

                "wind_speed_m_s":
                    wind_speed,

                "source_file":
                    str(source_path),

                "source_sheet":
                    sheet_name,

                "selected_source_row_index":
                    selected_row_index,

                "source_row_selection_method":
                    selection_method,

                "source_row_time_difference_minutes":
                    row_time_difference,
            }

            evidence_rows.append(
                evidence_record
            )

            scene_evidence.append(
                evidence_record
            )

        scene_evidence = pd.DataFrame(
            scene_evidence
        )

        short_term = scene_evidence[
            scene_evidence[
                "window_seconds"
            ].isin(
                SHORT_TERM_WINDOWS
            )
            & scene_evidence[
                "downwind_degrees"
            ].notna()
        ].copy()

        valid_count = len(
            short_term
        )

        fixed_downwind, concentration = (
            circular_mean_degrees(
                short_term[
                    "downwind_degrees"
                ].to_numpy()
            )
        )

        if (
            valid_count > 0
            and np.isfinite(
                fixed_downwind
            )
        ):
            deviations = (
                short_term[
                    "downwind_degrees"
                ]
                .map(
                    lambda value:
                        circular_difference(
                            value,
                            fixed_downwind,
                        )
                )
            )

            max_deviation = float(
                deviations.max()
            )

            fixed_upwind = (
                fixed_downwind
                + 180.0
            ) % 360.0

        else:
            max_deviation = np.nan
            fixed_upwind = np.nan

        if valid_count >= 2:
            if (
                max_deviation
                <= MAX_ACCEPTABLE_DIRECTION_DEVIATION
            ):
                status = (
                    "locked_short_term_circular_mean"
                )
            else:
                status = (
                    "manual_review_large_direction_shift"
                )

        elif valid_count == 1:
            status = (
                "manual_review_only_one_direction"
            )

        else:
            status = (
                "missing_short_term_direction"
            )

        plan_rows.append({
            "positive_id":
                positive_id,

            "event_id":
                row["event_id"],

            "scene_id":
                row["scene_id"],

            "site":
                row["site"],

            "acquisition_time_utc":
                target_time,

            "release_rate_kg_h":
                row[
                    "release_rate_kg_h"
                ],

            "valid_short_term_window_count":
                valid_count,

            "available_short_term_windows":
                "|".join(
                    short_term[
                        "window_seconds"
                    ]
                    .astype(str)
                    .tolist()
                ),

            "fixed_downwind_direction_degrees":
                fixed_downwind,

            "fixed_upwind_direction_degrees":
                fixed_upwind,

            "short_term_circular_concentration":
                concentration,

            "maximum_short_term_deviation_degrees":
                max_deviation,

            "wind_plan_status":
                status,

            "direction_method":
                (
                    "equal_weight_circular_mean_of_"
                    "meteorological_downwind_"
                    "directions_for_30_60_90_300s"
                ),

            "context_windows_excluded_from_plan":
                "600|900",

            "wind_plan_version":
                "s2_high_emission_wind_plan_v1",
        })

        print(
            "  Valid short-term windows:",
            valid_count,
        )

        print(
            "  Fixed downwind:",
            (
                f"{fixed_downwind:.2f}°"
                if np.isfinite(
                    fixed_downwind
                )
                else "missing"
            ),
        )

        print(
            "  Maximum deviation:",
            (
                f"{max_deviation:.2f}°"
                if np.isfinite(
                    max_deviation
                )
                else "missing"
            ),
        )

        print(
            "  Status:",
            status,
        )

    evidence = pd.DataFrame(
        evidence_rows
    )

    plan = pd.DataFrame(
        plan_rows
    )

    missing_audit = pd.DataFrame(
        missing_rows
    )

    evidence.to_csv(
        WIND_EVIDENCE_OUTPUT,
        index=False,
    )

    plan.to_csv(
        WIND_PLAN_OUTPUT,
        index=False,
    )

    missing_audit.to_csv(
        MISSING_OUTPUT,
        index=False,
    )

    locked_count = int(
        plan[
            "wind_plan_status"
        ].eq(
            "locked_short_term_circular_mean"
        ).sum()
    )

    manual_count = int(
        plan[
            "wind_plan_status"
        ].str.startswith(
            "manual_review",
            na=False,
        ).sum()
    )

    missing_count = (
        len(positives)
        - locked_count
        - manual_count
    )

    report_lines = [
        "=" * 115,
        (
            "SENTINEL-2 HIGH-EMISSION "
            "WIND PLAN REPORT V1"
        ),
        "=" * 115,
        "",
        f"Positive scenes: {len(positives)}",
        f"Wind-plan rows: {len(plan)}",
        f"Automatically locked: {locked_count}",
        f"Manual review required: {manual_count}",
        f"Missing/unresolved: {missing_count}",
        "",
        "Method:",
        (
            "Wind direction fields represent the "
            "meteorological direction from which the "
            "wind originated. Downwind direction was "
            "calculated by adding 180 degrees."
        ),
        (
            "The locked direction is the equal-weight "
            "circular mean of the available 30, 60, "
            "90 and 300 second downwind directions."
        ),
        (
            "The 600 and 900 second fields are retained "
            "as context but excluded from the locked "
            "near-source direction."
        ),
        "",
        "Wind plan:",
        (
            plan[
                [
                    "positive_id",
                    "release_rate_kg_h",
                    "available_short_term_windows",
                    "fixed_downwind_direction_degrees",
                    "maximum_short_term_deviation_degrees",
                    "wind_plan_status",
                ]
            ].to_string(index=False)
            if not plan.empty
            else "None"
        ),
    ]

    if not missing_audit.empty:
        report_lines.extend([
            "",
            "Missing wind audit:",
            missing_audit.to_string(
                index=False
            ),
        ])

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 115)
    print("HIGH-EMISSION WIND PLAN SUMMARY")
    print("=" * 115)

    print(
        "\nAutomatically locked:",
        locked_count,
    )

    print(
        "Manual review required:",
        manual_count,
    )

    print(
        "Missing/unresolved:",
        missing_count,
    )

    print("\nWind plan:")

    if plan.empty:
        print("None")
    else:
        print(
            plan[
                [
                    "positive_id",
                    "release_rate_kg_h",
                    "available_short_term_windows",
                    "fixed_downwind_direction_degrees",
                    "fixed_upwind_direction_degrees",
                    "maximum_short_term_deviation_degrees",
                    "wind_plan_status",
                ]
            ].to_string(
                index=False
            )
        )

    if not missing_audit.empty:
        print("\nMissing wind audit:")
        print(
            missing_audit.to_string(
                index=False
            )
        )

    print("\nSaved:")
    print(WIND_EVIDENCE_OUTPUT)
    print(WIND_PLAN_OUTPUT)
    print(MISSING_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
