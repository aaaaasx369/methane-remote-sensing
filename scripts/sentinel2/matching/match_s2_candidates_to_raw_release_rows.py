from pathlib import Path
import re

import numpy as np
import pandas as pd


CANDIDATE_INPUT = Path(
    "outputs/294_s2_positive_within_60min_candidates.csv"
)

RAW_INPUT = Path(
    "outputs/02_candidate_event_rows.csv"
)

OUTPUT = Path(
    "outputs/297_s2_candidate_raw_release_row_matches.csv"
)

BEST_OUTPUT = Path(
    "outputs/298_s2_candidate_best_raw_release_matches.csv"
)


TIME_COLUMNS = [
    "datetime_utc",
    "datetime_UTC",
    "DateTime (UTC)",
    "TimestampUTC",
    "Operator_Timestamp",
    "Stanford_timestamp",
    "overpass_datetime",
    "Timestamp (UTC)",
    "Timestamp",
    "DateOfSurvey",
    "Acquisition time",
    "Acquistion date",
    "Date",
    "date",
    "Time (UTC)",
    "Time (UTC) - from team",
    "Time (UTC) - from Stanford",
    "Time (UTC) - from Flightradar",
]


RELEASE_FIELDS = [
    "StartTime",
    "EndTime",
    "cr_start",
    "cr_end",
    "start_release",
    "end_release",
    "StartUTC",
    "EndUTC",
    "StartTime (UTC)",
    "EndTime (UTC)",
    "Start UTC",
    "End UTC",
    "Start UTC (s since midnight)",
    "End UTC (s since midnight)",
    "overpass_datetime_end",

    "FacilityEmissionRate",
    "FacilityEmissionRate (kg/hr)",
    "FacilityEmissionRate (kg/hour)",
    "UPDATED_ FacilityEmissionRate",
    "release_rate_kgh",
    "release_rate",
    "ch4_kgh_mean",
    "gas_kgh_mean",
    "cr_kgh_CH4_mean30",
    "cr_kgh_CH4_mean60",
    "cr_kgh_CH4_mean90",
    "flow_rate",
    "kgh_ch4",
    "CH4 Emission (kg h-1)",
    "Last 30s (kg/h) - from team",
    "Last 60s (kg/h) - from team",
    "Last 90s (kg/h) - from team",
]


def normalize_path_text(value):
    return (
        str(value)
        .replace("\\", "/")
        .strip()
        .lower()
    )


def parse_timestamp(value, reference_time):
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()

    if not text:
        return pd.NaT

    # 完整日期時間
    if re.search(r"\b(19|20)\d{2}\b", text):
        return pd.to_datetime(
            text,
            errors="coerce",
            utc=True,
        )

    # 只有 HH:MM 或 HH:MM:SS，使用事件日期補齊
    time_match = re.search(
        r"\b([01]?\d|2[0-3]):([0-5]\d)"
        r"(?::([0-5]\d(?:\.\d+)?))?\b",
        text,
    )

    if (
        time_match
        and pd.notna(reference_time)
    ):
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))

        second_text = time_match.group(3)
        second = (
            float(second_text)
            if second_text is not None
            else 0.0
        )

        base = reference_time.normalize()

        return (
            base
            + pd.Timedelta(hours=hour)
            + pd.Timedelta(minutes=minute)
            + pd.Timedelta(seconds=second)
        )

    return pd.NaT


def compact_release_fields(row):
    found = []

    for column in RELEASE_FIELDS:
        if column not in row.index:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        text = str(value).strip()

        if not text or text.lower() == "nan":
            continue

        found.append(
            f"{column}={text}"
        )

    return " | ".join(found)


def source_matches(candidate_source_text, raw_source):
    candidate_text = normalize_path_text(
        candidate_source_text
    )

    raw_text = normalize_path_text(
        raw_source
    )

    if not raw_text or raw_text == "nan":
        return False

    raw_basename = Path(raw_text).name

    return (
        raw_text in candidate_text
        or raw_basename in candidate_text
    )


def main():
    candidates = pd.read_csv(
        CANDIDATE_INPUT,
        low_memory=False,
    )

    raw = pd.read_csv(
        RAW_INPUT,
        low_memory=False,
    )

    candidates["original_event_time"] = (
        pd.to_datetime(
            candidates["original_event_time"],
            errors="coerce",
            utc=True,
        )
    )

    candidates["actual_s2_time"] = (
        pd.to_datetime(
            candidates["actual_s2_time"],
            errors="coerce",
            utc=True,
        )
    )

    source_column = None

    for column in [
        "source_files_str",
        "source_files",
    ]:
        if column in candidates.columns:
            source_column = column
            break

    if source_column is None:
        raise KeyError(
            "Candidate table has no source-file column."
        )

    if "source_file" not in raw.columns:
        raise KeyError(
            "Raw table has no source_file column."
        )

    available_time_columns = [
        column
        for column in TIME_COLUMNS
        if column in raw.columns
    ]

    match_rows = []

    for number, candidate in candidates.iterrows():
        event_id = str(candidate["event_id"])

        source_text = candidate.get(
            source_column,
            "",
        )

        reference_time = candidate[
            "original_event_time"
        ]

        source_mask = raw[
            "source_file"
        ].apply(
            lambda value:
                source_matches(
                    source_text,
                    value,
                )
        )

        source_rows = raw[
            source_mask
        ].copy()

        if source_rows.empty:
            match_rows.append({
                "event_id":
                    event_id,
                "candidate_rank":
                    candidate.get(
                        "candidate_rank"
                    ),
                "actual_s2_time":
                    candidate[
                        "actual_s2_time"
                    ],
                "original_event_time":
                    reference_time,
                "emission_kg_h_mean":
                    candidate.get(
                        "emission_kg_h_mean"
                    ),
                "emission_bin":
                    candidate.get(
                        "emission_bin"
                    ),
                "raw_match_status":
                    "no_source_file_match",
            })

            continue

        for raw_index, raw_row in (
            source_rows.iterrows()
        ):
            best_time = pd.NaT
            best_time_column = ""
            best_time_value = ""
            best_delta_minutes = np.nan

            for time_column in (
                available_time_columns
            ):
                parsed_time = parse_timestamp(
                    raw_row[time_column],
                    reference_time,
                )

                if (
                    pd.isna(parsed_time)
                    or pd.isna(reference_time)
                ):
                    continue

                delta_minutes = abs(
                    (
                        parsed_time
                        - reference_time
                    ).total_seconds()
                ) / 60.0

                if (
                    not np.isfinite(
                        best_delta_minutes
                    )
                    or delta_minutes
                    < best_delta_minutes
                ):
                    best_time = parsed_time
                    best_time_column = (
                        time_column
                    )
                    best_time_value = str(
                        raw_row[time_column]
                    )
                    best_delta_minutes = (
                        delta_minutes
                    )

            match_rows.append({
                "event_id":
                    event_id,
                "candidate_rank":
                    candidate.get(
                        "candidate_rank"
                    ),
                "filename":
                    candidate.get("filename"),
                "site_name":
                    candidate.get("site_name"),
                "satellite_from_paper":
                    candidate.get(
                        "satellite_from_paper"
                    ),
                "actual_s2_time":
                    candidate[
                        "actual_s2_time"
                    ],
                "original_event_time":
                    reference_time,
                "s2_event_difference_minutes":
                    candidate.get(
                        "s2_event_time_difference_minutes"
                    ),
                "emission_kg_h_mean":
                    candidate.get(
                        "emission_kg_h_mean"
                    ),
                "emission_bin":
                    candidate.get(
                        "emission_bin"
                    ),
                "raw_row_index":
                    raw_index,
                "raw_source_file":
                    raw_row.get(
                        "source_file"
                    ),
                "raw_source_sheet":
                    raw_row.get(
                        "source_sheet"
                    ),
                "best_raw_time_column":
                    best_time_column,
                "best_raw_time_value":
                    best_time_value,
                "best_raw_timestamp":
                    best_time,
                "raw_event_time_difference_minutes":
                    best_delta_minutes,
                "release_fields_found":
                    compact_release_fields(
                        raw_row
                    ),
                "raw_match_status":
                    (
                        "candidate_raw_row"
                        if np.isfinite(
                            best_delta_minutes
                        )
                        else "source_match_no_time"
                    ),
            })

        print(
            f"[{number + 1}/{len(candidates)}] "
            f"{event_id}: "
            f"{len(source_rows)} raw source rows",
            flush=True,
        )

    matches = pd.DataFrame(
        match_rows
    )

    matches = matches.sort_values(
        [
            "candidate_rank",
            "raw_event_time_difference_minutes",
            "raw_row_index",
        ],
        na_position="last",
    ).reset_index(drop=True)

    matches.to_csv(
        OUTPUT,
        index=False,
    )

    matched = matches[
        matches["raw_match_status"].eq(
            "candidate_raw_row"
        )
    ].copy()

    best = (
        matched.sort_values(
            [
                "candidate_rank",
                "raw_event_time_difference_minutes",
            ]
        )
        .groupby(
            "event_id",
            as_index=False,
        )
        .head(1)
        .reset_index(drop=True)
    )

    best.to_csv(
        BEST_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 115)
    print("BEST RAW RELEASE-ROW MATCH PER S2 CANDIDATE")
    print("=" * 115)

    print(
        "\nCandidate events:",
        candidates["event_id"].nunique(),
    )

    print(
        "Events with raw source match:",
        matches.loc[
            matches[
                "raw_match_status"
            ].ne("no_source_file_match"),
            "event_id",
        ].nunique(),
    )

    print(
        "Events with timestamped raw match:",
        best["event_id"].nunique(),
    )

    display_columns = [
        "candidate_rank",
        "event_id",
        "emission_kg_h_mean",
        "emission_bin",
        "actual_s2_time",
        "best_raw_time_column",
        "best_raw_timestamp",
        "raw_event_time_difference_minutes",
        "raw_source_file",
        "release_fields_found",
    ]

    print("\nBest matches:")
    print(
        best[
            display_columns
        ].to_string(
            index=False,
            max_colwidth=100,
        )
    )

    print("\nSaved:")
    print(OUTPUT)
    print(BEST_OUTPUT)


if __name__ == "__main__":
    main()
