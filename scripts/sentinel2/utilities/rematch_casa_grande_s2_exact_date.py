from pathlib import Path
import re

import numpy as np
import pandas as pd


REVIEW_INPUT = Path(
    "outputs/301_s2_strict_review_compact.csv"
)

RAW_INPUT = Path(
    "outputs/02_candidate_event_rows.csv"
)

OUTPUT = Path(
    "outputs/304_casa_grande_s2_exact_date_matches.csv"
)


TARGET_RANKS = [1, 3, 6, 8, 12]

FULL_DATETIME_COLUMNS = [
    "DateTime (UTC)",
    "datetime_utc",
    "datetime_UTC",
    "TimestampUTC",
    "Operator_Timestamp",
    "Stanford_timestamp",
    "overpass_datetime",
    "DateOfSurvey",
    "Timestamp (UTC)",
]

DATE_COLUMNS = [
    "Date",
    "date",
    "DateOfSurvey",
    "Acquistion date",
    "Acquisition date",
]

TIME_COLUMNS = [
    "Timestamp (UTC)",
    "Timestamp",
    "Time (UTC)",
    "Time (UTC) - from team",
    "Time (UTC) - from Stanford",
    "Time (UTC) - from Flightradar",
]

EVIDENCE_KEYWORDS = [
    "start",
    "end",
    "release",
    "emission",
    "facility",
    "flow",
    "ch4",
    "kgh",
    "rate",
]


def nonempty(value):
    if pd.isna(value):
        return False

    text = str(value).strip()

    return (
        text
        and text.lower()
        not in {"nan", "none", "null"}
    )


def parse_full_datetime(value):
    if not nonempty(value):
        return pd.NaT

    text = str(value).strip()

    if not re.search(
        r"\b(?:19|20)\d{2}\b",
        text,
    ):
        return pd.NaT

    return pd.to_datetime(
        text,
        errors="coerce",
        utc=True,
    )


def parse_date_and_time(
    date_value,
    time_value,
):
    if (
        not nonempty(date_value)
        or not nonempty(time_value)
    ):
        return pd.NaT

    date = pd.to_datetime(
        date_value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(date):
        return pd.NaT

    text = str(time_value).strip()

    match = re.search(
        r"\b([01]?\d|2[0-3]):([0-5]\d)"
        r"(?::([0-5]\d(?:\.\d+)?))?\b",
        text,
    )

    if not match:
        return pd.NaT

    hour = int(match.group(1))
    minute = int(match.group(2))

    second = (
        float(match.group(3))
        if match.group(3)
        else 0.0
    )

    return (
        date.normalize()
        + pd.Timedelta(hours=hour)
        + pd.Timedelta(minutes=minute)
        + pd.Timedelta(seconds=second)
    )


def row_timestamps(row):
    results = []

    for column in FULL_DATETIME_COLUMNS:
        if column not in row.index:
            continue

        timestamp = parse_full_datetime(
            row[column]
        )

        if pd.notna(timestamp):
            results.append(
                (column, timestamp)
            )

    for date_column in DATE_COLUMNS:
        if date_column not in row.index:
            continue

        for time_column in TIME_COLUMNS:
            if time_column not in row.index:
                continue

            timestamp = parse_date_and_time(
                row[date_column],
                row[time_column],
            )

            if pd.notna(timestamp):
                results.append(
                    (
                        f"{date_column}+{time_column}",
                        timestamp,
                    )
                )

    return results


def evidence_text(row):
    values = []

    for column, value in row.items():
        lower = column.lower()

        if not any(
            keyword in lower
            for keyword in EVIDENCE_KEYWORDS
        ):
            continue

        if not nonempty(value):
            continue

        values.append(
            f"{column}={value}"
        )

    return " | ".join(values)


def main():
    review = pd.read_csv(
        REVIEW_INPUT,
        low_memory=False,
    )

    raw = pd.read_csv(
        RAW_INPUT,
        low_memory=False,
    )

    targets = review[
        review["candidate_rank"].isin(
            TARGET_RANKS
        )
    ].copy()

    targets["actual_s2_time"] = pd.to_datetime(
        targets["actual_s2_time"],
        errors="coerce",
        utc=True,
    )

    # 只搜尋 2022 controlled-release 原始資料。
    raw_subset = raw[
        raw["source_file"]
        .fillna("")
        .astype(str)
        .str.contains(
            "2024_SU_Controlled_Releases",
            case=False,
            regex=False,
        )
    ].copy()

    print(
        "Raw Casa Grande candidate rows:",
        len(raw_subset),
    )

    result_rows = []

    for _, target in targets.iterrows():
        s2_time = target["actual_s2_time"]

        matches = []

        for raw_index, row in (
            raw_subset.iterrows()
        ):
            for source_column, timestamp in (
                row_timestamps(row)
            ):
                # 必須和 S2 在同一天。
                if (
                    timestamp.date()
                    != s2_time.date()
                ):
                    continue

                delta_minutes = abs(
                    (
                        timestamp
                        - s2_time
                    ).total_seconds()
                ) / 60.0

                if delta_minutes > 10:
                    continue

                matches.append({
                    "candidate_rank":
                        target[
                            "candidate_rank"
                        ],
                    "event_id":
                        target["event_id"],
                    "actual_s2_time":
                        s2_time,
                    "summary_emission_kg_h":
                        target[
                            "summary_emission_kg_h"
                        ],
                    "raw_row_index":
                        raw_index,
                    "raw_timestamp_source":
                        source_column,
                    "raw_timestamp":
                        timestamp,
                    "time_difference_minutes":
                        delta_minutes,
                    "raw_source_file":
                        row.get("source_file"),
                    "raw_source_sheet":
                        row.get("source_sheet"),
                    "evidence":
                        evidence_text(row),
                })

        matches = sorted(
            matches,
            key=lambda item: (
                item[
                    "time_difference_minutes"
                ],
                item["raw_row_index"],
            ),
        )

        result_rows.extend(
            matches[:20]
        )

        print(
            f"Candidate {target['candidate_rank']}: "
            f"{len(matches)} same-date matches"
        )

    result = pd.DataFrame(
        result_rows
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print("\n" + "=" * 110)
    print("CASA GRANDE EXACT-DATE MATCHES")
    print("=" * 110)

    if result.empty:
        print("No matches found.")
    else:
        for candidate_rank, group in (
            result.groupby(
                "candidate_rank"
            )
        ):
            print(
                f"\nCandidate {candidate_rank}"
            )

            print(
                group[
                    [
                        "raw_row_index",
                        "raw_timestamp_source",
                        "raw_timestamp",
                        "time_difference_minutes",
                        "raw_source_sheet",
                        "evidence",
                    ]
                ]
                .head(10)
                .to_string(
                    index=False,
                    max_colwidth=120,
                )
            )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
