from pathlib import Path
import re

import numpy as np
import pandas as pd


ALL_SUMMARIES = Path(
    "outputs/125_stanford_all_release_summaries.csv"
)

EXISTING_WINDOWS = Path(
    "outputs/138_evanston_detailed_release_windows.csv"
)

ALL_OUTPUT = Path(
    "outputs/158_additional_evanston_landsat_events.csv"
)

PRIORITY_OUTPUT = Path(
    "outputs/159_additional_evanston_high_flow_candidates.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/160_additional_evanston_candidate_summary.csv"
)


# 800 kg/h 只是搜尋用的寬鬆預篩選。
# 最終 high-emission 標籤仍固定為衛星過境時 >= 1000 kg/h。
SEARCH_PREFILTER_KG_H = 800.0
FROZEN_HIGH_EMISSION_THRESHOLD_KG_H = 1000.0


def first_existing_column(
    dataframe,
    candidates,
):
    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def derive_sensor_text(dataframe):
    likely_columns = [
        column
        for column in dataframe.columns
        if any(
            token in column.lower()
            for token in [
                "release",
                "instrument",
                "sensor",
                "platform",
                "source",
                "file",
                "path",
                "folder",
            ]
        )
    ]

    if not likely_columns:
        return pd.Series(
            "",
            index=dataframe.index,
        )

    return (
        dataframe[likely_columns]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
    )


def main():
    if not ALL_SUMMARIES.exists():
        raise FileNotFoundError(
            ALL_SUMMARIES
        )

    summaries = pd.read_csv(
        ALL_SUMMARIES,
        low_memory=False,
    )

    release_id_column = (
        first_existing_column(
            summaries,
            [
                "release_ID",
                "release_id",
                "event_id",
            ],
        )
    )

    if release_id_column is None:
        raise KeyError(
            "Could not find release ID column.\n"
            f"Columns: {summaries.columns.tolist()}"
        )

    location_column = (
        first_existing_column(
            summaries,
            [
                "location",
                "site_key",
                "site",
            ],
        )
    )

    if location_column is None:
        raise KeyError(
            "Could not find location/site column."
        )

    mean_flow_column = (
        first_existing_column(
            summaries,
            [
                "ch4_kgh_mean",
                "release_rate_kg_h",
                "mean_flow_kg_h",
                "flow_mean_kg_h",
            ],
        )
    )

    if mean_flow_column is None:
        raise KeyError(
            "Could not find summary flow column."
        )

    location_text = (
        summaries[location_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    evanston = summaries[
        location_text.str.contains(
            "evanston",
            na=False,
        )
    ].copy()

    sensor_text = derive_sensor_text(
        evanston
    )

    ls8_mask = sensor_text.str.contains(
        r"(^|[^A-Z0-9])LS8([^A-Z0-9]|$)",
        flags=re.IGNORECASE,
        regex=True,
        na=False,
    )

    ls9_mask = sensor_text.str.contains(
        r"(^|[^A-Z0-9])LS9([^A-Z0-9]|$)",
        flags=re.IGNORECASE,
        regex=True,
        na=False,
    )

    evanston["expected_sensor"] = np.select(
        [
            ls8_mask,
            ls9_mask,
        ],
        [
            "Landsat-8",
            "Landsat-9",
        ],
        default="",
    )

    evanston = evanston[
        evanston["expected_sensor"]
        .ne("")
    ].copy()

    evanston["release_id"] = (
        evanston[release_id_column]
        .astype(str)
        .str.strip()
    )

    evanston["summary_flow_kg_h"] = (
        pd.to_numeric(
            evanston[mean_flow_column],
            errors="coerce",
        )
    )

    if "datetime_utc" in evanston.columns:
        evanston["release_datetime_utc"] = (
            pd.to_datetime(
                evanston["datetime_utc"],
                errors="coerce",
                utc=True,
            )
        )

    elif (
        "date" in evanston.columns
        and "time_UTC" in evanston.columns
    ):
        evanston["release_datetime_utc"] = (
            pd.to_datetime(
                evanston["date"].astype(str)
                + " "
                + evanston[
                    "time_UTC"
                ].astype(str),
                errors="coerce",
                utc=True,
            )
        )

    else:
        evanston["release_datetime_utc"] = (
            pd.NaT
        )

    existing_ids = set()

    if EXISTING_WINDOWS.exists():
        existing = pd.read_csv(
            EXISTING_WINDOWS,
            low_memory=False,
        )

        if "release_id" in existing.columns:
            existing_ids = set(
                existing["release_id"]
                .dropna()
                .astype(str)
                .str.strip()
            )

    evanston["already_processed"] = (
        evanston["release_id"]
        .isin(existing_ids)
    )

    additional = evanston[
        ~evanston["already_processed"]
    ].copy()

    additional["search_priority"] = np.select(
        [
            additional[
                "summary_flow_kg_h"
            ] >= (
                FROZEN_HIGH_EMISSION_THRESHOLD_KG_H
            ),
            additional[
                "summary_flow_kg_h"
            ] >= SEARCH_PREFILTER_KG_H,
        ],
        [
            "summary_ge_1000",
            "summary_800_to_999",
        ],
        default="summary_below_800",
    )

    additional = (
        additional.sort_values(
            [
                "search_priority",
                "summary_flow_kg_h",
                "release_datetime_utc",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .drop_duplicates(
            subset=["release_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    priority = additional[
        additional[
            "summary_flow_kg_h"
        ] >= SEARCH_PREFILTER_KG_H
    ].copy()

    summary = (
        additional.groupby(
            [
                "expected_sensor",
                "search_priority",
            ],
            dropna=False,
        )
        .agg(
            event_count=(
                "release_id",
                "size",
            ),
            minimum_summary_flow_kg_h=(
                "summary_flow_kg_h",
                "min",
            ),
            median_summary_flow_kg_h=(
                "summary_flow_kg_h",
                "median",
            ),
            maximum_summary_flow_kg_h=(
                "summary_flow_kg_h",
                "max",
            ),
            first_event_time=(
                "release_datetime_utc",
                "min",
            ),
            last_event_time=(
                "release_datetime_utc",
                "max",
            ),
        )
        .reset_index()
    )

    additional.to_csv(
        ALL_OUTPUT,
        index=False,
    )

    priority.to_csv(
        PRIORITY_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("=" * 105)
    print("ADDITIONAL EVANSTON LANDSAT RELEASE EVENTS")
    print("=" * 105)

    print(
        "\nExisting processed release IDs:",
        len(existing_ids),
    )

    print(
        "All Evanston LS8/LS9 events:",
        len(evanston),
    )

    print(
        "Additional unprocessed events:",
        len(additional),
    )

    print(
        "Priority events with summary flow "
        f">= {SEARCH_PREFILTER_KG_H:.0f} kg/h:",
        len(priority),
    )

    print("\nAdditional events by priority:")
    print(
        additional[
            "search_priority"
        ].value_counts(
            dropna=False
        )
    )

    if not priority.empty:
        display_columns = [
            "release_id",
            "release_datetime_utc",
            "expected_sensor",
            "summary_flow_kg_h",
            "search_priority",
        ]

        print("\nPriority candidates:")
        print(
            priority[
                display_columns
            ].sort_values(
                "summary_flow_kg_h",
                ascending=False,
            ).to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.2f}",
            )
        )

    print("\nSaved:")
    print(ALL_OUTPUT)
    print(PRIORITY_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
