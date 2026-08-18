from pathlib import Path
import re

import numpy as np
import pandas as pd


RAW_INPUT = Path(
    "outputs/02_candidate_event_rows.csv"
)

SITE_INPUT = Path(
    "outputs/10_final_events_for_gee.csv"
)

ALL_OUTPUT = Path(
    "outputs/309_all_exact_release_intervals_for_s2.csv"
)

LOW_OUTPUT = Path(
    "outputs/310_low_emission_release_intervals_for_s2.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/311_release_interval_extraction_audit.csv"
)


MAX_DURATION_HOURS = 6.0
LOW_EMISSION_MAX_KG_H = 1000.0


DATE_CONTEXT_COLUMNS = [
    "Date",
    "date",
    "DateOfSurvey",
    "Acquistion date",
    "Acquisition date",
    "DateTime (UTC)",
    "datetime_utc",
    "datetime_UTC",
    "TimestampUTC",
    "Operator_Timestamp",
    "Stanford_timestamp",
    "overpass_datetime",
]


RATE_COLUMNS_2022 = [
    "ch4_kgh_mean",
    "gas_kgh_mean",
    "release_rate_kgh",
    "release_rate",
    "flow_rate",
    "kgh_ch4",
    "FacilityEmissionRate (kg/hr)",
    "FacilityEmissionRate (kg/hour)",
    "UPDATED_ FacilityEmissionRate",
    "FacilityEmissionRate",
    "CH4 Emission (kg h-1)",
]


RATE_COLUMNS_2021 = [
    "cr_kgh_CH4_mean300",
    "cr_kgh_CH4_mean60",
    "cr_kgh_CH4_mean30",
    "cr_kgh_CH4_mean90",
    "cr_kgh_CH4_mean600",
    "cr_kgh_CH4_mean900",
    "ch4_kgh_mean",
    "release_rate_kgh",
    "release_rate",
    "FacilityEmissionRate (kg/hr)",
    "FacilityEmissionRate (kg/hour)",
    "FacilityEmissionRate",
]


RATE_COLUMNS_GENERIC = [
    "ch4_kgh_mean",
    "cr_kgh_CH4_mean300",
    "cr_kgh_CH4_mean60",
    "cr_kgh_CH4_mean30",
    "gas_kgh_mean",
    "release_rate_kgh",
    "release_rate",
    "flow_rate",
    "kgh_ch4",
    "CH4 Emission (kg h-1)",
    "FacilityEmissionRate (kg/hr)",
    "FacilityEmissionRate (kg/hour)",
    "UPDATED_ FacilityEmissionRate",
    "FacilityEmissionRate",
]


SCHEMAS = [
    {
        "schema_name":
            "start_release_end_release",
        "start_column":
            "start_release",
        "end_column":
            "end_release",
        "rate_columns":
            RATE_COLUMNS_2022,
        "seconds_since_midnight":
            False,
    },
    {
        "schema_name":
            "cr_start_cr_end",
        "start_column":
            "cr_start",
        "end_column":
            "cr_end",
        "rate_columns":
            RATE_COLUMNS_2021,
        "seconds_since_midnight":
            False,
    },
    {
        "schema_name":
            "StartTime_EndTime",
        "start_column":
            "StartTime",
        "end_column":
            "EndTime",
        "rate_columns":
            RATE_COLUMNS_GENERIC,
        "seconds_since_midnight":
            False,
    },
    {
        "schema_name":
            "StartUTC_EndUTC",
        "start_column":
            "StartUTC",
        "end_column":
            "EndUTC",
        "rate_columns":
            RATE_COLUMNS_GENERIC,
        "seconds_since_midnight":
            False,
    },
    {
        "schema_name":
            "StartTimeUTC_EndTimeUTC",
        "start_column":
            "StartTime (UTC)",
        "end_column":
            "EndTime (UTC)",
        "rate_columns":
            RATE_COLUMNS_GENERIC,
        "seconds_since_midnight":
            False,
    },
    {
        "schema_name":
            "Start_UTC_End_UTC",
        "start_column":
            "Start UTC",
        "end_column":
            "End UTC",
        "rate_columns":
            RATE_COLUMNS_GENERIC,
        "seconds_since_midnight":
            False,
    },
    {
        "schema_name":
            "overpass_datetime_window",
        "start_column":
            "overpass_datetime",
        "end_column":
            "overpass_datetime_end",
        "rate_columns":
            RATE_COLUMNS_GENERIC,
        "seconds_since_midnight":
            False,
    },
    {
        "schema_name":
            "seconds_since_midnight",
        "start_column":
            "Start UTC (s since midnight)",
        "end_column":
            "End UTC (s since midnight)",
        "rate_columns":
            RATE_COLUMNS_GENERIC,
        "seconds_since_midnight":
            True,
    },
]


def empty_datetime_series(index):
    return pd.Series(
        pd.NaT,
        index=index,
        dtype="datetime64[ns, UTC]",
    )


def build_row_date(frame):
    result = empty_datetime_series(
        frame.index
    )

    for column in DATE_CONTEXT_COLUMNS:
        if column not in frame.columns:
            continue

        text = (
            frame[column]
            .astype("string")
            .str.strip()
        )

        contains_year = text.str.contains(
            r"(?:19|20)\d{2}",
            regex=True,
            na=False,
        )

        parsed = pd.to_datetime(
            text.where(contains_year),
            errors="coerce",
            utc=True,
        )

        parsed = parsed.dt.normalize()

        result = result.fillna(parsed)

    return result


def parse_datetime_values(
    values,
    row_dates,
    seconds_since_midnight=False,
):
    result = empty_datetime_series(
        values.index
    )

    if seconds_since_midnight:
        numeric = pd.to_numeric(
            values,
            errors="coerce",
        )

        valid = (
            row_dates.notna()
            & numeric.ge(0)
            & numeric.lt(24 * 60 * 60)
        )

        result.loc[valid] = (
            row_dates.loc[valid]
            + pd.to_timedelta(
                numeric.loc[valid],
                unit="s",
            )
        )

        return result

    text = (
        values
        .astype("string")
        .str.strip()
    )

    contains_year = text.str.contains(
        r"(?:19|20)\d{2}",
        regex=True,
        na=False,
    )

    result.loc[contains_year] = (
        pd.to_datetime(
            text.loc[contains_year],
            errors="coerce",
            utc=True,
        )
    )

    time_only = text.str.match(
        r"^\s*"
        r"(?:[01]?\d|2[0-3]):[0-5]\d"
        r"(?::[0-5]\d(?:\.\d+)?)?"
        r"\s*$",
        na=False,
    )

    use_row_date = (
        time_only
        & ~contains_year
        & row_dates.notna()
    )

    combined = (
        row_dates.loc[use_row_date]
        .dt.strftime("%Y-%m-%d")
        + " "
        + text.loc[use_row_date]
    )

    result.loc[use_row_date] = (
        pd.to_datetime(
            combined,
            errors="coerce",
            utc=True,
        )
    )

    return result


def select_rate(
    frame,
    rate_columns,
):
    rate = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    rate_source = pd.Series(
        "",
        index=frame.index,
        dtype="string",
    )

    for column in rate_columns:
        if column not in frame.columns:
            continue

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        fill_mask = (
            rate.isna()
            & values.notna()
        )

        rate.loc[fill_mask] = (
            values.loc[fill_mask]
        )

        rate_source.loc[fill_mask] = (
            column
        )

    return rate, rate_source


def infer_site(source_file):
    text = (
        source_file
        .fillna("")
        .astype(str)
        .str.lower()
    )

    return pd.Series(
        np.select(
            [
                text.str.contains(
                    "2024_su_controlled_releases",
                    regex=False,
                ),
                text.str.contains(
                    "2023_controlled_release_2021",
                    regex=False,
                ),
                text.str.contains(
                    "evanston",
                    regex=False,
                ),
            ],
            [
                "Casa_Grande_AZ_release_stacks",
                "Ehrenberg_AZ_release_stack",
                "Evanston_WY_release_site",
            ],
            default="unknown",
        ),
        index=source_file.index,
    )


def get_coordinate_lookup():
    if not SITE_INPUT.exists():
        return {}

    sites = pd.read_csv(
        SITE_INPUT,
        low_memory=False,
    )

    required = {
        "site_name",
        "lat",
        "lon",
    }

    if not required.issubset(
        sites.columns
    ):
        return {}

    sites["lat"] = pd.to_numeric(
        sites["lat"],
        errors="coerce",
    )

    sites["lon"] = pd.to_numeric(
        sites["lon"],
        errors="coerce",
    )

    grouped = (
        sites.dropna(
            subset=["lat", "lon"]
        )
        .groupby("site_name")[
            ["lat", "lon"]
        ]
        .median()
    )

    lookup = {}

    for site_name, row in grouped.iterrows():
        lower = str(site_name).lower()

        if "casa_grande" in lower:
            lookup[
                "Casa_Grande_AZ_release_stacks"
            ] = (
                float(row["lat"]),
                float(row["lon"]),
            )

        if "ehrenberg" in lower:
            lookup[
                "Ehrenberg_AZ_release_stack"
            ] = (
                float(row["lat"]),
                float(row["lon"]),
            )

        if "evanston" in lower:
            lookup[
                "Evanston_WY_release_site"
            ] = (
                float(row["lat"]),
                float(row["lon"]),
            )

    return lookup


def rate_priority(rate_source):
    source = (
        rate_source
        .fillna("")
        .astype(str)
    )

    priority = pd.Series(
        5,
        index=source.index,
        dtype=int,
    )

    priority.loc[
        source.isin([
            "ch4_kgh_mean",
            "cr_kgh_CH4_mean300",
        ])
    ] = 0

    priority.loc[
        source.isin([
            "cr_kgh_CH4_mean30",
            "cr_kgh_CH4_mean60",
            "cr_kgh_CH4_mean90",
            "cr_kgh_CH4_mean600",
            "cr_kgh_CH4_mean900",
        ])
    ] = 1

    priority.loc[
        source.eq("gas_kgh_mean")
    ] = 2

    priority.loc[
        source.isin([
            "release_rate_kgh",
            "release_rate",
            "flow_rate",
            "kgh_ch4",
            "CH4 Emission (kg h-1)",
        ])
    ] = 3

    priority.loc[
        source.str.contains(
            "FacilityEmissionRate",
            regex=False,
        )
    ] = 4

    return priority


def emission_bin(rate):
    return pd.cut(
        rate,
        bins=[
            0,
            200,
            500,
            1000,
            2000,
            np.inf,
        ],
        labels=[
            "0_to_200",
            "200_to_500",
            "500_to_1000",
            "1000_to_2000",
            "2000_plus",
        ],
        include_lowest=False,
        right=False,
    )


def main():
    if not RAW_INPUT.exists():
        raise FileNotFoundError(
            RAW_INPUT
        )

    header = pd.read_csv(
        RAW_INPUT,
        nrows=0,
    )

    wanted_columns = {
        "source_file",
        "source_sheet",
        "paper_guess",
        *DATE_CONTEXT_COLUMNS,
    }

    for schema in SCHEMAS:
        wanted_columns.add(
            schema["start_column"]
        )

        wanted_columns.add(
            schema["end_column"]
        )

        wanted_columns.update(
            schema["rate_columns"]
        )

    use_columns = [
        column
        for column in header.columns
        if column in wanted_columns
    ]

    print("=" * 110)
    print("LOADING RAW CONTROLLED-RELEASE DATA")
    print("=" * 110)

    print(
        "\nColumns loaded:",
        len(use_columns),
    )

    raw = pd.read_csv(
        RAW_INPUT,
        usecols=use_columns,
        low_memory=False,
    )

    print("Raw rows:", len(raw))

    row_dates = build_row_date(
        raw
    )

    candidate_frames = []
    audit_rows = []

    for schema in SCHEMAS:
        start_column = (
            schema["start_column"]
        )

        end_column = (
            schema["end_column"]
        )

        if (
            start_column not in raw.columns
            or end_column not in raw.columns
        ):
            continue

        possible_mask = (
            raw[start_column].notna()
            & raw[end_column].notna()
        )

        subset = raw.loc[
            possible_mask
        ].copy()

        subset_dates = row_dates.loc[
            possible_mask
        ]

        start_time = (
            parse_datetime_values(
                subset[start_column],
                subset_dates,
                seconds_since_midnight=
                    schema[
                        "seconds_since_midnight"
                    ],
            )
        )

        end_time = (
            parse_datetime_values(
                subset[end_column],
                subset_dates,
                seconds_since_midnight=
                    schema[
                        "seconds_since_midnight"
                    ],
            )
        )

        release_rate, rate_source = (
            select_rate(
                subset,
                schema["rate_columns"],
            )
        )

        duration_seconds = (
            (
                end_time
                - start_time
            )
            .dt.total_seconds()
        )

        valid = (
            start_time.notna()
            & end_time.notna()
            & duration_seconds.gt(0)
            & duration_seconds.le(
                MAX_DURATION_HOURS
                * 3600
            )
            & release_rate.notna()
            & release_rate.ge(0)
        )

        accepted = subset.loc[
            valid
        ].copy()

        if accepted.empty:
            audit_rows.append({
                "schema":
                    schema["schema_name"],
                "possible_rows":
                    int(possible_mask.sum()),
                "accepted_rows":
                    0,
            })

            continue

        accepted["raw_row_index"] = (
            accepted.index
        )

        accepted[
            "release_start_utc"
        ] = start_time.loc[valid]

        accepted[
            "release_end_utc"
        ] = end_time.loc[valid]

        accepted[
            "release_duration_minutes"
        ] = (
            duration_seconds.loc[valid]
            / 60.0
        )

        accepted[
            "release_rate_kg_h"
        ] = release_rate.loc[valid]

        accepted[
            "release_rate_source"
        ] = rate_source.loc[valid]

        accepted[
            "interval_schema"
        ] = schema["schema_name"]

        candidate_frames.append(
            accepted[
                [
                    column
                    for column in [
                        "raw_row_index",
                        "source_file",
                        "source_sheet",
                        "paper_guess",
                        "interval_schema",
                        "release_start_utc",
                        "release_end_utc",
                        "release_duration_minutes",
                        "release_rate_kg_h",
                        "release_rate_source",
                    ]
                    if column
                    in accepted.columns
                ]
            ]
        )

        audit_rows.append({
            "schema":
                schema["schema_name"],
            "possible_rows":
                int(possible_mask.sum()),
            "accepted_rows":
                int(valid.sum()),
        })

        print(
            f"{schema['schema_name']}: "
            f"{int(valid.sum())} valid intervals"
        )

    if not candidate_frames:
        raise RuntimeError(
            "No valid release intervals found."
        )

    intervals = pd.concat(
        candidate_frames,
        ignore_index=True,
        sort=False,
    )

    intervals["site"] = infer_site(
        intervals["source_file"]
    )

    coordinate_lookup = (
        get_coordinate_lookup()
    )

    intervals["lat"] = intervals[
        "site"
    ].map(
        lambda site:
            coordinate_lookup.get(
                site,
                (np.nan, np.nan),
            )[0]
    )

    intervals["lon"] = intervals[
        "site"
    ].map(
        lambda site:
            coordinate_lookup.get(
                site,
                (np.nan, np.nan),
            )[1]
    )

    intervals["rate_priority"] = (
        rate_priority(
            intervals[
                "release_rate_source"
            ]
        )
    )

    intervals["release_start_utc"] = (
        pd.to_datetime(
            intervals[
                "release_start_utc"
            ],
            errors="coerce",
            utc=True,
        )
    )

    intervals["release_end_utc"] = (
        pd.to_datetime(
            intervals[
                "release_end_utc"
            ],
            errors="coerce",
            utc=True,
        )
    )

    # 相同場址與完全相同區間，
    # 優先保留 meter-derived CH4 rate。
    intervals = (
        intervals.sort_values(
            [
                "site",
                "release_start_utc",
                "release_end_utc",
                "rate_priority",
                "raw_row_index",
            ]
        )
        .drop_duplicates(
            subset=[
                "site",
                "release_start_utc",
                "release_end_utc",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    intervals[
        "release_interval_id"
    ] = [
        f"CR_INTERVAL_{number:06d}"
        for number in range(
            1,
            len(intervals) + 1,
        )
    ]

    intervals["emission_bin"] = (
        emission_bin(
            intervals[
                "release_rate_kg_h"
            ]
        )
    )

    intervals[
        "coordinate_available"
    ] = (
        intervals["lat"].notna()
        & intervals["lon"].notna()
    )

    intervals[
        "strict_interval_candidate"
    ] = (
        intervals["site"].ne(
            "unknown"
        )
        & intervals[
            "coordinate_available"
        ]
        & intervals[
            "release_rate_kg_h"
        ].gt(0)
    )

    low = intervals[
        intervals[
            "strict_interval_candidate"
        ]
        & intervals[
            "release_rate_kg_h"
        ].gt(0)
        & intervals[
            "release_rate_kg_h"
        ].lt(
            LOW_EMISSION_MAX_KG_H
        )
    ].copy()

    intervals.to_csv(
        ALL_OUTPUT,
        index=False,
    )

    low.to_csv(
        LOW_OUTPUT,
        index=False,
    )

    pd.DataFrame(
        audit_rows
    ).to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 110)
    print("CONTROLLED-RELEASE INTERVAL INVENTORY")
    print("=" * 110)

    print(
        "\nUnique exact intervals:",
        len(intervals),
    )

    print(
        "Strict interval candidates:",
        int(
            intervals[
                "strict_interval_candidate"
            ].sum()
        ),
    )

    print(
        "Low-emission intervals "
        "(0–1000 kg/h):",
        len(low),
    )

    print("\nLow-emission intervals by site:")
    print(
        low["site"]
        .value_counts(
            dropna=False
        )
    )

    print("\nLow-emission bins:")
    print(
        low["emission_bin"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print("\nRate sources:")
    print(
        low[
            "release_rate_source"
        ].value_counts(
            dropna=False
        )
    )

    print("\nDate coverage:")
    if not low.empty:
        print(
            "First:",
            low[
                "release_start_utc"
            ].min(),
        )

        print(
            "Last:",
            low[
                "release_end_utc"
            ].max(),
        )

    print("\nSaved:")
    print(ALL_OUTPUT)
    print(LOW_OUTPUT)
    print(AUDIT_OUTPUT)


if __name__ == "__main__":
    main()
