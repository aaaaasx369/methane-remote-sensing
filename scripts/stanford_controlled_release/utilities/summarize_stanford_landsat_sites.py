from pathlib import Path
import re

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/125_stanford_all_release_summaries.csv"
)

SITE_SUMMARY_OUTPUT = Path(
    "outputs/161_stanford_landsat_site_summary.csv"
)

FOURTH_SITE_OUTPUT = Path(
    "outputs/162_stanford_fourth_site_candidates.csv"
)

EVENT_OUTPUT = Path(
    "outputs/163_stanford_fourth_site_event_candidates.csv"
)


SEARCH_PREFILTER_KG_H = 800.0
HIGH_EMISSION_THRESHOLD_KG_H = 1000.0

EXISTING_SITES = {
    "casa_grande",
    "ehrenberg",
    "evanston",
}


def first_existing_column(
    dataframe,
    candidates,
):
    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def normalize_site(value):
    text = str(value).strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    return text.strip("_")


def build_sensor_text(dataframe):
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
        likely_columns = list(
            dataframe.columns
        )

    return (
        dataframe[likely_columns]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    dataframe = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    release_id_column = (
        first_existing_column(
            dataframe,
            [
                "release_ID",
                "release_id",
                "event_id",
            ],
        )
    )

    site_column = first_existing_column(
        dataframe,
        [
            "location",
            "site_key",
            "site",
        ],
    )

    flow_column = first_existing_column(
        dataframe,
        [
            "ch4_kgh_mean",
            "release_rate_kg_h",
            "mean_flow_kg_h",
            "flow_mean_kg_h",
        ],
    )

    if release_id_column is None:
        raise KeyError(
            "Release ID column not found."
        )

    if site_column is None:
        raise KeyError(
            "Location/site column not found."
        )

    if flow_column is None:
        raise KeyError(
            "Mean flow column not found."
        )

    sensor_text = build_sensor_text(
        dataframe
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

    landsat = dataframe[
        ls8_mask | ls9_mask
    ].copy()

    landsat["expected_sensor"] = (
        np.select(
            [
                ls8_mask.loc[
                    landsat.index
                ],
                ls9_mask.loc[
                    landsat.index
                ],
            ],
            [
                "Landsat-8",
                "Landsat-9",
            ],
            default="unknown",
        )
    )

    landsat["release_id"] = (
        landsat[release_id_column]
        .astype(str)
        .str.strip()
    )

    landsat["site_name"] = (
        landsat[site_column]
        .astype(str)
        .str.strip()
    )

    landsat["site_key_normalized"] = (
        landsat["site_name"]
        .apply(normalize_site)
    )

    landsat["summary_flow_kg_h"] = (
        pd.to_numeric(
            landsat[flow_column],
            errors="coerce",
        )
    )

    if "datetime_utc" in landsat.columns:
        landsat["event_datetime_utc"] = (
            pd.to_datetime(
                landsat["datetime_utc"],
                errors="coerce",
                utc=True,
            )
        )

    elif (
        "date" in landsat.columns
        and "time_UTC"
        in landsat.columns
    ):
        landsat["event_datetime_utc"] = (
            pd.to_datetime(
                landsat["date"]
                .astype(str)
                + " "
                + landsat[
                    "time_UTC"
                ].astype(str),
                errors="coerce",
                utc=True,
            )
        )

    else:
        landsat["event_datetime_utc"] = (
            pd.NaT
        )

    landsat["summary_ge_800"] = (
        landsat["summary_flow_kg_h"]
        >= SEARCH_PREFILTER_KG_H
    )

    landsat["summary_ge_1000"] = (
        landsat["summary_flow_kg_h"]
        >= HIGH_EMISSION_THRESHOLD_KG_H
    )

    landsat["existing_site"] = (
        landsat[
            "site_key_normalized"
        ].isin(EXISTING_SITES)
    )

    aggregation = {
        "event_count": (
            "release_id",
            "nunique",
        ),
        "landsat8_count": (
            "expected_sensor",
            lambda values:
                int(
                    (
                        values
                        == "Landsat-8"
                    ).sum()
                ),
        ),
        "landsat9_count": (
            "expected_sensor",
            lambda values:
                int(
                    (
                        values
                        == "Landsat-9"
                    ).sum()
                ),
        ),
        "events_ge_800": (
            "summary_ge_800",
            "sum",
        ),
        "events_ge_1000": (
            "summary_ge_1000",
            "sum",
        ),
        "minimum_flow_kg_h": (
            "summary_flow_kg_h",
            "min",
        ),
        "median_flow_kg_h": (
            "summary_flow_kg_h",
            "median",
        ),
        "maximum_flow_kg_h": (
            "summary_flow_kg_h",
            "max",
        ),
        "first_event": (
            "event_datetime_utc",
            "min",
        ),
        "last_event": (
            "event_datetime_utc",
            "max",
        ),
    }

    if "lat" in landsat.columns:
        aggregation["median_latitude"] = (
            "lat",
            "median",
        )

    if "lon" in landsat.columns:
        aggregation["median_longitude"] = (
            "lon",
            "median",
        )

    site_summary = (
        landsat.groupby(
            [
                "site_key_normalized",
                "site_name",
                "existing_site",
            ],
            dropna=False,
        )
        .agg(**aggregation)
        .reset_index()
        .sort_values(
            [
                "existing_site",
                "events_ge_1000",
                "events_ge_800",
                "event_count",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    fourth_sites = site_summary[
        ~site_summary["existing_site"]
        & (
            site_summary["events_ge_800"]
            > 0
        )
    ].copy()

    fourth_site_keys = set(
        fourth_sites[
            "site_key_normalized"
        ]
    )

    candidate_events = landsat[
        landsat[
            "site_key_normalized"
        ].isin(fourth_site_keys)
        & landsat[
            "summary_ge_800"
        ]
    ].copy()

    candidate_events = (
        candidate_events.sort_values(
            [
                "site_key_normalized",
                "summary_flow_kg_h",
                "event_datetime_utc",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    SITE_SUMMARY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    site_summary.to_csv(
        SITE_SUMMARY_OUTPUT,
        index=False,
    )

    fourth_sites.to_csv(
        FOURTH_SITE_OUTPUT,
        index=False,
    )

    candidate_events.to_csv(
        EVENT_OUTPUT,
        index=False,
    )

    print("=" * 110)
    print("STANFORD LANDSAT SITE SUMMARY")
    print("=" * 110)

    print(
        "\nTotal Landsat release rows:",
        len(landsat),
    )

    print(
        "Unique Landsat release events:",
        landsat["release_id"].nunique(),
    )

    print(
        "Unique Landsat sites:",
        landsat[
            "site_key_normalized"
        ].nunique(),
    )

    print(
        "New candidate sites:",
        len(fourth_sites),
    )

    display_columns = [
        "site_name",
        "site_key_normalized",
        "existing_site",
        "event_count",
        "landsat8_count",
        "landsat9_count",
        "events_ge_800",
        "events_ge_1000",
        "median_flow_kg_h",
        "maximum_flow_kg_h",
        "first_event",
        "last_event",
    ]

    print("\nAll Landsat sites:")
    print(
        site_summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.2f}",
        )
    )

    if not fourth_sites.empty:
        print(
            "\nFourth-site candidates:"
        )

        print(
            fourth_sites[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.2f}",
            )
        )

        print(
            "\nHigh-flow candidate events:"
        )

        event_columns = [
            "release_id",
            "site_name",
            "expected_sensor",
            "event_datetime_utc",
            "summary_flow_kg_h",
            "summary_ge_1000",
        ]

        print(
            candidate_events[
                event_columns
            ].to_string(
                index=False,
                float_format=lambda value:
                    f"{value:.2f}",
            )
        )

    print("\nSaved:")
    print(SITE_SUMMARY_OUTPUT)
    print(FOURTH_SITE_OUTPUT)
    print(EVENT_OUTPUT)


if __name__ == "__main__":
    main()
