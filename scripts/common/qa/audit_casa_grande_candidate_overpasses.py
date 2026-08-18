from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_CSV = Path(
    "outputs/67_landsat_unique_candidate_overpasses.csv"
)

RELEASE_FILE_CANDIDATES = [
    Path(
        "raw_data/2024_SU_Controlled_Releases/"
        "sahar-elabbadi-SU-Controlled-Releases-2022-0a604d7/"
        "Satellite_overpasses_with_release_rates_20230404.csv"
    ),
    Path(
        "raw_data/2024_SU_Controlled_Releases/"
        "sahar-elabbadi-SU-Controlled-Releases-2022-0a604d7/"
        "00_raw_reports/"
        "Satellite_overpasses_with_release_rates_20230404.csv"
    ),
]

AUDIT_OUTPUT_CSV = Path(
    "outputs/69_casa_grande_candidate_release_audit.csv"
)

MATCH_OUTPUT_CSV = Path(
    "outputs/70_casa_grande_candidate_release_matches.csv"
)

DAILY_OUTPUT_CSV = Path(
    "outputs/71_casa_grande_release_log_daily_coverage.csv"
)


def locate_release_file():
    for path in RELEASE_FILE_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find Casa Grande release file:\n"
        + "\n".join(
            str(path)
            for path in RELEASE_FILE_CANDIDATES
        )
    )


def parse_utc(series):
    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )


def calculate_seconds_to_interval(
    candidate_time,
    release_table,
):
    result = pd.Series(
        np.nan,
        index=release_table.index,
        dtype=float,
    )

    before = (
        candidate_time
        < release_table["release_start_utc"]
    )

    after = (
        candidate_time
        > release_table["release_end_utc"]
    )

    overlap = ~(before | after)

    result.loc[overlap] = 0.0

    result.loc[before] = (
        release_table.loc[
            before,
            "release_start_utc",
        ]
        - candidate_time
    ).dt.total_seconds()

    result.loc[after] = (
        candidate_time
        - release_table.loc[
            after,
            "release_end_utc",
        ]
    ).dt.total_seconds()

    return result


def classify_overlap(overlapping):
    rates = pd.to_numeric(
        overlapping["ch4_kgh_mean"],
        errors="coerce",
    ).dropna()

    if len(rates) == 0:
        return {
            "release_review_status":
                "overlap_missing_rate",
            "recommended_label": np.nan,
            "negative_candidate_eligible": False,
            "review_reason": (
                "The Landsat acquisition overlaps a release "
                "interval, but ch4_kgh_mean is missing."
            ),
        }

    has_positive = bool(
        (rates > 0).any()
    )

    has_zero = bool(
        (rates == 0).any()
    )

    if has_positive and has_zero:
        return {
            "release_review_status":
                "conflicting_overlap_rates",
            "recommended_label": np.nan,
            "negative_candidate_eligible": False,
            "review_reason": (
                "Overlapping release rows contain both positive "
                "and zero flow values."
            ),
        }

    if has_positive:
        return {
            "release_review_status":
                "confirmed_positive_overlap",
            "recommended_label": 1,
            "negative_candidate_eligible": False,
            "review_reason": (
                "The Landsat acquisition overlaps a release "
                "interval with positive methane flow."
            ),
        }

    if has_zero and (rates == 0).all():
        return {
            "release_review_status":
                "confirmed_zero_flow_overlap",
            "recommended_label": 0,
            "negative_candidate_eligible": True,
            "review_reason": (
                "The Landsat acquisition overlaps a measured "
                "interval and all available methane-flow values "
                "are zero."
            ),
        }

    return {
        "release_review_status":
            "invalid_overlap_rate",
        "recommended_label": np.nan,
        "negative_candidate_eligible": False,
        "review_reason": (
            "The overlapping release-rate values could not be "
            "interpreted as positive or zero."
        ),
    }


def main():
    if not CANDIDATE_CSV.exists():
        raise FileNotFoundError(
            f"Missing candidate file: {CANDIDATE_CSV}"
        )

    release_path = locate_release_file()

    candidates = pd.read_csv(
        CANDIDATE_CSV,
        low_memory=False,
    )

    releases = pd.read_csv(
        release_path,
        low_memory=False,
    )

    required_candidate_columns = [
        "overpass_id",
        "site_key",
        "candidate_time_utc",
        "landsat_sensor",
        "LANDSAT_PRODUCT_ID",
    ]

    missing_candidate_columns = [
        column
        for column in required_candidate_columns
        if column not in candidates.columns
    ]

    if missing_candidate_columns:
        raise ValueError(
            "Missing candidate columns: "
            + ", ".join(missing_candidate_columns)
        )

    required_release_columns = [
        "start_release",
        "end_release",
        "ch4_kgh_mean",
    ]

    missing_release_columns = [
        column
        for column in required_release_columns
        if column not in releases.columns
    ]

    if missing_release_columns:
        raise ValueError(
            "Missing release columns: "
            + ", ".join(missing_release_columns)
        )

    candidates["candidate_time_utc"] = parse_utc(
        candidates["candidate_time_utc"]
    )

    releases["release_start_utc"] = parse_utc(
        releases["start_release"]
    )

    releases["release_end_utc"] = parse_utc(
        releases["end_release"]
    )

    releases["ch4_kgh_mean"] = pd.to_numeric(
        releases["ch4_kgh_mean"],
        errors="coerce",
    )

    if "ch4_kgh_sigma" in releases.columns:
        releases["ch4_kgh_sigma"] = pd.to_numeric(
            releases["ch4_kgh_sigma"],
            errors="coerce",
        )

    releases = releases[
        releases["release_start_utc"].notna()
        & releases["release_end_utc"].notna()
    ].copy()

    casa_candidates = candidates[
        candidates["site_key"]
        == "casa_grande"
    ].copy()

    casa_candidates = casa_candidates.sort_values(
        "candidate_time_utc"
    ).reset_index(drop=True)

    if len(casa_candidates) == 0:
        raise ValueError(
            "No Casa Grande candidate overpasses were found."
        )

    releases["release_date_utc"] = (
        releases["release_start_utc"]
        .dt.strftime("%Y-%m-%d")
    )

    daily_coverage = (
        releases.groupby(
            "release_date_utc"
        )
        .agg(
            interval_count=(
                "release_start_utc",
                "size",
            ),
            first_interval_start=(
                "release_start_utc",
                "min",
            ),
            last_interval_end=(
                "release_end_utc",
                "max",
            ),
            nonmissing_rate_count=(
                "ch4_kgh_mean",
                lambda series: int(
                    series.notna().sum()
                ),
            ),
            positive_rate_count=(
                "ch4_kgh_mean",
                lambda series: int(
                    (series > 0).sum()
                ),
            ),
            zero_rate_count=(
                "ch4_kgh_mean",
                lambda series: int(
                    (series == 0).sum()
                ),
            ),
            rate_min=(
                "ch4_kgh_mean",
                "min",
            ),
            rate_max=(
                "ch4_kgh_mean",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            "release_date_utc"
        )
    )

    DAILY_OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_coverage.to_csv(
        DAILY_OUTPUT_CSV,
        index=False,
    )

    audit_rows = []
    match_rows = []

    useful_release_columns = [
        column
        for column in [
            "Date",
            "Timestamp (UTC)",
            "DateTime (UTC)",
            "start_release",
            "end_release",
            "ch4_kgh_mean",
            "ch4_kgh_sigma",
            "ch4_fraction_km",
            "ch4_fraction_km_sigma",
        ]
        if column in releases.columns
    ]

    for _, candidate in casa_candidates.iterrows():
        candidate_time = candidate[
            "candidate_time_utc"
        ]

        candidate_date = (
            candidate_time.strftime("%Y-%m-%d")
        )

        overlapping = releases[
            (
                releases["release_start_utc"]
                <= candidate_time
            )
            & (
                releases["release_end_utc"]
                >= candidate_time
            )
        ].copy()

        same_day = releases[
            releases["release_date_utc"]
            == candidate_date
        ].copy()

        release_distances = (
            calculate_seconds_to_interval(
                candidate_time,
                releases,
            )
        )

        nearest_index = (
            release_distances.idxmin()
            if release_distances.notna().any()
            else None
        )

        if nearest_index is not None:
            nearest = releases.loc[
                nearest_index
            ]

            seconds_to_nearest = float(
                release_distances.loc[
                    nearest_index
                ]
            )
        else:
            nearest = pd.Series(
                dtype=object
            )

            seconds_to_nearest = np.nan

        if len(overlapping) > 0:
            decision = classify_overlap(
                overlapping
            )

            overlapping_rates = (
                overlapping["ch4_kgh_mean"]
                .dropna()
            )

            overlap_rate_min = (
                overlapping_rates.min()
                if len(overlapping_rates) > 0
                else np.nan
            )

            overlap_rate_max = (
                overlapping_rates.max()
                if len(overlapping_rates) > 0
                else np.nan
            )

        elif len(same_day) > 0:
            decision = {
                "release_review_status":
                    "same_day_but_no_exact_overlap",
                "recommended_label": np.nan,
                "negative_candidate_eligible": False,
                "review_reason": (
                    "Release-log intervals exist on the same UTC "
                    "date, but none overlaps the Landsat "
                    "acquisition time. This is not sufficient by "
                    "itself to assign Label 0."
                ),
            }

            overlap_rate_min = np.nan
            overlap_rate_max = np.nan

        else:
            decision = {
                "release_review_status":
                    "no_release_log_for_date",
                "recommended_label": np.nan,
                "negative_candidate_eligible": False,
                "review_reason": (
                    "No release-log interval was found on the "
                    "candidate UTC date. Additional campaign "
                    "schedule evidence is required before assigning "
                    "Label 0."
                ),
            }

            overlap_rate_min = np.nan
            overlap_rate_max = np.nan

        audit_row = {
            "overpass_id":
                candidate["overpass_id"],
            "candidate_time_utc":
                candidate_time,
            "candidate_date_utc":
                candidate_date,
            "landsat_sensor":
                candidate["landsat_sensor"],
            "LANDSAT_PRODUCT_ID":
                candidate["LANDSAT_PRODUCT_ID"],
            "WRS_PATH":
                candidate.get("WRS_PATH"),
            "WRS_ROW":
                candidate.get("WRS_ROW"),
            "CLOUD_COVER":
                candidate.get("CLOUD_COVER"),
            "same_reference_wrs":
                candidate.get(
                    "same_reference_wrs"
                ),
            "same_day_release_interval_count":
                len(same_day),
            "overlapping_interval_count":
                len(overlapping),
            "overlap_rate_min":
                overlap_rate_min,
            "overlap_rate_max":
                overlap_rate_max,
            "nearest_release_start_utc":
                nearest.get(
                    "release_start_utc"
                ),
            "nearest_release_end_utc":
                nearest.get(
                    "release_end_utc"
                ),
            "nearest_ch4_kgh_mean":
                nearest.get(
                    "ch4_kgh_mean"
                ),
            "seconds_to_nearest_interval":
                seconds_to_nearest,
            "hours_to_nearest_interval":
                (
                    seconds_to_nearest / 3600
                    if pd.notna(
                        seconds_to_nearest
                    )
                    else np.nan
                ),
            **decision,
        }

        audit_rows.append(
            audit_row
        )

        selected_matches = (
            overlapping.copy()
            if len(overlapping) > 0
            else same_day.copy()
        )

        if len(selected_matches) == 0:
            if nearest_index is not None:
                selected_matches = (
                    releases.loc[
                        [nearest_index]
                    ].copy()
                )

                match_type = (
                    "nearest_interval_only"
                )
            else:
                selected_matches = (
                    pd.DataFrame()
                )

                match_type = (
                    "no_release_match"
                )
        elif len(overlapping) > 0:
            match_type = (
                "exact_time_overlap"
            )
        else:
            match_type = (
                "same_day_interval"
            )

        for match_rank, (_, release_row) in enumerate(
            selected_matches.iterrows(),
            start=1,
        ):
            row = {
                "overpass_id":
                    candidate["overpass_id"],
                "candidate_time_utc":
                    candidate_time,
                "LANDSAT_PRODUCT_ID":
                    candidate["LANDSAT_PRODUCT_ID"],
                "match_type":
                    match_type,
                "match_rank":
                    match_rank,
                "release_start_utc":
                    release_row.get(
                        "release_start_utc"
                    ),
                "release_end_utc":
                    release_row.get(
                        "release_end_utc"
                    ),
            }

            for column in useful_release_columns:
                row[column] = (
                    release_row.get(column)
                )

            match_rows.append(row)

    audit_df = pd.DataFrame(
        audit_rows
    )

    matches_df = pd.DataFrame(
        match_rows
    )

    audit_df = audit_df.sort_values(
        "candidate_time_utc"
    ).reset_index(drop=True)

    audit_df.to_csv(
        AUDIT_OUTPUT_CSV,
        index=False,
    )

    matches_df.to_csv(
        MATCH_OUTPUT_CSV,
        index=False,
    )

    print("=" * 110)
    print("CASA GRANDE CANDIDATE RELEASE AUDIT")
    print("=" * 110)

    print(f"\nRelease file: {release_path}")
    print(f"Valid release intervals: {len(releases)}")
    print(
        f"Casa Grande candidate overpasses: "
        f"{len(audit_df)}"
    )

    print("\nRelease-log date range:")
    print(
        releases["release_start_utc"].min(),
        "to",
        releases["release_end_utc"].max(),
    )

    print("\nReview-status counts:")
    print(
        audit_df[
            "release_review_status"
        ].value_counts()
    )

    print("\nRecommended-label counts:")
    print(
        audit_df[
            "recommended_label"
        ].value_counts(
            dropna=False
        ).sort_index()
    )

    print("\nNegative-candidate eligibility:")
    print(
        audit_df[
            "negative_candidate_eligible"
        ].value_counts()
    )

    print("\nComplete Casa Grande audit:")
    display_columns = [
        "overpass_id",
        "candidate_time_utc",
        "landsat_sensor",
        "CLOUD_COVER",
        "same_day_release_interval_count",
        "overlapping_interval_count",
        "overlap_rate_min",
        "overlap_rate_max",
        "hours_to_nearest_interval",
        "release_review_status",
        "recommended_label",
        "negative_candidate_eligible",
    ]

    print(
        audit_df[
            display_columns
        ].to_string(index=False)
    )

    confirmed_zero = audit_df[
        audit_df[
            "release_review_status"
        ]
        == "confirmed_zero_flow_overlap"
    ]

    print("\nConfirmed zero-flow candidates:")

    if len(confirmed_zero) == 0:
        print("None")
    else:
        print(
            confirmed_zero[
                [
                    "overpass_id",
                    "candidate_time_utc",
                    "landsat_sensor",
                    "LANDSAT_PRODUCT_ID",
                    "CLOUD_COVER",
                    "overlap_rate_min",
                    "overlap_rate_max",
                ]
            ].to_string(index=False)
        )

    print("\nSaved:")
    print(AUDIT_OUTPUT_CSV)
    print(MATCH_OUTPUT_CSV)
    print(DAILY_OUTPUT_CSV)


if __name__ == "__main__":
    main()
