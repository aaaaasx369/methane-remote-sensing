from pathlib import Path
import pandas as pd


AUDIT_INPUT = Path(
    "outputs/293_s2_actual_acquisition_time_audit.csv"
)

EVENT_INPUT = Path(
    "outputs/10_final_events_for_gee.csv"
)

CANDIDATE_OUTPUT = Path(
    "outputs/294_s2_positive_within_60min_candidates.csv"
)

SOURCE_SUMMARY_OUTPUT = Path(
    "outputs/295_s2_candidate_source_file_summary.csv"
)

UNMATCHED_OUTPUT = Path(
    "outputs/296_s2_unmatched_event_ids.csv"
)


def main():
    audit = pd.read_csv(
        AUDIT_INPUT,
        low_memory=False,
    )

    events = pd.read_csv(
        EVENT_INPUT,
        low_memory=False,
    )

    for frame in [audit, events]:
        frame["event_id"] = (
            frame["event_id"]
            .astype(str)
            .str.strip()
        )

    event_columns = [
        column
        for column in [
            "event_id",
            "paper",
            "site_name",
            "satellite_from_paper",
            "source_files",
            "source_files_str",
            "n_rows_merged",
            "n_unique_times_merged",
        ]
        if column in events.columns
    ]

    events = (
        events[event_columns]
        .drop_duplicates(
            subset=["event_id"],
            keep="first",
        )
    )

    merged = audit.merge(
        events,
        on="event_id",
        how="left",
        suffixes=("", "_event"),
        indicator="event_source_merge",
    )

    merged[
        "s2_event_time_difference_minutes"
    ] = pd.to_numeric(
        merged[
            "s2_event_time_difference_minutes"
        ],
        errors="coerce",
    )

    candidates = merged[
        merged["label"].eq(1)
        & merged[
            "s2_event_time_difference_minutes"
        ].le(60)
    ].copy()

    candidates = candidates.sort_values(
        [
            "emission_kg_h_mean",
            "actual_s2_time",
            "event_id",
        ],
        na_position="last",
    ).reset_index(drop=True)

    candidates[
        "candidate_rank"
    ] = range(
        1,
        len(candidates) + 1,
    )

    candidates[
        "needs_exact_release_window_check"
    ] = True

    candidates[
        "strict_positive_status"
    ] = "pending_release_window_check"

    candidates.to_csv(
        CANDIDATE_OUTPUT,
        index=False,
    )

    source_column = None

    for column in [
        "source_files_str",
        "source_files",
    ]:
        if column in candidates.columns:
            source_column = column
            break

    if source_column is not None:
        source_summary = (
            candidates.groupby(
                [
                    "paper",
                    "site_name",
                    "satellite_from_paper",
                    source_column,
                ],
                dropna=False,
            )
            .agg(
                candidate_count=(
                    "event_id",
                    "size",
                ),
                low_emission_count=(
                    "emission_kg_h_mean",
                    lambda values:
                        int(
                            pd.to_numeric(
                                values,
                                errors="coerce",
                            ).lt(1000).sum()
                        ),
                ),
                minimum_emission_kg_h=(
                    "emission_kg_h_mean",
                    "min",
                ),
                maximum_emission_kg_h=(
                    "emission_kg_h_mean",
                    "max",
                ),
            )
            .reset_index()
        )
    else:
        source_summary = pd.DataFrame()

    source_summary.to_csv(
        SOURCE_SUMMARY_OUTPUT,
        index=False,
    )

    unmatched = merged[
        merged[
            "event_source_merge"
        ].ne("both")
    ].copy()

    unmatched.to_csv(
        UNMATCHED_OUTPUT,
        index=False,
    )

    display_columns = [
        column
        for column in [
            "candidate_rank",
            "event_id",
            "filename",
            "site_name",
            "satellite_from_paper",
            "original_event_time",
            "actual_s2_time",
            "s2_event_time_difference_minutes",
            "emission_kg_h_mean",
            "emission_bin",
            "source_files_str",
        ]
        if column in candidates.columns
    ]

    print("=" * 120)
    print("SENTINEL-2 STRICT POSITIVE CANDIDATE INVENTORY")
    print("=" * 120)

    print(
        "\nPositive candidates within 60 minutes:",
        len(candidates),
    )

    print(
        "Candidates below 1000 kg/h:",
        int(
            candidates[
                "emission_kg_h_mean"
            ].lt(1000).sum()
        ),
    )

    print(
        "Unique events:",
        candidates["event_id"].nunique(),
    )

    print(
        "Unique sites:",
        candidates[
            "site_name"
        ].nunique()
        if "site_name" in candidates.columns
        else "unknown",
    )

    print("\nEmission bins:")
    print(
        candidates[
            "emission_bin"
        ].value_counts(
            dropna=False
        ).sort_index()
    )

    print("\nCandidate events:")
    print(
        candidates[
            display_columns
        ].to_string(
            index=False,
            max_colwidth=90,
        )
    )

    print("\nSource groups:")
    if not source_summary.empty:
        print(
            source_summary.to_string(
                index=False,
                max_colwidth=100,
            )
        )
    else:
        print("No source-file column found.")

    print(
        "\nUnmatched S2 rows:",
        len(unmatched),
    )

    print("\nSaved:")
    print(CANDIDATE_OUTPUT)
    print(SOURCE_SUMMARY_OUTPUT)
    print(UNMATCHED_OUTPUT)


if __name__ == "__main__":
    main()
