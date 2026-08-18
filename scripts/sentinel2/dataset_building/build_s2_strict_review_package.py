from pathlib import Path
import pandas as pd


BEST_MATCH_INPUT = Path(
    "outputs/298_s2_candidate_best_raw_release_matches.csv"
)

RAW_INPUT = Path(
    "outputs/02_candidate_event_rows.csv"
)

REVIEW_OUTPUT = Path(
    "outputs/299_s2_strict_manual_review_template.csv"
)

EVIDENCE_OUTPUT = Path(
    "outputs/300_s2_strict_raw_evidence_long.csv"
)


KEYWORDS = [
    "start",
    "end",
    "time",
    "date",
    "release",
    "emission",
    "flow",
    "ch4",
    "kgh",
    "facility",
    "rate",
    "timestamp",
]


def is_nonempty(value):
    if pd.isna(value):
        return False

    text = str(value).strip()

    return (
        text != ""
        and text.lower()
        not in {"nan", "none", "null"}
    )


def is_relevant_column(column):
    lower = column.lower()

    return any(
        keyword in lower
        for keyword in KEYWORDS
    )


def main():
    best = pd.read_csv(
        BEST_MATCH_INPUT,
        low_memory=False,
    )

    raw = pd.read_csv(
        RAW_INPUT,
        low_memory=False,
    )

    evidence_rows = []
    review_rows = []

    best = best.sort_values(
        "candidate_rank"
    ).reset_index(drop=True)

    for _, candidate in best.iterrows():
        raw_index = pd.to_numeric(
            candidate["raw_row_index"],
            errors="coerce",
        )

        if pd.isna(raw_index):
            continue

        raw_index = int(raw_index)

        if (
            raw_index < 0
            or raw_index >= len(raw)
        ):
            raise IndexError(
                f"Invalid raw row index: "
                f"{raw_index}"
            )

        raw_row = raw.iloc[raw_index]

        evidence_count = 0

        for column, value in (
            raw_row.items()
        ):
            if (
                is_relevant_column(column)
                and is_nonempty(value)
            ):
                evidence_rows.append({
                    "candidate_rank":
                        candidate.get(
                            "candidate_rank"
                        ),
                    "event_id":
                        candidate.get(
                            "event_id"
                        ),
                    "actual_s2_time":
                        candidate.get(
                            "actual_s2_time"
                        ),
                    "raw_row_index":
                        raw_index,
                    "raw_source_file":
                        candidate.get(
                            "raw_source_file"
                        ),
                    "raw_source_sheet":
                        candidate.get(
                            "raw_source_sheet"
                        ),
                    "field_name":
                        column,
                    "field_value":
                        str(value),
                })

                evidence_count += 1

        event_id = str(
            candidate.get(
                "event_id",
                "",
            )
        )

        direct_sentinel2 = (
            "sentinel-2"
            in event_id.lower()
        )

        review_rows.append({
            "candidate_rank":
                candidate.get(
                    "candidate_rank"
                ),
            "event_id":
                event_id,
            "filename":
                candidate.get(
                    "filename"
                ),
            "site_name":
                candidate.get(
                    "site_name"
                ),
            "event_sensor":
                candidate.get(
                    "satellite_from_paper"
                ),
            "direct_sentinel2_event":
                direct_sentinel2,
            "review_priority":
                (
                    "1_direct_sentinel2"
                    if direct_sentinel2
                    else "2_cross_sensor_candidate"
                ),
            "actual_s2_time":
                candidate.get(
                    "actual_s2_time"
                ),
            "original_event_time":
                candidate.get(
                    "original_event_time"
                ),
            "s2_event_difference_minutes":
                candidate.get(
                    "s2_event_difference_minutes"
                ),
            "summary_emission_kg_h":
                candidate.get(
                    "emission_kg_h_mean"
                ),
            "summary_emission_bin":
                candidate.get(
                    "emission_bin"
                ),
            "raw_row_index":
                raw_index,
            "raw_source_file":
                candidate.get(
                    "raw_source_file"
                ),
            "raw_source_sheet":
                candidate.get(
                    "raw_source_sheet"
                ),
            "evidence_field_count":
                evidence_count,

            # 以下欄位由人工確認。
            "exact_release_start_utc":
                "",
            "exact_release_end_utc":
                "",
            "exact_release_rate_kg_h":
                "",
            "s2_inside_exact_release_window":
                "",
            "review_status":
                "pending",
            "evidence_fields_used":
                "",
            "review_notes":
                "",
        })

    review = pd.DataFrame(
        review_rows
    ).sort_values(
        [
            "review_priority",
            "candidate_rank",
        ]
    )

    evidence = pd.DataFrame(
        evidence_rows
    ).sort_values(
        [
            "candidate_rank",
            "field_name",
        ]
    )

    review.to_csv(
        REVIEW_OUTPUT,
        index=False,
    )

    evidence.to_csv(
        EVIDENCE_OUTPUT,
        index=False,
    )

    print("=" * 100)
    print("SENTINEL-2 STRICT REVIEW PACKAGE")
    print("=" * 100)

    print("\nCandidates:", len(review))

    print("\nBy review priority:")
    print(
        review[
            "review_priority"
        ].value_counts()
    )

    print(
        "\nEvidence rows:",
        len(evidence),
    )

    print("\nReview order:")
    print(
        review[
            [
                "candidate_rank",
                "event_id",
                "direct_sentinel2_event",
                "actual_s2_time",
                "summary_emission_kg_h",
                "review_priority",
            ]
        ].to_string(
            index=False,
            max_colwidth=90,
        )
    )

    print("\nSaved:")
    print(REVIEW_OUTPUT)
    print(EVIDENCE_OUTPUT)


if __name__ == "__main__":
    main()
