from pathlib import Path
import pandas as pd


REVIEW_INPUT = Path(
    "outputs/299_s2_strict_manual_review_template.csv"
)

EVIDENCE_INPUT = Path(
    "outputs/300_s2_strict_raw_evidence_long.csv"
)

OUTPUT = Path(
    "outputs/301_s2_strict_review_compact.csv"
)


START_KEYWORDS = [
    "start",
    "cr_start",
]

END_KEYWORDS = [
    "end",
    "cr_end",
]

RATE_KEYWORDS = [
    "emission",
    "flow",
    "rate",
    "kgh",
    "kg/h",
    "kg h",
    "facility",
    "ch4",
]

TIME_KEYWORDS = [
    "timestamp",
    "datetime",
    "date",
    "time",
]


def join_evidence(group, keywords):
    rows = []

    for _, row in group.iterrows():
        field_name = str(
            row["field_name"]
        )

        field_value = str(
            row["field_value"]
        )

        lower_name = field_name.lower()

        if any(
            keyword in lower_name
            for keyword in keywords
        ):
            rows.append(
                f"{field_name}={field_value}"
            )

    # 保留順序並移除完全相同的內容
    return " | ".join(
        dict.fromkeys(rows)
    )


def main():
    review = pd.read_csv(
        REVIEW_INPUT,
        low_memory=False,
    )

    evidence = pd.read_csv(
        EVIDENCE_INPUT,
        low_memory=False,
    )

    compact_rows = []

    for candidate_rank, group in (
        evidence.groupby(
            "candidate_rank",
            sort=True,
        )
    ):
        compact_rows.append({
            "candidate_rank":
                candidate_rank,

            "start_evidence":
                join_evidence(
                    group,
                    START_KEYWORDS,
                ),

            "end_evidence":
                join_evidence(
                    group,
                    END_KEYWORDS,
                ),

            "rate_evidence":
                join_evidence(
                    group,
                    RATE_KEYWORDS,
                ),

            "time_evidence":
                join_evidence(
                    group,
                    TIME_KEYWORDS,
                ),
        })

    compact = pd.DataFrame(
        compact_rows
    )

    result = review.merge(
        compact,
        on="candidate_rank",
        how="left",
        validate="one_to_one",
    )

    result["automatic_warning"] = ""

    no_start = (
        result["start_evidence"]
        .fillna("")
        .str.strip()
        .eq("")
    )

    no_end = (
        result["end_evidence"]
        .fillna("")
        .str.strip()
        .eq("")
    )

    no_rate = (
        result["rate_evidence"]
        .fillna("")
        .str.strip()
        .eq("")
    )

    result.loc[
        no_start,
        "automatic_warning",
    ] += "no_start_evidence;"

    result.loc[
        no_end,
        "automatic_warning",
    ] += "no_end_evidence;"

    result.loc[
        no_rate,
        "automatic_warning",
    ] += "no_rate_evidence;"

    cross_sensor = ~result[
        "direct_sentinel2_event"
    ].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    result.loc[
        cross_sensor,
        "automatic_warning",
    ] += "cross_sensor_candidate;"

    result = result.sort_values(
        [
            "review_priority",
            "candidate_rank",
        ]
    ).reset_index(drop=True)

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 100)
    print("SENTINEL-2 COMPACT REVIEW TABLE")
    print("=" * 100)

    print("\nRows:", len(result))

    print("\nDirect Sentinel-2 candidates:")
    print(
        result[
            "direct_sentinel2_event"
        ].value_counts(dropna=False)
    )

    print("\nWarnings:")
    print(
        result[
            "automatic_warning"
        ].value_counts(dropna=False)
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
