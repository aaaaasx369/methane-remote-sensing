from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_CSV = Path(
    "outputs/67_landsat_unique_candidate_overpasses.csv"
)

EVIDENCE_CSV = Path(
    "outputs/86_ehrenberg_candidate_release_summary.csv"
)

DECISION_OUTPUT = Path(
    "outputs/87_ehrenberg_final_label_decisions.csv"
)

PRIORITY_OUTPUT = Path(
    "outputs/88_ehrenberg_priority_download_batch.csv"
)

RESERVE_OUTPUT = Path(
    "outputs/89_ehrenberg_reserve_candidates.csv"
)


CAMPAIGN_START = pd.Timestamp(
    "2021-10-16 00:00:00",
    tz="UTC",
)

CAMPAIGN_END = pd.Timestamp(
    "2021-11-03 23:59:59",
    tz="UTC",
)


# 先下載這五張負樣本。
PRIORITY_NEGATIVES = [
    "OP_022",
    "OP_023",
    "OP_024",
    "OP_020",
    "OP_026",
]

# 再加入一張直接有正流量證據的正樣本。
PRIORITY_POSITIVES = [
    "OP_028",
]


def main():
    for path in [
        CANDIDATE_CSV,
        EVIDENCE_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input file: {path}"
            )

    candidates = pd.read_csv(
        CANDIDATE_CSV,
        low_memory=False,
    )

    evidence = pd.read_csv(
        EVIDENCE_CSV,
        low_memory=False,
    )

    # 只保留 Ehrenberg。
    if "site_key" not in candidates.columns:
        raise KeyError(
            "site_key column is missing from candidate table."
        )

    candidates = candidates[
        candidates["site_key"]
        .astype(str)
        .str.lower()
        .eq("ehrenberg")
    ].copy()

    candidates["candidate_time_utc"] = pd.to_datetime(
        candidates["candidate_time_utc"],
        errors="coerce",
        utc=True,
    )

    if candidates["candidate_time_utc"].isna().any():
        raise ValueError(
            "Some candidate acquisition times could not be parsed."
        )

    # 避免 evidence 表中的重複 overpass。
    evidence = (
        evidence
        .sort_values("overpass_id")
        .drop_duplicates(
            subset=["overpass_id"],
            keep="first",
        )
    )

    evidence_columns = [
        column
        for column in [
            "overpass_id",
            "evidence_status",
            "recommended_label",
            "files_covering_date",
            "nearby_evidence_rows",
            "reason",
        ]
        if column in evidence.columns
    ]

    merged = candidates.merge(
        evidence[evidence_columns],
        on="overpass_id",
        how="left",
        validate="one_to_one",
    )

    output_rows = []

    for _, row in merged.iterrows():
        acquisition_time = row[
            "candidate_time_utc"
        ]

        evidence_status = str(
            row.get(
                "evidence_status",
                "",
            )
        )

        inside_campaign = (
            CAMPAIGN_START
            <= acquisition_time
            <= CAMPAIGN_END
        )

        if (
            evidence_status
            == "confirmed_positive_near_overpass"
        ):
            final_label = 1
            final_status = (
                "confirmed_positive_direct_flow_evidence"
            )
            label_confidence = "high"
            label_source = (
                "release_log_near_satellite_overpass"
            )
            controlled_release_present = True

        elif not inside_campaign:
            final_label = 0
            final_status = (
                "schedule_supported_no_controlled_release"
            )
            label_confidence = "high_schedule"
            label_source = (
                "outside_official_campaign_period"
            )
            controlled_release_present = False

        else:
            final_label = np.nan
            final_status = (
                "ambiguous_inside_campaign"
            )
            label_confidence = "ambiguous"
            label_source = (
                "insufficient_release_evidence"
            )
            controlled_release_present = np.nan

        if acquisition_time < CAMPAIGN_START:
            days_from_campaign = (
                acquisition_time
                - CAMPAIGN_START
            ).total_seconds() / 86400

        elif acquisition_time > CAMPAIGN_END:
            days_from_campaign = (
                acquisition_time
                - CAMPAIGN_END
            ).total_seconds() / 86400

        else:
            days_from_campaign = 0.0

        overpass_id = str(
            row["overpass_id"]
        )

        if overpass_id in [
            "OP_022",
            "OP_023",
        ]:
            priority_tier = "A_near_campaign"

        elif overpass_id in PRIORITY_NEGATIVES:
            priority_tier = "B_additional_negative"

        elif overpass_id in PRIORITY_POSITIVES:
            priority_tier = "A_confirmed_positive"

        else:
            priority_tier = "reserve"

        selected_for_download = (
            overpass_id in PRIORITY_NEGATIVES
            or overpass_id in PRIORITY_POSITIVES
        )

        output_row = row.to_dict()

        output_row.update({
            "official_campaign_start_utc":
                CAMPAIGN_START,
            "official_campaign_end_utc":
                CAMPAIGN_END,
            "inside_official_campaign":
                inside_campaign,
            "days_from_campaign_boundary":
                days_from_campaign,
            "final_label":
                final_label,
            "controlled_release_present":
                controlled_release_present,
            "final_status":
                final_status,
            "label_confidence":
                label_confidence,
            "label_source":
                label_source,
            "label_definition":
                (
                    "Presence or absence of the experimental "
                    "controlled release at the Ehrenberg site; "
                    "not a claim that all methane sources in "
                    "the full satellite scene are absent."
                ),
            "schedule_reference":
                (
                    "Sherwin et al. 2023 Scientific Reports; "
                    "campaign 2021-10-16 through 2021-11-03"
                ),
            "priority_tier":
                priority_tier,
            "selected_for_download":
                selected_for_download,
        })

        output_rows.append(
            output_row
        )

    decision = pd.DataFrame(
        output_rows
    )

    # Product ID 必須存在而且唯一。
    required_columns = [
        "overpass_id",
        "LANDSAT_PRODUCT_ID",
        "candidate_time_utc",
        "final_label",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in decision.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    priority = decision[
        decision["selected_for_download"]
        == True
    ].copy()

    reserve = decision[
        decision["selected_for_download"]
        != True
    ].copy()

    expected_ids = set(
        PRIORITY_NEGATIVES
        + PRIORITY_POSITIVES
    )

    actual_ids = set(
        priority["overpass_id"]
        .astype(str)
    )

    if actual_ids != expected_ids:
        raise ValueError(
            "Selected overpass IDs do not match "
            f"the requested batch.\n"
            f"Expected: {sorted(expected_ids)}\n"
            f"Found: {sorted(actual_ids)}"
        )

    if priority[
        "LANDSAT_PRODUCT_ID"
    ].isna().any():
        raise ValueError(
            "A selected scene is missing LANDSAT_PRODUCT_ID."
        )

    if priority[
        "LANDSAT_PRODUCT_ID"
    ].duplicated().any():
        duplicates = priority.loc[
            priority[
                "LANDSAT_PRODUCT_ID"
            ].duplicated(
                keep=False
            ),
            [
                "overpass_id",
                "LANDSAT_PRODUCT_ID",
            ],
        ]

        raise ValueError(
            "Duplicate Landsat products found:\n"
            + duplicates.to_string(index=False)
        )

    # 固定輸出順序。
    priority_order = (
        PRIORITY_NEGATIVES
        + PRIORITY_POSITIVES
    )

    priority[
        "_priority_order"
    ] = priority[
        "overpass_id"
    ].map({
        value: index
        for index, value
        in enumerate(priority_order)
    })

    priority = priority.sort_values(
        "_priority_order"
    ).drop(
        columns="_priority_order"
    )

    decision = decision.sort_values(
        "candidate_time_utc"
    ).reset_index(drop=True)

    reserve = reserve.sort_values(
        "candidate_time_utc"
    ).reset_index(drop=True)

    DECISION_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    decision.to_csv(
        DECISION_OUTPUT,
        index=False,
    )

    priority.to_csv(
        PRIORITY_OUTPUT,
        index=False,
    )

    reserve.to_csv(
        RESERVE_OUTPUT,
        index=False,
    )

    print("=" * 100)
    print("EHRENBERG FINAL LABEL DECISIONS")
    print("=" * 100)

    print(
        f"\nOfficial campaign: "
        f"{CAMPAIGN_START} to {CAMPAIGN_END}"
    )

    print(
        f"\nTotal Ehrenberg candidates: "
        f"{len(decision)}"
    )

    print("\nFinal-status counts:")
    print(
        decision[
            "final_status"
        ].value_counts(
            dropna=False
        )
    )

    print("\nFinal-label counts:")
    print(
        decision[
            "final_label"
        ].value_counts(
            dropna=False
        ).sort_index()
    )

    print("\nPriority download batch:")
    print(
        priority[
            [
                "overpass_id",
                "candidate_time_utc",
                "landsat_sensor",
                "LANDSAT_PRODUCT_ID",
                "WRS_PATH",
                "WRS_ROW",
                "CLOUD_COVER",
                "final_label",
                "final_status",
                "label_confidence",
                "priority_tier",
            ]
        ].to_string(index=False)
    )

    print("\nPriority label counts:")
    print(
        priority[
            "final_label"
        ].value_counts()
        .sort_index()
    )

    print("\nReserve candidates:")
    print(
        reserve[
            [
                "overpass_id",
                "candidate_time_utc",
                "CLOUD_COVER",
                "final_label",
                "final_status",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(DECISION_OUTPUT)
    print(PRIORITY_OUTPUT)
    print(RESERVE_OUTPUT)


if __name__ == "__main__":
    main()
