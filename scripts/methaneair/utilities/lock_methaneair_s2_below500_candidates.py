from pathlib import Path

import pandas as pd


CANDIDATE_INPUT = Path(
    "outputs/457_methaneair_s2_below500_candidate_manifest_v1.csv"
)

QA_INPUT = Path(
    "outputs/456_methaneair_s2_below500_patch_qa_v1.csv"
)

LOCKED_OUTPUT = Path(
    "outputs/459_methaneair_s2_below500_locked_candidate_manifest_v1.csv"
)

EXCLUDED_OUTPUT = Path(
    "outputs/460_methaneair_s2_below500_excluded_patches_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/461_methaneair_s2_below500_locked_report_v1.txt"
)


def main():
    candidates = pd.read_csv(
        CANDIDATE_INPUT,
        low_memory=False,
    )

    qa = pd.read_csv(
        QA_INPUT,
        low_memory=False,
    )

    qa_columns = [
        "event_id",
        "qa_pass",
        "file_exists",
        "raster_read_success",
        "all_zero",
        "has_nan",
        "band_count",
        "width",
        "height",
        "dtype",
        "crs",
        "zero_fraction",
        "source_inside_bounds",
    ]

    qa_columns = [
        column
        for column in qa_columns
        if column in qa.columns
    ]

    frame = candidates.drop(
        columns=[
            column
            for column in qa_columns
            if column != "event_id"
            and column in candidates.columns
        ],
        errors="ignore",
    ).merge(
        qa[qa_columns],
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    frame["qa_pass"] = (
        frame["qa_pass"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    frame["emission_kg_hr"] = pd.to_numeric(
        frame["emission_kg_hr"],
        errors="coerce",
    )

    frame[
        "absolute_time_difference_hours"
    ] = pd.to_numeric(
        frame["absolute_time_difference_hours"],
        errors="coerce",
    )

    locked = frame[
        frame["qa_pass"]
        & frame["emission_kg_hr"].gt(0)
        & frame["emission_kg_hr"].lt(500)
        & frame[
            "absolute_time_difference_hours"
        ].le(6)
    ].copy()

    excluded = frame[
        ~frame["event_id"].isin(
            locked["event_id"]
        )
    ].copy()

    locked["dataset_role"] = (
        "external_low_emission_candidate"
    )

    locked["label"] = 1

    locked["label_status"] = (
        "candidate_positive_not_simultaneous_ground_truth"
    )

    locked["temporal_evidence_level"] = locked[
        "time_match_tier"
    ].map({
        "tier_A_within_1h":
            "strongest_external_temporal_match",

        "tier_B_1_to_3h":
            "moderate_external_temporal_match",

        "tier_C_3_to_6h":
            "weak_external_temporal_match",
    })

    locked["analysis_role"] = locked[
        "time_match_tier"
    ].map({
        "tier_A_within_1h":
            "primary_analysis",

        "tier_B_1_to_3h":
            "primary_analysis",

        "tier_C_3_to_6h":
            "sensitivity_analysis",
    })

    locked["evaluation_group"] = locked[
        "scene_id"
    ]

    locked["do_not_random_split"] = True

    locked["exclusion_reason"] = ""

    excluded["dataset_role"] = "excluded"

    excluded["exclusion_reason"] = ""

    excluded.loc[
        excluded["all_zero"].eq(True),
        "exclusion_reason",
    ] = "all_zero_raster_no_valid_image_data"

    excluded.loc[
        excluded["qa_pass"].eq(False)
        & excluded["exclusion_reason"].eq(""),
        "exclusion_reason",
    ] = "failed_patch_qa"

    locked = locked.sort_values(
        [
            "analysis_role",
            "emission_kg_hr",
            "event_id",
        ]
    ).reset_index(drop=True)

    excluded = excluded.sort_values(
        [
            "exclusion_reason",
            "event_id",
        ]
    ).reset_index(drop=True)

    locked.to_csv(
        LOCKED_OUTPUT,
        index=False,
    )

    excluded.to_csv(
        EXCLUDED_OUTPUT,
        index=False,
    )

    temporal_summary = (
        locked["time_match_tier"]
        .value_counts()
        .reindex(
            [
                "tier_A_within_1h",
                "tier_B_1_to_3h",
                "tier_C_3_to_6h",
            ],
            fill_value=0,
        )
    )

    emission_summary = (
        locked["emission_bin"]
        .value_counts()
        .reindex(
            [
                "0_to_200",
                "200_to_500",
            ],
            fill_value=0,
        )
    )

    analysis_summary = (
        locked["analysis_role"]
        .value_counts()
        .reindex(
            [
                "primary_analysis",
                "sensitivity_analysis",
            ],
            fill_value=0,
        )
    )

    report_lines = [
        "=" * 105,
        "METHANEAIR–S2 BELOW-500 KG/H LOCKED CANDIDATE DATASET V1",
        "=" * 105,
        "",
        f"Downloaded candidate patches: {len(frame)}",
        f"QA-pass locked candidates: {len(locked)}",
        f"Excluded patches: {len(excluded)}",
        (
            "Unique Sentinel-2 scenes after exclusion: "
            f"{locked['scene_id'].nunique()}"
        ),
        "",
        "Temporal tiers:",
        temporal_summary.to_string(),
        "",
        "Emission bins:",
        emission_summary.to_string(),
        "",
        "Analysis roles:",
        analysis_summary.to_string(),
        "",
        "Excluded events:",
        excluded[
            [
                "event_id",
                "emission_kg_hr",
                "time_match_tier",
                "exclusion_reason",
            ]
        ].to_string(index=False),
        "",
        "Interpretation:",
        (
            "These are external low-emission candidate positives, "
            "not locked simultaneous methane-positive ground truth."
        ),
        (
            "Tier A and Tier B are used for primary analysis. "
            "Tier C is reserved for sensitivity analysis."
        ),
        (
            "All train/test evaluation must be grouped by scene_id."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 105)
    print(
        "METHANEAIR–S2 BELOW-500 LOCKED CANDIDATES"
    )
    print("=" * 105)

    print(
        "\nDownloaded candidate patches:",
        len(frame),
    )

    print(
        "QA-pass locked candidates:",
        len(locked),
    )

    print(
        "Excluded patches:",
        len(excluded),
    )

    print(
        "Unique Sentinel-2 scenes after exclusion:",
        locked["scene_id"].nunique(),
    )

    print("\nTemporal tiers:")
    print(temporal_summary)

    print("\nEmission bins:")
    print(emission_summary)

    print("\nAnalysis roles:")
    print(analysis_summary)

    print("\nExcluded events:")
    print(
        excluded[
            [
                "event_id",
                "emission_kg_hr",
                "time_match_tier",
                "exclusion_reason",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(LOCKED_OUTPUT)
    print(EXCLUDED_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
