from pathlib import Path

import pandas as pd


CANDIDATE_INPUT = Path(
    "outputs/358_s2_high_emission_negative_candidates_v1.csv"
)

SELECTED_INPUT = Path(
    "outputs/359_s2_high_emission_matched_negative_manifest_v1.csv"
)

QA_INPUT = Path(
    "outputs/364_s2_high_emission_negative_local_qa_v1.csv"
)

FAILED_OUTPUT = Path(
    "outputs/366_s2_high_emission_failed_negative_qa_v1.csv"
)

MANIFEST_OUTPUT = Path(
    "outputs/367_s2_high_emission_matched_negative_manifest_v2.csv"
)

REPORT_OUTPUT = Path(
    "outputs/368_s2_high_emission_negative_replacement_report.txt"
)


EXPECTED_NEGATIVES = 28


def bool_pass(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def main():
    candidates = pd.read_csv(
        CANDIDATE_INPUT,
        low_memory=False,
    )

    selected = pd.read_csv(
        SELECTED_INPUT,
        low_memory=False,
    )

    qa = pd.read_csv(
        QA_INPUT,
        low_memory=False,
    )

    for frame in [
        candidates,
        selected,
        qa,
    ]:
        if "acquisition_time_utc" in frame.columns:
            frame["acquisition_time_utc"] = pd.to_datetime(
                frame["acquisition_time_utc"],
                errors="coerce",
                utc=True,
            )

    failed_qa = qa[
        ~bool_pass(
            qa["qa_pass_preliminary"]
        )
    ].copy()

    if len(failed_qa) != 1:
        raise RuntimeError(
            "預期剛好 1 張 negative QA 失敗，"
            f"實際找到 {len(failed_qa)} 張。"
        )

    failed_qa.to_csv(
        FAILED_OUTPUT,
        index=False,
    )

    failed_negative_id = str(
        failed_qa.iloc[0]["negative_id"]
    )

    failed_selected = selected[
        selected["negative_id"]
        .astype(str)
        .eq(failed_negative_id)
    ].copy()

    if len(failed_selected) != 1:
        raise RuntimeError(
            "無法在 selected manifest 中找到唯一的失敗 negative："
            f"{failed_negative_id}"
        )

    failed_row = failed_selected.iloc[0]

    positive_id = str(
        failed_row["positive_id"]
    )

    required_side = str(
        failed_row.get(
            "required_temporal_side",
            failed_row["temporal_side"],
        )
    )

    failed_scene_id = str(
        failed_row["scene_id"]
    )

    # 先移除失敗影像，保留其餘 27 張。
    retained = selected[
        ~selected["negative_id"]
        .astype(str)
        .eq(failed_negative_id)
    ].copy()

    used_scene_ids = set(
        retained["scene_id"]
        .dropna()
        .astype(str)
    )

    used_dates_for_positive = set(
        pd.to_datetime(
            retained.loc[
                retained["positive_id"]
                .astype(str)
                .eq(positive_id),
                "acquisition_time_utc",
            ],
            errors="coerce",
            utc=True,
        )
        .dropna()
        .dt.date
    )

    pool = candidates[
        candidates["candidate_status"]
        .astype(str)
        .eq("eligible")
        & candidates["positive_id"]
        .astype(str)
        .eq(positive_id)
        & candidates["temporal_side"]
        .astype(str)
        .eq(required_side)
        & ~candidates["scene_id"]
        .astype(str)
        .isin(used_scene_ids)
        & ~candidates["scene_id"]
        .astype(str)
        .eq(failed_scene_id)
    ].copy()

    pool["candidate_date"] = (
        pool["acquisition_time_utc"]
        .dt.date
    )

    # 維持同一 matched group 內日期不重複。
    pool = pool[
        ~pool["candidate_date"]
        .isin(used_dates_for_positive)
    ].copy()

    pool["absolute_days_from_positive"] = pd.to_numeric(
        pool["absolute_days_from_positive"],
        errors="coerce",
    )

    pool["scene_cloud_percentage"] = pd.to_numeric(
        pool["scene_cloud_percentage"],
        errors="coerce",
    )

    pool["nearest_nonzero_release_hours"] = pd.to_numeric(
        pool["nearest_nonzero_release_hours"],
        errors="coerce",
    )

    pool = pool.sort_values(
        [
            "absolute_days_from_positive",
            "scene_cloud_percentage",
            "nearest_nonzero_release_hours",
            "acquisition_time_utc",
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ],
    )

    if pool.empty:
        raise RuntimeError(
            "找不到符合相同 positive、相同 temporal side、"
            "且未重複日期的替代 negative。"
        )

    replacement = pool.iloc[0].to_dict()

    replacement.update({
        "negative_id":
            failed_negative_id,

        "required_temporal_side":
            required_side,

        "matching_slot_number":
            failed_row.get(
                "matching_slot_number",
                pd.NA,
            ),

        "label":
            0,

        "dataset_role":
            "high_emission_matched_negative",

        "selection_version":
            "s2_high_emission_negative_v2",

        "local_qa_status":
            "pending_replacement_qa",

        "replacement_for_scene_id":
            failed_scene_id,

        "replacement_reason":
            "original_negative_failed_local_qa",
    })

    replacement.pop(
        "candidate_date",
        None,
    )

    replacement_frame = pd.DataFrame(
        [replacement]
    )

    final = pd.concat(
        [
            retained,
            replacement_frame,
        ],
        ignore_index=True,
        sort=False,
    )

    final["acquisition_time_utc"] = pd.to_datetime(
        final["acquisition_time_utc"],
        errors="coerce",
        utc=True,
    )

    final = final.sort_values(
        [
            "matched_positive_time_utc",
            "temporal_side",
            "acquisition_time_utc",
        ]
    ).reset_index(drop=True)

    if len(final) != EXPECTED_NEGATIVES:
        raise RuntimeError(
            f"最終應有 {EXPECTED_NEGATIVES} 張 negatives，"
            f"實際為 {len(final)}。"
        )

    if final["scene_id"].nunique() != EXPECTED_NEGATIVES:
        raise RuntimeError(
            "Replacement 後 scene_id 仍有重複。"
        )

    per_positive = (
        final.groupby(
            "positive_id"
        )["scene_id"]
        .nunique()
    )

    if not per_positive.eq(4).all():
        raise RuntimeError(
            "Replacement 後不是每個 positive 都有 4 張 negatives：\n"
            + per_positive.to_string()
        )

    side_counts = (
        final.groupby(
            [
                "positive_id",
                "temporal_side",
            ]
        )["scene_id"]
        .nunique()
        .unstack(
            fill_value=0
        )
    )

    final.to_csv(
        MANIFEST_OUTPUT,
        index=False,
    )

    report_lines = [
        "=" * 110,
        "SENTINEL-2 HIGH-EMISSION NEGATIVE REPLACEMENT",
        "=" * 110,
        "",
        f"Failed negative ID: {failed_negative_id}",
        f"Failed scene ID: {failed_scene_id}",
        f"Matched positive ID: {positive_id}",
        f"Required temporal side: {required_side}",
        "",
        "Replacement:",
        (
            f"  Scene ID: "
            f"{replacement['scene_id']}"
        ),
        (
            f"  Acquisition: "
            f"{replacement['acquisition_time_utc']}"
        ),
        (
            f"  Days from positive: "
            f"{replacement['days_from_positive']}"
        ),
        (
            f"  Cloud percentage: "
            f"{replacement['scene_cloud_percentage']}"
        ),
        (
            "  Distance from nonzero release: "
            f"{replacement['nearest_nonzero_release_hours']} hours"
        ),
        "",
        f"Final negative rows: {len(final)}",
        (
            "Unique final scene IDs: "
            f"{final['scene_id'].nunique()}"
        ),
        "",
        "Negatives per positive:",
        per_positive.to_string(),
        "",
        "Temporal sides per positive:",
        side_counts.to_string(),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 110)
    print(
        "HIGH-EMISSION NEGATIVE REPLACEMENT COMPLETE"
    )
    print("=" * 110)

    print(
        "\nFailed negative:",
        failed_negative_id,
    )

    print(
        "Failed scene:",
        failed_scene_id,
    )

    print(
        "\nReplacement scene:",
        replacement["scene_id"],
    )

    print(
        "Replacement acquisition:",
        replacement[
            "acquisition_time_utc"
        ],
    )

    print(
        "Temporal side:",
        replacement[
            "temporal_side"
        ],
    )

    print(
        "Cloud percentage:",
        replacement[
            "scene_cloud_percentage"
        ],
    )

    print(
        "Distance from release (hours):",
        replacement[
            "nearest_nonzero_release_hours"
        ],
    )

    print(
        "\nFinal negatives:",
        len(final),
    )

    print(
        "Unique final scenes:",
        final["scene_id"].nunique(),
    )

    print("\nNegatives per positive:")
    print(per_positive)

    print("\nTemporal sides per positive:")
    print(side_counts)

    print("\nSaved:")
    print(FAILED_OUTPUT)
    print(MANIFEST_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
