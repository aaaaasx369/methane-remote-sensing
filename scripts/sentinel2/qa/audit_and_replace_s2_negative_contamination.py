from pathlib import Path
import numpy as np
import pandas as pd


CANDIDATE_INPUT = Path(
    "outputs/321_s2_low_emission_negative_candidates.csv"
)

SELECTED_INPUT = Path(
    "outputs/322_s2_low_emission_matched_negative_manifest_v1.csv"
)

OLD_S2_AUDIT_INPUT = Path(
    "outputs/293_s2_actual_acquisition_time_audit.csv"
)

DIRECT_GT_INPUT = Path(
    "outputs/307_s2_direct_strict_ground_truth_v1.csv"
)

LOW_EMISSION_MANIFEST_INPUT = Path(
    "outputs/317_s2_low_emission_scene_manifest_v1.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/323_s2_negative_contamination_audit.csv"
)

CLEAN_OUTPUT = Path(
    "outputs/324_s2_low_emission_matched_negative_manifest_v2.csv"
)


POSITIVE_TIME_TOLERANCE_MINUTES = 2.0
NEGATIVES_PER_POSITIVE = 4
NEGATIVES_PER_SIDE = 2


def infer_site(value):
    text = str(value).lower()

    if "casa_grande" in text:
        return "Casa_Grande_AZ_release_stacks"

    if "ehrenberg" in text:
        return "Ehrenberg_AZ_release_stack"

    if "evanston" in text:
        return "Evanston_WY_release_site"

    return "unknown"


def find_column(frame, candidates):
    for column in candidates:
        if column in frame.columns:
            return column

    return None


def positive_flag(frame):
    label_column = find_column(
        frame,
        [
            "physical_release_label",
            "strict_label",
            "label",
        ],
    )

    if label_column is None:
        return pd.Series(
            True,
            index=frame.index,
        )

    numeric = pd.to_numeric(
        frame[label_column],
        errors="coerce",
    )

    return numeric.eq(1)


def load_known_positive_times(
    path,
    source_name,
):
    if not path.exists():
        print(
            "Warning: missing known-positive file:",
            path,
        )

        return pd.DataFrame(
            columns=[
                "site",
                "known_positive_time_utc",
                "known_positive_source",
                "known_positive_reference",
            ]
        )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    time_column = find_column(
        frame,
        [
            "actual_s2_time",
            "acquisition_time_utc",
            "best_acquisition_time_utc",
            "s2_image_time",
            "original_event_time",
        ],
    )

    if time_column is None:
        print(
            "Warning: no acquisition-time column in:",
            path,
        )

        return pd.DataFrame(
            columns=[
                "site",
                "known_positive_time_utc",
                "known_positive_source",
                "known_positive_reference",
            ]
        )

    frame = frame[
        positive_flag(frame)
    ].copy()

    frame[
        "known_positive_time_utc"
    ] = pd.to_datetime(
        frame[time_column],
        errors="coerce",
        utc=True,
    )

    if "site" in frame.columns:
        frame["known_site"] = (
            frame["site"]
            .fillna("")
            .astype(str)
        )

        unknown = frame[
            "known_site"
        ].eq("") | frame[
            "known_site"
        ].eq("unknown")

        reference_column = find_column(
            frame,
            [
                "event_id",
                "scene_id",
                "best_scene_id",
            ],
        )

        if reference_column:
            frame.loc[
                unknown,
                "known_site",
            ] = frame.loc[
                unknown,
                reference_column,
            ].map(infer_site)

    else:
        reference_column = find_column(
            frame,
            [
                "event_id",
                "scene_id",
                "best_scene_id",
            ],
        )

        if reference_column:
            frame["known_site"] = frame[
                reference_column
            ].map(infer_site)
        else:
            frame["known_site"] = "unknown"

    reference_column = find_column(
        frame,
        [
            "event_id",
            "scene_id",
            "best_scene_id",
            "selected_release_interval_id",
        ],
    )

    if reference_column:
        references = frame[
            reference_column
        ].astype(str)
    else:
        references = pd.Series(
            "",
            index=frame.index,
        )

    result = pd.DataFrame({
        "site":
            frame["known_site"],

        "known_positive_time_utc":
            frame[
                "known_positive_time_utc"
            ],

        "known_positive_source":
            source_name,

        "known_positive_reference":
            references,
    })

    result = result.dropna(
        subset=[
            "known_positive_time_utc",
        ]
    )

    result = result[
        result["site"].ne("unknown")
    ]

    return result


def build_known_positive_table():
    frames = [
        load_known_positive_times(
            OLD_S2_AUDIT_INPUT,
            "293_s2_actual_acquisition_audit",
        ),

        load_known_positive_times(
            DIRECT_GT_INPUT,
            "307_s2_direct_strict_ground_truth",
        ),

        load_known_positive_times(
            LOW_EMISSION_MANIFEST_INPUT,
            "317_s2_low_emission_manifest",
        ),
    ]

    known = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    known = (
        known.sort_values(
            [
                "site",
                "known_positive_time_utc",
                "known_positive_source",
            ]
        )
        .drop_duplicates(
            subset=[
                "site",
                "known_positive_time_utc",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return known


def audit_against_known_positives(
    frame,
    known,
):
    audited_rows = []

    for _, row in frame.iterrows():
        record = row.to_dict()

        acquisition_time = pd.to_datetime(
            row["acquisition_time_utc"],
            errors="coerce",
            utc=True,
        )

        site = str(row["site"])

        site_known = known[
            known["site"].eq(site)
        ].copy()

        if (
            pd.isna(acquisition_time)
            or site_known.empty
        ):
            minimum_difference = np.nan
            nearest_time = pd.NaT
            nearest_source = ""
            nearest_reference = ""
            contaminated = False

        else:
            differences = (
                site_known[
                    "known_positive_time_utc"
                ]
                - acquisition_time
            ).abs().dt.total_seconds() / 60.0

            nearest_index = (
                differences.idxmin()
            )

            minimum_difference = float(
                differences.loc[
                    nearest_index
                ]
            )

            nearest_row = site_known.loc[
                nearest_index
            ]

            nearest_time = nearest_row[
                "known_positive_time_utc"
            ]

            nearest_source = nearest_row[
                "known_positive_source"
            ]

            nearest_reference = nearest_row[
                "known_positive_reference"
            ]

            contaminated = (
                minimum_difference
                <= POSITIVE_TIME_TOLERANCE_MINUTES
            )

        record.update({
            "nearest_known_positive_time_utc":
                nearest_time,

            "known_positive_time_difference_minutes":
                minimum_difference,

            "nearest_known_positive_source":
                nearest_source,

            "nearest_known_positive_reference":
                nearest_reference,

            "positive_contamination":
                contaminated,

            "contamination_status":
                (
                    "exclude_known_positive"
                    if contaminated
                    else "clean_candidate"
                ),
        })

        audited_rows.append(record)

    return pd.DataFrame(
        audited_rows
    )


def choose_replacements(
    candidates,
    clean_selected,
):
    final_groups = []

    global_used_scene_ids = set(
        clean_selected[
            "scene_id"
        ].dropna().astype(str)
    )

    positive_groups = (
        candidates[
            [
                "matched_positive_scene_id",
                "matched_positive_time_utc",
                "matched_positive_rate_kg_h",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "matched_positive_time_utc"
        )
    )

    for _, positive in (
        positive_groups.iterrows()
    ):
        positive_scene_id = positive[
            "matched_positive_scene_id"
        ]

        current = clean_selected[
            clean_selected[
                "matched_positive_scene_id"
            ].eq(positive_scene_id)
        ].copy()

        current = current.drop_duplicates(
            subset=["scene_id"]
        )

        used_dates = set(
            pd.to_datetime(
                current[
                    "acquisition_time_utc"
                ],
                errors="coerce",
                utc=True,
            ).dt.date.dropna()
        )

        pool = candidates[
            candidates[
                "matched_positive_scene_id"
            ].eq(positive_scene_id)
            & candidates[
                "candidate_status"
            ].eq("eligible")
            & ~candidates[
                "positive_contamination"
            ].astype(bool)
        ].copy()

        pool[
            "acquisition_time_utc"
        ] = pd.to_datetime(
            pool[
                "acquisition_time_utc"
            ],
            errors="coerce",
            utc=True,
        )

        pool = pool.sort_values(
            [
                "absolute_days_from_positive",
                "scene_cloud_percentage",
                "acquisition_time_utc",
            ]
        )

        selected_records = (
            current.to_dict("records")
        )

        for side in ["before", "after"]:
            current_side_count = sum(
                str(record.get(
                    "temporal_side"
                )) == side
                for record in selected_records
            )

            needed = max(
                0,
                NEGATIVES_PER_SIDE
                - current_side_count,
            )

            if needed == 0:
                continue

            side_pool = pool[
                pool[
                    "temporal_side"
                ].eq(side)
            ]

            for _, candidate in (
                side_pool.iterrows()
            ):
                scene_id = str(
                    candidate["scene_id"]
                )

                acquisition_date = (
                    candidate[
                        "acquisition_time_utc"
                    ].date()
                )

                if (
                    scene_id
                    in global_used_scene_ids
                ):
                    continue

                if acquisition_date in used_dates:
                    continue

                selected_records.append(
                    candidate.to_dict()
                )

                global_used_scene_ids.add(
                    scene_id
                )

                used_dates.add(
                    acquisition_date
                )

                needed -= 1

                if needed == 0:
                    break

        remaining_needed = (
            NEGATIVES_PER_POSITIVE
            - len(selected_records)
        )

        if remaining_needed > 0:
            for _, candidate in (
                pool.iterrows()
            ):
                scene_id = str(
                    candidate["scene_id"]
                )

                acquisition_date = (
                    candidate[
                        "acquisition_time_utc"
                    ].date()
                )

                if (
                    scene_id
                    in global_used_scene_ids
                ):
                    continue

                if acquisition_date in used_dates:
                    continue

                selected_records.append(
                    candidate.to_dict()
                )

                global_used_scene_ids.add(
                    scene_id
                )

                used_dates.add(
                    acquisition_date
                )

                remaining_needed -= 1

                if remaining_needed == 0:
                    break

        selected = pd.DataFrame(
            selected_records
        )

        if len(selected) > NEGATIVES_PER_POSITIVE:
            selected = (
                selected.sort_values(
                    [
                        "absolute_days_from_positive",
                        "scene_cloud_percentage",
                    ]
                )
                .head(
                    NEGATIVES_PER_POSITIVE
                )
            )

        final_groups.append(
            selected
        )

    final = pd.concat(
        final_groups,
        ignore_index=True,
        sort=False,
    )

    final[
        "acquisition_time_utc"
    ] = pd.to_datetime(
        final[
            "acquisition_time_utc"
        ],
        errors="coerce",
        utc=True,
    )

    final[
        "matched_positive_time_utc"
    ] = pd.to_datetime(
        final[
            "matched_positive_time_utc"
        ],
        errors="coerce",
        utc=True,
    )

    final = final.sort_values(
        [
            "matched_positive_time_utc",
            "acquisition_time_utc",
        ]
    ).reset_index(drop=True)

    final["label"] = 0
    final["dataset_role"] = (
        "matched_negative"
    )
    final["local_qa_status"] = (
        "pending"
    )
    final["selection_version"] = (
        "s2_low_emission_negative_v2"
    )
    final["positive_contamination"] = False
    final["contamination_status"] = (
        "clean_candidate"
    )

    negative_ids = []

    for group_number, (
        _,
        group,
    ) in enumerate(
        final.groupby(
            "matched_positive_scene_id",
            sort=False,
        ),
        start=1,
    ):
        for item_number in range(
            1,
            len(group) + 1,
        ):
            negative_ids.append(
                f"S2_NEG_{group_number:02d}_"
                f"{item_number:02d}"
            )

    final["negative_id"] = negative_ids

    return final


def main():
    candidates = pd.read_csv(
        CANDIDATE_INPUT,
        low_memory=False,
    )

    selected = pd.read_csv(
        SELECTED_INPUT,
        low_memory=False,
    )

    for frame in [
        candidates,
        selected,
    ]:
        frame[
            "acquisition_time_utc"
        ] = pd.to_datetime(
            frame[
                "acquisition_time_utc"
            ],
            errors="coerce",
            utc=True,
        )

        frame[
            "matched_positive_time_utc"
        ] = pd.to_datetime(
            frame[
                "matched_positive_time_utc"
            ],
            errors="coerce",
            utc=True,
        )

    known = (
        build_known_positive_table()
    )

    audited_candidates = (
        audit_against_known_positives(
            candidates,
            known,
        )
    )

    audited_selected = (
        audit_against_known_positives(
            selected,
            known,
        )
    )

    contaminated_selected = (
        audited_selected[
            audited_selected[
                "positive_contamination"
            ].astype(bool)
        ].copy()
    )

    clean_selected = audited_selected[
        ~audited_selected[
            "positive_contamination"
        ].astype(bool)
    ].copy()

    final = choose_replacements(
        candidates=audited_candidates,
        clean_selected=clean_selected,
    )

    audited_selected.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    final.to_csv(
        CLEAN_OUTPUT,
        index=False,
    )

    print("=" * 110)
    print(
        "SENTINEL-2 NEGATIVE CONTAMINATION AUDIT"
    )
    print("=" * 110)

    print(
        "\nKnown positive acquisition times:",
        len(known),
    )

    print(
        "Original selected negatives:",
        len(selected),
    )

    print(
        "Contaminated selected negatives:",
        len(contaminated_selected),
    )

    if not contaminated_selected.empty:
        print("\nExcluded contaminated scenes:")
        print(
            contaminated_selected[
                [
                    "negative_id",
                    "site",
                    "acquisition_time_utc",
                    "scene_id",
                    "nearest_known_positive_time_utc",
                    "known_positive_time_difference_minutes",
                    "nearest_known_positive_reference",
                ]
            ].to_string(
                index=False,
            )
        )

    print(
        "\nFinal clean negatives:",
        len(final),
    )

    print("\nFinal negatives per positive:")
    print(
        final.groupby(
            [
                "matched_positive_time_utc",
                "matched_positive_rate_kg_h",
            ]
        )["scene_id"].nunique()
    )

    print("\nTemporal side:")
    print(
        final[
            "temporal_side"
        ].value_counts(
            dropna=False
        )
    )

    print(
        "\nKnown-positive contamination "
        "remaining:",
        int(
            final[
                "positive_contamination"
            ].astype(bool).sum()
        ),
    )

    print(
        "\nMinimum distance from "
        "nonzero interval inventory (hours):",
        final[
            "nearest_nonzero_release_hours"
        ].min(),
    )

    print("\nClean final scenes:")
    print(
        final[
            [
                "negative_id",
                "matched_positive_time_utc",
                "acquisition_time_utc",
                "days_from_positive",
                "temporal_side",
                "scene_cloud_percentage",
                "nearest_nonzero_release_hours",
                "scene_id",
            ]
        ].to_string(
            index=False,
        )
    )

    print("\nSaved:")
    print(AUDIT_OUTPUT)
    print(CLEAN_OUTPUT)


if __name__ == "__main__":
    main()
