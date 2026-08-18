from pathlib import Path

import numpy as np
import pandas as pd


OVERLAP_AUDIT_INPUT = Path(
    "outputs/353_s2_high_low_scene_overlap_audit_v1.csv"
)

CLEAN_OUTPUT = Path(
    "outputs/355_s2_high_emission_positive_manifest_clean_v1.csv"
)

EXCLUDED_OUTPUT = Path(
    "outputs/356_s2_high_emission_excluded_scene_audit_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/357_s2_high_emission_positive_manifest_report.txt"
)


def parse_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "true",
            "1",
            "yes",
        ])
    )


def append_reason(
    frame,
    mask,
    reason,
):
    current = frame.loc[
        mask,
        "exclusion_reason",
    ].fillna("")

    frame.loc[
        mask,
        "exclusion_reason",
    ] = np.where(
        current.eq(""),
        reason,
        current + ";" + reason,
    )


def main():
    if not OVERLAP_AUDIT_INPUT.exists():
        raise FileNotFoundError(
            OVERLAP_AUDIT_INPUT
        )

    audit = pd.read_csv(
        OVERLAP_AUDIT_INPUT,
        low_memory=False,
    )

    required_columns = [
        "positive_id",
        "scene_id",
        "site_name",
        "resolved_acquisition_time_utc",
        "preferred_release_rate_kg_h",
        "lat",
        "lon",
        "resolution_status",
        "same_scene_in_low_pilot",
        "overlap_status",
    ]

    missing = [
        column
        for column in required_columns
        if column not in audit.columns
    ]

    if missing:
        raise KeyError(
            "Overlap audit 缺少欄位："
            + ", ".join(missing)
        )

    audit[
        "resolved_acquisition_time_utc"
    ] = pd.to_datetime(
        audit[
            "resolved_acquisition_time_utc"
        ],
        errors="coerce",
        utc=True,
    )

    audit[
        "preferred_release_rate_kg_h"
    ] = pd.to_numeric(
        audit[
            "preferred_release_rate_kg_h"
        ],
        errors="coerce",
    )

    audit["lat"] = pd.to_numeric(
        audit["lat"],
        errors="coerce",
    )

    audit["lon"] = pd.to_numeric(
        audit["lon"],
        errors="coerce",
    )

    audit[
        "same_scene_in_low_pilot"
    ] = parse_bool(
        audit[
            "same_scene_in_low_pilot"
        ]
    )

    audit["scene_id"] = (
        audit["scene_id"]
        .astype("string")
        .str.strip()
    )

    audit[
        "duplicate_within_high_inventory"
    ] = (
        audit["scene_id"].notna()
        & audit["scene_id"].ne("")
        & audit["scene_id"].duplicated(
            keep=False
        )
    )

    audit["exclusion_reason"] = ""

    unresolved = ~audit[
        "resolution_status"
    ].astype(str).eq(
        "resolved_exact"
    )

    append_reason(
        audit,
        unresolved,
        "scene_not_exactly_resolved",
    )

    missing_scene = (
        audit["scene_id"].isna()
        | audit["scene_id"].eq("")
    )

    append_reason(
        audit,
        missing_scene,
        "missing_scene_id",
    )

    invalid_core = (
        audit[
            [
                "resolved_acquisition_time_utc",
                "preferred_release_rate_kg_h",
                "lat",
                "lon",
            ]
        ]
        .isna()
        .any(axis=1)
    )

    append_reason(
        audit,
        invalid_core,
        "missing_core_metadata",
    )

    # 同一張影像已經存在 low-emission pilot：
    # 不論排放率是否一致，都不能在 high set 重複使用。
    low_overlap = audit[
        "same_scene_in_low_pilot"
    ]

    append_reason(
        audit,
        low_overlap,
        "same_scene_already_used_in_low_emission_pilot",
    )

    conflicting_rate = audit[
        "overlap_status"
    ].astype(str).eq(
        "same_scene_conflicting_rate"
    )

    append_reason(
        audit,
        conflicting_rate,
        "conflicting_release_rate_between_ground_truth_sources",
    )

    high_duplicate = audit[
        "duplicate_within_high_inventory"
    ]

    append_reason(
        audit,
        high_duplicate,
        "duplicate_scene_within_high_inventory",
    )

    audit[
        "final_include_high_emission"
    ] = audit[
        "exclusion_reason"
    ].eq("")

    clean = audit[
        audit[
            "final_include_high_emission"
        ]
    ].copy()

    excluded = audit[
        ~audit[
            "final_include_high_emission"
        ]
    ].copy()

    clean = clean.sort_values(
        [
            "resolved_acquisition_time_utc",
            "preferred_release_rate_kg_h",
        ]
    ).reset_index(drop=True)

    excluded = excluded.sort_values(
        [
            "resolved_acquisition_time_utc",
            "preferred_release_rate_kg_h",
        ]
    ).reset_index(drop=True)

    clean[
        "original_positive_id"
    ] = clean[
        "positive_id"
    ]

    clean["positive_id"] = [
        f"S2_HIGH_CLEAN_{number:02d}"
        for number in range(
            1,
            len(clean) + 1,
        )
    ]

    clean[
        "matched_group_id"
    ] = clean["scene_id"]

    clean[
        "acquisition_time_utc"
    ] = clean[
        "resolved_acquisition_time_utc"
    ]

    clean[
        "release_rate_kg_h"
    ] = clean[
        "preferred_release_rate_kg_h"
    ]

    clean["site"] = clean[
        "site_name"
    ]

    clean["label"] = 1

    clean[
        "dataset_role"
    ] = (
        "strict_high_emission_positive_clean"
    )

    clean[
        "ground_truth_status"
    ] = (
        "strict_primary_resolved_unique"
    )

    clean[
        "manifest_version"
    ] = (
        "s2_high_emission_positive_clean_v1"
    )

    preferred_columns = [
        "positive_id",
        "original_positive_id",
        "event_id",
        "scene_id",
        "matched_group_id",
        "site",
        "site_name",
        "acquisition_time_utc",
        "scene_lookup_time_utc",
        "resolved_acquisition_time_utc",
        "release_rate_kg_h",
        "preferred_release_rate_kg_h",
        "lat",
        "lon",
        "mgrs_tile",
        "cloudy_pixel_percentage",
        "scene_time_difference_seconds",
        "resolution_status",
        "label",
        "dataset_role",
        "ground_truth_status",
        "raw_source_file",
        "rate_source",
        "manifest_version",
    ]

    existing_columns = [
        column
        for column in preferred_columns
        if column in clean.columns
    ]

    remaining_columns = [
        column
        for column in clean.columns
        if column not in existing_columns
    ]

    clean = clean[
        existing_columns
        + remaining_columns
    ]

    clean.to_csv(
        CLEAN_OUTPUT,
        index=False,
    )

    excluded.to_csv(
        EXCLUDED_OUTPUT,
        index=False,
    )

    unique_clean_scenes = int(
        clean[
            "scene_id"
        ].nunique()
    )

    release_statistics = (
        clean[
            "release_rate_kg_h"
        ].describe()
        if not clean.empty
        else pd.Series(dtype=float)
    )

    report_lines = [
        "=" * 110,
        (
            "SENTINEL-2 CLEAN HIGH-EMISSION "
            "POSITIVE MANIFEST"
        ),
        "=" * 110,
        "",
        f"Input resolved rows: {len(audit)}",
        f"Included clean rows: {len(clean)}",
        f"Unique clean scene IDs: {unique_clean_scenes}",
        f"Excluded rows: {len(excluded)}",
        "",
        "Exclusion reasons:",
        (
            excluded[
                "exclusion_reason"
            ].value_counts(
                dropna=False
            ).to_string()
            if not excluded.empty
            else "None"
        ),
        "",
        "Release-rate statistics:",
        (
            release_statistics.to_string()
            if not release_statistics.empty
            else "No included scenes"
        ),
        "",
        "Scenes per site:",
        (
            clean[
                "site"
            ].value_counts().to_string()
            if not clean.empty
            else "No included scenes"
        ),
        "",
        "Included clean scenes:",
        (
            clean[
                [
                    "positive_id",
                    "site",
                    "acquisition_time_utc",
                    "release_rate_kg_h",
                    "mgrs_tile",
                    "scene_id",
                ]
            ].to_string(index=False)
            if not clean.empty
            else "None"
        ),
        "",
        "Excluded scenes:",
        (
            excluded[
                [
                    "positive_id",
                    "site_name",
                    "resolved_acquisition_time_utc",
                    "preferred_release_rate_kg_h",
                    "low_release_rate_kg_h",
                    "scene_id",
                    "overlap_status",
                    "exclusion_reason",
                ]
            ].to_string(index=False)
            if not excluded.empty
            else "None"
        ),
        "",
        "Ground-truth decision:",
        (
            "The 2021-11-03 Sentinel-2 scene is retained "
            "only in the locked low-emission pilot and "
            "excluded from the high-emission expansion "
            "because the same image has conflicting "
            "release-rate estimates."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 110)
    print(
        "FINALIZE CLEAN SENTINEL-2 "
        "HIGH-EMISSION POSITIVES"
    )
    print("=" * 110)

    print(
        "\nInput resolved rows:",
        len(audit),
    )

    print(
        "Included clean rows:",
        len(clean),
    )

    print(
        "Unique clean scene IDs:",
        unique_clean_scenes,
    )

    print(
        "Excluded rows:",
        len(excluded),
    )

    print("\nExclusion reasons:")

    if excluded.empty:
        print("None")
    else:
        print(
            excluded[
                "exclusion_reason"
            ].value_counts(
                dropna=False
            )
        )

    print(
        "\nClean high-emission positives:"
    )

    if clean.empty:
        print("None")
    else:
        print(
            clean[
                [
                    "positive_id",
                    "site",
                    "acquisition_time_utc",
                    "release_rate_kg_h",
                    "mgrs_tile",
                    "scene_id",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nExcluded scenes:")

    if excluded.empty:
        print("None")
    else:
        display_columns = [
            "positive_id",
            "site_name",
            "resolved_acquisition_time_utc",
            "preferred_release_rate_kg_h",
            "low_release_rate_kg_h",
            "scene_id",
            "overlap_status",
            "exclusion_reason",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in excluded.columns
        ]

        print(
            excluded[
                display_columns
            ].to_string(
                index=False
            )
        )

    print("\nSaved:")
    print(CLEAN_OUTPUT)
    print(EXCLUDED_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
