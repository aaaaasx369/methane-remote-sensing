from pathlib import Path
import re

import numpy as np
import pandas as pd


CANDIDATE_INPUT = Path(
    "outputs/61_landsat_matched_negative_candidates.csv"
)

POSITIVE_INPUT = Path(
    "outputs/396_landsat_final_confirmed_features_site_repaired_v1.csv"
)

INDEPENDENT_OUTPUT = Path(
    "outputs/393_landsat_independent_negative_candidates_v1.csv"
)

PAIRING_OUTPUT = Path(
    "outputs/394_landsat_negative_pairing_feasibility_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/395_landsat_negative_pairing_feasibility_report_v1.txt"
)


def find_column(
    frame,
    candidates,
    table_name,
    required=True,
):
    for column in candidates:
        if column in frame.columns:
            return column

    if required:
        raise KeyError(
            f"{table_name} 找不到欄位："
            + ", ".join(candidates)
        )

    return None


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


def normalize_site_alias(value):
    text = str(value).strip().lower()

    if "casa" in text:
        return "casa_grande"

    if "ehrenberg" in text:
        return "ehrenberg"

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    ).strip("_")


def main():
    if not CANDIDATE_INPUT.exists():
        raise FileNotFoundError(
            CANDIDATE_INPUT
        )

    if not POSITIVE_INPUT.exists():
        raise FileNotFoundError(
            POSITIVE_INPUT
        )

    candidates = pd.read_csv(
        CANDIDATE_INPUT,
        low_memory=False,
    )

    confirmed = pd.read_csv(
        POSITIVE_INPUT,
        low_memory=False,
    )

    required_candidate_columns = [
        "candidate_role",
        "candidate_time_utc",
        "site_name_normalized",
        "landsat_sensor",
        "LANDSAT_PRODUCT_ID",
        "CLOUD_COVER",
    ]

    missing_candidate_columns = [
        column
        for column in required_candidate_columns
        if column not in candidates.columns
    ]

    if missing_candidate_columns:
        raise KeyError(
            "Candidate table 缺少欄位："
            + ", ".join(
                missing_candidate_columns
            )
        )

    confirmed["label"] = pd.to_numeric(
        confirmed["label"],
        errors="raise",
    ).astype(int)

    positives = confirmed[
        confirmed["label"].eq(1)
    ].copy()

    if len(positives) != 7:
        raise RuntimeError(
            "預期 7 張 confirmed positives，"
            f"實際為 {len(positives)}。"
        )

    positive_site_column = find_column(
        positives,
        [
            "site",
            "site_name",
            "site_name_normalized",
            "release_site",
        ],
        "Confirmed positive table",
    )

    positive_time_column = find_column(
        positives,
        [
            "landsat_image_time",
            "acquisition_time_utc",
            "image_time_utc",
            "scene_time_utc",
        ],
        "Confirmed positive table",
    )

    positive_id_column = find_column(
        positives,
        [
            "raster_group_id",
            "sample_id",
            "pixel_hash",
            "LANDSAT_PRODUCT_ID",
            "scene_id",
        ],
        "Confirmed positive table",
    )

    positive_sensor_column = find_column(
        positives,
        [
            "landsat_sensor",
            "sensor",
            "platform",
        ],
        "Confirmed positive table",
        required=False,
    )

    release_rate_column = find_column(
        positives,
        [
            "release_rate_kg_h",
            "final_release_rate_kg_h",
            "preferred_release_rate_kg_h",
            "matched_release_rate_kg_h",
            "cr_kgh_CH4_mean300",
        ],
        "Confirmed positive table",
        required=False,
    )

    # 必須直接使用已整理好的 UTC 欄位，
    # 不使用數值型 system:time_start。
    candidates[
        "candidate_time_parsed_utc"
    ] = pd.to_datetime(
        candidates[
            "candidate_time_utc"
        ],
        errors="coerce",
        utc=True,
    )

    candidates["site_alias"] = (
        candidates[
            "site_name_normalized"
        ].map(
            normalize_site_alias
        )
    )

    candidates[
        "candidate_role_normalized"
    ] = (
        candidates[
            "candidate_role"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    candidates[
        "cloud_cover_numeric"
    ] = pd.to_numeric(
        candidates[
            "CLOUD_COVER"
        ],
        errors="coerce",
    )

    candidates[
        "priority_score_numeric"
    ] = pd.to_numeric(
        candidates.get(
            "candidate_priority_score",
            np.nan,
        ),
        errors="coerce",
    )

    candidates[
        "same_reference_wrs_bool"
    ] = (
        parse_bool(
            candidates[
                "same_reference_wrs"
            ]
        )
        if "same_reference_wrs"
        in candidates.columns
        else True
    )

    candidates[
        "sensor_match_bool"
    ] = (
        parse_bool(
            candidates[
                "sensor_matches_positive_at_site"
            ]
        )
        if "sensor_matches_positive_at_site"
        in candidates.columns
        else True
    )

    new_candidates = candidates[
        candidates[
            "candidate_role_normalized"
        ].str.startswith(
            "new"
        )
    ].copy()

    invalid_time_count = int(
        new_candidates[
            "candidate_time_parsed_utc"
        ].isna().sum()
    )

    new_candidates = (
        new_candidates.dropna(
            subset=[
                "candidate_time_parsed_utc",
                "site_alias",
                "LANDSAT_PRODUCT_ID",
            ]
        )
        .copy()
    )

    # 相同場址、衛星、日期的相鄰 WRS scenes
    # 視為同一次 overpass。
    new_candidates[
        "independent_overpass_key"
    ] = (
        new_candidates[
            "site_alias"
        ].astype(str)
        + "|"
        + new_candidates[
            "landsat_sensor"
        ].astype(str)
        + "|"
        + new_candidates[
            "candidate_time_parsed_utc"
        ].dt.strftime(
            "%Y-%m-%d"
        )
    )

    # 同一次 overpass 如果有相鄰 WRS scenes，
    # 優先保留 reference WRS、sensor match，
    # 然後使用較高 priority 和較低雲量。
    new_candidates = (
        new_candidates.sort_values(
            [
                "same_reference_wrs_bool",
                "sensor_match_bool",
                "priority_score_numeric",
                "cloud_cover_numeric",
                "candidate_time_parsed_utc",
            ],
            ascending=[
                False,
                False,
                True,
                True,
                True,
            ],
            na_position="last",
        )
        .drop_duplicates(
            subset=[
                "independent_overpass_key"
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    new_candidates[
        "independent_candidate_id"
    ] = [
        f"LANDSAT_NEG_CAND_{number:03d}"
        for number in range(
            1,
            len(new_candidates) + 1,
        )
    ]

    new_candidates.to_csv(
        INDEPENDENT_OUTPUT,
        index=False,
    )

    positives[
        "positive_time_utc"
    ] = pd.to_datetime(
        positives[
            positive_time_column
        ],
        errors="coerce",
        utc=True,
    )

    positives["site_alias"] = (
        positives[
            positive_site_column
        ].map(
            normalize_site_alias
        )
    )

    positives[
        "positive_id_standard"
    ] = positives[
        positive_id_column
    ].astype(str)

    if positive_sensor_column is None:
        positives[
            "positive_sensor"
        ] = "Landsat"
    else:
        positives[
            "positive_sensor"
        ] = positives[
            positive_sensor_column
        ].astype(str)

    if release_rate_column is None:
        positives[
            "positive_release_rate_kg_h"
        ] = np.nan
    else:
        positives[
            "positive_release_rate_kg_h"
        ] = pd.to_numeric(
            positives[
                release_rate_column
            ],
            errors="coerce",
        )

    if positives[
        "positive_time_utc"
    ].isna().any():
        raise RuntimeError(
            "部分 positive 的拍攝時間無法解析。"
        )

    pairing_rows = []
    feasibility_rows = []

    for _, positive in positives.iterrows():
        same_site = new_candidates[
            new_candidates[
                "site_alias"
            ].eq(
                positive[
                    "site_alias"
                ]
            )
        ].copy()

        before_count = 0
        after_count = 0
        same_sensor_before_count = 0
        same_sensor_after_count = 0

        for _, candidate in (
            same_site.iterrows()
        ):
            days_difference = (
                candidate[
                    "candidate_time_parsed_utc"
                ]
                - positive[
                    "positive_time_utc"
                ]
            ).total_seconds() / 86400.0

            if days_difference < 0:
                temporal_side = "before"
                before_count += 1

            elif days_difference > 0:
                temporal_side = "after"
                after_count += 1

            else:
                temporal_side = "same_day"

            same_sensor = (
                str(
                    candidate[
                        "landsat_sensor"
                    ]
                )
                ==
                str(
                    positive[
                        "positive_sensor"
                    ]
                )
            )

            if (
                same_sensor
                and temporal_side == "before"
            ):
                same_sensor_before_count += 1

            if (
                same_sensor
                and temporal_side == "after"
            ):
                same_sensor_after_count += 1

            pairing_rows.append({
                "positive_id":
                    positive[
                        "positive_id_standard"
                    ],

                "positive_site":
                    positive[
                        positive_site_column
                    ],

                "positive_site_alias":
                    positive[
                        "site_alias"
                    ],

                "positive_time_utc":
                    positive[
                        "positive_time_utc"
                    ],

                "positive_sensor":
                    positive[
                        "positive_sensor"
                    ],

                "positive_release_rate_kg_h":
                    positive[
                        "positive_release_rate_kg_h"
                    ],

                "candidate_id":
                    candidate[
                        "independent_candidate_id"
                    ],

                "candidate_scene_id":
                    candidate[
                        "LANDSAT_PRODUCT_ID"
                    ],

                "independent_overpass_key":
                    candidate[
                        "independent_overpass_key"
                    ],

                "candidate_time_utc":
                    candidate[
                        "candidate_time_parsed_utc"
                    ],

                "candidate_sensor":
                    candidate[
                        "landsat_sensor"
                    ],

                "same_sensor_as_positive":
                    same_sensor,

                "candidate_cloud":
                    candidate[
                        "cloud_cover_numeric"
                    ],

                "days_from_positive":
                    days_difference,

                "absolute_days_from_positive":
                    abs(
                        days_difference
                    ),

                "temporal_side":
                    temporal_side,

                "release_check_required":
                    (
                        candidate.get(
                            "release_check_required",
                            True,
                        )
                    ),

                "pairing_eligible_by_time":
                    temporal_side
                    in {
                        "before",
                        "after",
                    },
            })

        feasibility_rows.append({
            "positive_id":
                positive[
                    "positive_id_standard"
                ],

            "positive_site":
                positive[
                    positive_site_column
                ],

            "positive_site_alias":
                positive[
                    "site_alias"
                ],

            "positive_time_utc":
                positive[
                    "positive_time_utc"
                ],

            "positive_sensor":
                positive[
                    "positive_sensor"
                ],

            "positive_release_rate_kg_h":
                positive[
                    "positive_release_rate_kg_h"
                ],

            "total_candidates":
                len(same_site),

            "before_candidates":
                before_count,

            "after_candidates":
                after_count,

            "same_sensor_before_candidates":
                same_sensor_before_count,

            "same_sensor_after_candidates":
                same_sensor_after_count,

            "can_fill_two_before":
                before_count >= 2,

            "can_fill_two_after":
                after_count >= 2,

            "can_fill_locked_2_before_2_after":
                (
                    before_count >= 2
                    and after_count >= 2
                ),

            "can_fill_same_sensor_2_before_2_after":
                (
                    same_sensor_before_count >= 2
                    and same_sensor_after_count >= 2
                ),
        })

    pairing = pd.DataFrame(
        pairing_rows
    )

    pairing.to_csv(
        PAIRING_OUTPUT,
        index=False,
    )

    feasibility = pd.DataFrame(
        feasibility_rows
    ).sort_values(
        [
            "positive_site_alias",
            "positive_time_utc",
        ]
    )

    new_role_count = int(
        candidates[
            "candidate_role_normalized"
        ].str.startswith(
            "new"
        ).sum()
    )

    expected_slots = (
        len(positives) * 4
    )

    site_counts = (
        new_candidates[
            "site_alias"
        ].value_counts()
    )

    date_counts = (
        new_candidates.assign(
            acquisition_date=
                new_candidates[
                    "candidate_time_parsed_utc"
                ].dt.date
        )
        .groupby(
            [
                "site_alias",
                "landsat_sensor",
            ]
        )[
            "acquisition_date"
        ]
        .nunique()
    )

    all_fill = bool(
        feasibility[
            "can_fill_locked_2_before_2_after"
        ].all()
    )

    report_lines = [
        "=" * 115,
        "LANDSAT MATCHED-NEGATIVE PAIRING FEASIBILITY V2",
        "=" * 115,
        "",
        f"All candidate rows: {len(candidates)}",
        f"New candidate rows: {new_role_count}",
        (
            "New rows with invalid candidate_time_utc: "
            f"{invalid_time_count}"
        ),
        (
            "Independent negative candidate overpasses: "
            f"{len(new_candidates)}"
        ),
        f"Confirmed positives: {len(positives)}",
        f"Required negative slots: {expected_slots}",
        "",
        "Independent candidates per site:",
        site_counts.to_string(),
        "",
        "Unique acquisition dates by site and sensor:",
        date_counts.to_string(),
        "",
        "Per-positive feasibility:",
        feasibility.to_string(
            index=False
        ),
        "",
        (
            "All positives can fill "
            f"2 before + 2 after: {all_fill}"
        ),
        "",
        "Important:",
        (
            "These are candidate negative overpasses only. "
            "Rows marked release_check_required must still "
            "be checked against controlled-release intervals."
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 115)
    print(
        "LANDSAT PAIRING FEASIBILITY SUMMARY V2"
    )
    print("=" * 115)

    print(
        "\nAll candidate rows:",
        len(candidates),
    )

    print(
        "New candidate rows:",
        new_role_count,
    )

    print(
        "Invalid new candidate times:",
        invalid_time_count,
    )

    print(
        "Independent negative candidate overpasses:",
        len(new_candidates),
    )

    print(
        "Confirmed positives:",
        len(positives),
    )

    print(
        "Required negative slots:",
        expected_slots,
    )

    print(
        "\nIndependent candidates per site:"
    )

    print(site_counts)

    print(
        "\nPer-positive feasibility:"
    )

    print(
        feasibility.to_string(
            index=False
        )
    )

    print(
        "\nAll positives can fill "
        "2 before + 2 after:",
        all_fill,
    )

    print("\nSaved:")
    print(INDEPENDENT_OUTPUT)
    print(PAIRING_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
