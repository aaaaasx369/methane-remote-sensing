from pathlib import Path
import re

import numpy as np
import pandas as pd


DATASET_CSV = Path(
    "outputs/95_landsat_strict_core_v2_features.csv"
)

EVIDENCE_OUTPUT = Path(
    "outputs/104_landsat_positive_release_evidence_long.csv"
)

INVENTORY_OUTPUT = Path(
    "outputs/105_landsat_positive_release_inventory.csv"
)


# 可能包含 release ground truth 的現有輸出檔。
OPTIONAL_SOURCE_FILES = [
    Path("outputs/52_2022_landsat_scene_label_review.csv"),
    Path("outputs/56_2021_landsat_scene_label_review.csv"),
    Path("outputs/57_landsat_final_confirmed_features.csv"),
    Path("outputs/75_casa_grande_schedule_decision_table.csv"),
    Path("outputs/76_casa_grande_priority_downloads.csv"),
    Path("outputs/77_casa_grande_schedule_checks.csv"),
    Path("outputs/81_landsat_targeted_reviewed_patch_index.csv"),
    Path("outputs/82_landsat_targeted_reviewed_features.csv"),
    Path("outputs/85_ehrenberg_candidate_release_evidence.csv"),
    Path("outputs/86_ehrenberg_candidate_release_summary.csv"),
    Path("outputs/87_ehrenberg_final_label_decisions.csv"),
    Path("outputs/88_ehrenberg_priority_download_batch.csv"),
    Path("outputs/90_ehrenberg_priority_landsat_patch_index.csv"),
    Path("outputs/93_ehrenberg_priority_landsat_features.csv"),
    DATASET_CSV,
]


IDENTIFIER_KEYWORDS = [
    "scene_key",
    "raster_group_id",
    "scene_group_id",
    "overpass_id",
    "event_id",
    "landsat_product_id",
    "landsat_scene_id",
    "product_id",
]


TIME_KEYWORDS = [
    "satellite_time",
    "landsat_image_time",
    "candidate_time",
    "acquisition_time",
    "overpass_time",
    "image_time",
]


EVIDENCE_KEYWORDS = [
    "release",
    "flow",
    "rate",
    "scfh",
    "kg_h",
    "kg/h",
    "kgh",
    "overlap",
    "start",
    "end",
    "wind",
    "confidence",
    "status",
    "reason",
    "campaign",
    "controlled",
]


def normalize_name(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def clean_value(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
    }:
        return ""

    return text


def matching_columns(columns, keywords):
    matches = []

    for column in columns:
        normalized = normalize_name(column)

        if any(
            keyword in normalized
            for keyword in keywords
        ):
            matches.append(column)

    return matches


def first_existing_value(row, candidates):
    for column in candidates:
        if column not in row.index:
            continue

        value = clean_value(row[column])

        if value:
            return value

    return ""


def collect_identifier_values(row):
    identifier_columns = matching_columns(
        row.index,
        IDENTIFIER_KEYWORDS,
    )

    values = set()

    for column in identifier_columns:
        value = clean_value(row[column])

        if value:
            values.add(value.lower())

    return values


def find_time_column(dataframe):
    candidate_columns = matching_columns(
        dataframe.columns,
        TIME_KEYWORDS,
    )

    best_column = None
    best_score = -1

    for column in candidate_columns:
        parsed = pd.to_datetime(
            dataframe[column],
            errors="coerce",
            utc=True,
        )

        score = int(parsed.notna().sum())

        if score > best_score:
            best_score = score
            best_column = column

    return best_column


def infer_site(row):
    text = " ".join(
        clean_value(row.get(column, ""))
        for column in [
            "site_key_normalized",
            "site_key",
            "site_name",
            "site",
            "location_name",
            "event_id",
            "scene_key",
        ]
    ).lower()

    if "ehrenberg" in text:
        return "ehrenberg"

    if "casa" in text or "grande" in text:
        return "casa_grande"

    return "unknown"


def classify_evidence_column(column):
    normalized = normalize_name(column)

    if (
        "wind" in normalized
        and (
            "speed" in normalized
            or "velocity" in normalized
        )
    ):
        return "wind"

    if "overlap" in normalized:
        return "overlap"

    if (
        "start" in normalized
        and (
            "release" in normalized
            or "interval" in normalized
            or "cr_" in normalized
        )
    ):
        return "release_start"

    if (
        "end" in normalized
        and (
            "release" in normalized
            or "interval" in normalized
            or "cr_" in normalized
        )
    ):
        return "release_end"

    rate_terms = [
        "release_rate",
        "flow_rate",
        "kg_h",
        "kgh",
        "kg_per_h",
        "scfh",
        "flow",
    ]

    if any(
        term in normalized
        for term in rate_terms
    ):
        return "release_rate"

    if "confidence" in normalized:
        return "confidence"

    if "status" in normalized:
        return "status"

    if "reason" in normalized:
        return "reason"

    return "other"


def unique_join(values, limit=20):
    cleaned = []

    for value in values:
        text = clean_value(value)

        if not text:
            continue

        if len(text) > 300:
            text = text[:300] + "..."

        if text not in cleaned:
            cleaned.append(text)

    return " | ".join(
        cleaned[:limit]
    )


def main():
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_CSV}"
        )

    dataset = pd.read_csv(
        DATASET_CSV,
        low_memory=False,
    )

    dataset["label"] = pd.to_numeric(
        dataset["label"],
        errors="raise",
    ).astype(int)

    positives = dataset[
        dataset["label"] == 1
    ].copy()

    positives = positives.reset_index(
        drop=True
    )

    print("=" * 105)
    print("LANDSAT POSITIVE-SCENE RELEASE INVENTORY")
    print("=" * 105)

    print(f"\nStrict Core rows: {len(dataset)}")
    print(f"Positive scenes: {len(positives)}")

    if len(positives) != 9:
        print(
            "Warning: expected 9 positive scenes, "
            f"but found {len(positives)}."
        )

    source_files = [
        path
        for path in OPTIONAL_SOURCE_FILES
        if path.exists()
    ]

    print(
        f"\nEvidence source files found: "
        f"{len(source_files)}"
    )

    for path in source_files:
        print(f"  {path}")

    positive_records = []

    for positive_index, row in (
        positives.iterrows()
    ):
        scene_key = first_existing_value(
            row,
            [
                "scene_key",
                "raster_group_id",
                "overpass_id",
                "event_id",
            ],
        )

        satellite_time = pd.to_datetime(
            first_existing_value(
                row,
                [
                    "acquisition_time_utc",
                    "landsat_image_time_utc",
                    "landsat_image_time",
                    "candidate_time_utc",
                ],
            ),
            errors="coerce",
            utc=True,
        )

        positive_records.append({
            "positive_index": positive_index,
            "scene_key": scene_key,
            "raster_group_id":
                first_existing_value(
                    row,
                    ["raster_group_id"],
                ),
            "overpass_id":
                first_existing_value(
                    row,
                    ["overpass_id"],
                ),
            "event_id":
                first_existing_value(
                    row,
                    ["event_id"],
                ),
            "landsat_product_id":
                first_existing_value(
                    row,
                    [
                        "landsat_product_id_normalized",
                        "LANDSAT_PRODUCT_ID",
                        "landsat_product_id",
                    ],
                ),
            "site_key": infer_site(row),
            "landsat_sensor":
                first_existing_value(
                    row,
                    ["landsat_sensor"],
                ),
            "satellite_time":
                satellite_time,
            "identifier_values":
                collect_identifier_values(row),
            "dataset_row":
                row,
        })

    evidence_rows = []

    for source_path in source_files:
        try:
            source = pd.read_csv(
                source_path,
                low_memory=False,
            )

        except Exception as error:
            print(
                f"[READ ERROR] {source_path}: "
                f"{error}"
            )
            continue

        source_identifier_columns = (
            matching_columns(
                source.columns,
                IDENTIFIER_KEYWORDS,
            )
        )

        source_evidence_columns = (
            matching_columns(
                source.columns,
                EVIDENCE_KEYWORDS,
            )
        )

        source_time_column = (
            find_time_column(source)
        )

        if source_time_column is not None:
            source_times = pd.to_datetime(
                source[source_time_column],
                errors="coerce",
                utc=True,
            )
        else:
            source_times = pd.Series(
                pd.NaT,
                index=source.index,
                dtype="datetime64[ns, UTC]",
            )

        for positive in positive_records:
            mask = pd.Series(
                False,
                index=source.index,
            )

            match_reasons = []

            # 優先用 scene/product/overpass ID 配對。
            for column in source_identifier_columns:
                normalized_values = (
                    source[column]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                for identifier in positive[
                    "identifier_values"
                ]:
                    current_match = (
                        normalized_values
                        == identifier
                    )

                    if current_match.any():
                        mask = mask | current_match

                        match_reasons.append(
                            f"{column}={identifier}"
                        )

            # 若 ID 沒有配到，再使用衛星時間 ±5 分鐘。
            if (
                not mask.any()
                and pd.notna(
                    positive["satellite_time"]
                )
                and source_time_column
                is not None
            ):
                time_difference = (
                    source_times
                    - positive["satellite_time"]
                ).dt.total_seconds().abs()

                time_match = (
                    time_difference <= 300
                ).fillna(False)

                if time_match.any():
                    mask = mask | time_match

                    match_reasons.append(
                        f"{source_time_column}"
                        "_within_5_minutes"
                    )

            matched_source = source[
                mask
            ].copy()

            if len(matched_source) == 0:
                continue

            for source_row_index, (
                _,
                source_row,
            ) in enumerate(
                matched_source.iterrows()
            ):
                original_index = (
                    source_row.name
                )

                source_time = (
                    source_times.loc[
                        original_index
                    ]
                    if original_index
                    in source_times.index
                    else pd.NaT
                )

                if (
                    pd.notna(source_time)
                    and pd.notna(
                        positive[
                            "satellite_time"
                        ]
                    )
                ):
                    seconds_from_satellite = (
                        source_time
                        - positive[
                            "satellite_time"
                        ]
                    ).total_seconds()
                else:
                    seconds_from_satellite = (
                        np.nan
                    )

                for column in (
                    source_evidence_columns
                ):
                    value = clean_value(
                        source_row.get(
                            column
                        )
                    )

                    if not value:
                        continue

                    evidence_rows.append({
                        "positive_index":
                            positive[
                                "positive_index"
                            ],
                        "scene_key":
                            positive["scene_key"],
                        "raster_group_id":
                            positive[
                                "raster_group_id"
                            ],
                        "overpass_id":
                            positive[
                                "overpass_id"
                            ],
                        "site_key":
                            positive["site_key"],
                        "satellite_time":
                            positive[
                                "satellite_time"
                            ],
                        "source_file":
                            str(source_path),
                        "source_row_index":
                            original_index,
                        "match_reason":
                            " | ".join(
                                sorted(
                                    set(
                                        match_reasons
                                    )
                                )
                            ),
                        "source_time_column":
                            source_time_column,
                        "source_time":
                            source_time,
                        "seconds_from_satellite":
                            seconds_from_satellite,
                        "evidence_column":
                            column,
                        "evidence_category":
                            classify_evidence_column(
                                column
                            ),
                        "evidence_value":
                            value,
                    })

    evidence = pd.DataFrame(
        evidence_rows
    )

    inventory_rows = []

    for positive in positive_records:
        if len(evidence) > 0:
            scene_evidence = evidence[
                evidence["positive_index"]
                == positive["positive_index"]
            ].copy()
        else:
            scene_evidence = pd.DataFrame()

        def category_values(category):
            if len(scene_evidence) == 0:
                return ""

            return unique_join(
                scene_evidence.loc[
                    scene_evidence[
                        "evidence_category"
                    ] == category,
                    "evidence_value",
                ].tolist()
            )

        rate_candidates = category_values(
            "release_rate"
        )

        start_candidates = category_values(
            "release_start"
        )

        end_candidates = category_values(
            "release_end"
        )

        overlap_candidates = category_values(
            "overlap"
        )

        wind_candidates = category_values(
            "wind"
        )

        status_candidates = unique_join(
            scene_evidence.loc[
                scene_evidence[
                    "evidence_category"
                ].isin([
                    "status",
                    "confidence",
                    "reason",
                ]),
                "evidence_value",
            ].tolist()
            if len(scene_evidence) > 0
            else []
        )

        inventory_rows.append({
            "positive_index":
                positive["positive_index"],
            "scene_key":
                positive["scene_key"],
            "raster_group_id":
                positive["raster_group_id"],
            "overpass_id":
                positive["overpass_id"],
            "event_id":
                positive["event_id"],
            "landsat_product_id":
                positive[
                    "landsat_product_id"
                ],
            "site_key":
                positive["site_key"],
            "landsat_sensor":
                positive[
                    "landsat_sensor"
                ],
            "satellite_time":
                positive[
                    "satellite_time"
                ],
            "evidence_row_count":
                len(scene_evidence),
            "candidate_release_rate_values":
                rate_candidates,
            "candidate_release_start_values":
                start_candidates,
            "candidate_release_end_values":
                end_candidates,
            "candidate_overlap_values":
                overlap_candidates,
            "candidate_wind_values":
                wind_candidates,
            "candidate_status_values":
                status_candidates,

            # 下面是我們接下來要正式填入的欄位。
            "release_rate_kg_h":
                np.nan,
            "release_rate_original_value":
                "",
            "release_rate_original_unit":
                "",
            "release_start":
                "",
            "release_end":
                "",
            "exact_overlap":
                np.nan,
            "wind_speed_m_s":
                np.nan,
            "quality_pass":
                np.nan,
            "quality_exclusion_reason":
                "",
            "high_emission_label":
                np.nan,
            "review_status":
                "needs_manual_review",
        })

    inventory = pd.DataFrame(
        inventory_rows
    )

    EVIDENCE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence.to_csv(
        EVIDENCE_OUTPUT,
        index=False,
    )

    inventory.to_csv(
        INVENTORY_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 105)
    print("POSITIVE SCENE INVENTORY")
    print("=" * 105)

    display_columns = [
        "scene_key",
        "raster_group_id",
        "overpass_id",
        "site_key",
        "satellite_time",
        "evidence_row_count",
        "candidate_release_rate_values",
        "candidate_overlap_values",
        "candidate_status_values",
    ]

    print(
        inventory[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 105)
    print("MISSING-EVIDENCE SUMMARY")
    print("=" * 105)

    missing_rate = inventory[
        inventory[
            "candidate_release_rate_values"
        ].astype(str).str.strip().eq("")
    ]

    missing_overlap = inventory[
        inventory[
            "candidate_overlap_values"
        ].astype(str).str.strip().eq("")
    ]

    print(
        f"\nScenes with no candidate "
        f"release-rate value: "
        f"{len(missing_rate)}"
    )

    if len(missing_rate) > 0:
        print(
            missing_rate[
                [
                    "scene_key",
                    "raster_group_id",
                    "overpass_id",
                    "site_key",
                    "satellite_time",
                ]
            ].to_string(index=False)
        )

    print(
        f"\nScenes with no candidate "
        f"overlap value: "
        f"{len(missing_overlap)}"
    )

    if len(missing_overlap) > 0:
        print(
            missing_overlap[
                [
                    "scene_key",
                    "raster_group_id",
                    "overpass_id",
                    "site_key",
                    "satellite_time",
                ]
            ].to_string(index=False)
        )

    print("\nSaved:")
    print(EVIDENCE_OUTPUT)
    print(INVENTORY_OUTPUT)


if __name__ == "__main__":
    main()
