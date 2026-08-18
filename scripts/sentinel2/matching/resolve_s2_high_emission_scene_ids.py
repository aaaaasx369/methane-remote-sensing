from pathlib import Path
import os

import ee
import numpy as np
import pandas as pd


COLLECTION_ID = (
    "COPERNICUS/S2_SR_HARMONIZED"
)

HIGH_INVENTORY_INPUT = Path(
    "outputs/349_s2_high_emission_positive_inventory_v1.csv"
)

LOW_LOCKED_INPUT = Path(
    "outputs/341_s2_low_emission_pilot_v1_locked.csv"
)

CANDIDATE_OUTPUT = Path(
    "outputs/351_s2_high_emission_scene_resolution_candidates_v1.csv"
)

RESOLVED_OUTPUT = Path(
    "outputs/352_s2_high_emission_positive_manifest_resolved_v1.csv"
)

OVERLAP_OUTPUT = Path(
    "outputs/353_s2_high_low_scene_overlap_audit_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/354_s2_high_emission_scene_resolution_report.txt"
)


SEARCH_WINDOW_MINUTES = 10
EXACT_MATCH_TOLERANCE_SECONDS = 120
MAX_CANDIDATES_PER_POSITIVE = 10


def initialize_earth_engine():
    project = os.environ.get(
        "EE_PROJECT"
    )

    if not project:
        raise RuntimeError(
            "找不到 EE_PROJECT。請先執行：\n"
            'export EE_PROJECT="methane-release-gee"'
        )

    try:
        ee.Initialize(
            project=project
        )

    except Exception:
        print(
            "Earth Engine 尚未完成登入，"
            "開始驗證..."
        )

        ee.Authenticate()

        ee.Initialize(
            project=project
        )

    print(
        "Earth Engine initialized:",
        project,
    )


def normalize_scene_id(scene_id):
    scene_id = str(
        scene_id
    ).strip()

    if scene_id.startswith(
        COLLECTION_ID + "/"
    ):
        return scene_id

    return (
        COLLECTION_ID
        + "/"
        + scene_id
    )


def extract_candidates(row):
    target_time = pd.to_datetime(
        row["scene_lookup_time_utc"],
        errors="raise",
        utc=True,
    )

    latitude = float(
        row["lat"]
    )

    longitude = float(
        row["lon"]
    )

    start_time = (
        target_time
        - pd.Timedelta(
            minutes=SEARCH_WINDOW_MINUTES
        )
    )

    end_time = (
        target_time
        + pd.Timedelta(
            minutes=SEARCH_WINDOW_MINUTES
        )
    )

    point = ee.Geometry.Point([
        longitude,
        latitude,
    ])

    target_ee_date = ee.Date(
        int(
            target_time.timestamp()
            * 1000
        )
    )

    def add_time_difference(image):
        difference = (
            image.date()
            .difference(
                target_ee_date,
                "second",
            )
            .abs()
        )

        return image.set(
            "absolute_time_difference_seconds",
            difference,
        )

    collection = (
        ee.ImageCollection(
            COLLECTION_ID
        )
        .filterBounds(point)
        .filterDate(
            start_time.isoformat(),
            end_time.isoformat(),
        )
        .map(
            add_time_difference
        )
        .sort(
            "absolute_time_difference_seconds"
        )
    )

    candidate_count = int(
        collection.size().getInfo()
    )

    information = (
        collection
        .limit(
            MAX_CANDIDATES_PER_POSITIVE
        )
        .getInfo()
    )

    candidate_rows = []

    for rank, feature in enumerate(
        information.get(
            "features",
            [],
        ),
        start=1,
    ):
        properties = feature.get(
            "properties",
            {},
        )

        raw_scene_id = feature.get(
            "id",
            properties.get(
                "system:index",
                "",
            ),
        )

        scene_id = normalize_scene_id(
            raw_scene_id
        )

        acquisition_time = pd.to_datetime(
            properties.get(
                "system:time_start"
            ),
            unit="ms",
            errors="coerce",
            utc=True,
        )

        difference_seconds = pd.to_numeric(
            properties.get(
                "absolute_time_difference_seconds"
            ),
            errors="coerce",
        )

        candidate_rows.append({
            "positive_id":
                row["positive_id"],

            "event_id":
                row["event_id"],

            "site_name":
                row["site_name"],

            "target_time_utc":
                target_time,

            "release_rate_kg_h":
                row[
                    "preferred_release_rate_kg_h"
                ],

            "lat":
                latitude,

            "lon":
                longitude,

            "candidate_rank":
                rank,

            "candidate_count_in_window":
                candidate_count,

            "scene_id":
                scene_id,

            "system_index":
                properties.get(
                    "system:index"
                ),

            "acquisition_time_utc":
                acquisition_time,

            "time_difference_seconds":
                difference_seconds,

            "time_difference_minutes":
                (
                    float(
                        difference_seconds
                    ) / 60
                    if pd.notna(
                        difference_seconds
                    )
                    else np.nan
                ),

            "mgrs_tile":
                properties.get(
                    "MGRS_TILE"
                ),

            "cloudy_pixel_percentage":
                properties.get(
                    "CLOUDY_PIXEL_PERCENTAGE"
                ),

            "product_id":
                properties.get(
                    "PRODUCT_ID"
                ),

            "granule_id":
                properties.get(
                    "GRANULE_ID"
                ),

            "sensing_orbit_number":
                properties.get(
                    "SENSING_ORBIT_NUMBER"
                ),

            "sensing_orbit_direction":
                properties.get(
                    "SENSING_ORBIT_DIRECTION"
                ),

            "datatake_identifier":
                properties.get(
                    "DATATAKE_IDENTIFIER"
                ),
        })

    return candidate_rows


def classify_resolution(
    candidate_group,
):
    if candidate_group.empty:
        return {
            "resolution_status":
                "no_candidate",

            "time_tie_count":
                0,
        }

    minimum_difference = pd.to_numeric(
        candidate_group[
            "time_difference_seconds"
        ],
        errors="coerce",
    ).min()

    tie_count = int(
        np.isclose(
            pd.to_numeric(
                candidate_group[
                    "time_difference_seconds"
                ],
                errors="coerce",
            ),
            minimum_difference,
            atol=0.001,
            equal_nan=False,
        ).sum()
    )

    if tie_count > 1:
        status = (
            "ambiguous_equal_time_candidates"
        )

    elif (
        pd.notna(minimum_difference)
        and minimum_difference
        <= EXACT_MATCH_TOLERANCE_SECONDS
    ):
        status = "resolved_exact"

    else:
        status = (
            "unresolved_time_mismatch"
        )

    return {
        "resolution_status":
            status,

        "time_tie_count":
            tie_count,
    }


def build_resolved_manifest(
    inventory,
    candidates,
):
    resolved_rows = []

    for _, inventory_row in (
        inventory.iterrows()
    ):
        positive_id = str(
            inventory_row[
                "positive_id"
            ]
        )

        group = candidates[
            candidates[
                "positive_id"
            ].astype(str).eq(
                positive_id
            )
        ].copy()

        group = group.sort_values(
            [
                "time_difference_seconds",
                "candidate_rank",
            ]
        )

        resolution = (
            classify_resolution(
                group
            )
        )

        record = (
            inventory_row.to_dict()
        )

        if group.empty:
            record.update({
                "scene_id":
                    pd.NA,

                "resolved_acquisition_time_utc":
                    pd.NaT,

                "scene_time_difference_seconds":
                    np.nan,

                "scene_time_difference_minutes":
                    np.nan,

                "mgrs_tile":
                    pd.NA,

                "cloudy_pixel_percentage":
                    np.nan,

                "product_id":
                    pd.NA,

                "candidate_count_in_window":
                    0,

                **resolution,
            })

        else:
            best = group.iloc[0]

            record.update({
                "scene_id":
                    best["scene_id"],

                "resolved_acquisition_time_utc":
                    best[
                        "acquisition_time_utc"
                    ],

                "scene_time_difference_seconds":
                    best[
                        "time_difference_seconds"
                    ],

                "scene_time_difference_minutes":
                    best[
                        "time_difference_minutes"
                    ],

                "mgrs_tile":
                    best["mgrs_tile"],

                "cloudy_pixel_percentage":
                    best[
                        "cloudy_pixel_percentage"
                    ],

                "product_id":
                    best["product_id"],

                "candidate_count_in_window":
                    best[
                        "candidate_count_in_window"
                    ],

                **resolution,
            })

        resolved_rows.append(
            record
        )

    resolved = pd.DataFrame(
        resolved_rows
    )

    resolved[
        "scene_resolution_version"
    ] = (
        "s2_high_emission_resolution_v1"
    )

    return resolved


def build_overlap_audit(
    resolved,
):
    if not LOW_LOCKED_INPUT.exists():
        return pd.DataFrame()

    low = pd.read_csv(
        LOW_LOCKED_INPUT,
        low_memory=False,
    )

    low["label"] = pd.to_numeric(
        low["label"],
        errors="coerce",
    )

    low_positive = low[
        low["label"].eq(1)
    ].copy()

    low_positive = (
        low_positive[
            [
                "sample_id",
                "scene_id",
                "acquisition_time_utc",
                "release_rate_kg_h",
                "matched_group_id",
            ]
        ]
        .rename(
            columns={
                "sample_id":
                    "low_positive_sample_id",

                "acquisition_time_utc":
                    "low_acquisition_time_utc",

                "release_rate_kg_h":
                    "low_release_rate_kg_h",

                "matched_group_id":
                    "low_matched_group_id",
            }
        )
    )

    audit = resolved.merge(
        low_positive,
        on="scene_id",
        how="left",
        validate="many_to_one",
    )

    audit[
        "high_release_rate_kg_h"
    ] = pd.to_numeric(
        audit[
            "preferred_release_rate_kg_h"
        ],
        errors="coerce",
    )

    audit[
        "low_release_rate_kg_h"
    ] = pd.to_numeric(
        audit[
            "low_release_rate_kg_h"
        ],
        errors="coerce",
    )

    audit[
        "same_scene_in_low_pilot"
    ] = audit[
        "low_positive_sample_id"
    ].notna()

    audit[
        "release_rate_difference_kg_h"
    ] = (
        audit[
            "high_release_rate_kg_h"
        ]
        - audit[
            "low_release_rate_kg_h"
        ]
    )

    audit[
        "absolute_release_rate_difference_kg_h"
    ] = audit[
        "release_rate_difference_kg_h"
    ].abs()

    audit[
        "overlap_status"
    ] = "no_low_pilot_overlap"

    overlap = audit[
        "same_scene_in_low_pilot"
    ]

    matching_rate = (
        overlap
        & audit[
            "absolute_release_rate_difference_kg_h"
        ].le(1.0)
    )

    conflicting_rate = (
        overlap
        & audit[
            "absolute_release_rate_difference_kg_h"
        ].gt(1.0)
    )

    audit.loc[
        matching_rate,
        "overlap_status",
    ] = "same_scene_consistent_rate"

    audit.loc[
        conflicting_rate,
        "overlap_status",
    ] = "same_scene_conflicting_rate"

    return audit


def main():
    initialize_earth_engine()

    inventory = pd.read_csv(
        HIGH_INVENTORY_INPUT,
        low_memory=False,
    )

    inventory[
        "scene_lookup_time_utc"
    ] = pd.to_datetime(
        inventory[
            "scene_lookup_time_utc"
        ],
        errors="raise",
        utc=True,
    )

    inventory["lat"] = pd.to_numeric(
        inventory["lat"],
        errors="raise",
    )

    inventory["lon"] = pd.to_numeric(
        inventory["lon"],
        errors="raise",
    )

    print("=" * 110)
    print(
        "RESOLVE SENTINEL-2 HIGH-EMISSION "
        "SCENE IDS"
    )
    print("=" * 110)

    all_candidates = []

    for number, row in (
        inventory.iterrows()
    ):
        print(
            f"\n[{number + 1}/{len(inventory)}] "
            f"{row['positive_id']} | "
            f"{row['site_name']} | "
            f"{row['scene_lookup_time_utc']}",
            flush=True,
        )

        try:
            candidates = (
                extract_candidates(
                    row
                )
            )

            all_candidates.extend(
                candidates
            )

            print(
                "  Candidates:",
                len(candidates),
            )

            if candidates:
                best = candidates[0]

                print(
                    "  Best scene:",
                    best["scene_id"],
                )

                print(
                    "  Time difference:",
                    best[
                        "time_difference_seconds"
                    ],
                    "seconds",
                )

                print(
                    "  MGRS tile:",
                    best["mgrs_tile"],
                )

        except Exception as error:
            print(
                "  Resolution failed:",
                error,
            )

    candidates = pd.DataFrame(
        all_candidates
    )

    candidates.to_csv(
        CANDIDATE_OUTPUT,
        index=False,
    )

    resolved = (
        build_resolved_manifest(
            inventory,
            candidates,
        )
    )

    resolved.to_csv(
        RESOLVED_OUTPUT,
        index=False,
    )

    overlap = build_overlap_audit(
        resolved
    )

    overlap.to_csv(
        OVERLAP_OUTPUT,
        index=False,
    )

    resolved_count = int(
        resolved[
            "resolution_status"
        ].eq(
            "resolved_exact"
        ).sum()
    )

    unique_scene_count = int(
        resolved[
            "scene_id"
        ].dropna().nunique()
    )

    duplicated_scene_count = int(
        resolved[
            "scene_id"
        ].dropna().duplicated(
            keep=False
        ).sum()
    )

    conflict_count = (
        int(
            overlap[
                "overlap_status"
            ].eq(
                "same_scene_conflicting_rate"
            ).sum()
        )
        if not overlap.empty
        else 0
    )

    report_lines = [
        "=" * 110,
        (
            "SENTINEL-2 HIGH-EMISSION "
            "SCENE RESOLUTION REPORT"
        ),
        "=" * 110,
        "",
        (
            f"Inventory rows: "
            f"{len(inventory)}"
        ),
        (
            f"Exactly resolved rows: "
            f"{resolved_count}"
        ),
        (
            f"Unique resolved scene IDs: "
            f"{unique_scene_count}"
        ),
        (
            "Rows involved in duplicated "
            f"high-emission scene IDs: "
            f"{duplicated_scene_count}"
        ),
        (
            "High/low same-scene rate conflicts: "
            f"{conflict_count}"
        ),
        "",
        "Resolution status:",
        resolved[
            "resolution_status"
        ].value_counts(
            dropna=False
        ).to_string(),
        "",
        "Resolved scenes:",
        resolved[
            [
                "positive_id",
                "site_name",
                "scene_lookup_time_utc",
                "preferred_release_rate_kg_h",
                "scene_id",
                "resolved_acquisition_time_utc",
                "scene_time_difference_seconds",
                "mgrs_tile",
                "cloudy_pixel_percentage",
                "resolution_status",
            ]
        ].to_string(index=False),
    ]

    if not overlap.empty:
        report_lines.extend([
            "",
            "High/low overlap audit:",
            overlap[
                [
                    "positive_id",
                    "scene_id",
                    "high_release_rate_kg_h",
                    "low_positive_sample_id",
                    "low_release_rate_kg_h",
                    "absolute_release_rate_difference_kg_h",
                    "overlap_status",
                ]
            ].to_string(index=False),
        ])

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print("RESOLUTION SUMMARY")
    print("=" * 110)

    print(
        "\nInventory rows:",
        len(inventory),
    )

    print(
        "Exactly resolved rows:",
        resolved_count,
    )

    print(
        "Unique resolved scene IDs:",
        unique_scene_count,
    )

    print("\nResolution status:")
    print(
        resolved[
            "resolution_status"
        ].value_counts(
            dropna=False
        )
    )

    print("\nResolved high-emission scenes:")
    print(
        resolved[
            [
                "positive_id",
                "scene_lookup_time_utc",
                "preferred_release_rate_kg_h",
                "scene_id",
                "scene_time_difference_seconds",
                "mgrs_tile",
                "cloudy_pixel_percentage",
                "resolution_status",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nHigh/low overlap audit:")

    if overlap.empty:
        print(
            "Low-emission locked benchmark "
            "not found."
        )

    else:
        print(
            overlap[
                [
                    "positive_id",
                    "high_release_rate_kg_h",
                    "low_positive_sample_id",
                    "low_release_rate_kg_h",
                    "absolute_release_rate_difference_kg_h",
                    "overlap_status",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nSaved:")
    print(CANDIDATE_OUTPUT)
    print(RESOLVED_OUTPUT)
    print(OVERLAP_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
