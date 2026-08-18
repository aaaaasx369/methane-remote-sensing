from pathlib import Path
import os

import ee
import numpy as np
import pandas as pd


COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"

POSITIVE_INPUT = Path(
    "outputs/355_s2_high_emission_positive_manifest_clean_v1.csv"
)

RELEASE_INTERVAL_INPUT = Path(
    "outputs/309_all_exact_release_intervals_for_s2.csv"
)

DIRECT_GT_INPUT = Path(
    "outputs/307_s2_direct_strict_ground_truth_v1.csv"
)

LOW_LOCKED_INPUT = Path(
    "outputs/341_s2_low_emission_pilot_v1_locked.csv"
)

LOW_MANIFEST_INPUT = Path(
    "outputs/317_s2_low_emission_scene_manifest_v1.csv"
)

ALL_HIGH_RESOLVED_INPUT = Path(
    "outputs/352_s2_high_emission_positive_manifest_resolved_v1.csv"
)

CANDIDATE_OUTPUT = Path(
    "outputs/358_s2_high_emission_negative_candidates_v1.csv"
)

SELECTED_OUTPUT = Path(
    "outputs/359_s2_high_emission_matched_negative_manifest_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/360_s2_high_emission_matched_negative_report_v1.txt"
)


SEARCH_DAYS = 60
MAX_CLOUD_PERCENTAGE = 40.0
MIN_RELEASE_DISTANCE_HOURS = 24.0
KNOWN_POSITIVE_TIME_TOLERANCE_MINUTES = 2.0
NEGATIVES_PER_POSITIVE = 4
NEGATIVES_PER_SIDE = 2


def initialize_earth_engine():
    project = os.environ.get("EE_PROJECT")

    if not project:
        raise RuntimeError(
            "找不到 EE_PROJECT。請先執行：\n"
            'export EE_PROJECT="methane-release-gee"'
        )

    try:
        ee.Initialize(project=project)

    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)

    print(
        "Earth Engine initialized:",
        project,
    )


def find_column(
    frame,
    candidates,
    required=True,
):
    for column in candidates:
        if column in frame.columns:
            return column

    if required:
        raise KeyError(
            "找不到任何候選欄位："
            + ", ".join(candidates)
        )

    return None


def normalize_site(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
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


def normalize_scene_id(value):
    value = str(value).strip()

    if value.startswith(
        COLLECTION_ID + "/"
    ):
        return value

    return (
        COLLECTION_ID
        + "/"
        + value
    )


def load_release_intervals():
    frame = pd.read_csv(
        RELEASE_INTERVAL_INPUT,
        low_memory=False,
    )

    site_column = find_column(
        frame,
        [
            "site",
            "site_name",
            "release_site",
            "site_key",
        ],
    )

    start_column = find_column(
        frame,
        [
            "release_start_utc",
            "exact_release_start_utc",
            "start_utc",
            "start_time_utc",
            "interval_start_utc",
        ],
    )

    end_column = find_column(
        frame,
        [
            "release_end_utc",
            "exact_release_end_utc",
            "end_utc",
            "end_time_utc",
            "interval_end_utc",
        ],
    )

    rate_column = find_column(
        frame,
        [
            "release_rate_kg_h",
            "final_release_rate_kg_h",
            "preferred_release_rate_kg_h",
            "selected_release_rate_kg_h",
            "exact_release_rate_kg_h",
            "cr_kgh_CH4_mean300",
            "ch4_kgh_mean",
        ],
    )

    result = pd.DataFrame({
        "site_key":
            frame[
                site_column
            ].map(normalize_site),

        "release_start_utc":
            pd.to_datetime(
                frame[start_column],
                errors="coerce",
                utc=True,
            ),

        "release_end_utc":
            pd.to_datetime(
                frame[end_column],
                errors="coerce",
                utc=True,
            ),

        "release_rate_kg_h":
            pd.to_numeric(
                frame[rate_column],
                errors="coerce",
            ),
    })

    result = result.dropna(
        subset=[
            "site_key",
            "release_start_utc",
            "release_end_utc",
            "release_rate_kg_h",
        ]
    )

    result = result[
        result[
            "release_rate_kg_h"
        ].gt(0)
    ].copy()

    return result


def load_known_positive_times():
    frames = []

    if DIRECT_GT_INPUT.exists():
        direct = pd.read_csv(
            DIRECT_GT_INPUT,
            low_memory=False,
        )

        required = {
            "site_name",
            "actual_s2_time",
            "strict_label",
        }

        if required.issubset(
            direct.columns
        ):
            labels = pd.to_numeric(
                direct["strict_label"],
                errors="coerce",
            )

            direct = direct[
                labels.eq(1)
            ].copy()

            frames.append(
                pd.DataFrame({
                    "site_key":
                        direct[
                            "site_name"
                        ].map(
                            normalize_site
                        ),

                    "known_positive_time_utc":
                        pd.to_datetime(
                            direct[
                                "actual_s2_time"
                            ],
                            errors="coerce",
                            utc=True,
                        ),

                    "known_positive_source":
                        "307_direct_strict_ground_truth",
                })
            )

    if LOW_MANIFEST_INPUT.exists():
        low = pd.read_csv(
            LOW_MANIFEST_INPUT,
            low_memory=False,
        )

        site_column = find_column(
            low,
            [
                "site",
                "site_name",
            ],
            required=False,
        )

        time_column = find_column(
            low,
            [
                "acquisition_time_utc",
                "actual_s2_time",
            ],
            required=False,
        )

        if (
            site_column is not None
            and time_column is not None
        ):
            frames.append(
                pd.DataFrame({
                    "site_key":
                        low[
                            site_column
                        ].map(
                            normalize_site
                        ),

                    "known_positive_time_utc":
                        pd.to_datetime(
                            low[
                                time_column
                            ],
                            errors="coerce",
                            utc=True,
                        ),

                    "known_positive_source":
                        "317_low_emission_manifest",
                })
            )

    if LOW_LOCKED_INPUT.exists():
        low_locked = pd.read_csv(
            LOW_LOCKED_INPUT,
            low_memory=False,
        )

        labels = pd.to_numeric(
            low_locked["label"],
            errors="coerce",
        )

        low_positive = low_locked[
            labels.eq(1)
        ].copy()

        frames.append(
            pd.DataFrame({
                "site_key":
                    low_positive[
                        "site"
                    ].map(
                        normalize_site
                    ),

                "known_positive_time_utc":
                    pd.to_datetime(
                        low_positive[
                            "acquisition_time_utc"
                        ],
                        errors="coerce",
                        utc=True,
                    ),

                "known_positive_source":
                    "341_low_locked_positive",
            })
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "site_key",
                "known_positive_time_utc",
                "known_positive_source",
            ]
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = (
        result.dropna(
            subset=[
                "site_key",
                "known_positive_time_utc",
            ]
        )
        .drop_duplicates(
            subset=[
                "site_key",
                "known_positive_time_utc",
            ]
        )
        .reset_index(drop=True)
    )

    return result


def load_excluded_scene_ids():
    scene_ids = set()

    for path in [
        LOW_LOCKED_INPUT,
        ALL_HIGH_RESOLVED_INPUT,
        POSITIVE_INPUT,
    ]:
        if not path.exists():
            continue

        frame = pd.read_csv(
            path,
            low_memory=False,
        )

        if "scene_id" not in frame.columns:
            continue

        for value in (
            frame["scene_id"]
            .dropna()
            .astype(str)
        ):
            value = value.strip()

            if value and value != "nan":
                scene_ids.add(
                    normalize_scene_id(
                        value
                    )
                )

    return scene_ids


def nearest_release_distance_hours(
    acquisition_time,
    site_intervals,
):
    if site_intervals.empty:
        return np.inf

    starts = site_intervals[
        "release_start_utc"
    ]

    ends = site_intervals[
        "release_end_utc"
    ]

    before_distance = (
        starts - acquisition_time
    ).dt.total_seconds() / 3600.0

    after_distance = (
        acquisition_time - ends
    ).dt.total_seconds() / 3600.0

    distances = np.where(
        acquisition_time < starts,
        before_distance,
        np.where(
            acquisition_time > ends,
            after_distance,
            0.0,
        ),
    )

    finite = np.asarray(
        distances,
        dtype=float,
    )

    finite = finite[
        np.isfinite(finite)
    ]

    if finite.size == 0:
        return np.inf

    return float(
        np.min(finite)
    )


def nearest_known_positive_minutes(
    acquisition_time,
    known_positive_site,
):
    if known_positive_site.empty:
        return np.inf

    differences = (
        known_positive_site[
            "known_positive_time_utc"
        ]
        - acquisition_time
    ).abs().dt.total_seconds() / 60.0

    differences = differences.dropna()

    if differences.empty:
        return np.inf

    return float(
        differences.min()
    )


def search_positive_candidates(
    positive,
    release_intervals,
    known_positive_times,
    excluded_scene_ids,
):
    positive_time = pd.to_datetime(
        positive[
            "acquisition_time_utc"
        ],
        errors="raise",
        utc=True,
    )

    site_key = normalize_site(
        positive["site"]
    )

    start_time = (
        positive_time
        - pd.Timedelta(
            days=SEARCH_DAYS
        )
    )

    end_time = (
        positive_time
        + pd.Timedelta(
            days=SEARCH_DAYS
        )
        + pd.Timedelta(seconds=1)
    )

    point = ee.Geometry.Point([
        float(positive["lon"]),
        float(positive["lat"]),
    ])

    collection = (
        ee.ImageCollection(
            COLLECTION_ID
        )
        .filterBounds(point)
        .filterDate(
            start_time.isoformat(),
            end_time.isoformat(),
        )
        .filter(
            ee.Filter.eq(
                "MGRS_TILE",
                str(
                    positive[
                        "mgrs_tile"
                    ]
                ),
            )
        )
        .filter(
            ee.Filter.lte(
                "CLOUDY_PIXEL_PERCENTAGE",
                MAX_CLOUD_PERCENTAGE,
            )
        )
        .sort("system:time_start")
    )

    information = collection.getInfo()

    site_intervals = release_intervals[
        release_intervals[
            "site_key"
        ].eq(site_key)
    ]

    site_positive_times = (
        known_positive_times[
            known_positive_times[
                "site_key"
            ].eq(site_key)
        ]
    )

    rows = []

    for feature in information.get(
        "features",
        [],
    ):
        properties = feature.get(
            "properties",
            {},
        )

        scene_id = normalize_scene_id(
            feature.get(
                "id",
                properties.get(
                    "system:index",
                    "",
                ),
            )
        )

        acquisition_time = pd.to_datetime(
            properties.get(
                "system:time_start"
            ),
            unit="ms",
            errors="coerce",
            utc=True,
        )

        cloud = pd.to_numeric(
            properties.get(
                "CLOUDY_PIXEL_PERCENTAGE"
            ),
            errors="coerce",
        )

        if pd.isna(acquisition_time):
            continue

        time_delta_days = (
            acquisition_time
            - positive_time
        ).total_seconds() / 86400.0

        if time_delta_days < 0:
            temporal_side = "before"
        elif time_delta_days > 0:
            temporal_side = "after"
        else:
            temporal_side = "same_time"

        release_distance = (
            nearest_release_distance_hours(
                acquisition_time,
                site_intervals,
            )
        )

        positive_time_distance = (
            nearest_known_positive_minutes(
                acquisition_time,
                site_positive_times,
            )
        )

        reasons = []

        if scene_id in excluded_scene_ids:
            reasons.append(
                "excluded_existing_scene"
            )

        if (
            positive_time_distance
            <= KNOWN_POSITIVE_TIME_TOLERANCE_MINUTES
        ):
            reasons.append(
                "known_positive_acquisition_time"
            )

        if (
            release_distance
            <= MIN_RELEASE_DISTANCE_HOURS
        ):
            reasons.append(
                "within_24h_nonzero_release"
            )

        if temporal_side == "same_time":
            reasons.append(
                "same_time_as_positive"
            )

        candidate_status = (
            "eligible"
            if not reasons
            else "excluded"
        )

        rows.append({
            "positive_id":
                positive[
                    "positive_id"
                ],

            "matched_positive_scene_id":
                positive[
                    "scene_id"
                ],

            "matched_positive_time_utc":
                positive_time,

            "matched_positive_rate_kg_h":
                positive[
                    "release_rate_kg_h"
                ],

            "site":
                positive["site"],

            "mgrs_tile":
                positive[
                    "mgrs_tile"
                ],

            "lat":
                positive["lat"],

            "lon":
                positive["lon"],

            "scene_id":
                scene_id,

            "acquisition_time_utc":
                acquisition_time,

            "days_from_positive":
                time_delta_days,

            "absolute_days_from_positive":
                abs(time_delta_days),

            "temporal_side":
                temporal_side,

            "scene_cloud_percentage":
                cloud,

            "nearest_nonzero_release_hours":
                release_distance,

            "nearest_known_positive_minutes":
                positive_time_distance,

            "candidate_status":
                candidate_status,

            "exclusion_reason":
                ";".join(reasons),

            "system_index":
                properties.get(
                    "system:index"
                ),

            "product_id":
                properties.get(
                    "PRODUCT_ID"
                ),
        })

    return rows


def build_slot_edges(
    eligible,
):
    slot_edges = {}
    candidate_lookup = {}

    positive_ids = (
        eligible[
            "positive_id"
        ]
        .drop_duplicates()
        .tolist()
    )

    for positive_id in positive_ids:
        group = eligible[
            eligible[
                "positive_id"
            ].eq(positive_id)
        ]

        for side in [
            "before",
            "after",
        ]:
            side_group = group[
                group[
                    "temporal_side"
                ].eq(side)
            ].sort_values(
                [
                    "absolute_days_from_positive",
                    "scene_cloud_percentage",
                    "acquisition_time_utc",
                ]
            )

            scene_ids = (
                side_group[
                    "scene_id"
                ]
                .drop_duplicates()
                .tolist()
            )

            for slot_number in range(
                1,
                NEGATIVES_PER_SIDE + 1,
            ):
                slot_id = (
                    f"{positive_id}|"
                    f"{side}|"
                    f"{slot_number}"
                )

                slot_edges[slot_id] = (
                    scene_ids.copy()
                )

            for _, row in (
                side_group.iterrows()
            ):
                key = (
                    positive_id,
                    row["scene_id"],
                )

                if key not in candidate_lookup:
                    candidate_lookup[key] = (
                        row.to_dict()
                    )

    return (
        slot_edges,
        candidate_lookup,
    )


def maximum_unique_matching(
    slot_edges,
):
    scene_to_slot = {}
    slot_to_scene = {}

    ordered_slots = sorted(
        slot_edges,
        key=lambda slot:
            len(slot_edges[slot]),
    )

    def try_assign(
        slot_id,
        visited_scenes,
    ):
        for scene_id in slot_edges[
            slot_id
        ]:
            if scene_id in visited_scenes:
                continue

            visited_scenes.add(
                scene_id
            )

            current_slot = (
                scene_to_slot.get(
                    scene_id
                )
            )

            if (
                current_slot is None
                or try_assign(
                    current_slot,
                    visited_scenes,
                )
            ):
                scene_to_slot[
                    scene_id
                ] = slot_id

                slot_to_scene[
                    slot_id
                ] = scene_id

                return True

        return False

    for slot_id in ordered_slots:
        try_assign(
            slot_id,
            set(),
        )

    return slot_to_scene


def build_selected_manifest(
    eligible,
):
    (
        slot_edges,
        candidate_lookup,
    ) = build_slot_edges(
        eligible
    )

    slot_to_scene = (
        maximum_unique_matching(
            slot_edges
        )
    )

    rows = []

    for slot_id, scene_id in (
        slot_to_scene.items()
    ):
        (
            positive_id,
            required_side,
            slot_number,
        ) = slot_id.split("|")

        key = (
            positive_id,
            scene_id,
        )

        row = dict(
            candidate_lookup[key]
        )

        row[
            "required_temporal_side"
        ] = required_side

        row[
            "matching_slot_number"
        ] = int(slot_number)

        rows.append(row)

    selected = pd.DataFrame(
        rows
    )

    if selected.empty:
        return selected

    selected = selected.sort_values(
        [
            "matched_positive_time_utc",
            "temporal_side",
            "acquisition_time_utc",
        ]
    ).reset_index(drop=True)

    selected["negative_id"] = [
        f"S2_HIGH_NEG_{number:03d}"
        for number in range(
            1,
            len(selected) + 1,
        )
    ]

    selected["label"] = 0

    selected[
        "dataset_role"
    ] = (
        "high_emission_matched_negative"
    )

    selected[
        "selection_version"
    ] = (
        "s2_high_emission_negative_v1"
    )

    selected[
        "local_qa_status"
    ] = "pending"

    return selected


def main():
    initialize_earth_engine()

    positives = pd.read_csv(
        POSITIVE_INPUT,
        low_memory=False,
    )

    positives[
        "acquisition_time_utc"
    ] = pd.to_datetime(
        positives[
            "acquisition_time_utc"
        ],
        errors="raise",
        utc=True,
    )

    for column in [
        "lat",
        "lon",
        "release_rate_kg_h",
    ]:
        positives[column] = (
            pd.to_numeric(
                positives[column],
                errors="raise",
            )
        )

    release_intervals = (
        load_release_intervals()
    )

    known_positive_times = (
        load_known_positive_times()
    )

    excluded_scene_ids = (
        load_excluded_scene_ids()
    )

    print("=" * 115)
    print(
        "SEARCH SENTINEL-2 HIGH-EMISSION "
        "MATCHED NEGATIVES"
    )
    print("=" * 115)

    print(
        "\nClean positives:",
        len(positives),
    )

    print(
        "Nonzero release intervals:",
        len(release_intervals),
    )

    print(
        "Known positive acquisition times:",
        len(known_positive_times),
    )

    print(
        "Excluded existing scene IDs:",
        len(excluded_scene_ids),
    )

    all_rows = []

    for number, positive in (
        positives.iterrows()
    ):
        print(
            f"\n[{number + 1}/{len(positives)}] "
            f"{positive['positive_id']} | "
            f"{positive['site']} | "
            f"{positive['acquisition_time_utc']}",
            flush=True,
        )

        rows = search_positive_candidates(
            positive=positive,
            release_intervals=
                release_intervals,
            known_positive_times=
                known_positive_times,
            excluded_scene_ids=
                excluded_scene_ids,
        )

        all_rows.extend(rows)

        frame = pd.DataFrame(rows)

        if frame.empty:
            print(
                "  Earth Engine scenes: 0"
            )
            continue

        eligible = frame[
            frame[
                "candidate_status"
            ].eq("eligible")
        ]

        print(
            "  All scenes:",
            len(frame),
        )

        print(
            "  Eligible:",
            len(eligible),
        )

        print(
            "  Before:",
            int(
                eligible[
                    "temporal_side"
                ].eq("before").sum()
            ),
        )

        print(
            "  After:",
            int(
                eligible[
                    "temporal_side"
                ].eq("after").sum()
            ),
        )

    candidates = pd.DataFrame(
        all_rows
    )

    candidates.to_csv(
        CANDIDATE_OUTPUT,
        index=False,
    )

    eligible = candidates[
        candidates[
            "candidate_status"
        ].eq("eligible")
    ].copy()

    selected = build_selected_manifest(
        eligible
    )

    selected.to_csv(
        SELECTED_OUTPUT,
        index=False,
    )

    expected_count = (
        len(positives)
        * NEGATIVES_PER_POSITIVE
    )

    selected_count = len(selected)

    unique_selected = (
        int(
            selected[
                "scene_id"
            ].nunique()
        )
        if not selected.empty
        else 0
    )

    if not selected.empty:
        per_positive = (
            selected.groupby(
                [
                    "positive_id",
                    "matched_positive_time_utc",
                    "matched_positive_rate_kg_h",
                ]
            )["scene_id"].nunique()
        )

        side_counts = (
            selected[
                "temporal_side"
            ].value_counts()
        )

        minimum_release_distance = (
            selected[
                "nearest_nonzero_release_hours"
            ].min()
        )

        duplicate_count = int(
            selected[
                "scene_id"
            ].duplicated().sum()
        )
    else:
        per_positive = pd.Series(
            dtype=int
        )

        side_counts = pd.Series(
            dtype=int
        )

        minimum_release_distance = (
            np.nan
        )

        duplicate_count = 0

    report_lines = [
        "=" * 115,
        (
            "SENTINEL-2 HIGH-EMISSION "
            "MATCHED-NEGATIVE REPORT"
        ),
        "=" * 115,
        "",
        f"Positive scenes: {len(positives)}",
        (
            f"Target negatives: "
            f"{expected_count}"
        ),
        (
            f"All candidate rows: "
            f"{len(candidates)}"
        ),
        (
            f"Eligible candidate rows: "
            f"{len(eligible)}"
        ),
        (
            f"Selected negatives: "
            f"{selected_count}"
        ),
        (
            f"Unique selected scene IDs: "
            f"{unique_selected}"
        ),
        (
            "Duplicated selected scene IDs: "
            f"{duplicate_count}"
        ),
        (
            "Minimum distance from nonzero "
            f"release: {minimum_release_distance} hours"
        ),
        "",
        "Selected negatives per positive:",
        per_positive.to_string(),
        "",
        "Temporal side:",
        side_counts.to_string(),
    ]

    if selected_count < expected_count:
        missing_slots = (
            expected_count
            - selected_count
        )

        report_lines.extend([
            "",
            "WARNING:",
            (
                f"{missing_slots} matched-negative "
                "slots could not be filled under the "
                "locked ±60-day, 2-before/2-after, "
                "globally unique constraints."
            ),
        ])

    if not selected.empty:
        report_lines.extend([
            "",
            "Final selected scenes:",
            selected[
                [
                    "negative_id",
                    "positive_id",
                    "matched_positive_time_utc",
                    "matched_positive_rate_kg_h",
                    "acquisition_time_utc",
                    "days_from_positive",
                    "temporal_side",
                    "scene_cloud_percentage",
                    "nearest_nonzero_release_hours",
                    "scene_id",
                ]
            ].to_string(index=False),
        ])

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 115)
    print("MATCHED-NEGATIVE SUMMARY")
    print("=" * 115)

    print(
        "\nAll candidate rows:",
        len(candidates),
    )

    print(
        "Eligible candidate rows:",
        len(eligible),
    )

    print(
        "Target negatives:",
        expected_count,
    )

    print(
        "Selected negatives:",
        selected_count,
    )

    print(
        "Unique selected scene IDs:",
        unique_selected,
    )

    print(
        "Duplicated selected scene IDs:",
        duplicate_count,
    )

    print(
        "\nSelected negatives per positive:"
    )

    print(per_positive)

    print("\nTemporal side:")
    print(side_counts)

    print(
        "\nMinimum distance from "
        "nonzero release (hours):",
        minimum_release_distance,
    )

    if selected_count < expected_count:
        print(
            "\nWARNING: only",
            selected_count,
            "of",
            expected_count,
            "slots were filled."
        )

    if not selected.empty:
        print("\nFinal selected scenes:")

        print(
            selected[
                [
                    "negative_id",
                    "positive_id",
                    "matched_positive_time_utc",
                    "matched_positive_rate_kg_h",
                    "acquisition_time_utc",
                    "days_from_positive",
                    "temporal_side",
                    "scene_cloud_percentage",
                    "nearest_nonzero_release_hours",
                    "scene_id",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nSaved:")
    print(CANDIDATE_OUTPUT)
    print(SELECTED_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
